# Implementation derived from Time Series Library https://github.com/thuml/Time-Series-Library
from typing import Tuple

import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

from ...models import BaseModel
from ...utils.complex_ops import as_real, complex_abs, is_compile_mode
from ..layers.embed import DataEmbedding
from ..layers.inception import InceptionBlockV1

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True


def FFT_for_Period(x: torch.Tensor, k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    # [B, C, T]
    xf = torch.fft.rfft(x, dim=-1)
    if is_compile_mode():
        amplitudes = complex_abs(as_real(xf))
    else:
        amplitudes = torch.abs(xf)
    frequency_amplitudes = amplitudes.mean(dim=(0, 1))
    # zero out the zero-frequency component
    frequency_amplitudes[0] = -torch.inf
    top_list = torch.topk(frequency_amplitudes, k).indices.clamp_min(1)
    period = torch.div(x.shape[2], top_list , rounding_mode='floor')
    return period.to(torch.int64), amplitudes.mean(-2)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(
            self,
            window_size: int,
            top_k: int=5,
            d_model: int=64,
            d_ff: int=64,
            num_kernels: int=8
        ) -> None:
        super(TimesBlock, self).__init__()
        # TODO(AR): check if window_size is needed
        self.seq_len = window_size
        self.top_k = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            InceptionBlockV1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlockV1(d_ff, d_model, num_kernels=num_kernels)
        )
        # Hold the Inception conv weights in NHWC so cuDNN runs the convs
        # channels-last: it picks faster kernels (notably for the bf16 tensor-core
        # path) and skips the NCHW<->NHWC layout conversions seen in profiling.
        # Memory-format change only (output differs at fp rounding ~1e-5).
        self.conv = self.conv.to(memory_format=torch.channels_last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N = x.size()
        x = x.mT.contiguous()
        torch._check(T == self.seq_len, f"{T} != {self.seq_len}")
        periods, period_weight = FFT_for_Period(x, self.top_k)

        # In eager mode read all top_k periods to the host in a single D2H copy
        # instead of one blocking `.item()` sync per iteration. Under
        # torch.compile keep the per-period `.item()` so Dynamo's unbacked-symint
        # path (capture_scalar_outputs + the torch._check_is_size below) is
        # preserved unchanged.
        compile_mode = is_compile_mode()
        period_list = None if compile_mode else periods.tolist()

        # Fold each Inception block's parallel convs into a single equivalent conv ONCE,
        # before the period loop. self.conv is otherwise applied top_k times per forward
        # with identical weights, so this folds 2x instead of 2*top_k times -- the win
        # is largest on the bf16 training path, where the per-period refolds are
        # launch-bound. The folded kernels are reused across the loop within this
        # forward; in training the fold is differentiable and the gradient accumulates
        # correctly through the top_k reuses.
        inc1, act, inc2 = self.conv[0], self.conv[1], self.conv[2]
        w1, b1, p1 = inc1.folded_kernel()
        w2, b2, p2 = inc2.folded_kernel()

        # Softmax the FFT period weights up front so each period's output can be folded
        # into a running weighted sum, instead of stacking all top_k of them and then
        # reducing (which holds ~2*top_k full [B, C, T] tensors at peak).
        period_weight = F.softmax(period_weight, dim=1)

        res = None
        for i in range(self.top_k):
            period = periods[i].item() if compile_mode else period_list[i]
            torch._check_is_size(period, f"Period {period} should be [0, {T}]", max=T)
            out = F.pad(x, (0, (-self.seq_len) % period))
            torch._check(out.shape[-1] % period == 0)
            out = out.unflatten(-1, (-1, period))
            # 2D conv: from 1d Variation to 2d Variation (channels-last input). Apply the
            # pre-folded kernels directly so they are not refolded once per period.
            out = out.contiguous(memory_format=torch.channels_last)
            out = F.conv2d(out, w1, b1, padding=p1)
            out = act(out)
            out = F.conv2d(out, w2, b2, padding=p2)
            # reshape back, weight by this period's amplitude, accumulate
            out = out.flatten(-2)[..., :self.seq_len]
            term = out * period_weight[:, i].view(B, 1, 1)
            res = term if res is None else res + term
        # residual connection
        res = res + x
        return res.mT.contiguous()


class TimesNet(BaseModel):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(
        self,
        window_size: int,
        input_dim: int,
        top_k: int=5,
        d_model: int=64,
        d_ff: int=64,
        num_kernels: int=8,
        e_layers: int=2,
        dropout: float=0.1
    ) -> None:
        super(TimesNet, self).__init__()
        # Rename to window_size
        self.seq_len = window_size
        max_allowed_top_k = window_size // 2 + 1
        if top_k > max_allowed_top_k:
            print(f"top_k = {top_k}, max allowed top_k = {max_allowed_top_k}, reducing top_k.")
            top_k = max_allowed_top_k

        self.model = nn.ModuleList([nn.Sequential(TimesBlock(window_size, top_k, d_model, d_ff, num_kernels),
                                                  nn.LayerNorm(d_model),)
                                    for _ in range(e_layers)])
        self.enc_embedding = DataEmbedding(input_dim, d_model, dropout)
        self.layer = e_layers

        self.projection = nn.Linear(d_model, input_dim, bias=True)

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        x_enc = inputs[0]
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) +
                           torch.finfo(x_enc.dtype).eps).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.model[i](enc_out)
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        scale = stdev[:, 0, :].unsqueeze(1)
        bias = means[:, 0, :].unsqueeze(1)
        dec_out = dec_out * scale + bias
        return dec_out
