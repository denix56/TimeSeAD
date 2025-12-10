# Implementation derived from Time Series Library https://github.com/thuml/Time-Series-Library
from typing import Tuple

import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

from ...models import BaseModel
from ..layers.embed import DataEmbedding
from ..layers.inception import InceptionBlockV1


def FFT_for_Period(x: torch.Tensor, k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    amplitudes = torch.abs(xf)
    frequency_amplitudes = amplitudes.mean(0).mean(-1)
    # zero out the zero-frequency component
    frequency_amplitudes[0] = 0
    top_list = torch.topk(frequency_amplitudes, k).indices
    period = torch.div(x.shape[1], top_list, rounding_mode='floor')
    return period, amplitudes.mean(-1)[:, top_list]


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.top_k)

        periods = period_list.to(torch.int64)
        res = []
        for period in periods:
            period_int = int(period)
            padding_length = int(torch.remainder(-self.seq_len, period))
            padding = x.new_zeros((B, padding_length, N))
            out = torch.cat((x, padding), dim=1)
            length = out.shape[1]
            # reshape
            torch._check_is_size(period_int, max=out.shape[1])
            out = out.reshape(B, length // period_int, period_int, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :self.seq_len, :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.view(period_weight.shape[0], 1, 1, -1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


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
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + torch.finfo(x_enc.dtype).eps).detach()
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

