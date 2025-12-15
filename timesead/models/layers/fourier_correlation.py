# Implementation derived from Time Series Library https://github.com/thuml/Time-Series-Library
# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com

import math
import numpy as np
import torch
import torch.nn as nn
torch._dynamo.config.capture_scalar_outputs = True


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random':
        index = np.arange(0, seq_len // 2)
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = np.arange(0, modes)
    index.sort()
    return index


class FourierBlock(nn.Module):
    """
    Packed Fourier block: selected true modes are written into the first K bins.

    Options (any combination works):
      - fft_norm: FFT normalization ("backward"/"forward"/"ortho" or None)
      - w_init: "random" (scaled rand) or "randn" (fan-in-ish complex normal)
      - residual: add residual at the end (only if Ein==Eout)
      - freq_norm_mode: None (no norm) or {"sqrt","linear"} based on true freq index i
      - lrfop: low-rank per-frequency operator (rank r)

    Layout:
      q: (B, L, H, Ein)
      output: (B, L, H, Eout)
      spectrum: (B, F, H, *)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seq_len: int,
        num_heads: int = 8,
        modes: int = 0,
        mode_select_method: str = "random",
        fft_norm: str = "backward",
        w_init: str = "random",
        freq_norm_mode: str | None = None,  # None -> no normalization
        lrfop: bool = False,
        rank: int = 8,
    ):
        super().__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"

        self.seq_len = seq_len
        self.num_heads = num_heads
        self.Ein = in_channels // num_heads
        self.Eout = out_channels // num_heads

        self.fft_norm = None if fft_norm in (None, "backward") else fft_norm
        self.w_init = w_init
        self.freq_norm_mode = freq_norm_mode
        self.lrfop = lrfop

        # modes on frequency domain (sorted)
        index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        self.register_buffer("index", torch.as_tensor(index, dtype=torch.long))

        # params
        self.scale = 1.0 / (in_channels * out_channels)

        if not self.lrfop:
            # Full per-frequency complex operator: (K, H, Ein, Eout)
            self.weights = nn.Parameter(torch.empty(len(self.index), num_heads, self.Ein, self.Eout, dtype=torch.cfloat))
            self._init_complex_(self.weights, fan_in=self.Ein)
        else:
            self.rank = int(rank)
            if self.rank <= 0:
                raise ValueError("rank must be positive when lrfop=True")

            # Shared base factors (H, Ein, r) and (H, Eout, r)
            self.U0 = nn.Parameter(torch.empty(num_heads, self.Ein, self.rank, dtype=torch.cfloat))
            self.V0 = nn.Parameter(torch.empty(num_heads, self.Eout, self.rank, dtype=torch.cfloat))
            self._init_complex_lowrank_(self.U0, self.V0)

    def _init_complex_(self, p: torch.Tensor, fan_in: int) -> None:
        if self.w_init == "random":
            with torch.no_grad():
                p.real.uniform_(0.0, 1.0).mul_(self.scale)
                p.imag.uniform_(0.0, 1.0).mul_(self.scale)
        elif self.w_init == "randn":
            std = 1.0 / math.sqrt(max(fan_in, 1))
            with torch.no_grad():
                p.real.normal_(0.0, std)
                p.imag.normal_(0.0, std)
        else:
            raise ValueError("w_init must be 'random' or 'randn'")

    def _init_complex_lowrank_(self, U0: torch.Tensor, V0: torch.Tensor) -> None:
        u_std = 1.0 / math.sqrt(max(self.Ein, 1))
        v_std = 1.0 / math.sqrt(max(self.rank, 1))
        if self.w_init == "random":
            with torch.no_grad():
                U0.real.uniform_(-1.0, 1.0).mul_(0.5 * u_std)
                U0.imag.uniform_(-1.0, 1.0).mul_(0.5 * u_std)
                V0.real.uniform_(-1.0, 1.0).mul_(0.5 * v_std)
                V0.imag.uniform_(-1.0, 1.0).mul_(0.5 * v_std)
        elif self.w_init == "randn":
            with torch.no_grad():
                U0.real.normal_(0.0, u_std)
                U0.imag.normal_(0.0, u_std)
                V0.real.normal_(0.0, v_std)
                V0.imag.normal_(0.0, v_std)
        else:
            raise ValueError("w_init must be 'random' or 'randn'")

    def _freq_scale(self, idx: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # idx: (K,) true frequency bins
        if self.freq_norm_mode is None:
            return torch.ones_like(idx, dtype=dtype)
        x = idx.to(dtype) + 1.0
        if self.freq_norm_mode == "sqrt":
            return torch.rsqrt(x)   # 1/sqrt(i+1)
        if self.freq_norm_mode == "linear":
            return 1.0 / x          # 1/(i+1)
        raise ValueError(f"Unsupported freq_norm_mode={self.freq_norm_mode}")

    def _freq_pos(self, idx: torch.Tensor, F: int, dtype: torch.dtype) -> torch.Tensor:
        # normalized true frequency position in [0,1]
        denom = float(max(F - 1, 1))
        return (idx.to(dtype) / denom).view(idx.numel(), 1)

    def forward(self, q, k, v, mask):
        # q: (B, L, H, Ein) -> y: (B, L, H, Eout)
        B, L, H, Ein = q.shape
        assert H == self.num_heads and Ein == self.Ein

        x_in = q
        x = q.to(torch.float32)

        # FFT over time dim=1 => (B, F, H, Ein)
        x_ft = torch.fft.rfft(x, dim=1, norm=self.fft_norm)
        F = x_ft.size(1)

        valid_mask = self.index < F
        idx = self.index[valid_mask]  # (K,)
        K = idx.numel()
        if K == 0:
            return (x_in, None)

        # Gather selected bins: (B, K, H, Ein)
        x_sel = x_ft.index_select(dim=1, index=idx)

        # Optional frequency-index normalization (based on true idx)
        if self.freq_norm_mode is not None:
            fs = self._freq_scale(idx, dtype=x_sel.real.dtype)
            x_sel = x_sel * fs.view(1, K, 1, 1)

        # Output spectrum: (B, F, H, Eout)
        out_ft = x_ft.new_zeros((B, F, H, self.Eout))

        if not self.lrfop:
            # Full operator: (B,K,H,Ein) x (K,H,Ein,Eout) -> (B,K,H,Eout)
            W = self.weights[:K]  # (K,H,Ein,Eout)
            out_sel = torch.einsum("bkhi,khio->bkho", x_sel, W)

        else:
            # Low-rank
            U = self.U0.unsqueeze(0).expand(K, -1, -1, -1)         # (K,H,Ein,r)
            V = self.V0.unsqueeze(0).expand(K, -1, -1, -1)         # (K,H,Eout,r)

            # One-einsum low-rank apply:
            # out[b,k,h,o] = sum_{i,r} x_sel[b,k,h,i] * U[k,h,i,r] * V[k,h,o,r]
            out_sel = torch.einsum("bkhi,khir,khor->bkho", x_sel, U, V)

        # PACK into first K frequency bins (intentional)
        out_ft[:, :K] = out_sel

        # Back to time: (B, L, H, Eout)
        y = torch.fft.irfft(out_ft, n=L, dim=1, norm=self.fft_norm).to(x_in.dtype)

        return (y, None)


# ########## Fourier Cross Former ####################
class FourierCrossAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        seq_len_q,
        seq_len_kv,
        modes=64,
        mode_select_method='random',
        activation='tanh',
        policy=0,
        num_heads=8,
    ):
        super(FourierCrossAttention, self).__init__()
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform,
        attention mechanism and Inverse FFT.
        """
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim_in = in_channels // num_heads
        self.head_dim_out = out_channels // num_heads

        # get modes and store as buffers (like in FourierBlock)
        index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
        index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)
        self.register_buffer('index_q', torch.from_numpy(index_q))
        self.register_buffer('index_kv', torch.from_numpy(index_kv))

        self.scale = 1.0 / (in_channels * out_channels)

        # complex weights encoded as (..., 2) real tensor, similar to FourierBlock
        self.weights = nn.Parameter(
            self.scale
            * torch.rand(
                len(index_q),
                num_heads,
                self.head_dim_in,
                self.head_dim_out,
                dtype=torch.cfloat,
            )
        )

    def forward(self, q, k, v, mask):
        # q, k, v: [B, L, H, E]
        B, Lq, H, E = q.shape
        assert H == self.num_heads
        _, Lkv, _, _ = k.shape

        xq = q.to(torch.float32)
        xk = k.to(torch.float32)
        # original code does not actually use v; keep behavior
        # xv = v.permute(0, 2, 3, 1)

        # FFT
        xq_ft = torch.fft.rfft(xq, dim=1)   # [B, H, E, Lq//2+1]
        xk_ft = torch.fft.rfft(xk, dim=1)   # [B, H, E, Lkv//2+1]

        freq_len_q = xq_ft.size(1)
        freq_len_kv = xk_ft.size(1)
        valid_q = self.index_q < freq_len_q
        valid_kv = self.index_kv < freq_len_kv
        index_q = self.index_q[valid_q]
        index_kv = self.index_kv[valid_kv]

        # select requested modes in one shot (no Python loop)
        xq_ft_sel = xq_ft[:, index_q]      # [B, H, E, Mq]
        xk_ft_sel = xk_ft[:, index_kv]     # [B, H, E, Mk]

        # attention in frequency domain:
        # xqk_ft: [B, H, Mq, Mk]
        xqk_ft = torch.einsum('blhe,bmhe->bmhl',xq_ft_sel, xk_ft_sel)

        if self.activation == 'tanh':
            # TODO: check - do we perform per chanel tanh or complex tanh
            xqk_ft = torch.view_as_complex(torch.view_as_real(xqk_ft).tanh())
        elif self.activation == 'softmax':
            attn = torch.softmax(xqk_ft.abs(), dim=1)
            xqk_ft = torch.complex(attn, torch.zeros_like(attn))
        else:
            raise Exception(f'{self.activation} activation function is not implemented')

        # linear transform with complex weights
        weights_c = self.weights[valid_q]    # [H, Ein, Eout, Mq]

        # combine with keys again: [B, H, E, Mq]
        xqkvw = torch.einsum("bmhl,bmhe,lhek->blhk", xqk_ft, xk_ft_sel, weights_c)

        # place selected freqs back into full spectrum
        out_ft = xqk_ft.new_zeros(B, freq_len_q, H, self.head_dim_out)
        out_ft[:, valid_q] = xqkvw

        # iFFT back to time domain
        out = torch.fft.irfft(
            out_ft / (self.in_channels * self.out_channels),
            n=xq.size(-1),
            dim=1,
        )  # [B, L, H, Eout]

        return out.to(q.dtype), None
