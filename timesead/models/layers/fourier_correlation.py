from __future__ import annotations

import math
import logging

import numpy as np
import torch
import torch.nn as nn

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

from ...utils.complex_ops import (
    as_complex,
    as_real,
    complex_einsum_bkhi_khio,
    complex_einsum_lowrank,
    complex_energy,
    is_compile_mode,
)


def get_frequency_modes(seq_len, modes=64, mode_select_method="random"):
    modes = min(int(modes), seq_len // 2)
    if modes <= 0:
        return np.empty((0,), dtype=np.int64)

    if mode_select_method == "random":
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

    Selection policies:
      - "static": use precomputed self.index (original behavior)
      - "topk_batch": Top-k by energy from current batch
      - "topk_running": Top-k by EMA running energy

    Notes:
      - For best compile stability, keep topk_per_head=False (shared indices across heads).
      - If lrfop=False, weights are sized to max(len(index), topk) to avoid einsum K mismatch.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seq_len: int,
        num_heads: int = 8,
        modes: int = 0,
        mode_select_method: str = "random",
        fft_norm: str | None = "backward",
        w_init: str = "random",
        freq_norm_mode: str | None = None,
        lrfop: bool = False,
        rank: int = 8,
        # --- selection ---
        mode_policy: str = "static",  # {"static","topk_batch","topk_running"}
        topk: int = 0,
        topk_exclude_dc: bool = True,
        topk_exclude_nyquist: bool = False,
        topk_per_head: bool = False,
        topk_ema: float = 0.9,
        scatter_freq: bool = False,
        debug: bool = True,
    ):
        super().__init__()
        assert in_channels % num_heads == 0
        assert out_channels % num_heads == 0

        self.seq_len = int(seq_len)
        self.num_heads = int(num_heads)
        self.Ein = in_channels // num_heads
        self.Eout = out_channels // num_heads

        # Expected frequency length for rfft on a sequence of seq_len
        # This is static for a given module instance and lets us avoid
        # data-dependent shape reasoning inside torch.compile.
        self.freq_len = (self.seq_len // 2) + 1

        self.fft_norm = None if fft_norm in (None, "backward") else fft_norm
        self.w_init = str(w_init)
        self.freq_norm_mode = freq_norm_mode
        self.lrfop = bool(lrfop)

        self.mode_policy = str(mode_policy)
        self.topk = int(topk) if self.mode_policy != "static" else 0
        self.topk = min(self.topk, self.freq_len)
        self.topk_exclude_dc = bool(topk_exclude_dc)
        self.topk_exclude_nyquist = bool(topk_exclude_nyquist)
        self.topk_per_head = bool(topk_per_head)
        self.topk_ema = float(topk_ema)
        self.scatter_freq = bool(scatter_freq)

        index = get_frequency_modes(self.seq_len, modes=modes, mode_select_method=mode_select_method)
        static_idx = torch.as_tensor(index, dtype=torch.long)
        if static_idx.numel() == 0:
            raise ValueError("FourierBlock requires at least one static frequency mode")
        if int(static_idx.max()) >= self.freq_len:
            raise ValueError(
                "Precomputed frequency indices exceed available spectrum for the configured seq_len"
            )
        self.register_buffer("index", static_idx)

        if self.topk > 0 and self.topk > self.freq_len:
            raise ValueError("topk must not exceed the available frequency bins for seq_len")

        self.scale = 1.0 / (in_channels * out_channels)

        # IMPORTANT FIX #1: ensure we have at least max_slots rows for dynamic K
        static_slots = int(self.index.numel())
        dynamic_slots = int(self.topk) if self.topk > 0 else 0
        self.max_slots = max(static_slots, dynamic_slots, 1)

        if not self.lrfop:
            self.weights = nn.Parameter(
                torch.empty(self.max_slots, self.num_heads, self.Ein, self.Eout, dtype=torch.cfloat)
            )
            self._init_complex_(self.weights, fan_in=self.Ein)
        else:
            self.rank = int(rank)
            if self.rank <= 0:
                raise ValueError("rank must be positive when lrfop=True")
            self.U0 = nn.Parameter(torch.empty(self.num_heads, self.Ein, self.rank, dtype=torch.cfloat))
            self.V0 = nn.Parameter(torch.empty(self.num_heads, self.Eout, self.rank, dtype=torch.cfloat))
            self._init_complex_lowrank_(self.U0, self.V0)

        self.register_buffer("_running_score", torch.empty(0), persistent=False)

        self.debug = debug

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
        if self.freq_norm_mode is None:
            return torch.ones_like(idx, dtype=dtype)
        x = idx.to(dtype) + 1.0
        if self.freq_norm_mode == "sqrt":
            return torch.rsqrt(x)
        if self.freq_norm_mode == "linear":
            return 1.0 / x
        raise ValueError(f"Unsupported freq_norm_mode={self.freq_norm_mode}")

    def _static_idx_valid(self) -> torch.Tensor:
        # index is validated against freq_len at construction time, so we can
        # return it directly without data-dependent checks.
        return self.index

    def _score_bins(self, x_ft: torch.Tensor) -> torch.Tensor:
        """
        x_ft: (B,F,H,Ein[,2]) complex or real representation
        returns score: (F,) if topk_per_head=False else (H,F)
        """
        B, F, H, Ein = x_ft.shape[:4]

        # Ensure the runtime frequency length matches the static expectation to
        # keep torch.compile from inserting data-dependent guards.
        assert F == self.freq_len, f"Unexpected frequency length {F}, expected {self.freq_len}"
        e = complex_energy(x_ft).sum(dim=-1)  # (B,F,H)

        if self.topk_per_head:
            score = e.mean(dim=0).transpose(0, 1).contiguous()  # (H,F)
        else:
            score = e.mean(dim=0).mean(dim=-1).contiguous()     # (F,)

        # exclusions
        if self.topk_exclude_dc:
            if self.topk_per_head:
                score[:, 0] = -float("inf")
            else:
                score[0] = -float("inf")

        if self.topk_exclude_nyquist and (F > 1) and (self.seq_len % 2 == 0):
            nyq = F - 1
            if self.topk_per_head:
                score[:, nyq] = -float("inf")
            else:
                score[nyq] = -float("inf")

        # EMA running
        if self.mode_policy.endswith("_running"):
            with torch.no_grad():
                if self._running_score.numel() == 0 or self._running_score.shape != score.shape:
                    self._running_score = score.detach().clone()
                else:
                    self._running_score.mul_(self.topk_ema).add_(score.detach(), alpha=(1.0 - self.topk_ema))
            score = self._running_score

        return score

    def _topk_idx(self, score: torch.Tensor, K: int) -> torch.Tensor:
        if self.topk_per_head:
            return torch.topk(score, k=K, dim=-1, largest=True, sorted=True).indices  # (H,K)
        return torch.topk(score, k=K, dim=0, largest=True, sorted=True).indices      # (K,)

    def _select_indices(self, x_ft: torch.Tensor) -> tuple[torch.Tensor, int]:
        B, F, H, Ein = x_ft.shape[:4]

        # Static or no-topk
        if self.mode_policy == "static" or self.topk <= 0:
            idx = self._static_idx_valid()
            K = idx.shape[0]
            assert K > 0, "No valid static modes selected (K=0)."
            return idx, K

        K = self.topk
        assert K > 0, "topk must be > 0 for dynamic policies"

        score = self._score_bins(x_ft)
        top_idx = self._topk_idx(score, K=K)

        if self.mode_policy.startswith("topk_"):
            return top_idx, K

        raise ValueError(f"Unsupported mode_policy={self.mode_policy}")

    def forward(self, q, k, v, mask):
        B, L, H, Ein = q.shape
        assert H == self.num_heads
        assert Ein == self.Ein
        assert L == self.seq_len

        with torch.autocast(device_type=q.device.type, dtype=torch.float32, enabled=False):
            x = q.float()
            x_ft_c = torch.fft.rfft(x, dim=1, norm=self.fft_norm)  # (B,F,H,Ein)

            use_real = is_compile_mode()
            x_ft = as_real(x_ft_c) if use_real else x_ft_c
            F = x_ft.shape[1]
            assert F == L // 2 + 1

            idx, K = self._select_indices(x_ft)

            # gather selected: (B,K,H,Ein)
            if idx.dim() == 1:
                x_sel = x_ft.index_select(dim=1, index=idx)
            else:
                # Only possible if topk_per_head True for topk_* policies
                idx_g = idx.transpose(0, 1).unsqueeze(0).unsqueeze(-1)  # (1,K,H,1)
                idx_g = idx_g.expand(B, -1, -1, Ein)                    # (B,K,H,Ein)
                if use_real:
                    idx_g = idx_g.unsqueeze(-1).expand(-1, -1, -1, -1, 2)  # (B,K,H,Ein,2)
                x_sel = x_ft.gather(dim=1, index=idx_g)

            if self.freq_norm_mode is not None:
                scale = (lambda t: t.view(1, K, 1, 1, 1)) if use_real else (lambda t: t.view(1, K, 1, 1))
                if idx.dim() == 1:
                    fs = self._freq_scale(idx, dtype=x_sel.real.dtype)  # (K,)
                    x_sel = x_sel * scale(fs)
                else:
                    fs_hk = self._freq_scale(idx[:, :K].reshape(-1), dtype=x_sel.real.dtype).view(H, K)
                    x_sel = x_sel * scale(fs_hk.transpose(0, 1))

            out_ft = x_ft.new_zeros((B, F, H, self.Eout, 2)) if use_real else x_ft.new_zeros((B, F, H, self.Eout))

            if not self.lrfop:
                # IMPORTANT FIX #1: weights has max_slots >= K
                W = self.weights[:K] # (K,H,Ein,Eout)
                assert W.shape[0] == x_sel.shape[1], f"{W.shape[0]} != {x_sel.shape[1]}, {self.index.shape[0]}, {K}, {x_ft.shape}"
                if use_real:
                    out_sel = complex_einsum_bkhi_khio(x_sel, as_real(W))
                else:
                    out_sel = torch.einsum("bkhi,khio->bkho", x_sel, W)
            else:
                U = self.U0.unsqueeze(0).expand(K, -1, -1, -1)  # (K,H,Ein,r)
                V = self.V0.unsqueeze(0).expand(K, -1, -1, -1)  # (K,H,Eout,r)
                if use_real:
                    out_sel = complex_einsum_lowrank(x_sel, as_real(U), as_real(V))
                else:
                    out_sel = torch.einsum("bkhi,khir,khor->bkho", x_sel, U, V)
            if self.scatter_freq:
                if idx.dim() == 1:
                    out_ft.index_copy_(1, idx, out_sel)
                else:
                    for h in range(H):
                        out_ft[:, idx[h], h, :] = out_sel[:, :, h, :]
            else:
                out_ft[:, :K] = out_sel

            if use_real:
                y_ft = as_complex(out_ft.permute(0, 2, 3, 1, 4))
            else:
                y_ft = out_ft.permute(0, 2, 3, 1)
            y = torch.fft.irfft(
                y_ft, n=L, dim=-1, norm=self.fft_norm
            ).permute(0, 3, 1, 2)
        y = y.to(q.dtype)
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
