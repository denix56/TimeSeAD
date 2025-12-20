from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True


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
      - "hybrid_batch": union(static, topk_batch) with order = [static, topk_new], truncated/padded to K
      - "hybrid_running": union(static, topk_running) with same rule

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
        mode_policy: str = "static",  # {"static","topk_batch","topk_running","hybrid_batch","hybrid_running"}
        topk: int = 0,
        topk_exclude_dc: bool = True,
        topk_exclude_nyquist: bool = False,
        topk_per_head: bool = False,  # supported, but hybrid is implemented most robustly with False
        topk_ema: float = 0.9,
    ):
        super().__init__()
        assert in_channels % num_heads == 0
        assert out_channels % num_heads == 0

        self.seq_len = int(seq_len)
        self.num_heads = int(num_heads)
        self.Ein = in_channels // num_heads
        self.Eout = out_channels // num_heads

        self.fft_norm = None if fft_norm in (None, "backward") else fft_norm
        self.w_init = str(w_init)
        self.freq_norm_mode = freq_norm_mode
        self.lrfop = bool(lrfop)

        self.mode_policy = str(mode_policy)
        self.topk = int(topk)
        self.topk_exclude_dc = bool(topk_exclude_dc)
        self.topk_exclude_nyquist = bool(topk_exclude_nyquist)
        self.topk_per_head = bool(topk_per_head)
        self.topk_ema = float(topk_ema)

        index = get_frequency_modes(self.seq_len, modes=modes, mode_select_method=mode_select_method)
        self.register_buffer("index", torch.as_tensor(index, dtype=torch.long))

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

    def _static_idx_valid(self, F: int) -> torch.Tensor:
        if self.index.numel() == 0:
            return self.index
        return self.index[self.index < F]

    def _score_bins(self, x_ft: torch.Tensor) -> torch.Tensor:
        """
        x_ft: (B,F,H,Ein) complex
        returns score: (F,) if topk_per_head=False else (H,F)
        """
        B, F, H, Ein = x_ft.shape
        e = (x_ft.real.float().square() + x_ft.imag.float().square()).sum(dim=-1)  # (B,F,H)

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

    def _select_idx_shared(self, static_idx: torch.Tensor, top_idx: torch.Tensor, score: torch.Tensor, K: int) -> torch.Tensor:
        """
        Hybrid selection (shared across heads), compile-friendly, no Python loops.

        Order: [static_idx, top_idx_not_in_static, backfill_by_score_not_in_union] then truncate to K.
        """
        device = static_idx.device
        dtype = static_idx.dtype

        static_idx = static_idx[:K]
        n_static = static_idx.numel()

        if n_static >= K:
            return static_idx

        # remove duplicates from top_idx wrt static
        if n_static > 0:
            mask_new = ~torch.isin(top_idx, static_idx)
            top_new = top_idx[mask_new]
        else:
            top_new = top_idx

        need = K - n_static
        top_new = top_new[:need]

        idx = torch.cat([static_idx, top_new], dim=0)

        if idx.numel() >= K:
            return idx[:K]

        # backfill: take remaining highest-score bins not in idx
        F = score.numel()
        order = torch.argsort(score, descending=True)  # (F,)
        mask_fill = ~torch.isin(order, idx)
        fill = order[mask_fill][: (K - idx.numel())]

        idx = torch.cat([idx, fill.to(device=device, dtype=dtype)], dim=0)
        return idx[:K]

    def _select_indices(self, x_ft: torch.Tensor) -> tuple[torch.Tensor, int]:
        B, F, H, Ein = x_ft.shape

        # Static or no-topk
        if self.mode_policy == "static" or self.topk <= 0:
            idx = self._static_idx_valid(F)
            K = int(idx.numel())
            if K <= 0:
                raise RuntimeError("No valid static modes selected (K=0).")
            return idx, K

        K = min(self.topk, F)
        if K <= 0:
            raise RuntimeError("topk must be > 0 for dynamic policies")

        score = self._score_bins(x_ft)
        top_idx = self._topk_idx(score, K=K)

        if self.mode_policy.startswith("topk_"):
            return top_idx, K

        # Hybrid: for best compile stability, do shared selection.
        # If you truly need per-head hybrid, I can provide it, but it is heavier.
        if self.topk_per_head:
            raise RuntimeError("hybrid_* with topk_per_head=True is intentionally disabled for compile stability.")

        static_idx = self._static_idx_valid(F)
        idx = self._select_idx_shared(static_idx, top_idx, score, K=K)
        return idx, int(idx.numel())

    def forward(self, q, k, v, mask):
        B, L, H, Ein = q.shape
        assert H == self.num_heads
        assert Ein == self.Ein
        assert L == self.seq_len

        x_in = q
        x = q.to(torch.float32)

        x_ft = torch.fft.rfft(x, dim=1, norm=self.fft_norm)  # (B,F,H,Ein)
        F = x_ft.size(1)
        assert F == L // 2 + 1

        idx, K = self._select_indices(x_ft)

        # gather selected: (B,K,H,Ein)
        if idx.dim() == 1:
            x_sel = x_ft.index_select(dim=1, index=idx)
        else:
            # Only possible if topk_per_head True for topk_* policies (not hybrid)
            idx_g = idx.transpose(0, 1).unsqueeze(0).unsqueeze(-1)  # (1,K,H,1)
            idx_g = idx_g.expand(B, -1, -1, Ein)                    # (B,K,H,Ein)
            x_sel = x_ft.gather(dim=1, index=idx_g)

        if self.freq_norm_mode is not None:
            if idx.dim() == 1:
                fs = self._freq_scale(idx, dtype=x_sel.real.dtype)  # (K,)
                x_sel = x_sel * fs.view(1, K, 1, 1)
            else:
                fs_hk = self._freq_scale(idx[:, :K].reshape(-1), dtype=x_sel.real.dtype).view(H, K)
                x_sel = x_sel * fs_hk.transpose(0, 1).view(1, K, H, 1)

        #out_ft = x_ft.new_zeros((B, F, H, self.Eout))

        if not self.lrfop:
            # IMPORTANT FIX #1: weights has max_slots >= K
            W = self.weights  # (K,H,Ein,Eout)
            out_sel = torch.einsum("bkhi,khio->bkho", x_sel, W)
        else:
            U = self.U0.unsqueeze(0).expand(K, -1, -1, -1)  # (K,H,Ein,r)
            V = self.V0.unsqueeze(0).expand(K, -1, -1, -1)  # (K,H,Eout,r)
            out_sel = torch.einsum("bkhi,khir,khor->bkho", x_sel, U, V)

        out_ft = out_sel

        y = torch.fft.irfft(
            out_ft.permute(0, 2, 3, 1), n=L, dim=-1, norm=self.fft_norm
        ).permute(0, 3, 1, 2).to(x_in.dtype)

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
