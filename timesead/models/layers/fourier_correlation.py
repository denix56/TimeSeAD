# Implementation derived from Time Series Library https://github.com/thuml/Time-Series-Library
# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True


def get_frequency_modes(seq_len, modes=64, mode_select_method="random"):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
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

    Added policies for frequency selection:
      - "static": use precomputed self.index (original behavior)
      - "topk_batch": dynamic Top-k by energy from current batch
      - "topk_running": dynamic Top-k by EMA running energy
      - "hybrid_batch": always include static modes, fill remaining with Top-k (current batch)
      - "hybrid_running": always include static modes, fill remaining with Top-k (EMA running)

    Options (any combination works):
      - fft_norm: FFT normalization ("backward"/"forward"/"ortho" or None)
      - w_init: "random" (scaled rand) or "randn" (fan-in-ish complex normal)
      - freq_norm_mode: None (no norm) or {"sqrt","linear"} based on true freq index i
      - lrfop: low-rank per-frequency operator (rank r)
      - topk_per_head: if True, choose topk per head (more expressive, less stable)
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
        freq_norm_mode: str | None = None,  # None -> no normalization
        lrfop: bool = False,
        rank: int = 8,
        # ---- optional top-k selection / hybrid ----
        mode_policy: str = "static",  # {"static","topk_batch","topk_running","hybrid_batch","hybrid_running"}
        topk: int = 0,                # if >0, used by topk*/hybrid* policies
        topk_exclude_dc: bool = True,
        topk_exclude_nyquist: bool = False,
        topk_per_head: bool = False,  # False => shared across heads; True => per-head
        topk_ema: float = 0.9,        # only for *_running
    ):
        super().__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"

        self.seq_len = int(seq_len)
        self.num_heads = int(num_heads)
        self.Ein = in_channels // num_heads
        self.Eout = out_channels // num_heads

        self.fft_norm = None if fft_norm in (None, "backward") else fft_norm
        self.w_init = w_init
        self.freq_norm_mode = freq_norm_mode
        self.lrfop = bool(lrfop)

        # ---- selection config ----
        self.mode_policy = str(mode_policy)
        self.topk = int(topk)
        self.topk_exclude_dc = bool(topk_exclude_dc)
        self.topk_exclude_nyquist = bool(topk_exclude_nyquist)
        self.topk_per_head = bool(topk_per_head)
        self.topk_ema = float(topk_ema)

        # modes on frequency domain (sorted)
        index = get_frequency_modes(self.seq_len, modes=modes, mode_select_method=mode_select_method)
        self.register_buffer("index", torch.as_tensor(index, dtype=torch.long))

        # params
        self.scale = 1.0 / (in_channels * out_channels)

        if not self.lrfop:
            # Full per-frequency complex operator: (K, H, Ein, Eout)
            self.weights = nn.Parameter(
                torch.empty(len(self.index), self.num_heads, self.Ein, self.Eout, dtype=torch.cfloat)
            )
            self._init_complex_(self.weights, fan_in=self.Ein)
        else:
            self.rank = int(rank)
            if self.rank <= 0:
                raise ValueError("rank must be positive when lrfop=True")

            # Shared base factors (H, Ein, r) and (H, Eout, r)
            self.U0 = nn.Parameter(torch.empty(self.num_heads, self.Ein, self.rank, dtype=torch.cfloat))
            self.V0 = nn.Parameter(torch.empty(self.num_heads, self.Eout, self.rank, dtype=torch.cfloat))
            self._init_complex_lowrank_(self.U0, self.V0)

        # running spectrum score for *_running policies (initialized lazily once F is known)
        self.register_buffer("_running_score", torch.empty(0), persistent=False)

    # ---------------- init helpers ----------------

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

    # ---------------- frequency helpers ----------------

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

    def _compute_score(self, x_ft: torch.Tensor) -> torch.Tensor:
        """
        x_ft: (B, F, H, Ein), complex
        Returns:
          score: (F,) if topk_per_head=False
                 (H, F) if topk_per_head=True
        """
        # Energy per bin: |X|^2 summed over Ein, averaged over batch.
        # Compute in float32 for stability.
        e = (x_ft.real.float().square() + x_ft.imag.float().square()).sum(dim=-1)  # (B,F,H)

        if self.topk_per_head:
            score = e.mean(dim=0).transpose(0, 1).contiguous()  # (H,F)
        else:
            score = e.mean(dim=0).mean(dim=-1).contiguous()     # (F,)

        B, F, H, Ein = x_ft.shape

        # Exclusions (set to -inf so they are never chosen by topk)
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

        # EMA running (optional)
        if self.mode_policy.endswith("_running"):
            with torch.no_grad():
                if self._running_score.numel() == 0 or self._running_score.shape != score.shape:
                    self._running_score = score.detach().clone()
                else:
                    self._running_score.mul_(self.topk_ema).add_(score.detach(), alpha=(1.0 - self.topk_ema))
            score = self._running_score

        return score

    def _topk_indices_from_score(self, score: torch.Tensor, K: int) -> torch.Tensor:
        """
        score: (F,) or (H,F)
        Returns:
          idx: (K,) or (H,K), sorted=True as returned by torch.topk
        """
        if self.topk_per_head:
            return torch.topk(score, k=K, dim=-1, largest=True, sorted=True).indices  # (H,K)
        return torch.topk(score, k=K, dim=0, largest=True, sorted=True).indices      # (K,)

    def _static_indices(self, F: int) -> torch.Tensor:
        """
        Returns static indices clipped to <F : (Ks,)
        """
        if self.index.numel() == 0:
            return self.index
        valid = self.index < F
        return self.index[valid]

    def _select_indices(self, x_ft: torch.Tensor) -> tuple[torch.Tensor, int, torch.Tensor | None]:
        """
        Decide which true frequency bins to use.

        Returns:
          idx: (K,) or (H,K)
          K: int
          score: score tensor or None (for debug/optional uses)
        """
        B, F, H, Ein = x_ft.shape

        policy = self.mode_policy
        if policy == "static" or self.topk <= 0:
            idx = self._static_indices(F)
            K = int(idx.numel())
            if K <= 0:
                raise RuntimeError("No valid frequency modes selected (K=0) in static policy.")
            return idx, K, None

        if policy not in ("topk_batch", "topk_running", "hybrid_batch", "hybrid_running"):
            raise ValueError(f"Unsupported mode_policy={policy}")

        # Determine K to use in this forward pass
        K = min(int(self.topk), F)
        if K <= 0:
            raise RuntimeError("topk must be > 0 for topk*/hybrid* policies.")

        score = self._compute_score(x_ft)  # (F,) or (H,F)
        top_idx = self._topk_indices_from_score(score, K=K)  # (K,) or (H,K)

        if policy.startswith("topk_"):
            return top_idx, K, score

        # Hybrid policies: always include static modes, fill remainder with top-k (no duplicates)
        static_idx = self._static_indices(F)  # (Ks,)
        Ks = int(static_idx.numel())

        if Ks == 0:
            return top_idx, K, score

        if not self.topk_per_head:
            # shared across heads: idx is (K,)
            # 1) keep static (in ascending order), up to K
            if Ks >= K:
                return static_idx[:K], K, score

            # 2) fill remaining with top-k, skipping duplicates
            static_set = set(static_idx.tolist())
            fill = []
            for i in top_idx.tolist():
                if i not in static_set:
                    fill.append(i)
                if len(fill) >= (K - Ks):
                    break

            # If exclusions caused too few, backfill from all bins by descending score
            if len(fill) < (K - Ks):
                # make a global ranking of bins by score
                order = torch.argsort(score, descending=True)
                for i in order.tolist():
                    if i in static_set:
                        continue
                    if i in fill:
                        continue
                    fill.append(i)
                    if len(fill) >= (K - Ks):
                        break

            idx = torch.cat([static_idx, torch.as_tensor(fill, device=static_idx.device, dtype=static_idx.dtype)], dim=0)
            # gather order does not matter for "packed" output, but keep deterministic:
            # place static first, then fill (as selected).
            return idx, int(idx.numel()), score

        else:
            # per-head: top_idx is (H,K), static_idx shared
            # For each head: keep static then fill with head-specific top-k
            H = top_idx.size(0)
            out = []
            static_list = static_idx.tolist()

            # score: (H,F)
            assert score is not None and score.dim() == 2 and score.size(0) == H

            for h in range(H):
                if Ks >= K:
                    out.append(static_idx[:K])
                    continue

                static_set = set(static_list)
                fill = []
                cand = top_idx[h].tolist()
                for i in cand:
                    if i not in static_set:
                        fill.append(i)
                    if len(fill) >= (K - Ks):
                        break

                if len(fill) < (K - Ks):
                    order = torch.argsort(score[h], descending=True)
                    for i in order.tolist():
                        if i in static_set:
                            continue
                        if i in fill:
                            continue
                        fill.append(i)
                        if len(fill) >= (K - Ks):
                            break

                idx_h = torch.cat(
                    [
                        static_idx,
                        torch.as_tensor(fill, device=static_idx.device, dtype=static_idx.dtype),
                    ],
                    dim=0,
                )
                out.append(idx_h)

            idx = torch.stack(out, dim=0)  # (H,K') where K' should equal K (but be safe)
            # Ensure exactly K columns
            idx = idx[:, :K].contiguous()
            return idx, K, score

    def _gather_selected(self, x_ft: torch.Tensor, idx: torch.Tensor, K: int) -> torch.Tensor:
        """
        x_ft: (B,F,H,Ein)
        idx: (K,) or (H,K)
        Returns x_sel: (B,K,H,Ein)
        """
        B, F, H, Ein = x_ft.shape

        if idx.dim() == 1:
            # shared across heads
            return x_ft.index_select(dim=1, index=idx[:K])

        # per-head gather: idx (H,K)
        # We need indices shaped (B,K,H,Ein) to gather along dim=1
        idx_g = idx.transpose(0, 1).unsqueeze(0).unsqueeze(-1)  # (1,K,H,1)
        idx_g = idx_g.expand(B, -1, -1, Ein)                    # (B,K,H,Ein)
        return x_ft.gather(dim=1, index=idx_g)

    # ---------------- forward ----------------

    def forward(self, q, k, v, mask):
        # q: (B, L, H, Ein) -> y: (B, L, H, Eout)
        B, L, H, Ein = q.shape
        assert H == self.num_heads and Ein == self.Ein

        x_in = q
        x = q.to(torch.float32)

        # FFT over time dim=1 => (B, F, H, Ein)
        x_ft = torch.fft.rfft(x, dim=1, norm=self.fft_norm)
        F = x_ft.size(1)
        assert F == L // 2 + 1

        # Select indices (static / topk / hybrid)
        idx, K, _score = self._select_indices(x_ft)  # idx: (K,) or (H,K)

        # Gather selected bins: (B, K, H, Ein)
        x_sel = self._gather_selected(x_ft, idx, K)

        # Optional frequency-index normalization (based on true idx)
        if self.freq_norm_mode is not None:
            if idx.dim() == 1:
                fs = self._freq_scale(idx[:K], dtype=x_sel.real.dtype)  # (K,)
                x_sel = x_sel * fs.view(1, K, 1, 1)
            else:
                # per-head: idx is (H,K); scale per element
                # fs_hk: (H,K)
                fs_hk = self._freq_scale(idx[:, :K].reshape(-1), dtype=x_sel.real.dtype).view(H, K)
                # broadcast to (1,K,H,1) with transpose
                x_sel = x_sel * fs_hk.transpose(0, 1).view(1, K, H, 1)

        # Output spectrum: (B, F, H, Eout)
        out_ft = x_ft.new_zeros((B, F, H, self.Eout))

        if not self.lrfop:
            # Full operator uses the first K learned slots (packed convention).
            # W is stored for len(self.index) (static K); for topk/hybrid we still use first K slots.
            W = self.weights[:K]  # (K,H,Ein,Eout)
            out_sel = torch.einsum("bkhi,khio->bkho", x_sel, W)
        else:
            # Low-rank operator: shared across frequencies, packed convention
            U = self.U0.unsqueeze(0).expand(K, -1, -1, -1)  # (K,H,Ein,r)
            V = self.V0.unsqueeze(0).expand(K, -1, -1, -1)  # (K,H,Eout,r)
            out_sel = torch.einsum("bkhi,khir,khor->bkho", x_sel, U, V)

        # PACK into first K frequency bins (intentional)
        out_ft[:, :K] = out_sel

        # Back to time: (B, L, H, Eout)
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
