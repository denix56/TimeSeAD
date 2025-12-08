"""Triton implementations of Fourier modules.

The reference implementations live in :mod:`timesead.models.layers.fourier_correlation`.
These variants mirror their behavior but offload the heavy complex matrix multiplications
to custom Triton kernels for better GPU performance.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from .fourier_correlation import FourierBlock, FourierCrossAttention


def _check_device(q: torch.Tensor) -> None:
    if not q.is_cuda:
        raise RuntimeError("Triton Fourier kernels require CUDA tensors")


@triton.jit
def _fourier_block_kernel(
    x_real_ptr,
    x_imag_ptr,
    w_real_ptr,
    w_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    stride_x_b,
    stride_x_h,
    stride_x_e,
    stride_x_m,
    stride_w_h,
    stride_w_e,
    stride_w_o,
    stride_w_m,
    stride_out_b,
    stride_out_h,
    stride_out_o,
    stride_out_m,
    H: tl.constexpr,
    E: tl.constexpr,
    O: tl.constexpr,
    M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_o = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)

    b = pid_b // H
    h = pid_b - b * H

    mask_m = offs_m < M
    mask_o = offs_o < O

    # [BLOCK_O, BLOCK_M]
    acc_real = tl.zeros((BLOCK_O, BLOCK_M), dtype=tl.float32)
    acc_imag = tl.zeros((BLOCK_O, BLOCK_M), dtype=tl.float32)

    for e in range(0, E, BLOCK_E):
        offs_e = e + tl.arange(0, BLOCK_E)
        mask_e = offs_e < E

        x_real = tl.load(
            x_real_ptr
            + b * stride_x_b
            + h * stride_x_h
            + offs_e[:, None] * stride_x_e
            + offs_m[None, :] * stride_x_m,
            mask=mask_e[:, None] & mask_m[None, :],
            other=0.0,
        )
        x_imag = tl.load(
            x_imag_ptr
            + b * stride_x_b
            + h * stride_x_h
            + offs_e[:, None] * stride_x_e
            + offs_m[None, :] * stride_x_m,
            mask=mask_e[:, None] & mask_m[None, :],
            other=0.0,
        )

        w_real = tl.load(
            w_real_ptr
            + h * stride_w_h
            + offs_e[:, None] * stride_w_e
            + offs_o[None, :] * stride_w_o
            + offs_m[None, :] * stride_w_m,
            mask=mask_e[:, None] & mask_o[None, :] & mask_m[None, :],
            other=0.0,
        )
        w_imag = tl.load(
            w_imag_ptr
            + h * stride_w_h
            + offs_e[:, None] * stride_w_e
            + offs_o[None, :] * stride_w_o
            + offs_m[None, :] * stride_w_m,
            mask=mask_e[:, None] & mask_o[None, :] & mask_m[None, :],
            other=0.0,
        )

        # complex multiply accumulate
        acc_real += tl.dot(w_real, x_real) - tl.dot(w_imag, x_imag)
        acc_imag += tl.dot(w_real, x_imag) + tl.dot(w_imag, x_real)

    tl.store(
        out_real_ptr
        + b * stride_out_b
        + h * stride_out_h
        + offs_o[:, None] * stride_out_o
        + offs_m[None, :] * stride_out_m,
        acc_real,
        mask=mask_o[:, None] & mask_m[None, :],
    )
    tl.store(
        out_imag_ptr
        + b * stride_out_b
        + h * stride_out_h
        + offs_o[:, None] * stride_out_o
        + offs_m[None, :] * stride_out_m,
        acc_imag,
        mask=mask_o[:, None] & mask_m[None, :],
    )


@triton.jit
def _complex_bmm_kernel(
    a_real_ptr,
    a_imag_ptr,
    b_real_ptr,
    b_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    stride_ab,
    stride_ah,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bh,
    stride_bk,
    stride_bn,
    stride_ob,
    stride_oh,
    stride_om,
    stride_on,
    H: tl.constexpr,
    K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    b = pid_b // H
    h = pid_b - b * H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc_real = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_imag = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a_real = tl.load(
            a_real_ptr
            + b * stride_ab
            + h * stride_ah
            + offs_m[:, None] * stride_am
            + offs_k[None, :] * stride_ak,
            mask=mask_k[None, :],
            other=0.0,
        )
        a_imag = tl.load(
            a_imag_ptr
            + b * stride_ab
            + h * stride_ah
            + offs_m[:, None] * stride_am
            + offs_k[None, :] * stride_ak,
            mask=mask_k[None, :],
            other=0.0,
        )

        b_real = tl.load(
            b_real_ptr
            + b * stride_bb
            + h * stride_bh
            + offs_k[:, None] * stride_bk
            + offs_n[None, :] * stride_bn,
            mask=mask_k[:, None],
            other=0.0,
        )
        b_imag = tl.load(
            b_imag_ptr
            + b * stride_bb
            + h * stride_bh
            + offs_k[:, None] * stride_bk
            + offs_n[None, :] * stride_bn,
            mask=mask_k[:, None],
            other=0.0,
        )

        acc_real += tl.dot(a_real, b_real) - tl.dot(a_imag, b_imag)
        acc_imag += tl.dot(a_real, b_imag) + tl.dot(a_imag, b_real)

    tl.store(
        out_real_ptr
        + b * stride_ob
        + h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_n[None, :] * stride_on,
        acc_real,
    )
    tl.store(
        out_imag_ptr
        + b * stride_ob
        + h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_n[None, :] * stride_on,
        acc_imag,
    )


class FourierBlockTriton(FourierBlock):
    """Triton-accelerated variant of :class:`FourierBlock`."""

    def forward(self, q, k, v, mask):  # type: ignore[override]
        _check_device(q)
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)
        x_ft = torch.fft.rfft(x.to(torch.float32), dim=-1)

        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        index = self.index[self.index < x_ft.shape[3]]
        if index.numel() == 0:
            return torch.fft.irfft(out_ft, n=x.size(-1)).to(x.dtype), None

        x_sel = x_ft[..., index].contiguous()
        w_sel = self.weights[..., : index.shape[0]].contiguous()

        # Prepare real/imag buffers
        x_real, x_imag = x_sel.real, x_sel.imag
        w_real, w_imag = w_sel.real, w_sel.imag
        out_real = torch.empty(B, H, E, index.shape[0], device=x.device, dtype=torch.float32)
        out_imag = torch.empty_like(out_real)

        BH = B * H
        grid = (
            triton.cdiv(index.shape[0], BLOCK_M),
            triton.cdiv(E, BLOCK_O),
            BH,
        )
        _fourier_block_kernel[grid](
            x_real,
            x_imag,
            w_real,
            w_imag,
            out_real,
            out_imag,
            x_real.stride(0),
            x_real.stride(1),
            x_real.stride(2),
            x_real.stride(3),
            w_real.stride(0),
            w_real.stride(1),
            w_real.stride(2),
            w_real.stride(3),
        out_real.stride(0),
        out_real.stride(1),
        out_real.stride(2),
        out_real.stride(3),
        H=H,
        E=E,
        O=E,
        M=index.shape[0],
    )

        out_complex = torch.complex(out_real, out_imag)
        out_ft[..., : index.shape[0]] = out_complex
        x_time = torch.fft.irfft(out_ft, n=x.size(-1)).to(x.dtype)
        return (x_time, None)


class FourierCrossAttentionTriton(FourierCrossAttention):
    """Triton-accelerated variant of :class:`FourierCrossAttention`."""

    def forward(self, q, k, v, mask):  # type: ignore[override]
        _check_device(q)
        B, Lq, H, _ = q.shape
        _, Lkv, _, _ = k.shape

        xq = q.permute(0, 2, 3, 1)
        xk = k.permute(0, 2, 3, 1)

        xq_ft = torch.fft.rfft(xq.to(torch.float32), dim=-1)
        xk_ft = torch.fft.rfft(xk.to(torch.float32), dim=-1)

        index_q = self.index_q[self.index_q < xq_ft.shape[3]]
        index_kv = self.index_kv[self.index_kv < xk_ft.shape[3]]

        if index_q.numel() == 0 or index_kv.numel() == 0:
            return torch.zeros_like(q), None

        xq_sel = xq_ft[..., index_q].contiguous()
        xk_sel = xk_ft[..., index_kv].contiguous()

        # compute xqk_ft via Triton batch matmul
        a_real, a_imag = xq_sel.permute(0, 1, 3, 2).real.contiguous(), xq_sel.permute(0, 1, 3, 2).imag.contiguous()
        b_real, b_imag = xk_sel.real, xk_sel.imag
        Mq, Ein = a_real.shape[2], a_real.shape[3]
        Mk = b_real.shape[3]

        out_real = torch.empty(B, H, Mq, Mk, device=q.device, dtype=torch.float32)
        out_imag = torch.empty_like(out_real)

        BH = B * H
        grid = (
            triton.cdiv(Mq, BLOCK_M),
            triton.cdiv(Mk, BLOCK_N),
            BH,
        )
        _complex_bmm_kernel[grid](
            a_real,
            a_imag,
            b_real,
            b_imag,
            out_real,
            out_imag,
            a_real.stride(0),
            a_real.stride(1),
            a_real.stride(2),
            a_real.stride(3),
            b_real.stride(0),
            b_real.stride(1),
            b_real.stride(2),
            b_real.stride(3),
            out_real.stride(0),
            out_real.stride(1),
            out_real.stride(2),
            out_real.stride(3),
            H=H,
            K=Ein,
        )

        xqk_ft = torch.complex(out_real, out_imag)

        if self.activation == 'tanh':
            xqk_ft = torch.complex(xqk_ft.real.tanh(), xqk_ft.imag.tanh())
        elif self.activation == 'softmax':
            attn = torch.softmax(xqk_ft.abs(), dim=-1)
            xqk_ft = torch.complex(attn, torch.zeros_like(attn))
        else:
            raise Exception(f"{self.activation} activation function is not implemented")

        weights_c = self.weights[..., : index_q.shape[0]].contiguous()

        # reuse kernel to multiply xqk_ft (Mq, Mk) with xk_sel (E, Mk) -> temp (Mq, E)
        attn_real = xqk_ft.real
        attn_imag = xqk_ft.imag
        xk_real = xk_sel.permute(0, 1, 3, 2).real.contiguous()
        xk_imag = xk_sel.permute(0, 1, 3, 2).imag.contiguous()

        temp_real = torch.empty(B, H, Mq, xk_real.shape[3], device=q.device, dtype=torch.float32)
        temp_imag = torch.empty_like(temp_real)

        grid_temp = (
            triton.cdiv(Mq, BLOCK_M),
            triton.cdiv(xk_real.shape[3], BLOCK_N),
            BH,
        )
        _complex_bmm_kernel[grid_temp](
            attn_real,
            attn_imag,
            xk_real,
            xk_imag,
            temp_real,
            temp_imag,
            attn_real.stride(0),
            attn_real.stride(1),
            attn_real.stride(2),
            attn_real.stride(3),
            xk_real.stride(0),
            xk_real.stride(1),
            xk_real.stride(2),
            xk_real.stride(3),
            temp_real.stride(0),
            temp_real.stride(1),
            temp_real.stride(2),
            temp_real.stride(3),
            H=H,
            K=Mk,
        )

        temp = torch.complex(temp_real, temp_imag)

        # final projection with weights
        proj_out = torch.zeros(B, H, self.head_dim_out, xq_ft.shape[3], device=q.device, dtype=torch.cfloat)
        temp_sel = temp.permute(0, 1, 3, 2)  # [B, H, E, Mq]

        # einsum on GPU, much smaller than previous ops
        proj_sel = torch.einsum("bhey,heox->bhox", temp_sel, weights_c)
        proj_out[..., index_q] = proj_sel

        out = torch.fft.irfft(
            proj_out / (self.in_channels * self.out_channels),
            n=xq.size(-1),
        )
        return out.to(q.dtype), None


# Kernel tuning constants
BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 32
BLOCK_O = 32
BLOCK_E = 32

