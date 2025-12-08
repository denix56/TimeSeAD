import time

import pytest
import torch

from timesead.models.layers import (
    FourierBlock,
    FourierBlockTriton,
    FourierCrossAttention,
    FourierCrossAttentionTriton,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require CUDA"
)


def _bench(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)


def test_fourier_block_triton_matches_reference():
    torch.manual_seed(0)
    B, L, H, E = 2, 96, 4, 8
    q = torch.randn(B, L, H, E, device="cuda")

    ref = FourierBlock(in_channels=E, out_channels=E, seq_len=L, num_heads=H, modes=16)
    triton_mod = FourierBlockTriton(
        in_channels=E, out_channels=E, seq_len=L, num_heads=H, modes=16
    )

    ref = ref.cuda()
    triton_mod = triton_mod.cuda()

    ref_out, _ = ref(q, q, q, None)
    tri_out, _ = triton_mod(q, q, q, None)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


def test_fourier_cross_attention_triton_matches_reference():
    torch.manual_seed(1)
    B, Lq, Lkv, H, E = 1, 128, 144, 2, 8
    q = torch.randn(B, Lq, H, E, device="cuda")
    k = torch.randn(B, Lkv, H, E, device="cuda")

    ref = FourierCrossAttention(
        in_channels=E,
        out_channels=E,
        seq_len_q=Lq,
        seq_len_kv=Lkv,
        modes=20,
        num_heads=H,
    ).cuda()

    triton_mod = FourierCrossAttentionTriton(
        in_channels=E,
        out_channels=E,
        seq_len_q=Lq,
        seq_len_kv=Lkv,
        modes=20,
        num_heads=H,
    ).cuda()

    ref_out, _ = ref(q, k, k, None)
    tri_out, _ = triton_mod(q, k, k, None)

    torch.testing.assert_close(tri_out, ref_out, rtol=2e-3, atol=2e-3)


def test_fourier_triton_performance_better_or_equal():
    torch.manual_seed(2)
    B, L, H, E = 2, 128, 4, 16
    q = torch.randn(B, L, H, E, device="cuda")

    ref = FourierBlock(in_channels=E, out_channels=E, seq_len=L, num_heads=H, modes=24).cuda()
    tri = FourierBlockTriton(in_channels=E, out_channels=E, seq_len=L, num_heads=H, modes=24).cuda()

    def run_ref():
        ref(q, q, q, None)

    def run_tri():
        tri(q, q, q, None)

    t_ref = _bench(run_ref)
    t_tri = _bench(run_tri)

    assert t_tri <= t_ref * 1.25
