import numpy as np
import pytest

torch = pytest.importorskip("torch")

from timesead.models.layers.fourier_correlation import FourierBlock


def test_fourierblock_scatter_selected_modes():
    torch.manual_seed(0)
    np.random.seed(0)

    seq_len = 8
    in_channels = out_channels = 4
    num_heads = 2
    modes = 2

    block = FourierBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        seq_len=seq_len,
        num_heads=num_heads,
        modes=modes,
        mode_select_method="random",
        mode_policy="static",
        fft_norm="backward",
        w_init="randn",
        scatter_freq=True,
    )

    # Make the frequency transform a simple copy so the expected result is easy to compute
    with torch.no_grad():
        block.weights.real.zero_()
        block.weights.imag.zero_()
        eye = torch.eye(block.Ein, block.Eout, dtype=block.weights.dtype)
        for h in range(num_heads):
            for k in range(block.weights.shape[0]):
                block.weights[k, h].copy_(eye)

    q = torch.randn(1, seq_len, num_heads, block.Ein)
    y, _ = block(q, q, q, None)

    # Manual reconstruction that scatters selected frequencies back into the full spectrum
    x_ft = torch.fft.rfft(q.float(), dim=1, norm=block.fft_norm)
    idx = block.index
    out_sel = torch.einsum("bkhi,khio->bkho", x_ft.index_select(1, idx), block.weights[: idx.numel()])
    out_ft = x_ft.new_zeros((q.size(0), x_ft.size(1), num_heads, block.Eout))
    out_ft[:, idx] = out_sel
    y_expected = torch.fft.irfft(
        out_ft.permute(0, 2, 3, 1), n=seq_len, dim=-1, norm=block.fft_norm
    ).permute(0, 3, 1, 2).to(q.dtype)

    assert torch.allclose(y, y_expected)


def test_fourierblock_no_scatter_places_selected_modes_first():
    torch.manual_seed(0)
    np.random.seed(0)

    seq_len = 8
    in_channels = out_channels = 4
    num_heads = 2
    modes = 2

    block = FourierBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        seq_len=seq_len,
        num_heads=num_heads,
        modes=modes,
        mode_select_method="random",
        mode_policy="static",
        fft_norm="backward",
        w_init="randn",
        scatter_freq=False,
    )

    with torch.no_grad():
        block.weights.real.zero_()
        block.weights.imag.zero_()
        eye = torch.eye(block.Ein, block.Eout, dtype=block.weights.dtype)
        for h in range(num_heads):
            for k in range(block.weights.shape[0]):
                block.weights[k, h].copy_(eye)

    q = torch.randn(1, seq_len, num_heads, block.Ein)
    y, _ = block(q, q, q, None)

    x_ft = torch.fft.rfft(q.float(), dim=1, norm=block.fft_norm)
    idx = block.index
    out_sel = torch.einsum("bkhi,khio->bkho", x_ft.index_select(1, idx), block.weights[: idx.numel()])
    out_ft = x_ft.new_zeros((q.size(0), x_ft.size(1), num_heads, block.Eout))
    out_ft[:, : idx.numel()] = out_sel
    y_expected = torch.fft.irfft(
        out_ft.permute(0, 2, 3, 1), n=seq_len, dim=-1, norm=block.fft_norm
    ).permute(0, 3, 1, 2).to(q.dtype)

    assert torch.allclose(y, y_expected)

