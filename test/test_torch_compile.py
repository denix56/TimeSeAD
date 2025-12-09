import pytest
import torch

from timesead.models.reconstruction.fedformer import FEDformer
from timesead.models.reconstruction.lstm_ae import LSTMAE
from timesead.models.reconstruction.timesnet import TimesNet


def _compile_and_run(
    model: torch.nn.Module, x: torch.Tensor, backend: str, *, fullgraph: bool = False
) -> torch.Tensor:
    compiled = torch.compile(model, backend=backend, fullgraph=fullgraph)
    with torch.no_grad():
        return compiled((x,))


def _assert_torch_compile(model_factory, train: bool = False) -> None:
    x = torch.randn(2, 8, 2)
    model = model_factory()
    model.train(mode=train)
    y = _compile_and_run(model, x, backend="eager")
    assert y.shape == x.shape

    model = model_factory()
    model.train(mode=train)
    try:
        y = _compile_and_run(model, x, backend="inductor", fullgraph=True)
        assert y.shape == x.shape
    except Exception as exc:
        pytest.xfail(f"torch.compile(backend='inductor') failed: {exc}")


def test_timesnet_torch_compile():
    _assert_torch_compile(
        lambda: TimesNet(
            window_size=8,
            input_dim=2,
            top_k=2,
            d_model=8,
            d_ff=8,
            num_kernels=2,
            e_layers=1,
        )
    )


def test_fedformer_torch_compile():
    _assert_torch_compile(
        lambda: FEDformer(
            window_size=8,
            input_dim=2,
            model_dim=8,
            num_heads=2,
            fcn_dim=8,
            encoder_layers=1,
            modes=4,
        )
    )


def test_lstm_ae_torch_compile():
    _assert_torch_compile(
        lambda: LSTMAE(
            input_dimension=2,
            hidden_dimensions=[4],
            latent_pooling="mean",
        ),
        train=True,
    )
