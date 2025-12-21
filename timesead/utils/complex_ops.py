import torch


def is_compile_mode() -> bool:
    """Return True when torch.compile/torch._dynamo is tracing/compiling."""
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling"):
        return bool(torch.compiler.is_compiling())
    if hasattr(torch._dynamo, "is_compiling"):
        return bool(torch._dynamo.is_compiling())
    return False


def as_real(x: torch.Tensor) -> torch.Tensor:
    """Return a real view for complex tensors, passthrough otherwise."""
    return torch.view_as_real(x) if torch.is_complex(x) else x


def as_complex(x: torch.Tensor) -> torch.Tensor:
    """View (..., 2) real representation back as complex when possible."""
    if torch.is_complex(x):
        return x
    if x.size(-1) != 2:
        raise ValueError("Expected last dimension of size 2 for real-to-complex view")
    return torch.view_as_complex(x)


def complex_abs(x: torch.Tensor) -> torch.Tensor:
    """Absolute value for complex tensors represented with real components."""
    if torch.is_complex(x):
        return torch.abs(x)
    real, imag = x.unbind(dim=-1)
    return torch.sqrt(real.square() + imag.square())


def complex_energy(x: torch.Tensor) -> torch.Tensor:
    """Return squared magnitude summed over the last component dimension."""
    if torch.is_complex(x):
        real = x.real.float()
        imag = x.imag.float()
    else:
        real, imag = x.unbind(dim=-1)
        real = real.float()
        imag = imag.float()
    return real.square() + imag.square()


def complex_einsum_bkhi_khio(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Complex multiply/accumulate for FourierBlock without complex dtype ops."""
    x = as_real(x)
    w = as_real(w)

    x_real, x_imag = x.unbind(dim=-1)
    w_real, w_imag = w.unbind(dim=-1)

    real = torch.einsum("bkhi,khio->bkho", x_real, w_real) - torch.einsum("bkhi,khio->bkho", x_imag, w_imag)
    imag = torch.einsum("bkhi,khio->bkho", x_real, w_imag) + torch.einsum("bkhi,khio->bkho", x_imag, w_real)
    return torch.stack((real, imag), dim=-1)


def complex_einsum_lowrank(x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Complex multiply/accumulate for low-rank FourierBlock parameters."""
    x = as_real(x)
    u = as_real(u)
    v = as_real(v)

    xr, xi = x.unbind(dim=-1)
    ur, ui = u.unbind(dim=-1)
    vr, vi = v.unbind(dim=-1)

    xr_u = torch.einsum("bkhi,khir->bkhr", xr, ur)
    xi_u = torch.einsum("bkhi,khir->bkhr", xi, ur)
    xr_ui = torch.einsum("bkhi,khir->bkhr", xr, ui)
    xi_ui = torch.einsum("bkhi,khir->bkhr", xi, ui)

    real = torch.einsum("bkhr,khor->bkho", xr_u - xi_ui, vr) - torch.einsum("bkhr,khor->bkho", xr_ui + xi_u, vi)
    imag = torch.einsum("bkhr,khor->bkho", xr_u - xi_ui, vi) + torch.einsum("bkhr,khor->bkho", xr_ui + xi_u, vr)
    return torch.stack((real, imag), dim=-1)
