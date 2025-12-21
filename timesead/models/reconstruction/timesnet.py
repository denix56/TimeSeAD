# Implementation derived from Time Series Library https://github.com/thuml/Time-Series-Library
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

from ...models import BaseModel
from ...utils.complex_ops import as_real, complex_abs, is_compile_mode
from ..layers.embed import DataEmbedding
from ..layers.inception import InceptionBlockV1


_DPSS_CACHE: Dict[Tuple[int, int, float], Tuple[torch.Tensor, torch.Tensor]] = {}


def _dpss_windows(length: int, num_tapers: int, time_bandwidth: float = 2.5) -> Tuple[torch.Tensor, torch.Tensor]:
    cache_key = (length, num_tapers, time_bandwidth)
    if cache_key in _DPSS_CACHE:
        return _DPSS_CACHE[cache_key]

    W = time_bandwidth / length
    n = torch.arange(length, dtype=torch.float64)
    diff = n[:, None] - n[None, :]
    sinc_arg = 2 * torch.pi * W * diff
    concentration = torch.where(diff == 0, torch.full_like(diff, 2 * W), torch.sin(sinc_arg) / (torch.pi * diff))
    eigenvalues, eigenvectors = torch.linalg.eigh(concentration)

    sort_indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[sort_indices][:num_tapers].to(dtype=torch.float32)
    eigenvectors = eigenvectors[:, sort_indices][:, :num_tapers].to(dtype=torch.float32)

    _DPSS_CACHE[cache_key] = (eigenvectors, eigenvalues)
    return eigenvectors, eigenvalues


def multitaper_spectrum(
        x: torch.Tensor,
        num_tapers: int = 3,
        time_bandwidth: float = 2.5,
        eigenvalue_threshold: float = 0.85,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # [B, C, T]
    tapers, eigenvalues = _dpss_windows(x.shape[-1], num_tapers, time_bandwidth)
    tapers = tapers.to(device=x.device, dtype=x.dtype).mT  # [K, T]
    eigenvalues = eigenvalues.to(device=x.device, dtype=x.dtype)

    tapered_x = x.unsqueeze(0) * tapers[:, None, None, :]
    xf = torch.fft.rfft(tapered_x, dim=-1)
    if is_compile_mode():
        amplitudes = complex_abs(as_real(xf))
    else:
        amplitudes = torch.abs(xf)
    spectra = amplitudes.pow(2)

    quality = torch.where(eigenvalues >= eigenvalue_threshold * torch.max(eigenvalues), eigenvalues, torch.zeros_like(eigenvalues))
    if torch.sum(quality) <= 0:
        quality = torch.ones_like(eigenvalues)
    weights = quality / torch.sum(quality)

    combined_spectrum = torch.sum(weights[:, None, None, None] * spectra, dim=0)
    return combined_spectrum, weights


def FFT_for_Period(
        x: torch.Tensor,
        max_k: int = 5,
        energy_ratio: Optional[float] = None,
        num_tapers: int = 3,
        time_bandwidth: float = 2.5,
        eigenvalue_threshold: float = 0.85,
        use_multitaper: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if use_multitaper:
        combined_spectrum, _ = multitaper_spectrum(x, num_tapers, time_bandwidth, eigenvalue_threshold)
    else:
        xf = torch.fft.rfft(x, dim=-1)
        if is_compile_mode():
            amplitudes = complex_abs(as_real(xf))
        else:
            amplitudes = torch.abs(xf)
        combined_spectrum = amplitudes.pow(2)

    frequency_energy = combined_spectrum.mean(dim=(0, 1))
    if frequency_energy.numel() == 0:
        return [x.shape[2]], combined_spectrum.new_zeros((x.shape[0], 1))

    frequency_energy[0] = 0
    if frequency_energy.shape[-1] == 1:
        return [x.shape[2]], combined_spectrum.mean(-2)[:, :1]
    total_energy = frequency_energy.sum()

    if total_energy <= 0:
        return [x.shape[2]], combined_spectrum.mean(-2)[:, :1]

    if energy_ratio is not None:
        sorted_indices = torch.argsort(frequency_energy, descending=True)
        cumulative_energy = torch.cumsum(frequency_energy[sorted_indices], dim=0)
        target_energy = energy_ratio * total_energy
        cutoff_idx = torch.nonzero(cumulative_energy >= target_energy, as_tuple=False)
        if cutoff_idx.numel() == 0:
            num_selected = sorted_indices.shape[0]
        else:
            num_selected = int(cutoff_idx[0, 0].item()) + 1
        num_selected = min(num_selected, max_k)
        selected = sorted_indices[:num_selected]
    else:
        k = min(max_k, frequency_energy.shape[-1])
        selected = torch.topk(frequency_energy, k=k, dim=-1).indices

    selected = selected[selected > 0]
    if selected.numel() == 0:
        fallback_index = 1 if frequency_energy.shape[-1] > 1 else 0
        selected = torch.tensor([fallback_index], device=frequency_energy.device)

    period = torch.div(x.shape[2], selected, rounding_mode='floor')
    period_weight = combined_spectrum.mean(-2)[:, selected]
    return period.to(torch.int64).tolist(), period_weight


class TimesBlock(nn.Module):
    def __init__(
            self,
            window_size: int,
            max_k: int=5,
            d_model: int=64,
            d_ff: int=64,
            num_kernels: int=8,
            energy_ratio: Optional[float] = None,
            num_tapers: int = 3,
            time_bandwidth: float = 2.5,
            eigenvalue_threshold: float = 0.85,
            circular_padding: bool = False,
            use_spectral_norm: bool = False,
            use_multitaper: bool = False,
        ) -> None:
        super(TimesBlock, self).__init__()
        # TODO(AR): check if window_size is needed
        self.seq_len = window_size
        self.max_k = max_k
        self.energy_ratio = energy_ratio
        self.num_tapers = num_tapers
        self.time_bandwidth = time_bandwidth
        self.eigenvalue_threshold = eigenvalue_threshold
        self.use_multitaper = use_multitaper
        # parameter-efficient design
        self.conv = nn.Sequential(
            InceptionBlockV1(d_model, d_ff, num_kernels=num_kernels, circular_padding=circular_padding,
                             use_spectral_norm=use_spectral_norm),
            nn.GELU(),
            InceptionBlockV1(d_ff, d_model, num_kernels=num_kernels, circular_padding=circular_padding,
                             use_spectral_norm=use_spectral_norm)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N = x.size()
        x = x.mT.contiguous()
        assert T == self.seq_len
        periods, period_weight = FFT_for_Period(
            x,
            max_k=self.max_k,
            energy_ratio=self.energy_ratio,
            num_tapers=self.num_tapers,
            time_bandwidth=self.time_bandwidth,
            eigenvalue_threshold=self.eigenvalue_threshold,
            use_multitaper=self.use_multitaper,
        )
        res = []
        for period in periods:
            assert period > 0
            assert period <= T
            out = F.pad(x, (0, -self.seq_len % period))
            assert out.shape[-1] % period == 0
            out = out.unflatten(-1, (-1, period))
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            res.append(out.flatten(-2)[..., :self.seq_len])
        res = torch.stack(res, dim=0)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.mT[..., None, None]
        res = torch.sum(res * period_weight, 0)
        # residual connection
        res = res + x
        return res.mT.contiguous()


class TimesNet(BaseModel):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(
        self,
        window_size: int,
        input_dim: int,
        max_k: int=5,
        d_model: int=64,
        d_ff: int=64,
        num_kernels: int=8,
        e_layers: int=2,
        dropout: float=0.1,
        energy_ratio: Optional[float] = None,
        num_tapers: int = 3,
        time_bandwidth: float = 2.5,
        eigenvalue_threshold: float = 0.85,
        circular_padding: bool = False,
        use_spectral_norm: bool = False,
        use_multitaper: bool = False,
    ) -> None:
        super(TimesNet, self).__init__()
        # Rename to window_size
        self.seq_len = window_size
        self.model = nn.ModuleList([
            nn.Sequential(
                TimesBlock(
                    window_size,
                    max_k,
                    d_model,
                    d_ff,
                    num_kernels,
                    energy_ratio=energy_ratio,
                    num_tapers=num_tapers,
                    time_bandwidth=time_bandwidth,
                    eigenvalue_threshold=eigenvalue_threshold,
                    circular_padding=circular_padding,
                    use_spectral_norm=use_spectral_norm,
                    use_multitaper=use_multitaper,
                ),
                nn.LayerNorm(d_model),
            )
            for _ in range(e_layers)
        ])
        self.enc_embedding = DataEmbedding(input_dim, d_model, dropout)
        self.layer = e_layers

        self.projection = nn.Linear(d_model, input_dim, bias=True)

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        x_enc = inputs[0]
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) +
                           torch.finfo(x_enc.dtype).eps).detach()
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

