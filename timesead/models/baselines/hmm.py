from typing import List, Optional, Tuple
import warnings

import torch

from ..common import AnomalyDetector

# NOTE: pomegranate imports `apricot` eagerly at module load. The GMM-/Gaussian-HMM
# paths below never call apricot, but the import must resolve — see the project
# dependencies (apricot-select installed with a modern numba via the numba>=0.60
# override).
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal
from pomegranate.gmm import GeneralMixtureModel


def _pomegranate_covariance_type(covariance_type: str) -> str:
    # pomegranate's Normal supports 'full' and 'diag'. hmmlearn's 'tied'/'spherical'
    # have no direct equivalent, so fall back to the cheap diagonal form.
    return "full" if covariance_type == "full" else "diag"


class _BaseHMMAnomalyDetector(AnomalyDetector):
    """HMM baseline backed by pomegranate (PyTorch, GPU-capable).

    Replaces the previous hmmlearn implementation. Fitting and scoring are batched
    over *all* windows in a single tensor operation (pomegranate consumes a 3D
    ``(n_windows, window_length, features)`` tensor) and run on a GPU whenever one
    is visible to the process, instead of hmmlearn's single-threaded EM and the
    per-window ``model.score`` Python loop.
    """

    def __init__(self, model: DenseHMM, batch_first: bool = True):
        super(_BaseHMMAnomalyDetector, self).__init__()
        # Keep the pomegranate model out of the nn.Module registry (store it inside
        # a list) so Lightning's device placement does not move it from under the
        # explicit device handling in fit()/score().
        self._hmm: List[DenseHMM] = [model]
        self.batch_first = batch_first
        self._fitted = False

    @property
    def model(self) -> DenseHMM:
        return self._hmm[0]

    def _device(self) -> torch.device:
        # Prefer a visible GPU; otherwise fall back to the detector's own device.
        if torch.cuda.is_available():
            return torch.device("cuda")
        return self.dummy.device

    def _make_batch_first(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            return inputs
        return inputs.transpose(0, 1)

    def fit(self, dataset: torch.utils.data.DataLoader, **kwargs) -> None:
        windows: List[torch.Tensor] = []
        length: Optional[int] = None
        for batch_inputs, _ in dataset:
            (batch_input,) = batch_inputs
            batch_input = self._make_batch_first(batch_input).detach().to(torch.float32).cpu()
            if length is None:
                length = batch_input.shape[1]
            if batch_input.shape[1] != length:
                warnings.warn(
                    "pomegranate HMM baseline expects equal-length windows; "
                    "skipping a batch with a mismatched window length.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            windows.append(batch_input)

        if not windows:
            return

        x = torch.cat(windows, dim=0)  # (n_windows, window_length, features)
        device = self._device()
        try:
            model = self.model.to(device)
            model.fit(x.to(device))
            self._hmm[0] = model
            self._fitted = True
        except Exception as exc:
            warnings.warn(f"Failed to fit HMM baseline: {exc}", RuntimeWarning, stacklevel=2)

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        (batch_input,) = inputs
        batch_input = self._make_batch_first(batch_input).detach().to(torch.float32)
        out_device = self.dummy.device
        n = batch_input.shape[0]

        if not self._fitted:
            return torch.full((n,), float("inf"), dtype=torch.float32, device=out_device)

        device = self._device()
        try:
            with torch.no_grad():
                # One batched forward pass over all windows -> per-window log-likelihood.
                log_prob = self.model.to(device).log_probability(batch_input.to(device))
            return (-log_prob).to(dtype=torch.float32, device=out_device)
        except Exception:
            return torch.full((n,), float("inf"), dtype=torch.float32, device=out_device)

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        (target,) = targets
        return target[:, -1] if self.batch_first else target[-1]


class HMMAnomalyDetector(_BaseHMMAnomalyDetector):
    def __init__(
        self,
        n_components: int = 4,
        covariance_type: str = "diag",
        n_iter: int = 100,
        tol: float = 1e-2,
        min_covar: float = 1e-3,
        random_state: int = 42,
        batch_first: bool = True,
    ):
        ct = _pomegranate_covariance_type(covariance_type)
        distributions = [Normal(covariance_type=ct, min_cov=min_covar) for _ in range(n_components)]
        model = DenseHMM(distributions, max_iter=n_iter, tol=tol, random_state=random_state)
        super(HMMAnomalyDetector, self).__init__(model=model, batch_first=batch_first)


class GMMHMMAnomalyDetector(_BaseHMMAnomalyDetector):
    def __init__(
        self,
        n_components: int = 4,
        n_mix: int = 2,
        covariance_type: str = "diag",
        n_iter: int = 100,
        tol: float = 1e-2,
        min_covar: float = 1e-3,
        random_state: int = 42,
        batch_first: bool = True,
    ):
        ct = _pomegranate_covariance_type(covariance_type)
        distributions = [
            GeneralMixtureModel([Normal(covariance_type=ct, min_cov=min_covar) for _ in range(n_mix)])
            for _ in range(n_components)
        ]
        model = DenseHMM(distributions, max_iter=n_iter, tol=tol, random_state=random_state)
        super(GMMHMMAnomalyDetector, self).__init__(model=model, batch_first=batch_first)


HMMAD = HMMAnomalyDetector
GMMHMMAD = GMMHMMAnomalyDetector
