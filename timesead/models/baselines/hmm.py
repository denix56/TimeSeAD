from typing import Any, List, Tuple
import warnings

import numpy as np
import torch
from hmmlearn.hmm import GaussianHMM, GMMHMM

from ..common import AnomalyDetector


class _BaseHMMAnomalyDetector(AnomalyDetector):
    def __init__(self, model: Any, batch_first: bool = True):
        super(_BaseHMMAnomalyDetector, self).__init__()

        self.model = model
        self.batch_first = batch_first

    def _make_batch_first(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            return inputs

        return inputs.transpose(0, 1)

    def fit(self, dataset: torch.utils.data.DataLoader, **kwargs) -> None:
        all_sequences: List[np.ndarray] = []
        lengths: List[int] = []

        for batch_inputs, _ in dataset:
            (batch_input,) = batch_inputs
            batch_input = self._make_batch_first(batch_input)

            x_np = batch_input.detach().cpu().numpy().astype(np.float64, copy=False)
            for sequence in x_np:
                all_sequences.append(sequence)
                lengths.append(sequence.shape[0])

        if not all_sequences:
            return

        concatenated = np.concatenate(all_sequences, axis=0)
        try:
            self.model.fit(concatenated, lengths=lengths)
        except Exception as exc:
            warnings.warn(f"Failed to fit HMM baseline: {exc}", RuntimeWarning, stacklevel=2)

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        (batch_input,) = inputs
        batch_input = self._make_batch_first(batch_input)
        x_np = batch_input.detach().cpu().numpy().astype(np.float64, copy=False)

        scores = []
        for sequence in x_np:
            try:
                scores.append(-float(self.model.score(sequence)))
            except Exception:
                scores.append(float("inf"))

        return torch.tensor(scores, dtype=torch.float32, device=self.dummy.device)

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
        model = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            min_covar=min_covar,
            random_state=random_state,
        )
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
        model = GMMHMM(
            n_components=n_components,
            n_mix=n_mix,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            min_covar=min_covar,
            random_state=random_state,
        )
        super(GMMHMMAnomalyDetector, self).__init__(model=model, batch_first=batch_first)


HMMAD = HMMAnomalyDetector
GMMHMMAD = GMMHMMAnomalyDetector
