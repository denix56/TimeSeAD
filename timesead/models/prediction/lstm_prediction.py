from typing import List, Union, Callable, Tuple, Sequence
import logging

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import Tensor

from ..common import RNN, MLP, PredictionAnomalyDetector
from ...models import BaseModel
from ...data.transforms import PredictionTargetTransform, Transform
from ...utils import torch_utils
from ...utils.utils import halflife2alpha

logger = logging.getLogger(__name__)


class LSTMPrediction(BaseModel):
    def __init__(self, input_dim: int, lstm_hidden_dims: Sequence[int] = (30, 20), linear_hidden_layers: Sequence[int] = None,
                 linear_activation: Union[Callable, str] = torch.nn.ELU, prediction_horizon: int = 3):
        """
        LSTM prediction (Malhotra2015)
        :param input_dim:
        :param lstm_hidden_dims:
        :param linear_hidden_layers:
        :param linear_activation:
        :param prediction_horizon:
        """
        super(LSTMPrediction, self).__init__()

        self.prediction_horizon = prediction_horizon
        if linear_hidden_layers is None:
            linear_hidden_layers = []

        self.lstm = RNN('lstm', 's2fh', input_dimension=input_dim, hidden_dimensions=lstm_hidden_dims)
        self.mlp = MLP(lstm_hidden_dims[-1], linear_hidden_layers, prediction_horizon * input_dim, linear_activation())

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # x: (T, B, D)
        x, = inputs

        # hidden: (B, hidden_dims)
        hidden = self.lstm(x)
        # x_pred: (B, horizon * D)
        x_pred = self.mlp(hidden)
        # x_pred: (B, horizon, D)
        x_pred = x_pred.view(x_pred.shape[0], self.prediction_horizon, -1)
        # output: (horizon, B, D)
        return x_pred.transpose(0, 1)


class LSTMS2SPrediction(BaseModel):
    def __init__(self, input_dim: int, lstm_hidden_dims: Sequence[int] = (30, 20), linear_hidden_layers: Sequence[int] = None,
                 linear_activation: Union[Callable, str] = torch.nn.ELU, dropout: float = 0.0):
        """
        LSTM prediction (Filonov2016)
        :param input_dim:
        :param lstm_hidden_dims:
        :param linear_hidden_layers:
        :param linear_activation:
        """
        super(LSTMS2SPrediction, self).__init__()
        if linear_hidden_layers is None:
            linear_hidden_layers = []

        self.lstm = RNN('lstm', 's2s', input_dim, lstm_hidden_dims, dropout=dropout)
        self.mlp = MLP(lstm_hidden_dims[-1], linear_hidden_layers, input_dim, linear_activation())

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # x: (T, B, D)
        x, = inputs

        # hidden: (T, B, hidden_dims)
        hidden = self.lstm(x)
        # x_pred: (T, B, D)
        x_pred = self.mlp(hidden)

        return x_pred


class LSTMPredictionAnomalyDetector(PredictionAnomalyDetector):
    def __init__(self, model: LSTMPrediction):
        """
        Malhotra2016

        :param model:
        """
        super(LSTMPredictionAnomalyDetector, self).__init__()
        self.model = model
        self.register_buffer('mean', torch.zeros(1))
        self.register_buffer('precision', torch.zeros(1))

    @staticmethod
    def _append_window_errors(error_buckets: List[List[torch.Tensor]], error: torch.Tensor, window_start: int) -> None:
        horizon, window_count = error.shape[:2]
        for offset in range(horizon):
            start = window_start + offset
            end = start + window_count
            if len(error_buckets) < end:
                error_buckets.extend([[] for _ in range(end - len(error_buckets))])
            error_step = error[offset]
            bucket_slice = error_buckets[start:end]
            error_buckets[start:end] = [bucket + [value] for bucket, value in zip(bucket_slice, error_step)]

    @staticmethod
    def _consume_subsequence_batches(batch_size: int, windows_per_seq: List[int], subseq_idx: int, window_idx: int,
                                     on_empty: Callable[[], None],
                                     on_slice: Callable[[int, int, int], None],
                                     on_end: Callable[[], None]) -> Tuple[int, int]:
        batch_offset = 0
        while batch_offset < batch_size:
            while subseq_idx < len(windows_per_seq) and windows_per_seq[subseq_idx] == 0:
                on_empty()
                subseq_idx += 1
                window_idx = 0

            if subseq_idx >= len(windows_per_seq):
                return subseq_idx, window_idx

            remaining = windows_per_seq[subseq_idx] - window_idx
            take = min(batch_size - batch_offset, remaining)
            on_slice(window_idx, batch_offset, take)

            window_idx += take
            batch_offset += take

            if window_idx >= windows_per_seq[subseq_idx]:
                on_end()
                subseq_idx += 1
                window_idx = 0

        return subseq_idx, window_idx

    def fit(self, dataset: DataLoader, *, subseq_lengths: List[int], window_size: int, **kwargs) -> None:
        errors = []
        device = self.dummy.device
        windows_per_seq = [max(subseq_len - window_size + 1, 0) for subseq_len in subseq_lengths]
        current_errors = []
        subseq_idx = 0
        window_idx = 0

        # Compute mean and covariance over the entire validation dataset
        for idx, (b_inputs, b_targets) in enumerate(dataset):
            b_inputs = tuple(b_inp.to(device) for b_inp in b_inputs)
            b_targets = tuple(b_tar.to(device) for b_tar in b_targets)
            with torch.no_grad():
                pred = self.model(b_inputs)

            target, = b_targets

            error = target - pred
            batch_size = error.shape[1]
            def handle_empty() -> None:
                nonlocal current_errors
                current_errors = []

            def handle_slice(window_start: int, offset: int, take: int) -> None:
                error_slice = error[:, offset:offset + take]
                self._append_window_errors(current_errors, error_slice, window_start)

            def handle_end() -> None:
                nonlocal current_errors
                trim = self.model.prediction_horizon - 1
                end = -trim if trim > 0 else None
                errors.extend(current_errors[trim:end])
                current_errors = []

            subseq_idx, window_idx = self._consume_subsequence_batches(
                batch_size, windows_per_seq, subseq_idx, window_idx, handle_empty, handle_slice, handle_end
            )

        if window_idx > 0 and current_errors:
            trim = self.model.prediction_horizon - 1
            end = -trim if trim > 0 else None
            errors.extend(current_errors[trim:end])

        errors = torch_utils.nested_list2tensor(errors)
        errors = errors.view(errors.shape[0], -1)
        mean = torch.mean(errors, dim=0)
        errors -= mean
        cov = torch.matmul(errors.T, errors)
        cov /= errors.shape[0] - 1
        cov = 0.5 * (cov + cov.T)

        for i in range(5, -3, -1):
            try:
                cov.diagonal().add_(10**-i)
                cholesky = torch.linalg.cholesky(cov)
                if not torch.isnan(cholesky).any():
                    break
            except Exception as e:
                print(f"Cholesky decomposition failed: {e}, trying to fix by adding small value to diagonal.")
                # If the covariance matrix is not positive definite, we can try to add a small value to the diagonal until it becomes positive definite
                continue
        else:
            raise RuntimeError(f'Could not compute a valid covariance matrix! {cov}')

        precision = cov
        torch.cholesky_inverse(cholesky, out=precision)

        self.mean = mean
        self.precision = precision

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pass

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pass

    def get_labels_and_scores(self, dataset: DataLoader, *, subseq_lengths: List[int], window_size: int, **kwargs) -> Tuple[Tensor, Tensor]:
        errors = []
        labels = []
        windows_per_seq = [max(subseq_len - window_size + 1, 0) for subseq_len in subseq_lengths]
        current_errors = []
        current_labels = []
        subseq_idx = 0
        window_idx = 0
        # Compute mean and covariance over the entire validation dataset
        for b_inputs, b_targets in dataset:
            b_inputs = tuple(b_inp.to(self.dummy.device) for b_inp in b_inputs)
            b_targets = tuple(b_tar.to(self.dummy.device) for b_tar in b_targets)
            with torch.no_grad():
                pred = self.model(b_inputs)

            label, target = b_targets

            error = target - pred
            batch_size = error.shape[1]
            label_at_end = label[-1].detach().cpu()

            def handle_empty() -> None:
                nonlocal current_errors, current_labels
                current_errors = []
                current_labels = []

            def handle_slice(window_start: int, offset: int, take: int) -> None:
                error_slice = error[:, offset:offset + take]
                self._append_window_errors(current_errors, error_slice, window_start)
                current_labels.append(label_at_end[offset:offset + take])

            def handle_end() -> None:
                nonlocal current_errors, current_labels
                trim = self.model.prediction_horizon - 1
                end = -trim if trim > 0 else None
                errors.extend(current_errors[trim:end])
                if current_labels:
                    subseq_labels = torch.cat(current_labels, dim=0)
                    if trim > 0:
                        subseq_labels = subseq_labels[:-trim]
                    labels.append(subseq_labels)
                current_errors = []
                current_labels = []

            subseq_idx, window_idx = self._consume_subsequence_batches(
                batch_size, windows_per_seq, subseq_idx, window_idx, handle_empty, handle_slice, handle_end
            )
        if window_idx > 0 and (current_errors or current_labels):
            trim = self.model.prediction_horizon - 1
            end = -trim if trim > 0 else None
            errors.extend(current_errors[trim:end])
            if current_labels:
                subseq_labels = torch.cat(current_labels, dim=0)
                if trim > 0:
                    subseq_labels = subseq_labels[:-trim]
                labels.append(subseq_labels)
        errors = torch_utils.nested_list2tensor(errors)
        errors = errors.view(errors.shape[0], -1)
        labels = torch.cat(labels, dim=0)

        errors -= self.mean
        scores = F.bilinear(errors, errors, self.precision.unsqueeze(0)).squeeze(-1)

        assert labels.shape == scores.shape

        return labels, scores.cpu()


class LSTMS2SPredictionAnomalyDetector(PredictionAnomalyDetector):
    def __init__(self, model: LSTMS2SPrediction, half_life: int):
        """
        Filonov2016

        :param model:
        :param half_life:
        """
        super(LSTMS2SPredictionAnomalyDetector, self).__init__()

        self.model = model
        self.alpha = halflife2alpha(half_life)


    def fit(self, dataset: torch.utils.data.DataLoader, **kwargs) -> None:
        pass

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, torch.Tensor, float, float]) \
            -> Tuple[torch.Tensor, float, float]:
        # x: (T, B, D), target: (T, B, D), moving_avg: ()
        x, target, moving_avg_num, moving_avg_denom = inputs

        with torch.no_grad():
            x_pred = self.model((x,))

        sq_error = target - x_pred
        torch.square(sq_error, out=sq_error)
        sq_error = torch.sum(sq_error, dim=-1)

        T, B = sq_error.shape
        sq_error = sq_error.T.flatten()
        moving_avg_num, moving_avg_denom = torch_utils.exponential_moving_avg_(sq_error, self.alpha,
                                                                               avg_num=moving_avg_num,
                                                                               avg_denom=moving_avg_denom)

        return sq_error.view(B, T).T, moving_avg_num, moving_avg_denom

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pass

    def get_labels_and_scores(self, dataset: torch.utils.data.DataLoader, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        errors = []
        labels = []
        moving_avg_num = 0
        moving_avg_denom = 0

        # Compute exp moving average of error score
        for b_inputs, b_targets in dataset:
            b_inputs = tuple(b_inp.to(self.dummy.device) for b_inp in b_inputs)
            b_targets = tuple(b_tar.to(self.dummy.device) for b_tar in b_targets)

            x, = b_inputs
            label, target = b_targets

            sq_error, moving_avg_num, moving_avg_denom = self.compute_online_anomaly_score((x, target, moving_avg_num,
                                                                                            moving_avg_denom))
            errors.append(sq_error)
            labels.append(label.cpu())

        scores = torch.cat(errors, dim=1).transpose(0, 1).flatten()
        labels = torch.cat(labels, dim=1).transpose(0, 1).flatten()

        assert labels.shape == scores.shape

        return labels, scores.cpu()


class LSTMS2STargetTransform(PredictionTargetTransform):
    def __init__(self, parent: Transform, window_size: int, replace_labels: bool = False,
                 reverse: bool = False):
        super(LSTMS2STargetTransform, self).__init__(parent, window_size, window_size, replace_labels=replace_labels,
                                                     step_size=window_size, reverse=reverse)
