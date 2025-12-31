from typing import List, Union, Callable, Tuple, Sequence

import torch
from torch.nn import functional as F

from ..common import RNN, MLP, PredictionAnomalyDetector
from ...models import BaseModel
from ...data.transforms import PredictionTargetTransform, Transform
from ...utils import torch_utils
from ...utils.utils import halflife2alpha


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

        self.lstm = RNN('lstm', 's2fh', input_dim, lstm_hidden_dims)
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
        self._errors = []
        self._counter = 0

    def reset_state(self) -> None:
        self._errors = []
        self._counter = 0

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        errors = []
        device = self.dummy.device

        # Compute mean and covariance over the entire validation dataset
        counter = 0
        for b_inputs, b_targets in dataset:
            b_inputs = tuple(b_inp.to(device) for b_inp in b_inputs)
            b_targets = tuple(b_tar.to(device) for b_tar in b_targets)
            with torch.inference_mode():
                pred = self.model(b_inputs)

            target, = b_targets

            error = target - pred
            flipped = torch.flip(error, dims=[0]).cpu()
            for offset in range(error.shape[0] + error.shape[1] - 1):
                index = counter + offset
                if len(errors) <= index:
                    errors.extend([[] for _ in range(index + 1 - len(errors))])
                diag = torch.diagonal(flipped, offset=offset - (error.shape[0] - 1), dim1=0, dim2=1)
                errors[index].extend(diag)
            counter += error.shape[1]

        errors = errors[self.model.prediction_horizon - 1:-self.model.prediction_horizon + 1]

        errors = torch_utils.nested_list2tensor(errors)
        errors = errors.view(errors.shape[0], -1)
        mean = torch.mean(errors, dim=0)
        errors -= mean
        cov = torch.matmul(errors.T, errors)
        cov /= errors.shape[0] - 1

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
            raise RuntimeError('Could not compute a valid covariance matrix!')

        precision = cov
        torch.cholesky_inverse(cholesky, out=precision)

        self.register_buffer('mean', mean)
        self.register_buffer('precision', precision)

    def _accumulate_errors(self, x: torch.Tensor, target: torch.Tensor):
        with torch.inference_mode():
            pred = self.model((x,))

        error = target - pred
        flipped = torch.flip(error, dims=[0]).cpu()
        for offset in range(error.shape[0] + error.shape[1] - 1):
            index = self._counter + offset
            if len(self._errors) <= index:
                self._errors.extend([[] for _ in range(index + 1 - len(self._errors))])
            diag = torch.diagonal(flipped, offset=offset - (error.shape[0] - 1), dim1=0, dim2=1)
            self._errors[index].extend(diag)
        self._counter += error.shape[1]

    def forward(self, inputs: Tuple[torch.Tensor, ...]):
        x, target = inputs
        self._accumulate_errors(x, target)

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def _score_from_errors(self, errors: torch.Tensor) -> torch.Tensor:
        errors = errors - self.mean
        return F.bilinear(errors, errors, self.precision.unsqueeze(0)).squeeze(-1)

    def compute(self) -> torch.Tensor:
        errors = self._errors[self.model.prediction_horizon - 1:-self.model.prediction_horizon + 1]
        errors = torch_utils.nested_list2tensor(errors)
        errors = errors.view(errors.shape[0], -1)
        scores = self._score_from_errors(errors)
        self.reset_state()
        return scores

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def get_labels_and_scores(self, dataset: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        self.reset_state()
        labels = []
        for b_inputs, b_targets in dataset:
            b_inputs = tuple(b_inp.to(self.dummy.device) for b_inp in b_inputs)
            b_targets = tuple(b_tar.to(self.dummy.device) for b_tar in b_targets)

            label, target = b_targets

            self.forward((b_inputs[0], target))

            labels.append(label[-1].cpu())

        labels = torch.cat(labels, dim=0)
        labels = labels[:-self.model.prediction_horizon + 1]

        scores = self.compute()

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
        self.moving_avg_num = 0
        self.moving_avg_denom = 0
        self._errors = []


    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        pass

    def reset_state(self) -> None:
        self.moving_avg_num = 0
        self.moving_avg_denom = 0
        self._errors = []

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # x: (T, B, D), target: (T, B, D)
        x, target = inputs
        with torch.inference_mode():
            x_pred = self.model((x,))

        sq_error = target - x_pred
        torch.square(sq_error, out=sq_error)
        sq_error = torch.sum(sq_error, dim=-1)

        T, B = sq_error.shape
        sq_error = sq_error.T.flatten()
        self.moving_avg_num, self.moving_avg_denom = torch_utils.exponential_moving_avg_(
            sq_error,
            self.alpha,
            avg_num=self.moving_avg_num,
            avg_denom=self.moving_avg_denom,
        )

        sq_error = sq_error.view(B, T).T
        return sq_error

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, torch.Tensor, float, float]) \
            -> Tuple[torch.Tensor, float, float]:
        x, target, moving_avg_num, moving_avg_denom = inputs
        with torch.inference_mode():
            x_pred = self.model((x,))

        sq_error = target - x_pred
        torch.square(sq_error, out=sq_error)
        sq_error = torch.sum(sq_error, dim=-1)

        T, B = sq_error.shape
        sq_error = sq_error.T.flatten()
        moving_avg_num, moving_avg_denom = torch_utils.exponential_moving_avg_(
            sq_error,
            self.alpha,
            avg_num=moving_avg_num,
            avg_denom=moving_avg_denom,
        )

        return sq_error.view(B, T).T, moving_avg_num, moving_avg_denom

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def get_labels_and_scores(self, dataset: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = []
        self.reset_state()

        # Compute exp moving average of error score
        for b_inputs, b_targets in dataset:
            b_inputs = tuple(b_inp.to(self.dummy.device) for b_inp in b_inputs)
            b_targets = tuple(b_tar.to(self.dummy.device) for b_tar in b_targets)

            x, = b_inputs
            label, target = b_targets

            self.forward((x, target))
            labels.append(label.cpu())

        labels = torch.cat(labels, dim=1).transpose(0, 1).flatten()

        scores = self.compute()

        assert labels.shape == scores.shape

        return labels, scores.cpu()


class LSTMS2STargetTransform(PredictionTargetTransform):
    def __init__(self, parent: Transform, window_size: int, replace_labels: bool = False,
                 reverse: bool = False):
        super(LSTMS2STargetTransform, self).__init__(parent, window_size, window_size, replace_labels=replace_labels,
                                                     step_size=window_size, reverse=reverse)
