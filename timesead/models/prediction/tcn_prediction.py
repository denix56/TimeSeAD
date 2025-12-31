from typing import Sequence, Tuple, Union, Callable

import torch
from torch.nn import functional as F

from ..common import MLP, PredictionAnomalyDetector
from ..layers import SameCausalZeroPad1d
from ...models import BaseModel
from ...utils import torch_utils


class TCNS2SPrediction(BaseModel):
    def __init__(self, input_dim: int, filters: Sequence[int] = (64, 64, 64, 64, 64),
                 kernel_sizes: Sequence[int] = (3, 3, 3, 3, 3), dilations: Sequence[int] = (1, 2, 4, 8, 16),
                 last_n_layers_to_cat: int = 3, activation=torch.nn.ReLU()):
        """
        He2019

        :param input_dim:
        :param filters:
        :param kernel_sizes:
        :param dilations:
        :param last_n_layers_to_cat:
        :param activation:
        """
        super(TCNS2SPrediction, self).__init__()

        assert len(filters) == len(kernel_sizes) == len(dilations)
        assert 0 < last_n_layers_to_cat < len(filters)

        self.last_n_layers_to_cat = last_n_layers_to_cat
        self.activation = activation

        filters = [input_dim] + list(filters)

        modules = []
        for in_channels, out_channels, kernel_size, dilation in zip(filters[:-1], filters[1:], kernel_sizes, dilations):
            padding = SameCausalZeroPad1d(kernel_size, dilation=dilation)
            conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
            modules.append(torch.nn.Sequential(padding, conv))
        self.conv_layers = torch.nn.ModuleList(modules)

        # Final 1x1 conv to retrieve the output
        self.final_conv = torch.nn.Conv1d(sum(filters[-last_n_layers_to_cat:]), input_dim, 1)

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # x: (B, T, D)
        x, = inputs
        # x: (B, D, T)
        x = x.transpose(1, 2)

        outputs = []
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            x = self.activation(x)

            if i >= len(self.conv_layers) - self.last_n_layers_to_cat:
                outputs.append(x)

        x_cat = torch.cat(outputs, dim=1)

        x_pred = self.final_conv(x_cat)

        return x_pred.transpose(1, 2)


class TCNPrediction(BaseModel):
    def __init__(self, input_dim: int, window_size: int, filters: Sequence[int] = (32, 32),
                 kernel_sizes: Sequence[int] = (3, 3), linear_hidden_layers: Sequence[int] = (50,),
                 activation: Union[Callable, str] = torch.nn.ReLU(), prediction_horizon: int = 1):
        """
        DeepAnT aka TCN prediction (Munir2018)
        :param input_dim:
        :param filters:
        :param kernel_sizes:
        :param linear_hidden_layers:
        :param activation:
        :param prediction_horizon:
        """
        super(TCNPrediction, self).__init__()

        assert len(filters) == len(kernel_sizes)

        self.activation = activation
        self.prediction_horizon = prediction_horizon
        self.pooler = torch.nn.MaxPool1d(2)

        filters = [input_dim] + list(filters)

        modules = []
        for in_channels, out_channels, kernel_size in zip(filters[:-1], filters[1:], kernel_sizes):
            conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
            modules.append(conv)
        self.conv_layers = torch.nn.ModuleList(modules)

        final_output_size = filters[-1] * int(window_size * 0.5**(len(filters) - 1))
        self.mlp = MLP(final_output_size, list(linear_hidden_layers), prediction_horizon * input_dim, activation)

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # x: (B, T, D)
        x, = inputs
        x = x.transpose(1, 2)

        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = self.activation(x)
            x = self.pooler(x)

        # Flatten x
        x = x.view(x.shape[0], -1)

        # x_pred: (B, horizon * D)
        x_pred = self.mlp(x)
        # x_pred: (B, horizon, D)
        x_pred = x_pred.view(x_pred.shape[0], self.prediction_horizon, -1)
        return x_pred


class TCNS2SPredictionAnomalyDetector(PredictionAnomalyDetector):
    def __init__(self, model: TCNS2SPrediction, offset: int):
        """
        He2019

        :param model:
        """
        super(TCNS2SPredictionAnomalyDetector, self).__init__()

        self.model = model
        self.offset = offset
        self._errors = []
        self._counter = 0

    def reset_state(self) -> None:
        self._errors = []
        self._counter = 0

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        errors = []

        # Compute mean and covariance over the entire validation dataset
        counter = 0
        for b_inputs, b_targets in dataset:
            b_inputs = tuple(b_inp.to(self.dummy.device) for b_inp in b_inputs)
            b_targets = tuple(b_tar.to(self.dummy.device) for b_tar in b_targets)
            with torch.inference_mode():
                pred = self.model(b_inputs)

            target, = b_targets

            error = target[:, -self.offset:] - pred[:, -self.offset:]
            for offset in range(error.shape[0] + error.shape[1] - 1):
                index = counter + offset
                if len(errors) <= index:
                    errors.extend([[] for _ in range(index + 1 - len(errors))])
                diag = torch.diagonal(error, offset=offset - (error.shape[0] - 1), dim1=0, dim2=1)
                errors[index].extend(diag)
            counter += error.shape[0]

        # We will have up to offset predictions for each point. We use the mean as the final prediction
        errors = [sum(error) / len(error) for error in errors]
        errors = torch.stack(errors, dim=0)

        mean = torch.mean(errors, dim=0)
        errors -= mean

        cov = torch.matmul(errors.T, errors)
        cov /= errors.shape[0] - 1

        # Add a small epsilon to the diagonal of the matrix to make it non-singular
        cov.diagonal().add_(1e-5)

        # This construction ensures that the resulting precision matrix is pos. semi-definite, even if the condition
        # number of the cov matrix is large
        cholesky = torch.linalg.cholesky(cov)
        precision = cov
        torch.cholesky_inverse(cholesky, out=precision)

        self.register_buffer('mean', mean)
        self.register_buffer('precision', precision)

    def _accumulate_errors(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            pred = self.model((x,))

        error = target[:, -self.offset:] - pred[:, -self.offset:]
        for offset in range(error.shape[0] + error.shape[1] - 1):
            index = self._counter + offset
            if len(self._errors) <= index:
                self._errors.extend([[] for _ in range(index + 1 - len(self._errors))])
            diag = torch.diagonal(error, offset=offset - (error.shape[0] - 1), dim1=0, dim2=1)
            self._errors[index].extend(diag)
        self._counter += error.shape[0]
        return error

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        x, target = inputs
        return self._accumulate_errors(x, target)

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def _score_from_errors(self, errors: torch.Tensor) -> torch.Tensor:
        errors = errors - self.mean
        return F.bilinear(errors, errors, self.precision.unsqueeze(0)).squeeze(-1)

    def compute(self) -> torch.Tensor:
        errors = [sum(error) / len(error) for error in self._errors]
        errors = torch.stack(errors, dim=0)
        scores = self._score_from_errors(errors)
        self.reset_state()
        return scores

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        label, _ = targets
        return label[:, -self.offset]

    def get_labels_and_scores(self, dataset: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        self.reset_state()
        labels = []
        for b_inputs, b_targets in dataset:
            b_inputs = tuple(b_inp.to(self.dummy.device) for b_inp in b_inputs)
            b_targets = tuple(b_tar.to(self.dummy.device) for b_tar in b_targets)
            label, target = b_targets

            self.forward((b_inputs[0], target))

            labels.append(label[:, -self.offset].cpu())

        labels = torch.cat(labels, dim=0)

        scores = self.compute()

        assert labels.shape == scores.shape

        return labels, scores.cpu()


class TCNPredictionAnomalyDetector(PredictionAnomalyDetector):
    def __init__(self, model: TCNPrediction):
        """
        Munir2018

        :param model:
        """
        super(TCNPredictionAnomalyDetector, self).__init__()

        self.model = model
        self._errors = []
        self._counter = 0

    def reset_state(self) -> None:
        self._errors = []
        self._counter = 0

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        pass

    def _accumulate_errors(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            pred = self.model((x,))

        error = target - pred
        for offset in range(error.shape[0] + error.shape[1] - 1):
            index = self._counter + offset
            if len(self._errors) <= index:
                self._errors.extend([[] for _ in range(index + 1 - len(self._errors))])
            diag = torch.diagonal(error, offset=offset - (error.shape[0] - 1), dim1=0, dim2=1)
            self._errors[index].extend(diag)
        self._counter += error.shape[0]
        return error

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        x, target = inputs
        return self._accumulate_errors(x, target)

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def _score_from_errors(self, errors: torch.Tensor) -> torch.Tensor:
        return torch_utils.batched_dot(errors, errors)

    def compute(self) -> torch.Tensor:
        errors = [sum(error) / len(error) for error in self._errors]
        errors = torch.stack(errors, dim=0)
        scores = self._score_from_errors(errors)
        self.reset_state()
        return scores

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        label, _ = targets
        return label[:, -1]

    def get_labels_and_scores(self, dataset: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        self.reset_state()
        labels = []
        for i, (b_inputs, b_targets) in enumerate(dataset):
            b_inputs = tuple(b_inp.to(self.dummy.device) for b_inp in b_inputs)
            b_targets = tuple(b_tar.to(self.dummy.device) for b_tar in b_targets)

            label, target = b_targets

            self.forward((b_inputs[0], target))

            if i == 0:
                # Append the first few labels as well
                labels.append(label[0, :-1].cpu())
            labels.append(label[:, -1].cpu())

        labels = torch.cat(labels, dim=0)

        scores = self.compute()

        assert labels.shape == scores.shape

        return labels, scores.cpu()
