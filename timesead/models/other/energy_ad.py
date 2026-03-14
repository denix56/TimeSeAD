from typing import Tuple

from sympy import ShapeError
import torch

from ..common import AnomalyDetector
from ...models import BaseModel


class EnergyAD_AnomalyDetector(AnomalyDetector):
    """
    Anomaly detector using the line integral for deviance from a learned vector field of normal dynamics.
    """
    def __init__(self, model: BaseModel, batch_first: bool = True):
        super(EnergyAD_AnomalyDetector, self).__init__()

        self.model = model
        self.batch_first = batch_first

    def fit(self, dataset: torch.utils.data.DataLoader, **kwargs) -> None:
        pass

    def _trim_inputs_for_model(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        model_seq_len = getattr(self.model, 'seq_len', None)
        if model_seq_len is None:
            return inputs

        time_dim = 1 if self.batch_first else 0
        input_seq_len = inputs[0].size(time_dim)
        if input_seq_len != model_seq_len + 1:
            return inputs

        if self.batch_first:
            return tuple(inp[:, :-1, ...] for inp in inputs)

        return tuple(inp[:-1, ...] for inp in inputs)

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        :param inputs: Tensors of shape ``(B, T, D)`` or ``(T, B, D)``.
        :return: Tensor of shape ``(B,)`` with higher scores indicating more anomalous windows.
        """
        model_inputs = self._trim_inputs_for_model(inputs)
        with torch.no_grad():
            model_output = self.model(model_inputs)

        input_batch, = inputs
        if not self.batch_first:
            input_batch = input_batch.permute(1, 0, 2)
            model_output = model_output.permute(1, 0, 2)

        anomaly_scores = line_integral_score(model_output, input_batch)
        # Normal samples follow the vector field more closely, so we flip the sign.
        anomaly_scores.neg_()

        return anomaly_scores

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        :param targets: Tensors of shape ``(B, T)``, ``(B,)`` or ``(T, B)``.
        :return: Tensor of shape ``(B,)`` containing the label for the last point in each window.
        """
        labels = targets[0]
        if labels.ndim == 1:
            return labels

        return labels[:, -1] if self.batch_first else labels[-1]


def line_integral_score(model_output_batch: torch.Tensor, input_batch: torch.Tensor) -> torch.Tensor:
    """
    Calculate the path energy with respect to the vector field predicted by the model.

    The line integral is approximated as

    ``f(x) * (dg / dt)(x) * ||(dg / dt)(x)||_2``

    with finite differences ``dg / dt = x_{i+1} - x_i``.
    """
    _, output_window_size, _ = model_output_batch.shape
    _, input_window_size, _ = input_batch.shape

    if input_window_size < 2:
        raise ShapeError(
            f'Input window length ({input_window_size}) is less than two. '
            'Try removing incomplete windows from the data.'
        )

    if input_window_size != output_window_size:
        if input_window_size == output_window_size + 1:
            input_batch = input_batch[:, :-1, :]
        else:
            raise ShapeError(
                f'Model output length ({output_window_size}) does not match input length '
                f'({input_window_size}).'
            )

    path_derivatives = input_batch[:, 1:, :] - input_batch[:, :-1, :]
    norm_derivative = torch.norm(path_derivatives, p=2, dim=2)
    dot_products = (model_output_batch[:, :-1, :] * path_derivatives).sum(dim=2)

    return (dot_products * norm_derivative).sum(dim=1)


lineIntegral_score = line_integral_score
