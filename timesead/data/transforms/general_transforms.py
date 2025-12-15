import functools
import random
from typing import Tuple

import numpy as np
import torch

from .transform_base import Transform
from ...utils.utils import ceil_div, getitem


def _linear_interpolate_sequence(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Linearly interpolate `x` at the given (monotonic) `positions`.

    :param x: Tensor of shape `(T, ...)`.
    :param positions: Tensor of shape `(T,)` that specifies the sampling locations along the time dimension.
    :return: Interpolated tensor of the same shape as ``x``.
    """
    seq_len = x.shape[0]

    lower_idx = torch.clamp(torch.floor(positions), 0, seq_len - 1).long()
    upper_idx = torch.clamp(lower_idx + 1, max=seq_len - 1)
    upper_weight = positions - lower_idx
    lower_weight = 1 - upper_weight

    lower_vals = x[lower_idx]
    upper_vals = x[upper_idx]

    weight_shape = (positions.shape[0],) + (1,) * (x.dim() - 1)
    lower_weight = lower_weight.view(weight_shape)
    upper_weight = upper_weight.view(weight_shape)

    return lower_weight * lower_vals + upper_weight * upper_vals


class SubsampleTransform(Transform):
    """
    Subsample sequences by a specified factor. `subsampling_factor` consecutive datapoints in a sequence will be
    aggregated into one point using the `aggregation` function.
    """
    def __init__(self, parent: Transform, subsampling_factor: int, aggregation: str = 'first'):
        """

        :param parent: Another :class:`~timesead.data.transforms.Transform` which is used as the data source for this
            :class:`~timesead.data.transforms.Transform`.
        :param subsampling_factor: This specifies the number of consecutive data points that will be aggregated.
        :param aggregation: The function that should be applied to aggregate a window of data points.
           Can be either 'mean', 'last' or 'first'.
        """
        super(SubsampleTransform, self).__init__(parent)

        self.subsampling_factor = subsampling_factor
        if aggregation == 'mean':
            self.aggregate_fn = functools.partial(torch.mean, dim=1)
        if aggregation == 'last':
            self.aggregate_fn = functools.partial(getitem, item=(slice(None), -1))
        else:  # 'first'
            self.aggregate_fn = functools.partial(getitem, item=(slice(None), 0))

    def _process_tensor(self, inp: torch.Tensor) -> torch.Tensor:
        # Input has shape (T, ...). Reshape it to (T//subsampling_factor, subsampling_factor, ...) and apply
        # aggregate on the 2nd axis. We might need to add padding at the end for this to work
        inp_shape = inp.shape
        new_t, rest = divmod(inp_shape[0], self.subsampling_factor)
        if rest > 0:
            # Add padding. We pad with the result of aggregating the last (incomplete) window
            pad_value = self.aggregate_fn(inp[new_t * self.subsampling_factor:inp_shape[0]].unsqueeze(0))
            inp = torch.cat([inp] + (self.subsampling_factor - rest) * [pad_value])
            new_t += 1

        inp = inp.view(new_t, self.subsampling_factor, *inp_shape[1:])
        return self.aggregate_fn(inp)

    def _get_datapoint_impl(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        inputs, targets = self.parent.get_datapoint(item)

        inputs = tuple(self._process_tensor(inp) for inp in inputs)
        targets = tuple(self._process_tensor(tar) for tar in targets)

        return inputs, targets

    @property
    def seq_len(self):
        old_len = self.parent.seq_len
        if old_len is None:
            return None

        if isinstance(old_len, int):
            return ceil_div(old_len, self.subsampling_factor)

        return [ceil_div(old_l, self.subsampling_factor) for old_l in old_len]


class CacheTransform(Transform):
    """
    Caches the results from a previous :class:`~timesead.data.transforms.Transform` in memory so that expensive
    calculations do not have to be recomputed.
    """
    def __init__(self, parent: Transform):
        """

        :param parent: Another :class:`~timesead.data.transforms.Transform` which is used as the data source for this
            :class:`~timesead.data.transforms.Transform`.
        """
        super(CacheTransform, self).__init__(parent)

        self.cache = {}

    def _get_datapoint_impl(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        if item in self.cache:
            return self.cache[item]

        inputs, targets = self.parent.get_datapoint(item)
        self.cache[item] = (inputs, targets)

        return inputs, targets


class LimitTransform(Transform):
    """
    Limits the amount of data points returned.
    """
    def __init__(self, parent: Transform, count: int):
        """

        :param parent: Another :class:`~timesead.data.transforms.Transform` which is used as the data source for this
            :class:`~timesead.data.transforms.Transform`.
        :param count: The max number of sequences that should be returned by this
            :class:`~timesead.data.transforms.Transform`.
        """
        super(LimitTransform, self).__init__(parent)
        self.max_count = count

    def _get_datapoint_impl(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        if item >= self.max_count:
            raise IndexError

        return self.parent.get_datapoint(item)

    def __len__(self):
        if len(self.parent) is not None:
            return min(self.max_count, len(self.parent))

        return None


class _BaseInputTransform(Transform):
    """Utility base class for transforms that only modify the inputs."""

    def _transform_input(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _get_datapoint_impl(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        inputs, targets = self.parent.get_datapoint(item)
        inputs = tuple(self._transform_input(inp) for inp in inputs)
        return inputs, targets


class MagAddNoiseTransform(_BaseInputTransform):
    """Apply additive noise scaled by the local magnitude of the signal."""

    def __init__(self, parent: Transform, magnitude: float = 1.0):
        super().__init__(parent)
        self.magnitude = magnitude

    def _transform_input(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.magnitude <= 0 or tensor.shape[0] < 2:
            return tensor

        diffs = tensor[1:] - tensor[:-1]
        std = diffs.std(dim=0, keepdim=True).clamp_min(torch.finfo(tensor.dtype).eps)
        noise = torch.normal(0, 1 / 3, size=tensor.shape, device=tensor.device, dtype=tensor.dtype)
        return tensor + noise * std * self.magnitude


class MagScaleTransform(_BaseInputTransform):
    """Randomly scale the input magnitude."""

    def __init__(self, parent: Transform, magnitude: float = 0.5):
        super().__init__(parent)
        self.magnitude = magnitude

    def _transform_input(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.magnitude <= 0:
            return tensor

        rand = torch.abs(torch.randn((), device=tensor.device, dtype=tensor.dtype)).item()
        scale = (1 - (rand * self.magnitude) / 2) if random.random() > 1 / 3 else (1 + rand * self.magnitude)
        return tensor * scale


class TimeWarpTransform(_BaseInputTransform):
    """Apply a smooth random time warping to the sequence."""

    def __init__(self, parent: Transform, magnitude: float = 0.1, order: int = 6):
        super().__init__(parent)
        self.magnitude = magnitude
        self.order = order

    def _transform_input(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.magnitude <= 0 or tensor.shape[0] < 2:
            return tensor

        seq_len = tensor.shape[0]
        base_idx = torch.arange(seq_len, device=tensor.device, dtype=tensor.dtype)

        rand_curve = torch.tensor(np.random.randn(self.order, seq_len), device=tensor.device, dtype=tensor.dtype)
        weights = torch.abs(rand_curve).mean(dim=0)
        cum_curve = torch.cumsum(weights, dim=0)
        cum_curve = (cum_curve - cum_curve.min()) / (cum_curve.max() - cum_curve.min() + torch.finfo(tensor.dtype).eps)

        warped_positions = (1 - self.magnitude) * base_idx + self.magnitude * cum_curve * (seq_len - 1)
        warped_positions, _ = torch.sort(warped_positions)
        return _linear_interpolate_sequence(tensor, warped_positions)


class MaskOutTransform(_BaseInputTransform):
    """Mask out random values along the time dimension."""

    def __init__(self, parent: Transform, magnitude: float = 0.1, compensate: bool = False):
        super().__init__(parent)
        self.magnitude = magnitude
        self.compensate = compensate

    def _transform_input(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.magnitude <= 0:
            return tensor

        mask = torch.rand_like(tensor) < self.magnitude
        if self.compensate:
            masked_ratio = torch.sum(mask, dim=0, keepdim=True) / tensor.shape[0]
            masked_ratio = torch.clamp(masked_ratio, max=1 - torch.finfo(tensor.dtype).eps)
            tensor = tensor / (1 - masked_ratio)
        return tensor.masked_fill(mask, 0)


class TranslateXTransform(_BaseInputTransform):
    """Translate the sequence along the time axis by a random offset."""

    def __init__(self, parent: Transform, magnitude: float = 0.1):
        super().__init__(parent)
        self.magnitude = magnitude

    def _transform_input(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.magnitude <= 0:
            return tensor

        seq_len = tensor.shape[0]
        lambd = np.random.beta(self.magnitude, self.magnitude)
        lambd = min(lambd, 1 - lambd)
        shift = int(round(seq_len * lambd))
        if shift == 0 or shift == seq_len:
            return tensor

        if np.random.rand() < 0.5:
            shift = -shift

        new_start = max(0, shift)
        new_end = min(seq_len + shift, seq_len)
        start = max(0, -shift)
        end = min(seq_len - shift, seq_len)

        output = torch.zeros_like(tensor)
        output[new_start:new_end] = tensor[start:end]
        return output
