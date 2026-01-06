import functools
import math
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
    """Subsample sequences by aggregating consecutive observations.

    :param parent: Another :class:`~timesead.data.transforms.Transform` which is used as the data source for this
        :class:`~timesead.data.transforms.Transform`.
    :param subsampling_factor: Number of consecutive data points that will be aggregated into a single output point.
    :param aggregation: Aggregation strategy for each subsampling window. Can be either ``"mean"``, ``"last"`` or
        ``"first"``.
    """
    def __init__(self, parent: Transform, subsampling_factor: int, aggregation: str = 'first'):
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
    """Cache results from the parent transform to avoid recomputation.

    :param parent: Another :class:`~timesead.data.transforms.Transform` which is used as the data source for this
        :class:`~timesead.data.transforms.Transform`.
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
    """Limit the maximum number of datapoints exposed by a dataset chain.

    :param parent: Upstream :class:`~timesead.data.transforms.Transform` that provides datapoints.
    :param max_count: Maximum number of datapoints that can be accessed.
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

    def __init__(self, parent: Transform, apply_prob: float = 1.0):
        super().__init__(parent)
        self.apply_prob = apply_prob

    def _transform_input(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _get_datapoint_impl(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        if random.random() >= self.apply_prob:
            return self.parent.get_datapoint(item)

        inputs, targets = self.parent.get_datapoint(item)
        inputs = tuple(self._transform_input(inp) for inp in inputs)
        return inputs, targets


class MagAddNoiseTransform(_BaseInputTransform):
    r"""Apply additive Gaussian noise scaled by the local signal magnitude.

    The transform perturbs each timestep ``x_t`` with zero-mean Gaussian noise scaled by the standard deviation of
    the finite differences ``\sigma`` and the configured ``magnitude`` factor ``\lambda``:

    .. math::

       x'_t = x_t + \epsilon_t \cdot \sigma \cdot \lambda, \qquad \epsilon_t \sim \mathcal{N}(0, 1/3).

    :param parent: Upstream :class:`~timesead.data.transforms.Transform` that provides datapoints.
    :param magnitude: Scaling factor ``\lambda`` applied to the estimated standard deviation.
    :param apply_prob: Probability of applying the noise to a datapoint.
    """

    def __init__(self, parent: Transform, magnitude: float = 1.0, apply_prob: float = 1.0):
        super().__init__(parent, apply_prob)
        self.magnitude = magnitude

    def _transform_input(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.magnitude <= 0 or tensor.shape[0] < 2:
            return tensor

        diffs = tensor[1:] - tensor[:-1]
        std = diffs.std(dim=0, keepdim=True).clamp_min(torch.finfo(tensor.dtype).eps)
        noise = torch.normal(0, 1 / 3, size=tensor.shape, device=tensor.device, dtype=tensor.dtype)
        return tensor + noise * std * self.magnitude


class MagScaleTransform(_BaseInputTransform):
    r"""Randomly scale the input magnitude by a sampled factor.

    A random non-negative draw ``r`` from a half-normal distribution determines the scale ``s``; with probability
    :math:`\tfrac{1}{3}` the series is amplified and otherwise attenuated:

    .. math::

       s = \begin{cases}
           1 + r \cdot \lambda & \text{if } u < \tfrac{1}{3} \\
           1 - \tfrac{r \cdot \lambda}{2} & \text{otherwise}
       \end{cases}

    yielding the transformed series :math:`x'_t = s \cdot x_t`.

    :param parent: Upstream :class:`~timesead.data.transforms.Transform` that provides datapoints.
    :param magnitude: Magnitude ``\lambda`` controlling the maximum deviation from the original scale.
    :param apply_prob: Probability of applying the scaling to a datapoint.
    """

    def __init__(self, parent: Transform, magnitude: float = 0.5, apply_prob: float = 1.0):
        super().__init__(parent, apply_prob)
        self.magnitude = magnitude

    def _transform_input(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.magnitude <= 0:
            return tensor

        rand = torch.abs(torch.randn((), device=tensor.device, dtype=tensor.dtype))
        scale = (1 - (rand * self.magnitude) / 2) if random.random() > 1 / 3 else (1 + rand * self.magnitude)
        return tensor * scale


class TimeWarpTransform(_BaseInputTransform):
    r"""Apply a smooth random time warping to the sequence.

    A cumulative random curve ``w`` defines new monotonically increasing sampling positions ``\tau_t`` that blend the
    original time index ``t`` with the normalized curve using the magnitude ``\lambda``:

    .. math::

       \tau_t = (1 - \lambda) \cdot t + \lambda \cdot w_t \cdot (T - 1),

    and the output is obtained through linear interpolation :math:`x'_t = x(\tau_t)`.

    :param parent: Upstream :class:`~timesead.data.transforms.Transform` that provides datapoints.
    :param magnitude: Warping intensity ``\lambda`` in ``[0, 1]``.
    :param order: Number of random basis curves used to construct the warp field.
    :param apply_prob: Probability of applying the warp to a datapoint.
    """

    def __init__(self, parent: Transform, magnitude: float = 0.1, order: int = 6, apply_prob: float = 1.0):
        super().__init__(parent, apply_prob)
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
    r"""Mask out random values along the time dimension.

    Each element is independently dropped with probability ``\lambda`` using a Bernoulli mask ``m_t``. The optional
    compensation rescales the surviving values to preserve the expected magnitude:

    .. math::

       x'_t = (1 - m_t) \cdot x_t \cdot \frac{1}{1 - \lambda} \quad \text{if compensate, else} \quad x'_t = (1 - m_t) \cdot x_t.

    :param parent: Upstream :class:`~timesead.data.transforms.Transform` that provides datapoints.
    :param magnitude: Masking probability ``\lambda`` for each element.
    :param compensate: Whether to renormalize unmasked values by ``\tfrac{1}{1-\lambda}``.
    :param apply_prob: Probability of applying the masking to a datapoint.
    """

    def __init__(self, parent: Transform, magnitude: float = 0.1, compensate: bool = False, apply_prob: float = 1.0):
        super().__init__(parent, apply_prob)
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
    r"""Translate the sequence along the time axis by a random offset.

    A shift length ``k`` is sampled from a symmetric beta distribution scaled by the sequence length ``T``. The output
    sequence is defined as

    .. math::

       x'_t = \begin{cases}
           x_{t-k} & 0 \leq t-k < T \\
           0 & \text{otherwise}
       \end{cases}

    :param parent: Upstream :class:`~timesead.data.transforms.Transform` that provides datapoints.
    :param magnitude: Beta distribution concentration parameter controlling the expected shift proportion.
    :param apply_prob: Probability of applying the translation to a datapoint.
    """

    def __init__(self, parent: Transform, magnitude: float = 0.1, apply_prob: float = 1.0):
        super().__init__(parent, apply_prob)
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
