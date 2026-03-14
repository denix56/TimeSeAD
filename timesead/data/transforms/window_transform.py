import logging
from bisect import bisect_right
from itertools import accumulate
from typing import Tuple, Union, List, Optional, Iterable

import torch

from .._debug_timing import run_with_debug_timing
from .transform_base import Transform
from ...utils.utils import ceil_div

_logger = logging.getLogger(__name__)


class WindowTransform(Transform):
    """
    This :class:`~timesead.data.transforms.Transform` produces sliding windows from input sequences. Incomplete windows
        (that can appear if ``step_size>1``) will not be returned.
    """
    def __init__(self, parent: Transform, window_size: int, step_size: int = 1, reverse: bool = False):
        """

        :param parent: Another :class:`~timesead.data.transforms.Transform` which is used as the data source for this
            :class:`~timesead.data.transforms.Transform`.
        :param window_size: The size of each window.
        :param step_size: The step size at which the sliding window is moved along the sequence.
        :param reverse: If this is `True`, start the sliding window at the end of a sequence, instead of the start.
            Note that this will not reverse the order of sequences in the dataset and only applies within a single
            sequence.
        """
        super(WindowTransform, self).__init__(parent)
        self._window_size = window_size
        self.step_size = step_size
        self.reverse = reverse
        self._cached_parent_seq_len: Optional[Tuple[int, ...]] = None
        self._cached_parent_cum_windows: Optional[Tuple[int, ...]] = None

    def _compute_windowed_len(self, old_n: int, old_ts: Union[int, Iterable[int]]) -> int:
        if isinstance(old_ts, int):
            return old_n * ceil_div(max((old_ts - self._window_size + 1), 0), self.step_size)

        return sum(ceil_div(max((old_t - self._window_size + 1), 0), self.step_size) for old_t in old_ts)

    def _get_parent_window_index_cache(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        if self._cached_parent_seq_len is None or self._cached_parent_cum_windows is None:
            seq_len = self.parent.seq_len
            if isinstance(seq_len, int):
                raise TypeError('Cached parent window index metadata only exists for variable-length sequences.')

            self._cached_parent_seq_len = tuple(seq_len)
            window_counts = tuple(
                ceil_div(max((seq_l - self._window_size + 1), 0), self.step_size)
                for seq_l in self._cached_parent_seq_len
            )
            self._cached_parent_cum_windows = tuple(accumulate(window_counts))

        return self._cached_parent_seq_len, self._cached_parent_cum_windows

    def _inverse_transform_index(self, item, seq_len: Union[int, Iterable[int]]) -> Tuple[int, int]:
        ts_index = window_start = 0
        effective_seq_len = seq_len
        if isinstance(seq_len, int):
            # Every sequence has the same length
            windows_per_seq = ceil_div(max((seq_len - self._window_size + 1), 0), self.step_size)
            ts_index, window_start = divmod(item, windows_per_seq)
            window_start *= self.step_size
        else:
            # Sequences have different lengths. Cache cumulative window counts so each lookup is O(log n).
            seq_len_values, cum_window_counts = self._get_parent_window_index_cache()
            ts_index = bisect_right(cum_window_counts, item)
            prev_total_windows = cum_window_counts[ts_index - 1] if ts_index > 0 else 0
            window_start = (item - prev_total_windows) * self.step_size
            effective_seq_len = seq_len_values[ts_index]

        if self.reverse:
            window_start = effective_seq_len - window_start - self._window_size

        return ts_index, window_start

    def _get_datapoint_impl(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        old_i, start = self._inverse_transform_index(item, self.parent.seq_len)
        end = start + self._window_size
        inputs, targets = self.parent.get_datapoint(old_i)

        out_inputs = tuple(inp[start:end] for inp in inputs)
        out_targets = tuple(t[start:end] for t in targets)

        return out_inputs, out_targets

    def __len__(self):
        if not self._has_cached_len:
            old_n = len(self.parent)
            old_ts = self.parent.seq_len
            self._cached_len = self._compute_windowed_len(old_n, old_ts)
            self._has_cached_len = True

        return self._cached_len

    @property
    def seq_len(self):
        return self._window_size

    @property
    def window_size(self) -> Optional[int]:
        return None


class WindowTransformIfNotWindow(WindowTransform):
    def __init__(self, parent: Transform, window_size: int, step_size: int = 1, reverse: bool = False):
        super().__init__(parent, window_size, step_size=step_size, reverse=reverse)
        self._cached_flat_seq_len: Optional[Tuple[int, ...]] = None
        self._cached_flat_cum_seq_len: Optional[Tuple[int, ...]] = None

    def _get_parent_flatten_index_cache(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        if self._cached_flat_seq_len is None or self._cached_flat_cum_seq_len is None:
            seq_len = self.parent.seq_len
            if isinstance(seq_len, int):
                seq_len = (seq_len,) * len(self.parent)
            else:
                seq_len = tuple(seq_len)

            self._cached_flat_seq_len = seq_len
            self._cached_flat_cum_seq_len = tuple(accumulate(seq_len))

        return self._cached_flat_seq_len, self._cached_flat_cum_seq_len

    def _fetch_datapoint(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        if self.parent.ndim == 2:
            return super(WindowTransformIfNotWindow, self)._get_datapoint_impl(item)

        old_i, start = self._inverse_transform_index(item, self.parent.window_size)
        end = start + self._window_size
        _, cum_seq_len = self._get_parent_flatten_index_cache()
        idx = bisect_right(cum_seq_len, old_i)
        seq_start = cum_seq_len[idx - 1] if idx > 0 else 0
        item_idx = old_i - seq_start

        inputs, targets = self.parent.get_datapoint(idx)
        use_full_window = start == 0 and self._window_size == self.parent.window_size
        if use_full_window:
            inputs = tuple(inp[item_idx] for inp in inputs)
        else:
            inputs = tuple(inp[item_idx, start:end] for inp in inputs)

        targets = tuple(
            tgt[item_idx]
            if tgt.ndim == 1 or use_full_window
            else tgt[item_idx, start:end]
            for tgt in targets
        )
        return inputs, targets

    def _get_datapoint_impl(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        return run_with_debug_timing(
            _logger,
            'WindowTransformIfNotWindow._get_datapoint_impl',
            lambda: self._fetch_datapoint(item),
            index_label='item_idx',
            index_value=item,
            initialize_logging=True,
        )

    def __len__(self):
        if self.parent.ndim == 2:
            return super().__len__()

        if not self._has_cached_len:
            seq_len, _ = self._get_parent_flatten_index_cache()
            self._cached_len = sum(self._compute_windowed_len(sl, self.parent.window_size) for sl in seq_len)
            self._has_cached_len = True

        return self._cached_len

    @property
    def ndim(self) -> int:
        return 2

    @property
    def window_size(self) -> Optional[int]:
        return None
