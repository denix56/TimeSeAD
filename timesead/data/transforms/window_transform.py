from typing import Tuple, Union, List, Optional

import numpy as np
import torch

from .transform_base import Transform
from ...utils.utils import ceil_div


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

    def _inverse_transform_index(self, item) -> Tuple[int, int]:
        seq_len = self.parent.seq_len

        ts_index = window_start = 0
        if isinstance(seq_len, int):
            # Every sequence has the same length
            windows_per_seq = ceil_div(max((seq_len - self._window_size + 1), 0), self.step_size)
            ts_index, window_start = divmod(item, windows_per_seq)
            window_start *= self.step_size
        else:
            # Sequences have different length
            total_windows = old_total_windows = 0
            for i, seq_l in enumerate(seq_len):
                windows_per_seq = ceil_div(max((seq_l - self._window_size + 1), 0), self.step_size)
                old_total_windows = total_windows
                total_windows += windows_per_seq
                if total_windows > item:
                    ts_index = i
                    window_start = (item - old_total_windows) * self.step_size
                    break

        if self.reverse:
            window_start = seq_len - window_start - self._window_size

        return ts_index, window_start

    def _get_datapoint_impl(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        old_i, start = self._inverse_transform_index(item)
        end = start + self._window_size
        inputs, targets = self.parent.get_datapoint(old_i)

        out_inputs = tuple(inp[start:end] for inp in inputs)
        out_targets = tuple(t[start:end] for t in targets)

        return out_inputs, out_targets

    def __len__(self):
        old_n = len(self.parent)
        old_ts = self.parent.seq_len
        if isinstance(old_ts, int):
            return old_n * ceil_div(max((old_ts - self._window_size + 1), 0), self.step_size)

        return sum(ceil_div(max((old_t - self._window_size + 1), 0), self.step_size) for old_t in old_ts)

    @property
    def seq_len(self):
        return self._window_size

    def window_size(self) -> Optional[int]:
        return None


class WindowTransformIfNotWindow(WindowTransform):
    def _get_datapoint_impl(self, item: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        if self.parent.ndim == 2:
            return super()._get_datapoint_impl(item)
        else:
            seq_len = self.parent.seq_len
            if isinstance(seq_len, int):
                seq_len = [seq_len]
            seq_len = np.asarray(seq_len)
            cum_seq_len = np.cumsum(seq_len)
            idx = int(np.searchsorted(cum_seq_len, item, side='right'))
            item_idx = (item - cum_seq_len[idx - 1]) if idx > 0 else item

            inputs, targets = self.parent.get_datapoint(idx)
            inputs = tuple(inp[item_idx] for inp in inputs)
            targets = tuple(tgt[item_idx] for tgt in targets)
            return inputs, targets

    def __len__(self):
        if self.parent.ndim == 2:
            return super().__len__()
        else:
            seq_len = self.parent.seq_len
            if isinstance(seq_len, int):
                seq_len = [seq_len]
            seq_len = sum(seq_len)
            return seq_len

    def seq_len(self) -> Union[int, List[int]]:
        if self.parent.ndim == 2:
            return super().seq_len
        else:
            return self.parent.window_size

    def ndim(self) -> int:
        return 2

    def window_size(self) -> Optional[int]:
        return None