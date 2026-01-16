import abc
import collections.abc
import functools
from typing import Tuple, Union, Callable, Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
#from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
from torch.utils.data._utils.collate import collate


class BaseTSDataset(abc.ABC, Dataset):
    """
    Base class for all time-series datasets in TimeSeAD. Implementing the members in this abstract class provides the
    data pipeline system with the necessary information to process the data correctly.
    """
    @abc.abstractmethod
    def __len__(self) -> int:
        """
        This should return the number of independent time series in the dataset
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def seq_len(self) -> Union[int, List[int]]:
        """
        This should return the length of each time series. If the time series have different lengths, the return
        value should be a list that contains the length of each sequence. If all sequences are of equal length,
        this should return an int.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_features(self) -> Union[int, Tuple[int, ...]]:
        """
        Number of features of each datapoint. This can also be a tuple if the data has more than one feature dimension.
        """
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        return 2

    @staticmethod
    @abc.abstractmethod
    def get_default_pipeline() -> Dict[str, Dict[str, Any]]:
        """
        Return the default pipeline for this dataset that is used if the user does not specify a different pipeline.
        This must be a dict of the form::

            {
                '<name>': {'class': '<name-of-transform-class>', 'args': {'<args-for-constructor>', ...}},
                ...
            }
        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_feature_names() -> List[str]:
        """
        Return names for the features in the order they are present in the data tensors.

        :return: A list of strings with names for each feature.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """
        Access the timeseries at position `index` and its corresponding label sequence. A call to this function should
        return a single time series that was sampled independently of the other time series in this dataset.

        :param index: The zero-based index of the time series to retrieve.
        :return: A tuple `(inputs, targets)`, where inputs is again a tuple of :class:`~torch.Tensor`\s with shape
            `(T, D*)`, where `D*` can very between the tensors. `targets` contains labels for the time series as tensors
            of shape `(T,)`.
        """
        raise NotImplementedError


def collate_tensor_fn(
    batch,
    *,
    collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None,
    batch_dim: int = 0
):
    """
    Puts each data field into a tensor with outer dimension batch size.

    This is largely copied from PyTorch's
    :func:`~torch.utils.data._utils.collate_tensor_fn` function, except that it
    allows stacking :class:`~torch.Tensor`\\s along an arbitrary dimension
    instead of always using the first dimension.

    Nested tensors and sparse tensors are not supported and will raise a
    :class:`RuntimeError`. In a DataLoader worker process, the output is
    allocated in shared memory to avoid an extra copy.

    :param batch: A sequence of tensors to be collated into a single batch
        tensor. All tensors in the batch must have the same shape.
    :type batch: Sequence[torch.Tensor]
    :param collate_fn_map: Unused in this implementation. Present only to match
        the signature of the higher-level :func:`collate` helper.
    :type collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]]
    :param batch_dim: The index of the dimension along which to stack the
        elements in the batch. For example, ``batch_dim=0`` produces the usual
        ``(batch_size, ...)`` layout, while ``batch_dim=-1`` appends the batch
        dimension as the last dimension.
    :type batch_dim: int

    :raises RuntimeError: If the input consists of nested tensors or sparse
        tensors, which are not supported by this function.

    :return: A single batched tensor whose shape is the input tensor shape with
        an additional batch dimension inserted at ``batch_dim``.
    :rtype: torch.Tensor
    """
    elem = batch[0]
    out = None
    if elem.is_nested:
        raise RuntimeError(
            "Batches of nested tensors are not currently supported by the default collate_fn; "
            "please provide a custom collate_fn to handle them appropriately."
        )
    if elem.layout in {
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_bsr,
        torch.sparse_csc,
        torch.sparse_bsc,
    }:
        raise RuntimeError(
            "Batches of sparse tensors are not currently supported by the default collate_fn; "
            "please provide a custom collate_fn to handle them appropriately."
        )
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum(x.numel() for x in batch)
        storage = elem._typed_storage()._new_shared(numel, device=elem.device)
        shape = list(elem.size())
        shape.insert(batch_dim, len(batch))
        out = elem.new(storage).resize_(*shape)
    return torch.stack(batch, dim=batch_dim, out=out)


def collate_fn(batch_dim: int) -> Callable:
    """
    Factory function that creates a :class:`~torch.utils.data.DataLoader`
    ``collate_fn`` which stacks tensors along a configurable batch dimension.

    The returned callable can be passed directly to
    :class:`~torch.utils.data.DataLoader` via its ``collate_fn`` argument. It
    behaves like the default PyTorch collate function, but uses
    :func:`collate_tensor_fn` to collate tensors, allowing control over where
    the batch dimension is inserted.

    Examples
    --------
    Use a non-standard batch dimension (e.g., append batch as the last dim):

    .. code-block:: python

        loader = DataLoader(
            dataset,
            batch_size=32,
            collate_fn=collate_fn(batch_dim=-1),
        )

    :param batch_dim: The dimension along which batched tensors will be
        stacked. This value is forwarded to :func:`collate_tensor_fn`.
    :type batch_dim: int

    :return: A collate function suitable for use with
        :class:`~torch.utils.data.DataLoader`.
    :rtype: Callable
    """
    collate_fn_map = {torch.Tensor: functools.partial(collate_tensor_fn, batch_dim=batch_dim)}
    return functools.partial(collate, collate_fn_map=collate_fn_map)

