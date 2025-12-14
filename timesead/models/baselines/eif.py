from typing import Optional, Tuple, Union, Literal
import numpy as np

import torch
from treeple import ExtendedIsolationForest

from ..common import AnomalyDetector

class EIFAD(AnomalyDetector):
    def __init__(
        self,
        n_trees: int = 200,
        sample_size: Union[int, Literal["auto"]] = "auto",
        contamination: Union[float, Literal["auto"]] = "auto",
        max_features: Union[int, float] = 1.0,
        bootstrap: bool = False,
        feature_combinations: Optional[int] = 2,
        n_jobs: Optional[int] = None,
        input_shape: str = "btf",
    ) -> None:
        """"
        Extended Isolation Forest Anomaly Detector.

        Thin wrapper around :class:`treeple.ExtendedIsolationForest` [Hariri2019]_.
        All arguments except ``input_shape`` are forwarded directly to
        ``treeple.ExtendedIsolationForest``; see its documentation for the
        precise meaning and valid ranges of these parameters.

        Implementation derived from
        https://github.com/HPI-Information-Systems/TimeEval-algorithms

        Parameters
        ----------
        n_trees : int, optional
            Forwarded as ``n_trees``.
        sample_size : int or {"auto"}, optional
            Forwarded as ``sample_size``.
        contamination : float or {"auto"}, optional
            Forwarded as ``contamination``.
        max_features : int or float, optional
            Forwarded as ``max_features``.
        bootstrap : bool, optional
            Forwarded as ``bootstrap``.
        feature_combinations : int, optional
            Forwarded as ``feature_combinations``.
        n_jobs : int or None, optional
            Forwarded as ``n_jobs``.
        input_shape : str, optional
            Expected input layout for this detector (e.g. ``"btf"`` for
            batch × time × features). This is handled by ``EIFAD`` itself
            and is not passed to ``treeple.ExtendedIsolationForest``.

        References
        ----------
        .. [Hariri2019] S. Hariri, M. C. Kind and R. J. Brunner,
           "Extended Isolation Forest," IEEE Transactions on Knowledge and
           Data Engineering, vol. 33, no. 4, pp. 1479–1489, 1 April 2021,
           doi:10.1109/TKDE.2019.2947676.
        """
        super(EIFAD, self).__init__()

        self.n_trees = n_trees
        self.sample_size = sample_size
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.feature_combinations = feature_combinations
        self.n_jobs = n_jobs
        self.input_shape = input_shape
        self.model = None


    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        # Merge all batches as batch processing is not possible
        data_full = []
        for (b_inputs, b_targets) in dataset:
            data = b_inputs[0]
            print(data.shape)
            batch_size, window_size, n_features = data.shape
            self.window_size = window_size
            data_full.append(data.reshape(batch_size, window_size * n_features))
        data_full = torch.cat(data_full, dim=0)

        data = data_full.cpu().detach().numpy().astype(np.float32)

        self.model = ExtendedIsolationForest(
            n_estimators=self.n_trees,
            max_samples=self.sample_size,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            feature_combinations=self.feature_combinations,
            n_jobs=self.n_jobs,
        ).fit(data)


    def compute_online_anomaly_score(
        self, inputs: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        batch_input = inputs[0]
        # Convert input to (B, T, D) dimension
        if self.input_shape[0] == "t":
            batch_input = batch_input.permute(1, 0, 2)

        if not hasattr(self, 'window_size'):
            raise RuntimeError('Run "fit" function before trying to compute_anomaly_score')
        # Get the final window for each batch
        data = batch_input[:, -self.window_size:, :]
        data = data.reshape(data.shape[0], -1)

        # Convert to numpy
        data = data.cpu().detach().numpy().astype(np.float32)
        scores = -torch.tensor(self.model.score_samples(data))

        return scores

    def compute_offline_anomaly_score(
        self, inputs: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:

        # Input of shape (B, T) or (T, B), output of shape (B)
        (target,) = targets

        # Just return the last label of the window
        return target[:, -1] if self.input_shape[0] == "b" else target[-1]
