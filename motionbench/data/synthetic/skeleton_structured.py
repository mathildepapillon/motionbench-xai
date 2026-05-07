"""motionbench.data.synthetic.skeleton_structured — Skeleton-adjacency Gaussian dataset.

Data model::

    x ~ N(0, Sigma_joints ⊗ I_F ⊗ Sigma_time)

where:

* ``Sigma_joints`` encodes kinematic-tree graph distance via
  ``Sigma_joints[i, j] = decay ** d(i, j)``, with ``d`` the BFS distance
  on the H36M-17 kinematic tree.
* ``Sigma_time`` is an AR(1) matrix ``C[t, t'] = alpha ** |t - t'|``.

The combination yields correlations that are simultaneously structured in
the anatomical (skeleton graph) domain and the temporal (AR(1)) domain,
making it a useful synthetic benchmark for evaluating methods sensitive to
both spatial and temporal dependencies.

Conforms to :class:`~motionbench.data.base.GroundTruthDataset` via structural
subtyping (no inheritance required).

References
----------
Human3.6M skeleton (H36M-17 joint ordering):
    Ionescu, C., Papava, D., Olaru, V., & Sminchisescu, C. (2014).
    Human3.6M: Large scale datasets and predictive methods for 3D human
    sensing in natural environments. TPAMI 36(7).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from motionbench.data.synthetic.gaussian_motion import (
    GaussianMotionBenchmark,
    SigmaJointsFactory,
    SigmaTimeFactory,
)

if TYPE_CHECKING:
    from motionbench.oracles.gaussian_oracle import GaussianOracle

__all__ = ["SkeletonStructuredDataset"]

#: Default label function type: maps ``(x_np: (N, J, F, T), int)`` → ``(N,)`` int64 array.
LabelFunction = Callable[[npt.NDArray[Any], int], npt.NDArray[np.int64]]


def _default_label_fn(
    x_np: npt.NDArray[Any], n_classes: int
) -> npt.NDArray[np.int64]:
    """Quantile-split on joint-0 grand mean (proxy label until Task 1D).

    Args:
        x_np: ``(N, J, F, T)`` float32 array of samples.
        n_classes: Number of label classes.

    Returns:
        ``(N,)`` int64 class-label array in ``{0, ..., n_classes-1}``.
    """
    score = x_np[:, 0, :, :].mean(axis=(1, 2))  # (N,)
    bounds = np.percentile(score, np.linspace(0.0, 100.0, n_classes + 1)[1:-1])
    labels: npt.NDArray[np.int64] = np.searchsorted(bounds, score).astype(np.int64)
    return labels


class SkeletonStructuredDataset:
    """Gaussian motion with skeleton-adjacency Σ_joint and AR(1) Σ_time.

    Conforms to :class:`~motionbench.data.base.GroundTruthDataset` (structural
    Protocol, no inheritance required).

    The joint covariance ``Sigma_joints[i, j] = decay ** d(i, j)`` encodes
    kinematic-tree proximity: joints that are close on the skeleton tree are
    more correlated.  The temporal covariance is AR(1): nearby frames are
    correlated with coefficient ``alpha_time ** lag``.

    This combination is useful for benchmarking XAI methods that should
    respect both the spatial structure of the human body and the temporal
    autocorrelation of gait data.

    Args:
        J: Number of skeletal joints.  Must be 17 for the ``"h36m_17"``
            skeleton.
        F: Number of coordinates per joint (e.g. 3 for xyz).
        T: Number of frames per sequence.
        N: Number of sequences to pre-generate.
        alpha_time: AR(1) autocorrelation coefficient for temporal covariance.
        decay: Correlation decay per kinematic-tree hop.  ``decay=1`` →
            all-ones matrix; ``decay=0`` → identity.
        n_classes: Number of label classes.
        label_fn: Optional callable ``(x_np: (N, J, F, T), n_classes: int)``
            → ``(N,)`` int64 array.  Defaults to quantile-split on joint-0
            mean (proxy until Task 1D).
        seed: Random seed for sequence generation.

    Example:
        >>> ds = SkeletonStructuredDataset(J=17, T=81, N=100, seed=42)
        >>> x, y = ds[0]
        >>> x.shape
        torch.Size([17, 3, 81])
    """

    def __init__(
        self,
        J: int = 17,
        F: int = 3,
        T: int = 81,
        N: int = 1000,
        alpha_time: float = 0.9,
        decay: float = 0.5,
        n_classes: int = 3,
        label_fn: LabelFunction | None = None,
        seed: int = 0,
    ) -> None:
        self._J = J
        self._F = F
        self._T = T
        self._N = N
        self._n_classes = n_classes
        self._decay = decay
        self._alpha_time = alpha_time

        sigma_joints = SigmaJointsFactory.skeleton_adjacency(J=J, decay=decay)
        sigma_time = SigmaTimeFactory.ar1(T=T, alpha=alpha_time)

        self._benchmark = GaussianMotionBenchmark(
            J=J,
            F=F,
            T=T,
            sigma_joints=sigma_joints,
            sigma_time=sigma_time,
            sigma_joints_source="skeleton_adjacency_h36m17",
            sigma_time_source="ar1",
        )

        x_np = self._benchmark.sample(N, seed=seed)  # (N, J, F, T)

        if label_fn is None:
            y_np = _default_label_fn(x_np, n_classes)
        else:
            try:
                y_np = np.asarray(label_fn(x_np), dtype=np.int64)
            except TypeError:
                y_np = label_fn(x_np, n_classes)

        self._x: Tensor = torch.tensor(x_np, dtype=torch.float32)
        self._y: Tensor = torch.tensor(y_np, dtype=torch.int64)

        # Deferred import to avoid circular dependency at module level.
        from motionbench.oracles.gaussian_oracle import GaussianOracle  # noqa: PLC0415

        self._oracle: GaussianOracle = GaussianOracle(
            Sigma_joints=self._benchmark.Sigma_joints,
            Sigma_time=self._benchmark.Sigma_time,
        )

    # ------------------------------------------------------------------
    # GroundTruthDataset protocol
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return the ``idx``-th ``(x, y)`` pair.

        Args:
            idx: Sample index in ``[0, N)``.

        Returns:
            x: ``(J, F, T)`` float32 Tensor.
            y: scalar int64 Tensor.
        """
        return self._x[idx], self._y[idx]

    def __len__(self) -> int:
        """Number of pre-sampled sequences.

        Returns:
            N.
        """
        return self._N

    @property
    def shape(self) -> tuple[int, int, int]:
        """Spatial shape of every sample.

        Returns:
            ``(J, F, T)`` tuple.
        """
        return (self._J, self._F, self._T)

    @property
    def metadata(self) -> dict[str, object]:
        """Dataset-level metadata.

        Returns:
            Dict with keys ``skeleton``, ``frame_rate``, ``decay``,
            ``alpha_time``, ``n_classes``, ``sigma_joints_source``,
            ``sigma_time_source``.
        """
        return {
            "skeleton": "h36m_17",
            "frame_rate": 27.0,
            "decay": self._decay,
            "alpha_time": self._alpha_time,
            "n_classes": self._n_classes,
            "sigma_joints_source": self._benchmark.sigma_joints_source,
            "sigma_time_source": self._benchmark.sigma_time_source,
        }

    @property
    def oracle(self) -> GaussianOracle:
        """Ground-truth :class:`~motionbench.oracles.gaussian_oracle.GaussianOracle`.

        Returns:
            The oracle instance for this dataset.
        """
        return self._oracle
