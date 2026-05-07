"""motionbench.data.synthetic.low_rank_manifold — Low-rank-manifold Gaussian dataset.

Data model::

    x ~ N(0, Sigma_joints ⊗ I_F ⊗ Sigma_time)

where:

* ``Sigma_joints = U U^T + eps * I`` with ``U ∈ R^{J×rank}`` (low rank in joints).
  The data along the joint axis lives near a ``rank``-dimensional linear
  subspace of ``R^J``; ``eps`` controls the off-subspace noise floor.
* ``Sigma_time`` is an AR(1) matrix.

This construction exposes the on-/off-manifold distinction relevant for
SHAP-imputer evaluation (Frye et al. 2021; Chen et al. 2023): off-manifold
imputers (``ZeroImputer``, ``MeanImputer``) place mass perpendicular to the
column-span of ``U``, where the data has near-zero density. The exact-Gaussian
oracle preserves the manifold by virtue of the closed-form conditional formula.

Closed-form Shapley remains tractable because the data is still Gaussian; only
the covariance is rank-deficient (with a small ``eps`` ridge for numerical PSD).

Conforms to :class:`~motionbench.data.base.GroundTruthDataset` via structural
subtyping (no inheritance required).

References
----------
Frye, C., Rowat, C., & Feige, I. (2021).
    Shapley explainability on the data manifold. ICLR 2021.
Chen, J., et al. (2023).
    ManifoldShap: Shapley explainability with respect to the data manifold.
    arXiv:2301.04041.
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

__all__ = ["LowRankManifoldDataset"]

LabelFunction = Callable[[npt.NDArray[Any], int], npt.NDArray[np.int64]]


def _default_label_fn(
    x_np: npt.NDArray[Any], n_classes: int
) -> npt.NDArray[np.int64]:
    """Quantile-split on joint-0 grand mean.

    Args:
        x_np: ``(N, J, F, T)`` float32 array.
        n_classes: Number of label classes.

    Returns:
        ``(N,)`` int64 class-label array in ``{0, ..., n_classes-1}``.
    """
    score = x_np[:, 0, :, :].mean(axis=(1, 2))
    bounds = np.percentile(score, np.linspace(0.0, 100.0, n_classes + 1)[1:-1])
    labels: npt.NDArray[np.int64] = np.searchsorted(bounds, score).astype(np.int64)
    return labels


class LowRankManifoldDataset:
    """Gaussian motion with rank-deficient ``Sigma_joints`` (low-rank manifold).

    Conforms to :class:`~motionbench.data.base.GroundTruthDataset` (structural
    Protocol, no inheritance required).

    The joint covariance is constructed as ``U U^T + eps * I`` where
    ``U ∈ R^{J×rank}`` has i.i.d. Gaussian entries scaled by ``1/sqrt(rank)``.
    The data therefore lives near a ``rank``-dimensional linear subspace of
    ``R^J`` (per coordinate / time-slice), with off-subspace dispersion of
    standard deviation ``sqrt(eps)``.

    Args:
        J: Number of joints.
        F: Number of coordinates per joint.
        T: Number of frames per sequence.
        N: Number of sequences to pre-generate.
        rank: Effective rank of ``Sigma_joints``. Must be in ``[1, J]``.
        eps: Isotropic noise floor in ``Sigma_joints``.
        alpha_time: AR(1) coefficient for ``Sigma_time``.
        n_classes: Number of label classes.
        label_fn: Optional callable ``(x_np, n_classes) -> (N,) int64``.
            Defaults to quantile-split on joint-0 grand mean.
        seed: Random seed for ``U`` and sequence generation.
    """

    def __init__(
        self,
        J: int = 17,
        F: int = 3,
        T: int = 16,
        N: int = 200,
        rank: int = 4,
        eps: float = 1e-2,
        alpha_time: float = 0.9,
        n_classes: int = 3,
        label_fn: LabelFunction | None = None,
        seed: int = 0,
    ) -> None:
        if rank < 1 or rank > J:
            raise ValueError(f"rank must be in [1, J={J}]; got {rank}.")

        self._J = J
        self._F = F
        self._T = T
        self._N = N
        self._n_classes = n_classes
        self._rank = rank
        self._eps = eps
        self._alpha_time = alpha_time

        sigma_joints = SigmaJointsFactory.low_rank(J=J, rank=rank, eps=eps, seed=seed)
        sigma_time = SigmaTimeFactory.ar1(T=T, alpha=alpha_time)

        self._benchmark = GaussianMotionBenchmark(
            J=J,
            F=F,
            T=T,
            sigma_joints=sigma_joints,
            sigma_time=sigma_time,
            sigma_joints_source=f"low_rank(J={J},rank={rank},eps={eps})",
            sigma_time_source="ar1",
        )

        x_np = self._benchmark.sample(N, seed=seed)

        if label_fn is None:
            y_np = _default_label_fn(x_np, n_classes)
        else:
            try:
                y_np = np.asarray(label_fn(x_np), dtype=np.int64)
            except TypeError:
                y_np = label_fn(x_np, n_classes)

        self._x: Tensor = torch.tensor(x_np, dtype=torch.float32)
        self._y: Tensor = torch.tensor(y_np, dtype=torch.int64)

        from motionbench.oracles.gaussian_oracle import GaussianOracle  # noqa: PLC0415

        self._oracle: GaussianOracle = GaussianOracle(
            Sigma_joints=self._benchmark.Sigma_joints,
            Sigma_time=self._benchmark.Sigma_time,
        )

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return ``(x_idx, y_idx)``.

        Args:
            idx: Sample index in ``[0, N)``.

        Returns:
            x: ``(J, F, T)`` float32 Tensor.
            y: scalar int64 Tensor.
        """
        return self._x[idx], self._y[idx]

    def __len__(self) -> int:
        """Return ``N``."""
        return self._N

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return ``(J, F, T)``."""
        return (self._J, self._F, self._T)

    @property
    def metadata(self) -> dict[str, object]:
        """Return dataset-level metadata."""
        return {
            "skeleton": "low_rank_manifold",
            "rank": self._rank,
            "eps": self._eps,
            "alpha_time": self._alpha_time,
            "n_classes": self._n_classes,
            "sigma_joints_source": self._benchmark.sigma_joints_source,
            "sigma_time_source": self._benchmark.sigma_time_source,
        }

    @property
    def oracle(self) -> GaussianOracle:
        """Return the closed-form Gaussian oracle for this dataset."""
        return self._oracle
