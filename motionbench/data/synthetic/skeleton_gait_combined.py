"""motionbench.data.synthetic.skeleton_gait_combined — Skeleton-graph × gait-periodic dataset.

Fourth quadrant of the 2x2 (low/high spatial × low/high temporal) synthetic design
grid.  Combines the two sources of structure that the existing pillar datasets
isolate:

* high spatial coupling (anatomical chain) — taken from
  :class:`~motionbench.data.synthetic.skeleton_structured.SkeletonStructuredDataset`,
* strong periodic temporal kernel (sum-of-cosines stride covariance) — taken from
  :class:`~motionbench.data.synthetic.gait_periodic.GaitPeriodicDataset`.

The full per-element covariance is the Kronecker product

::

    Sigma = Sigma_skeleton (J x J)  ⊗  I_F  ⊗  Sigma_periodic (T x T)

so the closed-form Gaussian Shapley oracle from
:class:`~motionbench.oracles.gaussian_oracle.GaussianOracle` applies directly,
just like the other pillar Gaussian datasets.  This is the regime closest to real
motion data because both spatial (skeleton) and temporal (gait) structures coexist.

Conforms to :class:`~motionbench.data.base.GroundTruthDataset` via structural
subtyping (no inheritance required).

References
----------
Olsen, L. R., Glad, I. K., Hjort, N. L., & Tveten, M. (2022).
    Using Shapley Values and Variational Autoencoders To Explain Predictions
    from Neural Networks for Short-Term Wind Power Forecasting. JMLR 23(1), 1–38.
MacKay, D. J. C. (1998). Introduction to Gaussian Processes. §4.2 — Periodic kernels.
Ionescu, C. et al. (2014). Human3.6M: TPAMI 36(7) — H36M-17 skeleton.
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

__all__ = ["SkeletonGaitDataset"]

LabelFunction = Callable[[npt.NDArray[Any], int], npt.NDArray[np.int64]]


def _default_label_fn(
    x_np: npt.NDArray[Any], n_classes: int
) -> npt.NDArray[np.int64]:
    """Quantile-split on joint-0 grand mean (matches the other pillar datasets)."""
    score = x_np[:, 0, :, :].mean(axis=(1, 2))
    bounds = np.percentile(score, np.linspace(0.0, 100.0, n_classes + 1)[1:-1])
    labels: npt.NDArray[np.int64] = np.searchsorted(bounds, score).astype(np.int64)
    return labels


class SkeletonGaitDataset:
    """Gaussian motion with skeleton-adjacency Σ_joint and gait-periodic Σ_time.

    The fourth quadrant of the 2x2 synthetic design grid: high-spatial × high-temporal.

    Args:
        J: Number of skeletal joints. Must be 17 for ``"h36m_17"`` skeleton.
        F: Number of coordinates per joint (3 for xyz).
        T: Number of frames per sequence.
        N: Number of sequences to pre-generate.
        decay: Correlation decay per kinematic-tree hop. ``decay=1`` → all-ones
            matrix; ``decay=0`` → identity.  Same default as
            :class:`SkeletonStructuredDataset` (0.5).
        period_mean: Mean gait cycle period in frames. Same default as
            :class:`GaitPeriodicDataset` (period=7 → ~2.3 cycles in T=16 frames,
            non-integer T/period ratio so the cosine kernel does not cancel
            over the window and the grand-mean label has signal).
        period_std: Documented variability across stride periods (not currently
            used for sampling; covariance is built from ``period_mean``).
        n_harmonics: Number of cosine harmonics in the temporal kernel.
        n_classes: Number of label classes.
        label_fn: Optional callable ``(x_np: (N, J, F, T), n_classes: int)``
            → ``(N,)`` int64 array. Defaults to quantile-split on joint-0
            grand mean to match the other pillar Gaussian datasets.
        seed: Random seed for sequence generation.

    Example:
        >>> ds = SkeletonGaitDataset(J=17, T=16, N=100, seed=42)
        >>> x, y = ds[0]
        >>> x.shape
        torch.Size([17, 3, 16])
    """

    def __init__(
        self,
        J: int = 17,
        F: int = 3,
        T: int = 16,
        N: int = 200,
        decay: float = 0.5,
        period_mean: float = 7.0,
        period_std: float = 1.0,
        n_harmonics: int = 3,
        n_classes: int = 3,
        label_fn: LabelFunction | None = None,
        seed: int = 0,
    ) -> None:
        self._J = J
        self._F = F
        self._T = T
        self._N = N
        self._decay = decay
        self._period_mean = period_mean
        self._period_std = period_std
        self._n_harmonics = n_harmonics
        self._n_classes = n_classes

        sigma_joints = SigmaJointsFactory.skeleton_adjacency(J=J, decay=decay)
        sigma_time = SigmaTimeFactory.gait_periodic(
            T=T, period=period_mean, n_harmonics=n_harmonics
        )

        self._benchmark = GaussianMotionBenchmark(
            J=J,
            F=F,
            T=T,
            sigma_joints=sigma_joints,
            sigma_time=sigma_time,
            sigma_joints_source="skeleton_adjacency_h36m17",
            sigma_time_source="gait_periodic_toeplitz",
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
        return self._x[idx], self._y[idx]

    def __len__(self) -> int:
        return self._N

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self._J, self._F, self._T)

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "skeleton": "h36m_17",
            "frame_rate": 27.0,
            "decay": self._decay,
            "period_mean": self._period_mean,
            "period_std": self._period_std,
            "n_harmonics": self._n_harmonics,
            "n_classes": self._n_classes,
            "sigma_joints_source": self._benchmark.sigma_joints_source,
            "sigma_time_source": self._benchmark.sigma_time_source,
        }

    @property
    def oracle(self) -> GaussianOracle:
        return self._oracle
