"""motionbench.data.synthetic.gait_periodic — Gait-periodic Gaussian motion dataset.

Data model::

    x ~ N(0, Sigma_joints ⊗ I_F ⊗ Sigma_time)

where:

* ``Sigma_joints`` is an equicorrelated (diagonal / identity) matrix,
  i.e. ``rho=0`` — joints are independent.  The gait-periodic structure
  is entirely captured in the temporal covariance.
* ``Sigma_time`` is a sum-of-cosines (Toeplitz) kernel::

        k(t, t') = Σ_{h=1}^{n_harmonics} cos(2π h |t-t'| / period) + ε I

  This creates temporal autocorrelations that peak at the stride period
  and its harmonics, mimicking the periodicity of real gait cycles.

The ``period_mean`` and ``period_std`` parameters document the intended
distribution over stride periods.  The covariance matrix is built from
``period_mean`` (a fixed kernel per dataset instance).  Per-sequence period
variability — drawing a fresh period per sample from
``N(period_mean, period_std²)`` — is deferred to future work.

Conforms to :class:`~motionbench.data.base.GroundTruthDataset` via structural
subtyping (no inheritance required).

References
----------
Ported and refactored from ``CARE-PD/synthetic/diagnostic_motion.py``
(Fourier-series gait generator with independent joints).

MacKay, D. J. C. (1998).
    Introduction to Gaussian Processes.  §4.2 — Periodic kernels.
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

__all__ = ["GaitPeriodicDataset", "gait_stddev_label_fn"]

#: Label function type: ``(x_np: (N, J, F, T), n_classes: int)`` → ``(N,)`` int64.
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


def gait_stddev_label_fn(
    x_np: npt.NDArray[Any], n_classes: int
) -> npt.NDArray[np.int64]:
    """Quantile-split on temporal standard deviation of joint-0 signal.

    More learnable than the grand-mean default when the cosine temporal kernel
    produces near-zero mean across the window (e.g. near-integer T/period ratio).
    The temporal std dev reflects the amplitude of the gait oscillation and
    varies meaningfully across samples drawn from the gait covariance.

    Args:
        x_np: ``(N, J, F, T)`` float32 array of samples.
        n_classes: Number of label classes.

    Returns:
        ``(N,)`` int64 class-label array in ``{0, ..., n_classes-1}``.
    """
    score = x_np[:, 0, :, :].std(axis=(1, 2))  # temporal SD of joint-0  (N,)
    bounds = np.percentile(score, np.linspace(0.0, 100.0, n_classes + 1)[1:-1])
    labels: npt.NDArray[np.int64] = np.searchsorted(bounds, score).astype(np.int64)
    return labels


class GaitPeriodicDataset:
    """Gaussian motion with gait-periodic Toeplitz Σ_time.

    Conforms to :class:`~motionbench.data.base.GroundTruthDataset` (structural
    Protocol, no inheritance required).

    The temporal covariance is a sum-of-cosines Toeplitz kernel whose
    autocovariance function peaks at integer multiples of ``period_mean``
    frames.  Joints are independent (identity joint covariance), so the
    gait periodicity is the sole source of inter-sample structure.

    ``period_std`` documents the intended variability of stride periods across
    subjects / trials but does not currently affect the generated data (the
    covariance kernel is fixed at ``period_mean``).  Per-sequence period
    variability is left to future work.

    Args:
        J: Number of skeletal joints.
        F: Number of coordinates per joint (e.g. 3 for xyz).
        T: Number of frames per sequence.
        N: Number of sequences to pre-generate.
        period_mean: Mean gait cycle period in frames (e.g. 27 ≈ 1 s at
            27 fps).  Used as the kernel period.
        period_std: Standard deviation of the stride period distribution
            (frames).  Stored as metadata; not currently used for sampling.
        n_harmonics: Number of cosine harmonics in the kernel.
        n_classes: Number of label classes.
        label_fn: Optional callable ``(x_np: (N, J, F, T), n_classes: int)``
            → ``(N,)`` int64 array.  Defaults to quantile-split on joint-0
            mean (proxy until Task 1D).
        seed: Random seed for sequence generation.

    Example:
        >>> ds = GaitPeriodicDataset(T=81, N=100, period_mean=27.0, seed=42)
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
        period_mean: float = 27.0,
        period_std: float = 2.0,
        n_harmonics: int = 3,
        n_classes: int = 3,
        label_fn: LabelFunction | None = None,
        seed: int = 0,
    ) -> None:
        self._J = J
        self._F = F
        self._T = T
        self._N = N
        self._period_mean = period_mean
        self._period_std = period_std
        self._n_harmonics = n_harmonics
        self._n_classes = n_classes

        # Independent joints (rho=0): identity joint covariance.
        sigma_joints = SigmaJointsFactory.equicorrelated(J=J, rho=0.0)
        sigma_time = SigmaTimeFactory.gait_periodic(
            T=T, period=period_mean, n_harmonics=n_harmonics
        )

        self._benchmark = GaussianMotionBenchmark(
            J=J,
            F=F,
            T=T,
            sigma_joints=sigma_joints,
            sigma_time=sigma_time,
            sigma_joints_source="equicorrelated_rho0",
            sigma_time_source="gait_periodic_toeplitz",
        )

        x_np = self._benchmark.sample(N, seed=seed)  # (N, J, F, T)

        _label_fn: LabelFunction = label_fn if label_fn is not None else _default_label_fn
        y_np = _label_fn(x_np, n_classes)

        self._x: Tensor = torch.tensor(x_np, dtype=torch.float32)
        self._y: Tensor = torch.tensor(y_np, dtype=torch.int64)

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
            Dict with keys ``skeleton``, ``frame_rate``, ``period_mean``,
            ``period_std``, ``n_harmonics``, ``n_classes``,
            ``sigma_joints_source``, ``sigma_time_source``.
        """
        return {
            "skeleton": "synthetic_gait_periodic",
            "frame_rate": 27.0,
            "period_mean": self._period_mean,
            "period_std": self._period_std,
            "n_harmonics": self._n_harmonics,
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
