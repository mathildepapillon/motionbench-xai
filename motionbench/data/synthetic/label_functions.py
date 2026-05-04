"""motionbench.data.synthetic.label_functions — Synthetic label function library.

This module provides a hierarchy of :class:`LabelFunction` implementations for
assigning integer class labels to motion sequences of shape ``(N, J, F, T)``.
All implementations binarise a continuous score into ``n_classes`` classes via
quantile-based cutoffs (ternary 0/1/2 by default).

Every :class:`LabelFunction` exposes :py:meth:`important_players`, which returns
the set of player indices whose coordinates causally drive the label.  This set
is used as ground-truth by the ``TopKRecovery`` metric and related attribution
evaluators.

Label functions implemented
---------------------------
* :class:`Linear`                  — weighted sum of per-coordinate means.
* :class:`OlsenInteraction`        — nonlinear K-window interaction (Olsen et al.
                                     JMLR 2022 Eq. 12 adaptation).
* :class:`SpatialOlsen`            — Olsen term over 4 designated signal joints.
* :class:`LocalizedTemporal`       — score depends only on one temporal window.
* :class:`LocalizedSpatial`        — score depends only on one spatial joint.
* :class:`LocalizedSpatiotemporal` — score depends only on one (joint, window) cell.

References
----------
Olsen, L. R., Glad, I. K., Hjort, N. L., & Tveten, M. (2022).
    Using Shapley Values and Variational Autoencoders To Explain
    Predictions from Neural Networks for Short-Term Wind Power
    Forecasting.  JMLR 23(1), 1–38.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import torch
from scipy.special import ndtr

if TYPE_CHECKING:
    from collections.abc import Callable

    from motionbench.players.base import PlayerSet

__all__ = [
    "LabelFunction",
    "Linear",
    "OlsenInteraction",
    "SpatialOlsen",
    "LocalizedTemporal",
    "LocalizedSpatial",
    "LocalizedSpatiotemporal",
]

# Convenient type alias used throughout this module.
_Array = npt.NDArray[Any]


# ---------------------------------------------------------------------------
# Module-level helpers (ported from CARE-PD/synthetic/gaussian_motion.py)
# ---------------------------------------------------------------------------


def _per_window_grand_means(
    x: _Array,
    window_assignments: list[list[int]],
) -> _Array:
    """Compute per-window scalar grand means over all (j, f) in each window.

    Args:
        x: ``(N, J, F, T)`` float array.
        window_assignments: K lists of frame indices, one per window.

    Returns:
        ``(N, K)`` float64 array of per-window grand means.
    """
    return np.stack(
        [x[:, :, :, frames].mean(axis=(1, 2, 3)) for frames in window_assignments],
        axis=1,
    )


def _olsen_term(
    u: _Array,
    feat_idx: tuple[int, int, int, int],
    coeffs: tuple[float, float, float],
) -> _Array:
    """One Olsen-style nonlinear interaction term across four feature indices.

    Given standardised uniform features ``u`` (shape ``(N, P)``) and four feature
    indices ``(a, b, c, d)``, computes::

        c1 · sin(π · u_a · u_b) + c2 · u_c · exp(c3 · u_c · u_d)

    All four features appear in the output so each one has a well-defined Shapley
    contribution — neither the sin term nor the exp term factorises across its pair.

    Args:
        u: ``(N, P)`` array of standardised uniform features in ``[0, 1]``.
        feat_idx: Four feature indices ``(a, b, c, d)``.
        coeffs: Three scalar coefficients ``(c1, c2, c3)``.

    Returns:
        ``(N,)`` float64 array of term values.
    """
    a, b, c, d = feat_idx
    c1, c2, c3 = coeffs
    result: _Array = (
        c1 * np.sin(np.pi * u[:, a] * u[:, b])
        + c2 * u[:, c] * np.exp(c3 * u[:, c] * u[:, d])
    )
    return result


def _nonlinear_olsen_score(
    x: _Array,
    window_assignments: list[list[int]],
    sigma_k: _Array,
    coeffs: _Array,
) -> _Array:
    """Nonlinear response adapted from Olsen et al. (JMLR 2022) Eq. (12).

    Generalised to arbitrary K divisible by 4 by tiling the base Olsen term
    over consecutive groups of 4 windows.  For K=4 this reduces exactly to::

        score = c1·sin(π·u0·u1) + c2·u2·exp(c3·u2·u3)

    Args:
        x: ``(N, J, F, T)`` float array.
        window_assignments: K lists of frame indices.
        sigma_k: ``(K,)`` per-window standard deviations for normalisation.
        coeffs: ``(n_terms, 3)`` coefficient matrix where ``n_terms = K // 4``.

    Returns:
        ``(N,)`` float64 score array.

    Raises:
        ValueError: if ``K != 4 * n_terms``.
    """
    w = _per_window_grand_means(x, window_assignments)  # (N, K)
    u: _Array = np.asarray(ndtr(w / (sigma_k[np.newaxis] + 1e-12)))  # (N, K)
    coeffs_arr = np.asarray(coeffs, dtype=np.float64)
    if coeffs_arr.ndim == 1:
        coeffs_arr = coeffs_arr.reshape(1, 3)
    K = u.shape[1]
    n_terms = coeffs_arr.shape[0]
    if 4 * n_terms != K:
        raise ValueError(
            f"K={K} must equal 4 * n_terms={4 * n_terms} "
            f"(coeffs has {n_terms} rows)."
        )
    score: _Array = np.zeros(u.shape[0], dtype=np.float64)
    for t_idx in range(n_terms):
        offset = 4 * t_idx
        score = score + _olsen_term(
            u,
            (offset, offset + 1, offset + 2, offset + 3),
            (float(coeffs_arr[t_idx, 0]), float(coeffs_arr[t_idx, 1]), float(coeffs_arr[t_idx, 2])),
        )
    return score


def _spatial_olsen_score(
    x: _Array,
    signal_joints: list[int],
    sigma_j: _Array,
    coeffs: _Array,
) -> _Array:
    """Spatial variant of the Olsen score over 4 designated signal joints.

    Per-joint grand means (averaged over F and T) are standardised via the
    Gaussian CDF and combined with a single Olsen interaction term.  Only the
    4 signal joints enter the label.

    Args:
        x: ``(N, J, F, T)`` float array.
        signal_joints: Exactly 4 distinct joint indices in ``[0, J)``.
        sigma_j: ``(J,)`` per-joint standard deviations for normalisation.
        coeffs: ``(3,)`` coefficients ``[c1, c2, c3]`` for the Olsen term.

    Returns:
        ``(N,)`` float64 score array.

    Raises:
        ValueError: if ``signal_joints`` does not contain exactly 4 indices.
    """
    if len(signal_joints) != 4:
        raise ValueError(
            f"spatial_olsen_score requires exactly 4 signal joints; "
            f"got {len(signal_joints)}."
        )
    w = x.mean(axis=(2, 3))  # (N, J)
    u: _Array = np.asarray(ndtr(w / (sigma_j[np.newaxis] + 1e-12)))  # (N, J)
    coeffs_flat = np.asarray(coeffs, dtype=np.float64).reshape(-1)
    a, b, c, d = signal_joints
    return _olsen_term(
        u,
        (a, b, c, d),
        (float(coeffs_flat[0]), float(coeffs_flat[1]), float(coeffs_flat[2])),
    )


def _default_reduce(arr: _Array) -> _Array:
    """Reduce a ``(N, ...)`` array to ``(N,)`` by taking the grand mean.

    Args:
        arr: ``(N, ...)`` array with arbitrary trailing dimensions.

    Returns:
        ``(N,)`` float64 mean array.
    """
    result: _Array = arr.reshape(arr.shape[0], -1).mean(axis=1)
    return result


def _make_window_assignments(T: int, K: int) -> list[list[int]]:
    """Build K equal-width window assignments for T frames.

    Args:
        T: Total number of frames.
        K: Number of windows.

    Returns:
        List of K lists, each containing the frame indices for that window.
    """
    quarter = T // K
    return [
        list(range(k * quarter, (k + 1) * quarter if k < K - 1 else T))
        for k in range(K)
    ]


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class LabelFunction(ABC):
    """Abstract base class for all synthetic label functions.

    Concrete implementations must:

    * Accept ``x: np.ndarray`` of shape ``(N, J, F, T)`` and return
      ``(N,)`` int64 class labels in ``{0, ..., n_classes - 1}``.
    * Implement :py:meth:`important_players` returning the set of player
      indices whose coordinates causally determine the label.

    Labels are produced by binarising a continuous score via quantile cutoffs.
    The default is ternary (3 classes) with 33/67 percentile splits.

    Args:
        n_classes: Number of output classes.  Must be >= 2.
        percentiles: ``(n_classes - 1,)``-tuple of percentile cutoffs used to
            split the continuous score into integer labels.  Must have exactly
            ``n_classes - 1`` entries in strictly increasing order.

    Raises:
        ValueError: if ``n_classes < 2`` or ``len(percentiles) != n_classes - 1``.
    """

    def __init__(
        self,
        n_classes: int = 3,
        percentiles: tuple[float, ...] = (33.0, 67.0),
    ) -> None:
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2; got {n_classes}.")
        if len(percentiles) != n_classes - 1:
            raise ValueError(
                f"len(percentiles)={len(percentiles)} must equal "
                f"n_classes-1={n_classes - 1}."
            )
        self.n_classes = n_classes
        self.percentiles = percentiles

    @abstractmethod
    def __call__(self, x: _Array) -> npt.NDArray[np.int64]:
        """Apply the label function to a batch of sequences.

        Args:
            x: ``(N, J, F, T)`` float array.

        Returns:
            ``(N,)`` int64 array with values in ``{0, ..., n_classes - 1}``.
        """

    @abstractmethod
    def important_players(self, player_set: PlayerSet) -> set[int]:
        """Return player indices that causally drive this label.

        The returned indices are used as ground-truth by the ``TopKRecovery``
        metric and related attribution evaluators.

        Args:
            player_set: The :class:`~motionbench.players.base.PlayerSet`
                defining the M players being evaluated.

        Returns:
            Subset of ``{0, ..., player_set.n_players - 1}`` whose coordinates
            causally determine the label.
        """

    def _binarize(self, score: _Array) -> npt.NDArray[np.int64]:
        """Quantile-split a continuous score into integer class labels.

        Applies the stored ``percentiles`` to compute quantile cutoffs from the
        score distribution, then assigns each sample to the corresponding class.

        Args:
            score: ``(N,)`` float array of continuous response values.

        Returns:
            ``(N,)`` int64 array with values in ``{0, ..., n_classes - 1}``.
        """
        cutoffs = np.percentile(score, list(self.percentiles))
        labels = np.zeros(len(score), dtype=np.int64)
        for i, qi in enumerate(cutoffs, start=1):
            labels[score >= qi] = i
        result: npt.NDArray[np.int64] = labels
        return result


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class Linear(LabelFunction):
    """Linear label function: weighted sum of flattened per-coordinate values.

    The continuous score is::

        score_n = sum_i w_i · x[n].flatten()[i]

    i.e. a dot product of flattened ``(J*F*T,)`` weights with each sample
    after flattening.

    Important players are those whose coordinate range has at least one
    non-zero weight.  The coordinate range for each player is determined by
    calling :py:meth:`~motionbench.players.base.PlayerSet.coalition_mask` with
    a one-hot coalition vector.

    Args:
        weights: ``(J*F*T,)`` weight vector.  The shape must match the
            flattened per-sample shape of the data passed to ``__call__``.
        n_classes: Number of output classes.
        percentiles: Quantile cutoffs for binarisation.

    Raises:
        ValueError: if ``weights`` is not 1-D.
    """

    def __init__(
        self,
        weights: _Array,
        n_classes: int = 3,
        percentiles: tuple[float, ...] = (33.0, 67.0),
    ) -> None:
        super().__init__(n_classes=n_classes, percentiles=percentiles)
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1:
            raise ValueError(
                f"weights must be 1-D; got shape {w.shape}. "
                "Pass weights.flatten() if needed."
            )
        self.weights: npt.NDArray[np.float64] = w

    def __call__(self, x: _Array) -> npt.NDArray[np.int64]:
        """Apply the linear label function.

        Args:
            x: ``(N, J, F, T)`` float array.

        Returns:
            ``(N,)`` int64 class labels.
        """
        N = x.shape[0]
        x_flat = x.reshape(N, -1).astype(np.float64)  # (N, J*F*T)
        score: _Array = x_flat @ self.weights  # (N,)
        return self._binarize(score)

    def important_players(self, player_set: PlayerSet) -> set[int]:
        """Return players that have at least one non-zero weight coordinate.

        Args:
            player_set: The PlayerSet defining M players over ``(J, F, T)``.

        Returns:
            Subset of ``{0, ..., M-1}`` with non-zero weight contribution.
        """
        M = player_set.n_players
        result: set[int] = set()
        for p in range(M):
            z = torch.zeros(M, dtype=torch.int32)
            z[p] = 1
            mask: npt.NDArray[np.bool_] = player_set.coalition_mask(z).numpy()
            mask_flat = mask.flatten()
            if mask_flat.shape[0] == self.weights.shape[0] and np.abs(self.weights[mask_flat]).sum() > 0.0:
                result.add(p)
        return result


class OlsenInteraction(LabelFunction):
    """Nonlinear K-window interaction label (Olsen et al. JMLR 2022 Eq. 12).

    Generalised to arbitrary K divisible by 4 by tiling one Olsen term per
    group of 4 consecutive windows.  Per-window grand means are standardised
    to approximate uniform marginals via the Gaussian CDF.

    **Calibration:** ``sigma_k`` (per-window standard deviations) and the
    interaction coefficients are fitted lazily on the first call to
    :py:meth:`__call__` using the provided data batch.  Subsequent calls
    reuse the fitted parameters.

    **Important players:** All K windows enter the score (each group of 4
    contributes one Olsen term), so the full set ``{0, ..., K-1}`` is returned.

    Args:
        K: Number of temporal windows.  Must be a positive multiple of 4.
        n_classes: Number of output classes.
        percentiles: Quantile cutoffs for binarisation.
        seed: Random seed for sampling Olsen interaction coefficients.

    Raises:
        ValueError: if ``K`` is not a positive multiple of 4.
    """

    def __init__(
        self,
        K: int,
        n_classes: int = 3,
        percentiles: tuple[float, ...] = (33.0, 67.0),
        seed: int = 0,
    ) -> None:
        super().__init__(n_classes=n_classes, percentiles=percentiles)
        if K <= 0 or K % 4 != 0:
            raise ValueError(f"K must be a positive multiple of 4; got {K}.")
        self.K = K
        self.seed = seed
        self._sigma_k: npt.NDArray[np.float64] | None = None
        self._coeffs: npt.NDArray[np.float64] | None = None
        self._window_assignments: list[list[int]] | None = None

    def _fit(self, x: _Array) -> None:
        """Calibrate sigma_k and sample coefficients from the first data batch.

        Args:
            x: ``(N, J, F, T)`` float array used for calibration.
        """
        T = x.shape[3]
        self._window_assignments = _make_window_assignments(T, self.K)
        w_cal = _per_window_grand_means(x, self._window_assignments)  # (N, K)
        self._sigma_k = np.asarray(w_cal.std(axis=0).clip(min=1e-8), dtype=np.float64)

        rng = np.random.default_rng(self.seed)
        n_terms = self.K // 4
        self._coeffs = np.array(
            [
                [
                    float(rng.uniform(0.5, 2.0)),
                    float(rng.uniform(0.5, 2.0)),
                    float(rng.uniform(0.5, 1.0)),
                ]
                for _ in range(n_terms)
            ],
            dtype=np.float64,
        )

    def __call__(self, x: _Array) -> npt.NDArray[np.int64]:
        """Apply the Olsen interaction label function.

        Calibrates on the first call; reuses fitted parameters thereafter.

        Args:
            x: ``(N, J, F, T)`` float array.

        Returns:
            ``(N,)`` int64 class labels.
        """
        if self._sigma_k is None:
            self._fit(x)
        assert self._window_assignments is not None
        assert self._sigma_k is not None
        assert self._coeffs is not None
        score = _nonlinear_olsen_score(
            x, self._window_assignments, self._sigma_k, self._coeffs
        )
        return self._binarize(score)

    def important_players(self, player_set: PlayerSet) -> set[int]:
        """Return all K window indices as important players.

        All K windows contribute to the score via the tiled Olsen terms,
        so no window is irrelevant.

        Args:
            player_set: The PlayerSet defining M players.

        Returns:
            ``{0, 1, ..., K-1}`` (all temporal windows are important).
        """
        return set(range(self.K))


class SpatialOlsen(LabelFunction):
    """Spatial Olsen score over 4 designated signal joints.

    Per-joint grand means (averaged over F and T) are standardised via the
    Gaussian CDF and combined with a single Olsen interaction term.  The
    remaining J-4 joints are nuisance — they do not enter the label
    (although they may be correlated with signal joints).

    **Calibration:** ``sigma_j`` and interaction coefficients are fitted
    lazily on the first call to :py:meth:`__call__`.

    **Important players:** The 4 ``signal_joints`` indices.

    Args:
        signal_joints: Exactly 4 distinct joint indices that drive the label.
        n_classes: Number of output classes.
        percentiles: Quantile cutoffs for binarisation.
        seed: Random seed for sampling Olsen interaction coefficients.

    Raises:
        ValueError: if ``signal_joints`` does not contain exactly 4 indices.
    """

    def __init__(
        self,
        signal_joints: list[int],
        n_classes: int = 3,
        percentiles: tuple[float, ...] = (33.0, 67.0),
        seed: int = 0,
    ) -> None:
        super().__init__(n_classes=n_classes, percentiles=percentiles)
        if len(signal_joints) != 4:
            raise ValueError(
                f"signal_joints must contain exactly 4 indices; "
                f"got {len(signal_joints)}."
            )
        if len(set(signal_joints)) != 4:
            raise ValueError("signal_joints must contain 4 distinct indices.")
        self.signal_joints = list(signal_joints)
        self.seed = seed
        self._sigma_j: npt.NDArray[np.float64] | None = None
        self._coeffs: npt.NDArray[np.float64] | None = None

    def _fit(self, x: _Array) -> None:
        """Calibrate sigma_j and sample coefficients from the first data batch.

        Args:
            x: ``(N, J, F, T)`` float array used for calibration.
        """
        w_cal = x.mean(axis=(2, 3))  # (N, J)
        self._sigma_j = np.asarray(w_cal.std(axis=0).clip(min=1e-8), dtype=np.float64)

        rng = np.random.default_rng(self.seed)
        self._coeffs = np.array(
            [
                float(rng.uniform(0.5, 2.0)),
                float(rng.uniform(0.5, 2.0)),
                float(rng.uniform(0.5, 1.0)),
            ],
            dtype=np.float64,
        )

    def __call__(self, x: _Array) -> npt.NDArray[np.int64]:
        """Apply the spatial Olsen label function.

        Calibrates on the first call; reuses fitted parameters thereafter.

        Args:
            x: ``(N, J, F, T)`` float array.

        Returns:
            ``(N,)`` int64 class labels.
        """
        if self._sigma_j is None:
            self._fit(x)
        assert self._sigma_j is not None
        assert self._coeffs is not None
        score = _spatial_olsen_score(
            x, self.signal_joints, self._sigma_j, self._coeffs
        )
        return self._binarize(score)

    def important_players(self, player_set: PlayerSet) -> set[int]:
        """Return the 4 signal joint indices as important players.

        Args:
            player_set: The PlayerSet defining M players.

        Returns:
            The set of signal joint indices (always a subset of ``{0, ..., J-1}``).
        """
        return set(self.signal_joints)


class LocalizedTemporal(LabelFunction):
    """Label function that depends only on a single temporal window.

    The continuous score is computed by applying ``fn`` to the slice of ``x``
    corresponding to ``window_idx`` out of ``K`` equal-width windows.

    Args:
        window_idx: Index of the temporal window that drives the label.
            Must be in ``[0, K)``.
        K: Total number of equal-width windows (default: 4).
        fn: Reduction function ``(N, J, F, window_size) → (N,)`` applied to
            the window slice.  Defaults to grand mean over all non-batch axes.
        n_classes: Number of output classes.
        percentiles: Quantile cutoffs for binarisation.

    Raises:
        ValueError: if ``window_idx >= K``.
    """

    def __init__(
        self,
        window_idx: int,
        K: int = 4,
        fn: Callable[[_Array], _Array] | None = None,
        n_classes: int = 3,
        percentiles: tuple[float, ...] = (33.0, 67.0),
    ) -> None:
        super().__init__(n_classes=n_classes, percentiles=percentiles)
        if window_idx < 0 or window_idx >= K:
            raise ValueError(
                f"window_idx={window_idx} out of range [0, K={K})."
            )
        self.window_idx = window_idx
        self.K = K
        self._fn: Callable[[_Array], _Array] = fn if fn is not None else _default_reduce

    def __call__(self, x: _Array) -> npt.NDArray[np.int64]:
        """Apply the localised temporal label function.

        Args:
            x: ``(N, J, F, T)`` float array.

        Returns:
            ``(N,)`` int64 class labels driven by window ``window_idx``.
        """
        T = x.shape[3]
        assignments = _make_window_assignments(T, self.K)
        frames = assignments[self.window_idx]
        x_slice = x[:, :, :, frames]  # (N, J, F, window_size)
        score: _Array = self._fn(x_slice).reshape(x.shape[0])
        return self._binarize(score)

    def important_players(self, player_set: PlayerSet) -> set[int]:
        """Return the single temporal window index as the important player.

        Args:
            player_set: The PlayerSet defining M players.

        Returns:
            ``{window_idx}``.
        """
        return {self.window_idx}


class LocalizedSpatial(LabelFunction):
    """Label function that depends only on a single spatial joint.

    The continuous score is computed by applying ``fn`` to the slice of ``x``
    corresponding to ``joint_idx``.

    Args:
        joint_idx: Index of the joint that drives the label.
        fn: Reduction function ``(N, F, T) → (N,)`` applied to the joint slice.
            Defaults to grand mean over all non-batch axes.
        n_classes: Number of output classes.
        percentiles: Quantile cutoffs for binarisation.
    """

    def __init__(
        self,
        joint_idx: int,
        fn: Callable[[_Array], _Array] | None = None,
        n_classes: int = 3,
        percentiles: tuple[float, ...] = (33.0, 67.0),
    ) -> None:
        super().__init__(n_classes=n_classes, percentiles=percentiles)
        if joint_idx < 0:
            raise ValueError(f"joint_idx must be >= 0; got {joint_idx}.")
        self.joint_idx = joint_idx
        self._fn: Callable[[_Array], _Array] = fn if fn is not None else _default_reduce

    def __call__(self, x: _Array) -> npt.NDArray[np.int64]:
        """Apply the localised spatial label function.

        Args:
            x: ``(N, J, F, T)`` float array.

        Returns:
            ``(N,)`` int64 class labels driven by joint ``joint_idx``.
        """
        x_slice = x[:, self.joint_idx, :, :]  # (N, F, T)
        score: _Array = self._fn(x_slice).reshape(x.shape[0])
        return self._binarize(score)

    def important_players(self, player_set: PlayerSet) -> set[int]:
        """Return the single joint index as the important player.

        Args:
            player_set: The PlayerSet defining M players.

        Returns:
            ``{joint_idx}``.
        """
        return {self.joint_idx}


class LocalizedSpatiotemporal(LabelFunction):
    """Label function that depends only on a single (joint, window) cell.

    The continuous score is computed by applying ``fn`` to the slice of ``x``
    indexed by ``joint_idx`` and the frames belonging to ``window_idx`` out of
    ``K`` equal-width windows.

    For ``JointWindowCells`` player sets, both the joint and the window are
    distinct player axes; for other player sets (e.g. ``TemporalWindows`` or
    ``SpatialJoints``), :py:meth:`important_players` returns
    ``{joint_idx, window_idx}`` as a conservative over-approximation.

    Args:
        joint_idx: Index of the joint that drives the label.
        window_idx: Index of the temporal window that drives the label.
            Must be in ``[0, K)``.
        K: Total number of equal-width windows (default: 4).
        fn: Reduction function ``(N, F, window_size) → (N,)`` applied to the
            cell slice.  Defaults to grand mean over all non-batch axes.
        n_classes: Number of output classes.
        percentiles: Quantile cutoffs for binarisation.

    Raises:
        ValueError: if ``window_idx >= K``.
    """

    def __init__(
        self,
        joint_idx: int,
        window_idx: int,
        K: int = 4,
        fn: Callable[[_Array], _Array] | None = None,
        n_classes: int = 3,
        percentiles: tuple[float, ...] = (33.0, 67.0),
    ) -> None:
        super().__init__(n_classes=n_classes, percentiles=percentiles)
        if joint_idx < 0:
            raise ValueError(f"joint_idx must be >= 0; got {joint_idx}.")
        if window_idx < 0 or window_idx >= K:
            raise ValueError(
                f"window_idx={window_idx} out of range [0, K={K})."
            )
        self.joint_idx = joint_idx
        self.window_idx = window_idx
        self.K = K
        self._fn: Callable[[_Array], _Array] = fn if fn is not None else _default_reduce

    def __call__(self, x: _Array) -> npt.NDArray[np.int64]:
        """Apply the localised spatiotemporal label function.

        Args:
            x: ``(N, J, F, T)`` float array.

        Returns:
            ``(N,)`` int64 class labels driven by joint ``joint_idx``
            in window ``window_idx``.
        """
        T = x.shape[3]
        assignments = _make_window_assignments(T, self.K)
        frames = assignments[self.window_idx]
        x_slice = x[:, self.joint_idx, :, :][:, :, frames]  # (N, F, window_size)
        score: _Array = self._fn(x_slice).reshape(x.shape[0])
        return self._binarize(score)

    def important_players(self, player_set: PlayerSet) -> set[int]:
        """Return the joint and window indices as important players.

        For most player sets, returns ``{joint_idx, window_idx}``.  For
        ``JointWindowCells`` player sets the caller should compute the cell
        index ``joint_idx * K + window_idx`` directly.

        Args:
            player_set: The PlayerSet defining M players.

        Returns:
            ``{joint_idx, window_idx}``.
        """
        return {self.joint_idx, self.window_idx}
