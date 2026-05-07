"""motionbench.attribution.group_segment_shap — GroupSegmentSHAP attributor.

Computes exact Shapley values at the player level by enumerating all ``2^M``
coalitions of M players, estimating a scalar value function for each coalition
via the provided :class:`~motionbench.imputers.base.BaseImputer`, and then
applying the closed-form weighted-sum Shapley formula.

This is equivalent to ``direct_group_shapley`` from Jullum, Redelmeier & Aas
(2021) applied to a game where each "group" (player) is a set of
``(J, F, T)`` coordinates defined by the
:class:`~motionbench.players.base.PlayerSet`.  The implementation is a
self-contained port of the CARE-PD ``group_baselines.direct_group_shapley``
and ``_shapley_from_v`` functions — it does not import CARE-PD directly.

Complexity
----------
Exact enumeration over 2^M coalitions. For M ≤ 20 this is tractable (< 10^6
coalitions). For larger M, a sampling-based approximation (e.g. KernelSHAP)
should be used instead; sampling-based extensions are left to future work.

References
----------
Jullum, Redelmeier & Aas (2021) "Groupwise Shapley Feature Importance Values."
*Computational Statistics & Data Analysis* 162, 107.
Owen (1977) "Values of Games with a Priori Unions." *Essays in Mathematical
Economics and Game Theory*, pp. 76–88.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from motionbench.attribution.base import BaseAttributor
from motionbench.imputers.base import BaseImputer

if TYPE_CHECKING:
    from motionbench.players.base import PlayerSet


__all__ = ["GroupSegmentSHAPAttributor"]

_MAX_EXACT_M: int = 20  # 2^20 = ~1 M coalitions; beyond this raise.


# ---------------------------------------------------------------------------
# Core Shapley computation (ported from CARE-PD group_baselines.py)
# ---------------------------------------------------------------------------


def _bit_indices(M: int) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Precompute coalition sizes and log-factorial table for M players.

    Args:
        M: Number of players.

    Returns:
        Tuple ``(sizes, log_fact)`` where ``sizes[i]`` is the popcount of
        bitmask ``i`` (length ``2^M``) and ``log_fact`` is a ``(M+2,)``
        array of cumulative log-factorials from ``0!`` to ``(M+1)!``.
    """
    N = 1 << M
    sizes = np.array([bin(i).count("1") for i in range(N)], dtype=np.int64)
    log_fact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, M + 2)))])
    return sizes, log_fact


def _shapley_from_v(v: npt.NDArray[np.float64], M: int) -> npt.NDArray[np.float64]:
    """Vectorised exact Shapley formula from a complete value-function table.

    Args:
        v: ``(2^M,)`` float64 array where ``v[bitmask]`` is the value of the
            coalition encoded by ``bitmask``. Must satisfy ``v[0] == 0``
            (empty coalition has zero value).
        M: Number of players.

    Returns:
        ``(M,)`` float64 array of per-player Shapley values.
    """
    sizes, log_fact = _bit_indices(M)
    N = 1 << M
    phi = np.zeros(M, dtype=np.float64)
    for i in range(M):
        bit = 1 << i
        # Indices of coalitions not containing player i.
        mask_without_i = (np.arange(N, dtype=np.int64) & bit) == 0
        ids_s = np.flatnonzero(mask_without_i)
        ids_si = ids_s | bit
        s_sizes = sizes[ids_s]
        log_w = log_fact[s_sizes] + log_fact[M - s_sizes - 1] - log_fact[M]
        weights = np.exp(log_w)
        phi[i] = float(np.sum(weights * (v[ids_si] - v[ids_s])))
    return phi


# ---------------------------------------------------------------------------
# Attributor class
# ---------------------------------------------------------------------------


class GroupSegmentSHAPAttributor(BaseAttributor):
    """Exact group Shapley attributor for motion-sequence player sets.

    For each of the ``2^M`` coalitions of M players:

    1. Expand the binary coalition vector to an element-level mask via
       :meth:`~motionbench.players.base.PlayerSet.coalition_mask`.
    2. Impute the hidden coordinates using the provided
       :class:`~motionbench.imputers.base.BaseImputer`.
    3. Run the classifier on the imputed sequence to obtain a scalar value.

    After collecting all ``2^M`` values, the closed-form weighted-sum Shapley
    formula is applied to produce the ``(M,)`` attribution vector.

    Args:
        classifier: ``(B, J, F, T) float32 → (B,) float32`` callable.
        imputer: Fitted :class:`~motionbench.imputers.base.BaseImputer` used
            to fill in hidden player coordinates.
        n_coalitions: Controls the number of imputer draws used to estimate
            each coalition's value via Monte Carlo averaging. Each coalition
            value is estimated as the mean over ``n_coalitions`` independent
            imputation samples. Defaults to 256.
        seed: Optional random seed for reproducibility. Defaults to ``None``.

    Raises:
        ValueError: if ``players.n_players > _MAX_EXACT_M`` (20), as exact
            enumeration would require more than ``2^20`` classifier calls.
    """

    def __init__(
        self,
        classifier: Callable[[Tensor], Tensor],
        imputer: BaseImputer,
        n_coalitions: int = 256,
        seed: int | None = None,
    ) -> None:
        super().__init__(classifier)
        self._imputer = imputer
        self._n_coalitions = n_coalitions
        self._seed = seed

    # ------------------------------------------------------------------
    # BaseAttributor interface
    # ------------------------------------------------------------------

    def attribute(
        self,
        x: Tensor,
        players: "PlayerSet",
        target: int = 0,
    ) -> Tensor:
        """Compute exact group Shapley values at the player level.

        Args:
            x: ``(J, F, T)`` float32 input sequence.
            players: :class:`~motionbench.players.base.PlayerSet` with M players.
            target: Class index for attribution. Selects output column when the
                classifier returns ``(B, n_classes)``; ignored for scalar output.

        Returns:
            ``(M,)`` float32 Tensor of per-player Shapley values.

        Raises:
            ValueError: if ``M > 20`` (exact enumeration intractable).
        """
        M = players.n_players
        if M > _MAX_EXACT_M:
            raise ValueError(
                f"GroupSegmentSHAPAttributor requires M ≤ {_MAX_EXACT_M} for "
                f"exact enumeration, but players.n_players = {M}. "
                "Use KernelSHAP (TimeSHAPAttributor) for large M."
            )

        n_enum = 1 << M  # 2^M coalition bitmasks
        v = np.zeros(n_enum, dtype=np.float64)

        # Each coalition value is the mean classifier output over n_coalitions
        # independent imputation samples drawn for that coalition's mask.
        for bitmask in range(n_enum):
            z_bits = [(bitmask >> k) & 1 for k in range(M)]
            z = torch.tensor(z_bits, dtype=torch.bool)
            mask = players.coalition_mask(z)

            seed_k: int | None = (
                None if self._seed is None else int(self._seed + bitmask)
            )
            # Draw n_coalitions imputed samples for this mask.
            x_imp_batch = self._imputer.impute(
                x, mask, n_samples=self._n_coalitions, seed=seed_k
            )  # (n_coalitions, J, F, T)

            with torch.no_grad():
                preds = self._classifier(x_imp_batch)  # (n_coalitions,) or (n_coalitions, C)

            if preds.ndim > 1:
                preds = preds[:, target]
            v[bitmask] = float(preds.mean().item())

        # v[0] must be the empty-coalition value (used as reference).
        # Subtract it so v[0] == 0, then add it back via the SHAP constant.
        # (The Shapley formula already handles a non-zero v[0] correctly since
        # it only uses differences v[S ∪ {i}] - v[S], so no adjustment needed.)
        phi = _shapley_from_v(v, M)
        return torch.tensor(phi, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Short identifier for logging."""
        return "GroupSegmentSHAP"

    @property
    def requires_imputer(self) -> bool:
        """GroupSegmentSHAP requires a fitted imputer."""
        return True


# ---------------------------------------------------------------------------
# Module-level math helpers (also useful for testing)
# ---------------------------------------------------------------------------

def shapley_from_value_table(
    v: npt.NDArray[np.float64], M: int
) -> npt.NDArray[np.float64]:
    """Public alias for :func:`_shapley_from_v` for external use and testing.

    Args:
        v: ``(2^M,)`` float64 value-function table indexed by bitmask.
        M: Number of players.

    Returns:
        ``(M,)`` float64 Shapley values.
    """
    return _shapley_from_v(v, M)


def direct_group_shapley(
    v_feature: npt.NDArray[np.float64],
    M: int,
    group_assignments: list[list[int]],
) -> npt.NDArray[np.float64]:
    """Shapley on the contracted K-group game (Jullum et al. 2021).

    Contracts the M-player game to a K-player game where each player
    corresponds to one group, then applies the exact Shapley formula.

    Args:
        v_feature: ``(2^M,)`` float64 value-function table over M features,
            indexed by bitmask.
        M: Total number of individual features.
        group_assignments: List of K disjoint lists, each containing the
            feature indices that belong to that group.  Must cover ``[M]``.

    Returns:
        ``(K,)`` float64 array of per-group Shapley values.
    """
    K = len(group_assignments)
    v_group = np.zeros(1 << K, dtype=np.float64)
    for t_bits in range(1 << K):
        feat_bits = 0
        for k_idx, g_feats in enumerate(group_assignments):
            if (t_bits >> k_idx) & 1:
                for f in g_feats:
                    feat_bits |= 1 << f
        v_group[t_bits] = v_feature[feat_bits]
    return _shapley_from_v(v_group, K)


def _log_factorial_weight(s: int, M: int) -> float:
    """Shapley weight ``s! * (M-s-1)! / M!`` for coalition size ``s``.

    Args:
        s: Size of coalition ``S`` (not including player ``i``).
        M: Total number of players.

    Returns:
        Scalar Shapley weight.
    """
    return math.factorial(s) * math.factorial(M - s - 1) / math.factorial(M)
