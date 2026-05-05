"""motionbench.metrics.ground_truth — Ground-truth attribution quality metrics.

These metrics compare a per-player attribution vector ``φ ∈ ℝ^M`` against the
oracle's true Shapley values ``φ_oracle ∈ ℝ^M``.  All metrics require an
:class:`~motionbench.oracles.base.Oracle` to be passed to ``evaluate``.

Metric definitions
------------------
* **EC1** — Mean absolute error: ``mean |φ_m − φ_oracle_m|``.
* **EC2** — Mean squared error: ``mean (φ_m − φ_oracle_m)²``.
* **EC3** — ``1 − Pearson(φ, φ_oracle)``.  Range ``[0, 2]``; 0 = perfect.
* **EC1_norm** — EC1 normalised by ``mean |φ_oracle_m|``.  Equals 1 when
  ``φ = 0``.
* **TopKRecovery** — Fraction of the true top-k players (by absolute oracle
  value) that appear in the top-k players of ``φ``.  ``k`` defaults to
  ``ceil(M/2)``.
* **SpearmanRankMetric** — Spearman rank correlation between ``φ`` and
  ``φ_oracle``.
* **KendallRankMetric** — Kendall tau-b rank correlation.
* **EfficiencyErrorMetric** — ``|Σφ − (v(N) − v(∅))| / |v(N) − v(∅)|``.
  Should be ``< 1e-3`` for KernelSHAP with an oracle imputer.

Design decisions
----------------
* All oracle calls use ``oracle.true_shapley(x, classifier, players)``.
  ``n_mc`` and ``seed`` use their defaults (1000, None) in ``evaluate``
  unless overridden at metric construction time.
* ``EfficiencyErrorMetric`` computes ``v(N) − v(∅)`` via
  ``oracle.true_shapley(...).sum()``, which satisfies the Shapley efficiency
  axiom by construction (avoiding independent MC re-estimation).
* ``TopKRecovery`` also returns ``top1``: whether the single most-important
  oracle player was recovered.

References
----------
Lundberg & Lee (2017) "A unified approach to interpreting model predictions."
Aas, Jullum & Løland (2021) §3 — conditional expectation Shapley game.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

from scipy.stats import kendalltau, spearmanr

from motionbench.metrics.base import BaseMetric

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from motionbench.imputers.base import BaseImputer
    from motionbench.oracles.base import Oracle
    from motionbench.players.base import PlayerSet


__all__ = [
    "EC1Metric",
    "EC2Metric",
    "EC3Metric",
    "TopKRecovery",
    "SpearmanRankMetric",
    "KendallRankMetric",
    "EfficiencyErrorMetric",
]


# ---------------------------------------------------------------------------
# EC1 — Mean absolute error
# ---------------------------------------------------------------------------


class EC1Metric(BaseMetric):
    """Mean absolute error vs oracle: ``mean |φ_m − φ_oracle_m|``.

    Also returns ``ec1_norm``: EC1 normalised by ``mean |φ_oracle_m|``,
    clamped so that division by zero returns ``nan``.

    Args:
        n_mc: Monte Carlo samples for ``oracle.true_shapley``.
        oracle_seed: Random seed forwarded to the oracle.
    """

    requires_oracle: ClassVar[bool] = True

    def __init__(self, n_mc: int = 1000, oracle_seed: int | None = None) -> None:
        self._n_mc = n_mc
        self._oracle_seed = oracle_seed

    def evaluate(
        self,
        phi: Tensor,
        x: Tensor,
        classifier: Callable[[Tensor], Tensor],
        players: PlayerSet,
        target: int = 0,
        oracle: Oracle | None = None,
        imputer: BaseImputer | None = None,
    ) -> dict[str, float]:
        """Compute EC1 and EC1_norm.

        Args:
            phi: ``(M,)`` float32 attribution vector.
            x: ``(J, F, T)`` float32 input sequence.
            classifier: ``(B, J, F, T) → (B,)`` callable.
            players: PlayerSet defining the M players.
            target: Unused (kept for interface compatibility).
            oracle: Ground-truth oracle.  Must not be None.
            imputer: Unused.

        Returns:
            ``{"ec1": float, "ec1_norm": float}``.

        Raises:
            ValueError: if oracle is None.
        """
        self._check_deps(oracle, imputer)
        assert oracle is not None  # mypy narrowing
        phi_oracle = oracle.true_shapley(
            x, classifier, players, n_mc=self._n_mc, seed=self._oracle_seed
        )
        phi = phi.float()
        phi_oracle = phi_oracle.float()
        ec1 = float((phi - phi_oracle).abs().mean().item())
        denom = float(phi_oracle.abs().mean().item())
        ec1_norm = ec1 / denom if denom > 0.0 else float("nan")
        return {"ec1": ec1, "ec1_norm": ec1_norm}


# ---------------------------------------------------------------------------
# EC2 — Mean squared error
# ---------------------------------------------------------------------------


class EC2Metric(BaseMetric):
    """MSE vs oracle: ``mean (φ_m − φ_oracle_m)²``.

    Args:
        n_mc: Monte Carlo samples for ``oracle.true_shapley``.
        oracle_seed: Random seed forwarded to the oracle.
    """

    requires_oracle: ClassVar[bool] = True

    def __init__(self, n_mc: int = 1000, oracle_seed: int | None = None) -> None:
        self._n_mc = n_mc
        self._oracle_seed = oracle_seed

    def evaluate(
        self,
        phi: Tensor,
        x: Tensor,
        classifier: Callable[[Tensor], Tensor],
        players: PlayerSet,
        target: int = 0,
        oracle: Oracle | None = None,
        imputer: BaseImputer | None = None,
    ) -> dict[str, float]:
        """Compute EC2 (MSE).

        Args:
            phi: ``(M,)`` float32 attribution vector.
            x: ``(J, F, T)`` float32 input sequence.
            classifier: ``(B, J, F, T) → (B,)`` callable.
            players: PlayerSet defining the M players.
            target: Unused.
            oracle: Ground-truth oracle.  Must not be None.
            imputer: Unused.

        Returns:
            ``{"ec2": float}``.

        Raises:
            ValueError: if oracle is None.
        """
        self._check_deps(oracle, imputer)
        assert oracle is not None
        phi_oracle = oracle.true_shapley(
            x, classifier, players, n_mc=self._n_mc, seed=self._oracle_seed
        )
        phi = phi.float()
        phi_oracle = phi_oracle.float()
        ec2 = float(((phi - phi_oracle) ** 2).mean().item())
        return {"ec2": ec2}


# ---------------------------------------------------------------------------
# EC3 — 1 − Pearson correlation
# ---------------------------------------------------------------------------


class EC3Metric(BaseMetric):
    """``1 − Pearson(φ, φ_oracle)``.  Range ``[0, 2]``; 0 = perfect match.

    If either vector is constant (std = 0), the metric is defined as 1.0
    (no linear information).

    Args:
        n_mc: Monte Carlo samples for ``oracle.true_shapley``.
        oracle_seed: Random seed forwarded to the oracle.
    """

    requires_oracle: ClassVar[bool] = True

    def __init__(self, n_mc: int = 1000, oracle_seed: int | None = None) -> None:
        self._n_mc = n_mc
        self._oracle_seed = oracle_seed

    def evaluate(
        self,
        phi: Tensor,
        x: Tensor,
        classifier: Callable[[Tensor], Tensor],
        players: PlayerSet,
        target: int = 0,
        oracle: Oracle | None = None,
        imputer: BaseImputer | None = None,
    ) -> dict[str, float]:
        """Compute EC3 = 1 − Pearson(φ, φ_oracle).

        Args:
            phi: ``(M,)`` float32 attribution vector.
            x: ``(J, F, T)`` float32 input sequence.
            classifier: ``(B, J, F, T) → (B,)`` callable.
            players: PlayerSet defining the M players.
            target: Unused.
            oracle: Ground-truth oracle.  Must not be None.
            imputer: Unused.

        Returns:
            ``{"ec3": float}`` where value ∈ ``[0, 2]``.

        Raises:
            ValueError: if oracle is None.
        """
        self._check_deps(oracle, imputer)
        assert oracle is not None
        phi_oracle = oracle.true_shapley(
            x, classifier, players, n_mc=self._n_mc, seed=self._oracle_seed
        )
        phi = phi.float()
        phi_oracle = phi_oracle.float()
        phi_c = phi - phi.mean()
        oracle_c = phi_oracle - phi_oracle.mean()
        std_phi = phi_c.norm()
        std_oracle = oracle_c.norm()
        if std_phi < 1e-12 or std_oracle < 1e-12:
            pearson = 0.0
        else:
            pearson = float((phi_c * oracle_c).sum() / (std_phi * std_oracle))
            pearson = max(-1.0, min(1.0, pearson))
        ec3 = 1.0 - pearson
        return {"ec3": ec3}


# ---------------------------------------------------------------------------
# TopKRecovery
# ---------------------------------------------------------------------------


class TopKRecovery(BaseMetric):
    """Fraction of true top-k oracle players recovered by ``φ``.

    ``k`` defaults to ``ceil(M / 2)``.  Also reports ``top1``: whether the
    single most-important oracle player is the argmax of ``|φ|``.

    Args:
        k: Number of top players to recover.  If None, defaults to
           ``ceil(M / 2)`` at evaluation time.
        n_mc: Monte Carlo samples for ``oracle.true_shapley``.
        oracle_seed: Random seed forwarded to the oracle.
    """

    requires_oracle: ClassVar[bool] = True

    def __init__(
        self,
        k: int | None = None,
        n_mc: int = 1000,
        oracle_seed: int | None = None,
    ) -> None:
        self._k = k
        self._n_mc = n_mc
        self._oracle_seed = oracle_seed

    def evaluate(
        self,
        phi: Tensor,
        x: Tensor,
        classifier: Callable[[Tensor], Tensor],
        players: PlayerSet,
        target: int = 0,
        oracle: Oracle | None = None,
        imputer: BaseImputer | None = None,
    ) -> dict[str, float]:
        """Compute top-k recovery and top-1 recovery.

        Args:
            phi: ``(M,)`` float32 attribution vector.
            x: ``(J, F, T)`` float32 input sequence.
            classifier: ``(B, J, F, T) → (B,)`` callable.
            players: PlayerSet defining the M players.
            target: Unused.
            oracle: Ground-truth oracle.  Must not be None.
            imputer: Unused.

        Returns:
            ``{"topk_overlap": float, "top1": float}`` where
            ``topk_overlap`` ∈ ``[0, 1]`` and ``top1`` ∈ ``{0.0, 1.0}``.

        Raises:
            ValueError: if oracle is None.
        """
        self._check_deps(oracle, imputer)
        assert oracle is not None
        phi_oracle = oracle.true_shapley(
            x, classifier, players, n_mc=self._n_mc, seed=self._oracle_seed
        )
        phi = phi.float()
        phi_oracle = phi_oracle.float()
        M = phi.shape[0]
        k = self._k if self._k is not None else math.ceil(M / 2)
        k = max(1, min(k, M))

        oracle_topk = set(phi_oracle.abs().topk(k).indices.tolist())
        phi_topk = set(phi.abs().topk(k).indices.tolist())
        topk_overlap = len(oracle_topk & phi_topk) / k

        oracle_top1 = int(phi_oracle.abs().argmax().item())
        phi_top1 = int(phi.abs().argmax().item())
        top1 = 1.0 if oracle_top1 == phi_top1 else 0.0

        return {"topk_overlap": topk_overlap, "top1": top1}


# ---------------------------------------------------------------------------
# SpearmanRankMetric
# ---------------------------------------------------------------------------


class SpearmanRankMetric(BaseMetric):
    """Spearman rank correlation between ``φ`` and oracle ``φ_oracle``.

    Returns 0.0 when either vector is constant (undefined correlation).

    Args:
        n_mc: Monte Carlo samples for ``oracle.true_shapley``.
        oracle_seed: Random seed forwarded to the oracle.
    """

    requires_oracle: ClassVar[bool] = True

    def __init__(self, n_mc: int = 1000, oracle_seed: int | None = None) -> None:
        self._n_mc = n_mc
        self._oracle_seed = oracle_seed

    def evaluate(
        self,
        phi: Tensor,
        x: Tensor,
        classifier: Callable[[Tensor], Tensor],
        players: PlayerSet,
        target: int = 0,
        oracle: Oracle | None = None,
        imputer: BaseImputer | None = None,
    ) -> dict[str, float]:
        """Compute Spearman rank correlation.

        Args:
            phi: ``(M,)`` float32 attribution vector.
            x: ``(J, F, T)`` float32 input sequence.
            classifier: ``(B, J, F, T) → (B,)`` callable.
            players: PlayerSet defining the M players.
            target: Unused.
            oracle: Ground-truth oracle.  Must not be None.
            imputer: Unused.

        Returns:
            ``{"spearman": float}`` where value ∈ ``[-1, 1]``.

        Raises:
            ValueError: if oracle is None.
        """
        self._check_deps(oracle, imputer)
        assert oracle is not None
        phi_oracle = oracle.true_shapley(
            x, classifier, players, n_mc=self._n_mc, seed=self._oracle_seed
        )
        a = phi.float().cpu().numpy()
        b = phi_oracle.float().cpu().numpy()
        result = spearmanr(a, b)
        corr = float(result.statistic)
        if math.isnan(corr):
            corr = 0.0
        return {"spearman": corr}


# ---------------------------------------------------------------------------
# KendallRankMetric
# ---------------------------------------------------------------------------


class KendallRankMetric(BaseMetric):
    """Kendall tau-b rank correlation between ``φ`` and oracle ``φ_oracle``.

    Returns 0.0 when either vector is constant (undefined correlation).

    Args:
        n_mc: Monte Carlo samples for ``oracle.true_shapley``.
        oracle_seed: Random seed forwarded to the oracle.
    """

    requires_oracle: ClassVar[bool] = True

    def __init__(self, n_mc: int = 1000, oracle_seed: int | None = None) -> None:
        self._n_mc = n_mc
        self._oracle_seed = oracle_seed

    def evaluate(
        self,
        phi: Tensor,
        x: Tensor,
        classifier: Callable[[Tensor], Tensor],
        players: PlayerSet,
        target: int = 0,
        oracle: Oracle | None = None,
        imputer: BaseImputer | None = None,
    ) -> dict[str, float]:
        """Compute Kendall tau-b rank correlation.

        Args:
            phi: ``(M,)`` float32 attribution vector.
            x: ``(J, F, T)`` float32 input sequence.
            classifier: ``(B, J, F, T) → (B,)`` callable.
            players: PlayerSet defining the M players.
            target: Unused.
            oracle: Ground-truth oracle.  Must not be None.
            imputer: Unused.

        Returns:
            ``{"kendall": float}`` where value ∈ ``[-1, 1]``.

        Raises:
            ValueError: if oracle is None.
        """
        self._check_deps(oracle, imputer)
        assert oracle is not None
        phi_oracle = oracle.true_shapley(
            x, classifier, players, n_mc=self._n_mc, seed=self._oracle_seed
        )
        a = phi.float().cpu().numpy()
        b = phi_oracle.float().cpu().numpy()
        tau, _ = kendalltau(a, b)
        if math.isnan(tau):
            tau = 0.0
        return {"kendall": float(tau)}


# ---------------------------------------------------------------------------
# EfficiencyErrorMetric
# ---------------------------------------------------------------------------


class EfficiencyErrorMetric(BaseMetric):
    """Efficiency axiom error: ``|Σφ − (v(N) − v(∅))| / |v(N) − v(∅)|``.

    ``v(N) − v(∅)`` is obtained from ``oracle.true_shapley(...).sum()``, which
    satisfies the Shapley efficiency axiom exactly by construction:

    .. math::

        \\sum_{m=1}^{M} \\phi^{\\text{oracle}}_m = v(N) - v(\\emptyset)

    This avoids independent Monte Carlo re-estimation of ``v(N)`` and
    ``v(∅)`` (which would accumulate variance), and instead leverages the
    oracle's exact Shapley computation.

    For KernelSHAP with an oracle imputer the efficiency axiom is enforced
    as a hard WLS constraint (``Σφ = v(N)_ks − v(∅)_ks`` exactly), so the
    error reflects only the discrepancy between the KernelSHAP MC estimates
    and the oracle's MC estimates of the same game.  When both are computed
    on the same input with large-scale oracle data the error is ``< 1e-3``.

    Args:
        n_mc: Monte Carlo samples forwarded to ``oracle.true_shapley``.
        oracle_seed: Random seed forwarded to the oracle.
    """

    requires_oracle: ClassVar[bool] = True

    def __init__(self, n_mc: int = 1000, oracle_seed: int | None = None) -> None:
        self._n_mc = n_mc
        self._oracle_seed = oracle_seed

    def evaluate(
        self,
        phi: Tensor,
        x: Tensor,
        classifier: Callable[[Tensor], Tensor],
        players: PlayerSet,
        target: int = 0,
        oracle: Oracle | None = None,
        imputer: BaseImputer | None = None,
    ) -> dict[str, float]:
        """Compute efficiency error relative to oracle Shapley sum.

        ``v(N) − v(∅)`` is computed as ``oracle.true_shapley(...).sum()``,
        which satisfies the efficiency axiom by construction.

        Args:
            phi: ``(M,)`` float32 attribution vector.
            x: ``(J, F, T)`` float32 input sequence.
            classifier: ``(B, J, F, T) → (B,)`` callable.
            players: PlayerSet defining the M players.
            target: Unused.
            oracle: Ground-truth oracle.  Must not be None.
            imputer: Unused.

        Returns:
            ``{"efficiency_error": float}`` — relative efficiency error.
            Should be ``< 1e-3`` for KernelSHAP with an oracle imputer
            on inputs with large ``|v(N) − v(∅)|``.

        Raises:
            ValueError: if oracle is None.
        """
        self._check_deps(oracle, imputer)
        assert oracle is not None

        # v(N) - v(∅) from the oracle's efficiency axiom: Σφ_oracle = v(N) - v(∅)
        phi_oracle = oracle.true_shapley(
            x, classifier, players, n_mc=self._n_mc, seed=self._oracle_seed
        )
        grand_diff = float(phi_oracle.sum().item())

        efficiency_error: float
        if abs(grand_diff) < 1e-12:
            efficiency_error = 0.0
        else:
            efficiency_error = abs(float(phi.sum().item()) - grand_diff) / abs(grand_diff)

        return {"efficiency_error": efficiency_error}
