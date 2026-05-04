"""motionbench.metrics.base — BaseMetric abstract base class.

A *metric* takes a per-player attribution vector ``φ ∈ ℝ^M`` and evaluates
its quality along one axis (ground-truth accuracy, fidelity, stability, or
sanity).  All metrics return a ``dict[str, float]`` so that multiple
sub-scores can be reported under one metric call.

Metric taxonomy in this benchmark
----------------------------------

**Ground-truth metrics** (``requires_oracle = True``):
    EC1 (mean L1), EC2 (MSE), EC3 (1 − Pearson) — compare φ to oracle φ.
    TopKRecovery, SpearmanRank, KendallRank — ordinal agreement.
    EfficiencyError — checks Σφ ≈ v(N) − v(∅).

**Fidelity metrics** (``requires_imputer = True``):
    FaithfulnessCorrelation, PixelFlipping (deletion / insertion),
    MonotonicityCorrelation — all with on-manifold and off-manifold variants.

**Stability metrics** (no oracle or imputer required):
    MaxSensitivity, Continuity, LipschitzEstimate.

**Sanity-check metrics** (no oracle required):
    ModelParameterRandomisation, RandomLogit (Adebayo et al. 2018).

**Meta-metrics:**
    RankingAgreement — cross-protocol Spearman correlation matrix.

Class variables
---------------
Subclasses must declare:

* ``requires_oracle: ClassVar[bool]`` — whether ``oracle`` must be provided.
* ``requires_imputer: ClassVar[bool]`` — whether ``imputer`` must be provided.

If ``evaluate`` is called without a required dependency, it should raise
:py:exc:`ValueError` with a descriptive message.

References
----------
Adebayo et al. (2018) "Sanity Checks for Saliency Maps." NeurIPS.
Hedström et al. (2023) "Quantus: An Explainability Toolkit for Responsible
Evaluation of Neural Network Explanations." JMLR 24(34).
Aas et al. (2021) §3 — conditional expectation game.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, ClassVar, Optional

from torch import Tensor

if TYPE_CHECKING:
    from motionbench.imputers.base import BaseImputer
    from motionbench.oracles.base import Oracle
    from motionbench.players.base import PlayerSet


__all__ = ["BaseMetric"]


class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics.

    Shape conventions:

    * Attribution:   ``(M,)`` float32 Tensor  — per-player Shapley values.
    * Sequence:      ``(J, F, T)`` float32 Tensor.
    * Classifier:    callable ``(B, J, F, T) → (B,)`` float32.

    Class variables (must be set in every subclass)
    -----------------------------------------------
    ``requires_oracle``
        Whether this metric requires a ground-truth oracle.  Set ``True``
        for all ground-truth metrics (EC1–EC3, TopK, …).
    ``requires_imputer``
        Whether this metric requires a fitted imputer.  Set ``True`` for
        fidelity metrics (PixelFlipping, FaithfulnessCorrelation, …).
    """

    requires_oracle: ClassVar[bool] = False
    requires_imputer: ClassVar[bool] = False

    @abstractmethod
    def evaluate(
        self,
        phi: Tensor,
        x: Tensor,
        classifier: Callable[[Tensor], Tensor],
        players: "PlayerSet",
        target: int = 0,
        oracle: Optional["Oracle"] = None,
        imputer: Optional["BaseImputer"] = None,
    ) -> dict[str, float]:
        """Evaluate attribution quality for a single sequence.

        Args:
            phi: ``(M,)`` float32 per-player attribution vector produced
                by the attributor under evaluation.
            x: ``(J, F, T)`` float32 input sequence.
            classifier: Callable ``(B, J, F, T) → (B,)`` scalar target.
            players: :class:`~motionbench.players.base.PlayerSet` used to
                produce ``phi``.
            target: Class index (must match the one used to produce ``phi``).
            oracle: Ground-truth oracle.  Required if
                ``self.requires_oracle is True``; ignored otherwise.
            imputer: Fitted imputer.  Required if
                ``self.requires_imputer is True``; ignored otherwise.

        Returns:
            ``dict[str, float]`` mapping metric sub-score names to values.
            For example, EC1 returns ``{"ec1": 0.034}``; TopK returns
            ``{"top1": 1.0, "topk_overlap": 0.75}``.

        Raises:
            ValueError: if ``self.requires_oracle and oracle is None``
                or ``self.requires_imputer and imputer is None``.
        """

    def _check_deps(
        self,
        oracle: Optional["Oracle"],
        imputer: Optional["BaseImputer"],
    ) -> None:
        """Raise ValueError if required dependencies are missing.

        Call this at the top of ``evaluate`` implementations.
        """
        if self.requires_oracle and oracle is None:
            raise ValueError(
                f"{self.__class__.__name__} requires an oracle "
                "(requires_oracle=True) but oracle=None was passed."
            )
        if self.requires_imputer and imputer is None:
            raise ValueError(
                f"{self.__class__.__name__} requires a fitted imputer "
                "(requires_imputer=True) but imputer=None was passed."
            )

    @property
    def name(self) -> str:
        """Short string identifier for logging and leaderboard tables."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"requires_oracle={self.requires_oracle}, "
            f"requires_imputer={self.requires_imputer})"
        )
