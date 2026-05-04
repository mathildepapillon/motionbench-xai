"""motionbench.players.base — PlayerSet abstract base class.

A *player* is the atomic unit of explanation in the group-SHAP game: any
method that reports Shapley values in motionbench reports one number per
player, not one number per scalar coordinate.  A PlayerSet defines:

1. How many players M exist.
2. How a length-M coalition indicator z ∈ {0,1}^M maps to the element-level
   boolean mask over (J, F, T) coordinates that an imputer consumes.
3. How per-coordinate gradient/attribution tensors are aggregated back to the
   per-player level.

Canonical implementations
--------------------------
- ``TemporalWindows(K, T)``      — K equal-width time windows (WindowSHAP / TimeSHAP).
- ``SpatialJoints(J)``           — each skeletal joint is one player.
- ``AnatomicalGroups(groups)``   — pre-defined joint groups (left-leg, right-leg, …).
- ``GaitPhase(n_phases)``        — stride-aligned phases (stance, swing, double-support).
- ``JointWindowCells(J, K)``     — J × K spatiotemporal cells (fine-grained).

Reference
---------
Jullum et al. (2021) "Groupwise Shapley Feature Importance Values" §3 —
indivisible group masking.
C-SHAP (Jutte et al. 2024) Algorithm 1 — group coalition expansion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    pass


__all__ = ["PlayerSet"]


class PlayerSet(ABC):
    """Abstract base class for all player-set definitions.

    A concrete subclass must implement:

    * :py:meth:`n_players` — total number of players M.
    * :py:meth:`coalition_mask` — expand a binary coalition vector to an
      element-level boolean tensor.
    * :py:meth:`aggregate` — reduce per-coordinate attribution to per-player
      attribution.

    Shape conventions used throughout motionbench:

    * ``(J, F, T)`` — per-sample coordinate layout: J joints, F features per
      joint, T time-steps.  All imputers and attributors use this layout.
    * ``(M,)``      — per-player attribution vector (M = n_players).
    * ``(M,)`` binary — coalition indicator (1 = player observed / included).
    """

    # ------------------------------------------------------------------
    # Shape metadata — set by concrete subclasses in __init__
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def n_players(self) -> int:
        """Number of players M in this player set."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int, int]:
        """(J, F, T) element-space shape this player set operates over."""

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    @abstractmethod
    def coalition_mask(self, z: Tensor) -> Tensor:
        """Expand a coalition indicator to an element-level boolean mask.

        Implements *indivisible group masking* (Jullum 2021 §3): when
        player k is hidden (z[k] == 0), **every** coordinate assigned to
        player k is set to False (hidden).

        Args:
            z: ``(M,)`` binary int/bool tensor.  1 = player observed.

        Returns:
            ``(J, F, T)`` bool tensor.  True = coordinate is observed.

        Raises:
            ValueError: if ``z.shape != (M,)``.
        """

    @abstractmethod
    def aggregate(self, phi_coords: Tensor) -> Tensor:
        """Aggregate per-coordinate attributions to per-player level.

        Uses the additivity of Shapley values: the value assigned to a player
        group equals the sum of the values of its constituent elements (Jullum
        2021, Proposition 1).

        Args:
            phi_coords: ``(J, F, T)`` float tensor of per-coordinate
                attribution values (e.g. from Captum IG or SmoothGrad).

        Returns:
            ``(M,)`` float tensor of per-player attribution scores.

        Raises:
            ValueError: if ``phi_coords.shape != (J, F, T)``.
        """

    # ------------------------------------------------------------------
    # Optional helper — can be overridden for efficiency
    # ------------------------------------------------------------------

    def batch_coalition_masks(self, Z: Tensor) -> Tensor:
        """Expand a batch of coalition indicators to element-level masks.

        Default implementation loops over rows; concrete subclasses may
        override with a vectorised implementation.

        Args:
            Z: ``(N, M)`` binary int/bool tensor.

        Returns:
            ``(N, J, F, T)`` bool tensor.
        """
        masks = torch.stack([self.coalition_mask(Z[i]) for i in range(Z.shape[0])])
        return masks

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        J, F, T = self.shape
        return (
            f"{self.__class__.__name__}("
            f"n_players={self.n_players}, J={J}, F={F}, T={T})"
        )
