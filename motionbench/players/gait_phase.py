"""motionbench.players.gait_phase — Stride-aligned gait phase player set.

Partitions the time axis into ``n_phases`` equal-length phases aligned to one
full gait cycle (stride period).  Classic gait phases:
    - Stance (foot flat → heel off)
    - Push-off
    - Initial swing
    - Terminal swing

The number and boundaries of phases are configurable; they are mapped to
frame indices assuming the sequence spans exactly one stride.

For multi-stride sequences, set ``n_strides > 1`` to tile the phases.  Each
(stride, phase) combination becomes a distinct player, giving
``n_strides * n_phases`` players total.  This matches the temporal grouping
used in CARE-PD's group-SHAP analysis.

Args:
    n_phases: Number of gait phases per stride.
    T: Total number of time steps.
    J: Number of joints.
    F: Number of features per joint.
    n_strides: Number of strides in the sequence (default 1).
        If ``T`` is not exactly divisible by ``n_phases * n_strides``, the
        last phase of the last stride absorbs the remainder frames.

Example::

    # 4-phase single stride, H36M skeleton
    players = GaitPhase(n_phases=4, T=81, J=17, F=3)
    # 8 players for a 2-stride sequence
    players = GaitPhase(n_phases=4, T=162, J=17, F=3, n_strides=2)
"""

from __future__ import annotations

import torch
from torch import Tensor

from motionbench.players.base import PlayerSet

__all__ = ["GaitPhase"]


class GaitPhase(PlayerSet):
    """Stride-aligned gait-phase player set.

    Args:
        n_phases: Phases per stride (typically 2 or 4).
        T: Total time steps.
        J: Number of joints.
        F: Features per joint.
        n_strides: Number of strides (default 1).
    """

    def __init__(
        self,
        n_phases: int,
        T: int,
        J: int,
        F: int,
        n_strides: int = 1,
    ) -> None:
        self._n_phases = n_phases
        self._n_strides = n_strides
        self._T = T
        self._J = J
        self._F = F
        self._M = n_phases * n_strides

        # Compute frame boundaries for each (stride, phase) player
        total_cells = n_phases * n_strides
        base_len = T // total_cells
        remainder = T % total_cells

        self._boundaries: list[tuple[int, int]] = []
        t = 0
        for cell in range(total_cells):
            cell_len = base_len + (1 if cell < remainder else 0)
            self._boundaries.append((t, t + cell_len))
            t += cell_len

    @property
    def n_players(self) -> int:
        return self._M

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._J, self._F, self._T

    @property
    def phase_boundaries(self) -> list[tuple[int, int]]:
        """Frame boundaries ``(start, end)`` for each player (stride × phase)."""
        return list(self._boundaries)

    def coalition_mask(self, z: Tensor) -> Tensor:
        """Expand a coalition indicator to an element-level boolean mask.

        Args:
            z: ``(M,)`` binary tensor.  1 = phase/stride cell is observed.

        Returns:
            ``(J, F, T)`` bool tensor.

        Raises:
            ValueError: if ``z.shape != (M,)``.
        """
        if z.shape != (self._M,):
            raise ValueError(
                f"Expected z.shape==({self._M},); got {tuple(z.shape)}."
            )
        mask = torch.zeros(self._J, self._F, self._T, dtype=torch.bool)
        for p, (t0, t1) in enumerate(self._boundaries):
            if z[p]:
                mask[:, :, t0:t1] = True
        return mask

    def aggregate(self, phi_coords: Tensor) -> Tensor:
        """Aggregate per-coordinate attributions to per-phase level.

        Args:
            phi_coords: ``(J, F, T)`` float tensor.

        Returns:
            ``(M,)`` float tensor.

        Raises:
            ValueError: if ``phi_coords.shape != (J, F, T)``.
        """
        if phi_coords.shape != (self._J, self._F, self._T):
            raise ValueError(
                f"Expected phi_coords.shape=={(self._J, self._F, self._T)}; "
                f"got {tuple(phi_coords.shape)}."
            )
        phi = torch.zeros(self._M, dtype=phi_coords.dtype)
        for p, (t0, t1) in enumerate(self._boundaries):
            phi[p] = phi_coords[:, :, t0:t1].sum()
        return phi
