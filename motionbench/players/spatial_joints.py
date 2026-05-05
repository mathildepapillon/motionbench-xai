"""motionbench.players.spatial_joints — Per-joint player set.

Each skeletal joint is one player.  All features and time-steps of a joint
move together as a unit.  Natural for gradient-based methods and Captum IG
where the question is "which joint matters most?".
"""

from __future__ import annotations

import torch
from torch import Tensor

from motionbench.players.base import PlayerSet

__all__ = ["SpatialJoints"]


class SpatialJoints(PlayerSet):
    """One player per skeletal joint, spanning all features and time-steps.

    The ``j``-th player covers the full ``(F, T)`` slice for joint ``j``.

    Args:
        J: Number of joints (= number of players M).
        F: Number of features per joint.
        T: Number of time steps.

    Example::

        players = SpatialJoints(J=17, F=3, T=81)
        z = torch.zeros(17, dtype=torch.int)
        z[0] = 1  # only root joint observed
        mask = players.coalition_mask(z)  # (17, 3, 81): only joint 0 True
    """

    def __init__(self, J: int, F: int, T: int) -> None:
        self._J = J
        self._F = F
        self._T = T

    @property
    def n_players(self) -> int:
        return self._J

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._J, self._F, self._T

    def coalition_mask(self, z: Tensor) -> Tensor:
        """Expand a coalition indicator to an element-level boolean mask.

        Args:
            z: ``(J,)`` binary int/bool tensor.  1 = joint is observed.

        Returns:
            ``(J, F, T)`` bool tensor.

        Raises:
            ValueError: if ``z.shape != (J,)``.
        """
        if z.shape != (self._J,):
            raise ValueError(
                f"Expected z.shape==({self._J},); got {tuple(z.shape)}."
            )
        mask = torch.zeros(self._J, self._F, self._T, dtype=torch.bool)
        for j in range(self._J):
            if z[j]:
                mask[j, :, :] = True
        return mask

    def aggregate(self, phi_coords: Tensor) -> Tensor:
        """Aggregate per-coordinate attributions to per-joint level.

        Args:
            phi_coords: ``(J, F, T)`` float tensor.

        Returns:
            ``(J,)`` float tensor — sum over ``(F, T)`` for each joint.

        Raises:
            ValueError: if ``phi_coords.shape != (J, F, T)``.
        """
        if phi_coords.shape != (self._J, self._F, self._T):
            raise ValueError(
                f"Expected phi_coords.shape=={(self._J, self._F, self._T)}; "
                f"got {tuple(phi_coords.shape)}."
            )
        return phi_coords.sum(dim=(1, 2))
