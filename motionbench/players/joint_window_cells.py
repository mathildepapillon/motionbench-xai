"""motionbench.players.joint_window_cells — Spatiotemporal J×K cell player set.

Each player is one (joint, temporal-window) cell.  This is the finest
resolution player set: M = J × K players.  It enables analysis of which
joint is important in which phase of movement.

The player index for joint ``j`` and window ``k`` is ``j * K + k``.

Args:
    J: Number of joints.
    K: Number of temporal windows.  Must divide T evenly.
    F: Number of features per joint.
    T: Total number of time steps.

Example::

    players = JointWindowCells(J=17, K=4, F=3, T=81)
    # 68 players total (17 joints × 4 windows)
    phi = players.aggregate(phi_coords)  # (68,)
"""

from __future__ import annotations

import torch
from torch import Tensor

from motionbench.players.base import PlayerSet

__all__ = ["JointWindowCells"]


class JointWindowCells(PlayerSet):
    """Spatiotemporal (joint × temporal-window) cell player set.

    Args:
        J: Number of joints.
        K: Number of equal-width temporal windows.  Must divide T.
        F: Number of features per joint.
        T: Total number of time steps.

    Raises:
        ValueError: if ``T`` is not divisible by ``K``.
    """

    def __init__(self, J: int, K: int, F: int, T: int) -> None:
        if T % K != 0:
            raise ValueError(
                f"T={T} must be divisible by K={K} for equal-width windows."
            )
        self._J = J
        self._K = K
        self._F = F
        self._T = T
        self._ws = T // K
        self._M = J * K

    @property
    def n_players(self) -> int:
        return self._M

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._J, self._F, self._T

    def player_index(self, j: int, k: int) -> int:
        """Return the player index for joint ``j`` and window ``k``."""
        return j * self._K + k

    def coalition_mask(self, z: Tensor) -> Tensor:
        """Expand a coalition indicator to an element-level boolean mask.

        Args:
            z: ``(J*K,)`` binary tensor.  1 = cell is observed.

        Returns:
            ``(J, F, T)`` bool tensor.

        Raises:
            ValueError: if ``z.shape != (J*K,)``.
        """
        if z.shape != (self._M,):
            raise ValueError(
                f"Expected z.shape==({self._M},); got {tuple(z.shape)}."
            )
        mask = torch.zeros(self._J, self._F, self._T, dtype=torch.bool)
        for j in range(self._J):
            for k in range(self._K):
                p = j * self._K + k
                if z[p]:
                    t0 = k * self._ws
                    t1 = (k + 1) * self._ws
                    mask[j, :, t0:t1] = True
        return mask

    def aggregate(self, phi_coords: Tensor) -> Tensor:
        """Aggregate per-coordinate attributions to per-cell level.

        Args:
            phi_coords: ``(J, F, T)`` float tensor.

        Returns:
            ``(J*K,)`` float tensor.

        Raises:
            ValueError: if ``phi_coords.shape != (J, F, T)``.
        """
        if phi_coords.shape != (self._J, self._F, self._T):
            raise ValueError(
                f"Expected phi_coords.shape=={(self._J, self._F, self._T)}; "
                f"got {tuple(phi_coords.shape)}."
            )
        phi = torch.zeros(self._M, dtype=phi_coords.dtype)
        for j in range(self._J):
            for k in range(self._K):
                t0 = k * self._ws
                t1 = (k + 1) * self._ws
                phi[j * self._K + k] = phi_coords[j, :, t0:t1].sum()
        return phi
