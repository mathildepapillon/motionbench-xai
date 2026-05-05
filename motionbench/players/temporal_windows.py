"""motionbench.players.temporal_windows — Equal-width temporal window player set.

Used by TimeSHAP, WindowSHAP, and KernelSHAP experiments where the unit of
explanation is a time segment rather than an individual joint or frame.
"""

from __future__ import annotations

import torch
from torch import Tensor

from motionbench.players.base import PlayerSet

__all__ = ["TemporalWindows"]


class TemporalWindows(PlayerSet):
    """K equal-width, non-overlapping temporal windows spanning all joints.

    The ``k``-th player covers frames ``[k*ws, (k+1)*ws)`` across every joint
    and feature channel.  ``T`` must be divisible by ``K``.

    Args:
        K: Number of windows (players).
        T: Total number of time steps.
        J: Number of joints.
        F: Number of features per joint.

    Raises:
        ValueError: if ``T`` is not divisible by ``K``.

    Example::

        players = TemporalWindows(K=4, T=64, J=17, F=3)
        z = torch.ones(4, dtype=torch.int)  # all windows observed
        mask = players.coalition_mask(z)    # (17, 3, 64) all-True
    """

    def __init__(self, K: int, T: int, J: int, F: int) -> None:
        if T % K != 0:
            raise ValueError(
                f"T={T} must be divisible by K={K} for equal-width windows."
            )
        self._K = K
        self._T = T
        self._J = J
        self._F = F
        self._ws = T // K

    @property
    def n_players(self) -> int:
        return self._K

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._J, self._F, self._T

    def coalition_mask(self, z: Tensor) -> Tensor:
        """Expand a coalition indicator to an element-level boolean mask.

        Args:
            z: ``(K,)`` binary int/bool tensor.  1 = window is observed.

        Returns:
            ``(J, F, T)`` bool tensor.

        Raises:
            ValueError: if ``z.shape != (K,)``.
        """
        if z.shape != (self._K,):
            raise ValueError(
                f"Expected z.shape==({self._K},); got {tuple(z.shape)}."
            )
        mask = torch.zeros(self._J, self._F, self._T, dtype=torch.bool)
        for k in range(self._K):
            if z[k]:
                t0 = k * self._ws
                t1 = (k + 1) * self._ws
                mask[:, :, t0:t1] = True
        return mask

    def aggregate(self, phi_coords: Tensor) -> Tensor:
        """Aggregate per-coordinate attributions to per-window level.

        Uses Shapley additivity: the window's value equals the sum of all
        coordinate values within that window (Jullum 2021, Proposition 1).

        Args:
            phi_coords: ``(J, F, T)`` float tensor.

        Returns:
            ``(K,)`` float tensor.

        Raises:
            ValueError: if ``phi_coords.shape != (J, F, T)``.
        """
        if phi_coords.shape != (self._J, self._F, self._T):
            raise ValueError(
                f"Expected phi_coords.shape=={(self._J, self._F, self._T)}; "
                f"got {tuple(phi_coords.shape)}."
            )
        phi = torch.zeros(self._K, dtype=phi_coords.dtype)
        for k in range(self._K):
            t0 = k * self._ws
            t1 = (k + 1) * self._ws
            phi[k] = phi_coords[:, :, t0:t1].sum()
        return phi
