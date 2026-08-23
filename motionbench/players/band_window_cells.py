"""motionbench.players.band_window_cells — Frequency-band × window cell player set.

Each player is one (frequency band, temporal window) cell over a
spectrogram-shaped input ``(J, F, T)`` whose ``J`` axis indexes frequency
bins (e.g. 128 mel bins for ESC-50).  The ``J`` bins are partitioned into
``n_bands`` contiguous bands of (near-)equal width — the same partition as
the per-band spatial players in ``scripts/run_esc50_freq_shap.py`` — and the
``T`` axis into ``K`` equal windows, giving ``M = n_bands * K`` players.

The player index for band ``b`` and window ``k`` is ``b * K + k`` (same
convention as :class:`~motionbench.players.joint_window_cells.JointWindowCells`).

Example::

    players = BandWindowCells(J=128, n_bands=4, K=4, F=1, T=1024)
    # 16 players: 4 mel-bin quartiles x 4 windows of 256 frames
    mask = players.coalition_mask(z)  # (128, 1, 1024) bool
"""

from __future__ import annotations

import torch
from torch import Tensor

from motionbench.players.base import PlayerSet

__all__ = ["BandWindowCells"]


class BandWindowCells(PlayerSet):
    """Spectro-temporal (frequency-band × temporal-window) cell player set.

    The ``J`` frequency bins are split into ``n_bands`` contiguous bands via
    ``divmod(J, n_bands)`` (the first ``J % n_bands`` bands get one extra
    bin), matching the band partition of the ESC-50 per-band sweep.

    Args:
        J: Number of frequency bins (e.g. 128 mel bins).
        n_bands: Number of contiguous frequency bands.
        K: Number of equal-width temporal windows.  Must divide T.
        F: Number of features per bin (1 for spectrograms).
        T: Total number of time steps.

    Raises:
        ValueError: if ``T`` is not divisible by ``K`` or ``n_bands > J``.
    """

    def __init__(self, J: int, n_bands: int, K: int, F: int, T: int) -> None:
        if T % K != 0:
            raise ValueError(f"T={T} must be divisible by K={K} for equal-width windows.")
        if n_bands > J:
            raise ValueError(f"n_bands={n_bands} cannot exceed J={J} frequency bins.")
        self._J = J
        self._B = n_bands
        self._K = K
        self._F = F
        self._T = T
        self._ws = T // K
        self._M = n_bands * K
        base, extra = divmod(J, n_bands)
        edges: list[tuple[int, int]] = []
        start = 0
        for b in range(n_bands):
            end = start + base + (1 if b < extra else 0)
            edges.append((start, end))
            start = end
        self._band_edges = edges

    @property
    def n_players(self) -> int:
        """Number of players M = n_bands * K."""
        return self._M

    @property
    def shape(self) -> tuple[int, int, int]:
        """(J, F, T) element-space shape this player set operates over."""
        return self._J, self._F, self._T

    @property
    def band_edges(self) -> list[tuple[int, int]]:
        """Half-open ``(start, end)`` bin ranges of the frequency bands."""
        return list(self._band_edges)

    def player_index(self, b: int, k: int) -> int:
        """Return the player index for band ``b`` and window ``k``."""
        return b * self._K + k

    def coalition_mask(self, z: Tensor) -> Tensor:
        """Expand a coalition indicator to an element-level boolean mask.

        Args:
            z: ``(n_bands*K,)`` binary tensor.  1 = cell is observed.

        Returns:
            ``(J, F, T)`` bool tensor.

        Raises:
            ValueError: if ``z.shape != (n_bands*K,)``.
        """
        if z.shape != (self._M,):
            raise ValueError(f"Expected z.shape==({self._M},); got {tuple(z.shape)}.")
        mask = torch.zeros(self._J, self._F, self._T, dtype=torch.bool)
        for b in range(self._B):
            j0, j1 = self._band_edges[b]
            for k in range(self._K):
                if z[b * self._K + k]:
                    mask[j0:j1, :, k * self._ws : (k + 1) * self._ws] = True
        return mask

    def aggregate(self, phi_coords: Tensor) -> Tensor:
        """Aggregate per-coordinate attributions to per-cell level.

        Args:
            phi_coords: ``(J, F, T)`` float tensor.

        Returns:
            ``(n_bands*K,)`` float tensor.

        Raises:
            ValueError: if ``phi_coords.shape != (J, F, T)``.
        """
        if phi_coords.shape != (self._J, self._F, self._T):
            raise ValueError(
                f"Expected phi_coords.shape=={(self._J, self._F, self._T)}; "
                f"got {tuple(phi_coords.shape)}."
            )
        phi = torch.zeros(self._M, dtype=phi_coords.dtype)
        for b in range(self._B):
            j0, j1 = self._band_edges[b]
            for k in range(self._K):
                phi[b * self._K + k] = phi_coords[j0:j1, :, k * self._ws : (k + 1) * self._ws].sum()
        return phi

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(J={self._J}, n_bands={self._B}, "
            f"K={self._K}, F={self._F}, T={self._T})"
        )
