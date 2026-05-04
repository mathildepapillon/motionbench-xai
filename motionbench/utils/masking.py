"""motionbench.utils.masking — element-level mask utilities.

Helpers for converting coalition indicators to element-level boolean masks
and validating mask shapes throughout the motionbench pipeline.

A *coalition indicator* ``z`` is a ``(M,)`` binary tensor where each entry
signals whether the corresponding player is observed (1/True) or hidden
(0/False).  A *PlayerSet* defines the mapping from players to ``(J, F, T)``
coordinates via :py:meth:`~motionbench.players.base.PlayerSet.coalition_mask`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from motionbench.players.base import PlayerSet


__all__ = ["coalition_to_element_mask", "assert_mask_shape"]


def coalition_to_element_mask(z: Tensor, player_set: "PlayerSet") -> Tensor:
    """Convert a ``(M,)`` binary coalition indicator to a ``(J, F, T)`` element mask.

    Delegates to :py:meth:`~motionbench.players.base.PlayerSet.coalition_mask`,
    which is the canonical expansion logic for each player-set type.

    Args:
        z: ``(M,)`` binary int or bool tensor.  ``1``/``True`` = player
            is observed.
        player_set: :class:`~motionbench.players.base.PlayerSet` defining
            the ``M`` players and their coordinate assignments.

    Returns:
        ``(J, F, T)`` bool tensor.  ``True`` = coordinate is observed.

    Raises:
        ValueError: if ``z.shape != (M,)`` (propagated from ``player_set``).
    """
    return player_set.coalition_mask(z)


def assert_mask_shape(mask: Tensor, J: int, F: int, T: int) -> None:
    """Raise ``ValueError`` if ``mask.shape != (J, F, T)``.

    Args:
        mask: Tensor whose shape is to be validated.
        J: Expected number of joints.
        F: Expected number of features per joint.
        T: Expected number of time-steps.

    Raises:
        ValueError: if ``mask.shape != (J, F, T)``.
    """
    if tuple(mask.shape) != (J, F, T):
        raise ValueError(
            f"Expected mask shape ({J}, {F}, {T}), got {tuple(mask.shape)}"
        )
