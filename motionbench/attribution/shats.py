"""motionbench.attribution.shats — ShaTS attributor stub.

This module provides :class:`ShaTSAttributor`, a placeholder for the ShaTS
(Shapley Time Series) attribution method.

**Status: stub only.**

The ``shats`` library has not yet been released as a Python package on PyPI.
Once it becomes available, this stub should be replaced with a real wrapper
(see BACKLOG entry B-3C-02).  Until then, calling :meth:`ShaTSAttributor.attribute`
raises :exc:`NotImplementedError` with an actionable message.

References
----------
Bourdoukan & Durner (2024) "ShaTS: Shapley Time Series." (preprint)
BACKLOG B-3C-02: shats not yet available as Python package; stub only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from motionbench.attribution.base import BaseAttributor

if TYPE_CHECKING:
    from torch import Tensor

    from motionbench.players.base import PlayerSet


__all__ = ["ShaTSAttributor"]


class ShaTSAttributor(BaseAttributor):
    """Stub attributor for the ShaTS method.

    Raises :exc:`NotImplementedError` on every call to :meth:`attribute`
    because the ``shats`` Python package is not yet available on PyPI.
    Install from the upstream repository when it is released (see BACKLOG
    entry B-3C-02).

    Args:
        classifier: ``(B, J, F, T) float32 → (B,) float32`` callable.
            Accepted for interface compatibility but not used.
        **kwargs: Forwarded to :class:`~motionbench.attribution.base.BaseAttributor`.
    """

    def __init__(
        self,
        classifier: Callable[[Tensor], Tensor],
        **kwargs: object,
    ) -> None:
        super().__init__(classifier, **kwargs)

    # ------------------------------------------------------------------
    # BaseAttributor interface
    # ------------------------------------------------------------------

    def attribute(
        self,
        x: Tensor,
        players: PlayerSet,
        target: int = 0,
    ) -> Tensor:
        """Not implemented — shats library not yet available.

        Args:
            x: ``(J, F, T)`` float32 input sequence.
            players: :class:`~motionbench.players.base.PlayerSet`.
            target: Class index (unused).

        Raises:
            NotImplementedError: Always. Install ``shats`` from
                ``https://github.com/author/shats`` when released.
        """
        raise NotImplementedError(
            "shats library not available; install from "
            "https://github.com/author/shats when released"
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Short identifier for logging."""
        return "ShaTS"
