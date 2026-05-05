"""motionbench.attribution.timeshap — Temporal KernelSHAP surrogate for TimeSHAP.

This module provides :class:`TimeSHAPAttributor`, which wraps ``shap.KernelExplainer``
to compute temporal SHAP values over player-level coalitions.

**Why not use the ``timeshap`` library directly?**

``timeshap`` 1.0.4 is incompatible with ``shap >= 0.42`` because it imports
``shap.explainers._kernel.Kernel``, a private class that was renamed in shap 0.42
(see BACKLOG entry B-3C-01). Until upstream fixes the import, this module implements
a drop-in surrogate:

1. Build a predict function that maps ``(N, M)`` binary coalition indicators to
   scalar predictions, using ``players.coalition_mask`` and a pluggable
   :class:`~motionbench.imputers.base.BaseImputer` to fill hidden coordinates.
2. Pass that function to ``shap.KernelExplainer`` with an all-zeros background
   (all players absent → model output when nothing is observed).
3. Evaluate SHAP values at the all-ones point (all players present) to obtain
   ``(M,)`` attributions.

This is mathematically equivalent to running KernelSHAP on the M-player game
exactly as described in Bento et al. (2020) "TimeSHAP: Explaining recurrent
models through sequence perturbations."

References
----------
Bento et al. (2020) "TimeSHAP: Explaining recurrent models through sequence
perturbations." arXiv:2012.00073.
Lundberg & Lee (2017) "A unified approach to interpreting model predictions."
NeurIPS 2017.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import shap
import torch
from torch import Tensor

from motionbench.attribution.base import BaseAttributor

if TYPE_CHECKING:
    from motionbench.imputers.base import BaseImputer
    from motionbench.players.base import PlayerSet


__all__ = ["TimeSHAPAttributor"]


def _make_predict_fn(
    x: Tensor,
    classifier: Callable[[Tensor], Tensor],
    imputer: BaseImputer,
    players: PlayerSet,
    target: int,
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """Build a KernelExplainer-compatible predict function over player coalitions.

    Args:
        x: ``(J, F, T)`` float32 input sequence.
        classifier: ``(B, J, F, T) → (B,)`` callable.
        imputer: Fitted imputer for filling hidden coordinates.
        players: PlayerSet defining M players and coordinate mapping.
        target: Output index to select when classifier returns multi-dim output.

    Returns:
        A function ``f(z_batch) → predictions`` where ``z_batch`` is
        ``(N, M)`` float64 and the return is ``(N,)`` float64.
    """

    def predict_fn(z_batch: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        results: list[float] = []
        for z_row in z_batch:
            z_tensor = torch.as_tensor(z_row > 0.5, dtype=torch.bool)
            mask = players.coalition_mask(z_tensor)
            x_imp = imputer.impute(x, mask, n_samples=1)[0]
            with torch.no_grad():
                pred = classifier(x_imp.unsqueeze(0))
            scalar = (
                float(pred[0, target].item())
                if pred.ndim > 1
                else float(pred[0].item())
            )
            results.append(scalar)
        return np.array(results, dtype=np.float64)

    return predict_fn


class TimeSHAPAttributor(BaseAttributor):
    """Temporal KernelSHAP surrogate for TimeSHAP.

    Uses ``shap.KernelExplainer`` at the player level as a drop-in replacement
    for the ``timeshap`` library, which is currently incompatible with
    ``shap >= 0.42`` (see BACKLOG B-3C-01).

    Each player corresponds to a group of ``(J, F, T)`` coordinates defined by
    the :class:`~motionbench.players.base.PlayerSet`. The value function is
    estimated by imputing hidden coordinates with the provided
    :class:`~motionbench.imputers.base.BaseImputer`.

    Args:
        classifier: ``(B, J, F, T) float32 → (B,) float32`` callable.
        imputer: Fitted :class:`~motionbench.imputers.base.BaseImputer`.
        n_coalitions: Number of coalition samples passed to
            ``KernelExplainer.shap_values`` as ``nsamples``. Larger values
            yield more accurate SHAP estimates at higher compute cost.
            Defaults to 100.
        seed: Optional random seed for reproducibility. Passed to numpy
            before each call to ``KernelExplainer``. Defaults to ``None``.
    """

    def __init__(
        self,
        classifier: Callable[[Tensor], Tensor],
        imputer: BaseImputer,
        n_coalitions: int = 100,
        seed: int | None = None,
    ) -> None:
        super().__init__(classifier)
        self._imputer = imputer
        self._n_coalitions = n_coalitions
        self._seed = seed

    # ------------------------------------------------------------------
    # BaseAttributor interface
    # ------------------------------------------------------------------

    def attribute(
        self,
        x: Tensor,
        players: PlayerSet,
        target: int = 0,
    ) -> Tensor:
        """Compute per-player temporal SHAP values via KernelSHAP surrogate.

        Args:
            x: ``(J, F, T)`` float32 input sequence.
            players: :class:`~motionbench.players.base.PlayerSet` with M players.
            target: Class index for attribution. Selects output column when the
                classifier returns ``(B, n_classes)``; ignored for scalar output.

        Returns:
            ``(M,)`` float32 Tensor of per-player SHAP values.
        """
        M = players.n_players
        if self._seed is not None:
            np.random.seed(self._seed)

        predict_fn = _make_predict_fn(
            x, self._classifier, self._imputer, players, target
        )

        # Background: all players absent → reference output when nothing is observed.
        background = np.zeros((1, M), dtype=np.float64)
        explainer = shap.KernelExplainer(predict_fn, background)

        # Explain the fully observed point (all players present).
        x_test = np.ones((1, M), dtype=np.float64)
        raw = explainer.shap_values(x_test, nsamples=self._n_coalitions, silent=True)

        phi = np.asarray(raw).reshape(M)
        return torch.tensor(phi, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Short identifier for logging."""
        return "TimeSHAP"

    @property
    def requires_imputer(self) -> bool:
        """TimeSHAP requires a fitted imputer."""
        return True
