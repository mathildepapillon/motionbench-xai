"""motionbench.attribution.lrp — Layer-wise Relevance Propagation via Zennit.

LRP decomposes a model prediction by backpropagating relevance scores from the
output to the input layer, distributing relevance according to per-layer
propagation rules.  This module wraps the Zennit library to provide three rule
variants commonly used in the gait-analysis XAI literature.

Supported rules
---------------
* ``"epsilon"``    — Epsilon rule (Bach et al. 2015).  Numerically stable and
  satisfies near-perfect conservation; recommended by Slijepcevic et al. (2022)
  for temporal/fully-connected layers in skeleton-based action recognition.
* ``"gamma"``      — Gamma rule (Montavon et al. 2019).  Emphasises positive
  contributions by boosting positive weights with factor ``γ``; recommended by
  Horst et al. (2019) for convolutional gait-classification layers.
* ``"alpha_beta"`` — Alpha-Beta rule (Bach et al. 2015).  Separates positive
  and negative contributions via ``α`` and ``β`` weights; ``α=2, β=1`` by
  default (satisfies ``α − β = 1`` conservation constraint).

References
----------
Bach et al. (2015) "On pixel-wise explanations for non-linear classifier
    decisions by layer-wise relevance propagation." PLOS ONE 10(7).
Montavon et al. (2019) "Layer-wise relevance propagation: an overview."
    In Explainable AI, Lecture Notes in Computer Science.
Slijepcevic et al. (2022) "Explaining deep neural network-based movement
    analysis with layer-wise relevance propagation." Pattern Recognit. Lett.
Horst et al. (2019) "Explaining the unique nature of individual gait patterns
    with deep neural networks." Sci. Rep. 9, 2391.
Zennit: https://github.com/chr5tphr/zennit
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
from torch import Tensor
from zennit.attribution import Gradient
from zennit.composites import AlphaBeta, Epsilon, Gamma, LayerMapComposite, layer_map_base

from motionbench.attribution.base import BaseAttributor

if TYPE_CHECKING:
    from motionbench.players.base import PlayerSet

__all__ = ["LRPAttributor"]

_Rule = Literal["epsilon", "gamma", "alpha_beta"]
_VALID_RULES: tuple[str, ...] = ("epsilon", "gamma", "alpha_beta")


class LRPAttributor(BaseAttributor):
    """Layer-wise Relevance Propagation via Zennit.

    Supported rules: ``"epsilon"``, ``"gamma"``, ``"alpha_beta"``.
    Default rule choices follow the gait XAI literature:

    * Slijepcevic et al. (2022) recommend the **epsilon rule** for
      temporal/fully-connected layers in skeleton-based action recognition.
    * Horst et al. (2019) recommend the **gamma rule** for convolutional
      gait-classification layers.

    The classifier must be an :class:`torch.nn.Module` because Zennit registers
    forward and backward hooks on individual layers to apply the LRP propagation
    rules.

    References:
        Bach et al. (2015) "On pixel-wise explanations..." PLOS ONE.
        Slijepcevic et al. (2022) — rule selection guidance.
        Horst et al. (2019) — gait LRP application.
        Zennit: https://github.com/chr5tphr/zennit
    """

    @property
    def requires_gradient(self) -> bool:
        """LRP requires gradient flow through the classifier."""
        return True

    def __init__(
        self,
        classifier: nn.Module,
        rule: _Rule = "epsilon",
        epsilon: float = 1e-6,
        gamma: float = 0.25,
        alpha: float = 2.0,
        beta: float = 1.0,
        **kwargs: object,
    ) -> None:
        """Initialise LRPAttributor.

        Args:
            classifier: ``nn.Module`` mapping ``(B, J, F, T) → (B,)`` float32.
                Must be an ``nn.Module`` (not a plain callable) because Zennit
                attaches per-layer hooks to propagate relevance.
            rule: LRP propagation rule — one of ``"epsilon"``, ``"gamma"``,
                or ``"alpha_beta"``.  Defaults to ``"epsilon"``.
            epsilon: Stabiliser added to the denominator in the epsilon rule.
                Ignored for other rules.  Defaults to ``1e-6``.
            gamma: Enhancement factor applied to positive weights in the gamma
                rule (Horst et al. 2019).  Ignored for other rules.
            alpha: Weight on positive contributions in the alpha-beta rule.
                Must satisfy ``alpha - beta == 1`` for conservation.
            beta: Weight on negative contributions in the alpha-beta rule.
                Defaults to ``1.0`` (paired with ``alpha=2.0``).
            **kwargs: Forwarded to
                :class:`~motionbench.attribution.base.BaseAttributor`.

        Raises:
            TypeError: if ``classifier`` is not an ``nn.Module``.
            ValueError: if ``rule`` is not one of the supported values.
        """
        if not isinstance(classifier, nn.Module):
            raise TypeError(
                f"LRPAttributor requires an nn.Module classifier, got {type(classifier).__name__}."
            )
        if rule not in _VALID_RULES:
            raise ValueError(f"rule must be one of {_VALID_RULES!r}, got {rule!r}.")
        super().__init__(classifier=classifier, **kwargs)
        self._model: nn.Module = classifier
        self._rule: _Rule = rule
        self._epsilon: float = epsilon
        self._gamma: float = gamma
        self._alpha: float = alpha
        self._beta: float = beta
        self._composite: LayerMapComposite = self._build_composite()

    def _build_composite(self) -> LayerMapComposite:
        """Build the Zennit LayerMapComposite for the selected LRP rule.

        Each rule is applied uniformly to all ``nn.Linear`` and convolutional
        layers via :class:`zennit.composites.LayerMapComposite`.  Activation
        and batch-norm layers receive ``Pass`` (identity) via ``layer_map_base``.

        Returns:
            A :class:`~zennit.composites.LayerMapComposite` configured for the
            chosen rule.
        """
        import torch.nn as _nn

        if self._rule == "epsilon":
            rule: Epsilon | Gamma | AlphaBeta = Epsilon(epsilon=self._epsilon)
        elif self._rule == "gamma":
            rule = Gamma(gamma=self._gamma)
        else:
            rule = AlphaBeta(alpha=self._alpha, beta=self._beta)

        layer_map = layer_map_base() + [
            (_nn.Linear, rule),
        ]
        return LayerMapComposite(layer_map=layer_map)

    def attribute(
        self,
        x: Tensor,
        players: PlayerSet,
        target: int = 0,
    ) -> Tensor:
        """Compute ``(M,)`` LRP relevances for a single input sequence.

        Relevance is computed by performing a forward pass through the
        classifier then backpropagating relevance from the scalar output using
        the configured LRP rule.  Coordinate-level relevances are then
        aggregated to player level via :meth:`players.aggregate`.

        The ``target`` argument is accepted for API compatibility.  Because the
        classifier is expected to return a ``(B,)`` scalar already corresponding
        to the desired target class, the argument is not used internally.

        Args:
            x: ``(J, F, T)`` float32 input sequence (no batch dimension).
            players: :class:`~motionbench.players.base.PlayerSet` defining
                the M players and coordinate→player aggregation.
            target: Class index (accepted for interface compatibility; not
                used because the classifier returns a single scalar).

        Returns:
            ``(M,)`` float32 tensor of per-player LRP relevance scores.
        """
        x_in: Tensor = x.unsqueeze(0).float()  # (1, J, F, T)

        attributor = Gradient(model=self._model, composite=self._composite)
        with torch.enable_grad():
            _output, attribution = attributor(x_in)  # (1, J, F, T)

        phi_coords: Tensor = attribution.squeeze(0).detach()  # (J, F, T)
        return players.aggregate(phi_coords)
