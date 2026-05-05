"""motionbench.attribution.attention_rollout — Attention Rollout attribution.

Implements the Attention Rollout algorithm (Abnar & Zuidema 2020) for
transformer encoder classifiers operating on ``(J, F, T)`` motion sequences.

Algorithm
---------
Given a list of per-layer attention-weight tensors
``[A_0, A_1, ..., A_{L-1}]``, each of shape ``(B, H, S, S)`` (B = batch,
H = heads, S = sequence length):

1. For each layer *l*, average over attention heads:
   ``A_avg = A_l[0].mean(dim=0)``  (shape ``(S, S)``).
2. Add residual connection: ``A_hat = 0.5 * A_avg + 0.5 * I_S``.
3. Accumulate rollout matrix: ``A_rollout = A_rollout @ A_hat``
   (initialised to ``I_S``).
4. Extract the first-token (CLS) row: ``phi_flat = A_rollout[0]``
   (shape ``(S,)``).
5. Reshape ``phi_flat`` back to ``(J, F, T)`` using the inverse of the
   input flattening applied by the classifier:

   * If ``S == J * F * T``: reshape directly (every coordinate is a token).
   * If ``S == T``: each time step is a token; broadcast over ``J`` and ``F``.
   * If ``S == J * F``: each joint-feature pair is a token; broadcast over
     ``T``.
   * Otherwise: a ``ValueError`` is raised.

6. Aggregate ``(J, F, T)`` to ``(M,)`` via ``players.aggregate``.

Requirements
------------
The classifier must expose a method::

    def get_attention_weights(self) -> list[Tensor]:
        ...

where each returned tensor has shape ``(B, H, S, S)`` (B = batch size,
H = number of attention heads, S = sequence length).  This method is called
**after** a forward pass so it can simply return cached weights.

References
----------
Abnar, S. & Zuidema, W. (2020). "Quantifying Attention Flow in Transformers."
    ACL 2020.  https://arxiv.org/abs/2005.00928
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import torch
from torch import Tensor

from motionbench.attribution.base import BaseAttributor

if TYPE_CHECKING:
    from collections.abc import Callable

    from motionbench.players.base import PlayerSet


__all__ = ["AttentionRolloutAttributor"]


class AttentionRolloutAttributor(BaseAttributor):
    """Attention rollout for transformer encoder classifiers (Abnar & Zuidema 2020).

    Rolls out attention weights across all layers by recursive multiplication.
    Each layer's attention matrix is averaged over heads and mixed with the
    identity matrix (to account for residual connections), then multiplied
    into a cumulative rollout matrix.  The first-token row of the resulting
    matrix is used as the per-coordinate attribution.

    The classifier must expose a ``get_attention_weights()`` method that
    returns a list of per-layer attention tensors with shape
    ``(B, H, S, S)``.  The method is called after a forward pass.

    Args:
        classifier: Callable ``(B, J, F, T) → (B,)`` or ``(B, n_classes)``.
            Must implement ``get_attention_weights() -> list[Tensor]``.
    """

    #: Always ``False`` — attention rollout only requires forward passes.
    requires_gradient: ClassVar[bool] = False

    def __init__(self, classifier: Callable[[Tensor], Tensor], **kwargs: object) -> None:
        super().__init__(classifier, **kwargs)

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def attribute(
        self,
        x: Tensor,
        players: PlayerSet,
        target: int = 0,
    ) -> Tensor:
        """Compute per-player attention rollout attributions.

        Args:
            x: ``(J, F, T)`` float32 input sequence (no batch dimension).
            players: :class:`~motionbench.players.base.PlayerSet` defining
                the M players and coordinate-to-player aggregation.
            target: Class index (unused by rollout, kept for API consistency).

        Returns:
            ``(M,)`` float32 per-player attribution tensor.

        Raises:
            AttributeError: If the classifier does not expose
                ``get_attention_weights() -> list[Tensor[(B, H, S, S)]]``.
            ValueError: If the classifier returns no attention weights, or if
                the sequence length ``S`` cannot be mapped back to
                ``(J, F, T)``.
        """
        if not hasattr(self._classifier, "get_attention_weights"):
            raise AttributeError(
                "Classifier must expose get_attention_weights() -> list[Tensor[(B, H, S, S)]]"
            )

        J, F_dim, T = x.shape
        x_in = x.unsqueeze(0)  # (1, J, F, T)

        with torch.no_grad():
            self._classifier(x_in)

        attn_weights: list[Tensor] = self._classifier.get_attention_weights()  # type: ignore[union-attr]

        if not attn_weights:
            raise ValueError(
                "Classifier.get_attention_weights() returned an empty list. "
                "Run a forward pass before calling attribute()."
            )

        # attn_weights[l]: (B, H, S, S)
        S = attn_weights[0].shape[-1]
        device = x.device
        eye = torch.eye(S, device=device)

        # Compute rollout: A_rollout = ∏_l (0.5 * mean_heads(A_l) + 0.5 * I)
        rollout = eye.clone()
        for attn in attn_weights:
            # Average over heads: (B, H, S, S) → (S, S) using first sample
            A_avg = attn[0].mean(dim=0)  # (S, S)
            A_hat = 0.5 * A_avg + 0.5 * eye
            rollout = rollout @ A_hat

        # phi_flat: (S,) — row 0 gives CLS-token's distributed attention
        phi_flat = rollout[0]  # (S,)

        phi_coords = self._reshape_to_input(phi_flat, S, J, F_dim, T)
        return players.aggregate(phi_coords)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reshape_to_input(
        self,
        phi_flat: Tensor,
        S: int,
        J: int,
        F_dim: int,
        T: int,
    ) -> Tensor:
        """Reshape the flat rollout vector ``(S,)`` to ``(J, F, T)``.

        Supported layouts:

        * ``S == J * F * T``: direct reshape (all coords are tokens).
        * ``S == T``: time tokens — broadcast over J and F.
        * ``S == J * F``: joint-feature tokens — broadcast over T.

        Args:
            phi_flat: ``(S,)`` float tensor of per-token attributions.
            S: Sequence length.
            J: Number of joints.
            F_dim: Number of features per joint.
            T: Number of time frames.

        Returns:
            ``(J, F, T)`` contiguous float32 tensor.

        Raises:
            ValueError: If ``S`` does not match any supported layout.
        """
        if J * F_dim * T == S:
            return phi_flat.reshape(J, F_dim, T).contiguous()
        if T == S:
            return phi_flat.unsqueeze(0).unsqueeze(0).expand(J, F_dim, T).clone().contiguous()
        if J * F_dim == S:
            return phi_flat.reshape(J, F_dim).unsqueeze(-1).expand(J, F_dim, T).clone().contiguous()
        raise ValueError(
            f"Cannot map rollout sequence length S={S} to input shape "
            f"(J={J}, F={F_dim}, T={T}). "
            f"Expected S ∈ {{J*F*T={J*F_dim*T}, T={T}, J*F={J*F_dim}}}."
        )
