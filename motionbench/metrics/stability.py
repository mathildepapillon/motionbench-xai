"""motionbench.metrics.stability -- Stability / robustness metrics wrapping Quantus.

Provides three metrics that measure how sensitive attributions are to small
perturbations of the input.  All three delegate to Quantus robustness metrics,
which internally require an ``explain_func`` callable so they can recompute
attributions on perturbed inputs.  The default ``explain_func`` is a simple
input-gradient (d output / d input); it can be overridden by passing
``explain_func=...`` inside ``quantus_kwargs``.

Shape contract
--------------
Input ``x`` is ``(J, F, T)``; spatial dimensions are flattened to
``(J*F, T)`` so Quantus sees a 1-D time-series with ``J*F`` channels.
Attribution ``phi`` is ``(M,)`` per-player and is broadcast back to
``(J, F, T)`` coordinate space using :py:meth:`~PlayerSet.coalition_mask`
before flattening.

Design notes
------------
* The gradient explain_func is a simplification: in production you would
  pass the same attribution method that produced ``phi``.  This is noted in
  ``TASKS.md`` as a trade-off.
* ``RelativeInputStability`` is used for ``LipschitzEstimateMetric`` because
  it is the closest available Lipschitz-oriented metric in Quantus >= 0.5.5.
  ``LocalLipschitzEstimate`` (older class) is also present but
  ``RelativeInputStability`` is the recommended replacement.
* ``_QuantusWrapper.forward`` outputs ``(B, n_classes=2)`` instead of ``(B,)``
  so that Quantus metrics that call ``model.predict().argmax(-1)`` and perform
  2-D index selection (e.g. ``Continuity``) work correctly with a motionbench
  scalar classifier.

References
----------
Yeh, Chih-Kuan et al. (2019) "On the (in)fidelity and sensitivity for
    explanations." NeurIPS.
Montavon, Gregoire et al. (2018) "Methods for interpreting and understanding
    deep neural networks." Digital Signal Processing 73.
Alvarez-Melis, David & Jaakkola, Tommi (2018) "Towards robust interpretability
    with self-explaining neural networks." NeurIPS.
Hedstrom et al. (2023) "Quantus: An Explainability Toolkit for Neural
    Networks." JMLR 24(34).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from quantus.metrics.robustness.continuity import Continuity
from quantus.metrics.robustness.max_sensitivity import MaxSensitivity
from quantus.metrics.robustness.relative_input_stability import RelativeInputStability
from torch import Tensor

from motionbench.metrics.base import BaseMetric

if TYPE_CHECKING:
    from collections.abc import Callable

    from motionbench.imputers.base import BaseImputer
    from motionbench.oracles.base import Oracle
    from motionbench.players.base import PlayerSet


__all__ = [
    "MaxSensitivityMetric",
    "ContinuityMetric",
    "LipschitzEstimateMetric",
]


# ---------------------------------------------------------------------------
# Private utilities shared by this module
# ---------------------------------------------------------------------------


class _QuantusWrapper(nn.Module):
    """Adapt a motionbench ``(B, J, F, T)`` classifier to Quantus ``(B, J*F, T)``.

    When ``classifier`` is an :class:`~torch.nn.Module` it is registered as a
    tracked sub-module so that Quantus MPRT can reach its parameters.

    Output shape is ``(B, n_classes)`` (default n_classes=2) so that Quantus
    metrics that call ``model.predict().argmax(-1)`` and perform 2-D index
    selection work correctly with a motionbench scalar classifier.  Column 0
    holds the real model output; remaining columns are zero.

    Args:
        classifier: Callable ``(B, J, F, T) -> (B,)`` or :class:`~torch.nn.Module`.
        J: Number of joints.
        F: Number of features per joint.
        n_classes: Width of the padded output (default 2).
    """

    def __init__(
        self,
        classifier: Callable[..., Tensor],
        J: int,
        F: int,
        n_classes: int = 2,
        target: int = 0,
    ) -> None:
        super().__init__()
        self._J = J
        self._F = F
        self._n_classes = n_classes
        self._target = target
        if isinstance(classifier, nn.Module):
            # Assignment to nn.Module attribute registers it as a tracked
            # sub-module, enabling Quantus MPRT to enumerate / randomise params.
            self._module_clf: nn.Module | None = classifier
            self._fn_clf: Callable[..., Tensor] | None = None
        else:
            self._module_clf = None
            self._fn_clf = classifier

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass that handles both 3-D and 4-D Quantus input.

        Quantus ``Continuity`` inserts a height dimension H=1, producing
        ``(B, J*F, 1, T)``.  We squeeze it back to ``(B, J*F, T)`` before
        passing to the classifier.

        Args:
            x: ``(B, J*F, T)`` or ``(B, J*F, 1, T)`` float32 tensor.

        Returns:
            ``(B, n_classes)`` float32 tensor; column 0 is the scalar
            classifier output (or the target-class probability for multi-output
            classifiers), remaining columns are zero.
        """
        if x.ndim == 4:
            # Continuity inserts H=1: (B, C, 1, T) -> (B, C, T)
            x = x.squeeze(2)
        B, _D, T = x.shape
        x_4d = x.reshape(B, self._J, self._F, T)
        clf: Callable[..., Tensor] = (
            self._module_clf
            if self._module_clf is not None
            else self._fn_clf  # type: ignore[assignment]
        )
        raw_out = clf(x_4d)  # (B,) or (B, n_classes)
        # Multi-class modules return (B, n_classes); extract target class.
        if raw_out.ndim == 2:
            scalar_out = torch.softmax(raw_out, dim=-1)[:, self._target]
        else:
            scalar_out = raw_out
        out = torch.zeros(
            B, self._n_classes, dtype=scalar_out.dtype, device=scalar_out.device
        )
        out[:, 0] = scalar_out
        return out


def _gradient_explain_func(
    model: nn.Module,
    inputs: npt.NDArray[Any],
    targets: npt.NDArray[Any],
    **kwargs: Any,
) -> npt.NDArray[Any]:
    """Input-gradient attribution (default ``explain_func`` for Quantus metrics).

    Computes ``d(output[:, 0]) / d(inputs)`` as a proxy attribution when
    re-evaluating on perturbed inputs.  Returns an array with the same shape
    as ``inputs`` so Quantus's internal shape assertions pass regardless of
    whether ``inputs`` is 3-D ``(B, C, T)`` or 4-D ``(B, C, H, T)``.

    Args:
        model: PyTorch module accepting ``(B, J*F, T)`` or ``(B, J*F, 1, T)``
            float32 tensors.
        inputs: ``(B, J*F, T)`` or ``(B, J*F, 1, T)`` float32 numpy array.
        targets: ``(B,)`` int numpy array (unused; gradient is always w.r.t.
            the first output column).
        **kwargs: Ignored extra keyword arguments from Quantus internals.

    Returns:
        Gradient array with the same shape as ``inputs``.
    """
    x_t = torch.from_numpy(inputs.copy()).requires_grad_(True)
    out: Tensor = model(x_t)  # (B, n_classes)
    # Gradient of first-class output only (column 0 = real classifier output).
    loss = out[:, 0].sum() if out.ndim == 2 else out.sum()
    loss.backward()  # type: ignore[no-untyped-call]
    grad = x_t.grad
    if grad is None:
        return np.zeros_like(inputs)
    # Return absolute gradient magnitudes: Quantus assertions require
    # attributions to not be all-negative; magnitude is the correct
    # quantity for Max Sensitivity (measures size of change, not sign).
    return np.abs(grad.detach().cpu().numpy())


def _expand_phi(phi: Tensor, players: PlayerSet) -> Tensor:
    """Broadcast per-player attributions to coordinate space ``(J, F, T)``.

    Args:
        phi: ``(M,)`` float32 per-player attribution vector.
        players: :class:`~motionbench.players.base.PlayerSet` that produced ``phi``.

    Returns:
        ``(J, F, T)`` float32 tensor where each coordinate holds the score
        of its owning player.
    """
    J, F, T = players.shape
    M = players.n_players
    expanded = torch.zeros(J, F, T)
    for k in range(M):
        z_k = torch.zeros(M)
        z_k[k] = 1.0
        mask_k: Tensor = players.coalition_mask(z_k)
        expanded[mask_k] = phi[k].item()
    return expanded


def _prepare_quantus_inputs(
    phi: Tensor,
    x: Tensor,
    classifier: Callable[..., Tensor],
    players: PlayerSet,
    target: int,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], _QuantusWrapper]:
    """Prepare numpy arrays and a wrapped model for a Quantus metric call.

    Args:
        phi: ``(M,)`` per-player attribution.
        x: ``(J, F, T)`` input sequence.
        classifier: Callable ``(B, J, F, T) -> (B,)`` or :class:`~torch.nn.Module`.
            Passing a raw ``nn.Module`` is strongly preferred: it enables
            gradient flow (required by :func:`_gradient_explain_func`) and
            parameter enumeration (required by Quantus MPRT).
        players: :class:`~motionbench.players.base.PlayerSet`.
        target: Class / label index.

    Returns:
        Tuple ``(x_batch, a_batch, y_batch, wrapped_model)`` ready for Quantus.
        ``x_batch`` and ``a_batch`` are ``(1, J*F, T)`` float32 arrays;
        ``y_batch`` is ``(1,)`` int array.  The wrapped model is in eval mode.
    """
    J, F, T = players.shape
    x_np = x.detach().cpu().numpy()
    x_batch = x_np.reshape(1, J * F, T).astype(np.float32)
    y_batch = np.array([target])
    phi_coords = _expand_phi(phi.detach().cpu(), players)
    a_batch = phi_coords.numpy().reshape(1, J * F, T).astype(np.float32)
    wrapped = _QuantusWrapper(classifier, J, F, target=target)
    wrapped.eval()
    return x_batch, a_batch, y_batch, wrapped


# ---------------------------------------------------------------------------
# Public metric classes
# ---------------------------------------------------------------------------


class MaxSensitivityMetric(BaseMetric):
    """Max-Sensitivity: maximum attribution change under bounded input noise.

    Measures the maximum ratio ``||A(x+e) - A(x)|| / ||A(x)||`` over
    ``nr_samples`` uniform-noise perturbations.  Lower is better.

    Wraps :class:`quantus.MaxSensitivity` (Yeh et al. 2019).

    Args:
        **quantus_kwargs: Forwarded to ``quantus.MaxSensitivity.__init__``.
            Useful overrides: ``nr_samples`` (default 200),
            ``lower_bound`` (default 0.2), ``upper_bound``.

    Attributes:
        requires_oracle: ``False``.
        requires_imputer: ``False``.
    """

    requires_oracle: ClassVar[bool] = False
    requires_imputer: ClassVar[bool] = False

    def __init__(self, **quantus_kwargs: Any) -> None:
        quantus_kwargs.setdefault("disable_warnings", True)
        self._quantus: MaxSensitivity = MaxSensitivity(**quantus_kwargs)

    def evaluate(
        self,
        phi: Tensor,
        x: Tensor,
        classifier: Callable[[Tensor], Tensor],
        players: PlayerSet,
        target: int = 0,
        oracle: Oracle | None = None,
        imputer: BaseImputer | None = None,
    ) -> dict[str, float]:
        """Evaluate Max-Sensitivity for a single sequence.

        Args:
            phi: ``(M,)`` per-player attribution.
            x: ``(J, F, T)`` input sequence.
            classifier: Callable ``(B, J, F, T) -> (B,)``.
            players: :class:`~motionbench.players.base.PlayerSet` for ``phi``.
            target: Class index (must match the index used to produce ``phi``).
            oracle: Not required; ignored.
            imputer: Not required; ignored.

        Returns:
            ``{"max_sensitivity": float}`` -- lower values indicate more stable
            attributions.
        """
        self._check_deps(oracle, imputer)
        x_batch, a_batch, y_batch, wrapped = _prepare_quantus_inputs(
            phi, x, classifier, players, target
        )
        scores: list[float] = self._quantus(
            model=wrapped,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            explain_func=_gradient_explain_func,
            channel_first=True,
            softmax=False,
        )
        return {"max_sensitivity": float(np.nanmean(scores))}


class ContinuityMetric(BaseMetric):
    """Continuity: attribution smoothness under coordinate-shift perturbations.

    Measures how much attributions change when the input is shifted
    incrementally across coordinates.  Lower is better (more continuous).

    Wraps :class:`quantus.Continuity` (Montavon et al. 2018).

    Args:
        **quantus_kwargs: Forwarded to ``quantus.Continuity.__init__``.
            Useful override: ``nr_steps`` (default 28).

    Attributes:
        requires_oracle: ``False``.
        requires_imputer: ``False``.
    """

    requires_oracle: ClassVar[bool] = False
    requires_imputer: ClassVar[bool] = False

    def __init__(self, **quantus_kwargs: Any) -> None:
        quantus_kwargs.setdefault("disable_warnings", True)
        self._quantus: Continuity = Continuity(**quantus_kwargs)

    def evaluate(
        self,
        phi: Tensor,
        x: Tensor,
        classifier: Callable[[Tensor], Tensor],
        players: PlayerSet,
        target: int = 0,
        oracle: Oracle | None = None,
        imputer: BaseImputer | None = None,
    ) -> dict[str, float]:
        """Evaluate Continuity for a single sequence.

        Args:
            phi: ``(M,)`` per-player attribution.
            x: ``(J, F, T)`` input sequence.
            classifier: Callable ``(B, J, F, T) -> (B,)``.
            players: :class:`~motionbench.players.base.PlayerSet` for ``phi``.
            target: Class index.
            oracle: Not required; ignored.
            imputer: Not required; ignored.

        Returns:
            ``{"continuity": float}`` -- lower values indicate smoother
            attributions.
        """
        self._check_deps(oracle, imputer)
        x_batch, a_batch, y_batch, wrapped = _prepare_quantus_inputs(
            phi, x, classifier, players, target
        )
        scores: list[float] = self._quantus(
            model=wrapped,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            explain_func=_gradient_explain_func,
            channel_first=True,
            softmax=False,
        )
        return {"continuity": float(np.nanmean(scores))}


class LipschitzEstimateMetric(BaseMetric):
    """Lipschitz Estimate: relative input stability as a Lipschitz proxy.

    Approximates the local Lipschitz constant of the attribution map using
    ``RelativeInputStability`` (Alvarez-Melis & Jaakkola 2018 via Quantus).
    Higher stability (lower score) means attributions are more Lipschitz-
    continuous.

    Wraps :class:`quantus.RelativeInputStability`.  (The older
    ``quantus.LocalLipschitzEstimate`` is also present but
    ``RelativeInputStability`` is the recommended replacement in Quantus >= 0.5.)

    Args:
        **quantus_kwargs: Forwarded to ``quantus.RelativeInputStability.__init__``.
            Useful override: ``nr_samples`` (default 200).

    Attributes:
        requires_oracle: ``False``.
        requires_imputer: ``False``.
    """

    requires_oracle: ClassVar[bool] = False
    requires_imputer: ClassVar[bool] = False

    def __init__(self, **quantus_kwargs: Any) -> None:
        quantus_kwargs.setdefault("disable_warnings", True)
        self._quantus: RelativeInputStability = RelativeInputStability(**quantus_kwargs)

    def evaluate(
        self,
        phi: Tensor,
        x: Tensor,
        classifier: Callable[[Tensor], Tensor],
        players: PlayerSet,
        target: int = 0,
        oracle: Oracle | None = None,
        imputer: BaseImputer | None = None,
    ) -> dict[str, float]:
        """Evaluate Lipschitz Estimate for a single sequence.

        Args:
            phi: ``(M,)`` per-player attribution.
            x: ``(J, F, T)`` input sequence.
            classifier: Callable ``(B, J, F, T) -> (B,)``.
            players: :class:`~motionbench.players.base.PlayerSet` for ``phi``.
            target: Class index.
            oracle: Not required; ignored.
            imputer: Not required; ignored.

        Returns:
            ``{"lipschitz_estimate": float}`` -- lower values indicate a more
            Lipschitz-stable attribution map.
        """
        self._check_deps(oracle, imputer)
        x_batch, a_batch, y_batch, wrapped = _prepare_quantus_inputs(
            phi, x, classifier, players, target
        )
        scores: list[float] = self._quantus(
            model=wrapped,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            explain_func=_gradient_explain_func,
            channel_first=True,
            softmax=False,
        )
        return {"lipschitz_estimate": float(np.nanmean(scores))}
