"""motionbench.metrics.sanity_checks -- Sanity-check metrics wrapping Quantus.

Implements two sanity-check metrics that validate whether an attribution
method is actually sensitive to model parameters and not merely computing
input statistics:

* ``ModelParameterRandomisationMetric`` (Adebayo et al. 2018) -- cascadingly
  randomises model layers and measures how much attributions change (Spearman
  correlation by default).  A good attribution method should yield low
  correlation with the original after full randomisation.

* ``RandomLogitMetric`` -- replaces the prediction target with a random
  (incorrect) logit and measures attribution similarity.  Attributions that
  do not change should be considered unreliable.

Both metrics require an ``explain_func`` to recompute attributions for the
modified model / target.  The default is input-gradient (d output / d input),
which can be overridden via ``quantus_kwargs``.

Shape contract
--------------
Identical to :mod:`motionbench.metrics.stability`: input ``x`` is ``(J, F, T)``,
flattened to ``(1, J*F, T)`` for Quantus; ``phi`` ``(M,)`` is broadcast to
``(J*F, T)`` via :py:meth:`~PlayerSet.coalition_mask`.

References
----------
Adebayo, Julius et al. (2018) "Sanity Checks for Saliency Maps." NeurIPS.
Hedstrom et al. (2023) "Quantus: An Explainability Toolkit for Neural
    Networks." JMLR 24(34).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from quantus.metrics.randomisation.mprt import MPRT
from quantus.metrics.randomisation.random_logit import RandomLogit
from torch import Tensor

from motionbench.metrics.base import BaseMetric

if TYPE_CHECKING:
    from collections.abc import Callable

    from motionbench.imputers.base import BaseImputer
    from motionbench.oracles.base import Oracle
    from motionbench.players.base import PlayerSet


__all__ = [
    "ModelParameterRandomisationMetric",
    "RandomLogitMetric",
]


# ---------------------------------------------------------------------------
# Private utilities (duplicated from stability.py per AGENTS.md sec.1.4)
# ---------------------------------------------------------------------------


class _QuantusWrapper(nn.Module):
    """Adapt a motionbench ``(B, J, F, T)`` classifier to Quantus ``(B, J*F, T)``.

    When ``classifier`` is an :class:`~torch.nn.Module` it is registered as a
    tracked sub-module so that Quantus MPRT can reach and randomise its
    parameters.

    Output shape is ``(B, n_classes)`` (default n_classes=2) so that Quantus
    metrics calling ``model.predict().argmax(-1)`` and 2-D index selection
    work correctly with a motionbench scalar classifier.

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
            self._module_clf: nn.Module | None = classifier
            self._fn_clf: Callable[..., Tensor] | None = None
        else:
            self._module_clf = None
            self._fn_clf = classifier

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass handling 3-D and 4-D Quantus input.

        Quantus ``Continuity`` inserts a height dimension H=1 giving
        ``(B, J*F, 1, T)``; we squeeze it back to ``(B, J*F, T)``.

        Args:
            x: ``(B, J*F, T)`` or ``(B, J*F, 1, T)`` float32 tensor.

        Returns:
            ``(B, n_classes)`` float32 tensor; column 0 is the scalar
            classifier output (or the target-class probability for
            multi-output classifiers), remaining columns are zero.
        """
        if x.ndim == 4:
            x = x.squeeze(2)
        B, _D, T = x.shape
        x_4d = x.reshape(B, self._J, self._F, T)
        clf: Callable[..., Tensor] = (
            self._module_clf
            if self._module_clf is not None
            else self._fn_clf  # type: ignore[assignment]
        )
        raw_out = clf(x_4d)  # (B,) or (B, n_classes)
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

    Args:
        model: PyTorch module accepting ``(B, J*F, T)`` or ``(B, J*F, 1, T)``
            float32 tensors.
        inputs: ``(B, J*F, T)`` or ``(B, J*F, 1, T)`` float32 numpy array.
        targets: ``(B,)`` int numpy array (unused).
        **kwargs: Ignored.

    Returns:
        Gradient array with the same shape as ``inputs``.
    """
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    x_t = torch.from_numpy(inputs.copy()).to(model_device).requires_grad_(True)
    out: Tensor = model(x_t)  # (B, n_classes)
    loss = out[:, 0].sum() if out.ndim == 2 else out.sum()
    loss.backward()  # type: ignore[no-untyped-call]
    grad = x_t.grad
    if grad is None:
        return np.zeros_like(inputs)
    return np.abs(grad.detach().cpu().numpy())


def _expand_phi(phi: Tensor, players: PlayerSet) -> Tensor:
    """Broadcast per-player attributions to coordinate space ``(J, F, T)``.

    Args:
        phi: ``(M,)`` float32 per-player attribution vector.
        players: :class:`~motionbench.players.base.PlayerSet`.

    Returns:
        ``(J, F, T)`` float32 tensor.
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
        players: :class:`~motionbench.players.base.PlayerSet`.
        target: Class / label index.

    Returns:
        ``(x_batch, a_batch, y_batch, wrapped_model)`` -- all ready for Quantus.
        The wrapped model is in eval mode.
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


class ModelParameterRandomisationMetric(BaseMetric):
    """Model Parameter Randomisation Test (MPRT) by Adebayo et al. (2018).

    Cascadingly randomises the model's layers (top-down) and measures the
    Spearman rank correlation between the original attribution and the
    attribution re-computed on each progressively randomised model.  A
    *lower* average correlation indicates that the attribution method is
    sensitive to model parameters, which is the desired sanity property.

    Wraps :class:`quantus.MPRT` (``quantus.ModelParameterRandomisation`` is
    a deprecated alias for the same class).

    Args:
        **quantus_kwargs: Forwarded to ``quantus.MPRT.__init__``.  The default
            ``return_average_correlation=True`` is set so that the metric
            returns a single float per sample (average across layers) rather
            than a per-layer dict.  Override with
            ``return_average_correlation=False`` to get the full layer-wise
            dict (the evaluate() method will then return the mean over all
            layer keys).

    Attributes:
        requires_oracle: ``False``.
        requires_imputer: ``False``.

    Notes:
        The classifier passed to ``evaluate`` **must** be a
        :class:`~torch.nn.Module` so that Quantus can randomise its
        parameters.  Passing a plain Python callable will cause Quantus to
        raise an error when it tries to iterate over model layers.
    """

    requires_oracle: ClassVar[bool] = False
    requires_imputer: ClassVar[bool] = False

    def __init__(self, **quantus_kwargs: Any) -> None:
        quantus_kwargs.setdefault("disable_warnings", True)
        quantus_kwargs.setdefault("return_average_correlation", True)
        self._quantus: MPRT = MPRT(**quantus_kwargs)

    def evaluate(
        self,
        phi: Tensor,
        x: Tensor,
        classifier: Callable[[Tensor], Tensor],
        players: PlayerSet,
        target: int = 0,
        oracle: Oracle | None = None,
        imputer: BaseImputer | None = None,
        explain_func: Callable[..., Any] | None = None,
    ) -> dict[str, float]:
        """Evaluate MPRT sanity check for a single sequence.

        Args:
            phi: ``(M,)`` per-player attribution.
            x: ``(J, F, T)`` input sequence.
            classifier: :class:`~torch.nn.Module` with signature
                ``(B, J, F, T) -> (B,)``.  Must be an ``nn.Module`` for
                Quantus layer randomisation to work.
            players: :class:`~motionbench.players.base.PlayerSet` for ``phi``.
            target: Class index.
            oracle: Not required; ignored.
            imputer: Not required; ignored.
            explain_func: Method-specific Quantus-compatible re-attribution
                function ``(model, inputs, targets, **kw) -> np.ndarray``.
                Must be provided; raises ``ValueError`` if ``None``.  Pass the
                result of ``attributor.build_quantus_explain_func(players,
                target, device)`` to ensure correct method-specific behaviour.

        Returns:
            ``{"mprt_avg_correlation": float}`` -- average Spearman correlation
            across all layers and samples.  Lower values indicate that the
            attribution reacts appropriately to model randomisation.

        Raises:
            ValueError: if ``explain_func`` is ``None``.  MPRT must re-compute
                attributions using the same method; a gradient proxy would
                measure gradient sanity regardless of the actual method.
        """
        if explain_func is None:
            raise ValueError(
                "ModelParameterRandomisationMetric requires a method-specific "
                "explain_func. Pass attributor.build_quantus_explain_func(players, "
                "target, device). If the attributor returns None (e.g. KernelSHAP), "
                "skip this metric."
            )
        self._check_deps(oracle, imputer)
        x_batch, a_batch, y_batch, wrapped = _prepare_quantus_inputs(
            phi, x, classifier, players, target
        )
        raw: Any = self._quantus(
            model=wrapped,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            explain_func=explain_func,
            channel_first=True,
            softmax=False,
        )
        if isinstance(raw, list):
            score = float(np.nanmean(raw))
        else:
            all_vals: list[float] = []
            for vals in raw.values():
                all_vals.extend(vals)
            score = float(np.nanmean(all_vals))
        return {"mprt_avg_correlation": score}


class RandomLogitMetric(BaseMetric):
    """Random Logit sanity check (Adebayo et al. 2018).

    Replaces the prediction target with a randomly sampled incorrect class
    index and measures the similarity between the original attribution and
    the re-computed attribution.  Attributions that are insensitive to the
    target logit (high similarity score) should be considered unreliable.

    Wraps :class:`quantus.RandomLogit`.

    Args:
        **quantus_kwargs: Forwarded to ``quantus.RandomLogit.__init__``.
            Useful overrides: ``num_classes`` (default 1000), ``seed``
            (default 42), ``similarity_func``.

    Attributes:
        requires_oracle: ``False``.
        requires_imputer: ``False``.
    """

    requires_oracle: ClassVar[bool] = False
    requires_imputer: ClassVar[bool] = False

    def __init__(self, **quantus_kwargs: Any) -> None:
        quantus_kwargs.setdefault("disable_warnings", True)
        self._quantus: RandomLogit = RandomLogit(**quantus_kwargs)

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
        """Evaluate Random Logit sanity check for a single sequence.

        Args:
            phi: ``(M,)`` per-player attribution.
            x: ``(J, F, T)`` input sequence.
            classifier: Callable ``(B, J, F, T) -> (B,)``.
            players: :class:`~motionbench.players.base.PlayerSet` for ``phi``.
            target: Class index.
            oracle: Not required; ignored.
            imputer: Not required; ignored.

        Returns:
            ``{"random_logit": float}`` -- Spearman correlation between
            original and random-target attributions.  Lower values indicate
            that the attribution method correctly depends on the prediction
            target.

        Raises:
            ValueError: if dependency requirements are not met.
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
        return {"random_logit": float(np.nanmean(scores))}
