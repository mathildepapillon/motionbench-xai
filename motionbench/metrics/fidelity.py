"""motionbench.metrics.fidelity — Fidelity metrics with on/off-manifold imputer support.

All four metrics wrap Quantus 0.6.0 faithfulness classes and expose a unified
``evaluate(phi, x, classifier, players, ...)`` interface consistent with
:class:`~motionbench.metrics.base.BaseMetric`.

On-manifold vs off-manifold
----------------------------
Each metric accepts an :class:`~motionbench.imputers.base.BaseImputer` at
construction time via the ``imputer`` argument.  Passing a
:class:`~motionbench.imputers.off_manifold.ZeroImputer` (``perturb_baseline="black"``)
reproduces Quantus default behaviour.  Passing a learned imputer (VAEAC,
Flow, KNN …) enables on-manifold perturbation.

Quantus integration
--------------------
Quantus's ``perturb_func`` signature after ``make_perturb_func`` is::

    perturb_func(arr: np.ndarray, indices: np.ndarray, **kwargs) -> np.ndarray

where ``arr`` is ``(B, n_features)`` flat and ``indices`` is
``(B, n_perturb)`` flat feature indices *to perturb / replace*.  Our
:func:`_make_perturb_func` adapter bridges this to
:meth:`~motionbench.imputers.base.BaseImputer.impute`.

References
----------
Hedström et al. (2023) "Quantus: An Explainability Toolkit for Responsible
Evaluation of Neural Network Explanations." JMLR 24(34).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import quantus
import scipy.stats
import torch
import torch.nn as nn
from torch import Tensor

from motionbench.metrics.base import BaseMetric

if TYPE_CHECKING:
    from collections.abc import Callable

    from motionbench.imputers.base import BaseImputer
    from motionbench.oracles.base import Oracle
    from motionbench.players.base import PlayerSet


__all__ = [
    "FaithfulnessCorrelationMetric",
    "MonotonicityCorrelationMetric",
    "PixelFlippingMetric",
    "SelectivityMetric",
]


# ---------------------------------------------------------------------------
# Compatibility shim for Quantus 0.6.0 spearmanr bug with batch_size=1
# ---------------------------------------------------------------------------


def _spearman_batched(
    a: Any,
    b: Any,
    batched: bool = False,
    **kwargs: object,
) -> Any:
    """Spearman correlation compatible with batch_size=1.

    Quantus 0.6.0 ``correlation_spearman`` fails when ``batched=True`` and
    ``batch_size=1`` because ``scipy.stats.spearmanr(..., axis=1)`` returns a
    ``float`` (no ``.shape`` attribute) instead of an ``ndarray``.  This
    wrapper handles both cases explicitly.
    """
    if batched:
        corr = scipy.stats.spearmanr(a, b, axis=1)[0]
        if not hasattr(corr, "shape"):
            # batch_size=1: scipy returned a scalar, wrap in 1-element array
            return np.array([corr])
        # Multi-sample batch: corr is a correlation matrix; first column is
        # the pairwise correlation between a[i] and b[i].
        return corr[:, 0] if corr.ndim > 1 else corr
    return np.array([scipy.stats.spearmanr(a, b)[0]])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _QuantusModelWrapper(nn.Module):
    """Wraps a motionbench classifier callable as an ``nn.Module`` for Quantus.

    Reduces output to ``(B, 1)`` — a single score per sample at ``target``
    class index — so that Quantus can do ``predict(x)[batch_idx, 0]``
    uniformly regardless of whether the classifier is binary or multi-class.

    The module is always in eval mode.

    Args:
        fn: Classifier callable ``(B, J, F, T) -> (B,)`` or ``(B, n_classes)``.
        target: Class index to select when ``fn`` returns ``(B, n_classes)``.
    """

    def __init__(self, fn: Callable[[Tensor], Tensor], target: int) -> None:
        super().__init__()
        self._fn = fn
        self._target = target
        self.eval()

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Run classifier and return ``(B, 1)`` score tensor.

        Args:
            x: ``(B, J, F, T)`` float32 tensor.

        Returns:
            ``(B, 1)`` float32 tensor with the target-class score.
        """
        out = self._fn(x)  # (B,) or (B, n_classes)
        if out.dim() == 2:
            out = out[:, self._target]  # (B,)
        return out.unsqueeze(-1)  # (B, 1)


def _make_perturb_func(
    imputer: BaseImputer,
    shape: tuple[int, int, int],
) -> Callable[..., Any]:
    """Build a Quantus-compatible ``perturb_func`` that calls ``imputer.impute``.

    The returned function signature matches what Quantus passes after
    ``make_perturb_func``:

        perturb_func(arr, indices, **kwargs) -> arr

    Args:
        imputer: Fitted :class:`~motionbench.imputers.base.BaseImputer`.
        shape: ``(J, F, T)`` original sequence shape used to unflatten.

    Returns:
        Callable ``(arr: ndarray, indices: ndarray, **kwargs) -> ndarray``.
        ``arr`` shape: ``(B, J*F*T)`` flat.
        ``indices`` shape: ``(B, n_perturb)`` flat feature indices to replace.
    """
    J, F, T = shape
    n_features = J * F * T

    def perturb_func(
        arr: Any,
        indices: Any,
        **kwargs: object,
    ) -> Any:
        # arr:     (B, n_features) numpy float
        # indices: (B, n_perturb) numpy int  — features to REPLACE (hidden)
        batch_size = arr.shape[0]
        out = arr.copy()

        for i in range(batch_size):
            x_flat = arr[i]  # (n_features,)
            x_3d = torch.from_numpy(x_flat.astype(np.float32)).reshape(J, F, T)

            # Build mask: True = observed, False = hidden (perturbed)
            mask = torch.ones(n_features, dtype=torch.bool)
            mask[indices[i].astype(int)] = False
            mask_3d = mask.reshape(J, F, T)

            completed = imputer.impute(x_3d, mask_3d, n_samples=1)  # (1, J, F, T)
            out[i] = completed[0].detach().cpu().numpy().reshape(n_features)

        return out

    return perturb_func


def _expand_phi_to_coords(phi: Tensor, players: PlayerSet) -> Tensor:
    """Expand per-player attribution ``(M,)`` to coordinate space ``(J, F, T)``.

    Each coordinate belonging to player k receives the value ``phi[k]``.  This
    is the inverse of :meth:`~motionbench.players.base.PlayerSet.aggregate`.

    Args:
        phi: ``(M,)`` float32 per-player attribution vector.
        players: :class:`~motionbench.players.base.PlayerSet`.

    Returns:
        ``(J, F, T)`` float32 tensor of per-coordinate attribution values.
    """
    J, F, T = players.shape
    phi_coords = torch.zeros(J, F, T, dtype=torch.float32)
    M = players.n_players
    for k in range(M):
        z = torch.zeros(M, dtype=torch.int)
        z[k] = 1
        mask_k = players.coalition_mask(z)  # (J, F, T) bool — True for player k
        phi_coords[mask_k] = phi[k].item()
    return phi_coords


def _run_quantus_metric(
    metric_q: Any,
    classifier: Callable[[Tensor], Tensor],
    x: Tensor,
    phi_coords: Tensor,
    target: int,
) -> list[float]:
    """Prepare numpy arrays and invoke a Quantus faithfulness metric.

    Args:
        metric_q: Instantiated Quantus metric (e.g. ``FaithfulnessCorrelation``).
        classifier: Raw callable ``(B, J, F, T) -> (B,)`` or ``(B, n_classes)``.
        x: ``(J, F, T)`` float32 input sequence.
        phi_coords: ``(J, F, T)`` float32 per-coordinate attribution tensor.
        target: Class index.

    Returns:
        List of floats returned by Quantus (one per sample in the batch; here
        always a length-1 list).
    """
    wrapped_model = _QuantusModelWrapper(classifier, target)

    x_np = x.detach().cpu().numpy().astype(np.float32)[np.newaxis]  # (1, J, F, T)
    y_np = np.array([0], dtype=int)  # y=0 since wrapper returns (B, 1)
    # Pass attributions in same shape as x so Quantus expand_attribution_channel is a no-op.
    # Quantus flattens a_batch internally in evaluate_batch before computing att_sums.
    a_np = phi_coords.detach().cpu().numpy().astype(np.float32)[np.newaxis]  # (1, J, F, T)

    return metric_q(  # type: ignore[no-any-return]
        model=wrapped_model,
        x_batch=x_np,
        y_batch=y_np,
        a_batch=a_np,
        channel_first=True,
    )


# ---------------------------------------------------------------------------
# Metric classes
# ---------------------------------------------------------------------------


class FaithfulnessCorrelationMetric(BaseMetric):
    """Wraps :class:`quantus.FaithfulnessCorrelation`.

    Measures the Pearson correlation between the sum of attribution values for
    randomly sampled subsets and the corresponding prediction deltas after
    masking those subsets.  Higher is better (more faithful).

    Supports on-manifold and off-manifold perturbation via ``imputer``.

    Args:
        imputer: Fitted :class:`~motionbench.imputers.base.BaseImputer`.
        nr_runs: Number of random subset samples per evaluation. Default 100.
        subset_size: Size of each random feature subset. Default 10.
        disable_warnings: Suppress Quantus runtime warnings. Default True.
    """

    requires_oracle: ClassVar[bool] = False
    requires_imputer: ClassVar[bool] = True

    def __init__(
        self,
        imputer: BaseImputer,
        nr_runs: int = 100,
        subset_size: int = 10,
        disable_warnings: bool = True,
    ) -> None:
        self._imputer = imputer
        self._nr_runs = nr_runs
        self._subset_size = subset_size
        self._disable_warnings = disable_warnings

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
        """Evaluate faithfulness correlation for a single sequence.

        Args:
            phi: ``(M,)`` float32 per-player attribution vector.
            x: ``(J, F, T)`` float32 input sequence.
            classifier: Callable ``(B, J, F, T) -> (B,)`` or ``(B, n_classes)``.
            players: :class:`~motionbench.players.base.PlayerSet`.
            target: Class index used to produce ``phi``.
            oracle: Ignored (``requires_oracle = False``).
            imputer: If provided, overrides the imputer passed at ``__init__``.

        Returns:
            ``{"faithfulness_correlation": float}``
        """
        eff_imputer = imputer if imputer is not None else self._imputer
        self._check_deps(oracle, eff_imputer)

        J, F, T = x.shape
        n_features = J * F * T
        subset_size = min(self._subset_size, n_features)

        perturb_fn = _make_perturb_func(eff_imputer, (J, F, T))
        metric_q = quantus.FaithfulnessCorrelation(  # type: ignore[attr-defined]
            nr_runs=self._nr_runs,
            subset_size=subset_size,
            perturb_func=perturb_fn,
            normalise=False,
            abs=False,
            disable_warnings=self._disable_warnings,
            display_progressbar=False,
        )

        phi_coords = _expand_phi_to_coords(phi, players)
        result = _run_quantus_metric(metric_q, classifier, x, phi_coords, target)
        value = float(result[0]) if result else float("nan")
        return {"faithfulness_correlation": value}


class MonotonicityCorrelationMetric(BaseMetric):
    """Wraps :class:`quantus.MonotonicityCorrelation`.

    Measures Spearman correlation between feature attribution rank and the
    monotonicity of prediction changes when features are iteratively removed.
    Higher is better.

    Args:
        imputer: Fitted :class:`~motionbench.imputers.base.BaseImputer`.
        nr_samples: Number of random orderings to test. Default 100.
        features_in_step: Features removed per perturbation step. Default 1.
        disable_warnings: Suppress Quantus runtime warnings. Default True.
    """

    requires_oracle: ClassVar[bool] = False
    requires_imputer: ClassVar[bool] = True

    def __init__(
        self,
        imputer: BaseImputer,
        nr_samples: int = 100,
        features_in_step: int = 1,
        disable_warnings: bool = True,
    ) -> None:
        self._imputer = imputer
        self._nr_samples = nr_samples
        self._features_in_step = features_in_step
        self._disable_warnings = disable_warnings

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
        """Evaluate monotonicity correlation for a single sequence.

        Args:
            phi: ``(M,)`` float32 per-player attribution vector.
            x: ``(J, F, T)`` float32 input sequence.
            classifier: Callable ``(B, J, F, T) -> (B,)`` or ``(B, n_classes)``.
            players: :class:`~motionbench.players.base.PlayerSet`.
            target: Class index used to produce ``phi``.
            oracle: Ignored (``requires_oracle = False``).
            imputer: If provided, overrides the imputer passed at ``__init__``.

        Returns:
            ``{"monotonicity_correlation": float}``
        """
        eff_imputer = imputer if imputer is not None else self._imputer
        self._check_deps(oracle, eff_imputer)

        J, F, T = x.shape
        perturb_fn = _make_perturb_func(eff_imputer, (J, F, T))
        metric_q = quantus.MonotonicityCorrelation(  # type: ignore[attr-defined]
            similarity_func=_spearman_batched,
            nr_samples=self._nr_samples,
            features_in_step=self._features_in_step,
            perturb_func=perturb_fn,
            normalise=False,
            abs=False,
            disable_warnings=self._disable_warnings,
            display_progressbar=False,
        )

        phi_coords = _expand_phi_to_coords(phi, players)
        result = _run_quantus_metric(metric_q, classifier, x, phi_coords, target)
        value = float(result[0]) if result else float("nan")
        return {"monotonicity_correlation": value}


class PixelFlippingMetric(BaseMetric):
    """Deletion curve (PGU) — wraps :class:`quantus.PixelFlipping`.

    Iteratively removes features in descending attribution order and measures
    prediction change.  Returns the AUC of the deletion curve (lower is
    better — small AUC means attributions correctly identify important features).

    Adapted for time-series: treats each ``(j, f, t)`` coordinate as a
    "pixel" and removes them in attribution-ranked order.

    Args:
        imputer: Fitted :class:`~motionbench.imputers.base.BaseImputer`.
        features_in_step: Number of features removed per step. Default 1.
        disable_warnings: Suppress Quantus runtime warnings. Default True.
    """

    requires_oracle: ClassVar[bool] = False
    requires_imputer: ClassVar[bool] = True

    def __init__(
        self,
        imputer: BaseImputer,
        features_in_step: int = 1,
        disable_warnings: bool = True,
    ) -> None:
        self._imputer = imputer
        self._features_in_step = features_in_step
        self._disable_warnings = disable_warnings

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
        """Evaluate deletion-curve AUC for a single sequence.

        Args:
            phi: ``(M,)`` float32 per-player attribution vector.
            x: ``(J, F, T)`` float32 input sequence.
            classifier: Callable ``(B, J, F, T) -> (B,)`` or ``(B, n_classes)``.
            players: :class:`~motionbench.players.base.PlayerSet`.
            target: Class index used to produce ``phi``.
            oracle: Ignored (``requires_oracle = False``).
            imputer: If provided, overrides the imputer passed at ``__init__``.

        Returns:
            ``{"pixel_flipping_auc": float}`` — AUC of the deletion curve.
        """
        eff_imputer = imputer if imputer is not None else self._imputer
        self._check_deps(oracle, eff_imputer)

        J, F, T = x.shape
        perturb_fn = _make_perturb_func(eff_imputer, (J, F, T))
        metric_q = quantus.PixelFlipping(  # type: ignore[attr-defined]
            features_in_step=self._features_in_step,
            perturb_func=perturb_fn,
            normalise=False,
            abs=False,
            return_aggregate=False,
            disable_warnings=self._disable_warnings,
            display_progressbar=False,
        )

        phi_coords = _expand_phi_to_coords(phi, players)
        result = _run_quantus_metric(metric_q, classifier, x, phi_coords, target)
        # result[0] is the deletion curve (list of floats per step)
        # Compute AUC using trapezoidal rule
        curve = np.asarray(result[0]) if result else np.array([])
        if len(curve) > 0:
            n = len(curve)
            auc = float(np.trapz(curve, dx=1.0 / max(n - 1, 1)))
        else:
            auc = float("nan")
        return {"pixel_flipping_auc": auc}


class SelectivityMetric(BaseMetric):
    """Insertion curve (PGI) — wraps :class:`quantus.Selectivity`.

    Iteratively reveals (inserts) features in descending attribution order
    starting from a baseline and measures how quickly the model prediction
    recovers.  Returns the AUC of the insertion curve (higher is better —
    large AUC means the top-attributed features are sufficient to recover
    the prediction).

    Implementation: uses :class:`quantus.Selectivity` (patch-based deletion
    curve) with inverted attributions so that *least* important features are
    removed first, equivalent to inserting most important first.

    Args:
        imputer: Fitted :class:`~motionbench.imputers.base.BaseImputer`.
        patch_size: Patch size for Selectivity. Set to 1 for per-element
            granularity. Default 1.
        disable_warnings: Suppress Quantus runtime warnings. Default True.
    """

    requires_oracle: ClassVar[bool] = False
    requires_imputer: ClassVar[bool] = True

    def __init__(
        self,
        imputer: BaseImputer,
        patch_size: int = 1,
        disable_warnings: bool = True,
    ) -> None:
        self._imputer = imputer
        self._patch_size = patch_size
        self._disable_warnings = disable_warnings

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
        """Evaluate insertion-curve AUC for a single sequence.

        Args:
            phi: ``(M,)`` float32 per-player attribution vector.
            x: ``(J, F, T)`` float32 input sequence.
            classifier: Callable ``(B, J, F, T) -> (B,)`` or ``(B, n_classes)``.
            players: :class:`~motionbench.players.base.PlayerSet`.
            target: Class index used to produce ``phi``.
            oracle: Ignored (``requires_oracle = False``).
            imputer: If provided, overrides the imputer passed at ``__init__``.

        Returns:
            ``{"selectivity_auc": float}`` — AUC of the insertion curve.
        """
        eff_imputer = imputer if imputer is not None else self._imputer
        self._check_deps(oracle, eff_imputer)

        J, F, T = x.shape
        perturb_fn = _make_perturb_func(eff_imputer, (J, F, T))
        metric_q = quantus.Selectivity(  # type: ignore[attr-defined]
            patch_size=self._patch_size,
            perturb_func=perturb_fn,
            normalise=False,
            abs=False,
            return_aggregate=False,
            disable_warnings=self._disable_warnings,
            display_progressbar=False,
        )

        phi_coords = _expand_phi_to_coords(phi, players)
        # Invert attributions: Selectivity removes highest-first, so negating
        # means least-important features are removed first → insertion curve.
        phi_coords_inv = -phi_coords
        result = _run_quantus_metric(metric_q, classifier, x, phi_coords_inv, target)
        curve = np.asarray(result[0]) if result else np.array([])
        if len(curve) > 0:
            n = len(curve)
            auc = float(np.trapz(curve, dx=1.0 / max(n - 1, 1)))
        else:
            auc = float("nan")
        return {"selectivity_auc": auc}
