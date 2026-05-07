"""motionbench.attribution.timeshap_real — TimeSHAP wrapper via the timeshap pip package.

Wraps :func:`timeshap.explainer.local_event` (Bento et al., 2021) to produce
per-temporal-window attributions for spatiotemporal sequences.  The package
operates at *per-timestep* granularity with a pruning heuristic; we aggregate
the per-timestep Shapley values back to the K=T/window_len uniform-window
player set so the resulting attribution row is directly comparable with the
KernelSHAP variants and ``WindowSHAP-Stationary/Dynamic``.

Naming
------
This module exposes :class:`RealTimeSHAPAttributor` to disambiguate from the
pre-existing backward-compat alias
``motionbench.attribution.kernelshap_temporal.TimeSHAPAttributor`` (which is
itself an alias for ``KernelSHAPTemporalAttributor`` --- our hand-rolled
KernelSHAP-over-temporal-windows surrogate, kept for old result-file
deserialisation).  The ``Real`` prefix marks this wrapper as the actual
``timeshap`` pip-package implementation.

References
----------
Bento, J., Saleiro, P., Cruz, A. F., Figueiredo, M. A. T., & Bizarro, P.
(2021).  TimeSHAP: Explaining Recurrent Models through Sequence
Perturbations.  KDD 2021.  ``pip install timeshap``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from motionbench.attribution.base import BaseAttributor

if TYPE_CHECKING:
    from motionbench.players.base import PlayerSet


__all__ = ["RealTimeSHAPAttributor"]


def _resolve_classifier_device(
    classifier: Callable[[Tensor], Tensor],
    x_input: Tensor,
) -> torch.device:
    """Find the device of *classifier* (Module or softmax-wrapping closure).

    Falls back to ``x_input.device`` when the classifier is a plain
    callable with no introspectable parameters.
    """
    if hasattr(classifier, "parameters"):
        try:
            return next(classifier.parameters()).device  # type: ignore[union-attr]
        except (StopIteration, AttributeError):
            pass
    if getattr(classifier, "__closure__", None):
        for cell in classifier.__closure__:  # type: ignore[union-attr]
            obj = cell.cell_contents
            if hasattr(obj, "parameters"):
                try:
                    return next(obj.parameters()).device
                except StopIteration:
                    continue
    return x_input.device


class _TimeSHAPClassifierAdapter:
    """Adapt a torch ``(B, J, F, T)`` classifier to a numpy ``(B, T, J*F)`` predict.

    ``timeshap.explainer.local_event`` calls a user-supplied ``f(x)`` where
    ``x`` is shaped ``(B, T, F_flat)`` (its sequence-major convention).  This
    adapter reshapes back to the MotionBench ``(B, J, F, T)`` layout, runs the
    classifier on GPU, and returns the scalar probability of the target class.
    """

    def __init__(
        self,
        classifier: Callable[[Tensor], Tensor],
        J: int,
        F_coords: int,
        T: int,
        target: int,
        device: torch.device,
    ) -> None:
        self._classifier = classifier
        self._J = J
        self._F = F_coords
        self._T = T
        self._target = int(target)
        self._device = device

    def __call__(self, x_btf: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        if x_btf.ndim == 2:  # (T, F_flat)
            x_btf = x_btf[np.newaxis]
        B = x_btf.shape[0]
        x_jft = (
            x_btf.reshape(B, self._T, self._J, self._F)
            .transpose(0, 2, 3, 1)  # (B, J, F, T)
            .astype(np.float32, copy=False)
        )
        x_t = torch.from_numpy(x_jft).to(self._device)
        with torch.no_grad():
            probs = self._classifier(x_t)
        probs_np = probs.detach().cpu().numpy()
        if probs_np.ndim == 1:
            return probs_np.reshape(B, 1)
        return probs_np[:, self._target : self._target + 1]


class RealTimeSHAPAttributor(BaseAttributor):
    """Attribution via :func:`timeshap.explainer.local_event` (pip package).

    Args:
        classifier: ``(B, J, F, T) → (B, n_classes)`` callable returning
            softmax probabilities.  When constructed via the standard
            pipeline, ``_build_attributor`` wraps the raw classifier in a
            softmax so ``v(S) = softmax(f(x))[target]`` matches the oracle.
        window_len: Aggregation window length in time steps; ``T`` must be
            a multiple.  The pip-package call itself is per-timestep --- we
            sum the per-timestep Shapley values within each uniform window
            of length ``window_len`` to recover a K-vector that is
            commensurate with the temporal-window player set used by the
            other rows of Table~\\ref{tab:synth_ec1}.
        nsamples: KernelSHAP coalition budget passed to ``local_event``.
        seed: Random seed forwarded to ``local_event`` (``rs`` argument).
    """

    def __init__(
        self,
        classifier: Callable[[Tensor], Tensor],
        window_len: int = 4,
        nsamples: int = 32,
        seed: int = 42,
    ) -> None:
        super().__init__(classifier)
        self._window_len = int(window_len)
        self._nsamples = int(nsamples)
        self._seed = int(seed)

    @property
    def name(self) -> str:  # noqa: D401
        """Short identifier for logging."""
        return "TimeSHAP"

    def attribute(
        self,
        x: Tensor,
        players: PlayerSet,
        target: int = 0,
    ) -> Tensor:
        from timeshap.explainer import local_event  # noqa: PLC0415

        J, F_coords, T = x.shape
        if self._window_len >= T:
            raise ValueError(
                f"window_len={self._window_len} must be less than T={T}.",
            )
        if T % self._window_len != 0:
            raise ValueError(
                f"window_len={self._window_len} must divide T={T} evenly.",
            )
        K = T // self._window_len
        F_total = J * F_coords
        win = self._window_len

        # (J, F, T) → (1, T, J*F)
        x_np = (
            x.detach().cpu().permute(2, 0, 1).reshape(T, F_total).numpy()
        )[np.newaxis].astype(np.float32)
        baseline = np.zeros_like(x_np)  # (1, T, F_total) all-zeros baseline

        device = _resolve_classifier_device(self._classifier, x)
        adapter = _TimeSHAPClassifierAdapter(
            self._classifier, J, F_coords, T, target, device,
        )

        df = local_event(
            f=adapter,
            data=x_np,
            event_dict={"rs": self._seed, "nsamples": self._nsamples},
            entity_uuid="0",
            entity_col="entity",
            baseline=baseline,
            pruned_idx=0,
        )

        # ``local_event`` returns a DataFrame with one row per timestep.  As
        # of ``timeshap`` 1.0.4 the event-index column is named ``Feature``
        # and contains strings of the form ``"Event -i"``, where ``-i``
        # uses Python negative-indexing semantics (``-1`` is the most recent
        # event, ``-T`` is the oldest).  Older releases used ``"t"`` with
        # an integer index.  Both are handled below; rows whose index does
        # not parse to a valid timestep (e.g. "Pruned Events" group rows)
        # are dropped, since they have no per-timestep placement.
        phi_t = np.zeros(T, dtype=np.float32)
        idx_col = "Feature" if "Feature" in df.columns else "t"
        for _, row in df.iterrows():
            raw = row[idx_col]
            try:
                t_idx = int(raw)
            except (TypeError, ValueError):
                # Try "Event -i" -> Python negative indexing -> 0..T-1.
                if isinstance(raw, str) and "Event" in raw:
                    try:
                        signed = int(raw.split()[-1])
                    except (TypeError, ValueError, IndexError):
                        continue
                    if signed < 0:
                        t_idx = T + signed
                    else:
                        t_idx = signed
                else:
                    continue
            if not (0 <= t_idx < T):
                continue
            phi_t[t_idx] = float(row["Shapley Value"])

        # Numerical-pathology guard: the underlying ``shap.KernelExplainer``
        # solves a weighted least-squares system that can become ill-conditioned
        # when the value function is locally degenerate over a coalition slice
        # (all sampled coalitions yield identical predictions).  This produces
        # cancelling pairs of huge-magnitude attributions (e.g. ``+1e13`` and
        # ``-1e13``) that sum to near-zero but corrupt EC1 / EC2.  Because the
        # value function here is a softmax probability bounded in [0, 1], any
        # individual Shapley value with ``|phi| > 2`` is provably an artifact
        # (the maximum |phi| is bounded by the diameter of the value range,
        # i.e. ``v(N) - v(emptyset)`` ≤ 1, plus a Shapley-axiom slack of 1).
        # We replace such entries with zero rather than NaN so per-sequence
        # aggregation remains well defined and the affected sequence simply
        # drops out of the contribution to EC3 (Pearson correlation handles
        # zeros benignly).
        bad = ~np.isfinite(phi_t) | (np.abs(phi_t) > 2.0)
        if bad.any():
            phi_t = phi_t.copy()
            phi_t[bad] = 0.0

        # Aggregate per-timestep to per-window (K,), spread uniformly across
        # joints/coords within each window so the (J, F, T) tensor satisfies
        # sum_{t in window k} phi[j, f, t] == phi_window_k for each (j, f).
        per_step_per_coord = (phi_t / float(F_total))[:, np.newaxis]  # (T, 1)
        per_step_full = np.broadcast_to(
            per_step_per_coord, (T, F_total),
        ).copy()  # (T, F_total)
        phi_3d = per_step_full.reshape(T, J, F_coords)
        phi_coords = torch.as_tensor(phi_3d, dtype=torch.float32).permute(1, 2, 0)
        return players.aggregate(phi_coords)
