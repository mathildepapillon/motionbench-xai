"""motionbench.attribution.windowshap — WindowSHAP wrapper via windowshap library.

This module wraps :class:`windowshap.windowshap.SlidingWindowSHAP` to produce
per-player attribution scores for motion-capture sequences.

Shape mapping
-------------
The ``windowshap`` library operates on ``(N, T, F)``-shaped 3D numpy arrays,
where ``N`` is the number of samples, ``T`` is the number of time steps, and
``F`` is the number of features.  MotionBench sequences are ``(J, F, T)``
Tensors (joints × coordinates × frames).  This wrapper:

1. Permutes the input from ``(J, F_coords, T)`` to ``(T, J * F_coords)`` and
   adds a batch dimension, yielding ``(1, T, J * F_coords)``.
2. Uses all-zeros as background data, matching the zero-imputation baseline.
3. Wraps the classifier in a lightweight adapter that accepts
   ``(N, T, J * F_coords)`` NumPy arrays and returns ``(N,)`` NumPy arrays.
4. Runs ``SlidingWindowSHAP.shap_values()`` to obtain per-time-step,
   per-feature SHAP values of shape ``(1, T, J * F_coords)``.
5. Reshapes to ``(J, F_coords, T)`` and calls ``players.aggregate`` to
   produce the final ``(M,)`` attribution vector.

References
----------
Mosca & Maestre-Torreblanca (2022) "SHAP-based Explanation Methods: A Review
for NLP Interpretability." (WindowSHAP extended from SHAP KernelExplainer.)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from windowshap.windowshap import (  # type: ignore[import-untyped]
    DynamicWindowSHAP,
    SlidingWindowSHAP,
    StationaryWindowSHAP,
)

from motionbench.attribution.base import BaseAttributor

if TYPE_CHECKING:
    from motionbench.players.base import PlayerSet


__all__ = [
    "WindowSHAPAttributor",
    "StationaryWindowSHAPAttributor",
    "DynamicWindowSHAPAttributor",
]


class _Compat049SlidingWindowSHAP(SlidingWindowSHAP):
    """Subclass of SlidingWindowSHAP that fixes SHAP ≥0.46 output-format incompatibility.

    ``windowshap`` was written for the old ``shap.KernelExplainer.shap_values``
    return convention ``(C, N, F)``.  SHAP ≥0.46 returns ``(N, F)`` for scalar
    models and ``(N, F, C)`` for multi-output models.  This subclass overrides
    ``shap_values`` to normalise the array to ``(1, N, F)`` before WindowSHAP's
    reshape logic runs.
    """

    def shap_values(  # type: ignore[override]
        self,
        num_output: int = 1,
        nsamples: int | str = "auto",
    ) -> npt.NDArray[np.float32]:
        import shap as _shap  # noqa: PLC0415

        seq_len: int = self.background_ts.shape[1]
        num_sw: int = int(np.ceil((seq_len - self.window_len) / self.stride)) + 1
        ts_phi = np.zeros(
            (self.num_test, num_sw, 2, self.background_ts.shape[2]), dtype=np.float64
        )
        dem_phi = np.zeros((self.num_test, num_sw, self.num_dem_ftr), dtype=np.float64)

        if nsamples == "auto":
            nsamples = 10 * self.num_ts_ftr + 5 * self.num_dem_ftr

        for stride_cnt in range(num_sw):
            _start = stride_cnt * self.stride

            def _predict(x: npt.NDArray[np.float32], _s: int = _start) -> npt.NDArray[np.float32]:
                return self.wraper_predict(x, start_ind=_s)  # type: ignore[return-value]

            self.explainer = _shap.KernelExplainer(_predict, self.background_data)
            sv = self.explainer.shap_values(self.test_data, nsamples=nsamples)
            sv_np = np.array(sv, dtype=np.float64)

            # Normalise to (1, N, F) regardless of SHAP version:
            #   old SHAP: returns list[(N,F)] → np.array = (1,N,F)  ✓
            #   new SHAP scalar: returns (N,F) (2D) → add leading dim
            #   new SHAP multi:  returns (N,F,C) → transpose to (C,N,F)
            if sv_np.ndim == 2:
                sv_np = sv_np[np.newaxis]                    # (1, N, F)
            elif sv_np.ndim == 3 and sv_np.shape[-1] < sv_np.shape[-2]:
                sv_np = sv_np.transpose(2, 0, 1)             # (N,F,C) → (C,N,F)

            dem_sv = sv_np[:, :, : self.num_dem_ftr]
            ts_sv = sv_np[:, :, self.num_dem_ftr :]
            ts_sv = ts_sv.reshape((num_output, self.num_test, 2, self.num_ts_ftr))

            ts_phi[:, stride_cnt, :, :] = ts_sv[0]
            if self.num_dem_ftr > 0:
                dem_phi[:, stride_cnt, :] = dem_sv[0]

        ts_phi_agg = np.full(
            (self.num_test, num_sw, self.num_ts_step, self.num_ts_ftr), np.nan
        )
        for k in range(num_sw):
            ts_phi_agg[
                :, k, k * self.stride : k * self.stride + self.window_len, :
            ] = ts_phi[:, k, 0, :][:, np.newaxis, :]
        ts_phi_agg = np.nanmean(ts_phi_agg, axis=1).astype(np.float32)
        dem_phi = np.nanmean(dem_phi, axis=1).astype(np.float32)

        self.dem_phi = dem_phi
        self.ts_phi = ts_phi_agg
        return ts_phi_agg if self.num_dem_ftr == 0 else (dem_phi, ts_phi_agg)


class _ClassifierAdapter:
    """Thin adapter exposing a ``.predict()`` method for windowshap.

    The ``windowshap`` library calls ``model.predict(ts_x)`` where ``ts_x``
    is a ``(N, T, F_flat)`` NumPy array (``model_type='lstm'`` branch).
    This adapter reshapes the input to ``(N, J, F_coords, T)`` and calls the
    MotionBench classifier.

    Args:
        classifier: ``(B, J, F_coords, T) float32 → (B,) float32`` callable.
        J: Number of skeletal joints.
        F_coords: Number of coordinates per joint.
        T: Number of time frames.
        target: Class index to select when output is multi-dimensional.
    """

    def __init__(
        self,
        classifier: Callable[[Tensor], Tensor],
        J: int,
        F_coords: int,
        T: int,
        target: int,
    ) -> None:
        self._classifier = classifier
        self._J = J
        self._F = F_coords
        self._T = T
        self._target = target

    def _classifier_device(self) -> torch.device:
        """Return the device of the wrapped classifier (or CPU if not a Module)."""
        cls = self._classifier
        # The pipeline wraps the raw classifier in a softmax closure, so we
        # need to find the underlying nn.Module to read its device.  The
        # closure has a ``__closure__`` cell containing the classifier; if
        # we can't introspect it, fall back to CPU.
        if hasattr(cls, "parameters"):
            try:
                return next(cls.parameters()).device  # type: ignore[union-attr]
            except (StopIteration, AttributeError):
                return torch.device("cpu")
        if getattr(cls, "__closure__", None):
            for cell in cls.__closure__:  # type: ignore[union-attr]
                obj = cell.cell_contents
                if hasattr(obj, "parameters"):
                    try:
                        return next(obj.parameters()).device
                    except StopIteration:
                        continue
        return torch.device("cpu")

    def predict(self, ts_x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Run the classifier on a batch of flattened time series.

        Args:
            ts_x: ``(N, T, J * F_coords)`` float32 numpy array.

        Returns:
            ``(N,)`` float32 numpy array of scalar predictions.
        """
        N = ts_x.shape[0]
        device = self._classifier_device()
        x_tensor = torch.as_tensor(ts_x, dtype=torch.float32, device=device)
        # (N, T, J*F) → (N, T, J, F) → (N, J, F, T)
        x_tensor = x_tensor.reshape(N, self._T, self._J, self._F)
        x_tensor = x_tensor.permute(0, 2, 3, 1).contiguous()
        with torch.no_grad():
            out = self._classifier(x_tensor)
        if out.ndim > 1:
            out = out[:, self._target]
        return out.detach().cpu().numpy().astype(np.float32)  # type: ignore[return-value]


class WindowSHAPAttributor(BaseAttributor):
    """Attribution via SlidingWindowSHAP from the ``windowshap`` library.

    Wraps :class:`windowshap.windowshap.SlidingWindowSHAP` to compute
    temporally-structured SHAP values for motion-capture sequences.

    The library uses a sliding-window kernel approach: it explains each
    overlapping time window of length ``window_len`` via KernelSHAP, then
    aggregates the per-window results across time by averaging.  The final
    per-time-step, per-feature SHAP map is reshaped to ``(J, F, T)`` and
    summarised to ``(M,)`` via ``players.aggregate``.

    Args:
        classifier: ``(B, J, F, T) float32 → (B,) float32`` callable.
        window_len: Length of each sliding window in time steps. Must be
            less than ``T``. Defaults to 8.
        stride: Step size of the sliding window. Defaults to ``window_len``
            (non-overlapping windows).
        seed: Optional random seed forwarded to numpy before fitting.
            Defaults to ``None``.
    """

    def __init__(
        self,
        classifier: Callable[[Tensor], Tensor],
        window_len: int = 8,
        stride: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(classifier)
        self._window_len = window_len
        self._stride = stride  # resolved to window_len at attribute time if None
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
        """Compute per-player SHAP values using SlidingWindowSHAP.

        Args:
            x: ``(J, F, T)`` float32 input sequence.
            players: :class:`~motionbench.players.base.PlayerSet` with M players.
            target: Class index for attribution.

        Returns:
            ``(M,)`` float32 Tensor of per-player attribution scores.

        Raises:
            ValueError: if ``window_len >= T``.
        """
        J, F_coords, T = x.shape
        if self._window_len >= T:
            raise ValueError(
                f"window_len={self._window_len} must be less than T={T}."
            )

        stride = self._stride if self._stride is not None else self._window_len

        if self._seed is not None:
            np.random.seed(self._seed)

        # Reshape (J, F, T) → (1, T, J*F) for the windowshap library.
        x_np = x.detach().cpu().permute(2, 0, 1).reshape(T, J * F_coords).numpy()  # (T, J*F)
        x_np = x_np[np.newaxis].astype(np.float32)  # (1, T, J*F)

        # All-zeros background (one sample).
        bg_np = np.zeros_like(x_np)  # (1, T, J*F)

        model_adapter = _ClassifierAdapter(
            self._classifier, J, F_coords, T, target
        )

        explainer = _Compat049SlidingWindowSHAP(
            model=model_adapter,
            stride=stride,
            window_len=self._window_len,
            B_ts=bg_np,
            test_ts=x_np,
            model_type="lstm",
        )

        # ts_phi_agg shape: (1, T, J*F)
        ts_phi_agg: npt.NDArray[np.float32] = explainer.shap_values()

        # Reshape to (J, F, T): (1, T, J*F) → (T, J, F) → (J, F, T)
        phi_flat = ts_phi_agg[0]  # (T, J*F)
        phi_3d = phi_flat.reshape(T, J, F_coords)  # (T, J, F)
        phi_coords = torch.as_tensor(phi_3d, dtype=torch.float32).permute(1, 2, 0)

        return players.aggregate(phi_coords)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Short identifier for logging."""
        return "WindowSHAP"


class _UniformWindowAttributor(BaseAttributor):
    """Shared logic for K=T/window_len uniform-window WindowSHAP variants.

    Subclasses construct a per-sequence explainer (``StationaryWindowSHAP``
    or ``DynamicWindowSHAP``) and return ``(K, F_total)`` per-window SHAP
    values; this base distributes them uniformly across the timesteps in
    each window to recover a ``(J, F_coords, T)`` attribution map and then
    aggregates with ``players.aggregate``.

    The aggregation choice (uniform within a window) preserves
    :math:`\\sum_{t=t_k}^{t_{k+1}-1} \\phi_{j,f,t} = \\phi^{(\\mathrm{win})}_{k,j,f}`,
    so for a temporal-window ``PlayerSet`` whose window boundaries match
    those used by WindowSHAP the per-player attribution is identical.

    Args:
        classifier: ``(B, J, F, T) → (B, n_classes)`` callable returning
            softmax probabilities (when constructed via the standard
            pipeline, ``_build_attributor`` wraps the raw classifier in a
            softmax).
        window_len: Window length in time steps; ``T`` must be a multiple.
        nsamples: KernelSHAP coalition budget per window.
        seed: NumPy seed.
    """

    def __init__(
        self,
        classifier: Callable[[Tensor], Tensor],
        window_len: int = 4,
        nsamples: int = 256,
        seed: int | None = None,
    ) -> None:
        super().__init__(classifier)
        self._window_len = int(window_len)
        self._nsamples = int(nsamples)
        self._seed = seed

    def _explain_one(
        self,
        adapter: _ClassifierAdapter,
        x_seq_btf: npt.NDArray[np.float32],
        bg_btf: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Return ``(K, F_total)`` per-window attribution for one sequence."""
        raise NotImplementedError

    def attribute(
        self,
        x: Tensor,
        players: PlayerSet,
        target: int = 0,
    ) -> Tensor:
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

        if self._seed is not None:
            np.random.seed(self._seed)

        x_np = (
            x.detach().cpu().permute(2, 0, 1).reshape(T, F_total).numpy()
        )[np.newaxis].astype(np.float32)
        bg_np = np.zeros_like(x_np)

        adapter = _ClassifierAdapter(self._classifier, J, F_coords, T, target)
        sv_kF = self._explain_one(adapter, x_np, bg_np)

        # Distribute (K, F_total) uniformly across the win_len timesteps in each
        # window, yielding (T, F_total) → (J, F_coords, T) for players.aggregate.
        win = self._window_len
        per_step = sv_kF[:, np.newaxis, :] / float(win)  # (K, 1, F_total)
        ts_phi = np.broadcast_to(per_step, (K, win, F_total)).reshape(T, F_total)
        phi_3d = ts_phi.reshape(T, J, F_coords)
        phi_coords = torch.as_tensor(phi_3d, dtype=torch.float32).permute(1, 2, 0)
        return players.aggregate(phi_coords)


class StationaryWindowSHAPAttributor(_UniformWindowAttributor):
    """WindowSHAP with fixed uniform windows (``StationaryWindowSHAP``).

    Wraps :class:`windowshap.windowshap.StationaryWindowSHAP` from the
    official ``windowshap`` pip package.  Each sequence is explained
    independently with its own KernelSHAP solve over the K windows.
    """

    @property
    def name(self) -> str:  # noqa: D401
        """Short identifier for logging."""
        return "WindowSHAP-Stationary"

    def _explain_one(
        self,
        adapter: _ClassifierAdapter,
        x_seq_btf: npt.NDArray[np.float32],
        bg_btf: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        import shap as _shap  # noqa: PLC0415

        explainer = StationaryWindowSHAP(
            model=adapter,
            window_len=self._window_len,
            B_ts=bg_btf,
            test_ts=x_seq_btf,
            model_type="lstm",
        )
        # The package builds an explainer on first call with nsamples='auto';
        # rebuild it with our explicit budget so runs are reproducible.
        explainer.explainer = _shap.KernelExplainer(
            explainer.wraper_predict, explainer.background_data,
        )
        sv = np.asarray(
            explainer.explainer.shap_values(
                explainer.test_data, nsamples=self._nsamples, silent=True,
            ),
        )
        # Normalise to (1, n_features) regardless of SHAP version.
        if sv.ndim == 2:
            sv = sv[np.newaxis]
        elif sv.ndim == 3 and sv.shape[-1] < sv.shape[-2]:
            sv = sv.transpose(2, 0, 1)
        sv = sv[0, 0]  # (n_features,) where n_features = K * F_total
        K = x_seq_btf.shape[1] // self._window_len
        F_total = x_seq_btf.shape[2]
        return sv.reshape(K, F_total).astype(np.float32)


class DynamicWindowSHAPAttributor(_UniformWindowAttributor):
    """WindowSHAP with adaptive split-points (``DynamicWindowSHAP``).

    Wraps :class:`windowshap.windowshap.DynamicWindowSHAP` from the
    official ``windowshap`` pip package.  The package returns per-timestep
    SHAP values after iterating split-points; we then aggregate to the
    fixed K=T/window_len uniform windows so the row is comparable with
    the rest of Table~\\ref{tab:synth_ec1}.
    """

    @property
    def name(self) -> str:  # noqa: D401
        """Short identifier for logging."""
        return "WindowSHAP-Dynamic"

    def _explain_one(
        self,
        adapter: _ClassifierAdapter,
        x_seq_btf: npt.NDArray[np.float32],
        bg_btf: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        import shap as _shap  # noqa: PLC0415

        # DynamicWindowSHAP was written for an older shap API that returned
        # ``list[(N, F)]`` for classification models; SHAP ≥0.46 returns
        # ``(N, F, C)`` directly.  Patch the call site for the duration of
        # this attribute() call.
        _orig = _shap.KernelExplainer.shap_values

        def _patched(self_: object, X: object, **kw: object) -> np.ndarray:
            sv = _orig(self_, X, **kw)  # type: ignore[arg-type]
            sv = np.asarray(sv)
            if sv.ndim == 3:
                sv = sv.transpose(2, 0, 1)
            elif sv.ndim == 2:
                sv = sv[np.newaxis]
            return sv

        _shap.KernelExplainer.shap_values = _patched  # type: ignore[assignment]
        try:
            explainer = DynamicWindowSHAP(
                model=adapter,
                delta=0.05,
                n_w=8,
                B_ts=bg_btf,
                test_ts=x_seq_btf,
                model_type="lstm",
            )
            phi_full = explainer.shap_values(
                nsamples_in_loop=self._nsamples,
                nsamples_final=self._nsamples,
            )
            # phi_full shape: (1, T, F_total).  Aggregate to (K, F_total).
            T = x_seq_btf.shape[1]
            F_total = x_seq_btf.shape[2]
            K = T // self._window_len
            return (
                phi_full[0]
                .reshape(K, self._window_len, F_total)
                .sum(axis=1)
                .astype(np.float32)
            )
        finally:
            _shap.KernelExplainer.shap_values = _orig  # type: ignore[assignment]
