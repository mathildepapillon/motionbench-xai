"""motionbench.attribution.base — BaseAttributor abstract base class.

An *attributor* maps a single input sequence to a per-player attribution
vector ``φ ∈ ℝ^M`` that quantifies each player's contribution to the
model's prediction.

Taxonomy of attribution methods in this benchmark
--------------------------------------------------

**SHAP-based (use pluggable imputer):**
    :class:`~motionbench.attribution.kernel_shap.KernelShapAttributor`
    — wraps ``shap.KernelExplainer`` with a custom
    :class:`~motionbench.imputers.base.BaseImputer`-backed masker.

**Gradient-based (no imputer required):**
    :class:`~motionbench.attribution.captum_methods.IntegratedGradientsAttributor`,
    ``DeepLiftAttributor``, ``GradientShapAttributor``, ``SaliencyAttributor``,
    ``SmoothGradAttributor``, ``InputXGradientAttributor`` — thin wrappers
    around Captum methods.

**LRP (no imputer required):**
    :class:`~motionbench.attribution.lrp.LRPAttributor` — via Zennit.

**Temporal-SHAP variants:**
    ``TimeSHAPAttributor``, ``WindowSHAPAttributor``, ``ShaTS Attributor``,
    ``GroupSegmentSHAPAttributor`` — thin wrappers around their respective
    reference implementations.

**Activation-based:**
    :class:`~motionbench.attribution.grad_cam.GradCAMAttributor` — via
    Captum ``LayerGradCam``.

Interface contract
------------------
All attributors return a ``(M,)`` tensor aggregated to the player level.
The aggregation from per-coordinate to per-player uses the additivity of
Shapley values (Jullum et al. 2021, Proposition 1):

    φ_k = Σ_{i ∈ group_k} φ_i^{(coord)}

For gradient methods, this is a literal sum over grouped coordinates.
For KernelSHAP, the grouping is enforced via the player-set masker.

References
----------
Lundberg & Lee (2017) "A unified approach to interpreting model predictions."
Sundararajan et al. (2017) "Axiomatic attribution for deep networks."
Bach et al. (2015) "On pixel-wise explanations for non-linear classifier
decisions by layer-wise relevance propagation."
Bento et al. (2020) "TimeSHAP: Explaining recurrent models through sequence
perturbations."
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from motionbench.players.base import PlayerSet


__all__ = ["BaseAttributor"]


class BaseAttributor(ABC):
    """Abstract base class for all attribution methods.

    Shape conventions:

    * Input sequence:  ``(J, F, T)`` float32 Tensor (single sample, no batch dim).
    * Output:          ``(M,)`` float32 Tensor of per-player attribution scores.

    Constructor
    -----------
    All concrete subclasses must accept ``classifier`` as their first
    positional argument and ``**kwargs`` for method-specific hyperparameters.
    This convention allows the pipeline factory to instantiate any attributor
    uniformly from a Hydra config:

    .. code-block:: python

        cfg = OmegaConf.load("configs/methods/kernelshap_vaeac.yaml")
        attributor = REGISTRY[cfg.method](classifier, **cfg.kwargs)

    The ``classifier`` callable maps ``(B, J, F, T) → (B,)`` scalar target
    values (e.g. probability of class 0).  Attributors that require gradient
    flow through the classifier will call it inside a ``torch.enable_grad()``
    context; attributors that only need forward passes should call it inside
    ``torch.no_grad()``.
    """

    def __init__(self, classifier: Callable[[Tensor], Tensor], **kwargs: object) -> None:
        """Initialise the attributor with a fixed classifier.

        Args:
            classifier: Callable ``(B, J, F, T) float32 → (B,) float32``.
                Must return a single scalar per sample (e.g. class probability,
                not raw logits).
            **kwargs: Method-specific hyperparameters (e.g. ``n_samples``,
                ``baselines``, ``imputer``).
        """
        self._classifier = classifier

    @abstractmethod
    def attribute(
        self,
        x: Tensor,
        players: "PlayerSet",
        target: int = 0,
    ) -> Tensor:
        """Compute per-player attribution scores for a single sequence.

        Args:
            x: ``(J, F, T)`` float32 input sequence.  Must **not** include a
                batch dimension; the attributor adds it internally.
            players: :class:`~motionbench.players.base.PlayerSet` defining
                the M players and the coordinate→player aggregation map.
            target: Class index for which to compute attributions.  Passed to
                the classifier to select the output dimension.

        Returns:
            ``(M,)`` float32 Tensor.  ``output[k]`` is the attribution score
            for player ``k``.  May be positive or negative; the sign follows
            the SHAP / IG convention (positive = player increases model
            output toward ``target``).

        Raises:
            ValueError: if ``x.shape != (J, F, T)`` (wrong shape).
            RuntimeError: if the attributor has not been properly initialised
                (e.g. imputer not fitted).
        """

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Short string identifier for logging and leaderboard tables."""
        return self.__class__.__name__

    @property
    def requires_imputer(self) -> bool:
        """Whether this attributor requires a fitted BaseImputer.

        True for KernelSHAP variants; False for gradient-based methods.
        """
        return False

    @property
    def requires_gradient(self) -> bool:
        """Whether this attributor requires gradients through the classifier.

        True for IG, DeepLift, LRP; False for KernelSHAP.
        """
        return False

    # ------------------------------------------------------------------
    # Quantus integration
    # ------------------------------------------------------------------

    def build_quantus_explain_func(
        self,
        players: "PlayerSet",
        target: int,
        device: str = "cpu",
    ) -> Optional[Callable]:
        """Return a Quantus-compatible ``explain_func`` for this attributor.

        Quantus stability and sanity metrics (MaxSensitivity, MPRT) need to
        re-compute attributions after input perturbation or model-parameter
        randomisation.  The returned callable satisfies Quantus's expected
        signature::

            fn(model, inputs, targets, **kwargs) -> np.ndarray

        where ``inputs`` is ``(B, J*F, T)`` float32 and the return value is
        ``(B, J*F, T)`` float32 coordinate-level attribution magnitudes.

        The default implementation:

        * For attributors with ``requires_gradient = True`` (IG, DeepLift,
          LRP, etc.): returns a closure that calls ``self.attribute()`` on
          each element of the batch and expands player-level results back to
          coordinate space.  **This is the correct method-specific behaviour
          required for proper Quantus evaluation.**

        * For SHAP-based attributors (``requires_gradient = False``, e.g.
          KernelSHAP): returns ``None``.  Re-running KernelSHAP for every
          Quantus perturbation step would require O(2^M) model evaluations
          per perturbation, which is computationally infeasible.  The pipeline
          will skip stability/sanity metrics for these methods.

        Args:
            players: Player set used to produce the original attributions.
            target: Class index used to produce the original attributions.
            device: Device string (``"cpu"`` or ``"cuda"``) for tensor ops.

        Returns:
            A Quantus-compatible callable, or ``None`` if re-attribution is
            infeasible for this method.
        """
        if not self.requires_gradient:
            # SHAP-based methods: skip stability/sanity (too expensive).
            return None

        J, F, T = players.shape

        def _explain_fn(
            model: object,
            inputs: "np.ndarray",
            targets: "np.ndarray",
            **kwargs: object,
        ) -> "np.ndarray":
            # Quantus calls this for EVERY perturbation step (MaxSensitivity)
            # and for EVERY layer randomisation (MPRT).  `model` is a
            # _QuantusWrapper — for MaxSensitivity it wraps the original
            # classifier; for MPRT it wraps a **deepcopy** with randomised
            # parameters.  We MUST use `model`'s inner classifier here;
            # using `self._classifier` would give the original model's
            # attributions for every MPRT step → correlation = 1.0 always.
            clf_inner: object = (
                getattr(model, "_module_clf", None)
                or getattr(model, "_fn_clf", None)
                or self._classifier
            )

            # Temporarily swap `self._classifier` (and `self._model` for LRP)
            # so that `self.attribute()` runs with the correct classifier.
            orig_clf = self._classifier
            orig_model = getattr(self, "_model", None)
            try:
                self._classifier = clf_inner  # type: ignore[assignment]
                if orig_model is not None:
                    self._model = clf_inner  # type: ignore[assignment]
                B = inputs.shape[0]
                results = []
                for b in range(B):
                    x_b = torch.from_numpy(inputs[b].copy()).reshape(J, F, T).to(device)
                    phi_b = self.attribute(x_b, players, target=target)
                    phi_coords = players.expand_attributions(phi_b.detach().cpu())
                    a = np.abs(phi_coords.numpy().reshape(J * F, T).astype(np.float32))
                    results.append(a)
            finally:
                self._classifier = orig_clf  # type: ignore[assignment]
                if orig_model is not None:
                    self._model = orig_model  # type: ignore[assignment]
            return np.array(results)  # (B, J*F, T)

        return _explain_fn

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
