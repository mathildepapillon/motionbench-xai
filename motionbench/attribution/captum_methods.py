"""motionbench.attribution.captum_methods — Captum-based attribution wrappers.

Six :class:`~motionbench.attribution.base.BaseAttributor` subclasses that wrap
Captum gradient attribution methods.  Each attributor:

1. Accepts a ``classifier`` callable ``(B, J, F, T) → (B,)`` or ``(B, n_classes)``.
2. Wraps it in a thin :class:`_ModuleWrapper` so Captum can call it as ``nn.Module``.
3. Calls the appropriate Captum method to obtain per-coordinate ``(J, F, T)``
   attribution values.
4. Aggregates to ``(M,)`` player-level scores via ``players.aggregate(phi_coords)``.

Methods
-------
- :class:`IntegratedGradientsAttributor` — IG (Sundararajan et al. 2017).
- :class:`DeepLiftAttributor` — DeepLIFT (Shrikumar et al. 2017).
- :class:`GradientShapAttributor` — GradientSHAP (Erion et al. 2021).
- :class:`SaliencyAttributor` — |∂y/∂x| (Simonyan et al. 2013).
- :class:`SmoothGradAttributor` — NoiseTunnel + Saliency (Smilkov et al. 2017).
- :class:`InputXGradientAttributor` — x · ∂y/∂x (Kindermans et al. 2016).

All methods run on CPU and require gradient flow through the classifier.

References
----------
Sundararajan et al. (2017) ICML. "Axiomatic attribution for deep networks."
Shrikumar et al. (2017) ICML. "Learning important features through propagating
    activation differences."
Smilkov et al. (2017). "SmoothGrad: removing noise by adding noise."
Captum documentation: https://captum.ai/docs/
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
from captum.attr import (
    DeepLift,
    GradientShap,
    InputXGradient,
    IntegratedGradients,
    NoiseTunnel,
    Saliency,
)
from torch import Tensor

from motionbench.attribution.base import BaseAttributor

if TYPE_CHECKING:
    from collections.abc import Callable

    from motionbench.players.base import PlayerSet

__all__ = [
    "IntegratedGradientsAttributor",
    "DeepLiftAttributor",
    "GradientShapAttributor",
    "SaliencyAttributor",
    "SmoothGradAttributor",
    "InputXGradientAttributor",
]

BaselineType = Literal["zero", "mean", "gaussian"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _ModuleWrapper(nn.Module):
    """Wraps a callable as ``nn.Module`` for Captum.

    Reduces multi-class output ``(B, n_classes)`` to scalar ``(B,)`` by
    selecting the requested target class.  Single-output classifiers that
    already return ``(B,)`` are passed through unchanged.

    Args:
        fn: Classifier callable ``(B, J, F, T) → (B,)`` or ``(B, n_classes)``.
        target: Class index to extract when ``fn`` returns ``(B, n_classes)``.
    """

    def __init__(self, fn: Callable[[Tensor], Tensor], target: int) -> None:
        super().__init__()
        self._fn = fn
        self._target = target

    def forward(self, x: Tensor) -> Tensor:
        """Run classifier and return ``(B,)`` scalar output.

        Args:
            x: ``(B, J, F, T)`` float32 tensor.

        Returns:
            ``(B,)`` float32 tensor of scalar predictions.
        """
        out = self._fn(x)
        if out.dim() == 2:
            return out[:, self._target]
        return out


def _make_baseline(x: Tensor, strategy: BaselineType) -> Tensor:
    """Construct a baseline tensor with shape ``(1, J, F, T)``.

    Args:
        x: Input sample ``(J, F, T)`` float32.
        strategy: One of ``"zero"``, ``"mean"``, ``"gaussian"``.

    Returns:
        ``(1, J, F, T)`` float32 baseline tensor.

    Raises:
        ValueError: If ``strategy`` is not a recognised baseline type.
    """
    x_batched = x.unsqueeze(0)
    if strategy == "zero":
        return torch.zeros_like(x_batched)
    if strategy == "mean":
        return torch.full_like(x_batched, x.mean().item())
    if strategy == "gaussian":
        return torch.randn_like(x_batched)
    raise ValueError(f"Unknown baseline {strategy!r}. Choose 'zero', 'mean', or 'gaussian'.")


# ---------------------------------------------------------------------------
# Attributor subclasses
# ---------------------------------------------------------------------------


class IntegratedGradientsAttributor(BaseAttributor):
    """Integrated Gradients attribution (Sundararajan et al. 2017).

    Integrates gradients of the model output with respect to the input along a
    straight-line path from a baseline to the actual input using ``n_steps``
    Riemann approximation steps.

    Args:
        classifier: Callable ``(B, J, F, T) → (B,)`` or ``(B, n_classes)``.
        baseline: Baseline strategy — ``"zero"`` | ``"mean"`` | ``"gaussian"``.
            Defaults to ``"zero"``.
        n_steps: Number of integration steps.  Defaults to ``50``.
    """

    def __init__(
        self,
        classifier: Callable[[Tensor], Tensor],
        *,
        baseline: BaselineType = "zero",
        n_steps: int = 50,
    ) -> None:
        super().__init__(classifier)
        self._baseline_type: BaselineType = baseline
        self._n_steps = n_steps

    @property
    def requires_gradient(self) -> bool:
        """Always ``True`` — IG requires gradient flow."""
        return True

    def attribute(
        self,
        x: Tensor,
        players: PlayerSet,
        target: int = 0,
    ) -> Tensor:
        """Compute per-player Integrated Gradients attributions.

        Args:
            x: ``(J, F, T)`` float32 input sequence.
            players: :class:`~motionbench.players.base.PlayerSet` defining
                the M players and coordinate-to-player aggregation.
            target: Class index for which to compute attributions.

        Returns:
            ``(M,)`` float32 per-player attribution tensor.
        """
        model = _ModuleWrapper(self._classifier, target)
        ig = IntegratedGradients(model)
        x_in = x.unsqueeze(0)
        baseline = _make_baseline(x, self._baseline_type)
        with torch.enable_grad():
            attrs = ig.attribute(x_in, baselines=baseline, n_steps=self._n_steps)
        phi_coords = attrs.squeeze(0).detach()
        return players.aggregate(phi_coords)


class DeepLiftAttributor(BaseAttributor):
    """DeepLIFT attribution (Shrikumar et al. 2017).

    Computes attributions by comparing each neuron's activation to a
    reference activation obtained from the baseline input.

    Args:
        classifier: Callable ``(B, J, F, T) → (B,)`` or ``(B, n_classes)``.
        baseline: Baseline strategy — ``"zero"`` | ``"mean"`` | ``"gaussian"``.
            Defaults to ``"zero"``.
    """

    def __init__(
        self,
        classifier: Callable[[Tensor], Tensor],
        *,
        baseline: BaselineType = "zero",
    ) -> None:
        super().__init__(classifier)
        self._baseline_type: BaselineType = baseline

    @property
    def requires_gradient(self) -> bool:
        """Always ``True`` — DeepLIFT requires gradient flow."""
        return True

    def attribute(
        self,
        x: Tensor,
        players: PlayerSet,
        target: int = 0,
    ) -> Tensor:
        """Compute per-player DeepLIFT attributions.

        Args:
            x: ``(J, F, T)`` float32 input sequence.
            players: :class:`~motionbench.players.base.PlayerSet` defining
                the M players and coordinate-to-player aggregation.
            target: Class index for which to compute attributions.

        Returns:
            ``(M,)`` float32 per-player attribution tensor.
        """
        model = _ModuleWrapper(self._classifier, target)
        dl = DeepLift(model)
        x_in = x.unsqueeze(0)
        baseline = _make_baseline(x, self._baseline_type)
        with torch.enable_grad():
            attrs = dl.attribute(x_in, baselines=baseline)
        phi_coords = attrs.squeeze(0).detach()
        return players.aggregate(phi_coords)


class GradientShapAttributor(BaseAttributor):
    """GradientSHAP attribution (Erion et al. 2021).

    Approximates SHAP values by computing expected gradients, sampling noise
    around the given baseline using ``n_samples`` Monte-Carlo draws.

    Args:
        classifier: Callable ``(B, J, F, T) → (B,)`` or ``(B, n_classes)``.
        baseline: Baseline strategy — ``"zero"`` | ``"mean"`` | ``"gaussian"``.
            Defaults to ``"zero"``.
        n_samples: Number of noise samples.  Defaults to ``50``.
        stdevs: Standard deviation of added Gaussian noise.  Defaults to
            ``0.0`` (no noise beyond the baseline distribution).
    """

    def __init__(
        self,
        classifier: Callable[[Tensor], Tensor],
        *,
        baseline: BaselineType = "zero",
        n_samples: int = 50,
        stdevs: float = 0.0,
    ) -> None:
        super().__init__(classifier)
        self._baseline_type: BaselineType = baseline
        self._n_samples = n_samples
        self._stdevs = stdevs

    @property
    def requires_gradient(self) -> bool:
        """Always ``True`` — GradientSHAP requires gradient flow."""
        return True

    def attribute(
        self,
        x: Tensor,
        players: PlayerSet,
        target: int = 0,
    ) -> Tensor:
        """Compute per-player GradientSHAP attributions.

        Args:
            x: ``(J, F, T)`` float32 input sequence.
            players: :class:`~motionbench.players.base.PlayerSet` defining
                the M players and coordinate-to-player aggregation.
            target: Class index for which to compute attributions.

        Returns:
            ``(M,)`` float32 per-player attribution tensor.
        """
        model = _ModuleWrapper(self._classifier, target)
        gs = GradientShap(model)
        x_in = x.unsqueeze(0)
        baseline = _make_baseline(x, self._baseline_type)
        with torch.enable_grad():
            attrs = gs.attribute(
                x_in,
                baselines=baseline,
                n_samples=self._n_samples,
                stdevs=self._stdevs,
            )
        phi_coords = attrs.squeeze(0).detach()
        return players.aggregate(phi_coords)


class SaliencyAttributor(BaseAttributor):
    """Saliency (gradient magnitude) attribution (Simonyan et al. 2013).

    Computes the absolute value of the gradient of the model output with
    respect to the input: ``|∂y/∂x|``.  No baseline is required.

    Args:
        classifier: Callable ``(B, J, F, T) → (B,)`` or ``(B, n_classes)``.
        baseline: Accepted for API consistency but unused by this method.
    """

    def __init__(
        self,
        classifier: Callable[[Tensor], Tensor],
        *,
        baseline: BaselineType = "zero",
    ) -> None:
        super().__init__(classifier)
        self._baseline_type: BaselineType = baseline

    @property
    def requires_gradient(self) -> bool:
        """Always ``True`` — Saliency requires gradient flow."""
        return True

    def attribute(
        self,
        x: Tensor,
        players: PlayerSet,
        target: int = 0,
    ) -> Tensor:
        """Compute per-player saliency attributions.

        Args:
            x: ``(J, F, T)`` float32 input sequence.
            players: :class:`~motionbench.players.base.PlayerSet` defining
                the M players and coordinate-to-player aggregation.
            target: Class index for which to compute attributions.

        Returns:
            ``(M,)`` float32 per-player attribution tensor.
        """
        model = _ModuleWrapper(self._classifier, target)
        sal = Saliency(model)
        x_in = x.unsqueeze(0)
        with torch.enable_grad():
            attrs = sal.attribute(x_in, abs=True)
        phi_coords = attrs.squeeze(0).detach()
        return players.aggregate(phi_coords)


class SmoothGradAttributor(BaseAttributor):
    """SmoothGrad attribution — NoiseTunnel wrapping Saliency (Smilkov et al. 2017).

    Averages saliency maps computed over ``nt_samples`` noisy copies of the
    input to produce smoother, more stable attribution maps.  No baseline is
    required.

    Args:
        classifier: Callable ``(B, J, F, T) → (B,)`` or ``(B, n_classes)``.
        baseline: Accepted for API consistency but unused by this method.
        nt_samples: Number of noise samples.  Defaults to ``50``.
        stdevs: Standard deviation of added Gaussian noise.  Defaults to
            ``0.1``.
    """

    def __init__(
        self,
        classifier: Callable[[Tensor], Tensor],
        *,
        baseline: BaselineType = "zero",
        nt_samples: int = 50,
        stdevs: float = 0.1,
    ) -> None:
        super().__init__(classifier)
        self._baseline_type: BaselineType = baseline
        self._nt_samples = nt_samples
        self._stdevs = stdevs

    @property
    def requires_gradient(self) -> bool:
        """Always ``True`` — SmoothGrad requires gradient flow."""
        return True

    def attribute(
        self,
        x: Tensor,
        players: PlayerSet,
        target: int = 0,
    ) -> Tensor:
        """Compute per-player SmoothGrad attributions.

        Args:
            x: ``(J, F, T)`` float32 input sequence.
            players: :class:`~motionbench.players.base.PlayerSet` defining
                the M players and coordinate-to-player aggregation.
            target: Class index for which to compute attributions.

        Returns:
            ``(M,)`` float32 per-player attribution tensor.
        """
        model = _ModuleWrapper(self._classifier, target)
        nt = NoiseTunnel(Saliency(model))
        x_in = x.unsqueeze(0)
        with torch.enable_grad():
            attrs = nt.attribute(
                x_in,
                nt_type="smoothgrad",
                nt_samples=self._nt_samples,
                stdevs=self._stdevs,
                abs=True,
            )
        phi_coords = attrs.squeeze(0).detach()
        return players.aggregate(phi_coords)


class InputXGradientAttributor(BaseAttributor):
    """Input × Gradient attribution (Kindermans et al. 2016).

    Computes element-wise product of the input and its gradient with respect
    to the model output: ``x · ∂y/∂x``.  No baseline is required.

    Args:
        classifier: Callable ``(B, J, F, T) → (B,)`` or ``(B, n_classes)``.
        baseline: Accepted for API consistency but unused by this method.
    """

    def __init__(
        self,
        classifier: Callable[[Tensor], Tensor],
        *,
        baseline: BaselineType = "zero",
    ) -> None:
        super().__init__(classifier)
        self._baseline_type: BaselineType = baseline

    @property
    def requires_gradient(self) -> bool:
        """Always ``True`` — InputXGradient requires gradient flow."""
        return True

    def attribute(
        self,
        x: Tensor,
        players: PlayerSet,
        target: int = 0,
    ) -> Tensor:
        """Compute per-player InputXGradient attributions.

        Args:
            x: ``(J, F, T)`` float32 input sequence.
            players: :class:`~motionbench.players.base.PlayerSet` defining
                the M players and coordinate-to-player aggregation.
            target: Class index for which to compute attributions.

        Returns:
            ``(M,)`` float32 per-player attribution tensor.
        """
        model = _ModuleWrapper(self._classifier, target)
        ixg = InputXGradient(model)
        x_in = x.unsqueeze(0)
        with torch.enable_grad():
            attrs = ixg.attribute(x_in)
        phi_coords = attrs.squeeze(0).detach()
        return players.aggregate(phi_coords)
