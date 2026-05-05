"""motionbench.attribution.grad_cam — Grad-CAM attribution via Captum LayerGradCam.

Wraps Captum's ``LayerGradCam`` to produce per-player attribution vectors from
convolutional classifiers operating on ``(J, F, T)`` motion sequences.

The attribution pipeline is:

1. Add batch dimension: ``(J, F, T) → (1, J, F, T)``.
2. Run ``captum.attr.LayerGradCam`` on the target convolutional layer, yielding
   a channel-summed activation map of shape ``(1, 1, *spatial)``.
3. Upsample the spatial dimensions back to the input shape ``(J, F, T)`` using
   ``torch.nn.functional.interpolate``, broadcasting any missing dims.
4. Aggregate the ``(J, F, T)`` coordinate attribution to ``(M,)`` player-level
   scores via ``players.aggregate``.

Spatial-dim handling
--------------------
Captum's ``LayerGradCam`` (with ``attr_dim_summation=True``, the default)
already sums over the channel dimension, returning ``(B, 1, *spatial)``.
The remaining spatial dims depend on the layer:

* **1-D** (e.g. ``Conv1d`` on ``(B, J*F, T)``): upsampled from ``T_cam`` to
  ``T``, then broadcast over the ``J`` and ``F`` dimensions.
* **2-D** (e.g. ``Conv2d``): upsampled to ``(F, T)`` and broadcast over ``J``.
* **3-D**: upsampled directly to ``(J, F, T)``.

The ``interpolate_mode`` parameter applies only when the cam is 2-D
(``"bilinear"`` or ``"nearest"``).  For 1-D and 3-D cams ``"nearest"`` is
always used because PyTorch's ``F.interpolate`` does not support
``"bilinear"`` on non-2-D inputs.

References
----------
Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization." ICCV.
Captum documentation: https://captum.ai/docs/layer_gradcam
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

import torch
import torch.nn as nn
from captum.attr import LayerGradCam
from torch import Tensor
from torch.nn.functional import interpolate

from motionbench.attribution.base import BaseAttributor

if TYPE_CHECKING:
    from collections.abc import Callable

    from motionbench.players.base import PlayerSet


__all__ = ["GradCAMAttributor"]


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


class _ScalarWrapper(nn.Module):
    """Wraps a callable as ``nn.Module``, reducing multi-class output to scalar.

    When the classifier returns ``(B, n_classes)``, selects ``[:, target]``.
    When it already returns ``(B,)``, passes through unchanged.

    Args:
        fn: Classifier callable ``(B, J, F, T) → (B,)`` or ``(B, n_classes)``.
        target: Class index to extract from multi-class outputs.
    """

    def __init__(self, fn: Callable[[Tensor], Tensor], target: int) -> None:
        super().__init__()
        self._fn = fn
        self._target = target

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass, returning ``(B,)`` scalar predictions.

        Args:
            x: ``(B, J, F, T)`` float32 input.

        Returns:
            ``(B,)`` float32 scalar output.
        """
        out = self._fn(x)
        if out.dim() == 2:
            return out[:, self._target]
        return out


# ---------------------------------------------------------------------------
# GradCAMAttributor
# ---------------------------------------------------------------------------


class GradCAMAttributor(BaseAttributor):
    """Captum LayerGradCam wrapper for convolutional classifiers.

    Computes Grad-CAM attributions at a specified convolutional layer,
    upsamples the resulting activation map back to the input shape
    ``(J, F, T)``, and aggregates to per-player scores ``(M,)`` via
    ``players.aggregate``.

    Args:
        classifier: Callable ``(B, J, F, T) → (B,)`` or ``(B, n_classes)``.
            Should be an ``nn.Module`` whose graph contains ``layer``.
        layer: The ``nn.Module`` convolutional layer to target
            (e.g. ``model.conv1``).  Must be reachable in the classifier's
            forward pass so that Captum can attach hooks.
        interpolate_mode: Upsampling mode passed to
            ``torch.nn.functional.interpolate`` when the Grad-CAM activation
            map is 2-D (height × width).  For 1-D and 3-D spatial maps
            ``"nearest"`` is always used.  Choices: ``"bilinear"`` |
            ``"nearest"``.  Defaults to ``"nearest"``.
    """

    #: Always ``True`` — Grad-CAM requires gradient flow through the classifier.
    requires_gradient: ClassVar[bool] = True

    def __init__(
        self,
        classifier: Callable[[Tensor], Tensor],
        *,
        layer: nn.Module,
        interpolate_mode: Literal["bilinear", "nearest"] = "nearest",
    ) -> None:
        super().__init__(classifier)
        self._layer = layer
        self._interpolate_mode: Literal["bilinear", "nearest"] = interpolate_mode

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def attribute(
        self,
        x: Tensor,
        players: PlayerSet,
        target: int = 0,
    ) -> Tensor:
        """Compute per-player Grad-CAM attributions for a single sequence.

        Args:
            x: ``(J, F, T)`` float32 input sequence (no batch dimension).
            players: :class:`~motionbench.players.base.PlayerSet` defining
                the M players and coordinate-to-player aggregation.
            target: Class index for which to compute attributions.

        Returns:
            ``(M,)`` float32 per-player attribution tensor.
        """
        J, F_dim, T = x.shape
        x_in = x.unsqueeze(0)  # (1, J, F, T)

        model = _ScalarWrapper(self._classifier, target)
        lg = LayerGradCam(model, self._layer)

        with torch.enable_grad():
            # attr_dim_summation=True (default) sums over channels → (1, 1, *spatial)
            cam_attr = lg.attribute(x_in, target=None)

        # cam_attr: (1, 1, *spatial) — strip batch and channel dims
        cam = cam_attr.detach().squeeze(0).squeeze(0)  # (*spatial,)
        spatial_dims = cam.shape  # e.g. (T_cam,) or (H, W)

        phi_coords = self._upsample_to_input(cam, spatial_dims, J, F_dim, T)
        return players.aggregate(phi_coords)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _upsample_to_input(
        self,
        cam: Tensor,
        spatial_dims: torch.Size,
        J: int,
        F_dim: int,
        T: int,
    ) -> Tensor:
        """Upsample Grad-CAM activation map to ``(J, F, T)``.

        Args:
            cam: Activation map with shape ``(*spatial_dims)`` after
                channel reduction.
            spatial_dims: The spatial shape of ``cam``.
            J: Number of joints.
            F_dim: Number of features per joint.
            T: Number of time frames.

        Returns:
            ``(J, F, T)`` float32 tensor, contiguous in memory.

        Raises:
            ValueError: If the spatial dimensionality is not 1, 2, or 3.
        """
        n_spatial = len(spatial_dims)
        cam_4d = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, *spatial)

        if n_spatial == 1:
            # 1-D cam from Conv1d: upsample T dimension, broadcast J and F
            cam_t = interpolate(cam_4d, size=(T,), mode="nearest")  # (1, 1, T)
            phi_t = cam_t.squeeze(0).squeeze(0)  # (T,)
            phi_coords = phi_t.unsqueeze(0).unsqueeze(0).expand(J, F_dim, T).clone()

        elif n_spatial == 2:
            # 2-D cam: upsample (H, W) → (F, T), broadcast J
            mode = self._interpolate_mode
            cam_ft = interpolate(cam_4d, size=(F_dim, T), mode=mode)  # (1, 1, F, T)
            phi_ft = cam_ft.squeeze(0).squeeze(0)  # (F, T)
            phi_coords = phi_ft.unsqueeze(0).expand(J, F_dim, T).clone()

        elif n_spatial == 3:
            # 3-D cam: upsample directly to (J, F, T)
            cam_5d = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
            cam_jft = interpolate(cam_5d, size=(J, F_dim, T), mode="nearest")
            phi_coords = cam_jft.squeeze(0).squeeze(0)  # (J, F, T)

        else:
            raise ValueError(
                f"Grad-CAM activation map has {n_spatial} spatial dimensions; "
                "expected 1, 2, or 3."
            )

        return phi_coords.contiguous()
