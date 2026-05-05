"""motionbench.attribution — Attribution method wrappers."""

from motionbench.attribution.attention_rollout import AttentionRolloutAttributor
from motionbench.attribution.base import BaseAttributor
from motionbench.attribution.captum_methods import (
    DeepLiftAttributor,
    GradientShapAttributor,
    InputXGradientAttributor,
    IntegratedGradientsAttributor,
    SaliencyAttributor,
    SmoothGradAttributor,
)
from motionbench.attribution.grad_cam import GradCAMAttributor
from motionbench.attribution.group_segment_shap import GroupSegmentSHAPAttributor
from motionbench.attribution.kernel_shap import KernelShapAttributor
from motionbench.attribution.lrp import LRPAttributor
from motionbench.attribution.shats import ShaTSAttributor
from motionbench.attribution.timeshap import TimeSHAPAttributor
from motionbench.attribution.windowshap import WindowSHAPAttributor

__all__ = [
    "BaseAttributor",
    "IntegratedGradientsAttributor",
    "DeepLiftAttributor",
    "GradientShapAttributor",
    "SaliencyAttributor",
    "SmoothGradAttributor",
    "InputXGradientAttributor",
    "LRPAttributor",
    "TimeSHAPAttributor",
    "WindowSHAPAttributor",
    "ShaTSAttributor",
    "GroupSegmentSHAPAttributor",
    "GradCAMAttributor",
    "AttentionRolloutAttributor",
    "KernelShapAttributor",
]
