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
from motionbench.attribution.kernelshap_temporal import KernelSHAPTemporalAttributor, TimeSHAPAttributor  # TimeSHAPAttributor is a compat alias
from motionbench.attribution.timeshap_real import RealTimeSHAPAttributor
from motionbench.attribution.windowshap import (
    DynamicWindowSHAPAttributor,
    StationaryWindowSHAPAttributor,
    WindowSHAPAttributor,
)

__all__ = [
    "BaseAttributor",
    "IntegratedGradientsAttributor",
    "DeepLiftAttributor",
    "GradientShapAttributor",
    "SaliencyAttributor",
    "SmoothGradAttributor",
    "InputXGradientAttributor",
    "LRPAttributor",
    "KernelSHAPTemporalAttributor",
    "TimeSHAPAttributor",  # compat alias for KernelSHAPTemporalAttributor
    "RealTimeSHAPAttributor",  # actual ``timeshap`` pip-package wrapper
    "WindowSHAPAttributor",
    "StationaryWindowSHAPAttributor",
    "DynamicWindowSHAPAttributor",
    "ShaTSAttributor",
    "GroupSegmentSHAPAttributor",
    "GradCAMAttributor",
    "AttentionRolloutAttributor",
    "KernelShapAttributor",
]
