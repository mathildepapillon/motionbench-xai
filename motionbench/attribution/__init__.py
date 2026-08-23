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
from motionbench.attribution.enumerated_kernel_shap import (
    build_coalition_masks,
    kernel_shap_exact,
    shapley_kernel,
)
from motionbench.attribution.grad_cam import GradCAMAttributor
from motionbench.attribution.group_segment_shap import GroupSegmentSHAPAttributor
from motionbench.attribution.kernel_shap import KernelShapAttributor
from motionbench.attribution.kernelshap_temporal import (  # TimeSHAPAttributor is a compat alias
    KernelSHAPTemporalAttributor,
    TimeSHAPAttributor,
)
from motionbench.attribution.lrp import LRPAttributor
from motionbench.attribution.sampled_coalitions import (
    DEFAULT_COALITION_SEED,
    EXACT_MAX_M,
    phi_from_values,
    sampled_coalition_set,
)
from motionbench.attribution.shats import ShaTSAttributor
from motionbench.attribution.timeshap_real import RealTimeSHAPAttributor
from motionbench.attribution.windowshap import (
    DynamicWindowSHAPAttributor,
    StationaryWindowSHAPAttributor,
    WindowSHAPAttributor,
)

__all__ = [
    # Coalition designs and WLS solvers
    "DEFAULT_COALITION_SEED",
    "EXACT_MAX_M",
    "build_coalition_masks",
    "kernel_shap_exact",
    "phi_from_values",
    "sampled_coalition_set",
    "shapley_kernel",
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
