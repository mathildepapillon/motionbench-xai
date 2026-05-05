"""motionbench.metrics — Evaluation metrics."""

from motionbench.metrics.base import BaseMetric
from motionbench.metrics.ground_truth import (
    EC1Metric,
    EC2Metric,
    EC3Metric,
    EfficiencyErrorMetric,
    KendallRankMetric,
    SpearmanRankMetric,
    TopKRecovery,
)
from motionbench.metrics.sanity_checks import (
    ModelParameterRandomisationMetric,
    RandomLogitMetric,
)
from motionbench.metrics.stability import (
    ContinuityMetric,
    LipschitzEstimateMetric,
    MaxSensitivityMetric,
)

__all__ = [
    "BaseMetric",
    # Ground-truth metrics
    "EC1Metric",
    "EC2Metric",
    "EC3Metric",
    "TopKRecovery",
    "SpearmanRankMetric",
    "KendallRankMetric",
    "EfficiencyErrorMetric",
    # Stability metrics
    "MaxSensitivityMetric",
    "ContinuityMetric",
    "LipschitzEstimateMetric",
    # Sanity-check metrics
    "ModelParameterRandomisationMetric",
    "RandomLogitMetric",
]
