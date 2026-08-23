"""motionbench.metrics — Evaluation metrics."""

from motionbench.metrics.base import BaseMetric
from motionbench.metrics.coalition_table import (
    aopc_order,
    deletion_path,
    faithfulness_enumerated,
    faithfulness_sampled,
    player_aopc_enumerated,
)
from motionbench.metrics.fidelity import (
    CrossGranularityFaithfulnessMetric,
    FaithfulnessCorrelationMetric,
    ManifoldFidelityGapMetric,
    MonotonicityCorrelationMetric,
    PixelFlippingMetric,
    PlayerDeletionMetric,
    SelectivityMetric,
)
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
    # Coalition-table metrics (real-data tracks; pinned conventions)
    "aopc_order",
    "deletion_path",
    "faithfulness_enumerated",
    "faithfulness_sampled",
    "player_aopc_enumerated",
    # Fidelity metrics
    "CrossGranularityFaithfulnessMetric",
    "FaithfulnessCorrelationMetric",
    "ManifoldFidelityGapMetric",
    "MonotonicityCorrelationMetric",
    "PixelFlippingMetric",
    "PlayerDeletionMetric",
    "SelectivityMetric",
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
