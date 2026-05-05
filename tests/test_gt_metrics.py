"""Tests for motionbench.metrics.ground_truth.

Test plan
---------
1. For each metric: verify ValueError raised when oracle=None.
2. test_perfect_attributions_give_zero_ec1 — φ = oracle.true_shapley → EC1 = 0.
3. test_zero_attributions_give_ec1_norm_one — φ = 0 → EC1_norm ≈ 1.
4. test_topk_recovers_all_important — φ = oracle Shapley → TopKRecovery = 1.0.
5. test_efficiency_error_kernel_shap — KernelSHAP + oracle imputer → < 1e-3.

Fixtures
--------
All tests use the canonical shape J=5, F=3, T=16, M=4 from conftest.py.
A ``MockPlayerSet`` (equal-width temporal windows) and a ``MockOracle``
(returns prescribed Shapley values) are defined inline.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import torch
from torch import Tensor

from motionbench.metrics.ground_truth import (
    EC1Metric,
    EC2Metric,
    EC3Metric,
    EfficiencyErrorMetric,
    KendallRankMetric,
    SpearmanRankMetric,
    TopKRecovery,
)
from motionbench.players.base import PlayerSet

# ---------------------------------------------------------------------------
# Constants matching conftest.py
# ---------------------------------------------------------------------------

J, F, T, M = 5, 3, 16, 4

# ---------------------------------------------------------------------------
# MockPlayerSet — equal-width temporal windows
# ---------------------------------------------------------------------------


class MockPlayerSet(PlayerSet):
    """Equal-width temporal windows player set for testing.

    Args:
        J: Number of joints.
        F: Features per joint.
        T: Time-steps.
        M: Number of windows.  Must divide T evenly.
    """

    def __init__(self, j: int, f: int, t: int, m: int) -> None:
        self._J = j
        self._F = f
        self._T = t
        self._M = m
        self._ws = t // m

    @property
    def n_players(self) -> int:
        return self._M

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._J, self._F, self._T

    def coalition_mask(self, z: Tensor) -> Tensor:
        """Expand (M,) coalition indicator to (J, F, T) mask."""
        mask = torch.zeros(self._J, self._F, self._T, dtype=torch.bool)
        for k in range(self._M):
            if z[k]:
                t0 = k * self._ws
                t1 = (k + 1) * self._ws
                mask[:, :, t0:t1] = True
        return mask

    def aggregate(self, phi_coords: Tensor) -> Tensor:
        """Sum per-window coordinates to (M,)."""
        phi = torch.zeros(self._M)
        for k in range(self._M):
            t0 = k * self._ws
            t1 = (k + 1) * self._ws
            phi[k] = phi_coords[:, :, t0:t1].sum()
        return phi


# ---------------------------------------------------------------------------
# MockOracle — returns prescribed Shapley values
# ---------------------------------------------------------------------------


class MockOracle:
    """Oracle that returns a fixed Shapley vector and zero conditional samples.

    Args:
        phi_true: ``(M,)`` float32 tensor to return from ``true_shapley``.
        v_empty: Value to use for the empty coalition (marginal mean output).
    """

    def __init__(self, phi_true: Tensor, v_empty: float = 0.0) -> None:
        self._phi_true = phi_true.float()
        self._v_empty = v_empty

    def true_shapley(
        self,
        x: Tensor,
        classifier: Callable[[Tensor], Tensor],
        players: PlayerSet,
        n_mc: int = 1000,
        seed: int | None = None,
    ) -> Tensor:
        """Return the prescribed Shapley vector."""
        return self._phi_true.clone()

    def conditional_sample(
        self,
        x_obs: Tensor,
        mask: Tensor,
        n: int,
        seed: int | None = None,
    ) -> Tensor:
        """Return samples that produce classifier output == v_empty."""
        J_, F_, T_ = x_obs.shape
        # Return zero tensors — classifier (mean) will output 0.0
        return torch.zeros(n, J_, F_, T_)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _linear_clf(x: Tensor) -> Tensor:
    """Linear classifier: global mean.  (B, J, F, T) → (B,)."""
    return x.mean(dim=(1, 2, 3))


def _make_players() -> MockPlayerSet:
    return MockPlayerSet(J, F, T, M)


# ---------------------------------------------------------------------------
# 1. ValueError when oracle=None — one test per metric class
# ---------------------------------------------------------------------------


METRIC_CLASSES = [
    EC1Metric,
    EC2Metric,
    EC3Metric,
    TopKRecovery,
    SpearmanRankMetric,
    KendallRankMetric,
    EfficiencyErrorMetric,
]


@pytest.mark.parametrize("MetricCls", METRIC_CLASSES)
def test_metric_raises_without_oracle(MetricCls: type) -> None:
    """Every ground-truth metric must raise ValueError when oracle=None.

    Args:
        MetricCls: Metric class to test.
    """
    metric = MetricCls()
    players = _make_players()
    phi = torch.randn(M)
    x = torch.randn(J, F, T)
    with pytest.raises(ValueError, match="requires_oracle"):
        metric.evaluate(phi, x, _linear_clf, players, oracle=None)


# ---------------------------------------------------------------------------
# 2. Perfect attributions give EC1 = 0
# ---------------------------------------------------------------------------


def test_perfect_attributions_give_zero_ec1() -> None:
    """When φ = oracle.true_shapley, EC1 must equal 0.0 exactly."""
    phi_true = torch.tensor([0.5, -0.3, 0.1, 0.2])
    oracle = MockOracle(phi_true)
    players = _make_players()
    x = torch.randn(J, F, T)
    metric = EC1Metric()
    result = metric.evaluate(phi_true.clone(), x, _linear_clf, players, oracle=oracle)
    assert "ec1" in result
    assert result["ec1"] == pytest.approx(0.0, abs=1e-6), (
        f"EC1 should be 0 for perfect attributions; got {result['ec1']}"
    )


# ---------------------------------------------------------------------------
# 3. Zero attributions give EC1_norm ≈ 1
# ---------------------------------------------------------------------------


def test_zero_attributions_give_ec1_norm_one() -> None:
    """When φ = 0, EC1_norm should equal 1.0 (error equals oracle scale).

    EC1_norm = mean|φ − φ_oracle| / mean|φ_oracle|
             = mean|φ_oracle| / mean|φ_oracle| = 1  when φ = 0.
    """
    phi_true = torch.tensor([0.5, -0.3, 0.1, 0.2])
    oracle = MockOracle(phi_true)
    players = _make_players()
    x = torch.randn(J, F, T)
    phi_zero = torch.zeros(M)
    metric = EC1Metric()
    result = metric.evaluate(phi_zero, x, _linear_clf, players, oracle=oracle)
    assert "ec1_norm" in result
    assert result["ec1_norm"] == pytest.approx(1.0, abs=1e-5), (
        f"EC1_norm should be 1.0 when φ=0; got {result['ec1_norm']}"
    )


# ---------------------------------------------------------------------------
# 4. TopKRecovery = 1.0 when φ = oracle Shapley
# ---------------------------------------------------------------------------


def test_topk_recovers_all_important() -> None:
    """When φ = oracle.true_shapley, TopKRecovery must return 1.0."""
    phi_true = torch.tensor([0.5, -0.3, 0.1, 0.2])
    oracle = MockOracle(phi_true)
    players = _make_players()
    x = torch.randn(J, F, T)
    metric = TopKRecovery()
    result = metric.evaluate(phi_true.clone(), x, _linear_clf, players, oracle=oracle)
    assert "topk_overlap" in result
    assert result["topk_overlap"] == pytest.approx(1.0, abs=1e-6), (
        f"topk_overlap should be 1.0 when φ = oracle; got {result['topk_overlap']}"
    )
    assert result["top1"] == pytest.approx(1.0, abs=1e-6), (
        f"top1 should be 1.0 when φ = oracle; got {result['top1']}"
    )


# ---------------------------------------------------------------------------
# 5. Spearman and Kendall return 1.0 when φ = oracle (monotone identical)
# ---------------------------------------------------------------------------


def test_spearman_perfect_agreement() -> None:
    """When φ = φ_oracle, Spearman correlation must be 1.0."""
    phi_true = torch.tensor([0.5, -0.3, 0.1, 0.2])
    oracle = MockOracle(phi_true)
    players = _make_players()
    x = torch.randn(J, F, T)
    metric = SpearmanRankMetric()
    result = metric.evaluate(phi_true.clone(), x, _linear_clf, players, oracle=oracle)
    assert "spearman" in result
    assert result["spearman"] == pytest.approx(1.0, abs=1e-5), (
        f"Spearman should be 1.0 for identical vectors; got {result['spearman']}"
    )


def test_kendall_perfect_agreement() -> None:
    """When φ = φ_oracle, Kendall tau must be 1.0."""
    phi_true = torch.tensor([0.5, -0.3, 0.1, 0.2])
    oracle = MockOracle(phi_true)
    players = _make_players()
    x = torch.randn(J, F, T)
    metric = KendallRankMetric()
    result = metric.evaluate(phi_true.clone(), x, _linear_clf, players, oracle=oracle)
    assert "kendall" in result
    assert result["kendall"] == pytest.approx(1.0, abs=1e-5), (
        f"Kendall should be 1.0 for identical vectors; got {result['kendall']}"
    )


# ---------------------------------------------------------------------------
# 6. EC2 = 0 for perfect attributions
# ---------------------------------------------------------------------------


def test_ec2_zero_for_perfect_attributions() -> None:
    """EC2 must be 0.0 when φ = oracle.true_shapley."""
    phi_true = torch.tensor([0.5, -0.3, 0.1, 0.2])
    oracle = MockOracle(phi_true)
    players = _make_players()
    x = torch.randn(J, F, T)
    metric = EC2Metric()
    result = metric.evaluate(phi_true.clone(), x, _linear_clf, players, oracle=oracle)
    assert "ec2" in result
    assert result["ec2"] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 7. EC3 = 0 for perfect attributions
# ---------------------------------------------------------------------------


def test_ec3_zero_for_perfect_attributions() -> None:
    """EC3 must be 0.0 when φ = oracle.true_shapley (Pearson = 1)."""
    phi_true = torch.tensor([0.5, -0.3, 0.1, 0.2])
    oracle = MockOracle(phi_true)
    players = _make_players()
    x = torch.randn(J, F, T)
    metric = EC3Metric()
    result = metric.evaluate(phi_true.clone(), x, _linear_clf, players, oracle=oracle)
    assert "ec3" in result
    assert result["ec3"] == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 8. EfficiencyError with mock oracle (zero-vector φ and v(N)=v(∅)=0)
# ---------------------------------------------------------------------------


def test_efficiency_error_zero_when_consistent() -> None:
    """EfficiencyError = 0 when φ = oracle.true_shapley (efficiency axiom holds).

    The metric uses oracle.true_shapley.sum() as the reference.  When phi
    is exactly oracle.true_shapley, Σφ = oracle.true_shapley.sum()
    → efficiency_error = 0.
    """
    torch.manual_seed(42)
    x = torch.randn(J, F, T)
    phi_true = torch.tensor([0.5, -0.3, 0.1, 0.2])
    oracle = MockOracle(phi_true)
    players = _make_players()
    metric = EfficiencyErrorMetric(n_mc=50)
    # phi_true.sum() = 0.5 (non-zero), MockOracle returns phi_true from true_shapley
    result = metric.evaluate(phi_true.clone(), x, _linear_clf, players, oracle=oracle)
    assert "efficiency_error" in result
    assert result["efficiency_error"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 9. Slow: KernelSHAP + GaussianOracle → EfficiencyError < 1e-3
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_efficiency_error_kernel_shap() -> None:
    """KernelSHAP with GaussianOracle imputer satisfies efficiency axiom < 1e-3.

    Design notes
    ------------
    KernelSHAP enforces ``Σφ = v(N)_ks − v(∅)_ks`` as a hard WLS
    constraint.  ``EfficiencyErrorMetric`` uses ``oracle.true_shapley.sum()``
    as the reference, which satisfies the efficiency axiom by construction.

    To make the test robust to MC noise, we use ``x = scale * ones`` so
    that ``f(x) = scale`` (large signal) while ``v(∅) ≈ 0`` (zero-mean
    GaussianOracle).  With ``scale = 50`` and ``n_completion_samples = 200``
    the expected efficiency error is ≈ 2e-4, well below the 1e-3 threshold
    even at 3σ.
    """
    from motionbench.attribution.kernel_shap import KernelShapAttributor
    from motionbench.oracles.gaussian_oracle import GaussianOracle
    from motionbench.utils.coalitions import ar1_cov, equicorr

    torch.manual_seed(0)
    np.random.seed(0)

    Sigma_joints = equicorr(J, 0.3)
    Sigma_time = ar1_cov(T, 0.5)
    oracle = GaussianOracle(Sigma_joints, Sigma_time)

    # Use a large-valued constant input so that |f(x)| = 50 >> MC noise.
    # For a linear classifier mean(x), f(50*ones) = 50.
    # v(∅) = E[mean(x̃)] = 0 (zero-mean oracle), so v(N) - v(∅) = 50.
    # MC noise in v(∅) ≈ std_marginal / sqrt(n_completion) ≈ 0.15/sqrt(200)
    # → efficiency_error ≈ 0.011/50 = 2e-4 << 1e-3.
    x = torch.ones(J, F, T, dtype=torch.float32) * 50.0

    players = MockPlayerSet(J, F, T, M)

    attr = KernelShapAttributor(
        _linear_clf,
        oracle,
        n_samples=512,
        n_completion_samples=200,
        seed=42,
    )
    phi = attr.attribute(x, players)

    metric = EfficiencyErrorMetric(n_mc=1000, oracle_seed=1)
    result = metric.evaluate(phi, x, _linear_clf, players, oracle=oracle)
    assert result["efficiency_error"] < 1e-3, (
        f"EfficiencyError={result['efficiency_error']:.2e} ≥ 1e-3 "
        f"(Σφ={phi.sum():.6f}, expected≈50)"
    )


# ---------------------------------------------------------------------------
# 10. Return types are always dict[str, float]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("MetricCls", METRIC_CLASSES)
def test_return_type_is_dict_str_float(MetricCls: type) -> None:
    """evaluate() must always return dict[str, float].

    Args:
        MetricCls: Metric class to test.
    """
    phi_true = torch.tensor([0.5, -0.3, 0.1, 0.2])
    oracle = MockOracle(phi_true)
    players = _make_players()
    x = torch.randn(J, F, T)
    metric = MetricCls()
    result = metric.evaluate(phi_true.clone(), x, _linear_clf, players, oracle=oracle)
    assert isinstance(result, dict), f"{MetricCls.__name__} must return dict"
    for k, v in result.items():
        assert isinstance(k, str), f"Key {k!r} is not str"
        assert isinstance(v, float), f"Value for {k!r} is not float: {v!r}"
