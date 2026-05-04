"""Contract tests for motionbench.metrics.base.BaseMetric.

Verifies:
1. BaseMetric is abstract.
2. evaluate() returns dict[str, float].
3. requires_oracle/requires_imputer class vars work correctly.
4. _check_deps raises ValueError when required dependencies are missing.
5. Subclasses with requires_oracle=True refuse None oracle.
"""
from __future__ import annotations

import pytest
import torch
from torch import Tensor

from motionbench.metrics.base import BaseMetric
from tests.conftest import J, F, T, M


# ---------------------------------------------------------------------------
# Minimal mocks
# ---------------------------------------------------------------------------


class _MockPlayers:
    n_players = M
    shape = (J, F, T)

    def aggregate(self, phi_coords):
        ws = T // M
        phi = torch.zeros(M)
        for k in range(M):
            phi[k] = phi_coords[:, :, k * ws:(k + 1) * ws].sum()
        return phi


class _MockOracle:
    def conditional_sample(self, x_obs, mask, n, seed=None):
        J, F, T = x_obs.shape
        return torch.zeros(n, J, F, T)

    def true_shapley(self, x, classifier, players, n_mc=1000, seed=None):
        return torch.zeros(players.n_players)


class _MockImputer:
    name = "mock"
    is_on_manifold = False

    def fit(self, train_data):
        return self

    def impute(self, x_obs, mask, n_samples, seed=None):
        J, F, T = x_obs.shape
        out = torch.zeros(n_samples, J, F, T)
        out[:, mask] = x_obs[mask]
        return out


# ---------------------------------------------------------------------------
# Mock Metrics
# ---------------------------------------------------------------------------


class MockMetric(BaseMetric):
    """Simple metric that returns a constant — no oracle, no imputer required."""

    requires_oracle = False
    requires_imputer = False

    def evaluate(self, phi, x, classifier, players, target=0, oracle=None, imputer=None):
        self._check_deps(oracle, imputer)
        return {"mock_score": float(phi.mean().item())}


class MockOracleMetric(BaseMetric):
    """Metric that requires an oracle."""

    requires_oracle = True
    requires_imputer = False

    def evaluate(self, phi, x, classifier, players, target=0, oracle=None, imputer=None):
        self._check_deps(oracle, imputer)
        # Compute a trivial metric using the oracle.
        phi_oracle = oracle.true_shapley(x, classifier, players)
        ec1 = float((phi - phi_oracle).abs().mean().item())
        return {"ec1": ec1}


class MockImpMetric(BaseMetric):
    """Metric that requires an imputer."""

    requires_oracle = False
    requires_imputer = True

    def evaluate(self, phi, x, classifier, players, target=0, oracle=None, imputer=None):
        self._check_deps(oracle, imputer)
        mask = torch.ones(J, F, T, dtype=torch.bool)
        _ = imputer.impute(x, mask, 1)
        return {"fid": 0.0}


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


def test_basemetric_is_abstract():
    with pytest.raises(TypeError):
        BaseMetric()  # type: ignore[abstract]


def test_mock_metric_instantiates():
    m = MockMetric()
    assert m is not None


def test_evaluate_returns_dict(x_sample, classifier_fn):
    m = MockMetric()
    players = _MockPlayers()
    phi = torch.randn(M)
    result = m.evaluate(phi, x_sample, classifier_fn, players)
    assert isinstance(result, dict), "evaluate() must return dict"
    assert all(isinstance(v, float) for v in result.values()), (
        "All values in result dict must be float"
    )


def test_evaluate_returns_string_keys(x_sample, classifier_fn):
    m = MockMetric()
    players = _MockPlayers()
    phi = torch.randn(M)
    result = m.evaluate(phi, x_sample, classifier_fn, players)
    assert all(isinstance(k, str) for k in result.keys())


def test_requires_oracle_default():
    m = MockMetric()
    assert m.requires_oracle is False


def test_requires_imputer_default():
    m = MockMetric()
    assert m.requires_imputer is False


def test_oracle_metric_raises_without_oracle(x_sample, classifier_fn):
    m = MockOracleMetric()
    players = _MockPlayers()
    phi = torch.randn(M)
    with pytest.raises(ValueError, match="requires_oracle"):
        m.evaluate(phi, x_sample, classifier_fn, players, oracle=None)


def test_oracle_metric_works_with_oracle(x_sample, classifier_fn):
    m = MockOracleMetric()
    players = _MockPlayers()
    phi = torch.randn(M)
    oracle = _MockOracle()
    result = m.evaluate(phi, x_sample, classifier_fn, players, oracle=oracle)
    assert "ec1" in result
    assert isinstance(result["ec1"], float)


def test_imputer_metric_raises_without_imputer(x_sample, classifier_fn):
    m = MockImpMetric()
    players = _MockPlayers()
    phi = torch.randn(M)
    with pytest.raises(ValueError, match="requires_imputer"):
        m.evaluate(phi, x_sample, classifier_fn, players, imputer=None)


def test_imputer_metric_works_with_imputer(x_sample, classifier_fn):
    m = MockImpMetric()
    players = _MockPlayers()
    phi = torch.randn(M)
    imp = _MockImputer().fit(None)  # type: ignore[arg-type]
    result = m.evaluate(phi, x_sample, classifier_fn, players, imputer=imp)
    assert isinstance(result, dict)


def test_name_property():
    m = MockMetric()
    assert isinstance(m.name, str)


def test_repr():
    m = MockMetric()
    r = repr(m)
    assert "MockMetric" in r
    assert "requires_oracle" in r
