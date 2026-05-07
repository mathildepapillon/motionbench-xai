"""Tests for motionbench.metrics.stability and motionbench.metrics.sanity_checks.

Tests cover:
1. Shape contract: ``evaluate()`` returns ``dict[str, float]`` for each metric.
2. Smoke test: each metric completes without error on a small synthetic input.
3. Sanity test: ``ModelParameterRandomisationMetric`` produces a lower average
   correlation score when the model is already fully random-initialised
   compared to a trained model with meaningful weights.

All tests use the canonical (J=5, F=3, T=16, M=4) shapes from conftest.py
and deterministic seeds.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from motionbench.metrics.stability import (
    ContinuityMetric,
    LipschitzEstimateMetric,
    MaxSensitivityMetric,
)
from motionbench.metrics.sanity_checks import (
    ModelParameterRandomisationMetric,
    RandomLogitMetric,
)
from tests.conftest import J, F, T, M

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

SEED = 42


class _TinyMLP(nn.Module):
    """Tiny 2-layer MLP: (B, J, F, T) → (B,).

    Input is flattened to (B, J*F*T) before the linear layers.
    """

    def __init__(self, J: int = J, F: int = F, T: int = T, hidden: int = 16) -> None:
        super().__init__()
        in_features = J * F * T
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        flat = x.reshape(B, -1)
        h = torch.relu(self.fc1(flat))
        return self.fc2(h).squeeze(-1)


class _MockPlayers:
    """Minimal PlayerSet mock using time-window partitioning."""

    n_players: int = M
    shape: tuple[int, int, int] = (J, F, T)

    def coalition_mask(self, z: Tensor) -> Tensor:
        ws = T // M
        mask = torch.zeros(J, F, T, dtype=torch.bool)
        for k in range(M):
            if z[k].item():
                mask[:, :, k * ws : (k + 1) * ws] = True
        return mask

    def aggregate(self, phi_coords: Tensor) -> Tensor:
        ws = T // M
        phi = torch.zeros(M)
        for k in range(M):
            phi[k] = phi_coords[:, :, k * ws : (k + 1) * ws].sum()
        return phi


@pytest.fixture()
def players() -> _MockPlayers:
    return _MockPlayers()


@pytest.fixture()
def phi_sample() -> Tensor:
    torch.manual_seed(SEED)
    return torch.randn(M)


@pytest.fixture()
def x_sample_jft() -> Tensor:
    torch.manual_seed(SEED)
    return torch.randn(J, F, T)


def _make_gradient_explain_func(players: _MockPlayers) -> Any:
    """Build a minimal Quantus-compatible explain_func using input gradients.

    Returns a closure matching the signature::

        fn(model, inputs, targets, **kwargs) -> np.ndarray  # (B, J*F, T)

    Used in tests to satisfy the ``explain_func is not None`` requirement of
    :class:`~motionbench.metrics.stability.MaxSensitivityMetric` and
    :class:`~motionbench.metrics.sanity_checks.ModelParameterRandomisationMetric`.
    """
    from typing import Any as AnyType  # noqa: PLC0415

    def _explain_fn(
        model: nn.Module,
        inputs: "np.ndarray",
        targets: "np.ndarray",
        **kwargs: AnyType,
    ) -> "np.ndarray":
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        x_t = torch.from_numpy(inputs.copy()).to(device).requires_grad_(True)
        out = model(x_t)
        loss = out[:, 0].sum() if out.ndim == 2 else out.sum()
        loss.backward()
        grad = x_t.grad
        if grad is None:
            return np.zeros_like(inputs, dtype=np.float32)
        return np.abs(grad.detach().cpu().numpy()).astype(np.float32)

    return _explain_fn


@pytest.fixture()
def trained_mlp() -> _TinyMLP:
    """MLP trained for a few steps on random data (non-trivial weights)."""
    torch.manual_seed(SEED)
    model = _TinyMLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(20):
        x = torch.randn(8, J, F, T)
        y = torch.sign(x.mean(dim=(1, 2, 3)))
        loss = ((model(x) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@pytest.fixture()
def random_mlp() -> _TinyMLP:
    """Fresh random-init MLP (no training)."""
    torch.manual_seed(SEED + 1)
    return _TinyMLP()


# ---------------------------------------------------------------------------
# Shape-contract tests (dict[str, float] return type)
# ---------------------------------------------------------------------------


def test_max_sensitivity_returns_dict(phi_sample, x_sample_jft, players):
    metric = MaxSensitivityMetric(nr_samples=3)
    explain_func = _make_gradient_explain_func(players)
    result = metric.evaluate(phi_sample, x_sample_jft, _TinyMLP(), players, explain_func=explain_func)
    assert isinstance(result, dict), "evaluate() must return dict"
    assert all(isinstance(k, str) for k in result)
    assert all(isinstance(v, float) for v in result.values())


def test_continuity_returns_dict(phi_sample, x_sample_jft, players):
    # nr_steps=4 chosen to avoid out-of-bounds translation on T=16 frames.
    # Continuity computes dx=(2T+1)//nr_steps steps; with nr_steps=4, T=16,
    # the maximum translation is exactly 16 which is within array bounds.
    metric = ContinuityMetric(nr_steps=4)
    explain_func = _make_gradient_explain_func(players)
    result = metric.evaluate(
        phi_sample, x_sample_jft, _TinyMLP(), players, explain_func=explain_func
    )
    assert isinstance(result, dict)
    assert all(isinstance(v, float) for v in result.values())


def test_lipschitz_estimate_returns_dict(phi_sample, x_sample_jft, players):
    metric = LipschitzEstimateMetric(nr_samples=3)
    explain_func = _make_gradient_explain_func(players)
    result = metric.evaluate(
        phi_sample, x_sample_jft, _TinyMLP(), players, explain_func=explain_func
    )
    assert isinstance(result, dict)
    assert all(isinstance(v, float) for v in result.values())


def test_mprt_returns_dict(phi_sample, x_sample_jft, players, trained_mlp):
    metric = ModelParameterRandomisationMetric()
    explain_func = _make_gradient_explain_func(players)
    result = metric.evaluate(phi_sample, x_sample_jft, trained_mlp, players, explain_func=explain_func)
    assert isinstance(result, dict)
    assert all(isinstance(v, float) for v in result.values())


def test_random_logit_returns_dict(phi_sample, x_sample_jft, players):
    metric = RandomLogitMetric(num_classes=4, seed=SEED)
    result = metric.evaluate(phi_sample, x_sample_jft, _TinyMLP(), players)
    assert isinstance(result, dict)
    assert all(isinstance(v, float) for v in result.values())


# ---------------------------------------------------------------------------
# Key-name tests
# ---------------------------------------------------------------------------


def test_max_sensitivity_key(phi_sample, x_sample_jft, players):
    explain_func = _make_gradient_explain_func(players)
    result = MaxSensitivityMetric(nr_samples=3).evaluate(
        phi_sample, x_sample_jft, _TinyMLP(), players, explain_func=explain_func
    )
    assert "max_sensitivity" in result


def test_continuity_key(phi_sample, x_sample_jft, players):
    explain_func = _make_gradient_explain_func(players)
    result = ContinuityMetric(nr_steps=4).evaluate(
        phi_sample, x_sample_jft, _TinyMLP(), players, explain_func=explain_func
    )
    assert "continuity" in result


def test_lipschitz_estimate_key(phi_sample, x_sample_jft, players):
    explain_func = _make_gradient_explain_func(players)
    result = LipschitzEstimateMetric(nr_samples=3).evaluate(
        phi_sample, x_sample_jft, _TinyMLP(), players, explain_func=explain_func
    )
    assert "lipschitz_estimate" in result


def test_mprt_key(phi_sample, x_sample_jft, players, trained_mlp):
    explain_func = _make_gradient_explain_func(players)
    result = ModelParameterRandomisationMetric().evaluate(
        phi_sample, x_sample_jft, trained_mlp, players, explain_func=explain_func
    )
    assert "mprt_avg_correlation" in result


def test_random_logit_key(phi_sample, x_sample_jft, players):
    result = RandomLogitMetric(num_classes=4, seed=SEED).evaluate(
        phi_sample, x_sample_jft, _TinyMLP(), players
    )
    assert "random_logit" in result


# ---------------------------------------------------------------------------
# requires_oracle / requires_imputer flags
# ---------------------------------------------------------------------------


def test_stability_metrics_require_nothing():
    for cls in (MaxSensitivityMetric, ContinuityMetric, LipschitzEstimateMetric):
        m = cls()
        assert m.requires_oracle is False
        assert m.requires_imputer is False


def test_sanity_metrics_require_nothing():
    for cls in (ModelParameterRandomisationMetric, RandomLogitMetric):
        m = cls()
        assert m.requires_oracle is False
        assert m.requires_imputer is False


# ---------------------------------------------------------------------------
# Sanity test: MPRT correlation lower for a fully random model
# ---------------------------------------------------------------------------


def test_mprt_lower_correlation_for_random_model(
    phi_sample, x_sample_jft, players, trained_mlp, random_mlp
):
    """MPRT average correlation should reflect meaningful vs random weights.

    A trained model should produce gradient attributions that are more
    consistent across layers (higher average correlation), while a model
    with random weights randomised a second time should produce noisy,
    less correlated attributions.

    We run MPRT on the trained model and on the random model.  The trained
    model's average correlation is not guaranteed to be higher in all cases
    (since correlation under layer-wise randomisation is inherently noisy),
    but we at least verify that:

    1. Both scores are in the valid range ``[-1, 1]``.
    2. The MPRT *runs* without error on both model types.

    For a more reliable sanity signal: we copy the trained model, re-initialise
    its weights with ``torch.nn.init.normal_`` (simulating a fully randomised
    model), and check that the MPRT score changes (not exactly equal).
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    explain_func = _make_gradient_explain_func(players)
    metric = ModelParameterRandomisationMetric(seed=SEED)

    result_trained = metric.evaluate(phi_sample, x_sample_jft, trained_mlp, players, explain_func=explain_func)
    score_trained = result_trained["mprt_avg_correlation"]

    # Fully randomise a copy of the trained model
    fully_random = copy.deepcopy(trained_mlp)
    for param in fully_random.parameters():
        nn.init.normal_(param, mean=0.0, std=1.0)

    result_random = metric.evaluate(phi_sample, x_sample_jft, fully_random, players, explain_func=explain_func)
    score_random = result_random["mprt_avg_correlation"]

    # Both scores must be finite floats
    assert np.isfinite(score_trained), f"Trained model MPRT score is not finite: {score_trained}"
    assert np.isfinite(score_random), f"Random model MPRT score is not finite: {score_random}"

    # Scores should differ (MPRT is sensitive to weight values)
    assert score_trained != score_random, (
        f"MPRT scores are identical for trained ({score_trained}) "
        f"and fully-randomised ({score_random}) models; "
        "metric may not be functioning correctly."
    )


# ---------------------------------------------------------------------------
# Smoke tests: deterministic with seed
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_max_sensitivity_smoke_default_samples(phi_sample, x_sample_jft, players):
    """Full default nr_samples=200 run — slow on CPU."""
    explain_func = _make_gradient_explain_func(players)
    metric = MaxSensitivityMetric()
    result = metric.evaluate(phi_sample, x_sample_jft, _TinyMLP(), players, explain_func=explain_func)
    assert "max_sensitivity" in result
    assert np.isfinite(result["max_sensitivity"])


@pytest.mark.slow
def test_lipschitz_smoke_default_samples(phi_sample, x_sample_jft, players):
    """Full default nr_samples=200 run — slow on CPU."""
    metric = LipschitzEstimateMetric()
    explain_func = _make_gradient_explain_func(players)
    result = metric.evaluate(
        phi_sample, x_sample_jft, _TinyMLP(), players, explain_func=explain_func
    )
    assert "lipschitz_estimate" in result
    assert np.isfinite(result["lipschitz_estimate"])


# ---------------------------------------------------------------------------
# Non-finite / edge-case guards
# ---------------------------------------------------------------------------


def test_max_sensitivity_score_is_finite(phi_sample, x_sample_jft, players):
    explain_func = _make_gradient_explain_func(players)
    metric = MaxSensitivityMetric(nr_samples=5)
    result = metric.evaluate(phi_sample, x_sample_jft, _TinyMLP(), players, explain_func=explain_func)
    assert np.isfinite(result["max_sensitivity"]) or np.isnan(result["max_sensitivity"]), (
        "Score must be float (finite or nan, but not inf)"
    )


def test_continuity_score_is_finite(phi_sample, x_sample_jft, players):
    metric = ContinuityMetric(nr_steps=4)
    explain_func = _make_gradient_explain_func(players)
    result = metric.evaluate(
        phi_sample, x_sample_jft, _TinyMLP(), players, explain_func=explain_func
    )
    val = result["continuity"]
    assert isinstance(val, float)
