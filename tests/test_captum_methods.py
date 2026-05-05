"""Tests for motionbench.attribution.captum_methods.

Verifies that all six Captum-based attributors:
1. Instantiate correctly with a tiny differentiable model.
2. Return ``(M,)`` float32 tensors from ``attribute()``.
3. Accept ``target`` and ``baseline`` kwargs without error.

All tests run on CPU in < 10 seconds total.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from motionbench.attribution.captum_methods import (
    DeepLiftAttributor,
    GradientShapAttributor,
    InputXGradientAttributor,
    IntegratedGradientsAttributor,
    SaliencyAttributor,
    SmoothGradAttributor,
)
from tests.conftest import F, J, M, T

# ---------------------------------------------------------------------------
# Shared test infrastructure
# ---------------------------------------------------------------------------

N_CLASSES = 3


def _make_classifier() -> nn.Module:
    """Tiny differentiable model: (B, J, F, T) → (B, N_CLASSES)."""
    lin = nn.Linear(J * F * T, N_CLASSES)

    class _Clf(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = lin

        def forward(self, x: Tensor) -> Tensor:
            return self.linear(x.flatten(1))

    return _Clf()


class _MockPlayers:
    """Temporal-window PlayerSet for testing (M equal-width windows)."""

    n_players: int = M
    shape: tuple[int, int, int] = (J, F, T)

    def aggregate(self, phi_coords: Tensor) -> Tensor:
        ws = T // M
        phi = torch.zeros(M)
        for k in range(M):
            phi[k] = phi_coords[:, :, k * ws : (k + 1) * ws].sum()
        return phi


@pytest.fixture()
def clf() -> nn.Module:
    """Shared tiny classifier fixture."""
    torch.manual_seed(0)
    return _make_classifier()


@pytest.fixture()
def x() -> Tensor:
    """Single ``(J, F, T)`` sample."""
    torch.manual_seed(1)
    return torch.randn(J, F, T)


@pytest.fixture()
def players() -> _MockPlayers:
    """Temporal-window player set with M players."""
    return _MockPlayers()


# ---------------------------------------------------------------------------
# Shape + dtype tests (one per method)
# ---------------------------------------------------------------------------


def test_integrated_gradients_shape(clf: nn.Module, x: Tensor, players: _MockPlayers) -> None:
    attr = IntegratedGradientsAttributor(clf, n_steps=5)
    phi = attr.attribute(x, players, target=0)
    assert phi.shape == (M,), f"Expected ({M},), got {phi.shape}"
    assert phi.dtype == torch.float32


def test_deeplift_shape(clf: nn.Module, x: Tensor, players: _MockPlayers) -> None:
    attr = DeepLiftAttributor(clf)
    phi = attr.attribute(x, players, target=0)
    assert phi.shape == (M,), f"Expected ({M},), got {phi.shape}"
    assert phi.dtype == torch.float32


def test_gradient_shap_shape(clf: nn.Module, x: Tensor, players: _MockPlayers) -> None:
    attr = GradientShapAttributor(clf, n_samples=5)
    phi = attr.attribute(x, players, target=0)
    assert phi.shape == (M,), f"Expected ({M},), got {phi.shape}"
    assert phi.dtype == torch.float32


def test_saliency_shape(clf: nn.Module, x: Tensor, players: _MockPlayers) -> None:
    attr = SaliencyAttributor(clf)
    phi = attr.attribute(x, players, target=0)
    assert phi.shape == (M,), f"Expected ({M},), got {phi.shape}"
    assert phi.dtype == torch.float32


def test_smooth_grad_shape(clf: nn.Module, x: Tensor, players: _MockPlayers) -> None:
    attr = SmoothGradAttributor(clf, nt_samples=5)
    phi = attr.attribute(x, players, target=0)
    assert phi.shape == (M,), f"Expected ({M},), got {phi.shape}"
    assert phi.dtype == torch.float32


def test_input_x_gradient_shape(clf: nn.Module, x: Tensor, players: _MockPlayers) -> None:
    attr = InputXGradientAttributor(clf)
    phi = attr.attribute(x, players, target=0)
    assert phi.shape == (M,), f"Expected ({M},), got {phi.shape}"
    assert phi.dtype == torch.float32


# ---------------------------------------------------------------------------
# requires_gradient property
# ---------------------------------------------------------------------------


def test_all_require_gradient(clf: nn.Module) -> None:
    methods = [
        IntegratedGradientsAttributor(clf),
        DeepLiftAttributor(clf),
        GradientShapAttributor(clf),
        SaliencyAttributor(clf),
        SmoothGradAttributor(clf),
        InputXGradientAttributor(clf),
    ]
    for m in methods:
        assert m.requires_gradient is True, f"{m.name} should require gradient"


# ---------------------------------------------------------------------------
# Baseline variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("baseline", ["zero", "mean", "gaussian"])
def test_ig_baseline_variants(
    clf: nn.Module, x: Tensor, players: _MockPlayers, baseline: str
) -> None:
    attr = IntegratedGradientsAttributor(clf, baseline=baseline, n_steps=3)  # type: ignore[arg-type]
    phi = attr.attribute(x, players, target=0)
    assert phi.shape == (M,)


@pytest.mark.parametrize("baseline", ["zero", "mean", "gaussian"])
def test_deeplift_baseline_variants(
    clf: nn.Module, x: Tensor, players: _MockPlayers, baseline: str
) -> None:
    attr = DeepLiftAttributor(clf, baseline=baseline)  # type: ignore[arg-type]
    phi = attr.attribute(x, players, target=0)
    assert phi.shape == (M,)


# ---------------------------------------------------------------------------
# Multi-target compatibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", [0, 1, 2])
def test_ig_different_targets(
    clf: nn.Module, x: Tensor, players: _MockPlayers, target: int
) -> None:
    attr = IntegratedGradientsAttributor(clf, n_steps=3)
    phi = attr.attribute(x, players, target=target)
    assert phi.shape == (M,)


# ---------------------------------------------------------------------------
# repr / name
# ---------------------------------------------------------------------------


def test_repr_contains_class_name(clf: nn.Module) -> None:
    attr = IntegratedGradientsAttributor(clf)
    assert "IntegratedGradientsAttributor" in repr(attr)
    assert "IntegratedGradientsAttributor" in attr.name
