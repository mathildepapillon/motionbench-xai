"""Contract tests for motionbench.attribution.base.BaseAttributor.

Verifies:
1. BaseAttributor is abstract.
2. attribute() returns (M,) float32 tensor.
3. Introspection properties have correct defaults.
"""
from __future__ import annotations

import pytest
import torch
from torch import Tensor

from motionbench.attribution.base import BaseAttributor
from tests.conftest import J, F, T, M


# ---------------------------------------------------------------------------
# Mock PlayerSet
# ---------------------------------------------------------------------------


class _MockPlayers:
    n_players = M
    shape = (J, F, T)

    def coalition_mask(self, z):
        ws = T // M
        mask = torch.zeros(J, F, T, dtype=torch.bool)
        for k in range(M):
            if z[k]:
                mask[:, :, k * ws:(k + 1) * ws] = True
        return mask

    def aggregate(self, phi_coords):
        ws = T // M
        phi = torch.zeros(M)
        for k in range(M):
            phi[k] = phi_coords[:, :, k * ws:(k + 1) * ws].sum()
        return phi


# ---------------------------------------------------------------------------
# Mock Attributor
# ---------------------------------------------------------------------------


class MockAttributor(BaseAttributor):
    """Returns uniform attribution (sum of input, split equally)."""

    def attribute(
        self,
        x: Tensor,
        players,
        target: int = 0,
    ) -> Tensor:
        # Compute full attribution over all coords, then aggregate.
        phi_coords = x.abs()  # simple stand-in for "attribution map"
        return players.aggregate(phi_coords)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


def test_baseattributor_is_abstract():
    with pytest.raises(TypeError):
        BaseAttributor(classifier=lambda x: x.mean(dim=(1, 2, 3)))  # type: ignore[abstract]


def test_mock_attributor_instantiates(classifier_fn):
    attr = MockAttributor(classifier_fn)
    assert attr is not None


def test_attribute_output_shape(x_sample, classifier_fn):
    attr = MockAttributor(classifier_fn)
    players = _MockPlayers()
    phi = attr.attribute(x_sample, players, target=0)
    assert phi.shape == (M,), f"Expected ({M},), got {phi.shape}"
    assert phi.dtype == torch.float32


def test_attribute_output_nonnegative_for_abs_attributor(x_sample, classifier_fn):
    """MockAttributor uses abs() so all attributions are non-negative."""
    attr = MockAttributor(classifier_fn)
    players = _MockPlayers()
    phi = attr.attribute(x_sample, players)
    assert (phi >= 0).all(), "Expected non-negative attributions from abs-mock"


def test_attribute_different_targets_allowed(x_sample, classifier_fn):
    """attribute() must accept any target int without error."""
    attr = MockAttributor(classifier_fn)
    players = _MockPlayers()
    for target in [0, 1, 2]:
        phi = attr.attribute(x_sample, players, target=target)
        assert phi.shape == (M,)


def test_name_property(classifier_fn):
    attr = MockAttributor(classifier_fn)
    assert isinstance(attr.name, str)
    assert len(attr.name) > 0


def test_requires_imputer_default(classifier_fn):
    attr = MockAttributor(classifier_fn)
    assert attr.requires_imputer is False


def test_requires_gradient_default(classifier_fn):
    attr = MockAttributor(classifier_fn)
    assert attr.requires_gradient is False


def test_repr(classifier_fn):
    attr = MockAttributor(classifier_fn)
    r = repr(attr)
    assert "MockAttributor" in r
