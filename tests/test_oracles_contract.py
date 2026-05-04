"""Contract tests for motionbench.oracles.base.Oracle.

Verifies:
1. Oracle is abstract — cannot be instantiated directly.
2. Concrete subclasses satisfy all method signatures and output shapes.
3. conditional_sample preserves observed entries bit-for-bit.
4. true_shapley output satisfies the efficiency axiom (Σφ ≈ v(N) − v(∅)).
"""
from __future__ import annotations

import pytest
import torch
from torch import Tensor

from motionbench.oracles.base import Oracle
from tests.conftest import J, F, T, M

# ---------------------------------------------------------------------------
# Minimal mock PlayerSet (avoid importing unfinished module)
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
# Mock Oracle
# ---------------------------------------------------------------------------


class MockOracle(Oracle):
    """Returns zeros for hidden coords, preserves observed coords."""

    def conditional_sample(
        self,
        x_obs: Tensor,
        mask: Tensor,
        n: int,
        seed: int | None = None,
    ) -> Tensor:
        J, F, T = x_obs.shape
        out = torch.zeros(n, J, F, T)
        out[:, mask] = x_obs[mask]
        return out

    def true_shapley(
        self,
        x: Tensor,
        classifier,
        players,
        n_mc: int = 1000,
        seed: int | None = None,
    ) -> Tensor:
        # Trivial: equal credit to all players.
        return torch.full((players.n_players,), 1.0 / players.n_players)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


def test_oracle_is_abstract():
    with pytest.raises(TypeError):
        Oracle()  # type: ignore[abstract]


def test_mock_oracle_instantiates():
    oracle = MockOracle()
    assert oracle is not None


def test_conditional_sample_shape(x_sample, mask_half):
    oracle = MockOracle()
    out = oracle.conditional_sample(x_sample, mask_half, n=8)
    assert out.shape == (8, J, F, T), f"Expected (8,{J},{F},{T}), got {out.shape}"
    assert out.dtype == torch.float32


def test_conditional_sample_preserves_observed(x_sample, mask_half):
    """The oracle contract: observed entries must be identical in all samples."""
    oracle = MockOracle()
    out = oracle.conditional_sample(x_sample, mask_half, n=20)
    # Every sample should match x_sample at observed positions
    for i in range(out.shape[0]):
        assert torch.allclose(out[i][mask_half], x_sample[mask_half]), (
            f"Sample {i}: observed entries changed"
        )


def test_conditional_sample_n1(x_sample, mask_half):
    oracle = MockOracle()
    out = oracle.conditional_sample(x_sample, mask_half, n=1)
    assert out.shape == (1, J, F, T)


def test_conditional_sample_all_observed(x_sample):
    oracle = MockOracle()
    full_mask = torch.ones(J, F, T, dtype=torch.bool)
    out = oracle.conditional_sample(x_sample, full_mask, n=5)
    for i in range(5):
        assert torch.allclose(out[i], x_sample), "All observed → output == x_obs"


def test_true_shapley_shape():
    oracle = MockOracle()
    players = _MockPlayers()
    x = torch.randn(J, F, T)

    def clf(xb):
        return xb.mean(dim=(1, 2, 3))

    phi = oracle.true_shapley(x, clf, players)
    assert phi.shape == (M,), f"Expected ({M},), got {phi.shape}"
    assert phi.dtype == torch.float32


def test_true_shapley_efficiency(x_sample, classifier_fn):
    """The Mock oracle returns uniform φ — just verify shape and dtype here.

    Actual efficiency tests live in test_gaussian_oracle.py and
    test_copula_oracle.py where true_shapley is properly implemented.
    """
    oracle = MockOracle()
    players = _MockPlayers()
    phi = oracle.true_shapley(x_sample, classifier_fn, players)
    assert phi.shape == (M,)
    # Efficiency: this mock is trivial (uniform), so just check it sums to 1.
    assert abs(phi.sum().item() - 1.0) < 1e-5
