"""Tests for motionbench.attribution.sampled_coalitions.

Covers the fixed coalition design (exact regime, sampled regime, weights,
determinism) and the constrained WLS solve on it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from motionbench.attribution.sampled_coalitions import (
    DEFAULT_COALITION_SEED,
    EXACT_MAX_M,
    phi_from_values,
    sampled_coalition_set,
)
from motionbench.utils.coalitions import enumerate_coalitions, shapley_kernel_weight

# ---------------------------------------------------------------------------
# Exact regime (M <= EXACT_MAX_M)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("M", [2, 5, 8, 12])
def test_exact_regime_matches_enumeration(M):
    Z, w = sampled_coalition_set(M, budget=1024)
    Z_ref, w_ref = enumerate_coalitions(M)
    assert Z.shape == (2**M, M)
    np.testing.assert_array_equal(np.asarray(Z), np.asarray(Z_ref))
    np.testing.assert_allclose(w, w_ref, rtol=0, atol=0)


def test_exact_regime_kernel_weights_formula():
    M = 6
    Z, w = sampled_coalition_set(M, budget=999)
    for row, weight in zip(Z, w, strict=False):
        s = int(row.sum())
        assert weight == pytest.approx(shapley_kernel_weight(s, M), abs=0)


# ---------------------------------------------------------------------------
# Sampled regime (M > EXACT_MAX_M)
# ---------------------------------------------------------------------------


def test_sampled_boundary_rows_pinned():
    Z, w = sampled_coalition_set(17, budget=256)
    assert (Z[0] == 0).all() and (Z[1] == 1).all()
    assert w[0] == 1e6 and w[1] == 1e6
    # Interior rows: strictly between empty and full
    sizes = Z[2:].sum(axis=1)
    assert (sizes > 0).all() and (sizes < 17).all()


def test_sampled_row_count_and_uniqueness():
    M, budget = 17, 512
    Z, w = sampled_coalition_set(M, budget)
    assert Z.shape == (budget + 2, M)
    assert w.shape == (budget + 2,)
    uniq = {row.tobytes() for row in np.asarray(Z, dtype=np.int8)}
    assert len(uniq) == len(Z), "coalition rows must be distinct"


def test_sampled_determinism_and_seed_sensitivity():
    Z1, w1 = sampled_coalition_set(20, 300)
    Z2, w2 = sampled_coalition_set(20, 300)
    np.testing.assert_array_equal(Z1, Z2)
    np.testing.assert_array_equal(w1, w2)
    Z3, _ = sampled_coalition_set(20, 300, seed=DEFAULT_COALITION_SEED + 1)
    assert not np.array_equal(np.asarray(Z1), np.asarray(Z3))


def test_sampled_weights_positive_normalised():
    _Z, w = sampled_coalition_set(30, 400)
    interior = w[2:]
    assert (interior > 0).all()
    # Log-space rescaling puts the largest non-boundary weight at exactly 1.
    assert interior.max() == pytest.approx(1.0, abs=0)


def test_sampled_enumerated_sizes_carry_kernel_order():
    """Pair (1, M-1) has the largest kernel mass and is enumerated first."""
    M, budget = 17, 512
    Z, _w = sampled_coalition_set(M, budget)
    sizes = Z[2:].sum(axis=1)
    # 2*M rows of sizes 1 and M-1 must be fully enumerated within the budget.
    assert int((sizes == 1).sum()) == M
    assert int((sizes == M - 1).sum()) == M


def test_sampled_importance_weight_value():
    """Sampled rows carry (total leftover mass) / n_sampled, rescaled."""
    M, budget = 17, 512
    Z, w = sampled_coalition_set(M, budget)
    sizes = Z[2:].sum(axis=1)
    w_int = w[2:]
    # Enumerated sizes carry exact kernel weights, so relative weights within
    # the enumerated stratum must follow the kernel-weight ratio.
    w1 = w_int[sizes == 1][0]
    w2 = w_int[sizes == 2][0]
    expected_ratio = shapley_kernel_weight(1, M) / shapley_kernel_weight(2, M)
    assert w1 / w2 == pytest.approx(expected_ratio, rel=1e-12)
    # All sampled rows of the leftover stratum share one uniform weight.
    enumerated = set()
    remaining = budget
    for s in range(1, M // 2 + 1):
        cnt = math.comb(M, s) + (math.comb(M, M - s) if s != M - s else 0)
        if cnt > remaining:
            break
        enumerated |= {s, M - s}
        remaining -= cnt
    leftover_rows = w_int[~np.isin(sizes, sorted(enumerated))]
    assert len(np.unique(leftover_rows)) == 1


# ---------------------------------------------------------------------------
# phi_from_values
# ---------------------------------------------------------------------------


def _linear_game_values(Z, coeffs, v0=0.25):
    return v0 + np.asarray(Z, dtype=np.float64) @ coeffs


@pytest.mark.parametrize("M,budget", [(5, 64), (17, 512)])
def test_phi_linear_game_recovery(M, budget):
    """For a linear game the WLS solve is exact for any design: phi_i = c_i."""
    rng = np.random.default_rng(0)
    coeffs = rng.standard_normal(M)
    Z, w = sampled_coalition_set(M, budget)
    v = _linear_game_values(Z, coeffs)
    phi = phi_from_values(Z, w, v)
    np.testing.assert_allclose(phi, coeffs, atol=1e-7)


def test_phi_efficiency_axiom():
    rng = np.random.default_rng(1)
    M, budget = 17, 256
    Z, w = sampled_coalition_set(M, budget)
    v = rng.standard_normal(len(Z))
    phi = phi_from_values(Z, w, v)
    empty = int(np.flatnonzero(Z.sum(axis=1) == 0)[0])
    full = int(np.flatnonzero(Z.sum(axis=1) == M)[0])
    assert phi.sum() == pytest.approx(float(v[full] - v[empty]), abs=1e-5)


def test_phi_linear_in_values():
    """phi is linear in v for a fixed design (basis of value-blending)."""
    rng = np.random.default_rng(2)
    M, budget = 14, 128
    Z, w = sampled_coalition_set(M, budget)
    v1 = rng.standard_normal(len(Z))
    v2 = rng.standard_normal(len(Z))
    lam = 0.3
    direct = phi_from_values(Z, w, lam * v1 + (1 - lam) * v2)
    blended = lam * phi_from_values(Z, w, v1) + (1 - lam) * phi_from_values(Z, w, v2)
    np.testing.assert_allclose(direct, blended, atol=1e-9)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        sampled_coalition_set(1, 64)
    with pytest.raises(ValueError):
        sampled_coalition_set(EXACT_MAX_M + 1, 0)
