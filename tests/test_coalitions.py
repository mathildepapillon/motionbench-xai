"""Tests for motionbench.utils.coalitions.

Verifies shape, symmetry, and correctness of each coalition utility.
"""
from __future__ import annotations

import numpy as np
import pytest

from motionbench.utils.coalitions import (
    ar1_cov,
    enumerate_coalitions,
    equicorr,
    sample_kernelshap_coalitions,
    shapley_kernel_weight,
    solve_shapley_wls,
)

# ---------------------------------------------------------------------------
# ar1_cov
# ---------------------------------------------------------------------------


def test_ar1_cov_shape_and_symmetry() -> None:
    """ar1_cov returns square symmetric matrix with correct diagonal."""
    T = 8
    C = ar1_cov(T, alpha=0.7)
    assert C.shape == (T, T), f"Expected ({T},{T}), got {C.shape}"
    np.testing.assert_allclose(C, C.T, err_msg="ar1_cov must be symmetric")
    np.testing.assert_allclose(np.diag(C), np.ones(T), err_msg="diagonal must be 1")


def test_ar1_cov_values() -> None:
    """ar1_cov entries follow alpha^|t-t'|."""
    alpha = 0.5
    C = ar1_cov(4, alpha)
    np.testing.assert_allclose(C[0, 1], alpha, rtol=1e-10)
    np.testing.assert_allclose(C[0, 2], alpha**2, rtol=1e-10)
    np.testing.assert_allclose(C[0, 3], alpha**3, rtol=1e-10)


def test_ar1_cov_is_psd() -> None:
    """ar1_cov matrix is positive semi-definite."""
    C = ar1_cov(10, 0.8)
    eigvals = np.linalg.eigvalsh(C)
    assert eigvals.min() >= -1e-10, f"Minimum eigenvalue {eigvals.min():.3e} < 0."


# ---------------------------------------------------------------------------
# equicorr
# ---------------------------------------------------------------------------


def test_equicorr_diagonal_ones() -> None:
    """equicorr diagonal is 1 and off-diagonal is rho."""
    J, rho = 5, 0.4
    C = equicorr(J, rho)
    assert C.shape == (J, J)
    np.testing.assert_allclose(np.diag(C), np.ones(J))
    off = C[np.triu_indices(J, k=1)]
    np.testing.assert_allclose(off, rho * np.ones(len(off)))


def test_equicorr_symmetry() -> None:
    """equicorr is symmetric."""
    C = equicorr(6, 0.3)
    np.testing.assert_allclose(C, C.T)


# ---------------------------------------------------------------------------
# shapley_kernel_weight
# ---------------------------------------------------------------------------


def test_shapley_kernel_weight_boundary() -> None:
    """shapley_kernel_weight returns 0 for s=0 and s=M."""
    M = 5
    assert shapley_kernel_weight(0, M) == 0.0
    assert shapley_kernel_weight(M, M) == 0.0


def test_shapley_kernel_weight_interior() -> None:
    """shapley_kernel_weight is positive for 0 < s < M."""
    M = 4
    for s in range(1, M):
        w = shapley_kernel_weight(s, M)
        assert w > 0.0, f"Weight for s={s}, M={M} should be positive; got {w}."


def test_shapley_kernel_weight_symmetry() -> None:
    """w(s, M) = w(M-s, M) by construction of the SHAP kernel."""
    M = 6
    for s in range(1, M):
        assert shapley_kernel_weight(s, M) == pytest.approx(
            shapley_kernel_weight(M - s, M), rel=1e-10
        )


def test_shapley_kernel_weight_invalid() -> None:
    """shapley_kernel_weight raises ValueError for out-of-range s."""
    with pytest.raises(ValueError):
        shapley_kernel_weight(-1, 4)
    with pytest.raises(ValueError):
        shapley_kernel_weight(5, 4)


# ---------------------------------------------------------------------------
# enumerate_coalitions
# ---------------------------------------------------------------------------


def test_enumerate_coalitions_count() -> None:
    """enumerate_coalitions returns exactly 2^M rows."""
    for M in [3, 4, 5]:
        coalitions, weights = enumerate_coalitions(M)
        assert coalitions.shape == (2**M, M), (
            f"Expected ({2**M}, {M}), got {coalitions.shape}"
        )
        assert weights.shape == (2**M,)


def test_enumerate_coalitions_binary() -> None:
    """enumerate_coalitions rows are binary (0/1)."""
    coalitions, _ = enumerate_coalitions(4)
    assert set(np.unique(coalitions)).issubset({0, 1})


def test_enumerate_coalitions_boundary_weights() -> None:
    """All-zero and all-one rows have weight 0."""
    M = 4
    coalitions, weights = enumerate_coalitions(M)
    empty_idx = np.where(coalitions.sum(axis=1) == 0)[0][0]
    full_idx = np.where(coalitions.sum(axis=1) == M)[0][0]
    assert weights[empty_idx] == 0.0
    assert weights[full_idx] == 0.0


def test_enumerate_coalitions_raises_for_large_m() -> None:
    """enumerate_coalitions raises ValueError for M > 20."""
    with pytest.raises(ValueError, match="refusing"):
        enumerate_coalitions(21)


# ---------------------------------------------------------------------------
# sample_kernelshap_coalitions
# ---------------------------------------------------------------------------


def test_sample_kernelshap_coalitions_shape() -> None:
    """sample_kernelshap_coalitions returns (2*n_pairs, M) arrays."""
    M, n_pairs = 6, 50
    rng = np.random.default_rng(42)
    coalitions, weights = sample_kernelshap_coalitions(M, n_pairs, rng)
    assert coalitions.shape == (2 * n_pairs, M)
    assert weights.shape == (2 * n_pairs,)


def test_sample_kernelshap_coalitions_no_boundary() -> None:
    """Sampled coalitions never contain all-zero or all-one rows."""
    M, n_pairs = 5, 200
    rng = np.random.default_rng(7)
    coalitions, weights = sample_kernelshap_coalitions(M, n_pairs, rng)
    sizes = coalitions.sum(axis=1)
    assert (sizes > 0).all(), "Found all-zero coalition in sampled set."
    assert (sizes < M).all(), "Found all-one coalition in sampled set."


def test_sample_kernelshap_coalitions_weights_positive() -> None:
    """All returned weights are strictly positive."""
    M, n_pairs = 4, 30
    rng = np.random.default_rng(99)
    _, weights = sample_kernelshap_coalitions(M, n_pairs, rng)
    assert (weights > 0).all(), "Found non-positive weight."


# ---------------------------------------------------------------------------
# solve_shapley_wls
# ---------------------------------------------------------------------------


def test_solve_shapley_wls_efficiency() -> None:
    """Shapley values satisfy efficiency: sum(phi) = v_full - v_empty."""
    M = 4
    coalitions, weights = enumerate_coalitions(M)
    # Simple additive value function v(S) = sum of active players.
    values = coalitions.sum(axis=1).astype(np.float64)
    v_empty = 0.0
    v_full = float(M)
    phi = solve_shapley_wls(coalitions, values, weights, v_empty, v_full)
    assert phi.shape == (M,)
    np.testing.assert_allclose(
        phi.sum(), v_full - v_empty, atol=1e-6, err_msg="Efficiency axiom violated."
    )


def test_solve_shapley_wls_symmetric() -> None:
    """For a symmetric additive game, all Shapley values are equal."""
    M = 4
    coalitions, weights = enumerate_coalitions(M)
    values = coalitions.sum(axis=1).astype(np.float64)
    phi = solve_shapley_wls(coalitions, values, weights, 0.0, float(M))
    np.testing.assert_allclose(phi, np.ones(M), atol=1e-5)


def test_solve_shapley_wls_dummy_player() -> None:
    """A player that never contributes gets Shapley value 0."""
    M = 4
    # v(S) = S[0] + S[1] + S[2]  (player 3 is always dummy).
    coalitions, weights = enumerate_coalitions(M)
    values = coalitions[:, :3].sum(axis=1).astype(np.float64)
    v_full = 3.0
    v_empty = 0.0
    phi = solve_shapley_wls(coalitions, values, weights, v_empty, v_full)
    np.testing.assert_allclose(phi[3], 0.0, atol=1e-5)


def test_solve_shapley_wls_shape_mismatch() -> None:
    """solve_shapley_wls raises ValueError on shape mismatch."""
    coalitions, weights = enumerate_coalitions(3)
    with pytest.raises(ValueError):
        solve_shapley_wls(coalitions, np.ones(5), weights, 0.0, 1.0)
