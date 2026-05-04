"""Tests for motionbench.attribution.kernel_shap.KernelShapAttributor.

Test plan
---------
1. ``test_kernelshap_shape``      — attribute() returns (M,) for K=4 and K=8.
2. ``test_kernelshap_efficiency`` — |Σφ − (v(N) − v(∅))| < 0.1.
3. ``test_kernelshap_matches_oracle`` — with GaussianOracle imputer, values
   match oracle.true_shapley within MC tolerance for K=4. ``@pytest.mark.slow``
4. ``test_zero_imputer_runs``     — runs with ZeroImputer without error.

Fixtures
--------
All tests use the canonical shape J=5, F=3, T=16 from conftest.py.
A ``MockPlayerSet`` (temporal windows) is defined here following the same
pattern as in ``test_players_contract.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from motionbench.imputers.off_manifold import ZeroImputer
from torch import Tensor

from motionbench.attribution.kernel_shap import (
    KernelShapAttributor,
    _MotionBenchMasker,
)
from motionbench.players.base import PlayerSet

# ---------------------------------------------------------------------------
# Shared test parameters — match conftest.py canonical shape
# ---------------------------------------------------------------------------

J, F, T = 5, 3, 16


# ---------------------------------------------------------------------------
# Minimal PlayerSet implementation (temporal windows)
# ---------------------------------------------------------------------------


class MockPlayerSet(PlayerSet):
    """Equal-width temporal windows player set for testing.

    Args:
        J: Number of joints.
        F: Features per joint.
        T: Time-steps.
        M: Number of windows.  Must divide T evenly.
    """

    def __init__(self, J: int, F: int, T: int, M: int) -> None:
        self._J = J
        self._F = F
        self._T = T
        self._M = M
        if T % M != 0:
            raise ValueError(f"T={T} must be divisible by M={M} for MockPlayerSet.")
        self._ws = T // M

    @property
    def n_players(self) -> int:
        return self._M

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._J, self._F, self._T

    def coalition_mask(self, z: Tensor) -> Tensor:
        """Expand (M,) coalition indicator to (J, F, T) element mask.

        Args:
            z: ``(M,)`` binary int/bool tensor.  1 = player present.

        Returns:
            ``(J, F, T)`` bool tensor.

        Raises:
            ValueError: if z.shape != (M,).
        """
        if z.shape != (self._M,):
            raise ValueError(f"Expected z.shape==({self._M},); got {tuple(z.shape)}.")
        mask = torch.zeros(self._J, self._F, self._T, dtype=torch.bool)
        for k in range(self._M):
            if z[k]:
                t0 = k * self._ws
                t1 = (k + 1) * self._ws
                mask[:, :, t0:t1] = True
        return mask

    def aggregate(self, phi_coords: Tensor) -> Tensor:
        """Aggregate (J, F, T) attribution tensor to (M,) player level.

        Args:
            phi_coords: ``(J, F, T)`` float tensor.

        Returns:
            ``(M,)`` float tensor — sum over each window's coordinates.

        Raises:
            ValueError: if phi_coords.shape != (J, F, T).
        """
        if phi_coords.shape != (self._J, self._F, self._T):
            raise ValueError(
                f"Expected phi_coords.shape=={self.shape}; got {tuple(phi_coords.shape)}."
            )
        phi = torch.zeros(self._M)
        for k in range(self._M):
            t0 = k * self._ws
            t1 = (k + 1) * self._ws
            phi[k] = phi_coords[:, :, t0:t1].sum()
        return phi


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _linear_clf(x: Tensor) -> Tensor:
    """Linear classifier: mean over all elements.  (B, J, F, T) → (B,)."""
    return x.mean(dim=(1, 2, 3))


def _make_zero_imputer() -> ZeroImputer:
    """Return a ZeroImputer (fit is a no-op; impute works without it)."""
    return ZeroImputer()


# ---------------------------------------------------------------------------
# 1. Shape test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("K", [4, 8])
def test_kernelshap_shape(K: int) -> None:
    """attribute() must return a (M,) float32 tensor for K=4 and K=8 windows.

    Args:
        K: Number of temporal windows (players).
    """
    x = torch.randn(J, F, T)
    players = MockPlayerSet(J, F, T, K)
    imputer = _make_zero_imputer()

    attr = KernelShapAttributor(
        _linear_clf,
        imputer,
        n_samples=128,
        n_completion_samples=1,
        seed=0,
    )
    phi = attr.attribute(x, players)

    assert phi.shape == (K,), f"Expected ({K},), got {tuple(phi.shape)}"
    assert phi.dtype == torch.float32, f"Expected float32, got {phi.dtype}"
    assert not phi.isnan().any(), "Shapley values must not be NaN"


# ---------------------------------------------------------------------------
# 2. Efficiency test
# ---------------------------------------------------------------------------


def test_kernelshap_efficiency() -> None:
    """Efficiency axiom: |Σφ − (v(N) − v(∅))| < 0.1.

    KernelExplainer's WLS solve enforces efficiency as a hard constraint;
    the tolerance 0.1 covers floating-point rounding in the WLS solve
    and any residual MC noise from the imputer.
    """
    K = 4
    x = torch.randn(J, F, T)
    players = MockPlayerSet(J, F, T, K)
    imputer = _make_zero_imputer()

    attr = KernelShapAttributor(
        _linear_clf,
        imputer,
        n_samples=256,
        n_completion_samples=1,
        seed=42,
    )
    phi = attr.attribute(x, players)

    # v(N): classifier on the original sequence
    v_full = float(_linear_clf(x.unsqueeze(0)).item())

    # v(∅): classifier on the empty-coalition mean completion (all zeros for ZeroImputer)
    empty_mask = torch.zeros(J, F, T, dtype=torch.bool)
    completions = imputer.impute(x, empty_mask, n_samples=1)
    v_empty = float(_linear_clf(completions).item())

    efficiency_error = abs(phi.sum().item() - (v_full - v_empty))
    assert efficiency_error < 0.1, (
        f"Efficiency axiom violated: |Σφ − (v(N)−v(∅))| = {efficiency_error:.6f} ≥ 0.1\n"
        f"  phi={phi.tolist()}, sum={phi.sum().item():.6f}, "
        f"  v(N)-v(∅)={v_full - v_empty:.6f}"
    )


# ---------------------------------------------------------------------------
# 3. Oracle-matching test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_kernelshap_matches_oracle() -> None:
    """KernelSHAP with GaussianOracle ≈ oracle.true_shapley within MC tolerance.

    Uses a linear classifier so v(S) = f(E[x|x_S]) = E[f(x)|x_S] exactly,
    meaning the mean-completion estimator is unbiased.  With K=4 players
    (2^4 = 16 coalitions, fully enumerated), the residual error is purely
    from MC approximation of the conditional mean.
    """
    from motionbench.oracles.gaussian_oracle import GaussianOracle
    from motionbench.utils.coalitions import ar1_cov, equicorr

    K = 4
    rng = np.random.default_rng(0)

    Sigma_joints = equicorr(J, 0.3)
    Sigma_time = ar1_cov(T, 0.5)
    oracle = GaussianOracle(Sigma_joints, Sigma_time)

    # Draw a sample from the Gaussian model
    x_np = oracle._sample_unconditional(1, J, F, T, rng)[0]
    x = torch.tensor(x_np, dtype=torch.float32)

    players = MockPlayerSet(J, F, T, K)

    # --- Ground-truth Shapley values via oracle --------------------------------
    phi_oracle = oracle.true_shapley(
        x, _linear_clf, players, n_mc=500, n_coalitions=1000, seed=1
    )

    # --- KernelSHAP with GaussianOracle as imputer ----------------------------
    attr = KernelShapAttributor(
        _linear_clf,
        oracle,
        n_samples=2**10,
        n_completion_samples=100,
        seed=42,
    )
    phi_ks = attr.attribute(x, players)

    # Tolerance: 3σ MC upper bound.
    # For a linear clf and unit-scale Gaussian data with J=5,F=3,T=16,
    # std(f(x)) ≈ 1/sqrt(J*F*T) ≈ 0.065.  With n_mc=500 completions and
    # M=4 coalitions, σ per Shapley value < 0.01.  We use 0.05 as a
    # conservative 5σ bound to avoid flakiness.
    max_diff = (phi_ks - phi_oracle).abs().max().item()
    assert max_diff < 0.05, (
        f"KernelSHAP deviates from oracle: max|φ_ks − φ_oracle| = {max_diff:.4f} ≥ 0.05\n"
        f"  phi_ks={phi_ks.tolist()}\n"
        f"  phi_oracle={phi_oracle.tolist()}"
    )


# ---------------------------------------------------------------------------
# 4. ZeroImputer smoke test
# ---------------------------------------------------------------------------


def test_zero_imputer_runs() -> None:
    """KernelShapAttributor runs with ZeroImputer without error.

    Verifies that the full attribute() pipeline completes successfully
    and returns non-NaN values when using the simplest off-manifold imputer.
    """
    K = 4
    x = torch.randn(J, F, T)
    players = MockPlayerSet(J, F, T, K)
    imputer = _make_zero_imputer()

    attr = KernelShapAttributor(
        _linear_clf,
        imputer,
        n_samples=64,
        n_completion_samples=1,
        seed=7,
    )
    phi = attr.attribute(x, players)

    assert phi.shape == (K,), f"Expected ({K},), got {tuple(phi.shape)}"
    assert not phi.isnan().any(), "phi must not contain NaN"
    assert not phi.isinf().any(), "phi must not contain Inf"


# ---------------------------------------------------------------------------
# 5. _MotionBenchMasker unit tests
# ---------------------------------------------------------------------------


def test_masker_shape_full_coalition() -> None:
    """Masker with all-ones mask returns (1, J*F*T) mean completion."""
    K = 4
    x = torch.randn(J, F, T)
    players = MockPlayerSet(J, F, T, K)
    imputer = _make_zero_imputer()

    masker = _MotionBenchMasker(x, players, imputer, n_completion_samples=1)
    mask = np.ones(K, dtype=bool)
    (out,) = masker(mask, mask.astype(np.float64))

    assert out.shape == (1, J * F * T), f"Expected (1, {J*F*T}), got {out.shape}"


def test_masker_empty_coalition_zeros_out() -> None:
    """ZeroImputer + empty coalition mask → mean completion is all zeros."""
    K = 4
    x = torch.ones(J, F, T)  # non-zero so we can detect them
    players = MockPlayerSet(J, F, T, K)
    imputer = _make_zero_imputer()

    masker = _MotionBenchMasker(x, players, imputer, n_completion_samples=1)
    mask = np.zeros(K, dtype=bool)
    (out,) = masker(mask, mask.astype(np.float64))

    np.testing.assert_allclose(
        out, 0.0, atol=1e-6, err_msg="Empty coalition with ZeroImputer must give all zeros."
    )


def test_masker_full_coalition_preserves_x() -> None:
    """ZeroImputer + full coalition mask → mean completion equals x."""
    K = 4
    x = torch.randn(J, F, T)
    players = MockPlayerSet(J, F, T, K)
    imputer = _make_zero_imputer()

    masker = _MotionBenchMasker(x, players, imputer, n_completion_samples=3)
    mask = np.ones(K, dtype=bool)
    (out,) = masker(mask, mask.astype(np.float64))

    x_flat = x.numpy().flatten()
    np.testing.assert_allclose(
        out[0],
        x_flat,
        atol=1e-5,
        err_msg="Full coalition with ZeroImputer must preserve x.",
    )


def test_kernelshap_invalid_input_raises() -> None:
    """attribute() raises ValueError for a 4-D (batched) input."""
    K = 4
    x_batched = torch.randn(1, J, F, T)  # wrong: has batch dim
    players = MockPlayerSet(J, F, T, K)
    imputer = _make_zero_imputer()

    attr = KernelShapAttributor(_linear_clf, imputer, n_samples=32, seed=0)

    with pytest.raises(ValueError, match="batch dim"):
        attr.attribute(x_batched, players)
