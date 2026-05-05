"""Tests for motionbench.imputers.empirical.

Covers:
1. Shape contract: impute() returns (n_samples, J, F, T).
2. Observed-preservation contract: output[:, mask] == x_obs[mask].
3. Pre-fit guard: impute() before fit() raises RuntimeError.
4. Convergence (slow): EmpiricalConditionalImputer Shapley values converge
   to GaussianOracle within 0.15 on Gaussian data with N=5000 training rows.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from motionbench.imputers.empirical import (
    EmpiricalConditionalImputer,
    KNNConditionalImputer,
    VineCopulaImputer,
)
from tests.conftest import F, J, T

# ---------------------------------------------------------------------------
# Shared test dataset
# ---------------------------------------------------------------------------

N_TRAIN = 80


class _MockDataset:
    """Minimal dataset of (J, F, T) Gaussian samples for testing."""

    def __init__(self, n: int = N_TRAIN, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self._data = torch.tensor(
            rng.standard_normal((n, J, F, T)), dtype=torch.float32
        )
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._data[idx], torch.tensor(0)


def _all_imputers() -> list[tuple[str, object]]:
    """Return (name, imputer) pairs for all three empirical imputers."""
    return [
        ("KNN", KNNConditionalImputer(k=5)),
        ("Empirical", EmpiricalConditionalImputer()),
        ("VineCopula", VineCopulaImputer(max_vine_dim=5)),
    ]


# ---------------------------------------------------------------------------
# 1. Shape test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,imputer", _all_imputers())
def test_knn_shape(name, imputer, x_sample, mask_half):
    """impute() returns (n_samples, J, F, T) for each empirical imputer."""
    ds = _MockDataset()
    imputer.fit(ds)
    n = 6
    out = imputer.impute(x_sample, mask_half, n_samples=n, seed=0)
    assert out.shape == (n, J, F, T), (
        f"{name}: expected ({n}, {J}, {F}, {T}), got {out.shape}"
    )
    assert out.dtype == torch.float32, f"{name}: expected float32, got {out.dtype}"


# ---------------------------------------------------------------------------
# 2. Observed-preservation test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,imputer", _all_imputers())
def test_knn_observed_preserved(name, imputer, x_sample, mask_half):
    """output[:, mask] == x_obs[mask] for all completions."""
    ds = _MockDataset()
    imputer.fit(ds)
    out = imputer.impute(x_sample, mask_half, n_samples=8, seed=1)
    for i in range(out.shape[0]):
        assert torch.allclose(out[i][mask_half], x_sample[mask_half]), (
            f"{name}: sample {i} does not preserve observed entries."
        )


# ---------------------------------------------------------------------------
# 3. Pre-fit guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,imputer", _all_imputers())
def test_knn_fit_required(name, imputer, x_sample, mask_half):
    """impute() before fit() raises RuntimeError."""
    with pytest.raises(RuntimeError):
        imputer.impute(x_sample, mask_half, n_samples=1)


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------


def test_empirical_empty_mask(x_sample):
    """Empty mask (no observed entries) returns (n, J, F, T) from training pool."""
    imp = EmpiricalConditionalImputer()
    imp.fit(_MockDataset())
    empty_mask = torch.zeros(J, F, T, dtype=torch.bool)
    out = imp.impute(x_sample, empty_mask, n_samples=4, seed=2)
    assert out.shape == (4, J, F, T)


def test_empirical_full_mask(x_sample):
    """Full mask (all observed) returns copies of x_obs."""
    imp = EmpiricalConditionalImputer()
    imp.fit(_MockDataset())
    full_mask = torch.ones(J, F, T, dtype=torch.bool)
    out = imp.impute(x_sample, full_mask, n_samples=3, seed=3)
    for i in range(3):
        assert torch.allclose(out[i], x_sample), (
            f"Full mask: sample {i} differs from x_obs."
        )


def test_knn_deterministic(x_sample, mask_half):
    """Same seed → identical outputs."""
    ds = _MockDataset()
    imp_a = KNNConditionalImputer(k=5)
    imp_b = KNNConditionalImputer(k=5)
    imp_a.fit(ds)
    imp_b.fit(ds)
    out_a = imp_a.impute(x_sample, mask_half, n_samples=5, seed=99)
    out_b = imp_b.impute(x_sample, mask_half, n_samples=5, seed=99)
    assert torch.allclose(out_a, out_b), "Same seed must produce identical output."


def test_vine_copula_is_on_manifold():
    """All three empirical imputers report is_on_manifold=True."""
    for _, imp in _all_imputers():
        assert imp.is_on_manifold is True, f"{imp.__class__.__name__}.is_on_manifold"


# ---------------------------------------------------------------------------
# 4. Convergence test (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_empirical_convergence():
    """EmpiricalConditionalImputer Shapley values converge to GaussianOracle.

    With N=5000 Gaussian training samples, the empirical conditional
    approximation should match GaussianOracle Shapley values within 0.15
    absolute tolerance (Monte Carlo noise expected < 0.05 for n_mc=200).

    Uses M=4 temporal players covering T//M consecutive frames each,
    and a linear classifier f(x) = mean(x[0, 0, :]) (first joint, first coord).
    """
    from motionbench.oracles.gaussian_oracle import GaussianOracle
    from motionbench.utils.coalitions import (
        enumerate_coalitions,
        solve_shapley_wls,
    )

    rng = np.random.default_rng(42)

    # Gaussian data: AR(1) temporal correlation, equicorrelated joints.
    alpha_t = 0.7   # temporal AR(1)
    rho_j = 0.5     # joint equicorrelation
    J_c, F_c, T_c = 3, 2, 8
    M = 4           # temporal players
    N_train = 5000
    n_mc = 200      # MC samples per coalition

    # Build covariance matrices.
    lag = np.abs(np.arange(T_c)[:, None] - np.arange(T_c)[None, :])
    Sigma_time = alpha_t ** lag
    Sigma_joints = rho_j * np.ones((J_c, J_c)) + (1 - rho_j) * np.eye(J_c)

    oracle = GaussianOracle(Sigma_joints=Sigma_joints, Sigma_time=Sigma_time)

    # Generate training data from the Gaussian model.
    L_t = np.linalg.cholesky(Sigma_time + 1e-8 * np.eye(T_c))
    L_j = np.linalg.cholesky(Sigma_joints + 1e-8 * np.eye(J_c))
    z = rng.standard_normal((N_train, J_c, F_c, T_c))
    x_train = np.einsum("tT,njfT->njft", L_t, z)
    x_train = np.einsum("jJ,nJft->njft", L_j, x_train)
    train_tensor = torch.tensor(x_train, dtype=torch.float32)

    class _GaussDataset:
        def __len__(self) -> int:
            return N_train

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            return train_tensor[idx], torch.tensor(0)

    # Linear classifier: mean over first joint, first coord.
    def clf(x: torch.Tensor) -> torch.Tensor:
        return x[:, 0, 0, :].mean(dim=-1)

    # Test point.
    x_np = rng.standard_normal((J_c, F_c, T_c))
    x_t = torch.tensor(x_np, dtype=torch.float32)

    # Temporal players: player i covers frames [i*T_c//M, (i+1)*T_c//M).
    boundaries = [(i * T_c // M, (i + 1) * T_c // M) for i in range(M)]

    def make_mask(z_row: np.ndarray) -> torch.Tensor:
        mask = torch.zeros(J_c, F_c, T_c, dtype=torch.bool)
        for p, (t0, t1) in enumerate(boundaries):
            if z_row[p]:
                mask[:, :, t0:t1] = True
        return mask

    # Enumerate all 2^M coalitions.
    coalitions, weights = enumerate_coalitions(M)

    def compute_shapley(imputer_obj: object) -> np.ndarray:
        values = np.zeros(len(coalitions), dtype=np.float64)
        for i, z_row in enumerate(coalitions):
            mask_i = make_mask(z_row)
            if int(z_row.sum()) == M:
                with torch.no_grad():
                    values[i] = float(clf(x_t.unsqueeze(0)).item())
            elif int(z_row.sum()) == 0:
                samps = imputer_obj.impute(x_t, mask_i, n_samples=n_mc, seed=int(i))  # type: ignore[union-attr]
                with torch.no_grad():
                    values[i] = float(clf(samps).mean().item())
            else:
                samps = imputer_obj.impute(x_t, mask_i, n_samples=n_mc, seed=int(i))  # type: ignore[union-attr]
                with torch.no_grad():
                    values[i] = float(clf(samps).mean().item())
        v_empty = values[np.all(coalitions == 0, axis=1)][0]
        v_full = values[np.all(coalitions == 1, axis=1)][0]
        return solve_shapley_wls(coalitions, values, weights, float(v_empty), float(v_full))

    # Fit empirical imputer.
    emp = EmpiricalConditionalImputer(bandwidth="auto", eta=0.95)
    emp.fit(_GaussDataset())
    phi_emp = compute_shapley(emp)

    # Exact Shapley via GaussianOracle.
    phi_oracle = compute_shapley(oracle)

    max_abs_diff = float(np.max(np.abs(phi_emp - phi_oracle)))
    assert max_abs_diff < 0.15, (
        f"EmpiricalConditionalImputer Shapley values diverge from GaussianOracle "
        f"by {max_abs_diff:.4f} > 0.15.\n"
        f"  phi_emp    = {phi_emp}\n"
        f"  phi_oracle = {phi_oracle}"
    )
