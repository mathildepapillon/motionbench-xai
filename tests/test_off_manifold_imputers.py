"""Tests for motionbench.imputers.off_manifold and motionbench.utils.masking.

Canonical test shape: J=5, F=3, T=16 (small but non-trivial).
All tests use deterministic seeds via the autouse fixture in conftest.py.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from motionbench.imputers.off_manifold import (
    GaussianNoiseImputer,
    MarginalDonorImputer,
    MeanImputer,
    ZeroImputer,
)
from motionbench.utils.masking import assert_mask_shape, coalition_to_element_mask

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

J, F, T = 5, 3, 16
N_TRAIN = 50


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _MockDataset:
    """Minimal dataset that yields (randn(J,F,T), tensor(0)) tuples."""

    def __init__(self, n: int = N_TRAIN, seed: int = 0) -> None:
        self._n = n
        torch.manual_seed(seed)
        self._data = torch.randn(n, J, F, T)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self._data[idx], torch.tensor(0)

    def __iter__(self):
        for i in range(self._n):
            yield self[i]


@pytest.fixture()
def mock_dataset() -> _MockDataset:
    """Return a simple mock dataset with N_TRAIN samples of shape (J, F, T)."""
    return _MockDataset(n=N_TRAIN, seed=0)


@pytest.fixture()
def x_obs() -> Tensor:
    """A single (J, F, T) float32 observation."""
    return torch.randn(J, F, T)


@pytest.fixture()
def mask_half() -> Tensor:
    """(J, F, T) bool mask: first T//2 timesteps observed."""
    m = torch.zeros(J, F, T, dtype=torch.bool)
    m[:, :, : T // 2] = True
    return m


@pytest.fixture()
def mask_all() -> Tensor:
    """(J, F, T) bool mask: all entries observed."""
    return torch.ones(J, F, T, dtype=torch.bool)


@pytest.fixture()
def mask_none() -> Tensor:
    """(J, F, T) bool mask: no entries observed."""
    return torch.zeros(J, F, T, dtype=torch.bool)


def _all_imputers_fitted(dataset: _MockDataset) -> list:
    """Return all four imputers after calling fit(dataset)."""
    return [
        ZeroImputer().fit(dataset),
        MeanImputer().fit(dataset),
        MarginalDonorImputer().fit(dataset),
        GaussianNoiseImputer(scale=1.0).fit(dataset),
    ]


# ---------------------------------------------------------------------------
# ZeroImputer tests
# ---------------------------------------------------------------------------


def test_zero_imputer_shape(x_obs: Tensor, mask_half: Tensor, mock_dataset: _MockDataset) -> None:
    """impute() returns (n_samples, J, F, T)."""
    imp = ZeroImputer().fit(mock_dataset)
    out = imp.impute(x_obs, mask_half, n_samples=7)
    assert out.shape == (7, J, F, T)


def test_zero_imputer_hidden_is_zero(x_obs: Tensor, mask_half: Tensor, mock_dataset: _MockDataset) -> None:
    """All hidden coordinates are exactly 0.0."""
    imp = ZeroImputer().fit(mock_dataset)
    out = imp.impute(x_obs, mask_half, n_samples=4)
    hidden = out[:, ~mask_half]
    assert torch.all(hidden == 0.0), "hidden entries must be exactly 0.0"


def test_zero_imputer_observed_preserved(x_obs: Tensor, mask_half: Tensor, mock_dataset: _MockDataset) -> None:
    """Observed coordinates exactly match x_obs."""
    imp = ZeroImputer().fit(mock_dataset)
    out = imp.impute(x_obs, mask_half, n_samples=4)
    for i in range(out.shape[0]):
        assert torch.equal(out[i][mask_half], x_obs[mask_half])


# ---------------------------------------------------------------------------
# MeanImputer tests
# ---------------------------------------------------------------------------


def test_mean_imputer_shape(x_obs: Tensor, mask_half: Tensor, mock_dataset: _MockDataset) -> None:
    """impute() returns (n_samples, J, F, T)."""
    imp = MeanImputer().fit(mock_dataset)
    out = imp.impute(x_obs, mask_half, n_samples=5)
    assert out.shape == (5, J, F, T)


def test_mean_imputer_observed_preserved(x_obs: Tensor, mask_half: Tensor, mock_dataset: _MockDataset) -> None:
    """Observed coordinates exactly match x_obs."""
    imp = MeanImputer().fit(mock_dataset)
    out = imp.impute(x_obs, mask_half, n_samples=3)
    for i in range(out.shape[0]):
        assert torch.equal(out[i][mask_half], x_obs[mask_half])


def test_mean_imputer_hidden_matches_mean(x_obs: Tensor, mask_half: Tensor, mock_dataset: _MockDataset) -> None:
    """Hidden coordinates equal the per-coordinate training mean."""
    imp = MeanImputer().fit(mock_dataset)
    out = imp.impute(x_obs, mask_half, n_samples=3)
    expected_mean = imp._mean  # type: ignore[attr-defined]
    for i in range(out.shape[0]):
        assert torch.equal(out[i][~mask_half], expected_mean[~mask_half])


def test_mean_imputer_before_fit_raises(x_obs: Tensor, mask_half: Tensor) -> None:
    """RuntimeError is raised when impute() is called before fit()."""
    imp = MeanImputer()
    with pytest.raises(RuntimeError, match="fit"):
        imp.impute(x_obs, mask_half, n_samples=1)


# ---------------------------------------------------------------------------
# MarginalDonorImputer tests
# ---------------------------------------------------------------------------


def test_marginal_donor_shape(x_obs: Tensor, mask_half: Tensor, mock_dataset: _MockDataset) -> None:
    """impute() returns (n_samples, J, F, T)."""
    imp = MarginalDonorImputer().fit(mock_dataset)
    out = imp.impute(x_obs, mask_half, n_samples=6, seed=42)
    assert out.shape == (6, J, F, T)


def test_marginal_donor_observed_preserved(x_obs: Tensor, mask_half: Tensor, mock_dataset: _MockDataset) -> None:
    """Observed coordinates exactly match x_obs."""
    imp = MarginalDonorImputer().fit(mock_dataset)
    out = imp.impute(x_obs, mask_half, n_samples=4, seed=7)
    for i in range(out.shape[0]):
        assert torch.equal(out[i][mask_half], x_obs[mask_half])


def test_marginal_donor_hidden_from_pool(x_obs: Tensor, mask_half: Tensor, mock_dataset: _MockDataset) -> None:
    """Hidden coordinates come from actual training sequences in the pool."""
    imp = MarginalDonorImputer().fit(mock_dataset)
    pool: Tensor = imp._pool  # type: ignore[attr-defined]  # (N, J, F, T)
    out = imp.impute(x_obs, mask_half, n_samples=10, seed=99)

    for i in range(out.shape[0]):
        hidden_vals = out[i][~mask_half]
        # Check that at least one training sequence matches the hidden values
        found = False
        for j in range(pool.shape[0]):
            if torch.equal(pool[j][~mask_half], hidden_vals):
                found = True
                break
        assert found, (
            f"Sample {i}: hidden coords do not match any training sequence in pool"
        )


# ---------------------------------------------------------------------------
# GaussianNoiseImputer tests
# ---------------------------------------------------------------------------


def test_gaussian_noise_shape(x_obs: Tensor, mask_half: Tensor, mock_dataset: _MockDataset) -> None:
    """impute() returns (n_samples, J, F, T)."""
    imp = GaussianNoiseImputer(scale=1.0).fit(mock_dataset)
    out = imp.impute(x_obs, mask_half, n_samples=8, seed=0)
    assert out.shape == (8, J, F, T)


def test_gaussian_noise_observed_preserved(x_obs: Tensor, mask_half: Tensor, mock_dataset: _MockDataset) -> None:
    """Observed coordinates exactly match x_obs."""
    imp = GaussianNoiseImputer(scale=1.0).fit(mock_dataset)
    out = imp.impute(x_obs, mask_half, n_samples=4, seed=1)
    for i in range(out.shape[0]):
        assert torch.equal(out[i][mask_half], x_obs[mask_half])


@pytest.mark.slow
def test_gaussian_noise_mean_convergence(x_obs: Tensor, mask_half: Tensor, mock_dataset: _MockDataset) -> None:
    """With scale=0, mean of hidden coords == training mean (to float32 precision)."""
    imp = GaussianNoiseImputer(scale=0.0).fit(mock_dataset)
    out = imp.impute(x_obs, mask_half, n_samples=1000, seed=123)
    expected = imp._mean[~mask_half]  # type: ignore[attr-defined]
    actual_mean = out[:, ~mask_half].mean(dim=0)
    # scale=0 means noise is suppressed; all hidden fills equal the mean exactly
    assert torch.allclose(actual_mean, expected, atol=1e-5), (
        f"mean mismatch: max diff = {(actual_mean - expected).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# All-imputer parametric tests
# ---------------------------------------------------------------------------


def test_all_imputers_n_samples_1(x_obs: Tensor, mask_half: Tensor, mock_dataset: _MockDataset) -> None:
    """All four imputers work with n_samples=1."""
    for imp in _all_imputers_fitted(mock_dataset):
        out = imp.impute(x_obs, mask_half, n_samples=1)
        assert out.shape == (1, J, F, T), f"{imp.name} failed with n_samples=1"


def test_all_imputers_n_samples_100(x_obs: Tensor, mask_half: Tensor, mock_dataset: _MockDataset) -> None:
    """All four imputers work with n_samples=100."""
    for imp in _all_imputers_fitted(mock_dataset):
        out = imp.impute(x_obs, mask_half, n_samples=100)
        assert out.shape == (100, J, F, T), f"{imp.name} failed with n_samples=100"


def test_all_imputers_full_mask(x_obs: Tensor, mask_all: Tensor, mock_dataset: _MockDataset) -> None:
    """With all-True mask, every output row equals x_obs."""
    for imp in _all_imputers_fitted(mock_dataset):
        out = imp.impute(x_obs, mask_all, n_samples=4, seed=0)
        for i in range(out.shape[0]):
            assert torch.equal(out[i], x_obs.to(dtype=torch.float32)), (
                f"{imp.name}: all-observed output row {i} does not match x_obs"
            )


def test_all_imputers_empty_mask(x_obs: Tensor, mask_none: Tensor, mock_dataset: _MockDataset) -> None:
    """With all-False mask, returns (n_samples, J, F, T) without error."""
    for imp in _all_imputers_fitted(mock_dataset):
        out = imp.impute(x_obs, mask_none, n_samples=4, seed=0)
        assert out.shape == (4, J, F, T), (
            f"{imp.name}: wrong shape with empty mask"
        )


def test_all_imputers_is_on_manifold_false(mock_dataset: _MockDataset) -> None:
    """All four imputers have is_on_manifold == False."""
    for imp in _all_imputers_fitted(mock_dataset):
        assert imp.is_on_manifold is False, (
            f"{imp.name}.is_on_manifold expected False, got {imp.is_on_manifold}"
        )


# ---------------------------------------------------------------------------
# Masking utilities tests
# ---------------------------------------------------------------------------


class _MockPlayerSet:
    """Minimal PlayerSet mock that returns a trivial coalition mask.

    Each player owns a contiguous block of joints; F and T are fully observed
    for each owned joint when z[k] = 1.
    """

    def __init__(self, n_players: int, J: int, F: int, T: int) -> None:
        self._n_players = n_players
        self._J = J
        self._F = F
        self._T = T

    @property
    def n_players(self) -> int:
        return self._n_players

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._J, self._F, self._T

    def coalition_mask(self, z: Tensor) -> Tensor:
        """Expand z (M,) to (J, F, T) with equal-width joint blocks."""
        mask = torch.zeros(self._J, self._F, self._T, dtype=torch.bool)
        block = self._J // self._n_players
        for k in range(self._n_players):
            if z[k]:
                j_start = k * block
                j_end = j_start + block if k < self._n_players - 1 else self._J
                mask[j_start:j_end, :, :] = True
        return mask


def test_masking_utils_coalition_to_mask() -> None:
    """coalition_to_element_mask returns (J, F, T) bool tensor."""
    M = 4
    ps = _MockPlayerSet(n_players=M, J=J, F=F, T=T)
    z = torch.ones(M, dtype=torch.bool)
    result = coalition_to_element_mask(z, ps)
    assert result.shape == (J, F, T)
    assert result.dtype == torch.bool
    # All players observed => all coordinates observed
    assert result.all()


def test_masking_utils_assert_shape_raises() -> None:
    """assert_mask_shape raises ValueError on wrong shape."""
    mask = torch.zeros(J, F, T, dtype=torch.bool)
    # Correct shape — no raise
    assert_mask_shape(mask, J, F, T)

    # Wrong shape — must raise
    bad_mask = torch.zeros(J + 1, F, T, dtype=torch.bool)
    with pytest.raises(ValueError, match="Expected mask shape"):
        assert_mask_shape(bad_mask, J, F, T)
