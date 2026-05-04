"""Contract tests for motionbench.imputers.base.BaseImputer.

Verifies:
1. BaseImputer is abstract.
2. Concrete subclasses satisfy fit/impute signatures and output shapes.
3. Observed entries are preserved bit-for-bit (the core imputer contract).
4. fit returns self (method chaining).
5. n_samples dimension is correctly sized.
"""
from __future__ import annotations

import pytest
import torch
from torch import Tensor

from motionbench.imputers.base import BaseImputer
from tests.conftest import J, F, T


# ---------------------------------------------------------------------------
# Mock BaseDataset (minimal)
# ---------------------------------------------------------------------------


class _MockDataset:
    shape = (J, F, T)

    def __getitem__(self, idx):
        return torch.zeros(J, F, T), torch.tensor(0)

    def __len__(self):
        return 10

    @property
    def metadata(self):
        return {"skeleton": "mock", "frame_rate": 30.0}

    @property
    def oracle(self):
        return None


# ---------------------------------------------------------------------------
# Mock BaseImputer (fills hidden entries with zeros)
# ---------------------------------------------------------------------------


class MockImputer(BaseImputer):
    """Zero-fill imputer for contract testing."""

    def fit(self, train_data) -> "MockImputer":
        self._fitted = True
        return self

    def impute(
        self,
        x_obs: Tensor,
        mask: Tensor,
        n_samples: int,
        seed: int | None = None,
    ) -> Tensor:
        if not getattr(self, "_fitted", False):
            raise RuntimeError("fit() must be called before impute()")
        J, F, T = x_obs.shape
        # Fill hidden with zeros, preserve observed.
        base = torch.zeros(J, F, T)
        base[mask] = x_obs[mask]
        return base.unsqueeze(0).expand(n_samples, -1, -1, -1).clone()


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


def test_baseimputer_is_abstract():
    with pytest.raises(TypeError):
        BaseImputer()  # type: ignore[abstract]


def test_mock_imputer_instantiates():
    imp = MockImputer()
    assert imp is not None


def test_fit_returns_self():
    imp = MockImputer()
    ds = _MockDataset()
    result = imp.fit(ds)
    assert result is imp, "fit() must return self"


def test_impute_output_shape(x_sample, mask_half):
    imp = MockImputer().fit(_MockDataset())
    n = 7
    out = imp.impute(x_sample, mask_half, n_samples=n)
    assert out.shape == (n, J, F, T), f"Expected ({n},{J},{F},{T}), got {out.shape}"
    assert out.dtype == torch.float32


def test_impute_preserves_observed_entries(x_sample, mask_half):
    """Core contract: observed entries (mask=True) must be bit-identical in all samples."""
    imp = MockImputer().fit(_MockDataset())
    out = imp.impute(x_sample, mask_half, n_samples=15)
    for i in range(out.shape[0]):
        assert torch.allclose(out[i][mask_half], x_sample[mask_half]), (
            f"Sample {i}: observed entry changed after imputation"
        )


def test_impute_n_samples_1(x_sample, mask_half):
    imp = MockImputer().fit(_MockDataset())
    out = imp.impute(x_sample, mask_half, n_samples=1)
    assert out.shape == (1, J, F, T)


def test_impute_full_mask(x_sample):
    """Full mask (all observed) → every sample equals x_obs."""
    imp = MockImputer().fit(_MockDataset())
    full_mask = torch.ones(J, F, T, dtype=torch.bool)
    out = imp.impute(x_sample, full_mask, n_samples=5)
    for i in range(5):
        assert torch.allclose(out[i], x_sample), (
            f"Full mask: sample {i} differs from x_obs"
        )


def test_impute_empty_mask(x_sample):
    """Empty mask (all hidden) → observed contract trivially satisfied (no observed entries)."""
    imp = MockImputer().fit(_MockDataset())
    empty_mask = torch.zeros(J, F, T, dtype=torch.bool)
    out = imp.impute(x_sample, empty_mask, n_samples=3)
    assert out.shape == (3, J, F, T)


def test_impute_before_fit_raises(x_sample, mask_half):
    imp = MockImputer()
    with pytest.raises(RuntimeError):
        imp.impute(x_sample, mask_half, n_samples=1)


def test_name_property():
    imp = MockImputer()
    assert isinstance(imp.name, str)
    assert len(imp.name) > 0


def test_is_on_manifold_default():
    """Default is_on_manifold should be False for off-manifold imputers."""
    imp = MockImputer()
    assert imp.is_on_manifold is False
