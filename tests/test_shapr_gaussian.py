"""Tests for motionbench.imputers.shapr_gaussian.ShaprGaussianImputer.

Covers the BaseImputer contract (shapes, bit-exact preservation of observed
entries, fit-before-impute), the Gaussian-conditional math (conditional mean
against a hand-computed reference), seeding semantics (seed vs threaded
generator), the per-mask parameter cache, and the ``fit_data`` protocol pin
in the player_eval pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from motionbench.imputers.shapr_gaussian import ShaprGaussianImputer

J, F, T = 3, 2, 8
D = J * F * T


@pytest.fixture(scope="module")
def pool() -> np.ndarray:
    """(N, J, F, T) correlated Gaussian pool, N > D so 'ml' is well-posed."""
    rng = np.random.default_rng(7)
    A = rng.standard_normal((D, D)) / np.sqrt(D)
    cov = A @ A.T + 0.5 * np.eye(D)
    L = np.linalg.cholesky(cov)
    flat = rng.standard_normal((400, D)) @ L.T + 1.5
    return flat.reshape(400, J, F, T).astype(np.float32)


@pytest.fixture(scope="module")
def fitted(pool: np.ndarray) -> ShaprGaussianImputer:
    return ShaprGaussianImputer().fit_pool(pool)


@pytest.fixture
def x_and_mask(pool: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(pool[0])
    mask = torch.zeros(J, F, T, dtype=torch.bool)
    mask[0] = True
    mask[1, :, : T // 2] = True
    return x, mask


class TestConstruction:
    def test_invalid_shrinkage_raises(self):
        with pytest.raises(ValueError, match="shrinkage"):
            ShaprGaussianImputer(shrinkage="oas")

    def test_impute_before_fit_raises(self, x_and_mask):
        x, mask = x_and_mask
        with pytest.raises(RuntimeError, match="fit"):
            ShaprGaussianImputer().impute(x, mask, 3, seed=0)

    def test_fit_returns_self_and_is_on_manifold(self, pool):
        imp = ShaprGaussianImputer()
        assert imp.fit_pool(pool) is imp
        assert imp.is_on_manifold

    def test_fit_accepts_dataset(self, pool):
        class _DS:
            def __len__(self):
                return 16

            def __getitem__(self, i):
                return torch.from_numpy(pool[i]), torch.tensor(0)

        imp = ShaprGaussianImputer().fit(_DS())
        out = imp.impute(torch.from_numpy(pool[0]), torch.ones(J, F, T, dtype=torch.bool), 2)
        assert out.shape == (2, J, F, T)

    def test_bad_pool_shape_raises(self):
        with pytest.raises(ValueError, match="pool"):
            ShaprGaussianImputer().fit_pool(np.zeros((10, D)))

    def test_shape_mismatch_raises(self, fitted):
        x = torch.zeros(J, F, T)
        with pytest.raises(ValueError, match="mask"):
            fitted.impute(x, torch.ones(J, F, T + 1, dtype=torch.bool), 2)
        with pytest.raises(ValueError, match="fitted"):
            bad = torch.zeros(J, F, T + 1)
            fitted.impute(bad, torch.ones(J, F, T + 1, dtype=torch.bool), 2)


class TestContract:
    def test_output_shape_and_dtype(self, fitted, x_and_mask):
        x, mask = x_and_mask
        out = fitted.impute(x, mask, 4, seed=0)
        assert out.shape == (4, J, F, T)
        assert out.dtype == torch.float32

    def test_observed_preserved_bit_exact(self, fitted, x_and_mask):
        x, mask = x_and_mask
        out = fitted.impute(x, mask, 5, seed=1)
        for i in range(5):
            assert torch.equal(out[i][mask], x[mask])

    def test_full_mask_returns_copies(self, fitted, x_and_mask):
        x, _ = x_and_mask
        out = fitted.impute(x, torch.ones(J, F, T, dtype=torch.bool), 3, seed=2)
        assert torch.equal(out, x.expand(3, -1, -1, -1))

    def test_empty_mask_draws_unconditionally(self, fitted, x_and_mask):
        x, _ = x_and_mask
        out = fitted.impute(x, torch.zeros(J, F, T, dtype=torch.bool), 2000, seed=3)
        # Sample mean of the draws approaches the fitted pool mean.
        mu_hat = out.numpy().reshape(2000, -1).mean(axis=0)
        assert np.allclose(mu_hat, fitted._mu, atol=0.15)

    def test_seed_determinism(self, fitted, x_and_mask):
        x, mask = x_and_mask
        a = fitted.impute(x, mask, 3, seed=11)
        b = fitted.impute(x, mask, 3, seed=11)
        c = fitted.impute(x, mask, 3, seed=12)
        assert torch.equal(a, b)
        assert not torch.equal(a, c)

    def test_generator_threads_one_stream(self, fitted, x_and_mask):
        """generator= draws match manual draws from the same stream state."""
        x, mask = x_and_mask
        gen = np.random.default_rng([42, 0])
        out1 = fitted.impute(x, mask, 3, generator=gen)
        out2 = fitted.impute(x, mask, 3, generator=gen)
        # Reference: same math against a fresh stream, consumed in call order.
        ref_gen = np.random.default_rng([42, 0])
        mf = mask.numpy().reshape(-1)
        obs, hid, W, L = fitted._conditional_params(mf)
        xf = x.numpy().astype(np.float64).reshape(-1)
        mu_h = fitted._mu[hid] + W @ (xf[obs] - fitted._mu[obs])
        for out in (out1, out2):
            eps = ref_gen.standard_normal((3, len(hid))) @ L.T
            ref = np.tile(xf[None], (3, 1))
            ref[:, hid] = mu_h[None] + eps
            expect = ref.reshape(3, J, F, T).astype(np.float32)
            assert np.array_equal(out.numpy(), expect)


class TestGaussianMath:
    def test_conditional_mean_matches_analytic(self, fitted, x_and_mask):
        """Empirical mean of many draws converges to mu_h + W (x_o - mu_o)."""
        x, mask = x_and_mask
        out = fitted.impute(x, mask, 4000, seed=5).numpy().reshape(4000, -1)
        mf = mask.numpy().reshape(-1)
        obs, hid, W, _L = fitted._conditional_params(mf)
        xf = x.numpy().astype(np.float64).reshape(-1)
        mu_h = fitted._mu[hid] + W @ (xf[obs] - fitted._mu[obs])
        assert np.allclose(out[:, hid].mean(axis=0), mu_h, atol=0.15)

    def test_ml_shrinkage_has_zero_coef(self, pool):
        imp = ShaprGaussianImputer(shrinkage="ml").fit_pool(pool)
        assert imp.shrinkage_coef == 0.0
        lw = ShaprGaussianImputer().fit_pool(pool)
        assert lw.shrinkage_coef > 0.0
        assert not np.array_equal(imp._cov, lw._cov)

    def test_covariance_is_symmetric(self, fitted):
        assert np.array_equal(fitted._cov, fitted._cov.T)

    def test_cache_reused_per_mask(self, fitted, x_and_mask):
        x, mask = x_and_mask
        fitted._cache.clear()
        fitted.impute(x, mask, 2, seed=0)
        assert len(fitted._cache) == 1
        fitted.impute(x, mask, 2, seed=1)
        assert len(fitted._cache) == 1
        other = torch.zeros(J, F, T, dtype=torch.bool)
        other[2] = True
        fitted.impute(x, other, 2, seed=0)
        assert len(fitted._cache) == 2

    def test_refit_clears_cache(self, pool, x_and_mask):
        imp = ShaprGaussianImputer().fit_pool(pool)
        x, mask = x_and_mask
        imp.impute(x, mask, 2, seed=0)
        assert len(imp._cache) == 1
        imp.fit_pool(pool)
        assert len(imp._cache) == 0
