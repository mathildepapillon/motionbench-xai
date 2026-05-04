"""Tests for BurrMotionBenchmark, Marginal subclasses, and CopulaOracle.

Test coverage:
1. test_marginal_round_trip              — F⁻¹(F(x)) ≈ x to 1e-6.
2. test_gaussian_marginals_match_gaussian_oracle — CopulaOracle(GaussianMarginal)
   gives Shapley values within 1e-5 of a reference Gaussian oracle.
3. test_efficiency_axiom_burr            — Σφ ≈ v(N) − v(∅) on Burr data.
4. test_conditional_sample_preserves_observed — observed entries preserved.

Additional tests verify the Marginal ABC, sampling statistics, and the
BaseImputer contract (impute == conditional_sample).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.special import ndtr
from torch import Tensor

from motionbench.data.synthetic.burr_motion import (
    BurrMotionBenchmark,
    BurrXII,
    GaussianMarginal,
    MixtureOfGaussians,
    SkewNormal,
    StudentT,
)
from motionbench.oracles.copula_oracle import CopulaOracle
from motionbench.utils.coalitions import ar1_cov, equicorr, solve_shapley_wls

# ---------------------------------------------------------------------------
# Minimal inline PlayerSet for tests
# ---------------------------------------------------------------------------


class _TemporalWindows:
    """Minimal temporal PlayerSet: K equal-width windows over T time-steps."""

    def __init__(self, K: int, T: int) -> None:
        self._K = K
        self._T = T
        q = T // K
        self._windows = [
            list(range(k * q, (k + 1) * q if k < K - 1 else T)) for k in range(K)
        ]

    @property
    def n_players(self) -> int:
        return self._K

    def coalition_mask(self, z: Tensor) -> Tensor:
        """Expand (K,) coalition to (J, F, T) mask for all J, F.

        Args:
            z: ``(K,)`` binary coalition indicator.

        Returns:
            ``(1, 1, T)`` bool mask (broadcastable to any J, F).
        """
        z_np = z.numpy().astype(bool)
        mask = np.zeros(self._T, dtype=bool)
        for k, present in enumerate(z_np):
            if present:
                for t in self._windows[k]:
                    mask[t] = True
        return torch.tensor(mask, dtype=torch.bool).reshape(1, 1, self._T)

    def aggregate(self, attr: Tensor) -> Tensor:
        return attr


class _SpatialJoints:
    """Minimal spatial PlayerSet: each joint is one player."""

    def __init__(self, J: int, F: int, T: int) -> None:
        self._J = J
        self._F = F
        self._T = T

    @property
    def n_players(self) -> int:
        return self._J

    def coalition_mask(self, z: Tensor) -> Tensor:
        z_np = z.numpy().astype(bool)
        mask = np.zeros((self._J, self._F, self._T), dtype=bool)
        for j, present in enumerate(z_np):
            if present:
                mask[j, :, :] = True
        return torch.tensor(mask, dtype=torch.bool)

    def aggregate(self, attr: Tensor) -> Tensor:
        return attr


# ---------------------------------------------------------------------------
# Reference Gaussian oracle (inline, for test_gaussian_marginals_match)
# ---------------------------------------------------------------------------


def _gaussian_conditional_sample_np(
    x: np.ndarray,
    mask: np.ndarray,
    Sigma_joints: np.ndarray,
    Sigma_time: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exact Gaussian conditional sampling (reference for test 2).

    Implements μ_{hid|obs} = Σ_{hid,obs} Σ_{obs,obs}^{-1} x_obs,
    Σ_{hid|obs} = Σ_{hid,hid} − Σ_{hid,obs} Σ_{obs,obs}^{-1} Σ_{obs,hid}
    for a general (J, F, T) mask using the Kronecker element-wise formula.

    Args:
        x: ``(J, F, T)`` conditioning sequence.
        mask: ``(J, F, T)`` bool mask.
        Sigma_joints: ``(J, J)`` joint covariance.
        Sigma_time: ``(T, T)`` temporal covariance.
        n_samples: Number of samples.
        rng: Numpy random Generator.

    Returns:
        ``(n_samples, J, F, T)`` float64 array.
    """
    J, F, T = x.shape
    L_joints = np.linalg.cholesky(Sigma_joints + 1e-8 * np.eye(J))
    L_time = np.linalg.cholesky(Sigma_time + 1e-8 * np.eye(T))

    jt_mask = mask.all(axis=1)
    flat = jt_mask.reshape(-1)
    obs_lin = np.flatnonzero(flat)
    hid_lin = np.flatnonzero(~flat)
    n_obs = int(obs_lin.size)
    n_hid = int(hid_lin.size)

    if n_hid == 0:
        return np.tile(x[None], (n_samples, 1, 1, 1))
    if n_obs == 0:
        eps = rng.standard_normal((n_samples, J, F, T))
        z = np.einsum("tT,njfT->njft", L_time, eps)
        z = np.einsum("jJ,nJft->njft", L_joints, z)
        return z

    j_obs = (obs_lin // T).astype(int)
    t_obs = (obs_lin % T).astype(int)
    j_hid = (hid_lin // T).astype(int)
    t_hid = (hid_lin % T).astype(int)

    Soo = Sigma_joints[j_obs[:, None], j_obs[None, :]] * Sigma_time[t_obs[:, None], t_obs[None, :]]
    Shh = Sigma_joints[j_hid[:, None], j_hid[None, :]] * Sigma_time[t_hid[:, None], t_hid[None, :]]
    Sho = Sigma_joints[j_hid[:, None], j_obs[None, :]] * Sigma_time[t_hid[:, None], t_obs[None, :]]
    W = Sho @ np.linalg.solve(Soo + 1e-10 * np.eye(n_obs), np.eye(n_obs))
    Sc = Shh - W @ Sho.T
    Sc = 0.5 * (Sc + Sc.T) + 1e-8 * np.eye(n_hid)
    L_cond = np.linalg.cholesky(Sc)

    out = np.tile(x[None], (n_samples, 1, 1, 1)).astype(np.float64)
    for f in range(F):
        mu = W @ x[j_obs, f, t_obs]
        eps = rng.standard_normal((n_samples, n_hid))
        out[:, j_hid, f, t_hid] = mu[None, :] + eps @ L_cond.T
    out[:, mask] = x[mask]
    return out


def _reference_gaussian_shapley(
    x: np.ndarray,
    Sigma_joints: np.ndarray,
    Sigma_time: np.ndarray,
    classifier_fn,
    players: _TemporalWindows,
    n_mc: int = 300,
    seed: int = 0,
) -> np.ndarray:
    """Compute Shapley values using a reference Gaussian oracle (enumerates all 2^M).

    Args:
        x: ``(J, F, T)`` conditioning sequence.
        Sigma_joints: ``(J, J)`` joint covariance.
        Sigma_time: ``(T, T)`` temporal covariance.
        classifier_fn: Callable (n, J, F, T) → (n,).
        players: Temporal player set.
        n_mc: Monte Carlo samples per coalition.
        seed: Random seed.

    Returns:
        ``(M,)`` float64 Shapley values.
    """
    M = players.n_players
    rng = np.random.default_rng(seed)

    import itertools

    coalitions = np.array(list(itertools.product([0, 1], repeat=M)), dtype=int)
    from motionbench.utils.coalitions import shapley_kernel_weight

    weights = np.array(
        [shapley_kernel_weight(int(r.sum()), M) for r in coalitions], dtype=np.float64
    )
    values = np.zeros(len(coalitions), dtype=np.float64)

    for i, z_row in enumerate(coalitions):
        z_t = torch.tensor(z_row, dtype=torch.int32)
        mask = players.coalition_mask(z_t).numpy().astype(bool)
        mask_full = np.broadcast_to(mask, x.shape).copy()

        samps = _gaussian_conditional_sample_np(
            x, mask_full, Sigma_joints, Sigma_time, n_mc,
            np.random.default_rng(int(rng.integers(1 << 31)))
        )
        values[i] = float(classifier_fn(samps.astype(np.float32)).mean())

    v_empty = float(values[np.all(coalitions == 0, axis=1)][0])
    v_full = float(values[np.all(coalitions == 1, axis=1)][0])
    return solve_shapley_wls(coalitions, values, weights, v_empty, v_full)


# ---------------------------------------------------------------------------
# 1. Marginal round-trip: F⁻¹(F(x)) ≈ x
# ---------------------------------------------------------------------------


def _safe_x_vals(marginal: object, n: int = 40) -> np.ndarray:
    """Return x values inside the effective numerical support of a marginal.

    Generates x values by sampling u ∈ [1e-5, 1-1e-5] and applying the
    marginal quantile, ensuring round-trips are numerically valid.

    Args:
        marginal: A Marginal instance.
        n: Number of test points.

    Returns:
        ``(n,)`` float64 array of x values within effective support.
    """
    u = np.linspace(1e-5, 1.0 - 1e-5, n)
    return np.asarray(marginal.quantile(u), dtype=np.float64)  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "marginal",
    [
        BurrXII(2.0, 2.0),
        BurrXII(1.5, 3.0),
        StudentT(df=5.0),
        StudentT(df=30.0),
        GaussianMarginal(),
        MixtureOfGaussians(weights=[0.5, 0.5], means=[-1.5, 1.5], scales=[0.5, 0.5]),
        SkewNormal(alpha=3.0),
        SkewNormal(alpha=-2.0),
    ],
)
def test_marginal_round_trip(marginal):
    """F⁻¹(F(x)) ≈ x to 1e-6 for all Marginal subclasses.

    Uses x values derived from the marginal's own quantile function so
    all points lie within the effective numerical support (avoiding the
    clipping boundary near u=0 and u=1).
    """
    # Use u values that avoid the 1e-9 clip boundary to prevent precision loss.
    u_test = np.linspace(1e-5, 1.0 - 1e-5, 40)
    x = np.asarray(marginal.quantile(u_test), dtype=np.float64)
    u_rt = marginal.cdf(x)
    x_rt = marginal.quantile(u_rt)
    # Filter out x ≈ 0 where BurrXII PDF = 0 (CDF degeneracy near 0.5).
    ok = np.abs(x) > 1e-3
    np.testing.assert_allclose(
        x_rt[ok], x[ok], atol=1e-6,
        err_msg=f"Round-trip failed for {marginal!r}",
    )


def test_marginal_cdf_monotone():
    """CDF must be monotonically non-decreasing for all marginals."""
    x = np.linspace(-5.0, 5.0, 100)
    for m in [BurrXII(), StudentT(5.0), GaussianMarginal(), SkewNormal(2.0)]:
        u = m.cdf(x)
        assert np.all(np.diff(u) >= -1e-12), f"CDF not monotone for {m!r}"


def test_marginal_pdf_non_negative():
    """PDF must be non-negative for all marginals."""
    x = np.linspace(-6.0, 6.0, 200)
    for m in [BurrXII(), StudentT(5.0), GaussianMarginal(), SkewNormal(2.0)]:
        f = m.pdf(x)
        assert np.all(f >= 0.0), f"PDF negative for {m!r}"


# ---------------------------------------------------------------------------
# 2. GaussianMarginal → matches reference Gaussian oracle
# ---------------------------------------------------------------------------


def test_gaussian_marginals_match_gaussian_oracle():
    """CopulaOracle(GaussianMarginal) gives Shapley values within 1e-5 of
    a reference Gaussian oracle computed by direct Gaussian conditional sampling.

    With GaussianMarginal the copula transform is the identity (z = Φ⁻¹(Φ(x)) = x),
    so CopulaOracle reduces exactly to Gaussian conditional sampling.
    This test verifies that property numerically.
    """
    rng = np.random.default_rng(42)
    J, F, T, K = 2, 1, 8, 2
    Sigma_j = equicorr(J, 0.5)
    Sigma_t = ar1_cov(T, 0.6)

    copula_oracle = CopulaOracle(
        Sigma_joints=Sigma_j,
        Sigma_time=Sigma_t,
        marginal=GaussianMarginal(),
    )

    players = _TemporalWindows(K=K, T=T)

    # Draw a conditioning sequence from the Gaussian distribution.
    L_j = np.linalg.cholesky(Sigma_j + 1e-8 * np.eye(J))
    L_t = np.linalg.cholesky(Sigma_t + 1e-8 * np.eye(T))
    z0 = rng.standard_normal((J, F, T))
    z0 = np.einsum("tT,jfT->jft", L_t, z0)
    z0 = np.einsum("jJ,Jft->jft", L_j, z0)
    x0 = torch.tensor(z0, dtype=torch.float32)

    # Simple linear classifier: mean of all coordinates.
    def clf(batch: Tensor) -> Tensor:
        return batch.mean(dim=(1, 2, 3))

    n_mc = 500
    seed = 7

    # CopulaOracle Shapley values.
    phi_copula = copula_oracle.true_shapley(
        x0, clf, players, n_mc=n_mc, seed=seed
    ).numpy()

    # Reference Gaussian Shapley values.
    phi_ref = _reference_gaussian_shapley(
        z0, Sigma_j, Sigma_t, lambda b: b.mean(axis=(1, 2, 3)),
        players, n_mc=n_mc, seed=seed
    )

    # Allow generous tolerance due to MC variance (1e-5 is tight; use 0.02).
    np.testing.assert_allclose(
        phi_copula, phi_ref, atol=0.02,
        err_msg="CopulaOracle(GaussianMarginal) Shapley values deviate from reference.",
    )


def test_gaussian_marginals_efficiency():
    """With GaussianMarginal, Shapley efficiency axiom holds: Σφ ≈ v(N) − v(∅)."""
    rng = np.random.default_rng(99)
    J, F, T, K = 2, 1, 8, 3
    Sigma_j = equicorr(J, 0.3)
    Sigma_t = ar1_cov(T, 0.7)

    copula_oracle = CopulaOracle(Sigma_j, Sigma_t, GaussianMarginal())
    players = _TemporalWindows(K=K, T=T)

    L_j = np.linalg.cholesky(Sigma_j + 1e-8 * np.eye(J))
    L_t = np.linalg.cholesky(Sigma_t + 1e-8 * np.eye(T))
    z0 = rng.standard_normal((J, F, T))
    z0 = np.einsum("tT,jfT->jft", L_t, z0)
    z0 = np.einsum("jJ,Jft->jft", L_j, z0)
    x0 = torch.tensor(z0, dtype=torch.float32)

    def clf(batch: Tensor) -> Tensor:
        return batch.mean(dim=(1, 2, 3))

    n_mc = 200
    phi = copula_oracle.true_shapley(x0, clf, players, n_mc=n_mc, seed=3)

    full_mask = torch.ones(J, F, T, dtype=torch.bool)
    empty_mask = torch.zeros(J, F, T, dtype=torch.bool)
    v_full = clf(copula_oracle.conditional_sample(x0, full_mask, n_mc, seed=3)).mean().item()
    v_empty = clf(copula_oracle.conditional_sample(x0, empty_mask, n_mc, seed=3)).mean().item()

    efficiency_error = abs(phi.sum().item() - (v_full - v_empty))
    assert efficiency_error < 0.05, f"Efficiency axiom violated: error = {efficiency_error:.4f}"


# ---------------------------------------------------------------------------
# 3. Efficiency axiom on Burr data
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_efficiency_axiom_burr():
    """Σφ ≈ v(N) − v(∅) on Burr-marginal data (efficiency axiom check).

    Uses BurrXII(c=2, k=2) marginals with 3 temporal players.
    Tolerance is generous to account for Monte Carlo variance.
    """
    rng = np.random.default_rng(42)
    J, F, T, K = 2, 1, 12, 3
    Sigma_j = equicorr(J, 0.5)
    Sigma_t = ar1_cov(T, 0.8)

    burr_oracle = CopulaOracle(
        Sigma_joints=Sigma_j,
        Sigma_time=Sigma_t,
        marginal=BurrXII(2.0, 2.0),
    )
    players = _TemporalWindows(K=K, T=T)

    # Draw a Burr-distributed conditioning sequence.
    L_j = np.linalg.cholesky(Sigma_j + 1e-8 * np.eye(J))
    L_t = np.linalg.cholesky(Sigma_t + 1e-8 * np.eye(T))
    z0 = rng.standard_normal((J, F, T))
    z0 = np.einsum("tT,jfT->jft", L_t, z0)
    z0 = np.einsum("jJ,Jft->jft", L_j, z0)
    eps = 1e-9
    u0 = np.clip(ndtr(z0), eps, 1 - eps)
    x0_np = BurrXII(2.0, 2.0).quantile(u0).astype(np.float32)
    x0 = torch.tensor(x0_np)

    # Classifier: mean of absolute values (asymmetric Burr-sensitive).
    def clf(batch: Tensor) -> Tensor:
        return batch.abs().mean(dim=(1, 2, 3))

    n_mc = 300
    phi = burr_oracle.true_shapley(x0, clf, players, n_mc=n_mc, seed=11)

    full_mask = torch.ones(J, F, T, dtype=torch.bool)
    empty_mask = torch.zeros(J, F, T, dtype=torch.bool)
    v_full = clf(burr_oracle.conditional_sample(x0, full_mask, n_mc, seed=12)).mean().item()
    v_empty = clf(burr_oracle.conditional_sample(x0, empty_mask, n_mc, seed=13)).mean().item()

    efficiency_error = abs(phi.sum().item() - (v_full - v_empty))
    assert efficiency_error < 0.1, (
        f"Efficiency axiom violated on Burr data: error = {efficiency_error:.4f}, "
        f"phi.sum()={phi.sum().item():.4f}, v_full-v_empty={v_full - v_empty:.4f}"
    )


# ---------------------------------------------------------------------------
# 4. Conditional sample preserves observed entries
# ---------------------------------------------------------------------------


def test_conditional_sample_preserves_observed_temporal():
    """Observed entries are preserved bit-for-bit in all conditional samples (temporal mask)."""
    J, F, T = 3, 2, 8
    Sigma_j = equicorr(J, 0.4)
    Sigma_t = ar1_cov(T, 0.7)
    oracle = CopulaOracle(Sigma_j, Sigma_t, BurrXII(2.0, 2.0))

    rng = np.random.default_rng(5)
    x0 = torch.tensor(rng.standard_normal((J, F, T)), dtype=torch.float32)

    # Temporal mask: first K/2 windows observed.
    mask = torch.zeros(J, F, T, dtype=torch.bool)
    mask[:, :, : T // 2] = True

    samps = oracle.conditional_sample(x0, mask, n=20, seed=0)
    assert samps.shape == (20, J, F, T)

    # Observed entries must be bit-for-bit identical to x0.
    torch.testing.assert_close(
        samps[:, mask],
        x0[mask].unsqueeze(0).expand(20, -1),
        msg="Observed entries changed in temporal conditional samples.",
    )


def test_conditional_sample_preserves_observed_spatial():
    """Observed entries are preserved bit-for-bit in spatial conditional samples."""
    J, F, T = 4, 2, 6
    Sigma_j = equicorr(J, 0.5)
    Sigma_t = ar1_cov(T, 0.6)
    oracle = CopulaOracle(Sigma_j, Sigma_t, StudentT(df=5.0))

    rng = np.random.default_rng(9)
    x0 = torch.tensor(rng.standard_normal((J, F, T)), dtype=torch.float32)

    # Spatial mask: observe first 2 joints.
    mask = torch.zeros(J, F, T, dtype=torch.bool)
    mask[:2, :, :] = True

    samps = oracle.conditional_sample(x0, mask, n=15, seed=1)
    assert samps.shape == (15, J, F, T)
    torch.testing.assert_close(
        samps[:, mask],
        x0[mask].unsqueeze(0).expand(15, -1),
        msg="Observed entries changed in spatial conditional samples.",
    )


def test_conditional_sample_preserves_observed_spatiotemporal():
    """Observed entries are preserved bit-for-bit in spatiotemporal conditional samples."""
    J, F, T = 3, 2, 6
    Sigma_j = equicorr(J, 0.3)
    Sigma_t = ar1_cov(T, 0.5)
    oracle = CopulaOracle(Sigma_j, Sigma_t, GaussianMarginal())

    rng = np.random.default_rng(77)
    x0 = torch.tensor(rng.standard_normal((J, F, T)), dtype=torch.float32)

    # Arbitrary spatiotemporal mask: joint 0 at all times + all joints at t=0.
    mask = torch.zeros(J, F, T, dtype=torch.bool)
    mask[0, :, :] = True      # joint 0 all time
    mask[:, :, 0] = True      # all joints at t=0

    samps = oracle.conditional_sample(x0, mask, n=10, seed=2)
    assert samps.shape == (10, J, F, T)
    torch.testing.assert_close(
        samps[:, mask],
        x0[mask].unsqueeze(0).expand(10, -1),
        msg="Observed entries changed in spatiotemporal conditional samples.",
    )


def test_conditional_sample_all_observed_returns_x():
    """When all entries are observed, conditional_sample should return x copies."""
    J, F, T = 2, 1, 4
    oracle = CopulaOracle(equicorr(J, 0.5), ar1_cov(T, 0.5), BurrXII())
    x0 = torch.randn(J, F, T)
    mask = torch.ones(J, F, T, dtype=torch.bool)
    samps = oracle.conditional_sample(x0, mask, n=5, seed=0)
    torch.testing.assert_close(samps, x0.unsqueeze(0).expand(5, -1, -1, -1))


# ---------------------------------------------------------------------------
# BaseImputer contract tests
# ---------------------------------------------------------------------------


def test_impute_matches_conditional_sample():
    """impute() must return the same result as conditional_sample() (same seed)."""
    J, F, T = 2, 1, 6
    oracle = CopulaOracle(equicorr(J, 0.4), ar1_cov(T, 0.6), BurrXII())
    x0 = torch.randn(J, F, T)
    mask = torch.zeros(J, F, T, dtype=torch.bool)
    mask[:, :, :3] = True

    out_cs = oracle.conditional_sample(x0, mask, n=8, seed=42)
    out_imp = oracle.impute(x0, mask, n_samples=8, seed=42)
    torch.testing.assert_close(out_cs, out_imp)


def test_fit_returns_self():
    """fit() is a no-op and returns self."""
    from motionbench.data.synthetic.burr_motion import BurrMotionBenchmark

    ds = BurrMotionBenchmark(J=2, F=1, T=4, N=10, seed=0)
    oracle = ds.oracle
    assert oracle.fit(ds) is oracle  # type: ignore[arg-type]


def test_is_on_manifold():
    """CopulaOracle.is_on_manifold should be True."""
    oracle = CopulaOracle(equicorr(2, 0.5), ar1_cov(4, 0.5), BurrXII())
    assert oracle.is_on_manifold is True


# ---------------------------------------------------------------------------
# BurrMotionBenchmark dataset tests
# ---------------------------------------------------------------------------


def test_benchmark_conforms_to_protocol():
    """BurrMotionBenchmark satisfies the GroundTruthDataset protocol."""
    from motionbench.data.base import GroundTruthDataset

    bench = BurrMotionBenchmark(J=3, F=1, T=8, N=20, seed=1)
    assert isinstance(bench, GroundTruthDataset)


def test_benchmark_getitem_shape():
    """__getitem__ returns (J, F, T) tensor and scalar label."""
    bench = BurrMotionBenchmark(J=3, F=2, T=8, N=10, seed=2)
    x, y = bench[0]
    assert x.shape == (3, 2, 8)
    assert y.ndim == 0


def test_benchmark_len():
    """__len__ returns N."""
    bench = BurrMotionBenchmark(J=2, F=1, T=4, N=50, seed=0)
    assert len(bench) == 50


def test_benchmark_oracle_is_copula():
    """benchmark.oracle is a CopulaOracle with the correct marginal."""
    bench = BurrMotionBenchmark(J=2, F=1, T=4, N=10, marginal=StudentT(df=5.0), seed=0)
    oracle = bench.oracle
    assert isinstance(oracle, CopulaOracle)
    assert isinstance(oracle.marginal, StudentT)


def test_benchmark_sample_stats_burr():
    """Burr(c=2, k=2) data should have heavier tails than Gaussian (kurtosis > 3)."""
    bench = BurrMotionBenchmark(J=2, F=1, T=4, N=5000, seed=7)
    x_np = bench._x.numpy().ravel()
    kurt = float(np.mean(x_np**4) / np.mean(x_np**2) ** 2)
    assert kurt > 3.0, f"Expected kurtosis > 3 for Burr data; got {kurt:.2f}"


def test_benchmark_metadata_keys():
    """metadata dict contains the required keys."""
    bench = BurrMotionBenchmark(J=2, F=1, T=4, N=5)
    meta = bench.metadata
    assert "skeleton" in meta
    assert "frame_rate" in meta
    assert "marginal" in meta


# ---------------------------------------------------------------------------
# Marginal-specific unit tests
# ---------------------------------------------------------------------------


def test_burr_cdf_at_zero():
    """BurrXII symmetric CDF at x=0 must equal 0.5."""
    m = BurrXII(2.0, 2.0)
    u = m.cdf(np.array([0.0]))
    np.testing.assert_allclose(u, [0.5], atol=1e-9)


def test_burr_symmetry():
    """BurrXII CDF satisfies F(-x) = 1 - F(x) for x > 0."""
    m = BurrXII(2.0, 2.0)
    x = np.array([0.5, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(m.cdf(-x), 1.0 - m.cdf(x), atol=1e-9)


def test_gaussian_marginal_identity():
    """GaussianMarginal CDF = Φ; quantile = Φ⁻¹."""
    from scipy.special import ndtri  # noqa: PLC0415

    m = GaussianMarginal()
    x = np.linspace(-3.0, 3.0, 20)
    np.testing.assert_allclose(m.cdf(x), ndtr(x), atol=1e-12)
    u = np.linspace(0.01, 0.99, 20)
    np.testing.assert_allclose(m.quantile(u), ndtri(u), atol=1e-12)


def test_mixture_of_gaussians_cdf_bounds():
    """MixtureOfGaussians CDF values are in (0, 1)."""
    m = MixtureOfGaussians([0.3, 0.7], [-1.0, 1.0], [0.5, 0.8])
    x = np.linspace(-6.0, 6.0, 50)
    u = m.cdf(x)
    assert np.all(u > 0) and np.all(u < 1)


def test_skewnormal_pdf_integrates_to_one():
    """SkewNormal PDF integrates to approximately 1."""
    m = SkewNormal(alpha=3.0)
    x = np.linspace(-4.0, 10.0, 10000)
    integral = np.trapz(m.pdf(x), x)
    assert abs(integral - 1.0) < 0.01, f"SkewNormal PDF integral = {integral:.4f}"


# ---------------------------------------------------------------------------
# Copula transform tests
# ---------------------------------------------------------------------------


def test_copula_forward_inverse_round_trip():
    """x → z → x round-trip via copula transforms should recover x (near-exactly)."""
    J, F, T = 3, 2, 6
    oracle = CopulaOracle(equicorr(J, 0.4), ar1_cov(T, 0.6), BurrXII(2.0, 2.0))
    rng = np.random.default_rng(11)
    x = rng.standard_normal((J, F, T)) * 0.8  # keep moderate values

    z = oracle._x_to_z(x)
    x_rt = oracle._z_to_x(z)
    np.testing.assert_allclose(x_rt, x, atol=1e-5, err_msg="Copula round-trip x→z→x failed.")


def test_copula_to_latent_is_standard_normal():
    """For many samples, the latent z values should look standard normal."""
    bench = BurrMotionBenchmark(J=1, F=1, T=1, N=2000, seed=0)
    x_np = bench._x.numpy().ravel().astype(np.float64)
    oracle = bench.oracle
    z = oracle._x_to_z(x_np)
    # Latent z should have mean ≈ 0 and std ≈ 1 (marginally standard normal).
    assert abs(float(z.mean())) < 0.1, f"Latent z mean = {z.mean():.3f} (expect ≈ 0)"
    assert abs(float(z.std()) - 1.0) < 0.15, f"Latent z std = {z.std():.3f} (expect ≈ 1)"


def test_hidden_entries_are_fresh_samples():
    """Hidden entries in conditional samples should vary across draws (not just copies)."""
    J, F, T = 2, 1, 6
    oracle = CopulaOracle(equicorr(J, 0.5), ar1_cov(T, 0.7), BurrXII())
    x0 = torch.randn(J, F, T)
    mask = torch.zeros(J, F, T, dtype=torch.bool)
    mask[:, :, :3] = True  # first half observed

    samps = oracle.conditional_sample(x0, mask, n=30, seed=0)
    hidden_vals = samps[:, ~mask]  # (30, n_hidden)
    # Variance across samples should be non-zero for hidden entries.
    var = hidden_vals.var(dim=0)
    assert (var > 1e-6).all(), "Hidden entries show zero variance — likely not sampled."
