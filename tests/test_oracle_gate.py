"""Oracle validation gate: the six properties G1-G6 that Proposition 1 relies on.

These tests pin the released oracles to the contract Proposition 1 needs:

  G1  Unconditional sampler covariance matches Sigma_J (x) I_F (x) Sigma_T.
  G2  Conditional sampler mean/cov match the closed-form Gaussian
      conditional; observed entries preserved bit-for-bit.
  G3  Shapley efficiency: sum(phi*) == v(N) - v(empty) (WLS constraint).
  G4  Dummy player: under INDEPENDENT windows (alpha=0), a function that
      ignores windows 1..K-1 attributes ~0 to them.  (Under AR(1) the
      conditional game legitimately credits correlated dummies, so the axiom
      only holds with independent players.)
  G5  Known answer: for a LINEAR game the conditional value function is
      closed-form; the exact solution agrees with the MC oracle.
  G6  Copula round-trip: x -> z -> x identity (float32 tolerance); Burr
      marginal variance ~ 1; observed entries preserved.

Configuration matches the gauss_k4 / burr_m5 benchmark datasets
(J=5, F=3, T=16/20, equicorr rho=0.5, AR(1) alpha=0.8, Burr XII c=k=2).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from motionbench.data.synthetic.burr_motion import BurrXII
from motionbench.oracles.copula_oracle import CopulaOracle
from motionbench.oracles.gaussian_oracle import GaussianOracle
from motionbench.players.temporal_windows import TemporalWindows
from motionbench.utils.coalitions import ar1_cov, equicorr

J, F, T, K = 5, 3, 16, 4
RHO, ALPHA = 0.5, 0.8
SIGMA_J = equicorr(J, RHO)
SIGMA_T = ar1_cov(T, ALPHA)
WS = T // K


@pytest.fixture(scope="module")
def oracle():
    return GaussianOracle(SIGMA_J, SIGMA_T)


@pytest.fixture(scope="module")
def players():
    return TemporalWindows(K=K, T=T, J=J, F=F)


def _f_window0(batch: torch.Tensor) -> torch.Tensor:
    """Scalar game depending only on window 0's grand mean."""
    w0 = batch[:, :, :, :WS].double().mean(dim=(1, 2, 3))
    return (1.0 / (1.0 + torch.exp(-3.0 * w0))).float()


def test_g1_unconditional_covariance(oracle):
    n = 60_000
    xs = oracle._sample_unconditional(n, J, F, T, np.random.default_rng(0)).astype(np.float64)
    err_t = np.abs(np.cov(xs[:, 0, 0, :].T) - SIGMA_T).max()
    err_j = np.abs(np.cov(xs[:, :, 0, 0].T) - SIGMA_J).max()
    cross = abs(np.corrcoef(xs[:, 0, 0, 0], xs[:, 0, 1, 0])[0, 1])
    assert err_t < 0.05, f"Sigma_T mismatch {err_t:.4f}"
    assert err_j < 0.05, f"Sigma_J mismatch {err_j:.4f}"
    assert cross < 0.05, f"cross-channel correlation {cross:.4f} (I_F violated)"


def test_g2_conditional_moments(oracle, players):
    rng = np.random.default_rng(1)
    x0 = oracle._sample_unconditional(1, J, F, T, rng)[0].astype(np.float64)
    z = torch.tensor([1, 0, 1, 0], dtype=torch.int32)
    mask = players.coalition_mask(z)
    n = 40_000
    cs = (
        oracle.conditional_sample(torch.tensor(x0, dtype=torch.float32), mask, n, seed=2)
        .double()
        .numpy()
    )

    mask_np = mask.numpy()
    t_obs = np.flatnonzero(mask_np[0, 0, :])
    t_hid = np.flatnonzero(~mask_np[0, 0, :])
    W = SIGMA_T[np.ix_(t_hid, t_obs)] @ np.linalg.inv(SIGMA_T[np.ix_(t_obs, t_obs)])
    Sc = SIGMA_T[np.ix_(t_hid, t_hid)] - W @ SIGMA_T[np.ix_(t_obs, t_hid)]
    mu_true = np.einsum("ho,jfo->jfh", W, x0[:, :, t_obs])

    err_mu = np.abs(cs[:, :, :, t_hid].mean(axis=0) - mu_true).max()
    emp_cc = np.cov(cs[:, 0, 0][:, t_hid].T)
    err_cc = np.abs(emp_cc - Sc * SIGMA_J[0, 0]).max()
    obs_exact = float(np.abs(cs[:, mask_np] - x0[mask_np].astype(np.float32)[None]).max())
    assert err_mu < 0.05, f"conditional mean error {err_mu:.4f}"
    assert err_cc < 0.05, f"conditional covariance error {err_cc:.4f}"
    assert obs_exact == 0.0, "observed entries not preserved bit-for-bit"


def test_g3_efficiency(oracle, players):
    rng = np.random.default_rng(3)
    x0 = torch.tensor(oracle._sample_unconditional(1, J, F, T, rng)[0], dtype=torch.float32)
    phi = oracle.true_shapley(x0, _f_window0, players, n_mc=2000, seed=1)
    # Efficiency is enforced by the WLS boundary constraints: sum(phi) equals
    # v(N) - v(empty).  v(N) is deterministic; v(empty) is re-estimated here
    # with an independent MC stream, so the tolerance covers two independent
    # 2000-draw MC errors of the empty-coalition value.
    v_full = float(_f_window0(x0[None])[0])
    v_empty = float(
        _f_window0(
            torch.tensor(
                oracle._sample_unconditional(2000, J, F, T, np.random.default_rng(1)),
                dtype=torch.float32,
            )
        ).mean()
    )
    gap = abs(float(phi.sum()) - (v_full - v_empty))
    assert gap < 0.02, f"efficiency gap {gap:.4f}"


def test_g4_dummy_player(players):
    """Dummy axiom holds under independent windows (alpha=0)."""
    ind_oracle = GaussianOracle(equicorr(J, RHO), ar1_cov(T, 0.0))
    x = torch.tensor(
        ind_oracle._sample_unconditional(1, J, F, T, np.random.default_rng(4))[0],
        dtype=torch.float32,
    )
    phi = ind_oracle.true_shapley(x, _f_window0, players, n_mc=2000, seed=5)
    dummy_max = float(phi[1:].abs().max())
    assert dummy_max < 0.01, f"dummy attribution {dummy_max:.4f}"
    assert abs(float(phi[0])) > 5 * dummy_max, "window-0 signal not dominant"


def test_g5_linear_game_closed_form(oracle, players):
    """MC oracle matches the closed-form solution of a linear game."""
    from motionbench.utils.coalitions import enumerate_coalitions, solve_shapley_wls

    rng5 = np.random.default_rng(50)
    w_lin = rng5.standard_normal((J, F, T))
    probe = oracle._sample_unconditional(2000, J, F, T, np.random.default_rng(51))
    w_lin *= 0.1 / float(np.tensordot(probe.astype(np.float64), w_lin, axes=3).std())

    def f_linear(batch: torch.Tensor) -> torch.Tensor:
        arr = batch.double().numpy()
        return torch.tensor(np.tensordot(arr, w_lin, axes=3), dtype=torch.float32)

    def exact_linear_phi(xq: np.ndarray) -> np.ndarray:
        Z, wk = enumerate_coalitions(K)
        vals = np.zeros(len(Z))
        for i, zrow in enumerate(Z):
            mask = players.coalition_mask(torch.tensor(zrow, dtype=torch.int32)).numpy()
            if mask.all():
                vals[i] = float((w_lin * xq).sum())
                continue
            if not mask.any():
                vals[i] = 0.0  # E[x] = 0
                continue
            t_o = np.flatnonzero(mask[0, 0, :])
            t_h = np.flatnonzero(~mask[0, 0, :])
            Wc = SIGMA_T[np.ix_(t_h, t_o)] @ np.linalg.inv(SIGMA_T[np.ix_(t_o, t_o)])
            mu_h = np.einsum("ho,jfo->jfh", Wc, xq[:, :, t_o])
            vals[i] = float(
                (w_lin[:, :, t_o] * xq[:, :, t_o]).sum() + (w_lin[:, :, t_h] * mu_h).sum()
            )
        v_empty = vals[np.all(Z == 0, axis=1)][0]
        v_full = vals[np.all(Z == 1, axis=1)][0]
        return solve_shapley_wls(Z, vals, wk, float(v_empty), float(v_full))

    deltas = []
    for i in range(4):
        xi = oracle._sample_unconditional(1, J, F, T, np.random.default_rng(100 + i))[0].astype(
            np.float64
        )
        p_exact = exact_linear_phi(xi)
        p_mc = oracle.true_shapley(
            torch.tensor(xi, dtype=torch.float32),
            f_linear,
            players,
            n_mc=50,
            seed=200 + i,
        ).numpy()
        deltas.append(np.mean(np.abs(p_mc - p_exact)))
    mean_delta = float(np.mean(deltas))
    assert mean_delta < 0.01, f"MC-vs-closed-form delta {mean_delta:.5f}"


def test_g6_copula_roundtrip():
    T_b, K_b = 20, 5
    sigma_t_b = ar1_cov(T_b, ALPHA)
    oracle = CopulaOracle(SIGMA_J, sigma_t_b, marginal=BurrXII(2.0, 2.0))
    players = TemporalWindows(K=K_b, T=T_b, J=J, F=F)

    xb = (
        oracle.conditional_sample(
            torch.zeros(J, F, T_b), torch.zeros(J, F, T_b, dtype=torch.bool), 20_000, seed=2
        )
        .double()
        .numpy()
    )
    var_b = float(xb.var())
    assert abs(var_b - 1.0) < 0.15, f"Burr marginal variance {var_b:.3f}"

    z_rt = oracle._x_to_z(xb)
    x_rt = oracle._z_to_x(z_rt)
    rt_err = float(np.abs(x_rt - xb).max())
    assert rt_err < 1e-4, f"copula round-trip error {rt_err:.2e}"

    mask = players.coalition_mask(torch.tensor([1, 0, 1, 0, 1], dtype=torch.int32))
    cb = oracle.conditional_sample(
        torch.tensor(xb[0], dtype=torch.float32), mask, 2000, seed=3
    ).numpy()
    mask_np = mask.numpy()
    obs_err = float(np.abs(cb[:, mask_np] - xb[0][mask_np].astype(np.float32)[None]).max())
    assert obs_err == 0.0, "copula conditional does not preserve observed entries"
