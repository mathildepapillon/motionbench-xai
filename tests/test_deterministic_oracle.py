"""Tests for motionbench.oracles.deterministic.

The deterministic conditional fill must agree with a large-n Monte-Carlo
estimate of E[x_hid | x_obs] from the sampling oracles, on both the plain
Gaussian and the Gaussian-copula (Burr XII) families, for temporal, spatial
and general (cell) masks.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from motionbench.attribution.sampled_coalitions import sampled_coalition_set
from motionbench.data.synthetic.burr_motion import BurrXII
from motionbench.oracles.copula_oracle import CopulaOracle
from motionbench.oracles.deterministic import (
    GL_NODES,
    DeterministicConditionalOracle,
    cusp_split_nodes,
    marginal_fill,
)
from motionbench.oracles.gaussian_oracle import GaussianOracle
from motionbench.players.joint_window_cells import JointWindowCells
from motionbench.players.spatial_joints import SpatialJoints
from motionbench.players.temporal_windows import TemporalWindows
from motionbench.utils.coalitions import ar1_cov, equicorr

J, F, T, K = 3, 2, 8, 4
SIGMA_J = equicorr(J, 0.5)
SIGMA_T = ar1_cov(T, 0.8)


def _mc_cond_mean(oracle, x, mask, n=60_000, seed=0):
    samples = oracle.conditional_sample(
        torch.tensor(x, dtype=torch.float32), torch.tensor(mask), n, seed=seed
    )
    return samples.double().mean(dim=0).numpy()


def _draw_x(seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((J, F, T))
    Lj = np.linalg.cholesky(SIGMA_J + 1e-8 * np.eye(J))
    Lt = np.linalg.cholesky(SIGMA_T + 1e-8 * np.eye(T))
    return np.einsum("jk,kft->jft", Lj, np.einsum("ts,jfs->jft", Lt, z))


@pytest.mark.parametrize(
    "players,z",
    [
        (TemporalWindows(K=K, T=T, J=J, F=F), [1, 0, 1, 0]),
        (SpatialJoints(J=J, F=F, T=T), [1, 0, 1]),
        (JointWindowCells(J=J, K=K, F=F, T=T), [1, 0] * (J * K // 2)),
    ],
    ids=["temporal", "spatial", "cells"],
)
def test_gaussian_fill_matches_mc(players, z):
    """Deterministic Gaussian fill == MC mean of exact conditional draws."""
    oracle = GaussianOracle(SIGMA_J, SIGMA_T)
    Z = np.asarray([z], dtype=np.int8)
    det = DeterministicConditionalOracle(SIGMA_J, SIGMA_T, players, Z)
    x = _draw_x(seed=3)
    fill = det.fill_all(x)[0].astype(np.float64)

    mask = players.coalition_mask(torch.tensor(z, dtype=torch.bool)).numpy()
    mc = _mc_cond_mean(oracle, x, mask)
    hid = ~mask
    # MC error ~ sd/sqrt(n) ~ 0.004; use a safe 4-sigma band.
    assert np.abs(fill[hid] - mc[hid]).max() < 0.02
    # Observed entries preserved exactly.
    np.testing.assert_allclose(fill[mask], x[mask], atol=1e-6)


def test_copula_fill_matches_mc_temporal():
    """Cusp-split quadrature fill == MC conditional mean under the copula."""
    marginal = BurrXII(2.0, 2.0)
    oracle = CopulaOracle(SIGMA_J, SIGMA_T, marginal=marginal)
    players = TemporalWindows(K=K, T=T, J=J, F=F)
    z = [1, 0, 0, 1]
    Z = np.asarray([z], dtype=np.int8)
    det = DeterministicConditionalOracle(SIGMA_J, SIGMA_T, players, Z, marginal=marginal)
    # A sequence with valid Burr marginals: push a Gaussian draw through the copula.
    x = marginal.quantile(np.random.default_rng(11).uniform(0.05, 0.95, size=(J, F, T)))
    fill = det.fill_all(x)[0].astype(np.float64)
    mask = players.coalition_mask(torch.tensor(z, dtype=torch.bool)).numpy()
    mc = _mc_cond_mean(oracle, x, mask, n=120_000, seed=5)
    hid = ~mask
    assert np.abs(fill[hid] - mc[hid]).max() < 0.03
    np.testing.assert_allclose(fill[mask], x[mask], atol=1e-6)


def test_quadrature_node_doubling_converged():
    """Doubling the Gauss-Legendre nodes must not change the copula mean.

    (mu, sd) pairs are sampled from the domain that actually occurs for a
    unit-variance latent field: the conditional sd is at most 1 and the
    conditional mean has variance 1 - sd^2 (a nearly-unconstrained coordinate
    has sd near 1 and mu near 0).  Outside this domain — |mu| large together
    with sd near 1 — the marginal's probability-clip kink degrades the rule;
    that regime cannot arise from the benchmark's correlation-matrix models.
    """
    marginal = BurrXII(2.0, 2.0)
    rng = np.random.default_rng(0)
    sd = rng.uniform(0.05, 1.0, size=(5, 7))
    mu = rng.standard_normal((5, 7)) * np.sqrt(1.0 - sd**2)

    def copula_mean(n_nodes):
        from scipy.special import ndtr

        t, w = cusp_split_nodes(mu, sd, n=n_nodes)
        zz = mu[..., None] + sd[..., None] * t
        return np.sum(marginal.quantile(ndtr(zz)) * w, axis=-1)

    m1 = copula_mean(GL_NODES)
    m2 = copula_mean(2 * GL_NODES)
    assert np.abs(m1 - m2).max() < 1e-6


def test_boundary_coalitions():
    """Empty coalition fills with E[x] = 0; full coalition returns x."""
    players = TemporalWindows(K=K, T=T, J=J, F=F)
    Z = np.asarray([[0] * K, [1] * K], dtype=np.int8)
    det = DeterministicConditionalOracle(SIGMA_J, SIGMA_T, players, Z)
    x = _draw_x(seed=9)
    fills = det.fill_all(x).astype(np.float64)
    np.testing.assert_allclose(fills[0], 0.0, atol=0)
    np.testing.assert_allclose(fills[1], x, atol=1e-6)


def test_from_oracle_constructor():
    players = SpatialJoints(J=J, F=F, T=T)
    Z, _w = sampled_coalition_set(J, budget=8)
    gauss = DeterministicConditionalOracle.from_oracle(GaussianOracle(SIGMA_J, SIGMA_T), players, Z)
    assert gauss.marginal is None
    cop = DeterministicConditionalOracle.from_oracle(CopulaOracle(SIGMA_J, SIGMA_T), players, Z)
    assert cop.marginal is not None
    with pytest.raises(TypeError):
        DeterministicConditionalOracle.from_oracle(object(), players, Z)


def test_marginal_fill():
    x = _draw_x(seed=1)
    mask = np.zeros((J, F, T), dtype=bool)
    mask[0] = True
    out = marginal_fill(x, mask)
    np.testing.assert_allclose(out[0], x[0], atol=1e-6)
    assert (out[1:] == 0).all()
