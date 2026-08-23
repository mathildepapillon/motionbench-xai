"""scripts/_ablation_common.py — Shared machinery for the synthetic ablation scripts.

Single home for the functions previously duplicated across
``run_player_set_ablation.py``, ``run_player_set_budget.py``,
``run_scalability_test.py`` and ``run_nk_ablation.py``:

- ``ec_metrics`` / ``topk_metrics``: the ablation scripts' EC / rank-metric
  conventions (EC1_norm denominator ``+ 1e-8``, degenerate EC3 -> ``nan``,
  ``k = max(1, M // 2)``, ``nan -> 0.0`` coercion) — deliberately distinct
  from ``motionbench.metrics.ground_truth`` (which the graded pipelines use)
  and pinned to the stored ablation records.
- ``batched_oracle_shapley`` (+ Cholesky-cache helpers): oracle ground truth
  with exact Gaussian conditionals.  Draws use fresh entropy per call
  (``default_rng(None)``) — the documented release behaviour; do not seed.

Sources are the (typed, documented) run_player_set_ablation.py versions;
the sibling copies were logic-identical.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from motionbench.oracles.gaussian_oracle import mask_is_spatial, mask_is_temporal


def ec_metrics(phi_hat: np.ndarray, phi_true: np.ndarray) -> dict[str, float]:
    diff = phi_hat - phi_true
    ec1 = float(np.mean(np.abs(diff)))
    denom = float(np.mean(np.abs(phi_true)) + 1e-8)
    ec1_norm = ec1 / denom
    ec2 = float(np.mean(diff**2))
    if np.std(phi_hat) < 1e-10 or np.std(phi_true) < 1e-10:
        ec3 = float("nan")
    else:
        corr = float(np.corrcoef(phi_hat, phi_true)[0, 1])
        ec3 = 1.0 - corr
    return {"ec1": ec1, "ec1_norm": ec1_norm, "ec2": ec2, "ec3": ec3}


def topk_metrics(phi_hat: np.ndarray, phi_true: np.ndarray) -> dict[str, float]:
    abs_h = np.abs(phi_hat)
    abs_t = np.abs(phi_true)
    n = phi_hat.shape[0]
    out: dict[str, float] = {}
    if n >= 1:
        out["top1"] = float(int(np.argmax(abs_h) == np.argmax(abs_t)))
    K_top = max(1, n // 2)
    top_h = set(np.argsort(-abs_h)[:K_top].tolist())
    top_t = set(np.argsort(-abs_t)[:K_top].tolist())
    out["topk_overlap"] = float(len(top_h & top_t) / K_top)
    from scipy.stats import kendalltau, spearmanr

    sp, _ = spearmanr(phi_hat, phi_true)
    kt, _ = kendalltau(phi_hat, phi_true)
    out["spearman"] = float(sp) if not np.isnan(sp) else 0.0
    out["kendall"] = float(kt) if not np.isnan(kt) else 0.0
    return out


def _build_spatiotemporal_cond_params(
    oracle,
    mask_np: np.ndarray,
    J: int,
    F: int,
    T: int,
) -> tuple | None:
    """Pre-compute conditional parameters for a spatiotemporal mask.

    Returns ``(j_obs, j_hid, t_obs, t_hid, W, L_cond)`` or ``None`` if the
    mask is full or empty.  This is the expensive Cholesky step; the result
    can be cached and reused across sequences since it only depends on the
    mask, not on ``x``.
    """
    jt_mask = mask_np.all(axis=1)  # (J, T)
    flat = jt_mask.reshape(-1)
    obs_lin = np.flatnonzero(flat)
    hid_lin = np.flatnonzero(~flat)
    n_obs = int(obs_lin.size)
    n_hid = int(hid_lin.size)
    if n_hid == 0 or n_obs == 0:
        return None
    j_obs = (obs_lin // T).astype(int)
    t_obs = (obs_lin % T).astype(int)
    j_hid = (hid_lin // T).astype(int)
    t_hid = (hid_lin % T).astype(int)
    Sigma_oo = (
        oracle.Sigma_joints[j_obs[:, None], j_obs[None, :]]
        * oracle.Sigma_time[t_obs[:, None], t_obs[None, :]]
    )
    Sigma_hh = (
        oracle.Sigma_joints[j_hid[:, None], j_hid[None, :]]
        * oracle.Sigma_time[t_hid[:, None], t_hid[None, :]]
    )
    Sigma_ho = (
        oracle.Sigma_joints[j_hid[:, None], j_obs[None, :]]
        * oracle.Sigma_time[t_hid[:, None], t_obs[None, :]]
    )
    W = Sigma_ho @ np.linalg.solve(Sigma_oo + 1e-10 * np.eye(n_obs), np.eye(n_obs))
    Sigma_cond = Sigma_hh - W @ Sigma_ho.T
    Sigma_cond = 0.5 * (Sigma_cond + Sigma_cond.T) + 1e-8 * np.eye(n_hid)
    L_cond = np.linalg.cholesky(Sigma_cond)
    return (j_obs, j_hid, t_obs, t_hid, W, L_cond)


def _sample_with_cached_params(
    x_np: np.ndarray,
    params,
    n_mc: int,
    J: int,
    F: int,
    T: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from Gaussian conditional using cached (W, L_cond) params."""
    if params is None:
        # Full coalition: repeat x
        return np.tile(x_np[None].astype(np.float32), (n_mc, 1, 1, 1))
    j_obs, j_hid, t_obs, t_hid, W, L_cond = params
    n_hid = len(j_hid)
    out = np.tile(x_np[None], (n_mc, 1, 1, 1)).astype(np.float64)
    for f in range(F):
        x_obs_vals = x_np[j_obs, f, t_obs]
        mu = W @ x_obs_vals
        z = rng.standard_normal((n_mc, n_hid))
        z_corr = z @ L_cond.T
        out[:, j_hid, f, t_hid] = mu[None, :] + z_corr
    return out.astype(np.float32)


def batched_oracle_shapley(
    oracle,
    x: Tensor,
    clf_fn,  # (B, J, F, T) → (B,) float scalar
    players,
    n_mc: int,
    coalitions: np.ndarray,  # (N_coal, M) pre-sampled coalition matrix
    weights: np.ndarray,  # (N_coal,) Shapley kernel weights
    clf_chunk: int = 1024,
    cholesky_cache: dict | None = None,  # mutable cache passed by caller
) -> Tensor:
    """Batched oracle Shapley evaluation with Cholesky caching.

    Pre-generates all Gaussian conditional samples for every coalition in one
    numpy pass, then calls clf_fn once (chunked) on the stacked batch instead
    of making N_coal separate GPU calls.

    The ``cholesky_cache`` dict is keyed by coalition index and stores the
    expensive Cholesky factorization of the conditional covariance. Sharing
    the same cache across sequences gives a ~n_seq × speedup for the
    spatiotemporal sampling path (used for P_cell / JointWindowCells).

    Args:
        oracle: GaussianOracle instance.
        x: ``(J, F, T)`` float32 Tensor.
        clf_fn: Callable ``(B, J, F, T) → (B,)`` float scalar.
        players: PlayerSet used to build coalition masks.
        n_mc: Monte Carlo samples per coalition (Gaussian conditionals).
        coalitions: ``(N_coal, M)`` int array including boundary rows.
        weights: ``(N_coal,)`` Shapley kernel weights.
        clf_chunk: Maximum batch size per classifier forward pass.
        cholesky_cache: Optional shared dict for Cholesky factor caching.

    Returns:
        ``(M,)`` float32 Tensor of Shapley values.
    """
    from motionbench.utils.coalitions import solve_shapley_wls

    M = players.n_players
    N_coal = coalitions.shape[0]
    x_np = x.detach().cpu().numpy().astype(np.float64)
    J, F, T = x_np.shape
    rng = np.random.default_rng(None)

    # Phase 1: generate all conditional samples in numpy (CPU)
    all_samples: list[np.ndarray] = []  # each: (n_mc, J, F, T) float32
    for ci, z_row in enumerate(coalitions):
        n_obs_players = int(z_row.sum())

        if n_obs_players == M:
            s = np.tile(x_np[None].astype(np.float32), (n_mc, 1, 1, 1))
        elif n_obs_players == 0:
            s = oracle._sample_unconditional(
                n_mc,
                J,
                F,
                T,
                np.random.default_rng(int(rng.integers(1 << 31))),
            )
        else:
            # Try cache first; compute and store on miss
            if cholesky_cache is not None and ci in cholesky_cache:
                params = cholesky_cache[ci]
            else:
                z_t = torch.tensor(z_row, dtype=torch.int32)
                mask = players.coalition_mask(z_t)
                mask_np = mask.numpy().astype(bool)
                # Simple spatial/temporal patterns use the oracle's own
                # cached sampler; only spatiotemporal masks need a factor.
                if mask_is_temporal(mask_np) or mask_is_spatial(mask_np):
                    # Use oracle's built-in cached path
                    params = ("oracle", mask_np)
                else:
                    # Build and cache spatiotemporal Cholesky
                    params = _build_spatiotemporal_cond_params(oracle, mask_np, J, F, T)
                if cholesky_cache is not None:
                    cholesky_cache[ci] = params

            # Sample using cached params.
            # Sentinel ("oracle", mask_np) has len=2; spatiotemporal params
            # tuple has len=6 — distinguish by length, not by array equality.
            if isinstance(params, tuple) and len(params) == 2 and isinstance(params[0], str):
                # Use oracle's built-in sampler (has its own cache)
                _, mask_np_cached = params
                s = oracle._conditional_sample_np(
                    x_np,
                    mask_np_cached,
                    n_mc,
                    np.random.default_rng(int(rng.integers(1 << 31))),
                )
            else:
                s = _sample_with_cached_params(
                    x_np,
                    params,
                    n_mc,
                    J,
                    F,
                    T,
                    np.random.default_rng(int(rng.integers(1 << 31))),
                )

        all_samples.append(s.astype(np.float32))  # (n_mc, J, F, T)

    # Phase 2: stack → (N_coal * n_mc, J, F, T) and run classifier once
    stacked = torch.from_numpy(
        np.concatenate(all_samples, axis=0)  # (N_coal * n_mc, J, F, T)
    )
    vals_flat_list: list[Tensor] = []
    for s in range(0, len(stacked), clf_chunk):
        vals_flat_list.append(clf_fn(stacked[s : s + clf_chunk]))
    vals_flat = torch.cat(vals_flat_list).float()  # (N_coal * n_mc,)

    # Phase 3: reshape and average per coalition
    vals_mat = vals_flat.view(N_coal, n_mc)  # (N_coal, n_mc)
    values = vals_mat.mean(dim=1).numpy().astype(np.float64)  # (N_coal,)

    # Boundary values (first row = all-zero coalition, second = all-one)
    v_empty = float(values[0])
    v_full = float(values[1])

    phi = solve_shapley_wls(coalitions, values, weights, v_empty, v_full)
    return torch.tensor(phi, dtype=torch.float32)
