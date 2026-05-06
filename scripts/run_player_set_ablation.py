"""scripts/run_player_set_ablation.py — cross-player-set ablation sweep.

Runs two sweeps on GPU 7:
  1. skeleton_structured  × P_joint (M=17 spatial joints)
  2. skeleton_gait_combined × P_cell  (M=68 joint-window cells, J×K=17×4)

For each sweep, runs methods:
  - kernelshap_zero_{pjoint,pcell}
  - kernelshap_marginal_{pjoint,pcell}
  - kernelshap_vaeac_{pjoint,pcell}
  - kernelshap_flow_{pjoint,pcell}

KernelSHAP is computed via sampled coalitions (n_coalitions) since M>>12.
Oracle Shapley values are computed via GaussianOracle.true_shapley with the
same player set so EC1/EC2/EC3/spearman/kendall are well-defined.

Usage::

    conda activate motionbench-xai
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=7 python \\
        scripts/run_player_set_ablation.py

Results written to:
    results/synthetic/{dataset}/{clf}/kernelshap_{zero,marginal,vaeac,flow}_{pjoint,pcell}/result.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results" / "synthetic"

# ---------------------------------------------------------------------------
# Per-cell wall-clock timeout
# ---------------------------------------------------------------------------


class CellTimeout(Exception):
    pass


@contextmanager
def cell_timeout(seconds: int) -> Iterator[None]:
    if seconds <= 0:
        yield
        return

    def _handler(signum, frame):  # type: ignore[no-untyped-def]
        raise CellTimeout(f"exceeded {seconds}s wall-clock budget")

    prev_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)


# ---------------------------------------------------------------------------
# Metric helpers (identical semantics to existing scripts)
# ---------------------------------------------------------------------------


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


def efficiency_error(phi: np.ndarray, v_full: float, v_empty: float) -> float:
    return float(abs(phi.sum() - (v_full - v_empty)))


# ---------------------------------------------------------------------------
# Imputer factory
# ---------------------------------------------------------------------------


def build_imputer(method_base: str, dataset, device_str: str):
    """Build and fit the imputer for the given base method name.

    Args:
        method_base: One of {zero, marginal, vaeac, flow}.
        dataset: Fitted dataset object.
        device_str: e.g. "cuda:0".
    """
    if method_base == "zero":
        from motionbench.imputers.off_manifold import ZeroImputer
        imp = ZeroImputer()
        imp.fit(dataset)
        return imp

    if method_base == "marginal":
        from motionbench.imputers.off_manifold import MarginalDonorImputer
        imp = MarginalDonorImputer()
        imp.fit(dataset)
        return imp

    if method_base == "vaeac":
        from motionbench.imputers.carepd_imputer import (
            _CARE_PD_ROOT, _VAEAC_REGISTRY, _load_vaeac,
        )
        cls_key = type(dataset).__name__
        if cls_key not in _VAEAC_REGISTRY:
            raise RuntimeError(f"No VAEAC registry entry for {cls_key}")
        ckpt_rel, cfg_rel = _VAEAC_REGISTRY[cls_key]
        ckpt_dir = _CARE_PD_ROOT / ckpt_rel
        cfg_path = _CARE_PD_ROOT / cfg_rel
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"VAEAC checkpoint dir not found: {ckpt_dir}")
        return _load_vaeac(ckpt_dir, cfg_path, torch.device(device_str))

    if method_base == "flow":
        from motionbench.imputers.carepd_imputer import (
            _CARE_PD_ROOT, _FLOW_REGISTRY, _load_flow,
        )
        import tempfile
        cls_key = type(dataset).__name__
        if cls_key not in _FLOW_REGISTRY:
            raise RuntimeError(f"No Flow registry entry for {cls_key}")
        ckpt_rel, cfg_rel = _FLOW_REGISTRY[cls_key]
        ckpt_dir = _CARE_PD_ROOT / ckpt_rel
        cfg_path = _CARE_PD_ROOT / cfg_rel
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Flow checkpoint dir not found: {ckpt_dir}")
        cfg = json.loads(cfg_path.read_text())
        cfg["num_steps"] = 20   # speed: 20 ODE steps is sufficient for evaluation
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cfg, f)
            tmp_cfg = Path(f.name)
        try:
            imp = _load_flow(ckpt_dir, tmp_cfg, torch.device(device_str))
        finally:
            tmp_cfg.unlink(missing_ok=True)
        return imp

    raise ValueError(f"Unknown method_base: {method_base!r}")


# ---------------------------------------------------------------------------
# Batched oracle Shapley: pre-generate all conditional samples, run one
# big GPU batch, then solve WLS — avoids per-coalition GPU launch overhead.
# ---------------------------------------------------------------------------


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
    jt_mask = mask_np.all(axis=1)         # (J, T)
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
    W = Sigma_ho @ np.linalg.solve(
        Sigma_oo + 1e-10 * np.eye(n_obs), np.eye(n_obs)
    )
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
    clf_fn,                  # (B, J, F, T) → (B,) float scalar
    players,
    n_mc: int,
    coalitions: np.ndarray,  # (N_coal, M) pre-sampled coalition matrix
    weights: np.ndarray,     # (N_coal,) Shapley kernel weights
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
    all_samples: list[np.ndarray] = []   # each: (n_mc, J, F, T) float32
    for ci, z_row in enumerate(coalitions):
        n_obs_players = int(z_row.sum())

        if n_obs_players == M:
            s = np.tile(x_np[None].astype(np.float32), (n_mc, 1, 1, 1))
        elif n_obs_players == 0:
            s = oracle._sample_unconditional(
                n_mc, J, F, T,
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
                # Check if it's a simple spatial/temporal pattern first
                # (GaussianOracle._conditional_sample_np handles these with
                # its own cache — only skip to spatiotemporal when needed)
                from motionbench.oracles.gaussian_oracle import (
                    _mask_is_temporal, _mask_is_spatial,
                )
                if _mask_is_temporal(mask_np) or _mask_is_spatial(mask_np):
                    # Use oracle's built-in cached path
                    params = ("oracle", mask_np)
                else:
                    # Build and cache spatiotemporal Cholesky
                    params = _build_spatiotemporal_cond_params(
                        oracle, mask_np, J, F, T
                    )
                if cholesky_cache is not None:
                    cholesky_cache[ci] = params

            # Sample using cached params
            if isinstance(params, tuple) and params[0] == "oracle":
                # Use oracle's built-in sampler (has its own cache)
                _, mask_np_cached = params
                s = oracle._conditional_sample_np(
                    x_np, mask_np_cached, n_mc,
                    np.random.default_rng(int(rng.integers(1 << 31))),
                )
            else:
                s = _sample_with_cached_params(
                    x_np, params, n_mc, J, F, T,
                    np.random.default_rng(int(rng.integers(1 << 31))),
                )

        all_samples.append(s.astype(np.float32))   # (n_mc, J, F, T)

    # Phase 2: stack → (N_coal * n_mc, J, F, T) and run classifier once
    stacked = torch.from_numpy(
        np.concatenate(all_samples, axis=0)         # (N_coal * n_mc, J, F, T)
    )
    vals_flat_list: list[Tensor] = []
    for s in range(0, len(stacked), clf_chunk):
        vals_flat_list.append(clf_fn(stacked[s: s + clf_chunk]))
    vals_flat = torch.cat(vals_flat_list).float()   # (N_coal * n_mc,)

    # Phase 3: reshape and average per coalition
    vals_mat = vals_flat.view(N_coal, n_mc)         # (N_coal, n_mc)
    values = vals_mat.mean(dim=1).numpy().astype(np.float64)  # (N_coal,)

    # Boundary values (first row = all-zero coalition, second = all-one)
    v_empty = float(values[0])
    v_full = float(values[1])

    phi = solve_shapley_wls(coalitions, values, weights, v_empty, v_full)
    return torch.tensor(phi, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Core imputation: given an imputer and a (J, F, T) mask, return one completion
# ---------------------------------------------------------------------------


def impute_one(imp, x_obs: Tensor, mask: Tensor) -> Tensor:
    """Return one imputed completion (J, F, T) enforcing observed entries.

    Handles both motionbench BaseImputer (has .impute) and raw CARE-PD imputers
    (have .sample_completions).
    """
    if hasattr(imp, "impute"):
        # BaseImputer contract
        comp = imp.impute(x_obs, mask, n_samples=1)  # (1, J, F, T) or (J, F, T)
        if comp.ndim == 4:
            comp = comp[0]
    elif hasattr(imp, "sample_completions"):
        # Raw CARE-PD imputer (VAEACImputer / FlowImputer)
        from motionbench.imputers.carepd_imputer import _mask_to_coalition
        J, F, T = x_obs.shape
        device = imp._device
        x_in = x_obs.unsqueeze(0).to(device)
        pad = torch.ones(1, T, dtype=torch.bool, device=device)
        coalition_mask, _ = _mask_to_coalition(mask)
        coalition_mask = coalition_mask.to(device)
        completions = imp.sample_completions(
            x=x_in,
            y=None,
            mask=pad,
            lengths=None,
            coalition_mask=coalition_mask,
            n_samples=1,
        )                                  # list of 1 × (1, J, F, T)
        comp = torch.cat(completions, dim=0)[0].cpu()  # (J, F, T)
    else:
        raise TypeError(f"Imputer {type(imp)} has neither .impute nor .sample_completions")

    # Enforce observed entries bit-for-bit
    comp = comp.cpu().float()
    x_obs_cpu = x_obs.cpu().float()
    mask_cpu = mask.cpu()
    comp = torch.where(mask_cpu, x_obs_cpu, comp)
    return comp


# ---------------------------------------------------------------------------
# Per-cell driver
# ---------------------------------------------------------------------------


def run_one_cell(
    ds_name: str,
    dataset,
    players,               # SpatialJoints or JointWindowCells instance
    player_set_tag: str,   # "pjoint" or "pcell"
    clf_name: str,
    method_base: str,      # "zero" | "marginal" | "vaeac" | "flow"
    device: torch.device,
    n_seq: int,
    n_coalitions: int,
    cell_timeout_s: int,
) -> dict:
    M = players.n_players
    J, F, T = dataset.shape
    n_classes = int(dataset.metadata.get("n_classes", 3))
    K_ds = 4   # temporal windows for dataset (from YAML)

    method_name = f"kernelshap_{method_base}_{player_set_tag}"
    method_dir = RESULTS_DIR / ds_name / clf_name / method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    t_cell = time.time()

    # Load classifier
    clf_yaml = REPO / "configs" / "classifiers" / f"{clf_name}.yaml"
    clf_cfg = OmegaConf.load(clf_yaml)
    from motionbench.pipelines.synthetic_eval import _build_classifier
    clf = _build_classifier(clf_cfg, J, F, T, K_ds, n_classes).to(device)
    clf.eval()

    ckpt_path = (
        REPO / "motionbench" / "classifiers" / "checkpoints" / "synthetic"
        / ds_name / f"{clf_name}.pt"
    )
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        clf.load_state_dict(ckpt["model_state_dict"])
        log.info("    Loaded classifier from %s", ckpt_path)
    else:
        log.warning("    No checkpoint found at %s — using random weights", ckpt_path)

    # Load sequences and determine targets
    seqs = [dataset[i][0] for i in range(min(n_seq, len(dataset)))]
    n_seq_actual = len(seqs)
    X = torch.stack(seqs).to(device)
    with torch.no_grad():
        logits_X = clf(X)
    targets = (
        logits_X.argmax(dim=-1).cpu().numpy()
        if logits_X.ndim == 2
        else np.zeros(n_seq_actual, dtype=int)
    )

    # Build imputer
    log.info("    Building imputer for method=%s ...", method_base)
    device_str = str(device)
    imp = build_imputer(method_base, dataset, device_str)
    log.info("    Imputer ready.")

    # Sample KernelSHAP coalitions (shared across sequences)
    from motionbench.utils.coalitions import sample_kernelshap_coalitions, solve_shapley_wls
    rng_coal = np.random.default_rng(42)
    n_pairs = max(1, n_coalitions // 2)
    inner_coalitions, inner_weights = sample_kernelshap_coalitions(M, n_pairs, rng_coal)
    boundary_z = np.array([[0] * M, [1] * M], dtype=np.intp)
    boundary_w = np.zeros(2, dtype=np.float64)
    coalitions = np.vstack([boundary_z, inner_coalitions])   # (2+2*n_pairs, M)
    weights = np.concatenate([boundary_w, inner_weights])
    n_coal_total = coalitions.shape[0]
    log.info("    M=%d players, %d coalitions sampled", M, n_coal_total)

    # Pre-compute coalition masks (CPU tensor per coalition)
    coal_masks: list[Tensor] = []
    for ci in range(n_coal_total):
        z_t = torch.tensor(coalitions[ci], dtype=torch.int32)
        coal_masks.append(players.coalition_mask(z_t))   # (J, F, T) bool

    # Per-sequence loop
    phis = np.zeros((n_seq_actual, M), dtype=np.float32)
    v_all = np.zeros((n_seq_actual, n_coal_total), dtype=np.float32)
    phi_true_all = np.zeros((n_seq_actual, M), dtype=np.float32)
    oracle = dataset.oracle
    # Cache for Cholesky factors indexed by coalition index — shared across
    # sequences so the expensive spatiotemporal Cholesky is only computed once.
    cholesky_cache: dict = {}

    for i in range(n_seq_actual):
        x_i = seqs[i]   # (J, F, T) cpu
        target_i = int(targets[i])

        # Impute all coalitions
        comps = torch.zeros(n_coal_total, J, F, T, dtype=torch.float32)
        for ci in range(n_coal_total):
            mask_ci = coal_masks[ci]
            n_obs = int(mask_ci.sum().item())
            if n_obs == J * F * T:
                comps[ci] = x_i.float()
                continue
            if n_obs == 0:
                # empty coalition: zero or unconditional sample
                try:
                    comps[ci] = impute_one(imp, x_i, mask_ci)
                except Exception:
                    comps[ci] = torch.zeros(J, F, T)
                continue
            try:
                comps[ci] = impute_one(imp, x_i, mask_ci)
            except Exception as exc:
                if i == 0 and ci == 0:
                    log.warning("    impute_one failed for ci=%d: %s", ci, exc)
                comps[ci] = torch.zeros(J, F, T)

        # Classifier forward pass (batch over all coalitions)
        with torch.no_grad():
            logits_b = clf(comps.to(device))
        if logits_b.ndim == 2:
            v_b = torch.softmax(logits_b, dim=-1)[:, target_i]
        else:
            v_b = logits_b.squeeze(-1)
        v_all[i] = v_b.cpu().numpy()

        # WLS solve for KernelSHAP
        phi = solve_shapley_wls(
            coalitions,
            v_b.cpu().numpy().astype(np.float64),
            weights,
            float(v_all[i, 0]),   # v_empty = first row (all-zero coalition)
            float(v_all[i, 1]),   # v_full  = second row (all-one coalition)
        )
        phis[i] = phi.astype(np.float32)

        # Oracle true Shapley values for this player set (batched for speed)
        try:
            _target_i = target_i   # capture for closure

            def clf_fn(arr) -> Tensor:
                if isinstance(arr, np.ndarray):
                    t_arr = torch.from_numpy(arr.astype(np.float32)).to(device)
                elif isinstance(arr, Tensor):
                    t_arr = arr.float().to(device)
                else:
                    t_arr = torch.tensor(arr, dtype=torch.float32, device=device)
                with torch.no_grad():
                    o = clf(t_arr)
                if o.ndim == 2:
                    o = torch.softmax(o, dim=-1)[:, _target_i]
                return o.float().cpu()

            phi_true = batched_oracle_shapley(
                oracle=oracle,
                x=x_i,
                clf_fn=clf_fn,
                players=players,
                n_mc=20,
                coalitions=coalitions,
                weights=weights,
                cholesky_cache=cholesky_cache,
            )
            phi_true_all[i] = phi_true.numpy()
        except Exception as exc:
            if i == 0:
                log.warning("    batched_oracle_shapley failed: %s", exc)
                import traceback
                traceback.print_exc()

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_cell
            log.info(
                "    %s/%s/%s seq %d/%d (%.1fs)",
                ds_name, clf_name, method_name, i + 1, n_seq_actual, elapsed,
            )

    # Aggregate metrics
    ecs = [ec_metrics(phis[i], phi_true_all[i]) for i in range(n_seq_actual)]
    topks = [topk_metrics(phis[i], phi_true_all[i]) for i in range(n_seq_actual)]

    out_dict: dict = {
        "dataset": ds_name,
        "classifier": clf_name,
        "method": method_name,
        "n_sequences": int(n_seq_actual),
        "player_set": player_set_tag,
        "n_players": M,
        "n_coalitions": n_coal_total,
    }
    for k in ("ec1", "ec1_norm", "ec2", "ec3"):
        vals = [e[k] for e in ecs if not np.isnan(e[k])]
        out_dict[k] = float(np.mean(vals)) if vals else float("nan")
    for k in ("top1", "topk_overlap", "spearman", "kendall"):
        out_dict[k] = float(np.mean([t[k] for t in topks]))

    # Save
    np.savez_compressed(
        method_dir / "attributions.npz",
        phi=phis, phi_true=phi_true_all, v=v_all,
    )
    (method_dir / "result.json").write_text(json.dumps(out_dict, indent=2))

    wall = time.time() - t_cell
    out_dict["_wall_seconds"] = wall
    log.info(
        "    DONE %s/%s/%s in %.1fs  EC1=%.4f  spearman=%.3f",
        ds_name, clf_name, method_name, wall,
        out_dict.get("ec1", float("nan")),
        out_dict.get("spearman", float("nan")),
    )
    return out_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SWEEP_CONFIGS = [
    # (dataset_name, player_set_tag, player_factory)
    ("skeleton_structured",   "pjoint", "spatial_joints"),
    ("skeleton_gait_combined", "pcell",  "joint_window_cells"),
]

METHODS_BASE = ["zero", "marginal", "vaeac", "flow"]
CLASSIFIERS_DEFAULT = ["synthetic_mlp", "synthetic_cnn", "synthetic_transformer"]


def load_dataset(ds_name: str):
    ds_yaml = REPO / "configs" / "data" / f"{ds_name}.yaml"
    ds_cfg = OmegaConf.load(ds_yaml)
    ds_cfg_d = OmegaConf.to_container(ds_cfg, resolve=True)
    ds_cfg_d.pop("K", None)           # K is pipeline-only
    target = ds_cfg_d.pop("_target_")
    mod_path, cls_name = target.rsplit(".", 1)
    mod = __import__(mod_path, fromlist=[cls_name])
    DatasetCls = getattr(mod, cls_name)
    return DatasetCls(**ds_cfg_d)


def make_players(player_factory: str, J: int, F: int, T: int, K: int):
    if player_factory == "spatial_joints":
        from motionbench.players.spatial_joints import SpatialJoints
        return SpatialJoints(J=J, F=F, T=T)
    if player_factory == "joint_window_cells":
        from motionbench.players.joint_window_cells import JointWindowCells
        return JointWindowCells(J=J, K=K, F=F, T=T)
    raise ValueError(f"Unknown player_factory: {player_factory!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweeps", nargs="+",
                   default=["skeleton_structured_pjoint", "skeleton_gait_combined_pcell"],
                   help="Which sweep combos to run. Format: <dataset>_<pset>.")
    p.add_argument("--classifiers", nargs="+", default=CLASSIFIERS_DEFAULT)
    p.add_argument("--methods", nargs="+", default=METHODS_BASE,
                   choices=METHODS_BASE)
    p.add_argument("--n-seq", type=int, default=50)
    p.add_argument("--n-coalitions", type=int, default=2000)
    p.add_argument("--cell-timeout", type=int, default=3600,
                   help="Per-cell wall-clock timeout in seconds (0=unlimited).")
    p.add_argument("--device", default="cuda:0",
                   help="Torch device. CUDA_VISIBLE_DEVICES already restricts GPUs.")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if result.json already exists.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_total = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    log.info("Repo root: %s", REPO)
    log.info("Device: %s  (CUDA_VISIBLE_DEVICES=%s)",
             device, os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    log.info("n_seq=%d  n_coalitions=%d  cell_timeout=%ds",
             args.n_seq, args.n_coalitions, args.cell_timeout)

    # Resolve requested sweeps
    sweep_map = {
        "skeleton_structured_pjoint":    ("skeleton_structured",   "pjoint", "spatial_joints",     4),
        "skeleton_gait_combined_pcell":  ("skeleton_gait_combined", "pcell", "joint_window_cells",  4),
    }
    requested_sweeps = [sweep_map[s] for s in args.sweeps if s in sweep_map]
    if not requested_sweeps:
        log.error("No valid sweeps requested. Use: %s", list(sweep_map.keys()))
        sys.exit(1)

    summary: list[dict] = []

    for ds_name, player_set_tag, player_factory, K_ds in requested_sweeps:
        log.info("=" * 70)
        log.info("Sweep: %s × %s", ds_name, player_set_tag)

        dataset = load_dataset(ds_name)
        J, F, T = dataset.shape
        players = make_players(player_factory, J, F, T, K_ds)
        M = players.n_players
        log.info("  Dataset %s: J=%d F=%d T=%d  Players M=%d (%s)",
                 ds_name, J, F, T, M, player_set_tag)

        plan = [(clf, meth) for clf in args.classifiers for meth in args.methods]
        log.info("  Planning %d cells", len(plan))

        for clf_name, method_base in plan:
            method_name = f"kernelshap_{method_base}_{player_set_tag}"
            result_path = RESULTS_DIR / ds_name / clf_name / method_name / "result.json"

            if result_path.exists() and not args.force:
                log.info("  [SKIP cached] %s/%s/%s", ds_name, clf_name, method_name)
                try:
                    cached = json.loads(result_path.read_text())
                    summary.append({
                        "dataset": ds_name, "classifier": clf_name,
                        "method": method_name,
                        "ec1": cached.get("ec1"), "spearman": cached.get("spearman"),
                        "wall_seconds": None, "status": "cached",
                    })
                except Exception:
                    pass
                continue

            log.info("==> RUN  %s/%s/%s", ds_name, clf_name, method_name)
            try:
                with cell_timeout(args.cell_timeout):
                    out = run_one_cell(
                        ds_name=ds_name,
                        dataset=dataset,
                        players=players,
                        player_set_tag=player_set_tag,
                        clf_name=clf_name,
                        method_base=method_base,
                        device=device,
                        n_seq=args.n_seq,
                        n_coalitions=args.n_coalitions,
                        cell_timeout_s=args.cell_timeout,
                    )
                summary.append({
                    "dataset": ds_name, "classifier": clf_name,
                    "method": method_name,
                    "ec1": out.get("ec1"), "spearman": out.get("spearman"),
                    "wall_seconds": out.get("_wall_seconds"), "status": "ok",
                })
            except CellTimeout as exc:
                log.warning("[TIMEOUT] %s/%s/%s: %s", ds_name, clf_name, method_name, exc)
                summary.append({
                    "dataset": ds_name, "classifier": clf_name,
                    "method": method_name,
                    "ec1": None, "spearman": None,
                    "wall_seconds": args.cell_timeout, "status": "timeout",
                })
            except Exception as exc:
                log.warning("[FAIL] %s/%s/%s: %s", ds_name, clf_name, method_name, exc)
                import traceback
                traceback.print_exc()
                summary.append({
                    "dataset": ds_name, "classifier": clf_name,
                    "method": method_name,
                    "ec1": None, "spearman": None,
                    "wall_seconds": None, "status": f"error:{type(exc).__name__}",
                })

    log.info("=" * 70)
    log.info("Total wall-clock: %.1fs", time.time() - t_total)
    log.info("=== Per-cell summary ===")
    for row in summary:
        log.info(
            "  %-45s  %-22s  EC1=%-8s  spearman=%-8s  wall=%-7s  %s",
            f"{row['dataset']}/{row['classifier']}",
            row["method"],
            f"{row['ec1']:.4f}" if row["ec1"] is not None else "  N/A",
            f"{row['spearman']:.3f}" if row["spearman"] is not None else "  N/A",
            f"{row['wall_seconds']:.1f}s" if row["wall_seconds"] is not None else "  N/A",
            row["status"],
        )

    n_ok = sum(1 for r in summary if r["status"] in ("ok", "cached"))
    n_tot = len(summary)
    log.info("Cells complete: %d / %d", n_ok, n_tot)
    sys.exit(0 if n_ok == n_tot else 1)


if __name__ == "__main__":
    main()
