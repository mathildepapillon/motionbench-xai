"""scripts/run_scalability_test.py — High-n scalability test for KernelSHAP variants.

Addresses reviewer concern C10: "does the on-manifold (VAEAC) advantage over
marginal imputer survive at much larger player counts (M=16 frame-level players
vs M=4 window players)?"

Setup
-----
Dataset : gaussian_k4  (J=5, F=3, T=16, N=200)
Methods : kernelshap_vaeac, kernelshap_marginal
Players :
  - "window"  → TemporalWindows(K=4,  T=16)   M=4  (baseline)
  - "frame"   → TemporalWindows(K=16, T=16)   M=16 (individual frames, high-n)
Budgets : n_coalitions ∈ {64, 256, 1024}  (inner pairs; actual = budget+2)
Seeds   : 3 per (budget × M × method)

EC1 is computed against oracle Shapley values (GaussianOracle with 2000-coalition
WLS + 50 MC samples per coalition).

Usage::

    conda activate motionbench-xai
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=7 python \\
        scripts/run_scalability_test.py [--n-seq 30] [--device cuda:0]

Results::

    results/scalability/{player_type}/{budget}/seed{s}/{method}/result.json
    results/scalability/summary.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RESULTS_DIR = REPO / "results" / "scalability"

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
DATASET_NAME = "gaussian_k4"
CLF_NAME = "synthetic_mlp"
J, F, T = 5, 3, 16
N_CLASSES = 3

PLAYER_CONFIGS = [
    ("window", 4),    # M=4 temporal windows (baseline)
    ("frame",  16),   # M=T=16 individual frames (high-n)
]

BUDGETS = [64, 256, 1024]
SEEDS = [0, 1, 2]
METHODS = ["vaeac", "marginal"]

# Oracle evaluation budget: use 1000 pairs → 2002 coalitions
N_ORACLE_PAIRS = 1000
N_ORACLE_MC = 50   # Gaussian conditional MC samples per coalition


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def ec_metrics(phi_hat: np.ndarray, phi_true: np.ndarray) -> dict[str, float]:
    diff = phi_hat - phi_true
    ec1 = float(np.mean(np.abs(diff)))
    denom = float(np.mean(np.abs(phi_true)) + 1e-8)
    ec1_norm = ec1 / denom
    ec2 = float(np.mean(diff ** 2))
    if np.std(phi_hat) < 1e-10 or np.std(phi_true) < 1e-10:
        ec3 = float("nan")
    else:
        corr = float(np.corrcoef(phi_hat, phi_true)[0, 1])
        ec3 = 1.0 - corr
    return {"ec1": ec1, "ec1_norm": ec1_norm, "ec2": ec2, "ec3": ec3}


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def load_dataset():
    from omegaconf import OmegaConf
    ds_yaml = REPO / "configs" / "data" / f"{DATASET_NAME}.yaml"
    ds_cfg = OmegaConf.to_container(OmegaConf.load(ds_yaml), resolve=True)
    ds_cfg.pop("K", None)
    target = ds_cfg.pop("_target_")
    mod_path, cls_name = target.rsplit(".", 1)
    mod = __import__(mod_path, fromlist=[cls_name])
    DatasetCls = getattr(mod, cls_name)
    return DatasetCls(**ds_cfg)


# ---------------------------------------------------------------------------
# Classifier loader
# ---------------------------------------------------------------------------


def load_classifier(device: torch.device):
    from omegaconf import OmegaConf
    clf_yaml = REPO / "configs" / "classifiers" / f"{CLF_NAME}.yaml"
    clf_cfg = OmegaConf.load(clf_yaml)
    from motionbench.pipelines.synthetic_eval import _build_classifier
    clf = _build_classifier(clf_cfg, J, F, T, 4, N_CLASSES).to(device)
    ckpt_path = (
        REPO / "motionbench" / "classifiers" / "checkpoints" / "synthetic"
        / DATASET_NAME / f"{CLF_NAME}.pt"
    )
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        clf.load_state_dict(ckpt["model_state_dict"])
        log.info("Loaded classifier from %s", ckpt_path)
    else:
        log.warning("No checkpoint at %s — using random weights", ckpt_path)
    clf.eval()
    return clf


# ---------------------------------------------------------------------------
# Imputer factory
# ---------------------------------------------------------------------------


def build_marginal_imputer(dataset):
    from motionbench.imputers.off_manifold import MarginalDonorImputer
    imp = MarginalDonorImputer()
    imp.fit(dataset)
    return imp


def build_vaeac_imputer(dataset, device_str: str):
    """Load raw CARE-PD VAEAC for GaussianMotionDataset."""
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


# ---------------------------------------------------------------------------
# Single coalition imputation (handles both BaseImputer and raw CARE-PD)
# ---------------------------------------------------------------------------


def impute_one(imp, x_obs: Tensor, mask: Tensor) -> Tensor:
    """Return one imputed (J, F, T) completion; preserve observed entries."""
    if hasattr(imp, "impute"):
        comp = imp.impute(x_obs, mask, n_samples=1)
        if comp.ndim == 4:
            comp = comp[0]
    elif hasattr(imp, "sample_completions"):
        from motionbench.imputers.carepd_imputer import _mask_to_coalition
        device = imp._device
        x_in = x_obs.unsqueeze(0).to(device)
        pad = torch.ones(1, T, dtype=torch.bool, device=device)
        coalition_mask, _ = _mask_to_coalition(mask)
        coalition_mask = coalition_mask.to(device)
        completions = imp.sample_completions(
            x=x_in, y=None, mask=pad, lengths=None,
            coalition_mask=coalition_mask, n_samples=1,
        )
        comp = torch.cat(completions, dim=0)[0].cpu()
    else:
        raise TypeError(f"Imputer {type(imp)} has neither .impute nor .sample_completions")

    comp = comp.cpu().float()
    comp = torch.where(mask.cpu(), x_obs.cpu().float(), comp)
    return comp


# ---------------------------------------------------------------------------
# Oracle Shapley — reuse batched path from run_player_set_ablation.py
# ---------------------------------------------------------------------------


def _build_spatiotemporal_cond_params(oracle, mask_np, rng=None):
    """Pre-compute conditional covariance Cholesky for a given mask."""
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


def batched_oracle_shapley(
    oracle,
    x: Tensor,
    clf_fn,
    players,
    n_mc: int,
    coalitions: np.ndarray,
    weights: np.ndarray,
    clf_chunk: int = 512,
    cholesky_cache: dict | None = None,
) -> Tensor:
    """Batched oracle Shapley with Cholesky caching.

    Uses exact Gaussian conditionals; caches Cholesky factors across
    sequences for significant speedup.
    """
    from motionbench.utils.coalitions import solve_shapley_wls
    from motionbench.oracles.gaussian_oracle import _mask_is_temporal, _mask_is_spatial

    M = players.n_players
    N_coal = coalitions.shape[0]
    x_np = x.detach().cpu().numpy().astype(np.float64)
    rng = np.random.default_rng(None)

    # Phase 1: generate all conditional samples (CPU numpy)
    all_samples: list[np.ndarray] = []
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
            if cholesky_cache is not None and ci in cholesky_cache:
                params = cholesky_cache[ci]
            else:
                z_t = torch.tensor(z_row, dtype=torch.int32)
                mask = players.coalition_mask(z_t)
                mask_np = mask.numpy().astype(bool)
                if _mask_is_temporal(mask_np) or _mask_is_spatial(mask_np):
                    params = ("oracle", mask_np)
                else:
                    params = _build_spatiotemporal_cond_params(oracle, mask_np)
                if cholesky_cache is not None:
                    cholesky_cache[ci] = params

            if isinstance(params, tuple) and params[0] == "oracle":
                _, mask_np_cached = params
                s = oracle._conditional_sample_np(
                    x_np, mask_np_cached, n_mc,
                    np.random.default_rng(int(rng.integers(1 << 31))),
                )
            elif params is None:
                s = np.tile(x_np[None].astype(np.float32), (n_mc, 1, 1, 1))
            else:
                j_obs, j_hid, t_obs, t_hid, W, L_cond = params
                n_hid = len(j_hid)
                out_np = np.tile(x_np[None], (n_mc, 1, 1, 1)).astype(np.float64)
                sample_rng = np.random.default_rng(int(rng.integers(1 << 31)))
                for f in range(F):
                    x_obs_vals = x_np[j_obs, f, t_obs]
                    mu = W @ x_obs_vals
                    z_noise = sample_rng.standard_normal((n_mc, n_hid))
                    z_corr = z_noise @ L_cond.T
                    out_np[:, j_hid, f, t_hid] = mu[None, :] + z_corr
                s = out_np.astype(np.float32)
        all_samples.append(s)

    # Phase 2: batch classifier
    stacked = torch.from_numpy(np.concatenate(all_samples, axis=0))
    vals_list: list[Tensor] = []
    for start in range(0, len(stacked), clf_chunk):
        vals_list.append(clf_fn(stacked[start: start + clf_chunk]))
    vals_flat = torch.cat(vals_list).float()

    vals_mat = vals_flat.view(N_coal, n_mc)
    values = vals_mat.mean(dim=1).numpy().astype(np.float64)
    v_empty = float(values[0])
    v_full = float(values[1])

    phi = solve_shapley_wls(coalitions, values, weights, v_empty, v_full)
    return torch.tensor(phi, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Run one scalability cell
# ---------------------------------------------------------------------------


def run_cell(
    method: str,        # "vaeac" or "marginal"
    player_type: str,   # "window" or "frame"
    M: int,
    n_coalitions: int,
    seed: int,
    players,
    seqs: list[Tensor],
    targets: np.ndarray,
    oracle_phis: np.ndarray,   # (n_seq, M) precomputed oracle Shapley
    imp,                       # pre-built imputer
    clf,
    device: torch.device,
) -> dict[str, Any]:
    """Run KernelSHAP with the given budget and seed; return metrics."""
    from motionbench.utils.coalitions import sample_kernelshap_coalitions, solve_shapley_wls

    n_seq = len(seqs)
    rng = np.random.default_rng(seed * 1000 + n_coalitions)
    n_pairs = max(1, n_coalitions // 2)
    inner_coal, inner_w = sample_kernelshap_coalitions(M, n_pairs, rng)
    boundary_z = np.array([[0] * M, [1] * M], dtype=np.intp)
    boundary_w = np.zeros(2, dtype=np.float64)
    coalitions = np.vstack([boundary_z, inner_coal])
    weights = np.concatenate([boundary_w, inner_w])
    n_coal_total = coalitions.shape[0]

    # Pre-compute coalition masks (reused across sequences)
    coal_masks: list[Tensor] = []
    for ci in range(n_coal_total):
        z_t = torch.tensor(coalitions[ci], dtype=torch.int32)
        coal_masks.append(players.coalition_mask(z_t))

    phis = np.zeros((n_seq, M), dtype=np.float32)
    t_start = time.time()

    for i, x_i in enumerate(seqs):
        target_i = int(targets[i])
        comps = torch.zeros(n_coal_total, J, F, T, dtype=torch.float32)

        for ci in range(n_coal_total):
            mask_ci = coal_masks[ci]
            n_obs = int(mask_ci.sum().item())
            if n_obs == J * F * T:
                comps[ci] = x_i.float()
            elif n_obs == 0:
                try:
                    comps[ci] = impute_one(imp, x_i, mask_ci)
                except Exception:
                    comps[ci] = torch.zeros(J, F, T)
            else:
                try:
                    comps[ci] = impute_one(imp, x_i, mask_ci)
                except Exception as exc:
                    if i == 0:
                        log.debug("impute_one failed ci=%d: %s", ci, exc)
                    comps[ci] = torch.zeros(J, F, T)

        # Classifier batch forward
        with torch.no_grad():
            logits_b = clf(comps.to(device))
        if logits_b.ndim == 2:
            v_b = torch.softmax(logits_b, dim=-1)[:, target_i].cpu().numpy()
        else:
            v_b = logits_b.squeeze(-1).cpu().numpy()

        # WLS solve
        phi = solve_shapley_wls(
            coalitions,
            v_b.astype(np.float64),
            weights,
            float(v_b[0]),
            float(v_b[1]),
        )
        phis[i] = phi.astype(np.float32)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            log.info(
                "    %s/%s/budget%d/seed%d  seq %d/%d  %.1fs",
                method, player_type, n_coalitions, seed, i + 1, n_seq, elapsed,
            )

    wall_time = time.time() - t_start

    # Per-sequence EC1 vs oracle
    ec1_per_seq = [
        float(np.mean(np.abs(phis[i] - oracle_phis[i])))
        for i in range(n_seq)
    ]
    ec1_mean = float(np.mean(ec1_per_seq))
    ec1_std = float(np.std(ec1_per_seq))

    # Additional metrics averaged over sequences
    metrics_per_seq = [ec_metrics(phis[i], oracle_phis[i]) for i in range(n_seq)]
    ec1_norm_mean = float(np.mean([m["ec1_norm"] for m in metrics_per_seq]))

    result = {
        "ec1_mean": ec1_mean,
        "ec1_std": ec1_std,
        "ec1_norm_mean": ec1_norm_mean,
        "n_sequences": n_seq,
        "method": f"kernelshap_{method}",
        "player_type": player_type,
        "n_players": M,
        "n_coalitions": n_coal_total,
        "n_coalitions_budget": n_coalitions,
        "seed": seed,
        "wall_time_s": wall_time,
        "wall_time_per_seq_s": wall_time / max(n_seq, 1),
        "wall_time_per_coalition_s": wall_time / max(n_coal_total * n_seq, 1),
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-seq", type=int, default=30,
                   help="Number of sequences to evaluate per cell.")
    p.add_argument("--device", default="cuda:0",
                   help="Torch device string.")
    p.add_argument("--player-types", nargs="+", default=["window", "frame"],
                   choices=["window", "frame"],
                   help="Which player configurations to run.")
    p.add_argument("--methods", nargs="+", default=["vaeac", "marginal"],
                   choices=["vaeac", "marginal"])
    p.add_argument("--budgets", type=int, nargs="+", default=BUDGETS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--force", action="store_true",
                   help="Re-run even if result.json already exists.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info("Device: %s  (CUDA_VISIBLE_DEVICES=%s)",
             device, os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))

    # ----- Load dataset and classifier -----
    log.info("Loading dataset %s ...", DATASET_NAME)
    dataset = load_dataset()
    log.info("Dataset shape: J=%d F=%d T=%d  N=%d", J, F, T, len(dataset))

    log.info("Loading classifier %s ...", CLF_NAME)
    clf = load_classifier(device)
    oracle = dataset.oracle

    # ----- Sample sequences (shared across all cells) -----
    n_seq = min(args.n_seq, len(dataset))
    seqs = [dataset[i][0] for i in range(n_seq)]
    X = torch.stack(seqs).to(device)
    with torch.no_grad():
        logits_X = clf(X)
    targets = (
        logits_X.argmax(dim=-1).cpu().numpy()
        if logits_X.ndim == 2
        else np.zeros(n_seq, dtype=int)
    )

    # Build per-sequence classifier function factory
    def make_clf_fn(target_i: int):
        def clf_fn(arr) -> Tensor:
            if isinstance(arr, np.ndarray):
                t_arr = torch.from_numpy(arr.astype(np.float32)).to(device)
            else:
                t_arr = arr.float().to(device)
            with torch.no_grad():
                o = clf(t_arr)
            if o.ndim == 2:
                o = torch.softmax(o, dim=-1)[:, target_i]
            return o.float().cpu()
        return clf_fn

    # ----- Build imputers (once each) -----
    log.info("Building imputers ...")
    vaeac_available = False
    vaeac_imp = None
    if "vaeac" in args.methods:
        try:
            device_str = str(device)
            vaeac_imp = build_vaeac_imputer(dataset, device_str)
            vaeac_available = True
            log.info("VAEAC imputer loaded.")
        except Exception as exc:
            log.warning("VAEAC imputer unavailable: %s", exc)

    marginal_imp = None
    if "marginal" in args.methods:
        marginal_imp = build_marginal_imputer(dataset)
        log.info("Marginal imputer ready (pool size=%d).", len(dataset))

    summary: list[dict] = []

    # ----- Per-player-type loop -----
    requested_player_configs = [
        (pt, K) for (pt, K) in PLAYER_CONFIGS if pt in args.player_types
    ]

    for player_type, K in requested_player_configs:
        from motionbench.players.temporal_windows import TemporalWindows
        players = TemporalWindows(K=K, T=T, J=J, F=F)
        M = players.n_players
        log.info("=" * 70)
        log.info(
            "Player config: %s  K=%d  M=%d  (window_size=%d)",
            player_type, K, M, T // K,
        )

        # ----- Pre-compute oracle Shapley values for this player config -----
        log.info(
            "Pre-computing oracle Shapley values for M=%d "
            "(n_oracle_pairs=%d, n_mc=%d) ...",
            M, N_ORACLE_PAIRS, N_ORACLE_MC,
        )
        oracle_phis = np.zeros((n_seq, M), dtype=np.float32)
        oracle_coal_rng = np.random.default_rng(99999)
        from motionbench.utils.coalitions import sample_kernelshap_coalitions, solve_shapley_wls
        oracle_inner_coal, oracle_inner_w = sample_kernelshap_coalitions(
            M, N_ORACLE_PAIRS, oracle_coal_rng
        )
        oracle_boundary_z = np.array([[0] * M, [1] * M], dtype=np.intp)
        oracle_boundary_w = np.zeros(2, dtype=np.float64)
        oracle_coalitions = np.vstack([oracle_boundary_z, oracle_inner_coal])
        oracle_weights = np.concatenate([oracle_boundary_w, oracle_inner_w])
        cholesky_cache: dict = {}

        t_oracle = time.time()
        for i, x_i in enumerate(seqs):
            target_i = int(targets[i])
            clf_fn = make_clf_fn(target_i)
            try:
                phi_oracle = batched_oracle_shapley(
                    oracle=oracle,
                    x=x_i,
                    clf_fn=clf_fn,
                    players=players,
                    n_mc=N_ORACLE_MC,
                    coalitions=oracle_coalitions,
                    weights=oracle_weights,
                    cholesky_cache=cholesky_cache,
                )
                oracle_phis[i] = phi_oracle.numpy()
            except Exception as exc:
                log.warning("Oracle Shapley failed for seq %d: %s", i, exc)

            if (i + 1) % 10 == 0:
                log.info(
                    "  Oracle %s: seq %d/%d  %.1fs",
                    player_type, i + 1, n_seq, time.time() - t_oracle,
                )

        log.info(
            "  Oracle done in %.1fs  |phi_oracle| mean=%.4f",
            time.time() - t_oracle,
            float(np.mean(np.abs(oracle_phis))),
        )

        # Save oracle Shapley for inspection
        oracle_dir = RESULTS_DIR / player_type / "oracle"
        oracle_dir.mkdir(parents=True, exist_ok=True)
        np.save(oracle_dir / "phi_oracle.npy", oracle_phis)

        # ----- Per-budget/seed/method sweep -----
        for n_coalitions in args.budgets:
            for seed in args.seeds:
                for method in args.methods:
                    # Output path
                    out_dir = (
                        RESULTS_DIR / player_type
                        / str(n_coalitions)
                        / f"seed{seed}"
                        / f"kernelshap_{method}"
                    )
                    out_path = out_dir / "result.json"
                    if out_path.exists() and not args.force:
                        log.info("[SKIP] %s", out_path)
                        try:
                            cached = json.loads(out_path.read_text())
                            summary.append(cached)
                        except Exception:
                            pass
                        continue

                    # Select imputer
                    if method == "vaeac":
                        if not vaeac_available:
                            log.warning("Skipping vaeac (not available)")
                            continue
                        imp = vaeac_imp
                    else:
                        imp = marginal_imp

                    log.info(
                        "==> %s / %s / budget=%d / seed=%d",
                        player_type, method, n_coalitions, seed,
                    )

                    try:
                        result = run_cell(
                            method=method,
                            player_type=player_type,
                            M=M,
                            n_coalitions=n_coalitions,
                            seed=seed,
                            players=players,
                            seqs=seqs,
                            targets=targets,
                            oracle_phis=oracle_phis,
                            imp=imp,
                            clf=clf,
                            device=device,
                        )
                    except Exception as exc:
                        import traceback
                        log.error("FAILED %s/%s/%d/%d: %s", player_type, method,
                                  n_coalitions, seed, exc)
                        traceback.print_exc()
                        result = {
                            "ec1_mean": float("nan"),
                            "ec1_std": float("nan"),
                            "ec1_norm_mean": float("nan"),
                            "n_sequences": n_seq,
                            "method": f"kernelshap_{method}",
                            "player_type": player_type,
                            "n_players": M,
                            "n_coalitions": n_coalitions + 2,
                            "n_coalitions_budget": n_coalitions,
                            "seed": seed,
                            "wall_time_s": float("nan"),
                            "wall_time_per_seq_s": float("nan"),
                            "wall_time_per_coalition_s": float("nan"),
                            "error": str(exc),
                        }

                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps(result, indent=2))
                    log.info(
                        "  EC1=%.4f±%.4f  wall=%.1fs  -> %s",
                        result.get("ec1_mean", float("nan")),
                        result.get("ec1_std", float("nan")),
                        result.get("wall_time_s", float("nan")),
                        out_path,
                    )
                    summary.append(result)

    # ----- Aggregate summary -----
    log.info("=" * 70)
    log.info("Building summary ...")

    # Group by (player_type, n_coalitions_budget, method) and aggregate over seeds
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in summary:
        if "error" in r:
            continue
        key = (
            r.get("player_type", ""),
            r.get("n_coalitions_budget", r.get("n_coalitions", 0)),
            r.get("method", ""),
        )
        groups[key].append(r)

    agg_rows: list[dict] = []
    for (player_type, budget, method), rows in sorted(groups.items()):
        ec1_vals = [r["ec1_mean"] for r in rows if not np.isnan(r["ec1_mean"])]
        wall_vals = [r["wall_time_s"] for r in rows
                     if r.get("wall_time_s") and not np.isnan(r["wall_time_s"])]
        n_players_vals = [r.get("n_players", 0) for r in rows]
        agg_rows.append({
            "player_type": player_type,
            "n_players": int(np.mean(n_players_vals)) if n_players_vals else 0,
            "n_coalitions_budget": budget,
            "method": method,
            "n_seeds": len(rows),
            "ec1_mean_across_seeds": float(np.mean(ec1_vals)) if ec1_vals else float("nan"),
            "ec1_std_across_seeds": float(np.std(ec1_vals)) if ec1_vals else float("nan"),
            "wall_time_mean_s": float(np.mean(wall_vals)) if wall_vals else float("nan"),
        })

    summary_path = RESULTS_DIR / "summary.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "per_cell": summary,
        "aggregated": agg_rows,
    }
    summary_path.write_text(json.dumps(summary_data, indent=2))
    log.info("Summary written to %s", summary_path)

    # Print aggregated table
    log.info("\n=== Aggregated Results ===")
    log.info(
        "  %-8s  %-4s  %-12s  %-25s  EC1±std (across seeds)  wall(s)",
        "player", "M", "budget", "method",
    )
    log.info("-" * 90)
    for row in agg_rows:
        log.info(
            "  %-8s  %-4d  %-12d  %-25s  %.4f ± %.4f            %.1f",
            row["player_type"],
            row["n_players"],
            row["n_coalitions_budget"],
            row["method"],
            row.get("ec1_mean_across_seeds", float("nan")),
            row.get("ec1_std_across_seeds", float("nan")),
            row.get("wall_time_mean_s", float("nan")),
        )

    log.info("\nAll done. Results in: %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
