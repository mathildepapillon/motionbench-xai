"""scripts/run_player_set_budget.py — coalition-budget equalization sweep (reviewer C5/W6).

Re-runs the player-set comparison at **fixed coalition budgets** {16, 64, 256} to
disentangle whether the Ptemp advantage over Pjoint arises from structural
properties or merely from the fact that Ptemp (K=4) can be fully enumerated in
~16 coalitions while Pjoint (M=17) requires sampling.

Combinations
------------
Datasets  : gaussian_k4, skeleton_structured, skeleton_gait_combined
Methods   : kernelshap_vaeac, kernelshap_marginal
Player sets:
  temporal      — TemporalWindows(K=4): M=4 players
  spatial_joint — SpatialJoints(J=J):  M=J players (5 for gaussian_k4, 17 for skeleton)
Budgets   : 16, 64, 256  (passed as --n-coalitions)

Results written to::

    results/player_set_budget/{dataset}/{player_set}/{budget}/{method}/result.json

Usage::

    conda activate motionbench-xai
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python \\
        scripts/run_player_set_budget.py

    # Individual run:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_player_set_budget.py \\
        --datasets skeleton_structured --player-sets temporal spatial_joint \\
        --budgets 16 64 256 --methods kernelshap_vaeac kernelshap_marginal

Skip any combination whose result.json already exists (override with --force).
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
RESULTS_DIR = REPO / "results" / "player_set_budget"

# ---------------------------------------------------------------------------
# Timeout helper
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

    prev = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


# ---------------------------------------------------------------------------
# Metric helpers
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


# ---------------------------------------------------------------------------
# Imputer factory
# ---------------------------------------------------------------------------


def build_imputer(method_base: str, dataset, device_str: str):
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
            raise RuntimeError(f"No VAEAC registry entry for dataset class {cls_key!r}")
        ckpt_rel, cfg_rel = _VAEAC_REGISTRY[cls_key]
        ckpt_dir = _CARE_PD_ROOT / ckpt_rel
        cfg_path = _CARE_PD_ROOT / cfg_rel
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"VAEAC checkpoint dir not found: {ckpt_dir}")
        return _load_vaeac(ckpt_dir, cfg_path, torch.device(device_str))

    raise ValueError(f"Unknown method_base for this script: {method_base!r}. Use 'marginal' or 'vaeac'.")


# ---------------------------------------------------------------------------
# Batched oracle Shapley (copied verbatim from run_player_set_ablation.py)
# ---------------------------------------------------------------------------


def _build_spatiotemporal_cond_params(oracle, mask_np, J, F, T):
    jt_mask = mask_np.all(axis=1)
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


def _sample_with_cached_params(x_np, params, n_mc, J, F, T, rng):
    if params is None:
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
    oracle, x, clf_fn, players, n_mc, coalitions, weights,
    clf_chunk=1024, cholesky_cache=None,
):
    from motionbench.utils.coalitions import solve_shapley_wls
    M = players.n_players
    N_coal = coalitions.shape[0]
    x_np = x.detach().cpu().numpy().astype(np.float64)
    J, F, T = x_np.shape
    rng = np.random.default_rng(None)

    all_samples: list[np.ndarray] = []
    for ci, z_row in enumerate(coalitions):
        n_obs_players = int(z_row.sum())
        if n_obs_players == M:
            s = np.tile(x_np[None].astype(np.float32), (n_mc, 1, 1, 1))
        elif n_obs_players == 0:
            s = oracle._sample_unconditional(
                n_mc, J, F, T, np.random.default_rng(int(rng.integers(1 << 31))),
            )
        else:
            if cholesky_cache is not None and ci in cholesky_cache:
                params = cholesky_cache[ci]
            else:
                z_t = torch.tensor(z_row, dtype=torch.int32)
                mask = players.coalition_mask(z_t)
                mask_np = mask.numpy().astype(bool)
                from motionbench.oracles.gaussian_oracle import (
                    _mask_is_temporal, _mask_is_spatial,
                )
                if _mask_is_temporal(mask_np) or _mask_is_spatial(mask_np):
                    params = ("oracle", mask_np)
                else:
                    params = _build_spatiotemporal_cond_params(oracle, mask_np, J, F, T)
                if cholesky_cache is not None:
                    cholesky_cache[ci] = params

            if isinstance(params, tuple) and params[0] == "oracle":
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
        all_samples.append(s.astype(np.float32))

    stacked = torch.from_numpy(np.concatenate(all_samples, axis=0))
    vals_flat_list: list[Tensor] = []
    for s in range(0, len(stacked), clf_chunk):
        vals_flat_list.append(clf_fn(stacked[s: s + clf_chunk]))
    vals_flat = torch.cat(vals_flat_list).float()
    vals_mat = vals_flat.view(N_coal, n_mc)
    values = vals_mat.mean(dim=1).numpy().astype(np.float64)

    v_empty = float(values[0])
    v_full = float(values[1])
    phi = solve_shapley_wls(coalitions, values, weights, v_empty, v_full)
    return torch.tensor(phi, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Imputation helper
# ---------------------------------------------------------------------------


def impute_one(imp, x_obs: Tensor, mask: Tensor) -> Tensor:
    if hasattr(imp, "impute"):
        comp = imp.impute(x_obs, mask, n_samples=1)
        if comp.ndim == 4:
            comp = comp[0]
    elif hasattr(imp, "sample_completions"):
        from motionbench.imputers.carepd_imputer import _mask_to_coalition
        J, F, T = x_obs.shape
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
    x_obs_cpu = x_obs.cpu().float()
    mask_cpu = mask.cpu()
    comp = torch.where(mask_cpu, x_obs_cpu, comp)
    return comp


# ---------------------------------------------------------------------------
# Dataset / player factory helpers
# ---------------------------------------------------------------------------


def load_dataset(ds_name: str):
    ds_yaml = REPO / "configs" / "data" / f"{ds_name}.yaml"
    ds_cfg = OmegaConf.load(ds_yaml)
    ds_cfg_d = OmegaConf.to_container(ds_cfg, resolve=True)
    ds_cfg_d.pop("K", None)
    target = ds_cfg_d.pop("_target_")
    mod_path, cls_name = target.rsplit(".", 1)
    mod = __import__(mod_path, fromlist=[cls_name])
    return getattr(mod, cls_name)(**ds_cfg_d)


def make_players(player_set: str, J: int, F: int, T: int):
    """Instantiate a player set.

    Args:
        player_set: ``"temporal"`` → TemporalWindows(K=4) or
                    ``"spatial_joint"`` → SpatialJoints(J=J).
        J, F, T: Dataset shape.
    """
    if player_set == "temporal":
        from motionbench.players.temporal_windows import TemporalWindows
        K = 4
        if T % K != 0:
            raise ValueError(f"T={T} must be divisible by K={K}")
        return TemporalWindows(K=K, T=T, J=J, F=F)
    if player_set == "spatial_joint":
        from motionbench.players.spatial_joints import SpatialJoints
        return SpatialJoints(J=J, F=F, T=T)
    raise ValueError(f"Unknown player_set: {player_set!r}. Use 'temporal' or 'spatial_joint'.")


# ---------------------------------------------------------------------------
# Core per-run driver
# ---------------------------------------------------------------------------


def run_one_cell(
    ds_name: str,
    dataset,
    players,
    player_set_tag: str,   # "temporal" | "spatial_joint"
    clf_name: str,
    method_base: str,      # "marginal" | "vaeac"
    budget: int,           # n_coalitions
    device: torch.device,
    n_seq: int,
    cell_timeout_s: int,
) -> dict:
    M = players.n_players
    J, F, T = dataset.shape
    n_classes = int(dataset.metadata.get("n_classes", 3))
    K_ds = 4

    method_name = f"kernelshap_{method_base}"
    out_dir = RESULTS_DIR / ds_name / player_set_tag / str(budget) / method_name
    out_dir.mkdir(parents=True, exist_ok=True)
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
        log.warning("    No checkpoint at %s — using random weights", ckpt_path)

    # Load sequences
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
    imp = build_imputer(method_base, dataset, str(device))
    log.info("    Imputer ready.")

    # Sample KernelSHAP coalitions at the requested budget
    from motionbench.utils.coalitions import sample_kernelshap_coalitions, solve_shapley_wls
    rng_coal = np.random.default_rng(42)
    n_pairs = max(1, budget // 2)
    inner_coalitions, inner_weights = sample_kernelshap_coalitions(M, n_pairs, rng_coal)
    boundary_z = np.array([[0] * M, [1] * M], dtype=np.intp)
    boundary_w = np.zeros(2, dtype=np.float64)
    coalitions = np.vstack([boundary_z, inner_coalitions])   # (2 + 2*n_pairs, M)
    weights = np.concatenate([boundary_w, inner_weights])
    n_coal_total = coalitions.shape[0]
    log.info("    player_set=%s M=%d budget=%d → %d coalitions",
             player_set_tag, M, budget, n_coal_total)

    # Pre-compute coalition masks
    coal_masks: list[Tensor] = []
    for ci in range(n_coal_total):
        z_t = torch.tensor(coalitions[ci], dtype=torch.int32)
        coal_masks.append(players.coalition_mask(z_t))

    phis = np.zeros((n_seq_actual, M), dtype=np.float32)
    v_all = np.zeros((n_seq_actual, n_coal_total), dtype=np.float32)
    phi_true_all = np.zeros((n_seq_actual, M), dtype=np.float32)
    oracle = dataset.oracle
    cholesky_cache: dict = {}

    for i in range(n_seq_actual):
        x_i = seqs[i]
        target_i = int(targets[i])

        comps = torch.zeros(n_coal_total, J, F, T, dtype=torch.float32)
        for ci in range(n_coal_total):
            mask_ci = coal_masks[ci]
            n_obs = int(mask_ci.sum().item())
            if n_obs == J * F * T:
                comps[ci] = x_i.float()
                continue
            if n_obs == 0:
                try:
                    comps[ci] = impute_one(imp, x_i, mask_ci)
                except Exception:
                    comps[ci] = torch.zeros(J, F, T)
                continue
            try:
                comps[ci] = impute_one(imp, x_i, mask_ci)
            except Exception as exc:
                if i == 0 and ci < 3:
                    log.warning("    impute_one failed ci=%d: %s", ci, exc)
                comps[ci] = torch.zeros(J, F, T)

        with torch.no_grad():
            logits_b = clf(comps.to(device))
        if logits_b.ndim == 2:
            v_b = torch.softmax(logits_b, dim=-1)[:, target_i]
        else:
            v_b = logits_b.squeeze(-1)
        v_all[i] = v_b.cpu().numpy()

        phi = solve_shapley_wls(
            coalitions,
            v_b.cpu().numpy().astype(np.float64),
            weights,
            float(v_all[i, 0]),
            float(v_all[i, 1]),
        )
        phis[i] = phi.astype(np.float32)

        # Oracle Shapley values
        try:
            _target_i = target_i

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
            log.info("    seq %d/%d (%.1fs)", i + 1, n_seq_actual, time.time() - t_cell)

    # Aggregate metrics
    ecs = [ec_metrics(phis[i], phi_true_all[i]) for i in range(n_seq_actual)]
    topks = [topk_metrics(phis[i], phi_true_all[i]) for i in range(n_seq_actual)]

    out_dict: dict = {
        "dataset": ds_name,
        "classifier": clf_name,
        "method": method_name,
        "player_set": player_set_tag,
        "n_players": M,
        "n_sequences": int(n_seq_actual),
        "n_coalitions": n_coal_total,
        "budget": budget,
    }
    for k in ("ec1", "ec1_norm", "ec2", "ec3"):
        vals = [e[k] for e in ecs if not np.isnan(e[k])]
        out_dict[k] = float(np.mean(vals)) if vals else float("nan")
        out_dict[f"{k}_mean"] = out_dict[k]   # alias expected by caller
    for k in ("top1", "topk_overlap", "spearman", "kendall"):
        out_dict[k] = float(np.mean([t[k] for t in topks]))

    np.savez_compressed(
        out_dir / "attributions.npz",
        phi=phis, phi_true=phi_true_all, v=v_all,
    )
    (out_dir / "result.json").write_text(json.dumps(out_dict, indent=2))

    wall = time.time() - t_cell
    out_dict["_wall_seconds"] = wall
    log.info(
        "    DONE  dataset=%s  pset=%s  budget=%d  method=%s  EC1=%.4f  spearman=%.3f  (%.1fs)",
        ds_name, player_set_tag, budget, method_name,
        out_dict.get("ec1", float("nan")),
        out_dict.get("spearman", float("nan")),
        wall,
    )
    return out_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DATASETS_DEFAULT = ["gaussian_k4", "skeleton_structured", "skeleton_gait_combined"]
PLAYER_SETS_DEFAULT = ["temporal", "spatial_joint"]
BUDGETS_DEFAULT = [16, 64, 256]
METHODS_DEFAULT = ["kernelshap_vaeac", "kernelshap_marginal"]
CLASSIFIER_DEFAULT = "synthetic_mlp"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=DATASETS_DEFAULT)
    p.add_argument("--player-sets", nargs="+", default=PLAYER_SETS_DEFAULT,
                   choices=["temporal", "spatial_joint"])
    p.add_argument("--budgets", nargs="+", type=int, default=BUDGETS_DEFAULT)
    p.add_argument("--methods", nargs="+", default=METHODS_DEFAULT,
                   choices=["kernelshap_vaeac", "kernelshap_marginal"])
    p.add_argument("--classifier", default=CLASSIFIER_DEFAULT,
                   help="Synthetic classifier name (default: synthetic_mlp).")
    p.add_argument("--n-seq", type=int, default=50)
    p.add_argument("--cell-timeout", type=int, default=3600)
    p.add_argument("--device", default="cuda:0")
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
    log.info("datasets=%s  player_sets=%s  budgets=%s  methods=%s  classifier=%s",
             args.datasets, args.player_sets, args.budgets, args.methods, args.classifier)

    methods_base = [m.removeprefix("kernelshap_") for m in args.methods]

    summary: list[dict] = []

    for ds_name in args.datasets:
        log.info("=" * 70)
        log.info("Dataset: %s", ds_name)
        dataset = load_dataset(ds_name)
        J, F, T = dataset.shape
        log.info("  Shape J=%d F=%d T=%d", J, F, T)

        for player_set in args.player_sets:
            players = make_players(player_set, J, F, T)
            M = players.n_players
            log.info("  Player set: %s  M=%d", player_set, M)

            for budget in args.budgets:
                for method_base in methods_base:
                    method_name = f"kernelshap_{method_base}"
                    result_path = (
                        RESULTS_DIR / ds_name / player_set / str(budget)
                        / method_name / "result.json"
                    )

                    if result_path.exists() and not args.force:
                        log.info("  [SKIP] %s/%s/%d/%s", ds_name, player_set, budget, method_name)
                        try:
                            cached = json.loads(result_path.read_text())
                            summary.append({
                                "dataset": ds_name, "player_set": player_set,
                                "budget": budget, "method": method_name,
                                "n_players": M,
                                "ec1": cached.get("ec1"),
                                "spearman": cached.get("spearman"),
                                "wall_seconds": None, "status": "cached",
                            })
                        except Exception:
                            pass
                        continue

                    label = f"{ds_name}/{player_set}/budget={budget}/{method_name}"
                    log.info("==> RUN  %s", label)
                    try:
                        with cell_timeout(args.cell_timeout):
                            out = run_one_cell(
                                ds_name=ds_name,
                                dataset=dataset,
                                players=players,
                                player_set_tag=player_set,
                                clf_name=args.classifier,
                                method_base=method_base,
                                budget=budget,
                                device=device,
                                n_seq=args.n_seq,
                                cell_timeout_s=args.cell_timeout,
                            )
                        summary.append({
                            "dataset": ds_name, "player_set": player_set,
                            "budget": budget, "method": method_name,
                            "n_players": M,
                            "ec1": out.get("ec1"),
                            "spearman": out.get("spearman"),
                            "wall_seconds": out.get("_wall_seconds"),
                            "status": "ok",
                        })
                    except CellTimeout as exc:
                        log.warning("[TIMEOUT] %s: %s", label, exc)
                        summary.append({
                            "dataset": ds_name, "player_set": player_set,
                            "budget": budget, "method": method_name,
                            "n_players": M,
                            "ec1": None, "spearman": None,
                            "wall_seconds": args.cell_timeout, "status": "timeout",
                        })
                    except Exception as exc:
                        log.warning("[FAIL] %s: %s", label, exc)
                        import traceback
                        traceback.print_exc()
                        summary.append({
                            "dataset": ds_name, "player_set": player_set,
                            "budget": budget, "method": method_name,
                            "n_players": M,
                            "ec1": None, "spearman": None,
                            "wall_seconds": None,
                            "status": f"error:{type(exc).__name__}",
                        })

    # Final summary
    log.info("=" * 70)
    log.info("Total wall-clock: %.1fs", time.time() - t_total)
    log.info("=== Summary ===")
    log.info("%-25s %-15s %6s %-24s %7s %8s %8s  %s",
             "dataset", "player_set", "budget", "method", "M", "EC1", "spearman", "status")
    for r in summary:
        log.info(
            "%-25s %-15s %6d %-24s %7d %8s %8s  %s",
            r["dataset"], r["player_set"], r["budget"], r["method"],
            r.get("n_players", 0),
            f"{r['ec1']:.4f}" if r["ec1"] is not None else "  N/A",
            f"{r['spearman']:.3f}" if r["spearman"] is not None else "  N/A",
            r["status"],
        )

    n_ok = sum(1 for r in summary if r["status"] in ("ok", "cached"))
    n_tot = len(summary)
    log.info("Cells complete: %d / %d", n_ok, n_tot)
    sys.exit(0 if n_ok == n_tot else 1)


if __name__ == "__main__":
    main()
