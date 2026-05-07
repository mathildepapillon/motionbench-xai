"""scripts/run_care_pd_extended.py — multi-fold CARE-PD sweep (n=200) with
per-sequence faithfulness/AOPC stored for downstream CI computation.

Standalone CARE-PD extended runner.  The earlier `run_care_pd_fast.py`
prototype (now in `archive/scripts/`) hardcoded `N_SEQ=50` and was unsafe to
leave in the active tree; this script supersedes it with three additions:

* Takes a ``--fold`` argument selecting one of fold1/fold2/fold3 (default 1).
* Default ``--n_seq=200`` (vs 50 in the fast script).
* Stores **per-sequence** faithfulness and AOPC arrays alongside the means
  in ``results/care_pd_extended/foldN/<method>/result.json`` so that
  ``scripts/compute_real_cis.py`` can pool across folds and bootstrap CIs.

Imputers: only fold1 imputer checkpoints exist in CARE-PD's experiment_outs
(VAEAC and Flow). The fold1 imputer is reused for fold2/fold3 — this is
acceptable because both are pre-trained generative models of healthy/PD gait
that do not depend on the held-out classifier-evaluation clips. The fold-
specific quantity is the MotionBERT classifier (UPDRS-gait predictor).

Usage::

    conda activate motionbench-xai
    CUDA_VISIBLE_DEVICES=0 python scripts/run_care_pd_extended.py --fold 1
    CUDA_VISIBLE_DEVICES=0 python scripts/run_care_pd_extended.py --fold 2
    CUDA_VISIBLE_DEVICES=0 python scripts/run_care_pd_extended.py --fold 3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "care_pd_extended"
CARE_PD_ROOT = Path(os.environ.get("CARE_PD_ROOT", REPO_ROOT.parent / "CARE-PD"))
CACHE_TEMPLATE = str(
    CARE_PD_ROOT / "cache" / "flow_matching"
    / "BMCLab_h36m_80_classifier23fold_fold{fold}_eval" / "cache.npz"
)
# Per-fold _eval caches are validation-only (x1_train is empty). The training
# pool from the legacy fold1 cache is reused as a generic donor source for the
# marginal imputer — these are the same BMCLab clips the imputers were
# trained on, fold-agnostic in distribution.
TRAIN_POOL_CACHE = CARE_PD_ROOT / "cache" / "flow_matching" / "BMCLab_h36m_80_fold1" / "cache.npz"
# Backwards-compat alias used by older callsites in this script.
CAREPD_ROOT = REPO_ROOT
CKPT_TEMPLATE = (
    "motionbench/classifiers/checkpoints/real/"
    "carepd_bmclab_fold{fold}_motionbert.pt"
)

K = 4              # temporal windows
DEVICE = "cuda:0"  # overridden by CUDA_VISIBLE_DEVICES


# ---------------------------------------------------------------------- #
# Coalition masks                                                         #
# ---------------------------------------------------------------------- #


def build_coalition_masks(K: int, T: int) -> tuple[Tensor, Tensor]:
    """Return (2^K, K) bool coalition indicators and (2^K, T) frame masks."""
    n_coal = 1 << K
    win_size = T // K
    z_bin = np.zeros((n_coal, K), dtype=bool)
    frame_mask = np.zeros((n_coal, T), dtype=bool)
    for ci in range(n_coal):
        for k in range(K):
            if (ci >> k) & 1:
                z_bin[ci, k] = True
                t0 = k * win_size
                t1 = t0 + win_size if k < K - 1 else T
                frame_mask[ci, t0:t1] = True
    return torch.from_numpy(z_bin), torch.from_numpy(frame_mask)


# ---------------------------------------------------------------------- #
# Imputer batched paths                                                   #
# ---------------------------------------------------------------------- #


def build_completions_offmanifold(
    x: Tensor, frame_mask_2k: Tensor, kind: str, mean_per_jf: Tensor | None = None,
) -> Tensor:
    """Construct (16, J, F, T) batched completions for off-manifold imputers."""
    J, F, T = x.shape
    n_coal = frame_mask_2k.shape[0]
    obs = frame_mask_2k.view(n_coal, 1, 1, T).expand(n_coal, J, F, T)

    if kind == "zero":
        fill = torch.zeros_like(x)
    elif kind == "mean":
        if mean_per_jf is None:
            raise ValueError("mean kind requires mean_per_jf")
        fill = mean_per_jf.view(J, F, 1).expand(J, F, T)
    elif kind == "marginal":
        if mean_per_jf is None:
            raise ValueError("marginal kind requires donor in mean_per_jf (J,F,T)")
        fill = mean_per_jf
    else:
        raise ValueError(f"unknown kind {kind!r}")

    x_b = x.view(1, J, F, T).expand(n_coal, J, F, T)
    fill_b = fill.view(1, J, F, T).expand(n_coal, J, F, T)
    return torch.where(obs, x_b, fill_b).contiguous()


# ---------------------------------------------------------------------- #
# Closed-form KernelSHAP for K small                                      #
# ---------------------------------------------------------------------- #


def shapley_kernel(K: int, s: int) -> float:
    """Lundberg & Lee (2017) Shapley kernel weight for size s coalition."""
    if s == 0 or s == K:
        return 1e6
    from math import comb
    return (K - 1) / (comb(K, s) * s * (K - s))


def kernel_shap_exact(z_bin: Tensor, v_vals: Tensor, K: int) -> Tensor:
    """Solve KernelSHAP weighted least squares over all 2^K coalitions."""
    Z = z_bin.float().numpy()
    v = v_vals.float().numpy()
    n = Z.shape[0]
    sizes = Z.sum(axis=1).astype(int)
    w = np.array([shapley_kernel(K, int(s)) for s in sizes])
    Z_ext = np.concatenate([np.ones((n, 1)), Z], axis=1)
    W = np.diag(w)
    A = Z_ext.T @ W @ Z_ext
    b = Z_ext.T @ W @ v
    A += 1e-8 * np.eye(A.shape[0])
    sol = np.linalg.solve(A, b)
    phi = sol[1:]
    return torch.from_numpy(phi).float()


# ---------------------------------------------------------------------- #
# Metrics                                                                 #
# ---------------------------------------------------------------------- #


def faithfulness_correlation(z_bin: Tensor, v_vals: Tensor, phi: Tensor) -> float:
    """Pearson correlation between sum-of-attributions-of-removed-players
    and (v(N) - v(S)) over all coalitions."""
    z = z_bin.float()
    not_z = (1.0 - z)
    sum_phi_absent = not_z @ phi
    delta = v_vals[-1] - v_vals
    a = sum_phi_absent.numpy()
    b = delta.float().numpy()
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def player_aopc(v_vals: Tensor, z_bin: Tensor, phi: Tensor, K: int) -> float:
    """Area Over Player-removal Curve."""
    order = torch.argsort(phi.abs(), descending=True).tolist()
    v_full = v_vals[-1].item()
    drops: list[float] = []
    cur_z = z_bin[-1].clone()
    for k in order:
        cur_z[k] = False
        idx = int((cur_z.int() * (1 << torch.arange(K))).sum().item())
        drops.append(v_full - v_vals[idx].item())
    if not drops:
        return 0.0
    return float(np.mean(drops))


# ---------------------------------------------------------------------- #
# Main                                                                    #
# ---------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fold", type=int, default=1, choices=list(range(1, 24)),
                    help="CARE-PD classifier fold (1..23). Imputers always use fold1 ckpts.")
    ap.add_argument("--n_seq", type=int, default=200,
                    help="Number of validation sequences to evaluate.")
    ap.add_argument("--methods", type=str, nargs="+", default=None,
                    help="Subset of methods to run; default = all 5.")
    ap.add_argument("--results_dir", type=str, default=str(RESULTS_DIR),
                    help="Output root (per-fold subdir created automatically).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    fold = args.fold
    N_SEQ = int(args.n_seq)
    results_root = Path(args.results_dir)
    cache_path = Path(CACHE_TEMPLATE.format(fold=fold))
    if not cache_path.exists():
        raise FileNotFoundError(f"CARE-PD cache not found for fold{fold}: {cache_path}")
    ckpt_path = CAREPD_ROOT / CKPT_TEMPLATE.format(fold=fold)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"MotionBERT checkpoint missing: {ckpt_path}. "
            f"Extract from CARE-PD/experiment_outs/Hypertune/.../fold{fold}/latest_epoch.pth.tr."
        )

    t_total = time.time()
    device = torch.device(DEVICE)

    # ------------------------------------------------------------- data
    log.info("[fold%d] loading BMCLab cache from %s", fold, cache_path)
    d = np.load(cache_path, allow_pickle=True)
    x_val = np.transpose(d["x1_val"], (0, 2, 3, 1)).astype(np.float32)
    y_val = np.asarray(d["meta_updrs_gait_val"], dtype=np.int64)
    keep = y_val >= 0
    x_val = x_val[keep][:N_SEQ]

    # Eval caches are val-only — pull the donor/mean pool from the legacy
    # fold1 cache (full BMCLab train pool, fold-agnostic in distribution).
    if d["x1_train"].shape[0] == 0:
        log.info("[fold%d] eval cache has no train pool; loading legacy "
                 "BMCLab_h36m_80_fold1 cache for marginal donors", fold)
        d_train = np.load(TRAIN_POOL_CACHE, allow_pickle=True)
        x_train = np.transpose(d_train["x1_train"], (0, 2, 3, 1)).astype(np.float32)
    else:
        x_train = np.transpose(d["x1_train"], (0, 2, 3, 1)).astype(np.float32)
    N, J, F, T = x_val.shape
    log.info("[fold%d] data: N=%d, J=%d, F=%d, T=%d, val_class_counts=%s, "
             "train_pool_N=%d",
             fold, N, J, F, T,
             np.bincount(y_val[y_val >= 0], minlength=3).tolist(),
             x_train.shape[0])

    if N < N_SEQ:
        log.warning("[fold%d] only %d valid sequences available (asked for %d)",
                    fold, N, N_SEQ)

    # ------------------------------------------------------------- coalitions
    z_bin, frame_mask = build_coalition_masks(K, T)
    n_coal = 1 << K
    log.info("[fold%d] coalitions: %d (K=%d, T_window=%d)", fold, n_coal, K, T // K)

    # ------------------------------------------------------------- classifier
    log.info("[fold%d] loading MotionBERT (fold%d ckpt)...", fold, fold)
    t_load = time.time()
    from motionbench.classifiers.ported_care_pd.motionbert import MotionBERTClassifier
    clf = MotionBERTClassifier(
        n_classes=3,
        checkpoint_path=str(ckpt_path),
    ).to(device)
    clf.eval()
    log.info("[fold%d] MotionBERT loaded in %.1fs", fold, time.time() - t_load)

    with torch.no_grad():
        x_val_t = torch.from_numpy(x_val).to(device)
        logits = clf(x_val_t)
    targets = logits.argmax(dim=-1).cpu().numpy()
    log.info("[fold%d] target distribution: %s", fold, np.bincount(targets, minlength=3).tolist())

    # ------------------------------------------------------------- imputer setup
    mean_jf = torch.from_numpy(x_train.mean(axis=(0, 3))).float()  # (J, F)

    rng = np.random.default_rng(42 + fold)
    donor_idx = rng.integers(0, x_train.shape[0], size=N)
    donors = torch.from_numpy(x_train[donor_idx]).float()           # (N, J, F, T)

    vaeac_imputer = None
    flow_imputer = None

    def get_vaeac():
        nonlocal vaeac_imputer
        if vaeac_imputer is None:
            log.info("[fold%d] loading VAEAC imputer (fold1 ckpt — reused)...", fold)
            t = time.time()
            from motionbench.imputers.carepd_imputer import _load_vaeac, _CARE_PD_ROOT
            ckpt_dir = _CARE_PD_ROOT / "experiment_outs/vaeac_real/bmclab_fold1_real_gait_bm"
            cfg_path = _CARE_PD_ROOT / "configs/vaeac/bmclab_fold1_real_gait_bm.json"
            vaeac_imputer = _load_vaeac(ckpt_dir, cfg_path, device)
            log.info("[fold%d] VAEAC loaded in %.1fs", fold, time.time() - t)
        return vaeac_imputer

    def get_flow():
        nonlocal flow_imputer
        if flow_imputer is None:
            log.info("[fold%d] loading Flow imputer (fold1 ckpt — reused)...", fold)
            t = time.time()
            from motionbench.imputers.carepd_imputer import _load_flow, _CARE_PD_ROOT
            ckpt_dir = _CARE_PD_ROOT / "experiment_outs/flow_matching/bmclab_h36m3d_fold1"
            cfg_path = _CARE_PD_ROOT / "configs/flow_matching/bmclab_h36m3d_fold1.json"
            cfg = json.loads(cfg_path.read_text())
            cfg["num_steps"] = 20
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(cfg, f)
                tmp_cfg = Path(f.name)
            try:
                flow_imputer = _load_flow(ckpt_dir, tmp_cfg, device)
            finally:
                tmp_cfg.unlink()
            log.info("[fold%d] Flow loaded in %.1fs (num_steps=20)", fold, time.time() - t)
        return flow_imputer

    # ------------------------------------------------------------- methods
    all_methods = [
        "kernelshap_zero", "kernelshap_mean", "kernelshap_marginal",
        "kernelshap_vaeac", "kernelshap_flow",
    ]
    methods = args.methods if args.methods else all_methods
    for m in methods:
        if m not in all_methods:
            raise ValueError(f"unknown method {m!r}; valid: {all_methods}")
    fold_dir = results_root / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for method in methods:
        method_dir = fold_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        result_path = method_dir / "result.json"
        log.info("=" * 60)
        log.info("[fold%d] Method: %s", fold, method)
        t_method = time.time()

        phis = np.zeros((N, K), dtype=np.float32)
        v_all = np.zeros((N, n_coal), dtype=np.float32)

        if method == "kernelshap_vaeac":
            imp = get_vaeac()
        elif method == "kernelshap_flow":
            imp = get_flow()
        else:
            imp = None

        for i in range(N):
            x_i = torch.from_numpy(x_val[i])
            target_i = int(targets[i])

            if method == "kernelshap_zero":
                comps = build_completions_offmanifold(x_i, frame_mask, "zero")
            elif method == "kernelshap_mean":
                comps = build_completions_offmanifold(x_i, frame_mask, "mean", mean_jf)
            elif method == "kernelshap_marginal":
                comps = build_completions_offmanifold(
                    x_i, frame_mask, "marginal", donors[i],
                )
            elif method in ("kernelshap_vaeac", "kernelshap_flow"):
                with torch.no_grad():
                    out = imp.sample_completions_batched(
                        x=x_i.unsqueeze(0).to(device),
                        mask=torch.ones(1, T, dtype=torch.bool, device=device),
                        coalition_masks=frame_mask.to(device),
                        n_samples=1,
                    )                                            # (16, 1, J, F, T)
                comps = out.squeeze(1).cpu().contiguous()
                obs = frame_mask.view(n_coal, 1, 1, T).expand(n_coal, J, F, T)
                comps = torch.where(obs, x_i.unsqueeze(0).expand_as(comps), comps).contiguous()
            else:
                raise ValueError(f"unknown method {method}")

            with torch.no_grad():
                logits_b = clf(comps.to(device))
            v_b = torch.softmax(logits_b, dim=-1)[:, target_i]
            v_all[i] = v_b.cpu().numpy()

            phi = kernel_shap_exact(z_bin, v_b.cpu(), K)
            phis[i] = phi.numpy()

            if (i + 1) % 25 == 0 or i == N - 1:
                log.info("  [fold%d] %s seq %d/%d (avg %.2fs/seq)",
                         fold, method, i + 1, N,
                         (time.time() - t_method) / (i + 1))

        # ------------------------------------------------------------- metrics
        faiths = []
        aopcs = []
        for i in range(N):
            v_i = torch.from_numpy(v_all[i])
            phi_i = torch.from_numpy(phis[i])
            f_val = faithfulness_correlation(z_bin, v_i, phi_i)
            a_val = player_aopc(v_i, z_bin, phi_i, K)
            faiths.append(f_val)
            aopcs.append(a_val)

        faiths_arr = np.asarray(faiths, dtype=np.float64)
        aopcs_arr = np.asarray(aopcs, dtype=np.float64)
        finite = np.isfinite(faiths_arr)
        n_finite = int(finite.sum())

        faith_mean = float(np.nanmean(faiths_arr))
        faith_std = float(np.nanstd(faiths_arr, ddof=1)) if n_finite > 1 else float("nan")
        aopc_mean = float(np.mean(aopcs_arr))
        aopc_std = float(np.std(aopcs_arr, ddof=1)) if N > 1 else float("nan")

        np.savez_compressed(
            method_dir / "attributions.npz",
            phi=phis,
            x=x_val,
            target=targets,
            v=v_all,
        )
        result = {
            "dataset": "care_pd_bmclab_cache",
            "classifier": "motionbert",
            "fold": int(fold),
            "method": method,
            "n_sequences": int(N),
            "n_finite_faithfulness": n_finite,
            "faithfulness_correlation": faith_mean,
            "faithfulness_correlation_std": faith_std,
            "player_aopc": aopc_mean,
            "player_aopc_std": aopc_std,
            "phi_mean": phis.mean(axis=0).tolist(),
            "phi_std": phis.std(axis=0).tolist(),
            # per-sequence arrays for downstream bootstrap CIs
            "faithfulness_per_seq": faiths_arr.tolist(),
            "player_aopc_per_seq": aopcs_arr.tolist(),
            "targets_per_seq": targets.tolist(),
        }
        result_path.write_text(json.dumps(result, indent=2))
        summary_rows.append({
            "method": method,
            "fold": int(fold),
            "n": int(N),
            "faith_mean": faith_mean,
            "faith_std": faith_std,
            "aopc_mean": aopc_mean,
            "aopc_std": aopc_std,
        })
        log.info("  [fold%d] done %s in %.1fs — faith=%.3f±%.3f, aopc=%.3f±%.3f (n_finite=%d)",
                 fold, method, time.time() - t_method,
                 faith_mean, faith_std, aopc_mean, aopc_std, n_finite)

    # Per-fold summary
    summary_path = fold_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2))
    log.info("=" * 60)
    log.info("[fold%d] ALL DONE in %.1fs.  Results at %s",
             fold, time.time() - t_total, fold_dir)
    for r in summary_rows:
        log.info("  %-25s faith=%+.3f±%.3f  aopc=%+.3f±%.3f  (n=%d)",
                 r["method"], r["faith_mean"], r["faith_std"],
                 r["aopc_mean"], r["aopc_std"], r["n"])


if __name__ == "__main__":
    main()
