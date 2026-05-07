"""scripts/run_care_pd_multiclf.py — multi-classifier CARE-PD SHAP sweep.

Extends ``run_care_pd_extended.py`` to support multiple classifiers:
- ``motionbert``  : MotionBERT (DSTformer), T=80, 3-D H36M + crop_scale
- ``potr``        : POTR (GCN + Transformer), T=80, 3-D H36M + root-center + zscore
- ``motionagformer`` : MotionAGFormer (Attention + Graph), T=81 (zero-pad from 80),
                       3-D H36M + crop_scale

Usage::

    conda activate motionbench-xai
    # Run one classifier × one fold (parallelise across GPUs):
    CUDA_VISIBLE_DEVICES=0 python scripts/run_care_pd_multiclf.py --classifier motionbert  --fold 1
    CUDA_VISIBLE_DEVICES=1 python scripts/run_care_pd_multiclf.py --classifier motionbert  --fold 2
    CUDA_VISIBLE_DEVICES=2 python scripts/run_care_pd_multiclf.py --classifier motionbert  --fold 3
    CUDA_VISIBLE_DEVICES=3 python scripts/run_care_pd_multiclf.py --classifier potr        --fold 1
    CUDA_VISIBLE_DEVICES=4 python scripts/run_care_pd_multiclf.py --classifier potr        --fold 2
    CUDA_VISIBLE_DEVICES=5 python scripts/run_care_pd_multiclf.py --classifier potr        --fold 3
    CUDA_VISIBLE_DEVICES=6 python scripts/run_care_pd_multiclf.py --classifier motionagformer --fold 1
    CUDA_VISIBLE_DEVICES=7 python scripts/run_care_pd_multiclf.py --classifier motionagformer --fold 2

Results are written to::

    results/care_pd_multiclf/{classifier}/fold{fold}/{method}/result.json
    results/care_pd_multiclf/{classifier}/fold{fold}/summary.json

Run ``scripts/compute_real_cis_multiclf.py`` afterwards to pool and bootstrap.
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
RESULTS_ROOT = REPO_ROOT / "results" / "care_pd_multiclf"
CARE_PD_ROOT = Path(os.environ.get("CARE_PD_ROOT", REPO_ROOT.parent / "CARE-PD"))

CACHE_TEMPLATE = str(
    CARE_PD_ROOT / "cache" / "flow_matching"
    / "BMCLab_h36m_80_classifier23fold_fold{fold}_eval" / "cache.npz"
)
TRAIN_POOL_CACHE = CARE_PD_ROOT / "cache" / "flow_matching" / "BMCLab_h36m_80_fold1" / "cache.npz"

# Per-classifier checkpoint templates (relative to REPO_ROOT)
CKPT_TEMPLATES = {
    "motionbert":      "motionbench/classifiers/checkpoints/real/carepd_bmclab_fold{fold}_motionbert.pt",
    "potr":            "motionbench/classifiers/checkpoints/real/carepd_bmclab_fold{fold}_potr.pt",
    "motionagformer":  "motionbench/classifiers/checkpoints/real/carepd_bmclab_fold{fold}_motionagformer.pt",
}

K = 4       # temporal windows
DEVICE = "cuda:0"


# ---------------------------------------------------------------------- #
# Coalition masks                                                         #
# ---------------------------------------------------------------------- #

def build_coalition_masks(K: int, T: int) -> tuple[Tensor, Tensor]:
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
# Off-manifold completions                                               #
# ---------------------------------------------------------------------- #

def build_completions_offmanifold(
    x: Tensor, frame_mask_2k: Tensor, kind: str, mean_per_jf: Tensor | None = None,
) -> Tensor:
    J, F, T = x.shape
    n_coal = frame_mask_2k.shape[0]
    obs = frame_mask_2k.view(n_coal, 1, 1, T).expand(n_coal, J, F, T)
    if kind == "zero":
        fill = torch.zeros_like(x)
    elif kind == "mean":
        fill = mean_per_jf.view(J, F, 1).expand(J, F, T)
    elif kind == "marginal":
        fill = mean_per_jf
    else:
        raise ValueError(f"unknown kind {kind!r}")
    x_b = x.view(1, J, F, T).expand(n_coal, J, F, T)
    fill_b = fill.view(1, J, F, T).expand(n_coal, J, F, T)
    return torch.where(obs, x_b, fill_b).contiguous()


# ---------------------------------------------------------------------- #
# KernelSHAP (exact, K small)                                             #
# ---------------------------------------------------------------------- #

def shapley_kernel(K: int, s: int) -> float:
    if s == 0 or s == K:
        return 1e6
    from math import comb
    return (K - 1) / (comb(K, s) * s * (K - s))


def kernel_shap_exact(z_bin: Tensor, v_vals: Tensor, K: int) -> Tensor:
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
    return torch.from_numpy(sol[1:]).float()


# ---------------------------------------------------------------------- #
# Metrics                                                                 #
# ---------------------------------------------------------------------- #

def faithfulness_correlation(z_bin: Tensor, v_vals: Tensor, phi: Tensor) -> float:
    z = z_bin.float()
    not_z = 1.0 - z
    sum_phi_absent = not_z @ phi
    delta = v_vals[-1] - v_vals
    a, b = sum_phi_absent.numpy(), delta.float().numpy()
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def player_aopc(v_vals: Tensor, z_bin: Tensor, phi: Tensor, K: int) -> float:
    order = torch.argsort(phi.abs(), descending=True).tolist()
    v_full = v_vals[-1].item()
    drops: list[float] = []
    cur_z = z_bin[-1].clone()
    for k in order:
        cur_z[k] = False
        idx = int((cur_z.int() * (1 << torch.arange(K))).sum().item())
        drops.append(v_full - v_vals[idx].item())
    return float(np.mean(drops)) if drops else 0.0


# ---------------------------------------------------------------------- #
# Per-classifier loading helpers                                          #
# ---------------------------------------------------------------------- #

def load_classifier(clf_name: str, fold: int, device: torch.device,
                    stats_mean=None, stats_std=None):
    ckpt_rel = CKPT_TEMPLATES[clf_name].format(fold=fold)
    ckpt_path = REPO_ROOT / ckpt_rel
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint missing: {ckpt_path}")

    if clf_name == "motionbert":
        from motionbench.classifiers.ported_care_pd.motionbert import MotionBERTClassifier
        clf = MotionBERTClassifier(n_classes=3, checkpoint_path=str(ckpt_path))
    elif clf_name == "potr":
        from motionbench.classifiers.ported_care_pd.potr import POTRClassifier
        clf = POTRClassifier(
            n_classes=3,
            checkpoint_path=str(ckpt_path),
            source_seq_length=80,
            stats_mean=stats_mean,
            stats_std=stats_std,
        )
    elif clf_name == "motionagformer":
        from motionbench.classifiers.ported_care_pd.motionagformer import MotionAGFormerClassifier
        clf = MotionAGFormerClassifier(
            n_classes=3,
            checkpoint_path=str(ckpt_path),
            n_frames=81,
        )
    else:
        raise ValueError(f"Unknown classifier: {clf_name!r}")

    clf = clf.to(device)
    clf.eval()
    log.info("[%s fold%d] classifier loaded from %s", clf_name, fold, ckpt_path.name)
    return clf


def clf_forward(clf_name: str, clf, x_batch: Tensor, device: torch.device) -> Tensor:
    """Run the classifier on x_batch, handling per-classifier T requirements."""
    x_batch = x_batch.to(device)
    if clf_name == "motionagformer":
        # Pad T=80 → T=81 with a zero frame (treated as padding in crop_scale_and_conf)
        pad = torch.zeros(*x_batch.shape[:-1], 1, device=device, dtype=x_batch.dtype)
        x_batch = torch.cat([x_batch, pad], dim=-1)   # (B, J, 3, 81)
    with torch.no_grad():
        return clf(x_batch)


# ---------------------------------------------------------------------- #
# Main                                                                    #
# ---------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--classifier", type=str, default="motionbert",
                    choices=list(CKPT_TEMPLATES.keys()),
                    help="Classifier to evaluate.")
    ap.add_argument("--fold", type=int, default=1, choices=list(range(1, 24)),
                    help="CARE-PD classifier fold (1..23).")
    ap.add_argument("--n_seq", type=int, default=200)
    ap.add_argument("--methods", type=str, nargs="+", default=None)
    ap.add_argument("--results_dir", type=str, default=str(RESULTS_ROOT))
    ap.add_argument("--device", type=str, default=DEVICE)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    clf_name = args.classifier
    fold = args.fold
    N_SEQ = int(args.n_seq)
    results_root = Path(args.results_dir)
    device = torch.device(args.device)

    cache_path = Path(CACHE_TEMPLATE.format(fold=fold))
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")

    t_total = time.time()

    # ---------------------------------------------------------- data
    log.info("[%s fold%d] loading cache from %s", clf_name, fold, cache_path)
    d = np.load(cache_path, allow_pickle=True)
    stats_mean = np.asarray(d["stats_mean"], dtype=np.float32)  # (17, 3)
    stats_std  = np.asarray(d["stats_std"],  dtype=np.float32)  # (17, 3)

    x_val = np.transpose(d["x1_val"], (0, 2, 3, 1)).astype(np.float32)
    y_val = np.asarray(d["meta_updrs_gait_val"], dtype=np.int64)
    keep = y_val >= 0
    x_val = x_val[keep][:N_SEQ]

    if d["x1_train"].shape[0] == 0:
        d_train = np.load(TRAIN_POOL_CACHE, allow_pickle=True)
        x_train = np.transpose(d_train["x1_train"], (0, 2, 3, 1)).astype(np.float32)
    else:
        x_train = np.transpose(d["x1_train"], (0, 2, 3, 1)).astype(np.float32)

    N, J, F, T = x_val.shape
    log.info("[%s fold%d] N=%d J=%d F=%d T=%d train_pool=%d",
             clf_name, fold, N, J, F, T, x_train.shape[0])

    # ---------------------------------------------------------- coalitions
    z_bin, frame_mask = build_coalition_masks(K, T)
    n_coal = 1 << K

    # ---------------------------------------------------------- classifier
    t_load = time.time()
    clf = load_classifier(clf_name, fold, device, stats_mean=stats_mean, stats_std=stats_std)
    log.info("[%s fold%d] classifier loaded in %.1fs", clf_name, fold, time.time() - t_load)

    # Get predicted targets
    with torch.no_grad():
        logits_all = clf_forward(clf_name, clf, torch.from_numpy(x_val), device)
    targets = logits_all.cpu().argmax(dim=-1).numpy()
    log.info("[%s fold%d] target distribution: %s",
             clf_name, fold, np.bincount(targets, minlength=3).tolist())

    # ---------------------------------------------------------- imputer setup
    mean_jf = torch.from_numpy(x_train.mean(axis=(0, 3))).float()  # (J, F)
    rng = np.random.default_rng(42 + fold)
    donor_idx = rng.integers(0, x_train.shape[0], size=N)
    donors = torch.from_numpy(x_train[donor_idx]).float()

    vaeac_imputer = None
    flow_imputer = None

    def get_vaeac():
        nonlocal vaeac_imputer
        if vaeac_imputer is None:
            from motionbench.imputers.carepd_imputer import _load_vaeac, _CARE_PD_ROOT
            ckpt_dir = _CARE_PD_ROOT / "experiment_outs/vaeac_real/bmclab_fold1_real_gait_bm"
            cfg_path = _CARE_PD_ROOT / "configs/vaeac/bmclab_fold1_real_gait_bm.json"
            vaeac_imputer = _load_vaeac(ckpt_dir, cfg_path, device)
        return vaeac_imputer

    def get_flow():
        nonlocal flow_imputer
        if flow_imputer is None:
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
        return flow_imputer

    # ---------------------------------------------------------- methods
    all_methods = [
        "kernelshap_zero", "kernelshap_mean", "kernelshap_marginal",
        "kernelshap_vaeac", "kernelshap_flow",
    ]
    methods = args.methods if args.methods else all_methods
    fold_dir = results_root / clf_name / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for method in methods:
        method_dir = fold_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        result_path = method_dir / "result.json"
        if result_path.exists():
            log.info("[%s fold%d] %s already done, skipping.", clf_name, fold, method)
            with open(result_path) as f:
                summary_rows.append(json.load(f))
            continue

        log.info("=" * 60)
        log.info("[%s fold%d] Method: %s", clf_name, fold, method)
        t_method = time.time()

        phis = np.zeros((N, K), dtype=np.float32)
        v_all = np.zeros((N, n_coal), dtype=np.float32)

        imp = None
        if method == "kernelshap_vaeac":
            imp = get_vaeac()
            log.info("[%s fold%d] VAEAC loaded", clf_name, fold)
        elif method == "kernelshap_flow":
            imp = get_flow()
            log.info("[%s fold%d] Flow loaded", clf_name, fold)

        for i in range(N):
            x_i = torch.from_numpy(x_val[i])
            target_i = int(targets[i])

            if method == "kernelshap_zero":
                comps = build_completions_offmanifold(x_i, frame_mask, "zero")
            elif method == "kernelshap_mean":
                comps = build_completions_offmanifold(x_i, frame_mask, "mean", mean_jf)
            elif method == "kernelshap_marginal":
                comps = build_completions_offmanifold(x_i, frame_mask, "marginal", donors[i])
            elif method in ("kernelshap_vaeac", "kernelshap_flow"):
                with torch.no_grad():
                    out = imp.sample_completions_batched(
                        x=x_i.unsqueeze(0).to(device),
                        mask=torch.ones(1, T, dtype=torch.bool, device=device),
                        coalition_masks=frame_mask.to(device),
                        n_samples=1,
                    )  # (16, 1, J, F, T)
                comps = out.squeeze(1).cpu().contiguous()
                obs = frame_mask.view(n_coal, 1, 1, T).expand(n_coal, J, F, T)
                comps = torch.where(obs, x_i.unsqueeze(0).expand_as(comps), comps)
            else:
                raise ValueError(f"unknown method {method}")

            # Feed completions through classifier
            logits_b = clf_forward(clf_name, clf, comps, device)
            v_b = torch.softmax(logits_b, dim=-1)[:, target_i].cpu()
            v_all[i] = v_b.numpy()

            phi = kernel_shap_exact(z_bin, v_b, K)
            phis[i] = phi.numpy()

            if (i + 1) % 25 == 0 or i == N - 1:
                log.info("  [%s fold%d] %s  %d/%d  (%.2fs/seq)",
                         clf_name, fold, method, i + 1, N,
                         (time.time() - t_method) / (i + 1))

        # -------------------------------------------------------- metrics
        faiths, aopcs = [], []
        for i in range(N):
            v_i  = torch.from_numpy(v_all[i])
            phi_i = torch.from_numpy(phis[i])
            faiths.append(faithfulness_correlation(z_bin, v_i, phi_i))
            aopcs.append(player_aopc(v_i, z_bin, phi_i, K))

        faiths_arr = np.asarray(faiths, dtype=np.float64)
        aopcs_arr  = np.asarray(aopcs,  dtype=np.float64)
        n_finite   = int(np.isfinite(faiths_arr).sum())

        faith_mean = float(np.nanmean(faiths_arr))
        faith_std  = float(np.nanstd(faiths_arr, ddof=1)) if n_finite > 1 else float("nan")
        aopc_mean  = float(np.mean(aopcs_arr))
        aopc_std   = float(np.std(aopcs_arr, ddof=1))  if N > 1 else float("nan")

        np.savez_compressed(
            method_dir / "attributions.npz",
            phi=phis, x=x_val, target=targets, v=v_all,
        )
        result = {
            "dataset": "care_pd_bmclab_cache",
            "classifier": clf_name,
            "fold": int(fold),
            "method": method,
            "n_sequences": int(N),
            "n_finite_faithfulness": n_finite,
            "faithfulness_correlation": faith_mean,
            "faithfulness_correlation_std": faith_std,
            "player_aopc": aopc_mean,
            "player_aopc_std": aopc_std,
            "phi_mean": phis.mean(axis=0).tolist(),
            "phi_std":  phis.std(axis=0).tolist(),
            "faithfulness_per_seq": faiths_arr.tolist(),
            "player_aopc_per_seq":  aopcs_arr.tolist(),
            "targets_per_seq": targets.tolist(),
        }
        result_path.write_text(json.dumps(result, indent=2))
        summary_rows.append(result)
        log.info("  [%s fold%d] %s done %.1fs — faith=%+.3f aopc=%+.3f (n_fin=%d)",
                 clf_name, fold, method, time.time() - t_method,
                 faith_mean, aopc_mean, n_finite)

    (fold_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2))
    log.info("=" * 60)
    log.info("[%s fold%d] ALL DONE in %.1fs", clf_name, fold, time.time() - t_total)


if __name__ == "__main__":
    main()
