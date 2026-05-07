"""scripts/recompute_fixed_deletion_metrics.py

Recompute PlayerAOPC and faithfulness correlation under a SINGLE FIXED
reference deletion imputer (the marginal-donor protocol from
``kernelshap_marginal``) for the PTB-XL and CARE-PD KernelSHAP sweeps.

Motivation (reviewer W2)
------------------------
The existing PTB-XL/CARE-PD faithfulness and PlayerAOPC scores use each
method's OWN imputer for both attribution computation AND for the
deletion/coalition value function v(S).  This is potentially circular for
KS-Marginal/VAEAC/Flow because the same marginal/learned-donor distribution
is reused at evaluation time.  This script recomputes the two metrics with
a FIXED reference deletion imputer (marginal donors) so the comparison
across methods is apples-to-apples.

The attributions ``phi`` are loaded as-is from each method's saved
``attributions.npz``; only the value function ``v_ref(S)`` used for
metric evaluation is replaced.

Outputs
-------
``results/{ptbxl|care_pd_extended}/fold{F}/kernelshap_*/result_fixed_deletion.json``

Existing ``result.json`` files are NEVER modified.  If a
``result_fixed_deletion.json`` already exists for a (fold, method) cell,
that cell is skipped (the script is restartable).

Usage
-----
    conda activate motionbench-xai
    CUDA_VISIBLE_DEVICES=0 python scripts/recompute_fixed_deletion_metrics.py \\
        --dataset ptbxl --fold 1 --device cuda:0 \\
        --ptbxl_data_path "$PTBXL_DATA_ROOT"

    CUDA_VISIBLE_DEVICES=0 python scripts/recompute_fixed_deletion_metrics.py \\
        --dataset care_pd --fold 1 --device cuda:0

Environment variables:
    CARE_PD_ROOT: root of the CARE-PD codebase (cache lookup).
    PTBXL_DATA_ROOT: root of the PTB-XL raw-data directory.
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

import numpy as np
import torch
from torch import Tensor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

CARE_PD_ROOT = Path(os.environ.get("CARE_PD_ROOT", REPO_ROOT.parent / "CARE-PD"))

# Import shared utilities from the existing pipeline
from run_care_pd_multiclf import (   # noqa: E402
    build_coalition_masks,
    faithfulness_correlation,
    player_aopc,
)

K = 4
N_SEQ_DEFAULT = 200

ALL_METHODS = [
    "kernelshap_zero",
    "kernelshap_mean",
    "kernelshap_marginal",
    "kernelshap_vaeac",
    "kernelshap_flow",
]

# ---------------------------------------------------------------- data loaders


def load_ptbxl_train_pool(fold: int, ptbxl_data_path: str) -> tuple[np.ndarray, dict]:
    """Mirror the train-pool construction in scripts/run_ptbxl_shap.py."""
    ckpt_dir = REPO_ROOT / "motionbench" / "classifiers" / "checkpoints" / "real"
    stats_path = ckpt_dir / f"ptbxl_fold{fold}_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError(f"PTB-XL stats not found: {stats_path}")
    stats = np.load(stats_path)
    train_stats = (stats["mean"].astype(np.float32),
                   stats["std"].astype(np.float32))
    train_fold_ids = stats["train_folds"].tolist()

    from motionbench.data.real.ptbxl import PTBXLDataset, _FOLD_SPLITS
    _FOLD_SPLITS["_train_folds"] = (train_fold_ids,)
    train_ds = PTBXLDataset(
        data_path=ptbxl_data_path,
        split="_train_folds",
        normalize=True,
        max_sequences=2000,
        train_stats=train_stats,
    )
    del _FOLD_SPLITS["_train_folds"]
    x_train = np.stack([s[0] for s in train_ds._samples], axis=0)
    x_train = x_train.transpose(0, 2, 1)[:, :, np.newaxis, :]  # (N_tr, J=12, F=1, T=1000)
    return x_train.astype(np.float32), {
        "train_fold_ids": train_fold_ids,
        "stats_path": str(stats_path),
    }


def load_carepd_train_pool() -> np.ndarray:
    """Mirror the train-pool construction in run_care_pd_extended.py."""
    pool = CARE_PD_ROOT / "cache" / "flow_matching" / "BMCLab_h36m_80_fold1" / "cache.npz"
    if not pool.exists():
        raise FileNotFoundError(f"CARE-PD train pool not found: {pool}")
    d = np.load(pool, allow_pickle=True)
    x_train = np.transpose(d["x1_train"], (0, 2, 3, 1)).astype(np.float32)
    return x_train  # (N_tr, J=17, F=3, T=80)


def load_ptbxl_classifier(fold: int, device: torch.device):
    ckpt_path = (REPO_ROOT / "motionbench" / "classifiers"
                 / "checkpoints" / "real" / f"ptbxl_fold{fold}.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"PTB-XL classifier ckpt missing: {ckpt_path}")
    from motionbench.classifiers.ported_ptbxl.resnet1d import ECGResNet1dClassifier
    clf = ECGResNet1dClassifier(n_classes=2,
                                checkpoint_path=str(ckpt_path)).to(device)
    clf.eval()
    return clf, str(ckpt_path)


def load_carepd_classifier(fold: int, device: torch.device):
    ckpt_path = (REPO_ROOT / "motionbench" / "classifiers"
                 / "checkpoints" / "real"
                 / f"carepd_bmclab_fold{fold}_motionbert.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"CARE-PD MotionBERT ckpt missing: {ckpt_path}")
    from motionbench.classifiers.ported_care_pd.motionbert import MotionBERTClassifier
    clf = MotionBERTClassifier(n_classes=3,
                               checkpoint_path=str(ckpt_path)).to(device)
    clf.eval()
    return clf, str(ckpt_path)


# ---------------------------------------------------------------- core
def build_marginal_completions(
    x_i: Tensor, donor_i: Tensor, frame_mask: Tensor,
) -> Tensor:
    """Replace masked windows in x_i with values from the same donor.

    Args:
        x_i: (J, F, T) — single test sequence.
        donor_i: (J, F, T) — single donor sequence (same shape).
        frame_mask: (n_coal, T) bool — True = observed at that frame.
    Returns:
        (n_coal, J, F, T) completions.
    """
    J, F, T = x_i.shape
    n_coal = frame_mask.shape[0]
    obs = frame_mask.view(n_coal, 1, 1, T).expand(n_coal, J, F, T)
    x_b = x_i.view(1, J, F, T).expand(n_coal, J, F, T)
    fill_b = donor_i.view(1, J, F, T).expand(n_coal, J, F, T)
    return torch.where(obs, x_b, fill_b).contiguous()


def recompute_one_method(
    method: str,
    fold: int,
    method_dir: Path,
    classifier_forward,            # callable(x_batch_dev) -> logits
    donors: Tensor,                # (N, J, F, T) on CPU
    frame_mask: Tensor,            # (n_coal, T) bool on CPU
    z_bin: Tensor,                 # (n_coal, K) bool on CPU
    device: torch.device,
    log_prefix: str,
) -> dict | None:
    """Return result dict (or None if skipped)."""
    out_path = method_dir / "result_fixed_deletion.json"
    if out_path.exists():
        log.info("%s %s — already done, skipping.", log_prefix, method)
        with open(out_path) as fh:
            return json.load(fh)

    attr_path = method_dir / "attributions.npz"
    if not attr_path.exists():
        log.warning("%s %s — attributions.npz missing, skipping.",
                    log_prefix, method)
        return None

    d = np.load(attr_path)
    phi_all = d["phi"].astype(np.float32)             # (N, K)
    x_all = d["x"].astype(np.float32)                 # (N, J, F, T)
    target_all = d["target"].astype(np.int64)         # (N,)
    N = phi_all.shape[0]
    assert donors.shape[0] == N, (donors.shape, N)

    n_coal = z_bin.shape[0]
    v_ref = np.zeros((N, n_coal), dtype=np.float32)

    t_method = time.time()
    for i in range(N):
        x_i = torch.from_numpy(x_all[i])
        donor_i = donors[i]
        comps = build_marginal_completions(x_i, donor_i, frame_mask)
        logits = classifier_forward(comps.to(device))
        target_i = int(target_all[i])
        v_b = torch.softmax(logits, dim=-1)[:, target_i].detach().cpu()
        v_ref[i] = v_b.numpy()

        if (i + 1) % 50 == 0 or i == N - 1:
            log.info("  %s %s  %d/%d  (%.2fs/seq)",
                     log_prefix, method, i + 1, N,
                     (time.time() - t_method) / (i + 1))

    # ---- metrics
    faiths, aopcs = [], []
    for i in range(N):
        v_i = torch.from_numpy(v_ref[i])
        phi_i = torch.from_numpy(phi_all[i])
        faiths.append(faithfulness_correlation(z_bin, v_i, phi_i))
        aopcs.append(player_aopc(v_i, z_bin, phi_i, K))

    faiths_arr = np.asarray(faiths, dtype=np.float64)
    aopcs_arr = np.asarray(aopcs, dtype=np.float64)
    n_finite = int(np.isfinite(faiths_arr).sum())

    faith_mean = float(np.nanmean(faiths_arr))
    faith_std = float(np.nanstd(faiths_arr, ddof=1)) if n_finite > 1 else float("nan")
    aopc_mean = float(np.mean(aopcs_arr))
    aopc_std = float(np.std(aopcs_arr, ddof=1)) if N > 1 else float("nan")

    result = {
        "fold": int(fold),
        "method": method,
        "n_sequences": int(N),
        "n_finite_faithfulness": n_finite,
        "deletion_protocol": "fixed_marginal_donor_per_sequence",
        "faithfulness_correlation_fixed": faith_mean,
        "faithfulness_correlation_fixed_std": faith_std,
        "player_aopc_fixed": aopc_mean,
        "player_aopc_fixed_std": aopc_std,
        "faithfulness_per_seq_fixed": faiths_arr.tolist(),
        "player_aopc_per_seq_fixed": aopcs_arr.tolist(),
    }
    out_path.write_text(json.dumps(result, indent=2))
    log.info("  %s %s done %.1fs — faith_fixed=%+.3f aopc_fixed=%+.3f (n_fin=%d)",
             log_prefix, method, time.time() - t_method,
             faith_mean, aopc_mean, n_finite)
    return result


# ---------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", choices=["ptbxl", "care_pd"], required=True)
    ap.add_argument("--fold", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--ptbxl_data_path", type=str,
                    default=os.environ.get("PTBXL_DATA_ROOT",
                                           str(REPO_ROOT / "data" / "ptb-xl")))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    fold = args.fold
    methods = args.methods if args.methods else ALL_METHODS

    t_total = time.time()

    # ---- coalitions (T differs by dataset)
    if args.dataset == "ptbxl":
        T = 1000
        results_root = REPO_ROOT / "results" / "ptbxl"
        log.info("[ptbxl fold%d] loading train pool & classifier...", fold)
        x_train, _meta = load_ptbxl_train_pool(fold, args.ptbxl_data_path)
        clf, ckpt_str = load_ptbxl_classifier(fold, device)

        def clf_fwd(x_batch_dev: Tensor) -> Tensor:
            with torch.no_grad():
                return clf(x_batch_dev)

        log_prefix = f"[ptbxl fold{fold}]"
    else:  # care_pd
        T = 80
        results_root = REPO_ROOT / "results" / "care_pd_extended"
        log.info("[care_pd fold%d] loading train pool & classifier...", fold)
        x_train = load_carepd_train_pool()
        clf, ckpt_str = load_carepd_classifier(fold, device)

        def clf_fwd(x_batch_dev: Tensor) -> Tensor:
            with torch.no_grad():
                return clf(x_batch_dev)

        log_prefix = f"[care_pd fold{fold}]"

    log.info("%s train pool: %d records  ckpt=%s",
             log_prefix, x_train.shape[0], Path(ckpt_str).name)

    z_bin, frame_mask = build_coalition_masks(K, T)
    n_coal = 1 << K

    # ---- donors: same RNG as the original kernelshap_marginal pipeline.
    rng = np.random.default_rng(42 + fold)
    donor_idx = rng.integers(0, x_train.shape[0], size=N_SEQ_DEFAULT)
    donors_np = x_train[donor_idx]                      # (N, J, F, T)
    donors = torch.from_numpy(donors_np).float()        # CPU
    log.info("%s donors drawn (seed=%d, N=%d)",
             log_prefix, 42 + fold, donors_np.shape[0])

    fold_dir = results_root / f"fold{fold}"
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold dir missing: {fold_dir}")

    summary = []
    for method in methods:
        method_dir = fold_dir / method
        if not method_dir.exists():
            log.warning("%s %s — method dir missing, skipping.",
                        log_prefix, method)
            continue
        res = recompute_one_method(
            method=method,
            fold=fold,
            method_dir=method_dir,
            classifier_forward=clf_fwd,
            donors=donors,
            frame_mask=frame_mask,
            z_bin=z_bin,
            device=device,
            log_prefix=log_prefix,
        )
        if res is not None:
            summary.append(res)

    log.info("=" * 60)
    log.info("%s ALL DONE in %.1fs (%d methods)",
             log_prefix, time.time() - t_total, len(summary))


if __name__ == "__main__":
    main()
