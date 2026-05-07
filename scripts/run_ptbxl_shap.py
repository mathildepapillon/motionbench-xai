"""scripts/run_ptbxl_shap.py — KernelSHAP sweep on PTB-XL ECG.

Mirrors ``run_care_pd_multiclf.py`` for the PTB-XL dataset.  Runs KernelSHAP
with five imputation strategies (Zero / Mean / Marginal / VAEAC / Flow) on
PTB-XL NORM-vs-MI test sequences using the trained ECGResNet1dClassifier.

Shared utility functions are imported directly from
``scripts/run_care_pd_multiclf.py`` to avoid code duplication:

    build_coalition_masks       — build (z_bin, frame_mask) tensors
    kernel_shap_exact           — weighted least-squares SHAP solver
    shapley_kernel              — Shapley kernel weights
    faithfulness_correlation    — per-sequence faithfulness metric
    player_aopc                 — per-sequence PlayerAOPC metric

Player set
----------
K=4 temporal windows over T=1000 (each window = 250 time-steps).

Results are written to::

    results/ptbxl/{fold}/{method}/result.json

Run ``scripts/compute_ptbxl_cis.py`` afterwards to pool and bootstrap CIs.

Usage::

    conda activate motionbench-xai

    # Train classifier first (if not already done)
    python scripts/train_ptbxl_classifier.py --data_path /data/ptb-xl --fold 1

    # Run SHAP sweep
    CUDA_VISIBLE_DEVICES=0 python scripts/run_ptbxl_shap.py \\
        --data_path /data/ptb-xl --fold 1 --device cuda:0

    # All three folds
    for fold in 1 2 3; do
        CUDA_VISIBLE_DEVICES=${fold-1} python scripts/run_ptbxl_shap.py \\
            --data_path /data/ptb-xl --fold $fold
    done
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).parents[1]
SCRIPTS_DIR = Path(__file__).parent

# ------------------------------------------------------------------- helpers
# Import shared KernelSHAP utilities from run_care_pd_multiclf — no duplication.
sys.path.insert(0, str(SCRIPTS_DIR))
from run_care_pd_multiclf import (   # noqa: E402 (import after sys.path manipulation)
    build_coalition_masks,
    faithfulness_correlation,
    kernel_shap_exact,
    player_aopc,
    shapley_kernel,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------- constants
RESULTS_ROOT = REPO_ROOT / "results" / "ptbxl"
CKPT_DIR = REPO_ROOT / "motionbench" / "classifiers" / "checkpoints" / "real"
STATS_TEMPLATE = "ptbxl_fold{fold}_stats.npz"
CKPT_TEMPLATE  = "ptbxl_fold{fold}.pt"

K = 4       # temporal windows; each window = T//K = 250 time-steps
DEVICE = "cuda:0"

ALL_METHODS = [
    "kernelshap_zero",
    "kernelshap_mean",
    "kernelshap_marginal",
    "kernelshap_vaeac",
    "kernelshap_flow",
]


# ---------------------------------------------------------------------- helpers

def build_completions_offmanifold(
    x: Tensor,
    frame_mask_2k: Tensor,
    kind: str,
    mean_per_jf: Tensor | None = None,
) -> Tensor:
    """Build off-manifold completions for all 2^K coalitions.

    This function replicates the same off-manifold strategy used for CARE-PD
    so that the PTB-XL sweep is directly comparable.

    Args:
        x: ``(J, F, T)`` float32 single sequence.
        frame_mask_2k: ``(2^K, T)`` bool — True = observed frame.
        kind: ``"zero"`` | ``"mean"`` | ``"marginal"``.
        mean_per_jf: ``(J, F, T)`` donor or ``(J, F)`` mean tensor.

    Returns:
        ``(2^K, J, F, T)`` float32 completions.
    """
    J, F, T = x.shape
    n_coal = frame_mask_2k.shape[0]
    obs = frame_mask_2k.view(n_coal, 1, 1, T).expand(n_coal, J, F, T)
    if kind == "zero":
        fill = torch.zeros_like(x)
    elif kind == "mean":
        fill = mean_per_jf.view(J, F, 1).expand(J, F, T)
    elif kind == "marginal":
        fill = mean_per_jf   # (J, F, T) donor
    else:
        raise ValueError(f"unknown kind {kind!r}")
    x_b    = x.view(1, J, F, T).expand(n_coal, J, F, T)
    fill_b = fill.view(1, J, F, T).expand(n_coal, J, F, T)
    return torch.where(obs, x_b, fill_b).contiguous()


# ------------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data_path", type=str, required=True,
                    help="Root directory of the downloaded PTB-XL dataset.")
    ap.add_argument("--fold", type=int, default=2, choices=[1, 2, 3],
                    help="Script fold index (1–3); matches train_ptbxl_classifier.py.")
    ap.add_argument("--n_seq", type=int, default=200,
                    help="Number of test sequences to evaluate.")
    ap.add_argument("--methods", type=str, nargs="+", default=None,
                    help="Subset of methods to run (default: all five).")
    ap.add_argument("--results_dir", type=str, default=str(RESULTS_ROOT))
    ap.add_argument("--device", type=str, default=DEVICE)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    fold = args.fold
    N_SEQ = int(args.n_seq)
    results_root = Path(args.results_dir)
    device = torch.device(args.device)

    t_total = time.time()

    # ------------------------------------------------------------------ data
    log.info("[fold%d] loading PTB-XL test data from %s", fold, args.data_path)

    # Load normalisation statistics from the classifier training run
    stats_path = CKPT_DIR / STATS_TEMPLATE.format(fold=fold)
    if stats_path.exists():
        stats = np.load(stats_path)
        train_stats = (stats["mean"].astype(np.float32), stats["std"].astype(np.float32))
        test_folds  = stats["test_folds"].tolist()
        log.info("[fold%d] loaded stats from %s; test_folds=%s", fold, stats_path, test_folds)
    else:
        log.warning(
            "[fold%d] stats file %s not found — using raw (unnormalized) data. "
            "Run train_ptbxl_classifier.py first.",
            fold, stats_path,
        )
        train_stats = None
        test_folds = [10]  # default to standard held-out fold

    # Import here to avoid circular issues
    from motionbench.data.real.ptbxl import PTBXLDataset, _FOLD_SPLITS

    # Build a dataset for the test fold(s)
    _FOLD_SPLITS["_test_folds"] = (test_folds,)
    test_ds = PTBXLDataset(
        data_path=args.data_path,
        split="_test_folds",
        normalize=(train_stats is not None),
        max_sequences=N_SEQ,
        train_stats=train_stats,
    )
    del _FOLD_SPLITS["_test_folds"]

    # Extract raw arrays for efficient batch processing
    x_val = np.stack(
        [s[0] for s in test_ds._samples[:N_SEQ]], axis=0
    )  # (N, T=1000, J=12)
    # Transpose to (N, J=12, F=1, T=1000)
    x_val = x_val.transpose(0, 2, 1)[:, :, np.newaxis, :]  # (N, 12, 1, 1000)
    N, J, F, T = x_val.shape
    log.info("[fold%d] N=%d J=%d F=%d T=%d", fold, N, J, F, T)

    # Train pool for marginal donor sampling: use training folds
    if stats_path.exists():
        train_fold_ids = stats["train_folds"].tolist()
    else:
        train_fold_ids = list(range(1, 9))

    _FOLD_SPLITS["_train_folds"] = (train_fold_ids,)
    train_ds = PTBXLDataset(
        data_path=args.data_path,
        split="_train_folds",
        normalize=(train_stats is not None),
        max_sequences=2000,
        train_stats=train_stats,
    )
    del _FOLD_SPLITS["_train_folds"]
    x_train = np.stack(
        [s[0] for s in train_ds._samples], axis=0
    )  # (N_tr, T, J)
    x_train = x_train.transpose(0, 2, 1)[:, :, np.newaxis, :]  # (N_tr, J, 1, T)
    log.info("[fold%d] train pool: %d records", fold, x_train.shape[0])

    # ---------------------------------------------------------- coalitions
    z_bin, frame_mask = build_coalition_masks(K, T)
    n_coal = 1 << K

    # --------------------------------------------------------- classifier
    ckpt_path = CKPT_DIR / CKPT_TEMPLATE.format(fold=fold)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {ckpt_path}\n"
            f"Run: python scripts/train_ptbxl_classifier.py --data_path {args.data_path} --fold {fold}"
        )
    from motionbench.classifiers.ported_ptbxl.resnet1d import ECGResNet1dClassifier
    clf = ECGResNet1dClassifier(n_classes=2, checkpoint_path=str(ckpt_path)).to(device)
    clf.eval()
    log.info("[fold%d] classifier loaded from %s", fold, ckpt_path.name)

    # Get predicted targets on the full test set
    with torch.no_grad():
        logits_all = clf(torch.from_numpy(x_val).to(device))
    targets = logits_all.cpu().argmax(dim=-1).numpy()
    log.info("[fold%d] target distribution: %s",
             fold, np.bincount(targets, minlength=2).tolist())

    # --------------------------------------------------------- imputer setup
    mean_jf = torch.from_numpy(x_train.mean(axis=(0, 3))).float()  # (J, F)
    rng = np.random.default_rng(42 + fold)
    donor_idx = rng.integers(0, x_train.shape[0], size=N)
    donors = torch.from_numpy(x_train[donor_idx]).float()

    vaeac_imputer = None
    flow_imputer  = None

    def get_vaeac():
        nonlocal vaeac_imputer
        if vaeac_imputer is None:
            from motionbench.imputers.ptbxl_imputer import (
                PTBXLVAEACImputer,
                _VAEAC_CKPT_DIR,
                _resolve_cfg,
                _VAEAC_DEFAULT_CFG,
            )
            cfg_path = _resolve_cfg(
                _VAEAC_CKPT_DIR, "ptbxl_vaeac_cfg.json", _VAEAC_DEFAULT_CFG
            )
            from motionbench.imputers.carepd_imputer import _load_vaeac
            vaeac_imputer = _load_vaeac(_VAEAC_CKPT_DIR, cfg_path, device)
        return vaeac_imputer

    def get_flow():
        nonlocal flow_imputer
        if flow_imputer is None:
            from motionbench.imputers.ptbxl_imputer import (
                _FLOW_CKPT_DIR,
                _resolve_cfg,
                _FLOW_DEFAULT_CFG,
            )
            from motionbench.imputers.carepd_imputer import _load_flow
            cfg_path = _resolve_cfg(
                _FLOW_CKPT_DIR, "ptbxl_flow_cfg.json", _FLOW_DEFAULT_CFG
            )
            cfg = json.loads(cfg_path.read_text())
            cfg["num_steps"] = 20   # speed-up for sweep
            import tempfile
            tf = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            json.dump(cfg, tf)
            tf.close()
            try:
                flow_imputer = _load_flow(_FLOW_CKPT_DIR, Path(tf.name), device)
            finally:
                Path(tf.name).unlink(missing_ok=True)
        return flow_imputer

    # --------------------------------------------------------- methods
    methods = args.methods if args.methods else ALL_METHODS
    fold_dir = results_root / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    for method in methods:
        method_dir = fold_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        result_path = method_dir / "result.json"
        if result_path.exists():
            log.info("[fold%d] %s already done, skipping.", fold, method)
            with open(result_path) as fh:
                summary_rows.append(json.load(fh))
            continue

        log.info("=" * 60)
        log.info("[fold%d] Method: %s", fold, method)
        t_method = time.time()

        phis  = np.zeros((N, K), dtype=np.float32)
        v_all = np.zeros((N, n_coal), dtype=np.float32)

        imp = None
        if method == "kernelshap_vaeac":
            try:
                imp = get_vaeac()
                log.info("[fold%d] VAEAC loaded", fold)
            except Exception as exc:
                log.warning("[fold%d] VAEAC unavailable (%s) — skipping method.", fold, exc)
                continue
        elif method == "kernelshap_flow":
            try:
                imp = get_flow()
                log.info("[fold%d] Flow loaded", fold)
            except Exception as exc:
                log.warning("[fold%d] Flow unavailable (%s) — skipping method.", fold, exc)
                continue

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
            with torch.no_grad():
                logits_b = clf(comps.to(device))
            v_b = torch.softmax(logits_b, dim=-1)[:, target_i].cpu()
            v_all[i] = v_b.numpy()
            phis[i]  = kernel_shap_exact(z_bin, v_b, K).numpy()

            if (i + 1) % 25 == 0 or i == N - 1:
                elapsed = time.time() - t_method
                log.info("  [fold%d] %s  %d/%d  (%.2fs/seq)",
                         fold, method, i + 1, N, elapsed / (i + 1))

        # ------------------------------------------------------ metrics
        faiths, aopcs = [], []
        for i in range(N):
            v_i   = torch.from_numpy(v_all[i])
            phi_i = torch.from_numpy(phis[i])
            faiths.append(faithfulness_correlation(z_bin, v_i, phi_i))
            aopcs.append(player_aopc(v_i, z_bin, phi_i, K))

        faiths_arr = np.asarray(faiths, dtype=np.float64)
        aopcs_arr  = np.asarray(aopcs,  dtype=np.float64)
        n_finite   = int(np.isfinite(faiths_arr).sum())

        faith_mean = float(np.nanmean(faiths_arr))
        faith_std  = float(np.nanstd(faiths_arr, ddof=1)) if n_finite > 1 else float("nan")
        aopc_mean  = float(np.mean(aopcs_arr))
        aopc_std   = float(np.std(aopcs_arr, ddof=1)) if N > 1 else float("nan")

        np.savez_compressed(
            method_dir / "attributions.npz",
            phi=phis, x=x_val, target=targets, v=v_all,
        )
        result = {
            "dataset": "ptbxl",
            "classifier": "ecg_resnet1d",
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
        log.info("  [fold%d] %s done %.1fs — faith=%+.3f aopc=%+.3f (n_fin=%d)",
                 fold, method, time.time() - t_method,
                 faith_mean, aopc_mean, n_finite)

    (fold_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2))
    log.info("=" * 60)
    log.info("[fold%d] ALL DONE in %.1fs", fold, time.time() - t_total)


if __name__ == "__main__":
    main()
