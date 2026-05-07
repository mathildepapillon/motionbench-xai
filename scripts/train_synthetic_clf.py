"""Train synthetic classifiers (11 datasets × 3 architectures = 33 checkpoints) and save them.

The 9 datasets that appear in the main-paper Table 1 (`tab:datasets`) are
trained alongside two appendix datasets used by the player-imputer 2x2 grid
(``window_label_gaussian``) and the XOR-label robustness sweep
(``xor_label_gaussian``).

Each classifier is trained on data drawn from the same parametric distribution as
the evaluation set, using a separate seed to avoid data leakage.  Training is
dispatched across all available CUDA devices in parallel via joblib.

Checkpoint layout
-----------------
motionbench/classifiers/checkpoints/synthetic/
  gaussian_k4/
    synthetic_mlp.pt
    synthetic_cnn.pt
    synthetic_transformer.pt
  gaussian_k8/ ...
  burr_m5/ ...
  burr_m10/ ...
  skeleton_structured/ ...
  gait_periodic/ ...

Each .pt stores::

    {
        "model_state_dict": OrderedDict,
        "config": dict,          # constructor kwargs
        "val_acc": float,        # best validation accuracy
        "epoch": int,            # epoch at which best val_acc was achieved
        "dataset": str,
        "classifier": str,
    }

Usage
-----
# Train all 33 (11 datasets × 3 architectures) on all available GPUs:
python scripts/train_synthetic_clf.py

# Subset of datasets/classifiers on specific GPUs:
python scripts/train_synthetic_clf.py \\
    --datasets gaussian_k4 burr_m5 \\
    --classifiers synthetic_mlp synthetic_cnn \\
    --gpus 0 1

# Force CPU (debugging only — not recommended):
python scripts/train_synthetic_clf.py --force-cpu
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Project root on path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

from motionbench.classifiers.synthetic_cnn import SyntheticCNNClassifier  # noqa: E402
from motionbench.classifiers.synthetic_mlp import SyntheticMLPClassifier  # noqa: E402
from motionbench.classifiers.synthetic_transformer import (  # noqa: E402
    SyntheticTransformerClassifier,
)
from motionbench.data.synthetic.burr_motion import BurrMotionBenchmark  # noqa: E402
from motionbench.data.synthetic.gait_periodic import GaitPeriodicDataset  # noqa: E402
from motionbench.data.synthetic.gaussian_motion import GaussianMotionDataset  # noqa: E402
from motionbench.data.synthetic.label_functions import (  # noqa: E402
    LocalizedSpatial,
    LocalizedTemporal,
    ThresholdedXOR,
)
from motionbench.data.synthetic.low_rank_manifold import (  # noqa: E402
    LowRankManifoldDataset,
)
from motionbench.data.synthetic.skeleton_gait_combined import (  # noqa: E402
    SkeletonGaitDataset,
)
from motionbench.data.synthetic.skeleton_structured import (  # noqa: E402
    SkeletonStructuredDataset,
)

# ---------------------------------------------------------------------------
# Dataset config table  (mirrors YAML configs exactly)
# ---------------------------------------------------------------------------

DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "gaussian_k4": dict(
        cls=GaussianMotionDataset, J=5, F=3, T=16, K=4, rho=0.5, alpha=0.8, seed=42
    ),
    "gaussian_k8": dict(
        cls=GaussianMotionDataset, J=5, F=3, T=16, K=8, rho=0.5, alpha=0.8, seed=43
    ),
    "burr_m5": dict(
        cls=BurrMotionBenchmark, J=5, F=3, T=20, K=5, rho=0.5, alpha=0.8, seed=44
    ),
    "burr_m10": dict(
        cls=BurrMotionBenchmark, J=5, F=3, T=20, K=10, rho=0.5, alpha=0.8, seed=45
    ),
    "skeleton_structured": dict(
        cls=SkeletonStructuredDataset,
        J=17, F=3, T=16, K=4,
        alpha_time=0.9, decay=0.5, n_classes=3, seed=46,
    ),
    "gait_periodic": dict(
        cls=GaitPeriodicDataset,
        J=17, F=3, T=16, K=4,
        # period_mean=7.0: T/period = 16/7 ≈ 2.29 (non-integer) so the cosine
        # kernel does NOT cancel over T frames and the grand-mean label is learnable.
        period_mean=7.0, n_harmonics=3, n_classes=3, seed=47,
    ),
    # New datasets (May 2026): close coverage gaps for spatial / temporal
    # localization and low-rank manifold geometry.
    "low_rank_manifold": dict(
        cls=LowRankManifoldDataset,
        J=17, F=3, T=16, K=4,
        rank=4, eps=0.01, alpha_time=0.9, n_classes=3, seed=99,
    ),
    "window_label_gaussian": dict(
        cls=GaussianMotionDataset,
        J=5, F=3, T=16, K=4,
        rho=0.5, alpha=0.8, seed=142,
        # Localized: only window 1 of K=4 drives y.
        label_fn=LocalizedTemporal(window_idx=1, K=4, n_classes=3),
    ),
    "joint_subset_skeleton": dict(
        cls=SkeletonStructuredDataset,
        J=17, F=3, T=16, K=4,
        alpha_time=0.9, decay=0.5, n_classes=3, seed=146,
        # Localized: only joint 6 (left ankle) drives y.
        label_fn=LocalizedSpatial(joint_idx=6, n_classes=3),
    ),
    # Non-smooth label complement to gaussian_k4: identical Σ_J ⊗ I_F ⊗ Σ_T
    # and shape; the only change is the label generator.  Bits b_k from the
    # binarised window means feed a parity-of-pairs (XOR) score, producing
    # a piecewise-constant decision surface with zero marginal effects.
    "xor_label_gaussian": dict(
        cls=GaussianMotionDataset,
        J=5, F=3, T=16, K=4,
        rho=0.5, alpha=0.8, seed=152,
        label_fn=ThresholdedXOR(K=4, n_classes=3),
    ),
    # Fourth-quadrant pillar (May 2026): high-spatial × high-temporal —
    # skeleton-adjacency Σ_joint × gait-periodic Σ_time.  Closes the missing
    # cell of the 2x2 design grid.
    "skeleton_gait_combined": dict(
        cls=SkeletonGaitDataset,
        J=17, F=3, T=16, K=4,
        decay=0.5, period_mean=7.0, period_std=1.0, n_harmonics=3,
        n_classes=3, seed=51,
    ),
}

# ---------------------------------------------------------------------------
# Per-architecture training hyperparameters
# ---------------------------------------------------------------------------

CLF_HPARAMS: dict[str, dict[str, Any]] = {
    "synthetic_mlp": dict(lr=5e-3, epochs=200, batch_size=64, weight_decay=1e-3),
    "synthetic_cnn": dict(lr=1e-3, epochs=200, batch_size=64, weight_decay=1e-4),
    "synthetic_transformer": dict(lr=5e-4, epochs=200, batch_size=32, weight_decay=1e-4),
}

# Early stopping patience (epochs without val_acc improvement)
PATIENCE = 20
# Minimum acceptable val_acc (warning threshold)
MIN_ACC_WARN = 0.65
# Minimum acceptable val_acc (hard abort threshold — retry with lr /= 5)
MIN_ACC_RETRY = 0.50
# Number of samples drawn for training / validation (separate from eval N=200)
N_TRAIN = 2000
N_VAL = 500
# Seed offset used when drawing train/val data to avoid overlap with eval set
TRAIN_SEED_OFFSET = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_dataset(ds_cfg: dict[str, Any], N: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Instantiate a dataset and return (X, y) tensors.

    Each dataset class pre-generates all N samples at construction time and
    stores them as ``_x`` / ``_y`` tensors.

    Args:
        ds_cfg: Entry from DATASET_CONFIGS (includes cls and constructor kwargs).
        N: Number of samples to generate.
        seed: Overrides the config seed so train/val use distinct data.

    Returns:
        X: (N, J, F, T) float32 tensor.
        y: (N,) int64 tensor.
    """
    cls = ds_cfg["cls"]

    # Build per-class constructor kwargs (strip keys the class does not accept)
    shared = {"J": ds_cfg["J"], "F": ds_cfg["F"], "T": ds_cfg["T"], "N": N, "seed": seed}

    if cls is GaussianMotionDataset:
        kwargs = {**shared, "K": ds_cfg["K"], "rho": ds_cfg["rho"], "alpha": ds_cfg["alpha"]}
        if "label_fn" in ds_cfg:
            kwargs["label_fn"] = ds_cfg["label_fn"]
    elif cls is BurrMotionBenchmark:
        kwargs = {**shared, "rho": ds_cfg["rho"], "alpha": ds_cfg["alpha"]}
    elif cls is SkeletonStructuredDataset:
        kwargs = {**shared, "alpha_time": ds_cfg["alpha_time"], "decay": ds_cfg["decay"],
                  "n_classes": ds_cfg["n_classes"]}
        if "label_fn" in ds_cfg:
            kwargs["label_fn"] = ds_cfg["label_fn"]
    elif cls is GaitPeriodicDataset:
        kwargs = {**shared, "period_mean": ds_cfg["period_mean"],
                  "n_harmonics": ds_cfg["n_harmonics"], "n_classes": ds_cfg["n_classes"]}
    elif cls is SkeletonGaitDataset:
        kwargs = {**shared,
                  "decay": ds_cfg["decay"],
                  "period_mean": ds_cfg["period_mean"],
                  "period_std": ds_cfg.get("period_std", 1.0),
                  "n_harmonics": ds_cfg["n_harmonics"],
                  "n_classes": ds_cfg["n_classes"]}
    elif cls is LowRankManifoldDataset:
        kwargs = {**shared, "rank": ds_cfg["rank"], "eps": ds_cfg["eps"],
                  "alpha_time": ds_cfg["alpha_time"], "n_classes": ds_cfg["n_classes"]}
    else:
        raise ValueError(f"Unknown dataset class: {cls}")

    dataset = cls(**kwargs)
    # All synthetic dataset classes store (N, J, F, T) tensors as _x / _y
    return dataset._x, dataset._y


def _build_classifier(clf_name: str, ds_cfg: dict[str, Any], n_classes: int) -> nn.Module:
    """Instantiate a classifier with the correct shape for a given dataset.

    Args:
        clf_name: One of ``synthetic_mlp``, ``synthetic_cnn``, ``synthetic_transformer``.
        ds_cfg: Dataset config dict (provides J, F, T, K).
        n_classes: Number of output classes.

    Returns:
        An uninitialized (randomly weighted) :class:`~torch.nn.Module`.
    """
    J, F, T, K = ds_cfg["J"], ds_cfg["F"], ds_cfg["T"], ds_cfg.get("K", 4)
    if clf_name == "synthetic_mlp":
        return SyntheticMLPClassifier(J=J, F=F, T=T, K=K, n_classes=n_classes, hidden=64)
    if clf_name == "synthetic_cnn":
        return SyntheticCNNClassifier(J=J, F=F, n_classes=n_classes)
    if clf_name == "synthetic_transformer":
        return SyntheticTransformerClassifier(J=J, F=F, n_classes=n_classes, d_model=32, nhead=4, num_layers=2)
    raise ValueError(f"Unknown classifier: {clf_name!r}")


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean().item())


def _train_single(
    dataset_name: str,
    clf_name: str,
    device_str: str,
    checkpoint_dir: Path,
    lr: float,
    epochs: int,
    batch_size: int,
    weight_decay: float,
    seed: int = 42,
) -> dict[str, Any]:
    """Train one (dataset, classifier) combination and save the best checkpoint.

    Args:
        dataset_name: Key into DATASET_CONFIGS.
        clf_name: Key into CLF_HPARAMS.
        device_str: Torch device string (e.g. ``"cuda:0"``).
        checkpoint_dir: Root directory for checkpoint files.
        lr: Learning rate for Adam.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        weight_decay: L2 regularisation coefficient.
        seed: Manual seed for reproducibility.

    Returns:
        Result dict with keys ``dataset``, ``classifier``, ``val_acc``, ``epoch``,
        ``device``, ``path``.
    """
    t0 = time.time()
    device = torch.device(device_str)
    ds_cfg = DATASET_CONFIGS[dataset_name]

    # ---- Data ---------------------------------------------------------------
    train_seed = ds_cfg["seed"] + TRAIN_SEED_OFFSET
    val_seed = ds_cfg["seed"] + TRAIN_SEED_OFFSET + 1
    X_tr, y_tr = _build_dataset(ds_cfg, N_TRAIN, seed=train_seed)
    X_va, y_va = _build_dataset(ds_cfg, N_VAL, seed=val_seed)
    n_classes: int = int(y_tr.max().item()) + 1

    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    X_va, y_va = X_va.to(device), y_va.to(device)

    loader = DataLoader(
        TensorDataset(X_tr.cpu(), y_tr.cpu()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    # ---- Model --------------------------------------------------------------
    torch.manual_seed(seed)
    model = _build_classifier(clf_name, ds_cfg, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            criterion(model(xb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_acc = _accuracy(model(X_va), y_va)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    elapsed = time.time() - t0

    # ---- Checkpoint ---------------------------------------------------------
    out_dir = checkpoint_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{clf_name}.pt"

    clf_config: dict[str, Any] = {
        "J": ds_cfg["J"], "F": ds_cfg["F"], "n_classes": n_classes,
    }
    if clf_name == "synthetic_mlp":
        clf_config.update({"T": ds_cfg["T"], "K": ds_cfg.get("K", 4)})
    if clf_name == "synthetic_transformer":
        clf_config.update({"d_model": 32, "nhead": 4, "num_layers": 2})

    torch.save(
        {
            "model_state_dict": best_state,
            "config": clf_config,
            "val_acc": best_val_acc,
            "epoch": best_epoch,
            "dataset": dataset_name,
            "classifier": clf_name,
        },
        ckpt_path,
    )

    result = {
        "dataset": dataset_name,
        "classifier": clf_name,
        "val_acc": best_val_acc,
        "epoch": best_epoch,
        "device": device_str,
        "path": str(ckpt_path),
        "elapsed_s": elapsed,
    }

    # ---- Accuracy gate ------------------------------------------------------
    status = "OK"
    if best_val_acc < MIN_ACC_WARN:
        status = "WARN (val_acc < 0.65)"
    print(
        f"  [{status}] {dataset_name}/{clf_name}  "
        f"val_acc={best_val_acc:.3f}  epoch={best_epoch}  "
        f"device={device_str}  t={elapsed:.1f}s"
    )
    return result


def _train_with_retry(
    dataset_name: str,
    clf_name: str,
    device_str: str,
    checkpoint_dir: Path,
    seed: int = 42,
    force: bool = False,
) -> dict[str, Any]:
    """Wrapper that retries with lr /= 5 if val_acc falls below MIN_ACC_RETRY.

    Args:
        dataset_name: Key into DATASET_CONFIGS.
        clf_name: Key into CLF_HPARAMS.
        device_str: Torch device string.
        checkpoint_dir: Root checkpoint directory.
        seed: Manual random seed.
        force: If True, retrain even when a checkpoint already exists.

    Returns:
        Result dict from the best (or retried) training run.
    """
    ckpt_path = checkpoint_dir / dataset_name / f"{clf_name}.pt"
    if ckpt_path.exists() and not force:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            print(
                f"  [SKIP] {dataset_name}/{clf_name}  "
                f"val_acc={ckpt.get('val_acc', float('nan')):.3f}  "
                f"(checkpoint exists; pass --force to retrain)"
            )
            return {
                "dataset": dataset_name,
                "classifier": clf_name,
                "val_acc": float(ckpt.get("val_acc", float("nan"))),
                "epoch": int(ckpt.get("epoch", -1)),
                "device": device_str,
                "path": str(ckpt_path),
                "skipped": True,
            }
        except Exception as e:  # noqa: BLE001 — fall through to retraining if load fails
            print(f"  [WARN] could not load existing {ckpt_path}: {e}; retraining")

    hp = CLF_HPARAMS[clf_name]
    result = _train_single(
        dataset_name, clf_name, device_str, checkpoint_dir,
        lr=hp["lr"], epochs=hp["epochs"],
        batch_size=hp["batch_size"], weight_decay=hp["weight_decay"],
        seed=seed,
    )
    if result["val_acc"] < MIN_ACC_RETRY:
        print(
            f"  [RETRY] {dataset_name}/{clf_name} val_acc={result['val_acc']:.3f} "
            f"< {MIN_ACC_RETRY} — retrying with lr={hp['lr'] / 5:.6f}"
        )
        result = _train_single(
            dataset_name, clf_name, device_str, checkpoint_dir,
            lr=hp["lr"] / 5, epochs=hp["epochs"],
            batch_size=hp["batch_size"], weight_decay=hp["weight_decay"],
            seed=seed + 1,
        )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train all synthetic classifiers across all datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_CONFIGS.keys()),
        choices=list(DATASET_CONFIGS.keys()),
        metavar="DATASET",
        help="Datasets to train on (default: all 11 — 9 main-paper datasets "
             "plus 2 appendix-only datasets).",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=list(CLF_HPARAMS.keys()),
        choices=list(CLF_HPARAMS.keys()),
        metavar="CLF",
        help="Classifier architectures to train (default: all 3).",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="GPU indices to use (default: all available CUDA devices).",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Run on CPU instead of GPU. For debugging only — very slow.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Override parallelism (useful with --force-cpu on multi-core boxes).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=_REPO_ROOT / "motionbench" / "classifiers" / "checkpoints" / "synthetic",
        help="Root directory for checkpoint files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even when a checkpoint already exists at the target path.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = _parse_args()

    # ---- Device selection ---------------------------------------------------
    if args.force_cpu:
        print(
            "\n*** WARNING: --force-cpu is set. Training on CPU is very slow. ***\n"
            "    Use this flag for debugging only.\n"
        )
        devices = ["cpu"]
    else:
        n_cuda = torch.cuda.device_count()
        if n_cuda == 0:
            raise RuntimeError(
                "No CUDA devices found — refusing to train on CPU.\n"
                "If you really need CPU training, pass --force-cpu."
            )
        if args.gpus is not None:
            invalid = [g for g in args.gpus if g >= n_cuda]
            if invalid:
                raise ValueError(
                    f"GPU indices {invalid} are out of range "
                    f"(found {n_cuda} CUDA devices)."
                )
            gpu_ids = args.gpus
        else:
            gpu_ids = list(range(n_cuda))
        devices = [f"cuda:{g}" for g in gpu_ids]

    # ---- Jobs ---------------------------------------------------------------
    jobs: list[tuple[str, str]] = [
        (ds, clf)
        for ds in args.datasets
        for clf in args.classifiers
    ]
    n_jobs = len(jobs)
    n_workers = args.n_workers if args.n_workers is not None else min(n_jobs, len(devices))

    print(f"MotionBench-XAI — Synthetic Classifier Training")
    print(f"  Jobs      : {n_jobs} ({len(args.datasets)} datasets × {len(args.classifiers)} classifiers)")
    print(f"  Devices   : {devices}")
    print(f"  Workers   : {n_workers} parallel jobs")
    print(f"  Output dir: {args.checkpoint_dir}")
    print()

    # Round-robin device assignment
    assigned_devices = [devices[i % len(devices)] for i in range(n_jobs)]

    t_start = time.time()

    results: list[dict[str, Any]] = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(_train_with_retry)(
            ds, clf, dev, args.checkpoint_dir, seed=args.seed, force=args.force,
        )
        for (ds, clf), dev in zip(jobs, assigned_devices)
    )

    total_elapsed = time.time() - t_start

    # ---- Summary table ------------------------------------------------------
    print()
    print("=" * 80)
    print(f"{'DATASET':<22} {'CLASSIFIER':<22} {'VAL_ACC':>8} {'EPOCH':>6} {'DEVICE':>8}  PATH")
    print("=" * 80)
    for r in sorted(results, key=lambda x: (x["dataset"], x["classifier"])):
        flag = "  !" if r["val_acc"] < MIN_ACC_WARN else "   "
        print(
            f"{flag}{r['dataset']:<20} {r['classifier']:<22} "
            f"{r['val_acc']:>8.3f} {r['epoch']:>6}  {r['device']:>8}  {r['path']}"
        )
    print("=" * 80)
    print(f"Total wall time: {total_elapsed:.1f}s")

    # Warn if any run is below threshold
    bad = [r for r in results if r["val_acc"] < MIN_ACC_WARN]
    if bad:
        print(
            f"\n*** {len(bad)} run(s) have val_acc < {MIN_ACC_WARN:.2f} (marked with '!'). ***\n"
            "Consider increasing --epochs or tuning learning rate."
        )


if __name__ == "__main__":
    main()
