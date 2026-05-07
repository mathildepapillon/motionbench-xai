"""scripts/train_ptbxl_classifier.py — Train ECGResNet1dClassifier on PTB-XL.

Trains the 1D ResNet (Wang et al. 2017 ``resnet1d_wang``) on the PTB-XL
binary classification task (NORM vs. Myocardial Infarction) for a single fold
and writes a checkpoint to:

    motionbench/classifiers/checkpoints/real/ptbxl_fold{fold}.pt

The PTB-XL official 10-fold split is used:
    train = strat_fold ∈ {1, …, 8}
    val   = strat_fold == 9
    test  = strat_fold == 10  (not used during training)

To match the 3-fold evaluation structure of CARE-PD we train 3 variants,
each using a different held-out fold as the *test* partition (folds 8, 9, 10).
Fold 1 / 2 / 3 in this script correspond to PTB-XL ``strat_fold`` 8 / 9 / 10
as the held-out test:

    --fold 1  →  train on strat_fold ∈ {1..7}, val = strat_fold 9, test = 8
    --fold 2  →  train on strat_fold ∈ {1..8}, val = strat_fold 9, test = 10 (default)
    --fold 3  →  train on strat_fold ∈ {1..7,9}, val = strat_fold 8, test = 10

In practice, for the initial paper result we use the standard split
(fold 2 in this convention) which matches the Strodthoff et al. benchmark.

Usage::

    conda activate motionbench-xai
    python scripts/train_ptbxl_classifier.py \\
        --data_path /data/ptb-xl \\
        --fold 1 \\
        --epochs 30 \\
        --device cuda:0

Expected runtime: ~30 min / fold on a single GPU.

References
----------
* Wagner et al. (2020). PTB-XL dataset. Scientific Data, 7, 154.
* Wang et al. (2017). Time Series Classification from Scratch. IJCNN.
* Strodthoff et al. (2021). Deep Learning for ECG Analysis. IEEE JBHI.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

# Ensure repo root is on path
REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT))

from motionbench.classifiers.ported_ptbxl.resnet1d import ECGResNet1dClassifier
from motionbench.data.real.ptbxl import PTBXLDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-fold split configuration
# ---------------------------------------------------------------------------
# Maps script fold index (1, 2, 3) → (train_folds, val_folds, test_folds)
# using PTB-XL strat_fold values (1–10).
_FOLD_CONFIG: dict[int, dict[str, list[int]]] = {
    1: {"train": list(range(1, 8)), "val": [9],  "test": [8]},
    2: {"train": list(range(1, 9)), "val": [9],  "test": [10]},
    3: {"train": list(range(1, 8)) + [9], "val": [8], "test": [10]},
}


class _PTBXLSplitDataset(PTBXLDataset):
    """PTBXLDataset variant that accepts explicit fold lists instead of named splits."""

    def __init__(
        self,
        data_path: str | Path,
        fold_ids: list[int],
        train_stats: tuple[np.ndarray, np.ndarray] | None = None,
        max_sequences: int | None = None,
    ) -> None:
        # We bypass the parent split logic by monkey-patching _FOLD_SPLITS.
        # Import at function scope to avoid circular issues.
        import motionbench.data.real.ptbxl as _ptbxl_mod

        original = _ptbxl_mod._FOLD_SPLITS.copy()
        _ptbxl_mod._FOLD_SPLITS["_custom"] = (fold_ids,)
        try:
            super().__init__(
                data_path=data_path,
                split="_custom",
                normalize=True,
                max_sequences=max_sequences,
                train_stats=train_stats,
            )
        finally:
            _ptbxl_mod._FOLD_SPLITS.clear()
            _ptbxl_mod._FOLD_SPLITS.update(original)


def _make_weighted_sampler(dataset: PTBXLDataset) -> WeightedRandomSampler:
    """Build a sampler that over-samples the minority class.

    Args:
        dataset: A loaded PTBXLDataset instance.

    Returns:
        WeightedRandomSampler that balances NORM / MI class counts.
    """
    labels = [dataset._samples[i][1] for i in range(len(dataset))]
    class_counts = np.bincount(labels, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = torch.tensor([class_weights[l] for l in labels], dtype=torch.float)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def _evaluate(
    clf: ECGResNet1dClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate *clf* on *loader*, returning loss / acc / AUC metrics.

    Args:
        clf: Trained classifier.
        loader: DataLoader for the evaluation split.
        criterion: Loss function.
        device: Compute device.

    Returns:
        Dict with keys ``"loss"``, ``"acc"``, ``"auc"``.
    """
    from sklearn.metrics import roc_auc_score  # type: ignore[import]

    clf.eval()
    total_loss = 0.0
    all_probs: list[float] = []
    all_labels: list[int] = []
    correct = 0
    n = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = clf(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * len(y)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = logits.argmax(dim=-1)
            correct += int((preds == y).sum().item())
            n += len(y)
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    try:
        auc = float(roc_auc_score(all_labels, all_probs))
    except ValueError:
        auc = float("nan")

    return {
        "loss": total_loss / max(n, 1),
        "acc":  correct / max(n, 1),
        "auc":  auc,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train ECGResNet1dClassifier on PTB-XL NORM vs MI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data_path", type=str, required=True,
                    help="Root directory of the downloaded PTB-XL dataset.")
    ap.add_argument("--fold", type=int, default=2, choices=[1, 2, 3],
                    help="Script fold index (1–3); see module docstring for mapping.")
    ap.add_argument("--epochs", type=int, default=30,
                    help="Number of training epochs.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Initial learning rate (Adam).")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--max_train_seq", type=int, default=None,
                    help="Optionally cap training set size (for debugging).")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--output_dir", type=str,
                    default=str(REPO_ROOT / "motionbench" / "classifiers" / "checkpoints" / "real"),
                    help="Directory to write the checkpoint.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fold_cfg = _FOLD_CONFIG[args.fold]

    log.info("=" * 70)
    log.info("PTB-XL ECGResNet1d training — fold %d", args.fold)
    log.info("  data_path  : %s", args.data_path)
    log.info("  train folds: %s", fold_cfg["train"])
    log.info("  val   folds: %s", fold_cfg["val"])
    log.info("  test  folds: %s (not used in training)", fold_cfg["test"])
    log.info("  device     : %s", device)
    log.info("=" * 70)

    # ------------------------------------------------------------------ data
    log.info("Loading datasets …")
    train_ds = _PTBXLSplitDataset(
        data_path=args.data_path,
        fold_ids=fold_cfg["train"],
        max_sequences=args.max_train_seq,
    )
    train_stats = train_ds.train_stats
    val_ds = _PTBXLSplitDataset(
        data_path=args.data_path,
        fold_ids=fold_cfg["val"],
        train_stats=train_stats,
    )

    log.info("train: %d  val: %d  (class balance approx 50/50 after sampling)",
             len(train_ds), len(val_ds))

    sampler = _make_weighted_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=128, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ------------------------------------------------------------- model
    clf = ECGResNet1dClassifier(n_classes=2).to(device)
    n_params = sum(p.numel() for p in clf.parameters() if p.requires_grad)
    log.info("ECGResNet1dClassifier: %s trainable parameters", f"{n_params:,}")

    # ----------------------------------------------------------- training
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_auc = -1.0
    best_epoch = -1
    best_state: dict | None = None

    for epoch in range(1, args.epochs + 1):
        clf.train()
        train_loss = 0.0
        n_train = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            logits = clf(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(clf.parameters(), max_norm=1.0)
            optim.step()
            train_loss += float(loss.item()) * len(y)
            n_train += len(y)

        val_metrics = _evaluate(clf, val_loader, criterion, device)
        scheduler.step(val_metrics["auc"])

        log.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.3f  val_auc=%.4f%s",
            epoch, args.epochs,
            train_loss / max(n_train, 1),
            val_metrics["loss"],
            val_metrics["acc"],
            val_metrics["auc"],
            " ← best" if val_metrics["auc"] > best_val_auc else "",
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in clf.state_dict().items()}

    log.info("Training complete. Best val AUC=%.4f at epoch %d.", best_val_auc, best_epoch)

    # --------------------------------------------------------- save checkpoint
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"ptbxl_fold{args.fold}.pt"
    assert best_state is not None, "No best state found — training may have failed."
    torch.save(best_state, ckpt_path)
    log.info("Checkpoint written to %s", ckpt_path)

    # Also save normalisation statistics so inference scripts can use them
    import numpy as np
    stats_path = out_dir / f"ptbxl_fold{args.fold}_stats.npz"
    np.savez_compressed(
        stats_path,
        mean=train_stats[0],
        std=train_stats[1],
        train_folds=np.array(fold_cfg["train"]),
        val_folds=np.array(fold_cfg["val"]),
        test_folds=np.array(fold_cfg["test"]),
        best_val_auc=np.array([best_val_auc]),
    )
    log.info("Normalisation stats written to %s", stats_path)


if __name__ == "__main__":
    main()
