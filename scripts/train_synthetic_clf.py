"""Train all three synthetic classifiers on a Gaussian randn dataset and save checkpoints.

Generates synthetic data directly using numpy/torch (no dependency on
motionbench.data.synthetic, which is developed in parallel under Task 1A).

Label rule: 3-class quantile split on mean of x[:, 0, 0, :] (grand mean of
joint-0, feature-0, all frames).

Usage
-----
    python scripts/train_synthetic_clf.py [--J 17] [--F 3] [--T 81] [--epochs 30]

Checkpoint format
-----------------
    {"model_state_dict": ..., "config": {...}, "val_acc": float}

Checkpoints are written to:
    motionbench/classifiers/checkpoints/synthetic_mlp_k4.pt
    motionbench/classifiers/checkpoints/synthetic_cnn.pt
    motionbench/classifiers/checkpoints/synthetic_transformer.pt
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from motionbench.classifiers.synthetic_cnn import SyntheticCNNClassifier
from motionbench.classifiers.synthetic_mlp import SyntheticMLPClassifier
from motionbench.classifiers.synthetic_transformer import SyntheticTransformerClassifier

# On GPU: pin to 1 CPU thread to avoid contention (GPU does the work).
# On CPU: let PyTorch use all available cores for reasonable throughput.
if torch.cuda.is_available():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

CHECKPOINT_DIR = Path(__file__).parent.parent / "motionbench" / "classifiers" / "checkpoints"


def make_dataset(
    N: int,
    J: int,
    F: int,
    T: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic (N, J, F, T) dataset with 3-class quantile labels.

    Args:
        N: Number of samples.
        J: Number of joints.
        F: Features per joint.
        T: Frames per clip.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (x, y) where x is (N, J, F, T) float32 and y is (N,) int64.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N, J, F, T)).astype(np.float32)
    score = x[:, 0, 0, :].mean(axis=-1)
    q33, q67 = float(np.percentile(score, 33)), float(np.percentile(score, 67))
    y = np.where(score < q33, 0, np.where(score < q67, 1, 2)).astype(np.int64)
    return x, y


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy.

    Args:
        logits: (N, n_classes) logit tensor.
        labels: (N,) integer label tensor.

    Returns:
        Fraction of correctly classified samples in [0, 1].
    """
    return float((logits.argmax(dim=-1) == labels).float().mean().item())


def train_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int = 42,
) -> float:
    """Train a model and return final validation accuracy.

    Args:
        model: PyTorch model to train.
        x_train: (N, J, F, T) float32 training inputs.
        y_train: (N,) int64 training labels.
        x_val: (M, J, F, T) float32 validation inputs.
        y_val: (M,) int64 validation labels.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate for Adam.
        device: Torch device.
        seed: Manual seed for reproducibility.

    Returns:
        Validation accuracy after final epoch.
    """
    torch.manual_seed(seed)
    model.to(device)

    X_tr = torch.tensor(x_train)
    Y_tr = torch.tensor(y_train)
    X_va = torch.tensor(x_val).to(device)
    Y_va = torch.tensor(y_val).to(device)

    loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_acc = accuracy(model(X_va), Y_va)
            train_loss = total_loss / len(X_tr)
            print(f"  epoch {epoch + 1:3d}/{epochs}  loss={train_loss:.4f}  val_acc={val_acc:.3f}")

    model.eval()
    with torch.no_grad():
        val_acc = accuracy(model(X_va), Y_va)
    return val_acc


def main() -> None:
    """Train all three classifiers and save checkpoints."""
    parser = argparse.ArgumentParser(description="Train synthetic classifiers")
    parser.add_argument("--J", type=int, default=5, help="Number of joints")
    parser.add_argument("--F", type=int, default=3, help="Features per joint")
    parser.add_argument("--T", type=int, default=16, help="Frames per clip")
    parser.add_argument("--K", type=int, default=4, help="Temporal windows for MLP")
    parser.add_argument("--n_classes", type=int, default=3, help="Number of classes")
    parser.add_argument("--n_train", type=int, default=2000, help="Training samples")
    parser.add_argument("--n_val", type=int, default=500, help="Validation samples")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: J={args.J}, F={args.F}, T={args.T}, K={args.K}, n_classes={args.n_classes}")
    print(f"Data:   n_train={args.n_train}, n_val={args.n_val}, epochs={args.epochs}")
    print()

    x_train, y_train = make_dataset(args.n_train, args.J, args.F, args.T, seed=args.seed)
    x_val, y_val = make_dataset(args.n_val, args.J, args.F, args.T, seed=args.seed + 1)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # MLP — temporal mode
    # ------------------------------------------------------------------
    print("=== SyntheticMLPClassifier (temporal) ===")
    mlp = SyntheticMLPClassifier(
        J=args.J, F=args.F, T=args.T, K=args.K, n_classes=args.n_classes
    )
    val_acc_mlp = train_model(
        mlp, x_train, y_train, x_val, y_val,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=device, seed=args.seed,
    )
    ckpt_mlp = CHECKPOINT_DIR / "synthetic_mlp_k4.pt"
    torch.save(
        {
            "model_state_dict": mlp.state_dict(),
            "config": {
                "J": args.J, "F": args.F, "T": args.T, "K": args.K,
                "n_classes": args.n_classes, "player_mode": "temporal",
            },
            "val_acc": val_acc_mlp,
        },
        ckpt_mlp,
    )
    print(f"Saved {ckpt_mlp}  (val_acc={val_acc_mlp:.3f})\n")

    # ------------------------------------------------------------------
    # CNN
    # ------------------------------------------------------------------
    print("=== SyntheticCNNClassifier ===")
    cnn = SyntheticCNNClassifier(J=args.J, F=args.F, n_classes=args.n_classes)
    val_acc_cnn = train_model(
        cnn, x_train, y_train, x_val, y_val,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=device, seed=args.seed,
    )
    ckpt_cnn = CHECKPOINT_DIR / "synthetic_cnn.pt"
    torch.save(
        {
            "model_state_dict": cnn.state_dict(),
            "config": {"J": args.J, "F": args.F, "n_classes": args.n_classes},
            "val_acc": val_acc_cnn,
        },
        ckpt_cnn,
    )
    print(f"Saved {ckpt_cnn}  (val_acc={val_acc_cnn:.3f})\n")

    # ------------------------------------------------------------------
    # Transformer
    # ------------------------------------------------------------------
    print("=== SyntheticTransformerClassifier ===")
    tfm = SyntheticTransformerClassifier(J=args.J, F=args.F, n_classes=args.n_classes)
    val_acc_tfm = train_model(
        tfm, x_train, y_train, x_val, y_val,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=device, seed=args.seed,
    )
    ckpt_tfm = CHECKPOINT_DIR / "synthetic_transformer.pt"
    torch.save(
        {
            "model_state_dict": tfm.state_dict(),
            "config": {"J": args.J, "F": args.F, "n_classes": args.n_classes},
            "val_acc": val_acc_tfm,
        },
        ckpt_tfm,
    )
    print(f"Saved {ckpt_tfm}  (val_acc={val_acc_tfm:.3f})\n")

    print("=== Summary ===")
    print(f"  MLP        val_acc={val_acc_mlp:.3f}")
    print(f"  CNN        val_acc={val_acc_cnn:.3f}")
    print(f"  Transformer val_acc={val_acc_tfm:.3f}")


if __name__ == "__main__":
    main()
