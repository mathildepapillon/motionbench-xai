"""scripts/train_vaeac.py — Standalone training script for VAEACImputer.

Ports the VAEAC training loop from ``CARE-PD/train_vaeac.py`` to a plain
argparse-driven PyTorch script.  The Hydra config at
``configs/methods/train_vaeac.yaml`` is a stub for future pipeline integration.

Pipeline
--------
1. Optionally load a ``.npz`` data file containing ``x_train`` of shape
   ``(N, J, F, T)``.  If ``--data_path`` is omitted, random Gaussian data of
   the requested shape is generated (useful for smoke-testing).
2. Create a :class:`~motionbench.imputers.vaeac.VAEACImputer`.
3. Run :meth:`~motionbench.imputers.vaeac.VAEACImputer._fit_epochs`.
4. Save the trained checkpoint to ``--checkpoint_path``.

Usage
-----
::

    # Smoke-test on random data (no dataset required)
    python scripts/train_vaeac.py \\
        --J 17 --F 3 --T 81 \\
        --latent_dim 64 --hidden_dim 256 \\
        --epochs 5 --batch_size 16 --lr 1e-3 \\
        --checkpoint_path checkpoints/vaeac.pt

    # Train on a pre-built cache
    python scripts/train_vaeac.py \\
        --data_path data/cache.npz \\
        --checkpoint_path checkpoints/vaeac.pt
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running as a script from the repo root without editable install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from motionbench.imputers.vaeac import VAEACImputer  # noqa: E402


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_data(args: argparse.Namespace) -> torch.Tensor:
    """Load or generate training data of shape ``(N, J, F, T)``."""
    if args.data_path is not None:
        d = np.load(args.data_path, allow_pickle=True)
        # Accept 'x_train' or 'x1_train' (CARE-PD cache format)
        if "x_train" in d:
            x_np = d["x_train"].astype(np.float32)
        elif "x1_train" in d:
            # CARE-PD cache: (N, T, J, C) → permute to (N, J, C, T) motionbench layout
            x_np = d["x1_train"].astype(np.float32)  # (N, T, J, C)
            x_np = x_np.transpose(0, 2, 3, 1)        # (N, J, C, T)
        else:
            raise KeyError(
                f"data_path '{args.data_path}' must contain 'x_train' or 'x1_train'. "
                f"Found keys: {list(d.keys())}"
            )
        print(f"[data] loaded {x_np.shape} from {args.data_path}")
        return torch.from_numpy(x_np)

    # Generate random Gaussian data
    print(
        f"[data] generating synthetic Gaussian data: "
        f"N={args.n_synthetic}, J={args.J}, F={args.F}, T={args.T}"
    )
    return torch.randn(args.n_synthetic, args.J, args.F, args.T)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train a VAEACImputer on skeletal motion data."
    )
    # Architecture
    p.add_argument("--J", type=int, default=17, help="Number of skeletal joints")
    p.add_argument("--F", type=int, default=3, help="Coordinates per joint")
    p.add_argument("--T", type=int, default=81, help="Frames per clip")
    p.add_argument("--latent_dim", type=int, default=64, help="Per-frame latent dim")
    p.add_argument("--hidden_dim", type=int, default=256, help="Transformer d_model")
    # Training
    p.add_argument("--epochs", type=int, default=50, help="Training epochs")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    # Data
    p.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to .npz file with 'x_train' (N, J, F, T) or "
        "'x1_train' (N, T, J, C) array.  Omit to use synthetic Gaussian data.",
    )
    p.add_argument(
        "--n_synthetic",
        type=int,
        default=256,
        help="Number of synthetic samples when --data_path is omitted",
    )
    # Output
    p.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/vaeac.pt",
        help="Where to save the trained checkpoint",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device: 'cpu', 'cuda', 'auto' (default: auto)",
    )

    args = p.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[train] device = {device}")

    _seed_everything(args.seed)

    x_data = _load_data(args)
    print(
        f"[train] shape = {tuple(x_data.shape)}  "
        f"J={args.J}  F={args.F}  T={args.T}  "
        f"latent_dim={args.latent_dim}  hidden_dim={args.hidden_dim}"
    )

    imputer = VAEACImputer(
        J=args.J,
        F=args.F,
        T=args.T,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    print(f"[train] starting training: epochs={args.epochs}  lr={args.lr}")
    losses = imputer._fit_epochs(
        x_data.to(device),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    for epoch, loss in enumerate(losses, 1):
        print(f"  epoch {epoch:4d}  loss={loss:.4f}")

    ckpt_path = Path(args.checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    imputer.save(ckpt_path)
    print(f"[train] checkpoint saved → {ckpt_path}")

    if len(losses) >= 2:
        initial = losses[0]
        final = losses[-1]
        print(f"[train] initial loss = {initial:.4f}  final loss = {final:.4f}")
        if final < initial:
            print("[train] ✓ loss decreased")
        else:
            print("[train] ⚠ loss did not decrease — consider more epochs or lower lr")


if __name__ == "__main__":
    main()
