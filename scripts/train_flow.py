"""scripts/train_flow — Training entry point for FlowMatchingImputer.

Trains a :class:`~motionbench.imputers.flow_matching.FlowMatchingImputer`
on a pre-saved ``.pt`` tensor dataset and saves the checkpoint.

Usage
-----
Train on a pre-saved tensor dataset (shape ``(N, J, F, T)``)::

    python scripts/train_flow.py \\
        --data_path data/burr_motion.pt \\
        --save_path checkpoints/flow_burr.pt \\
        --J 5 --F 3 --T 16 \\
        --hidden_dim 256 --num_steps 100 \\
        --noise_init_scale 1.0 \\
        --n_epochs 200 --batch_size 32 --lr 1e-3

Generate a synthetic Gaussian dataset on-the-fly for quick debugging::

    python scripts/train_flow.py \\
        --synthetic --n_synthetic 500 \\
        --J 5 --F 3 --T 16 \\
        --n_epochs 20 --save_path /tmp/flow_debug.pt

The script saves the checkpoint to ``save_path`` and prints training
loss per epoch.  The checkpoint can be loaded with::

    from motionbench.imputers.flow_matching import FlowMatchingImputer
    imp = FlowMatchingImputer.load(save_path)
    samples = imp.impute(x_obs, mask, n_samples=20)

Hydra config (alternative invocation)
--------------------------------------
See ``configs/methods/train_flow.yaml`` for a Hydra configuration file
that maps the same CLI flags via ``hydra.main``.

Notes
-----
* The script does **not** require a GPU; training falls back to CPU
  automatically.
* All random seeds are fixed via ``--seed`` (default 42).
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import Tensor

from motionbench.imputers.flow_matching import FlowMatchingImputer


# ---------------------------------------------------------------------------
# Minimal in-memory dataset wrapper
# ---------------------------------------------------------------------------


class _TensorDataset:
    """Wraps a ``(N, J, F, T)`` tensor as a :class:`~motionbench.data.base.BaseDataset`.

    This is the simplest possible dataset for training the flow imputer.
    For real-world data loaders see the ``motionbench.data`` sub-package.

    Args:
        x: ``(N, J, F, T)`` float32 tensor.
        labels: Optional ``(N,)`` int64 label tensor; zeros if omitted.
        skeleton: Skeleton identifier string (stored in metadata).
        frame_rate: Clip frame rate in Hz (stored in metadata).
    """

    def __init__(
        self,
        x: Tensor,
        labels: Tensor | None = None,
        skeleton: str = "generic",
        frame_rate: float = 30.0,
    ) -> None:
        assert x.dim() == 4, f"Expected (N, J, F, T), got {tuple(x.shape)}"
        self._x = x.float()
        self._y = (
            labels.long() if labels is not None else torch.zeros(len(x), dtype=torch.long)
        )
        self._meta: dict[str, object] = {
            "skeleton": skeleton,
            "frame_rate": frame_rate,
        }

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self._x[idx], self._y[idx]

    @property
    def shape(self) -> tuple[int, int, int]:
        _, J, F, T = self._x.shape
        return J, F, T

    @property
    def metadata(self) -> dict[str, object]:
        return self._meta

    @property
    def oracle(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def _load_tensor_dataset(path: Path, J: int, F: int, T: int) -> _TensorDataset:
    """Load a pre-saved ``.pt`` file containing an ``(N, J, F, T)`` tensor.

    Args:
        path: Path to the ``.pt`` file.
        J: Expected number of joints.
        F: Expected number of coordinates per joint.
        T: Expected number of frames.

    Returns:
        :class:`_TensorDataset` wrapping the loaded tensor.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the loaded tensor has unexpected shape.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    raw = torch.load(path, map_location="cpu")
    x: Tensor
    if isinstance(raw, dict):
        x = raw["x"]
    elif isinstance(raw, Tensor):
        x = raw
    else:
        raise ValueError(f"Expected Tensor or dict with key 'x'; got {type(raw)}")
    if x.dim() != 4:
        raise ValueError(f"Expected (N, J, F, T) tensor; got shape {tuple(x.shape)}")
    n, actual_j, actual_f, actual_t = x.shape
    if (actual_j, actual_f, actual_t) != (J, F, T):
        raise ValueError(
            f"Shape mismatch: file has (J={actual_j}, F={actual_f}, T={actual_t}) "
            f"but --J {J} --F {F} --T {T} were requested."
        )
    print(f"[data] Loaded {n} samples ({J}, {F}, {T}) from {path}")
    return _TensorDataset(x)


def _make_synthetic_dataset(
    n_samples: int, J: int, F: int, T: int, seed: int = 42
) -> _TensorDataset:
    """Generate a synthetic Gaussian motion dataset for debugging.

    Args:
        n_samples: Number of clips to generate.
        J: Number of joints.
        F: Number of coordinates per joint.
        T: Number of frames.
        seed: Random seed.

    Returns:
        :class:`_TensorDataset` with i.i.d. ``N(0, I)`` samples.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    x = torch.randn(n_samples, J, F, T, generator=rng)
    print(f"[data] Generated {n_samples} synthetic Gaussian samples ({J}, {F}, {T})")
    return _TensorDataset(x)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train FlowMatchingImputer and save checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    data_grp = p.add_mutually_exclusive_group(required=True)
    data_grp.add_argument(
        "--data_path", type=Path,
        help="Path to pre-saved (N, J, F, T) tensor (.pt file).",
    )
    data_grp.add_argument(
        "--synthetic", action="store_true",
        help="Generate synthetic Gaussian data for debugging.",
    )
    p.add_argument(
        "--n_synthetic", type=int, default=500,
        help="Number of synthetic samples (only used with --synthetic).",
    )
    # Shape
    p.add_argument("--J", type=int, required=True, help="Number of joints.")
    p.add_argument("--F", type=int, required=True, help="Coordinates per joint.")
    p.add_argument("--T", type=int, required=True, help="Frames per clip.")
    # Model
    p.add_argument("--hidden_dim", type=int, default=256, help="VelocityNet d_model.")
    p.add_argument("--num_steps", type=int, default=100, help="ODE integration steps.")
    p.add_argument(
        "--noise_init_scale", type=float, default=1.0,
        help=(
            "Gaussian source std. σ²_data for Burr-XII(c=2,k=2) > 1; "
            "increasing this (e.g. 2.0) can mitigate H2 regression (see module docstring)."
        ),
    )
    p.add_argument("--solver", type=str, default="midpoint", choices=["midpoint", "euler"])
    # Training
    p.add_argument("--n_epochs", type=int, default=200, help="Training epochs.")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    # Output
    p.add_argument(
        "--save_path", type=Path, required=True,
        help="Output checkpoint path (e.g. checkpoints/flow.pt).",
    )
    # Misc
    p.add_argument("--seed", type=int, default=42, help="Global random seed.")
    p.add_argument(
        "--log_every", type=int, default=10,
        help="Print loss every N epochs (0 = quiet).",
    )
    return p


def main() -> None:
    """Entry point: parse args, build dataset, train, save."""
    parser = _build_parser()
    args = parser.parse_args()

    _seed_all(args.seed)

    # --- Dataset -----------------------------------------------------------
    if args.synthetic:
        dataset = _make_synthetic_dataset(
            n_samples=args.n_synthetic,
            J=args.J, F=args.F, T=args.T,
            seed=args.seed,
        )
    else:
        dataset = _load_tensor_dataset(
            path=args.data_path, J=args.J, F=args.F, T=args.T
        )

    # --- Model -------------------------------------------------------------
    imputer = FlowMatchingImputer(
        J=args.J, F=args.F, T=args.T,
        hidden_dim=args.hidden_dim,
        num_steps=args.num_steps,
        noise_init_scale=args.noise_init_scale,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        solver=args.solver,
    )
    dev = imputer._device
    print(f"[train_flow] device={dev}  J={args.J} F={args.F} T={args.T}")
    print(f"[train_flow] hidden_dim={args.hidden_dim}  num_steps={args.num_steps}  "
          f"noise_init_scale={args.noise_init_scale}  solver={args.solver}")
    print(f"[train_flow] n_epochs={args.n_epochs}  batch_size={args.batch_size}  "
          f"lr={args.lr}")

    # --- Training ----------------------------------------------------------
    t_start = time.time()
    imputer.fit(dataset)
    elapsed = time.time() - t_start

    if args.log_every > 0:
        losses = imputer.train_losses
        for ep, loss in enumerate(losses):
            if ep % args.log_every == 0 or ep == len(losses) - 1:
                print(f"[train_flow] epoch {ep:4d}/{len(losses)}  loss={loss:.6f}")

    if len(imputer.train_losses) >= 2:
        ratio = imputer.train_losses[0] / max(imputer.train_losses[-1], 1e-12)
        print(
            f"[train_flow] Training complete in {elapsed:.1f}s — "
            f"initial loss={imputer.train_losses[0]:.4f}  "
            f"final loss={imputer.train_losses[-1]:.4f}  "
            f"ratio={ratio:.2f}x"
        )

    # --- Save --------------------------------------------------------------
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    imputer.save(save_path)
    print(f"[train_flow] Checkpoint saved to {save_path}")


if __name__ == "__main__":
    main()
