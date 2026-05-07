"""scripts/build_skeleton_gait_cache.py — build VAEAC/Flow cache for skeleton_gait_combined.

Produces a (N, T, J, F) cache.npz at
  ``$CARE_PD_ROOT/cache/vaeac_synthetic/skeleton_gait_combined/cache.npz``

Generates 1000 sequences (850 train + 150 val) using a different seed from
the eval-time draws so the imputers do not see eval data.

Environment variables:
    CARE_PD_ROOT: Root of the CARE-PD codebase (used for cache output).
        Defaults to the sibling directory of this repo.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from motionbench.data.synthetic.skeleton_gait_combined import SkeletonGaitDataset


CARE_PD_ROOT = Path(os.environ.get("CARE_PD_ROOT", REPO.parent / "CARE-PD"))
CARE_PD_CACHE = CARE_PD_ROOT / "cache" / "vaeac_synthetic"


def _build(name: str, J: int = 17, T: int = 16, n_total: int = 1000) -> Path:
    print(f"=== Building {name}: J={J}, T={T}, n_total={n_total} ===")
    ds = SkeletonGaitDataset(
        J=J, F=3, T=T, N=n_total,
        decay=0.5, period_mean=7.0, period_std=1.0, n_harmonics=3,
        n_classes=3, seed=2026,
    )
    Xs = []
    for i in range(len(ds)):
        x_i, _ = ds[i]
        Xs.append(x_i.numpy())
    X = np.stack(Xs, axis=0)                              # (N, J, F, T)
    X = np.transpose(X, (0, 3, 1, 2)).astype(np.float32)  # (N, T, J, F)

    n_train = int(0.85 * n_total)
    x1_train = X[:n_train]
    x1_val = X[n_train:]

    mask_train = np.ones((x1_train.shape[0], T), dtype=bool)
    mask_val = np.ones((x1_val.shape[0], T), dtype=bool)

    out_dir = CARE_PD_CACHE / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cache.npz"
    np.savez_compressed(
        out_path,
        x1_train=x1_train,
        mask_train=mask_train,
        x1_val=x1_val,
        mask_val=mask_val,
    )
    print(f"  wrote {out_path}: x1_train={x1_train.shape}, x1_val={x1_val.shape}")
    return out_path


if __name__ == "__main__":
    _build("skeleton_gait_combined", J=17, T=16, n_total=1000)
    print("Done.")
