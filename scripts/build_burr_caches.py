"""scripts/build_burr_caches.py — build VAEAC/Flow caches for Burr datasets.

Produces caches at ``$CARE_PD_ROOT/cache/vaeac_synthetic/burr_m{5,10}_jft/``
matching the (N, T, J, F) layout that ``train_vaeac.py`` / ``train_flow_matching.py``
expect.

Generates 1000 sequences per dataset (850 train + 150 val) using a different seed
than the eval-time draws so the imputers do not see eval data.

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

from motionbench.data.synthetic.burr_motion import BurrMotionBenchmark


CARE_PD_ROOT = Path(os.environ.get("CARE_PD_ROOT", REPO.parent / "CARE-PD"))
CARE_PD_CACHE = CARE_PD_ROOT / "cache" / "vaeac_synthetic"


def _build_one(name: str, J: int, T: int, M: int, n_total: int = 1000) -> Path:
    """Build a (N, T, J, F=3) cache from BurrMotionBenchmark draws."""
    print(f"=== Building {name}: J={J}, T={T}, M={M}, n_total={n_total} ===")
    # Draw enough samples to cover train+val
    ds = BurrMotionBenchmark(J=J, F=3, T=T, N=n_total, rho=0.5, alpha=0.8, seed=99)
    Xs = []
    for i in range(len(ds)):
        x_i, _ = ds[i]
        Xs.append(x_i.numpy())
    X = np.stack(Xs, axis=0)                         # (N, J, F, T)
    # Transpose to (N, T, J, F) which is what the trainer expects.
    X = np.transpose(X, (0, 3, 1, 2)).astype(np.float32)
    n_train = int(0.85 * n_total)
    x1_train = X[:n_train]
    x1_val = X[n_train:]

    # Stats over training set (per-joint per-coord)
    stats_mean = x1_train.reshape(-1, J, 3).mean(axis=0)        # (J, 3)
    stats_std = x1_train.reshape(-1, J, 3).std(axis=0) + 1e-6   # (J, 3)

    # Masks (all valid)
    mask_train = np.ones((x1_train.shape[0], T), dtype=bool)
    mask_val = np.ones((x1_val.shape[0], T), dtype=bool)

    out_dir = CARE_PD_CACHE / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cache.npz"

    # Empty meta arrays so trainer's metadata-loading code doesn't crash
    n_tr, n_va = x1_train.shape[0], x1_val.shape[0]
    np.savez_compressed(
        out_path,
        x1_train=x1_train,
        mask_train=mask_train,
        x1_val=x1_val,
        mask_val=mask_val,
        stats_mean=stats_mean.astype(np.float32),
        stats_std=stats_std.astype(np.float32),
        seq_len=np.int32(T),
        fold=np.int32(0),
        num_folds=np.int32(1),
        clip_stride_train=np.int32(T),
        clip_stride_val=np.int32(T),
        dataset=name,
        data_type="synthetic",
        meta_pid_train=np.array([f"p{i}" for i in range(n_tr)], dtype=object),
        meta_walk_id_train=np.array([f"w{i}" for i in range(n_tr)], dtype=object),
        meta_seq_key_train=np.array([f"s{i}" for i in range(n_tr)], dtype=object),
        meta_clip_idx_train=np.zeros(n_tr, dtype=np.int32),
        meta_updrs_gait_train=np.zeros(n_tr, dtype=np.int32),
        meta_medication_train=np.array(["off"] * n_tr, dtype=object),
        meta_pid_val=np.array([f"p{i}" for i in range(n_va)], dtype=object),
        meta_walk_id_val=np.array([f"w{i}" for i in range(n_va)], dtype=object),
        meta_seq_key_val=np.array([f"s{i}" for i in range(n_va)], dtype=object),
        meta_clip_idx_val=np.zeros(n_va, dtype=np.int32),
        meta_updrs_gait_val=np.zeros(n_va, dtype=np.int32),
        meta_medication_val=np.array(["off"] * n_va, dtype=object),
    )
    print(f"  wrote {out_path}: x1_train={x1_train.shape}, x1_val={x1_val.shape}")
    return out_path


if __name__ == "__main__":
    # Both burr datasets have J=5, F=3, T=20.  K (window count) is a pipeline-time
    # parameter and does not affect the underlying generative process or imputer
    # training; we therefore train a single VAEAC/Flow per (J, T) combination.
    _build_one("burr_jft_t20_j5", J=5, T=20, M=5)
    print("Done.")
