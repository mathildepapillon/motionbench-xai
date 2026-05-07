#!/usr/bin/env python3
"""Build a .npz cache for PTB-XL imputer training.

Loads PTBXLDataset on training folds (1-8) and saves:
    x_train: (N, J=12, F=1, T=1000) float32

Usage:
    python scripts/build_ptbxl_cache.py --data_path "$PTBXL_DATA_ROOT"
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO))

from motionbench.data.real.ptbxl import PTBXLDataset, _FOLD_SPLITS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--max_sequences", type=int, default=None)
    ap.add_argument("--output", type=str,
                    default=str(REPO / "results" / "ptbxl_imputers" / "ptbxl_train_cache.npz"))
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading PTB-XL training data (folds 1-8) from {args.data_path} ...")
    ds = PTBXLDataset(
        data_path=args.data_path,
        split="train",
        normalize=True,
        max_sequences=args.max_sequences,
    )
    print(f"Loaded {len(ds)} records, shape {ds.shape}")

    # Stack into (N, J, F, T) array
    xs = np.stack([ds[i][0].numpy() for i in range(len(ds))], axis=0)  # (N, J, F, T)
    print(f"x_train shape: {xs.shape}  dtype: {xs.dtype}")

    np.savez_compressed(out_path, x_train=xs)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
