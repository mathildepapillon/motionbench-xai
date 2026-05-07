"""scripts/build_synthetic_imputer_caches.py — build VAEAC/Flow training caches
for every synthetic dataset in motionbench-xai.

Generic re-implementation of the dataset-specific cache builders that previously
lived in the CARE-PD codebase.  Caches are written to
``$CARE_PD_ROOT/cache/vaeac_synthetic/<cache_name>/cache.npz`` in the
``(N, T, J, F)`` layout that ``train_vaeac.py`` and ``train_flow_matching.py``
expect.

Usage::

    python scripts/build_synthetic_imputer_caches.py
    python scripts/build_synthetic_imputer_caches.py --datasets gaussian_k4 burr_m5
    python scripts/build_synthetic_imputer_caches.py --care_pd_root /custom/CARE-PD

Each cache contains a different generative seed than the eval-time seeds so the
imputer never sees evaluation data.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# (cache_name, dataset_module, dataset_class, dataset_kwargs)
DATASET_SPECS: list[tuple[str, str, str, dict]] = [
    (
        "gaussian_k4_t16",
        "motionbench.data.synthetic.gaussian_motion",
        "GaussianMotionDataset",
        {"J": 5, "F": 3, "T": 16, "K": 4, "N": 1000, "rho": 0.5, "alpha": 0.8, "seed": 99},
    ),
    (
        "gaussian_k8_t16",
        "motionbench.data.synthetic.gaussian_motion",
        "GaussianMotionDataset",
        {"J": 5, "F": 3, "T": 16, "K": 8, "N": 1000, "rho": 0.5, "alpha": 0.8, "seed": 99},
    ),
    (
        "skeleton_t16",
        "motionbench.data.synthetic.skeleton_structured",
        "SkeletonStructuredDataset",
        {"J": 17, "F": 3, "T": 16, "K": 4, "N": 1000, "alpha": 0.8, "seed": 99},
    ),
    (
        "gait_t16",
        "motionbench.data.synthetic.gait_periodic",
        "GaitPeriodicDataset",
        {"J": 17, "F": 3, "T": 16, "K": 4, "N": 1000, "period": 16, "seed": 99},
    ),
    (
        "joint_subset_skel_t16",
        "motionbench.data.synthetic.joint_subset_skeleton",
        "JointSubsetSkeletonDataset",
        {"J": 17, "F": 3, "T": 16, "K": 4, "N": 1000, "alpha": 0.8, "seed": 99},
    ),
    (
        "low_rank_manifold_t16",
        "motionbench.data.synthetic.low_rank_manifold",
        "LowRankManifoldDataset",
        {"J": 17, "F": 3, "T": 16, "K": 4, "N": 1000, "rank": 4, "alpha": 0.8, "seed": 99},
    ),
    (
        "burr_jft_t20_j5",
        "motionbench.data.synthetic.burr_motion",
        "BurrMotionBenchmark",
        {"J": 5, "F": 3, "T": 20, "N": 1000, "rho": 0.5, "alpha": 0.8, "seed": 99},
    ),
]


def _instantiate(module: str, cls: str, kwargs: dict):
    import importlib
    mod = importlib.import_module(module)
    klass = getattr(mod, cls)
    sig_kwargs = {k: v for k, v in kwargs.items() if k != "K"}  # K is pipeline-only
    return klass(**sig_kwargs)


def build_cache(name: str, module: str, cls: str, kwargs: dict, care_pd_root: Path,
                force: bool = False) -> Path:
    out_dir = care_pd_root / "cache" / "vaeac_synthetic" / name
    out_path = out_dir / "cache.npz"
    if out_path.exists() and not force:
        print(f"  [SKIP] {name} (cached)")
        return out_path

    print(f"  [BUILD] {name}: {cls}({kwargs})")
    ds = _instantiate(module, cls, kwargs)
    Xs = []
    for i in range(len(ds)):
        x_i, _ = ds[i]
        Xs.append(x_i.numpy() if hasattr(x_i, "numpy") else np.asarray(x_i))
    X = np.stack(Xs, axis=0)                     # (N, J, F, T)
    if X.ndim != 4:
        raise ValueError(f"unexpected X ndim={X.ndim} for {name}")
    X = np.transpose(X, (0, 3, 1, 2)).astype(np.float32)  # (N, T, J, F)
    N, T, J, F = X.shape

    n_train = int(0.85 * N)
    x1_train, x1_val = X[:n_train], X[n_train:]
    stats_mean = x1_train.reshape(-1, J, F).mean(axis=0)
    stats_std = x1_train.reshape(-1, J, F).std(axis=0) + 1e-6

    out_dir.mkdir(parents=True, exist_ok=True)
    n_tr, n_va = x1_train.shape[0], x1_val.shape[0]
    np.savez_compressed(
        out_path,
        x1_train=x1_train,
        mask_train=np.ones((n_tr, T), dtype=bool),
        x1_val=x1_val,
        mask_val=np.ones((n_va, T), dtype=bool),
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
    print(f"    wrote {out_path}: train={x1_train.shape}, val={x1_val.shape}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Subset of cache names to build; default is all.")
    _default_care_pd = Path(__file__).resolve().parent.parent.parent / "CARE-PD"
    parser.add_argument("--care_pd_root", type=Path,
                        default=Path(os.environ.get("CARE_PD_ROOT",
                                                    str(_default_care_pd))))
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if cache exists.")
    args = parser.parse_args()

    print(f"CARE_PD_ROOT = {args.care_pd_root}")
    args.care_pd_root.mkdir(parents=True, exist_ok=True)

    selected = args.datasets if args.datasets else [s[0] for s in DATASET_SPECS]
    for name, module, cls, kwargs in DATASET_SPECS:
        if name not in selected:
            continue
        try:
            build_cache(name, module, cls, kwargs, args.care_pd_root, force=args.force)
        except Exception as exc:
            print(f"    [FAIL] {name}: {exc}")
    print("Done.")


if __name__ == "__main__":
    main()
