"""scripts/validate_imputers.py — Validate VAEAC/Flow imputation realism.

Loads each available CARE-PD VAEAC/Flow checkpoint, applies it to held-out
test sequences with each player abstraction (temporal windows, spatial joints,
joint x phase), and reports:

  - Reconstruction MSE on hidden coordinates only
  - Sample std vs ground-truth std (manifold-realism proxy)
  - For Gaussian datasets: Oracle MSE as gold-standard reference

Run:
    conda activate motionbench-xai
    python scripts/validate_imputers.py --device cuda:0

Outputs:
    results/imputer_validation.md       (per-(dataset, imputer, player) table)
    results/imputer_validation.json     (full numerical results)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Datasets to validate (only those with available checkpoints)
# ---------------------------------------------------------------------------

DATASETS = {
    # Synthetic (T=16): need synthetic VAEAC/Flow ckpts
    "gaussian_k4": {
        "_target_": "motionbench.data.synthetic.gaussian_motion.GaussianMotionDataset",
        "kwargs": {"J": 5, "F": 3, "T": 16, "K": 4, "N": 100, "rho": 0.5, "alpha": 0.8, "seed": 42},
        "has_oracle": True,
    },
    "gaussian_k8": {
        "_target_": "motionbench.data.synthetic.gaussian_motion.GaussianMotionDataset",
        "kwargs": {"J": 5, "F": 3, "T": 16, "K": 8, "N": 100, "rho": 0.5, "alpha": 0.8, "seed": 42},
        "has_oracle": True,
    },
    "skeleton_structured": {
        "_target_": "motionbench.data.synthetic.skeleton_structured.SkeletonStructuredDataset",
        "kwargs": {"J": 17, "F": 3, "T": 16, "N": 100, "alpha_time": 0.9, "decay": 0.5, "seed": 42},
        "has_oracle": True,
    },
    "gait_periodic": {
        "_target_": "motionbench.data.synthetic.gait_periodic.GaitPeriodicDataset",
        "kwargs": {"J": 17, "F": 3, "T": 16, "N": 100, "seed": 42},
        "has_oracle": True,
    },
    # Real (T=80): we ship a separate CARE-PD cache adapter
    "bmclab_real": {
        "_target_": "motionbench.data.real.care_pd_cache.BMCLabCacheDataset",
        "kwargs": {
            "cache_path": str(
                Path(os.environ.get("CARE_PD_ROOT",
                                    Path(__file__).resolve().parent.parent.parent / "CARE-PD"))
                / "cache" / "flow_matching" / "BMCLab_h36m_80_fold1" / "cache.npz"
            ),
            "split": "val",
            "max_sequences": 50,
        },
        "has_oracle": False,
    },
}

# ---------------------------------------------------------------------------
# Player abstractions to test
# ---------------------------------------------------------------------------


def _make_player_masks(J: int, F: int, T: int, K: int = 4) -> dict[str, Tensor]:
    """Return a dict of mask Tensors of shape (J, F, T) with True = observed.

    Each mask hides ~half of the coordinates following one player abstraction:
      - temporal_half: hide K/2 contiguous time windows
      - spatial_half:  hide J/2 joints
      - random_half:   hide 50% of (j, f, t) coordinates uniformly at random
    """
    masks = {}

    # Temporal half: hide first K/2 windows
    ws = T // K
    m_temp = torch.ones(J, F, T, dtype=torch.bool)
    m_temp[:, :, : (K // 2) * ws] = False
    masks["temporal_half"] = m_temp

    # Spatial half: hide first J/2 joints
    m_spat = torch.ones(J, F, T, dtype=torch.bool)
    m_spat[: J // 2, :, :] = False
    masks["spatial_half"] = m_spat

    # Random half (joint x phase ~analog: each (j, t) cell uniformly random)
    rng = np.random.default_rng(0)
    flat = rng.random((J, T)) > 0.5  # (J, T) bool, True = observed
    m_rand = torch.from_numpy(flat).unsqueeze(1).expand(J, F, T).contiguous()
    masks["random_half_jt"] = m_rand

    return masks


# ---------------------------------------------------------------------------
# Imputers to test (per dataset)
# ---------------------------------------------------------------------------


def _build_imputer(name: str, dataset: Any, device: str) -> Any:
    """Build and fit an imputer by name."""
    from motionbench.imputers.carepd_imputer import CarepdFlowImputer, CarepdVAEACImputer
    from motionbench.imputers.off_manifold import ZeroImputer

    if name == "vaeac":
        imp = CarepdVAEACImputer(n_completion_samples=5, device=device)
    elif name == "flow":
        imp = CarepdFlowImputer(num_steps=50, device=device)
    elif name == "zero":
        imp = ZeroImputer()
    elif name == "oracle":
        return dataset.oracle if hasattr(dataset, "oracle") else None
    else:
        raise ValueError(f"Unknown imputer {name}")
    imp.fit(dataset)
    if getattr(imp, "_skip", False):
        return None
    return imp


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------


def _impute(imputer: Any, x: Tensor, mask: Tensor, n_samples: int = 5) -> Tensor:
    """Call .impute or .conditional_sample uniformly. Returns (n, J, F, T)."""
    if hasattr(imputer, "impute"):
        return imputer.impute(x, mask, n_samples=n_samples)
    if hasattr(imputer, "conditional_sample"):
        return imputer.conditional_sample(x, mask, n=n_samples)
    raise TypeError(f"Imputer {type(imputer).__name__} has no impute / conditional_sample method")


def _validate_one(
    dataset_name: str,
    dataset: Any,
    imputer_name: str,
    imputer: Any,
    masks: dict[str, Tensor],
    n_seqs: int = 20,
    n_samples: int = 5,
) -> dict[str, dict[str, float]]:
    """Return per-mask MSE / std stats."""
    out: dict[str, dict[str, float]] = {}
    for mask_name, mask in masks.items():
        mse_hidden_list = []
        std_hidden_list = []
        std_truth_list = []
        for i in range(min(n_seqs, len(dataset))):
            x_true, _ = dataset[i]
            try:
                samples = _impute(imputer, x_true, mask, n_samples=n_samples)
            except Exception as exc:
                print(f"    !! impute failed: {exc!r}")
                continue
            samples_np = samples.detach().cpu().numpy()
            x_np = x_true.detach().cpu().numpy()
            m_np = mask.detach().cpu().numpy()
            hidden = ~m_np                                 # bool
            if hidden.sum() == 0:
                continue
            err = (samples_np - x_np[None]) ** 2           # (n, J, F, T)
            mse_hidden_list.append(err[:, hidden].mean())
            std_hidden_list.append(samples_np[:, hidden].std())
            std_truth_list.append(x_np[hidden].std())
        if not mse_hidden_list:
            out[mask_name] = {"mse_hidden": float("nan"), "std_imp": float("nan"), "std_true": float("nan")}
        else:
            out[mask_name] = {
                "mse_hidden": float(np.mean(mse_hidden_list)),
                "std_imp": float(np.mean(std_hidden_list)),
                "std_true": float(np.mean(std_truth_list)),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-seqs", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    args.out_dir.mkdir(parents=True, exist_ok=True)
    full: dict[str, dict[str, dict[str, dict[str, float]]]] = {}

    ds_list = args.datasets or list(DATASETS)

    for ds_name in ds_list:
        if ds_name not in DATASETS:
            print(f"Unknown dataset: {ds_name}, skipping")
            continue
        info = DATASETS[ds_name]
        print(f"\n{'='*70}\nDataset: {ds_name}\n{'='*70}")
        cfg = OmegaConf.create({"_target_": info["_target_"], **info["kwargs"]})
        try:
            dataset = instantiate(cfg)
        except Exception as exc:
            print(f"  !! failed to instantiate: {exc}")
            continue

        J, F, T = dataset.shape
        K = info["kwargs"].get("K", 4)
        masks = _make_player_masks(J, F, T, K=K)

        ds_out: dict[str, dict[str, dict[str, float]]] = {}
        for imp_name in ["zero", "oracle", "vaeac", "flow"]:
            print(f"  Imputer: {imp_name}")
            imp = _build_imputer(imp_name, dataset, args.device)
            if imp is None:
                print(f"    (skipped — no checkpoint)")
                continue
            results = _validate_one(
                ds_name, dataset, imp_name, imp, masks,
                n_seqs=args.n_seqs, n_samples=args.n_samples,
            )
            ds_out[imp_name] = results
            for mask_name, stats in results.items():
                print(f"    {mask_name:18s}  MSE={stats['mse_hidden']:.4f}  "
                      f"std_imp={stats['std_imp']:.3f}  std_true={stats['std_true']:.3f}")
        full[ds_name] = ds_out

    json_path = args.out_dir / "imputer_validation.json"
    json_path.write_text(json.dumps(full, indent=2))
    print(f"\nFull JSON: {json_path}")

    md_path = args.out_dir / "imputer_validation.md"
    lines = ["# Imputer Validation Report\n",
             "MSE on hidden coordinates only (lower is better). std_imp/std_true compares imputed-sample dispersion to ground-truth dispersion (closer is better).\n"]
    for ds, ds_out in full.items():
        lines.append(f"\n## {ds}\n")
        lines.append("| Mask | Imputer | MSE↓ | std_imp | std_true |")
        lines.append("|------|---------|------|---------|----------|")
        for mask_name in ["temporal_half", "spatial_half", "random_half_jt"]:
            for imp_name in ["zero", "oracle", "vaeac", "flow"]:
                if imp_name not in ds_out:
                    continue
                r = ds_out[imp_name].get(mask_name)
                if r is None:
                    continue
                lines.append(
                    f"| {mask_name} | {imp_name} | {r['mse_hidden']:.4f} "
                    f"| {r['std_imp']:.3f} | {r['std_true']:.3f} |"
                )
    md_path.write_text("\n".join(lines))
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
