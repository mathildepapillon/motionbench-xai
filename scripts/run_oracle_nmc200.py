"""scripts/run_oracle_nmc200.py — Re-run kernelshap_oracle at n_mc=200.

Targets datasets whose oracle result.json was previously missing or was a stub
(≤4 keys).  Uses the pipeline's _run_cell which automatically loads cached
attributions.npz when available (fast path: only metric evaluation is re-run).

GPU assignments (set via CUDA_VISIBLE_DEVICES before launching):
  GPU 2:  gaussian_k4/synthetic_transformer
          gaussian_k8/synthetic_transformer
  GPU 5:  skeleton_structured × {mlp, cnn, transformer}
          gait_periodic × {mlp, cnn, transformer}
  GPU 6:  joint_subset_skeleton × {mlp, cnn, transformer}
          low_rank_manifold × {mlp, cnn, transformer}

Usage::

    conda activate motionbench-xai
    # GPU 2 batch
    CUDA_VISIBLE_DEVICES=2 python scripts/run_oracle_nmc200.py --gpu-batch gpu2

    # GPU 5 batch
    CUDA_VISIBLE_DEVICES=5 python scripts/run_oracle_nmc200.py --gpu-batch gpu5

    # GPU 6 batch
    CUDA_VISIBLE_DEVICES=6 python scripts/run_oracle_nmc200.py --gpu-batch gpu6

    # or run specific datasets
    CUDA_VISIBLE_DEVICES=2 python scripts/run_oracle_nmc200.py \\
        --datasets gaussian_k4 gaussian_k8 --classifiers synthetic_transformer

    # dry-run to preview which cells will be processed
    python scripts/run_oracle_nmc200.py --gpu-batch gpu5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results" / "synthetic"
OVERNIGHT_CFG = REPO / "configs" / "experiments" / "overnight_synthetic_sweep.yaml"

# ---------------------------------------------------------------------------
# GPU batch definitions
# ---------------------------------------------------------------------------

GPU_BATCHES: dict[str, list[tuple[str, str]]] = {
    "gpu2": [
        # skeleton_gait_combined already done (24 keys); only transformer missing
        ("gaussian_k4", "synthetic_transformer"),
        ("gaussian_k8", "synthetic_transformer"),
    ],
    "gpu5": [
        ("skeleton_structured", "synthetic_mlp"),
        ("skeleton_structured", "synthetic_cnn"),
        ("skeleton_structured", "synthetic_transformer"),
        ("gait_periodic", "synthetic_mlp"),
        ("gait_periodic", "synthetic_cnn"),
        ("gait_periodic", "synthetic_transformer"),
    ],
    "gpu6": [
        ("joint_subset_skeleton", "synthetic_mlp"),
        ("joint_subset_skeleton", "synthetic_cnn"),
        ("joint_subset_skeleton", "synthetic_transformer"),
        ("low_rank_manifold", "synthetic_mlp"),
        ("low_rank_manifold", "synthetic_cnn"),
        ("low_rank_manifold", "synthetic_transformer"),
    ],
}

METHOD = "kernelshap_oracle"
# Number of keys a non-stub result.json must have (stubs have exactly 4)
MIN_KEYS_FOR_VALID = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_valid_result(result_path: Path) -> bool:
    """Return True if result.json exists and has more than MIN_KEYS_FOR_VALID keys."""
    if not result_path.exists():
        return False
    try:
        data = json.loads(result_path.read_text())
        return len(data) > MIN_KEYS_FOR_VALID
    except Exception:
        return False


def build_cfg(n_mc: int = 200, device: str = "cuda"):
    """Load overnight YAML and override oracle n_mc and device."""
    from omegaconf import OmegaConf  # noqa: PLC0415

    cfg = OmegaConf.load(OVERNIGHT_CFG)
    # Override critical fields
    overrides = OmegaConf.create({
        "metric_oracle_n_mc": n_mc,
        "device": device,
        # Keep n_sequences at 50 (same as overnight)
        "n_sequences": 50,
        # Absolute path so _run_cell can resolve checkpoints
        "checkpoint_dir": str(
            REPO / "motionbench" / "classifiers" / "checkpoints" / "synthetic"
        ),
        "results_dir": str(RESULTS_DIR),
        # Disable wandb
        "wandb": {"mode": "disabled"},
    })
    cfg = OmegaConf.merge(cfg, overrides)
    return cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    batch_group = p.add_mutually_exclusive_group()
    batch_group.add_argument(
        "--gpu-batch", choices=["gpu2", "gpu5", "gpu6"],
        help="Pre-defined GPU batch to run.",
    )
    batch_group.add_argument(
        "--datasets", nargs="+",
        help="Explicit dataset list (used with --classifiers).",
    )
    p.add_argument(
        "--classifiers", nargs="+",
        default=["synthetic_mlp", "synthetic_cnn", "synthetic_transformer"],
        help="Classifiers to run (with --datasets).",
    )
    p.add_argument(
        "--n-mc", type=int, default=200,
        help="Oracle MC samples (default: 200).",
    )
    p.add_argument(
        "--device", default="cuda",
        help="Torch device (default: cuda; falls back to cpu if unavailable).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-run even if a valid (>4-key) result.json already exists.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned cells without running them.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Build cell list
    if args.gpu_batch:
        cells = GPU_BATCHES[args.gpu_batch]
    elif args.datasets:
        cells = [(ds, clf) for ds in args.datasets for clf in args.classifiers]
    else:
        # No filter: run all pre-defined cells across all batches
        cells = [c for batch in GPU_BATCHES.values() for c in batch]

    log.info("Repo root: %s", REPO)
    log.info("Results dir: %s", RESULTS_DIR)
    log.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    log.info("Planned %d cells (method=%s, n_mc=%d):", len(cells), METHOD, args.n_mc)
    for ds, clf in cells:
        rp = RESULTS_DIR / ds / clf / METHOD / "result.json"
        status = "DONE" if is_valid_result(rp) else "MISSING"
        log.info("  %-35s  %-22s  [%s]", ds, clf, status)

    if args.dry_run:
        log.info("[DRY-RUN] Exiting without running any cells.")
        return

    # Build OmegaConf cfg (with cwd set to repo root for _load_sub_config)
    import os as _os
    _os.chdir(REPO)

    try:
        import torch
        device_str = args.device
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA not available — falling back to cpu")
            device_str = "cpu"
    except ImportError:
        device_str = "cpu"

    cfg = build_cfg(n_mc=args.n_mc, device=device_str)

    from motionbench.pipelines.synthetic_eval import _run_cell  # noqa: PLC0415

    t_total = time.time()
    summary: list[dict] = []

    for ds, clf in cells:
        rp = RESULTS_DIR / ds / clf / METHOD / "result.json"

        # Skip valid non-stub results unless --force
        if not args.force and is_valid_result(rp):
            data = json.loads(rp.read_text())
            log.info(
                "[SKIP] %s/%s/%s — already valid (%d keys, ec1=%.4f)",
                ds, clf, METHOD, len(data), data.get("ec1", float("nan")),
            )
            summary.append({
                "dataset": ds, "classifier": clf,
                "ec1": data.get("ec1"), "spearman": data.get("spearman"),
                "status": "cached",
            })
            continue

        # Delete stub/invalid result.json so _run_cell doesn't skip
        if rp.exists():
            log.info("[STUB] Removing invalid result.json for %s/%s/%s", ds, clf, METHOD)
            rp.unlink()

        log.info("==> RUN %s/%s/%s (n_mc=%d, device=%s)", ds, clf, METHOD, args.n_mc, device_str)
        t0 = time.time()
        try:
            result = _run_cell(ds, clf, METHOD, cfg)
            wall = time.time() - t0
            ec1 = result.get("ec1", float("nan"))
            sp = result.get("spearman", float("nan"))
            log.info(
                "    DONE %s/%s/%s in %.1fs  ec1=%.4f  spearman=%.3f  keys=%d",
                ds, clf, METHOD, wall, ec1, sp, len(result),
            )
            summary.append({
                "dataset": ds, "classifier": clf,
                "ec1": ec1, "spearman": sp,
                "status": "ok" if "error" not in result else "error",
                "wall": wall,
            })
        except Exception as exc:
            wall = time.time() - t0
            log.error("[FAIL] %s/%s/%s in %.1fs: %s", ds, clf, METHOD, wall, exc)
            import traceback
            traceback.print_exc()
            summary.append({
                "dataset": ds, "classifier": clf,
                "ec1": None, "spearman": None,
                "status": f"error:{type(exc).__name__}",
                "wall": wall,
            })

    log.info("Total wall-clock: %.1fs", time.time() - t_total)
    log.info("=== Summary ===")
    for row in summary:
        ec1_str = f"{row['ec1']:.4f}" if row["ec1"] is not None else "  N/A"
        sp_str = f"{row['spearman']:.3f}" if row.get("spearman") is not None else "  N/A"
        log.info(
            "  %-35s  %-22s  ec1=%-8s  spearman=%-6s  %s",
            row["dataset"], row["classifier"], ec1_str, sp_str, row["status"],
        )

    n_ok = sum(1 for r in summary if r["status"] in ("ok", "cached"))
    log.info("Cells OK: %d / %d", n_ok, len(summary))
    sys.exit(0 if n_ok == len(summary) else 1)


if __name__ == "__main__":
    main()
