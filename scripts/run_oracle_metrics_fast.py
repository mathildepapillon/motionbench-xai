"""scripts/run_oracle_metrics_fast.py — evaluate oracle metrics using cached attributions.

For kernelshap_oracle cells that already have attributions.npz written (from a
high n_mc run), but are missing result.json, this script re-runs ONLY the
metric evaluation step using a lower metric_oracle_n_mc (default 50) so that
the reference oracle calls are much faster.

The attributions are loaded from cache; the fast metric evaluation then
computes EC1/EC2/EC3/spearman/kendall against a reference oracle at n_mc=50.

Usage::

    conda activate motionbench-xai
    # All datasets
    python scripts/run_oracle_metrics_fast.py

    # Specific datasets
    python scripts/run_oracle_metrics_fast.py \\
        --datasets skeleton_gait_combined skeleton_structured gait_periodic \\
        --metric-n-mc 50

    # Dry-run
    python scripts/run_oracle_metrics_fast.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results" / "synthetic"
METHOD = "kernelshap_oracle"

# Datasets with high J (slow oracle), targeting these by default
DEFAULT_DATASETS = [
    "skeleton_gait_combined",
    "skeleton_structured",
    "gait_periodic",
    "joint_subset_skeleton",
    "low_rank_manifold",
]
ALL_CLASSIFIERS = ["synthetic_mlp", "synthetic_cnn", "synthetic_transformer"]


def has_valid_result(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        d = json.loads(path.read_text())
        return any(k in d for k in ("ec1", "faithfulness_correlation", "player_aopc"))
    except Exception:
        return False


def has_valid_attributions(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = np.load(path, allow_pickle=True)
        keys = list(data.keys())
        return len(keys) > 0
    except Exception:
        return False


def build_cfg(metric_n_mc: int = 50, device: str = "cuda") -> object:
    """Load overnight config with fast metric evaluation settings."""
    from omegaconf import OmegaConf  # noqa: PLC0415

    overnight_cfg = REPO / "configs" / "experiments" / "overnight_synthetic_sweep.yaml"
    cfg = OmegaConf.load(overnight_cfg)
    overrides = {
        "metric_oracle_n_mc": metric_n_mc,
        "metric_oracle_n_coalitions": 64,  # use overnight default
        "device": device,
        "n_sequences": 50,
        "checkpoint_dir": str(REPO / "motionbench" / "classifiers" / "checkpoints" / "synthetic"),
    }
    return OmegaConf.merge(cfg, OmegaConf.create(overrides))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--classifiers", nargs="+", default=ALL_CLASSIFIERS)
    parser.add_argument("--metric-n-mc", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if result.json already exists")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.chdir(REPO)

    log.info("=== run_oracle_metrics_fast ===")
    log.info("metric_oracle_n_mc=%d  device=%s", args.metric_n_mc, args.device)
    log.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))

    # Plan which cells to run
    cells = []
    for ds in args.datasets:
        for clf in args.classifiers:
            result_path = RESULTS_DIR / ds / clf / METHOD / "result.json"
            attr_path = RESULTS_DIR / ds / clf / METHOD / "attributions.npz"

            if has_valid_result(result_path) and not args.force:
                log.info("[SKIP] %s/%s — result.json already valid", ds, clf)
                continue
            if not has_valid_attributions(attr_path):
                log.info("[SKIP] %s/%s — no valid attributions.npz (attr size=%s)",
                         ds, clf, attr_path.stat().st_size if attr_path.exists() else "missing")
                continue
            log.info("[TODO] %s/%s — attributions.npz=%d bytes, no valid result.json",
                     ds, clf, attr_path.stat().st_size)
            cells.append((ds, clf))

    if not cells:
        log.info("No cells to process.")
        return

    if args.dry_run:
        log.info("[DRY-RUN] %d cells planned. Exiting.", len(cells))
        return

    try:
        import torch
        device_str = args.device
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA not available — falling back to cpu")
            device_str = "cpu"
    except ImportError:
        device_str = "cpu"

    cfg = build_cfg(metric_n_mc=args.metric_n_mc, device=device_str)

    from motionbench.pipelines.synthetic_eval import _run_cell  # noqa: PLC0415

    summary = []
    t_total = time.time()

    for ds, clf in cells:
        result_path = RESULTS_DIR / ds / clf / METHOD / "result.json"

        # Remove invalid/stub result.json so _run_cell doesn't skip
        if result_path.exists() and not has_valid_result(result_path):
            log.info("[STUB] Removing invalid result.json for %s/%s/%s", ds, clf, METHOD)
            result_path.unlink()

        log.info("==> RUN %s/%s/%s (metric_n_mc=%d, device=%s)",
                 ds, clf, METHOD, args.metric_n_mc, device_str)
        t0 = time.time()
        try:
            result = _run_cell(ds, clf, METHOD, cfg)
            wall = time.time() - t0
            ec1 = result.get("ec1", float("nan"))
            sp = result.get("spearman", float("nan"))
            log.info("    DONE %s/%s in %.1fs  ec1=%.4f  spearman=%.3f  keys=%d",
                     ds, clf, wall, ec1, sp, len(result))
            summary.append({
                "dataset": ds, "classifier": clf,
                "ec1": ec1, "spearman": sp,
                "status": "ok" if "error" not in result else "error",
                "wall": wall,
            })
        except Exception as exc:
            wall = time.time() - t0
            log.error("[FAIL] %s/%s in %.1fs: %s", ds, clf, wall, exc)
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
        log.info("  %-30s  %-22s  ec1=%-8s  spearman=%-6s  %s",
                 row["dataset"], row["classifier"], ec1_str, sp_str, row["status"])

    n_ok = sum(1 for r in summary if r["status"] in ("ok",))
    log.info("Cells OK: %d / %d", n_ok, len(summary))


if __name__ == "__main__":
    main()
