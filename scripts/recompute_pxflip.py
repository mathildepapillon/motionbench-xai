"""scripts/recompute_pxflip.py — Targeted recompute of pixel_flipping_auc.

The original sweep produced pixel_flipping_auc = 1.000 for all methods due to
a bug where quantus.PixelFlipping defaulted to softmax=True, which always
returns 1.0 when the model wrapper outputs a single scalar (B, 1).

This script:
1. Finds all result.json files where pixel_flipping_auc ≈ 1.000.
2. Re-runs only the pixel_flipping metric in a temp directory (so the skip-on-
   existing-result logic does not fire on the original path).
3. Patches the original result.json in place with the corrected value.
4. Prints a summary of before/after values.

Usage::

    cd "$REPO_ROOT"
    conda run -n motionbench-xai python scripts/recompute_pxflip.py

    # Dry-run (inspect which cells would be recomputed without running them):
    conda run -n motionbench python scripts/recompute_pxflip.py --dry-run

    # Recompute even cells whose existing value is not ~1.0 (force full redo):
    conda run -n motionbench python scripts/recompute_pxflip.py --force
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path so imports work when run from scripts/
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))


def _is_saturated(result: dict) -> bool:
    """Return True when pixel_flipping_auc is missing, NaN, or rounds to 1.0."""
    val = result.get("pixel_flipping_auc")
    if val is None:
        return True
    try:
        fval = float(val)
        return np.isnan(fval) or fval >= 0.999
    except (TypeError, ValueError):
        return True


def _load_base_cfg() -> OmegaConf:
    """Load overnight_synthetic_sweep config with absolute paths resolved."""
    cfg_path = _REPO_ROOT / "configs" / "experiments" / "overnight_synthetic_sweep.yaml"
    cfg = OmegaConf.load(cfg_path)
    # Resolve relative results_dir to an absolute path so temp dirs work correctly.
    if not Path(str(cfg.results_dir)).is_absolute():
        OmegaConf.update(cfg, "results_dir", str(_REPO_ROOT / cfg.results_dir))
    return cfg


def main(dry_run: bool = False, force: bool = False) -> None:
    base_cfg = _load_base_cfg()
    results_root = Path(base_cfg.results_dir)

    result_files = sorted(results_root.glob("*/*/*/result.json"))
    print(f"Found {len(result_files)} result.json files under {results_root}")

    to_recompute: list[tuple[str, str, str, Path]] = []
    for rf in result_files:
        parts = rf.parts
        # Structure: results_root / dataset / classifier / method / result.json
        method = parts[-2]
        clf = parts[-3]
        dataset = parts[-4]
        result = json.loads(rf.read_text())
        if force or _is_saturated(result):
            to_recompute.append((dataset, clf, method, rf))

    print(f"{len(to_recompute)} cells need pixel_flipping_auc recompute"
          f" (force={force}, dry_run={dry_run})")

    if dry_run:
        for dataset, clf, method, rf in to_recompute:
            val = json.loads(rf.read_text()).get("pixel_flipping_auc")
            val_str = f"{float(val):.4f}" if val is not None else "MISSING"
            print(f"  WOULD recompute  {dataset}/{clf}/{method}  (current={val_str})")
        return

    # Lazy import — only needed when actually running cells.
    from motionbench.pipelines.synthetic_eval import _run_cell  # noqa: PLC0415

    patched = 0
    errors = 0
    for dataset, clf, method, original_path in to_recompute:
        old_val = json.loads(original_path.read_text()).get("pixel_flipping_auc", float("nan"))
        print(f"Recomputing {dataset}/{clf}/{method}  (old={old_val:.4f}) ...", end=" ", flush=True)

        tmpdir = tempfile.mkdtemp(prefix="pxflip_recompute_")
        try:
            # Build a patched config: only pixel_flipping metric, temp results_dir.
            patch = {
                "results_dir": tmpdir,
                "metrics": {
                    "gt": [],
                    "fidelity": ["pixel_flipping"],
                    "stability": [],
                    "sanity": [],
                },
            }
            cfg_patch = OmegaConf.merge(base_cfg, OmegaConf.create(patch))

            cell_result = _run_cell(dataset, clf, method, cfg_patch)

            new_val = cell_result.get("pixel_flipping_auc", float("nan"))
            print(f"new={new_val:.4f}")

            # Patch original result.json in place (merge, not overwrite).
            original_result = json.loads(original_path.read_text())
            original_result["pixel_flipping_auc"] = new_val
            original_path.write_text(json.dumps(original_result, indent=2))
            patched += 1

        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}")
            errors += 1
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\nDone. Patched {patched} cells, {errors} errors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which cells would be recomputed without running them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute all cells regardless of current value.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run, force=args.force)
