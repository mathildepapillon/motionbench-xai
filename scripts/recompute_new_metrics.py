"""scripts/recompute_new_metrics.py — Compute player_deletion and faith_gap metrics.

Adds two new metrics to all existing result.json files without re-running the
full attribution pipeline:

* ``player_aopc_comp`` / ``player_aopc_suff`` — player-level AOPC from
  :class:`~motionbench.metrics.fidelity.PlayerDeletionMetric`.
* ``faith_gap`` / ``faith_off`` / ``faith_on`` — manifold fidelity gap from
  :class:`~motionbench.metrics.fidelity.ManifoldFidelityGapMetric`
  (synthetic-dataset cells only; skipped when oracle is not available).

Strategy
--------
For each existing result.json that is missing the new metrics:

1. Load the base config, patch ``metrics`` to only include the new metrics and
   ``results_dir`` to a temp directory (so the skip-on-existing check doesn't
   fire on the original).
2. Call ``_run_cell(dataset, clf, method, cfg_patch)``.  The cell will re-run
   attribution (needed to get φ) but skip all previously-computed metrics.
3. Extract the new metric values from the temp result.
4. Merge them into the original result.json (non-destructive patch).

Usage::

    cd "$REPO_ROOT"
    conda run -n motionbench-xai python scripts/recompute_new_metrics.py

    # Dry-run:
    conda run -n motionbench python scripts/recompute_new_metrics.py --dry-run

    # Only one metric group (player_deletion or faith_gap):
    conda run -n motionbench python scripts/recompute_new_metrics.py --metrics player_deletion
    conda run -n motionbench python scripts/recompute_new_metrics.py --metrics faith_gap

    # Force recompute even if metrics are already present:
    conda run -n motionbench python scripts/recompute_new_metrics.py --force

Notes
-----
* faith_gap requires an oracle → automatically skipped for real-world datasets
  (the metric's ``requires_oracle=True`` guard in ``_evaluate_metrics`` handles this).
* n_jobs can be set via ``--n-jobs`` (defaults to config value).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# New metric keys produced by each group.
_METRIC_KEYS: dict[str, list[str]] = {
    "player_deletion": ["player_aopc_comp", "player_aopc_suff"],
    "faith_gap": ["faith_gap", "faith_off", "faith_on"],
    "cgfs": [
        "cgfs_aopc_joint",
        "cgfs_aopc_phase4",
        "cgfs_aopc_joint_phase4",
        "cgfs_sigma",
        "cgfs_range",
    ],
}


def _load_base_cfg() -> OmegaConf:
    cfg_path = _REPO_ROOT / "configs" / "experiments" / "overnight_synthetic_sweep.yaml"
    cfg = OmegaConf.load(cfg_path)
    if not Path(str(cfg.results_dir)).is_absolute():
        OmegaConf.update(cfg, "results_dir", str(_REPO_ROOT / cfg.results_dir))
    return cfg


def _needs_recompute(result: dict, metric_group: str, force: bool) -> bool:
    """Return True if any key from the metric group is missing."""
    if force:
        return True
    return any(k not in result for k in _METRIC_KEYS[metric_group])


def _run_group(
    dataset: str,
    clf: str,
    method: str,
    base_cfg: OmegaConf,
    metric_group: str,
) -> dict:
    """Run a single cell with only the specified metric group; return result dict."""
    from motionbench.pipelines.synthetic_eval import _run_cell  # noqa: PLC0415

    tmpdir = tempfile.mkdtemp(prefix=f"newmetric_{metric_group}_")
    try:
        fidelity_metrics = ["player_deletion"] if metric_group == "player_deletion" else []
        manifold_metrics: list[str] = []
        if metric_group == "faith_gap":
            manifold_metrics = ["faith_gap"]
        elif metric_group == "cgfs":
            manifold_metrics = ["cgfs"]

        patch = {
            "results_dir": tmpdir,
            "metrics": {
                "gt": [],
                "fidelity": fidelity_metrics,
                "manifold": manifold_metrics,
                "stability": [],
                "sanity": [],
            },
        }
        cfg_patch = OmegaConf.merge(base_cfg, OmegaConf.create(patch))
        return _run_cell(dataset, clf, method, cfg_patch)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main(
    dry_run: bool = False,
    force: bool = False,
    metric_groups: list[str] | None = None,
    n_jobs: int | None = None,
) -> None:
    groups = metric_groups or list(_METRIC_KEYS.keys())
    base_cfg = _load_base_cfg()
    results_root = Path(base_cfg.results_dir)

    result_files = sorted(results_root.glob("*/*/*/result.json"))
    print(f"Found {len(result_files)} result.json files under {results_root}")

    # Build work list: (dataset, clf, method, original_path, group)
    work: list[tuple[str, str, str, Path, str]] = []
    for rf in result_files:
        parts = rf.parts
        method = parts[-2]
        clf = parts[-3]
        dataset = parts[-4]
        result = json.loads(rf.read_text())
        for group in groups:
            if _needs_recompute(result, group, force):
                work.append((dataset, clf, method, rf, group))

    print(f"{len(work)} (cell, metric_group) pairs to compute "
          f"(force={force}, dry_run={dry_run}, groups={groups})")

    if dry_run:
        for dataset, clf, method, rf, group in work:
            print(f"  WOULD compute  {dataset}/{clf}/{method}  group={group}")
        return

    patched = 0
    errors = 0
    for dataset, clf, method, original_path, group in work:
        print(f"Computing {group:20s}  {dataset}/{clf}/{method} ...", end=" ", flush=True)
        try:
            cell_result = _run_group(dataset, clf, method, base_cfg, group)

            new_keys = {k: cell_result[k] for k in _METRIC_KEYS[group] if k in cell_result}
            if not new_keys:
                print("SKIPPED (metric not in result — oracle may be absent)")
                continue

            original_result = json.loads(original_path.read_text())
            original_result.update(new_keys)
            original_path.write_text(json.dumps(original_result, indent=2))
            vals = ", ".join(f"{k}={v:.4f}" for k, v in new_keys.items())
            print(f"OK  ({vals})")
            patched += 1

        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}")
            errors += 1

    print(f"\nDone. Patched {patched} cells, {errors} errors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--metrics",
        choices=list(_METRIC_KEYS.keys()),
        default=None,
        dest="metrics",
        help="Restrict to a single metric group (default: all groups).",
    )
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()
    main(
        dry_run=args.dry_run,
        force=args.force,
        metric_groups=[args.metrics] if args.metrics else None,
        n_jobs=args.n_jobs,
    )
