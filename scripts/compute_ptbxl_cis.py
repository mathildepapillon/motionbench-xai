"""scripts/compute_ptbxl_cis.py — Bootstrap CIs for the PTB-XL SHAP sweep.

Mirrors ``compute_real_cis_multiclf.py`` for the PTB-XL dataset.  Reads
individual fold results written by ``run_ptbxl_shap.py`` and computes
pooled means with 95% bootstrap confidence intervals.

Reads from::

    results/ptbxl/fold{f}/{method}/result.json

Writes::

    results/ptbxl/summary_ptbxl.json   ← consumed by generate_paper_tables.py

Shared helper functions are imported directly from
``compute_real_cis_multiclf.py`` to avoid code duplication:

    bootstrap_ci_mean       — bootstrap mean + CI
    paired_bootstrap_pvalue — paired bootstrap p-value
    summarize_method        — aggregate fold-level data

Usage::

    conda activate motionbench-xai
    python scripts/compute_ptbxl_cis.py
    python scripts/compute_ptbxl_cis.py --folds 1 2 3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parents[1]
SCRIPTS_DIR = Path(__file__).parent

# Import shared helpers — no duplication
sys.path.insert(0, str(SCRIPTS_DIR))
from compute_real_cis_multiclf import (   # noqa: E402
    bootstrap_ci_mean,
    paired_bootstrap_pvalue,
    summarize_method,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_RESULTS = REPO_ROOT / "results" / "ptbxl"

ALL_METHODS = [
    "kernelshap_zero",
    "kernelshap_mean",
    "kernelshap_marginal",
    "kernelshap_vaeac",
    "kernelshap_flow",
]


# ---------------------------------------------------------------------- helpers

def load_method_data(results_dir: Path, folds: list[int]) -> dict:
    """Load ``{method: {fold: {faith, aopc, ...}}}`` from PTB-XL results.

    Args:
        results_dir: Root directory containing ``fold{f}/`` subdirectories.
        folds: List of fold indices to load.

    Returns:
        Nested dict keyed by method then fold.
    """
    out: dict = {}
    for m in ALL_METHODS:
        out[m] = {}
        for f in folds:
            p = results_dir / f"fold{f}" / m / "result.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            out[m][f] = {
                "faith": np.asarray(r.get("faithfulness_per_seq", []), dtype=np.float64),
                "aopc":  np.asarray(r.get("player_aopc_per_seq",  []), dtype=np.float64),
            }
    return out


# ---------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS))
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--B", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    folds = args.folds
    B, seed = args.B, args.seed

    data = load_method_data(results_dir, folds)
    methods_present = [m for m in ALL_METHODS if data[m]]
    log.info("PTB-XL CIs: found %d methods across folds %s", len(methods_present), folds)

    summary_methods: dict = {}
    for m in methods_present:
        summary_methods[m] = summarize_method(data[m], B=B, seed=seed)

    # Paired tests: standard comparisons as in CARE-PD
    paired_tests: list[dict] = []
    for ma, mb in [
        ("kernelshap_marginal", "kernelshap_vaeac"),
        ("kernelshap_marginal", "kernelshap_flow"),
        ("kernelshap_vaeac",    "kernelshap_flow"),
    ]:
        if ma not in data or mb not in data:
            continue
        diffs_f, diffs_a = [], []
        shared = sorted(set(data[ma].keys()) & set(data[mb].keys()))
        for f in shared:
            a_arr = data[ma][f]["faith"]
            b_arr = data[mb][f]["faith"]
            n = min(a_arr.size, b_arr.size)
            diffs_f.extend((a_arr[:n] - b_arr[:n]).tolist())
            a_arr2 = data[ma][f]["aopc"]
            b_arr2 = data[mb][f]["aopc"]
            n2 = min(a_arr2.size, b_arr2.size)
            diffs_a.extend((a_arr2[:n2] - b_arr2[:n2]).tolist())
        ptest = {
            "method_a": ma,
            "method_b": mb,
            "metric": "faithfulness_correlation",
            "diff_definition": f"{ma} - {mb}",
            **paired_bootstrap_pvalue(np.asarray(diffs_f), B=B, seed=seed + 7),
            "secondary_metric_player_aopc": paired_bootstrap_pvalue(
                np.asarray(diffs_a), B=B, seed=seed + 8
            ),
        }
        paired_tests.append(ptest)

    summary = {
        "dataset": "ptbxl",
        "classifier": "ecg_resnet1d",
        "folds": folds,
        "B_resamples": B,
        "methods": summary_methods,
        "paired_tests": paired_tests,
    }
    out_path = results_dir / "summary_ptbxl.json"
    out_path.write_text(json.dumps(summary, indent=2))
    log.info("Wrote PTB-XL CI summary → %s", out_path)

    # Pretty-print table
    print("\n" + "=" * 80)
    print("PTB-XL SHAP SUMMARY")
    print("=" * 80)
    hdr = f"{'method':<25} {'n':>4}  {'faith mean [95%CI]':<28}  {'AOPC mean [95%CI]':<28}"
    print(hdr)
    for m, md in summary["methods"].items():
        fk = md["faithfulness"]
        ak = md["player_aopc"]
        n  = md["n_total_sequences"]
        fs = f"{fk['pooled_mean']:+.3f} [{fk['ci95_low']:+.3f},{fk['ci95_high']:+.3f}]"
        as_ = f"{ak['pooled_mean']:+.3f} [{ak['ci95_low']:+.3f},{ak['ci95_high']:+.3f}]"
        print(f"{m:<25} {n:>4}  {fs:<28}  {as_:<28}")
    print("=" * 80)


if __name__ == "__main__":
    main()
