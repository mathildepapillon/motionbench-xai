"""scripts/compute_esc50_summary.py — Pool ESC-50 SHAP results and compute bootstrap CIs.

Loads all result.json files from results/esc50/{fold}/{method}/,
pools faithfulness and AOPC across 3 folds, computes mean ± std and
95% bootstrap CI (B=1000 resamples), and prints a summary table.

Usage::

    python scripts/compute_esc50_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parents[1]
RESULTS_ROOT = REPO_ROOT / "results" / "esc50"

ALL_METHODS = [
    "kernelshap_zero",
    "kernelshap_mean",
    "kernelshap_marginal",
    "kernelshap_vaeac",
    "kernelshap_flow",
]

METHOD_LABELS = {
    "kernelshap_zero": "KS-Zero",
    "kernelshap_mean": "KS-Mean",
    "kernelshap_marginal": "KS-Marginal",
    "kernelshap_vaeac": "KS-VAEAC",
    "kernelshap_flow": "KS-Flow",
}


def bootstrap_ci(arr: np.ndarray, B: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
    """Returns (mean, ci_low, ci_high) via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(B)])
    return float(arr.mean()), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def main() -> None:
    method_results = {}

    for method in ALL_METHODS:
        faiths_all = []
        aopcs_all  = []
        for fold in [1, 2, 3]:
            rp = RESULTS_ROOT / f"fold{fold}" / method / "result.json"
            if not rp.exists():
                print(f"[WARNING] Missing: {rp}")
                continue
            with open(rp) as f:
                r = json.load(f)
            faiths_all.extend(r["faithfulness_per_seq"])
            aopcs_all.extend(r["player_aopc_per_seq"])

        if not faiths_all:
            print(f"[WARNING] No data for {method}")
            continue

        faiths = np.array([v for v in faiths_all if np.isfinite(v)], dtype=np.float64)
        aopcs  = np.array(aopcs_all, dtype=np.float64)

        faith_mean, faith_ci_lo, faith_ci_hi = bootstrap_ci(faiths)
        aopc_mean, aopc_ci_lo, aopc_ci_hi    = bootstrap_ci(aopcs)
        faith_std = float(np.std(faiths, ddof=1))
        aopc_std  = float(np.std(aopcs, ddof=1))

        method_results[method] = {
            "n_total": len(faiths_all),
            "n_finite_faithfulness": len(faiths),
            "faithfulness_mean": faith_mean,
            "faithfulness_std": faith_std,
            "faithfulness_ci_95": [faith_ci_lo, faith_ci_hi],
            "aopc_mean": aopc_mean,
            "aopc_std": aopc_std,
            "aopc_ci_95": [aopc_ci_lo, aopc_ci_hi],
        }

    # Compute ranks
    faith_ranks = {m: r for r, m in enumerate(
        sorted(method_results, key=lambda m: method_results[m]["faithfulness_mean"], reverse=True), 1
    )}
    aopc_ranks = {m: r for r, m in enumerate(
        sorted(method_results, key=lambda m: method_results[m]["aopc_mean"], reverse=True), 1
    )}

    # Print summary table
    print("\n" + "=" * 100)
    print("ESC-50 KernelSHAP Benchmark Summary (pooled across 3 folds)")
    print("=" * 100)
    header = f"{'Method':<18} {'Faithfulness':>22} {'AOPC':>22} {'Faith Rank':>12} {'AOPC Rank':>10}"
    print(header)
    print("-" * 100)
    for method in ALL_METHODS:
        if method not in method_results:
            print(f"  {METHOD_LABELS[method]:<16} — missing —")
            continue
        r = method_results[method]
        faith_str = f"{r['faithfulness_mean']:+.4f} ± {r['faithfulness_std']:.4f}  [{r['faithfulness_ci_95'][0]:+.4f}, {r['faithfulness_ci_95'][1]:+.4f}]"
        aopc_str  = f"{r['aopc_mean']:+.4f} ± {r['aopc_std']:.4f}  [{r['aopc_ci_95'][0]:+.4f}, {r['aopc_ci_95'][1]:+.4f}]"
        print(f"  {METHOD_LABELS[method]:<16} {faith_str:>40}  {aopc_str:>40}  {faith_ranks[method]:>8}  {aopc_ranks[method]:>8}")
    print("=" * 100)

    # Save summary.json
    summary = {
        "dataset": "esc50",
        "methods": method_results,
        "faith_ranks": faith_ranks,
        "aopc_ranks": aopc_ranks,
    }
    out_path = RESULTS_ROOT / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
