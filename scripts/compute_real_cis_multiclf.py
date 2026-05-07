"""scripts/compute_real_cis_multiclf.py — bootstrap CIs for the
multi-classifier CARE-PD sweep produced by ``run_care_pd_multiclf.py``.

Reads from::

    results/care_pd_multiclf/{classifier}/fold{f}/{method}/result.json

Writes::

    results/care_pd_multiclf/{classifier}/summary_with_ci.json   (per clf)
    results/care_pd_multiclf/summary_multi.json                  (combined)

Usage::

    conda activate motionbench-xai
    python scripts/compute_real_cis_multiclf.py
    python scripts/compute_real_cis_multiclf.py --classifiers motionbert potr
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parents[1]
DEFAULT_RESULTS = REPO_ROOT / "results" / "care_pd_multiclf"

ALL_METHODS = [
    "kernelshap_zero",
    "kernelshap_mean",
    "kernelshap_marginal",
    "kernelshap_vaeac",
    "kernelshap_flow",
]
DEFAULT_CLASSIFIERS = ["motionbert", "potr", "motionagformer"]


# ------------------------------------------------------------------ helpers

def bootstrap_ci_mean(x: np.ndarray, B: int = 10_000, alpha: float = 0.05,
                       seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(B, x.size))
    boot = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def paired_bootstrap_pvalue(diffs: np.ndarray, B: int = 10_000, seed: int = 0) -> dict:
    diffs = np.asarray(diffs, dtype=np.float64)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return {"n": 0, "mean_diff": float("nan"), "ci95_low": float("nan"),
                "ci95_high": float("nan"), "p_le0": float("nan"),
                "p_ge0": float("nan"), "p_two_sided": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(B, diffs.size))
    boot = diffs[idx].mean(axis=1)
    lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    p_le0 = float((boot <= 0.0).mean())
    p_ge0 = float((boot >= 0.0).mean())
    return {"n": int(diffs.size), "mean_diff": float(diffs.mean()),
            "ci95_low": lo, "ci95_high": hi, "p_le0": p_le0,
            "p_ge0": p_ge0, "p_two_sided": float(min(1.0, 2 * min(p_le0, p_ge0)))}


def load_method_data(clf_dir: Path, folds: list[int]) -> dict:
    """Load {method: {fold: {faith, aopc, ...}}}."""
    out: dict = {}
    for m in ALL_METHODS:
        out[m] = {}
        for f in folds:
            p = clf_dir / f"fold{f}" / m / "result.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            out[m][f] = {
                "faith": np.asarray(r.get("faithfulness_per_seq", []), dtype=np.float64),
                "aopc":  np.asarray(r.get("player_aopc_per_seq",  []), dtype=np.float64),
            }
    return out


def summarize_method(method_data: dict, B: int, seed: int) -> dict:
    folds_present = sorted(method_data.keys())
    f_fold, a_fold, f_pool, a_pool, n_per = [], [], [], [], []
    for f in folds_present:
        fd = method_data[f]
        f_fold.append(float(np.nanmean(fd["faith"])) if fd["faith"].size else float("nan"))
        a_fold.append(float(np.mean(fd["aopc"]))   if fd["aopc"].size  else float("nan"))
        f_pool.extend(fd["faith"].tolist())
        a_pool.extend(fd["aopc"].tolist())
        n_per.append(int(fd["faith"].size))
    f_arr = np.asarray(f_pool, dtype=np.float64)
    a_arr = np.asarray(a_pool, dtype=np.float64)
    fm, fl, fh = bootstrap_ci_mean(f_arr, B=B, seed=seed)
    am, al, ah = bootstrap_ci_mean(a_arr, B=B, seed=seed + 1)
    return {
        "n_folds": len(folds_present),
        "folds": folds_present,
        "n_total_sequences": int(f_arr.size),
        "n_per_fold": n_per,
        "faithfulness": {
            "fold_means": f_fold,
            "mean_of_fold_means": float(np.nanmean(f_fold)),
            "std_of_fold_means": (float(np.nanstd(f_fold, ddof=1)) if len(f_fold) > 1 else float("nan")),
            "pooled_mean": fm, "ci95_low": fl, "ci95_high": fh,
            "n_finite": int(np.isfinite(f_arr).sum()), "B_resamples": B,
        },
        "player_aopc": {
            "fold_means": a_fold,
            "mean_of_fold_means": float(np.nanmean(a_fold)),
            "std_of_fold_means": (float(np.nanstd(a_fold, ddof=1)) if len(a_fold) > 1 else float("nan")),
            "pooled_mean": am, "ci95_low": al, "ci95_high": ah,
            "n_finite": int(np.isfinite(a_arr).sum()), "B_resamples": B,
        },
    }


def process_classifier(clf_dir: Path, clf_name: str, folds: list[int], B: int,
                        seed: int) -> dict:
    data = load_method_data(clf_dir, folds)
    methods_present = [m for m in ALL_METHODS if data[m]]
    log.info("[%s] found %d methods across folds %s", clf_name, len(methods_present), folds)

    summary_methods: dict = {}
    for m in methods_present:
        summary_methods[m] = summarize_method(data[m], B=B, seed=seed)

    # Paired test: marginal vs vaeac (canonical)
    paired_tests = []
    for ma, mb in [("kernelshap_marginal", "kernelshap_vaeac"),
                   ("kernelshap_marginal", "kernelshap_flow"),
                   ("kernelshap_vaeac",    "kernelshap_flow")]:
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
            "method_a": ma, "method_b": mb,
            "metric": "faithfulness_correlation",
            "diff_definition": f"{ma} - {mb}",
            **paired_bootstrap_pvalue(np.asarray(diffs_f), B=B, seed=seed + 7),
            "secondary_metric_player_aopc": paired_bootstrap_pvalue(
                np.asarray(diffs_a), B=B, seed=seed + 8),
        }
        paired_tests.append(ptest)

    clf_summary = {
        "classifier": clf_name,
        "folds": folds,
        "B_resamples": B,
        "methods": summary_methods,
        "paired_tests": paired_tests,
    }
    out_path = clf_dir / "summary_with_ci.json"
    out_path.write_text(json.dumps(clf_summary, indent=2))
    log.info("[%s] wrote %s", clf_name, out_path)
    return clf_summary


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS))
    ap.add_argument("--classifiers", type=str, nargs="+", default=DEFAULT_CLASSIFIERS)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--B", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results_root = Path(args.results_dir)
    combined: dict = {}

    for clf_name in args.classifiers:
        clf_dir = results_root / clf_name
        if not clf_dir.is_dir():
            log.warning("directory not found for %s, skipping", clf_name)
            continue
        clf_summary = process_classifier(clf_dir, clf_name, args.folds, args.B, args.seed)
        combined[clf_name] = clf_summary

    # Combined summary keyed by classifier
    combined_path = results_root / "summary_multi.json"
    combined_path.write_text(json.dumps(combined, indent=2))
    log.info("wrote combined summary → %s", combined_path)

    # Pretty-print table
    print("\n" + "=" * 90)
    print("MULTI-CLASSIFIER REAL-WORLD SUMMARY")
    print("=" * 90)
    hdr = f"{'clf':<14} {'method':<25} {'n':>4}  {'faith mean [95%CI]':<28}  {'AOPC mean [95%CI]':<28}"
    print(hdr)
    for clf_name, cs in combined.items():
        for m, md in cs.get("methods", {}).items():
            f = md["faithfulness"]
            a = md["player_aopc"]
            n = md["n_total_sequences"]
            faith_str = f"{f['pooled_mean']:+.3f} [{f['ci95_low']:+.3f},{f['ci95_high']:+.3f}]"
            aopc_str  = f"{a['pooled_mean']:+.3f} [{a['ci95_low']:+.3f},{a['ci95_high']:+.3f}]"
            print(f"{clf_name:<14} {m:<25} {n:>4}  {faith_str:<28}  {aopc_str:<28}")
    print("=" * 90)


if __name__ == "__main__":
    main()
