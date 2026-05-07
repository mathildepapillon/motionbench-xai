"""scripts/compute_real_cis.py — bootstrap 95% CIs and paired tests for the
extended CARE-PD real-world sweep.

Inputs
------
``results/care_pd_extended/foldN/<method>/result.json`` (N=1..3) — produced
by ``scripts/run_care_pd_extended.py``.  Each result.json must contain the
per-sequence arrays ``faithfulness_per_seq`` and ``player_aopc_per_seq``.

Outputs
-------
``results/care_pd_extended/summary_with_ci.json`` — one row per method:

    {
      "method": "kernelshap_zero",
      "n_folds": 3,
      "n_total_sequences": 600,
      "faithfulness": {
        "fold_means": [...],            # length-K_folds list
        "mean_of_fold_means": float,
        "std_of_fold_means": float,
        "pooled_mean": float,           # mean over all sequences pooled
        "ci95_low": float,              # bootstrap 95% CI on pooled mean
        "ci95_high": float,
        "n_finite": int
      },
      "player_aopc": { same structure },
      ...
    }

It also reports a paired bootstrap test for the difference
``faithfulness(KS-Marginal) - faithfulness(KS-VAEAC)``.  Pairing is by
sequence index *within* each fold (the extended script uses the same
seed/donor logic for every method, so sequence i is the same clip across
methods within a fold).

Usage
-----
    conda activate motionbench-xai
    python scripts/compute_real_cis.py
    python scripts/compute_real_cis.py --B 10000
    python scripts/compute_real_cis.py --paired marginal vaeac --paired marginal flow
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


CAREPD_ROOT = Path(__file__).parents[1]
DEFAULT_RESULTS = CAREPD_ROOT / "results" / "care_pd_extended"

ALL_METHODS = [
    "kernelshap_zero",
    "kernelshap_mean",
    "kernelshap_marginal",
    "kernelshap_vaeac",
    "kernelshap_flow",
]


def load_per_method(results_root: Path, folds: list[int]) -> dict:
    """Return ``{method: {fold: {"faith": np.ndarray, "aopc": np.ndarray, ...}}}``."""
    out: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for m in ALL_METHODS:
        out[m] = {}
        for f in folds:
            p = results_root / f"fold{f}" / m / "result.json"
            if not p.exists():
                log.warning("missing %s — skipping", p)
                continue
            r = json.loads(p.read_text())
            out[m][f] = {
                "faith": np.asarray(r.get("faithfulness_per_seq", []), dtype=np.float64),
                "aopc": np.asarray(r.get("player_aopc_per_seq", []), dtype=np.float64),
                "targets": np.asarray(r.get("targets_per_seq", []), dtype=np.int64),
                "n": int(r.get("n_sequences", 0)),
            }
    return out


def bootstrap_ci_mean(
    x: np.ndarray, B: int = 10_000, alpha: float = 0.05, seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of ``x`` (NaNs ignored).

    Returns ``(mean, ci_low, ci_high)``.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(B, x.size))
    boot_means = x[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return float(x.mean()), lo, hi


def paired_bootstrap_pvalue(
    diffs: np.ndarray, B: int = 10_000, seed: int = 0,
) -> dict:
    """Two-sided paired bootstrap p-value testing H0: E[diff] = 0.

    Returns dict with mean diff, CI95, and p-values (one-sided & two-sided).
    The one-sided ``p_le0`` is the bootstrap probability that the resampled
    mean diff is <= 0 — i.e. evidence that diff > 0 in the population.
    """
    diffs = np.asarray(diffs, dtype=np.float64)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return {"n": 0, "mean_diff": float("nan"),
                "ci95_low": float("nan"), "ci95_high": float("nan"),
                "p_le0": float("nan"), "p_ge0": float("nan"),
                "p_two_sided": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(B, diffs.size))
    boot = diffs[idx].mean(axis=1)
    lo = float(np.quantile(boot, 0.025))
    hi = float(np.quantile(boot, 0.975))
    p_le0 = float((boot <= 0.0).mean())
    p_ge0 = float((boot >= 0.0).mean())
    p_two = 2.0 * min(p_le0, p_ge0)
    return {
        "n": int(diffs.size),
        "mean_diff": float(diffs.mean()),
        "ci95_low": lo,
        "ci95_high": hi,
        "p_le0": p_le0,
        "p_ge0": p_ge0,
        "p_two_sided": float(min(1.0, p_two)),
    }


def summarize_method(method_data: dict[int, dict], B: int, seed: int) -> dict:
    """Build per-method summary across folds."""
    folds_present = sorted(method_data.keys())
    fold_faith_means: list[float] = []
    fold_aopc_means: list[float] = []
    pooled_faith: list[float] = []
    pooled_aopc: list[float] = []
    n_per_fold: list[int] = []

    for f in folds_present:
        fd = method_data[f]
        faith_arr = fd["faith"]
        aopc_arr = fd["aopc"]
        finite = np.isfinite(faith_arr)
        fold_faith_means.append(
            float(np.nanmean(faith_arr)) if faith_arr.size else float("nan")
        )
        fold_aopc_means.append(
            float(np.mean(aopc_arr)) if aopc_arr.size else float("nan")
        )
        pooled_faith.extend(faith_arr.tolist())
        pooled_aopc.extend(aopc_arr.tolist())
        n_per_fold.append(int(faith_arr.size))

    pooled_faith_arr = np.asarray(pooled_faith, dtype=np.float64)
    pooled_aopc_arr = np.asarray(pooled_aopc, dtype=np.float64)
    n_total = int(pooled_faith_arr.size)
    n_finite_faith = int(np.isfinite(pooled_faith_arr).sum())

    f_mean, f_lo, f_hi = bootstrap_ci_mean(pooled_faith_arr, B=B, seed=seed)
    a_mean, a_lo, a_hi = bootstrap_ci_mean(pooled_aopc_arr, B=B, seed=seed + 1)

    return {
        "n_folds": len(folds_present),
        "folds": folds_present,
        "n_total_sequences": n_total,
        "n_per_fold": n_per_fold,
        "faithfulness": {
            "fold_means": fold_faith_means,
            "mean_of_fold_means": float(np.nanmean(fold_faith_means)),
            "std_of_fold_means": (
                float(np.nanstd(fold_faith_means, ddof=1))
                if len(fold_faith_means) > 1 else float("nan")
            ),
            "pooled_mean": f_mean,
            "ci95_low": f_lo,
            "ci95_high": f_hi,
            "n_finite": n_finite_faith,
            "B_resamples": B,
        },
        "player_aopc": {
            "fold_means": fold_aopc_means,
            "mean_of_fold_means": float(np.nanmean(fold_aopc_means)),
            "std_of_fold_means": (
                float(np.nanstd(fold_aopc_means, ddof=1))
                if len(fold_aopc_means) > 1 else float("nan")
            ),
            "pooled_mean": a_mean,
            "ci95_low": a_lo,
            "ci95_high": a_hi,
            "n_finite": int(np.isfinite(pooled_aopc_arr).sum()),
            "B_resamples": B,
        },
    }


def paired_diff_array(
    method_a: dict[int, dict],
    method_b: dict[int, dict],
    metric: str,
) -> np.ndarray:
    """Return per-sequence (a-b) differences pooled across shared folds.

    Pairing is by sequence index within each fold; both methods must run
    over the same N val sequences in the same order (which they do, since
    ``run_care_pd_extended.py`` enumerates the cache identically for every
    method).
    """
    diffs: list[float] = []
    shared = sorted(set(method_a.keys()) & set(method_b.keys()))
    for f in shared:
        a = method_a[f][metric]
        b = method_b[f][metric]
        n = min(a.size, b.size)
        diffs.extend((a[:n] - b[:n]).tolist())
    return np.asarray(diffs, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS),
                    help="Root with foldN/<method>/result.json files.")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--B", type=int, default=10_000,
                    help="Bootstrap resamples (default 10000).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--paired", action="append", nargs=2, metavar=("A", "B"),
                    default=None,
                    help=("Paired bootstrap comparisons of the form "
                          "'kernelshap_marginal kernelshap_vaeac'. "
                          "Short keys (e.g. 'marginal' 'vaeac') accepted. "
                          "Defaults to marginal vs vaeac if not specified."))
    ap.add_argument("--out", type=str, default=None,
                    help="Output summary path (default <results_dir>/summary_with_ci.json).")
    args = ap.parse_args()

    results_root = Path(args.results_dir)
    out_path = Path(args.out) if args.out else results_root / "summary_with_ci.json"

    paired_pairs = args.paired or [["marginal", "vaeac"]]

    def normalise(name: str) -> str:
        return name if name in ALL_METHODS else f"kernelshap_{name}"

    log.info("loading per-method results from %s (folds=%s)",
             results_root, args.folds)
    data = load_per_method(results_root, args.folds)

    summary: dict = {
        "results_dir": str(results_root),
        "folds": args.folds,
        "B_resamples": int(args.B),
        "seed": int(args.seed),
        "methods": {},
        "paired_tests": [],
    }

    for m in ALL_METHODS:
        if not data[m]:
            log.warning("no fold results for %s — skipping", m)
            continue
        log.info("summarising %s (folds=%s)", m, sorted(data[m].keys()))
        summary["methods"][m] = summarize_method(data[m], B=args.B, seed=args.seed)

    for raw_a, raw_b in paired_pairs:
        a_key = normalise(raw_a)
        b_key = normalise(raw_b)
        if a_key not in data or b_key not in data:
            log.warning("paired test %s vs %s skipped (missing data)", a_key, b_key)
            continue
        diffs_faith = paired_diff_array(data[a_key], data[b_key], "faith")
        diffs_aopc = paired_diff_array(data[a_key], data[b_key], "aopc")
        ptest = {
            "method_a": a_key,
            "method_b": b_key,
            "metric": "faithfulness_correlation",
            "diff_definition": f"{a_key} - {b_key}",
            **paired_bootstrap_pvalue(diffs_faith, B=args.B, seed=args.seed + 7),
            "secondary_metric_player_aopc": paired_bootstrap_pvalue(
                diffs_aopc, B=args.B, seed=args.seed + 8,
            ),
        }
        summary["paired_tests"].append(ptest)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    log.info("wrote %s", out_path)

    print("\n" + "=" * 78)
    print(f"REAL-WORLD EXTENDED SUMMARY  (results_dir={results_root})")
    print("=" * 78)
    print(f"{'method':<25} {'n':>4}  {'faith mean [95% CI]':<26}  "
          f"{'AOPC mean [95% CI]':<26}  {'fold std (faith)':>16}")
    for m, mdata in summary["methods"].items():
        f = mdata["faithfulness"]
        a = mdata["player_aopc"]
        n = mdata["n_total_sequences"]
        faith_str = f"{f['pooled_mean']:+.3f} [{f['ci95_low']:+.3f},{f['ci95_high']:+.3f}]"
        aopc_str = f"{a['pooled_mean']:+.3f} [{a['ci95_low']:+.3f},{a['ci95_high']:+.3f}]"
        print(f"{m:<25} {n:>4}  {faith_str:<26}  {aopc_str:<26}  "
              f"{f['std_of_fold_means']:>16.4f}")
    print("=" * 78)
    print("PAIRED BOOTSTRAP TESTS (faithfulness correlation):")
    for t in summary["paired_tests"]:
        sign = ">" if t["mean_diff"] > 0 else "<"
        sig = "*" if t["p_two_sided"] < 0.05 else " "
        print(f"  {t['method_a']:<22} {sign} {t['method_b']:<22} | "
              f"diff={t['mean_diff']:+.4f} [95% CI {t['ci95_low']:+.4f},{t['ci95_high']:+.4f}] | "
              f"p_two_sided={t['p_two_sided']:.4f}{sig}  (p_le0={t['p_le0']:.4f}, "
              f"p_ge0={t['p_ge0']:.4f}, n={t['n']})")
    print("=" * 78)


if __name__ == "__main__":
    main()
