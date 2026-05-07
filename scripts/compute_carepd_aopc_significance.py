"""scripts/compute_carepd_aopc_significance.py — paired bootstrap p-values for
PlayerAOPC differences across the four KernelSHAP imputers shown in Figure 1
of the paper.

Background
----------
``compute_real_cis_multiclf.py`` already produces 95% bootstrap CIs and a
canonical ``Marginal − VAEAC`` paired test for every classifier; this script
extends the protocol to **all** ordered pairs of {Zero, Mean, Marginal,
VAEAC} for the PlayerAOPC metric.  The pairing is per (classifier, fold,
sequence) so the test is paired at the finest possible granularity.  We
report:

* per-classifier paired tests (n=600 sequences = 200/fold × 3 folds);
* a classifier-pooled summary (n=1800), aggregated by stacking the
  per-sequence ``player_aopc`` arrays across MotionBERT, POTR and
  MotionAGFormer.

The pooled summary is the quantity referenced in the Figure 1 caption.
For each ordered pair ``(A, B)`` we report the bootstrap two-sided
p-value for the null hypothesis ``mean(player_aopc_A − player_aopc_B) = 0``.

Usage
-----
::

    conda activate motionbench-xai
    python scripts/compute_carepd_aopc_significance.py
    python scripts/compute_carepd_aopc_significance.py --B 20000 --seed 1

Output
------
Writes ``results/care_pd_multiclf/aopc_pairwise_significance.json`` with the
full per-classifier and pooled paired-test grids, and prints a compact
console table.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parents[1]
DEFAULT_RESULTS = REPO_ROOT / "results" / "care_pd_multiclf"

# Methods shown in Figure 1 (a) / (b) — see scripts/generate_paper_figures.py.
FIGURE1_METHODS = [
    "kernelshap_zero",
    "kernelshap_mean",
    "kernelshap_marginal",
    "kernelshap_vaeac",
]
DEFAULT_CLASSIFIERS = ["motionbert", "potr", "motionagformer"]
DEFAULT_FOLDS = [1, 2, 3]


# ----------------------------------------------------------------------- #
# Bootstrap helpers
# ----------------------------------------------------------------------- #


def paired_bootstrap_pvalue(
    diffs: np.ndarray, B: int = 10_000, seed: int = 0
) -> dict[str, float | int]:
    """Bootstrap CI and two-sided p-value for the mean of paired differences.

    Args:
        diffs: ``(n,)`` paired differences ``a_i − b_i``.
        B: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        Dict with ``n``, ``mean_diff``, ``ci95_low``, ``ci95_high``,
        ``p_le0``, ``p_ge0`` and ``p_two_sided``.
    """
    diffs = np.asarray(diffs, dtype=np.float64)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return {
            "n": 0,
            "mean_diff": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "p_le0": float("nan"),
            "p_ge0": float("nan"),
            "p_two_sided": float("nan"),
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(B, diffs.size))
    boot = diffs[idx].mean(axis=1)
    lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    p_le0 = float((boot <= 0.0).mean())
    p_ge0 = float((boot >= 0.0).mean())
    return {
        "n": int(diffs.size),
        "mean_diff": float(diffs.mean()),
        "ci95_low": lo,
        "ci95_high": hi,
        "p_le0": p_le0,
        "p_ge0": p_ge0,
        "p_two_sided": float(min(1.0, 2 * min(p_le0, p_ge0))),
    }


def load_aopc(
    results_root: Path, clf: str, method: str, folds: list[int]
) -> np.ndarray:
    """Concatenate per-sequence PlayerAOPC arrays across folds.

    Args:
        results_root: Root directory containing ``{clf}/fold{f}/{method}/result.json``.
        clf: Classifier name.
        method: Method name.
        folds: Fold indices to concatenate.

    Returns:
        ``(sum_f n_f,)`` float64 array.  Empty array if no files are found.
    """
    out: list[float] = []
    for f in folds:
        p = results_root / clf / f"fold{f}" / method / "result.json"
        if not p.exists():
            log.warning("missing %s", p)
            continue
        r = json.loads(p.read_text())
        out.extend(r.get("player_aopc_per_seq", []))
    return np.asarray(out, dtype=np.float64)


# ----------------------------------------------------------------------- #
# Pretty printing helpers
# ----------------------------------------------------------------------- #


def _fmt_p(p: float) -> str:
    """Format a p-value: ``"<10⁻³"`` (or smaller) when rounding hits the floor."""
    if not np.isfinite(p):
        return "  NA"
    if p == 0.0:
        return "<1e-4"
    if p < 1e-3:
        return f"{p:.0e}"
    return f"{p:.3f}"


def _stars(p: float) -> str:
    """Significance stars used in figure annotations and the caption."""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ----------------------------------------------------------------------- #
# Main pairwise loop
# ----------------------------------------------------------------------- #


def pairwise_grid(
    aopc_by_method: dict[str, np.ndarray], B: int, seed: int
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute paired bootstrap p-values for every ordered pair (A, B).

    The orientation ``A − B`` is preserved so that ``mean_diff > 0`` means
    method A has the larger PlayerAOPC.

    Args:
        aopc_by_method: Mapping ``method → (n,)`` PlayerAOPC array.  All
            arrays must be the same length and aligned by sequence index.
        B: Bootstrap resamples.
        seed: Random seed.

    Returns:
        Nested dict ``[method_a][method_b] → paired_bootstrap result``.
    """
    grid: dict[str, dict[str, dict[str, Any]]] = {}
    for ma, a_arr in aopc_by_method.items():
        grid[ma] = {}
        for mb, b_arr in aopc_by_method.items():
            if ma == mb:
                continue
            n = min(a_arr.size, b_arr.size)
            if n == 0:
                grid[ma][mb] = paired_bootstrap_pvalue(np.array([]), B=B, seed=seed)
                continue
            grid[ma][mb] = paired_bootstrap_pvalue(
                a_arr[:n] - b_arr[:n], B=B, seed=seed
            )
    return grid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS))
    ap.add_argument("--classifiers", type=str, nargs="+", default=DEFAULT_CLASSIFIERS)
    ap.add_argument("--folds", type=int, nargs="+", default=DEFAULT_FOLDS)
    ap.add_argument("--methods", type=str, nargs="+", default=FIGURE1_METHODS)
    ap.add_argument("--B", type=int, default=10_000, help="Bootstrap resamples.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results_root = Path(args.results_dir)
    summary: dict[str, Any] = {
        "B_resamples": args.B,
        "seed": args.seed,
        "metric": "player_aopc",
        "methods": list(args.methods),
        "per_classifier": {},
        "pooled_across_classifiers": {},
    }

    pooled_by_method: dict[str, list[float]] = {m: [] for m in args.methods}

    for clf in args.classifiers:
        aopc_by_method: dict[str, np.ndarray] = {}
        for m in args.methods:
            arr = load_aopc(results_root, clf, m, args.folds)
            aopc_by_method[m] = arr
            pooled_by_method[m].extend(arr.tolist())
            log.info("%s/%s: n=%d  mean=%.4f", clf, m, arr.size, float(arr.mean()) if arr.size else float("nan"))
        per_clf = pairwise_grid(aopc_by_method, B=args.B, seed=args.seed)
        means = {m: float(aopc_by_method[m].mean()) if aopc_by_method[m].size else float("nan")
                 for m in args.methods}
        summary["per_classifier"][clf] = {"means": means, "pairwise": per_clf}

    pooled_arrays = {m: np.asarray(pooled_by_method[m], dtype=np.float64)
                     for m in args.methods}
    pooled_grid = pairwise_grid(pooled_arrays, B=args.B, seed=args.seed)
    pooled_means = {m: float(pooled_arrays[m].mean()) if pooled_arrays[m].size else float("nan")
                    for m in args.methods}
    summary["pooled_across_classifiers"] = {
        "n_total": int(min(arr.size for arr in pooled_arrays.values()) if pooled_arrays else 0),
        "means": pooled_means,
        "pairwise": pooled_grid,
    }

    out_path = results_root / "aopc_pairwise_significance.json"
    out_path.write_text(json.dumps(summary, indent=2))
    log.info("wrote %s", out_path)

    print("\n" + "=" * 92)
    print("CARE-PD PlayerAOPC pairwise paired bootstrap test  (B=%d)" % args.B)
    print("Direction:  diff = mean(player_aopc[A]) - mean(player_aopc[B])")
    print("=" * 92)
    method_labels = {
        "kernelshap_zero": "KS-Zero",
        "kernelshap_mean": "KS-Mean",
        "kernelshap_marginal": "KS-Marginal",
        "kernelshap_vaeac": "KS-VAEAC",
        "kernelshap_flow": "KS-Flow",
    }
    for clf_name, payload in (
        list(summary["per_classifier"].items())
        + [("POOLED", summary["pooled_across_classifiers"])]
    ):
        print(f"\n[{clf_name}]")
        n_total = payload.get("n_total", "")
        if n_total:
            print(f"  n_per_method = {n_total}")
        hdr = f"  {'A':>13}  {'B':>13}  {'mean diff':>10}  {'95% CI':>22}  {'p (two-sided)':>15}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for ma in args.methods:
            for mb in args.methods:
                if ma == mb:
                    continue
                t = payload["pairwise"][ma][mb]
                ci = f"[{t['ci95_low']:+.4f},{t['ci95_high']:+.4f}]"
                p = t["p_two_sided"]
                print(
                    f"  {method_labels[ma]:>13}  {method_labels[mb]:>13}  "
                    f"{t['mean_diff']:+10.4f}  {ci:>22}  {_fmt_p(p):>10} {_stars(p)}"
                )
    print("=" * 92)


if __name__ == "__main__":
    main()
