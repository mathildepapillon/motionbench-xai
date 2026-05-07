"""scripts/framework_2x2.py — Top-down 2×2 interpretive framework for MotionBench-XAI.

FRAMEWORK OVERVIEW
==================

MotionBench-XAI benchmarks SHAP imputation strategies for spatiotemporal motion data
along two independent failure axes:

    Axis A — PLAYER ABSTRACTION: what is the unit of attribution?
    ┌──────────────────────────────────────────────────────────────┐
    │  Temporal players  (TemporalWindows)   → time segments       │
    │  Spatial players   (SpatialJoints)     → body joints         │
    └──────────────────────────────────────────────────────────────┘

    Axis B — IMPUTER QUALITY: how are masked features filled?
    ┌──────────────────────────────────────────────────────────────┐
    │  Off-manifold   (Zero, Mean)           → invalid poses       │
    │  Weak on-manifold (Marginal, Empirical)→ marginal samples    │
    │  Strong on-manifold (Oracle, VAEAC, Flow) → p(x_hid|x_obs)  │
    └──────────────────────────────────────────────────────────────┘

THE 2×2 FAILURE MODE MATRIX
============================

                      ┌─────────────────┬─────────────────┐
                      │  Temporal label  │  Spatial label  │
                      │ (window_label_*) │(joint_subset_*) │
    ┌─────────────────┼─────────────────┼─────────────────┤
    │Temporal players │  CORRECT axis   │  WRONG axis     │
    │(KS-Temporal,    │ (can identify   │ (joints grouped │
    │ WindowSHAP)     │  label window)  │  per timestep)  │
    ├─────────────────┼─────────────────┼─────────────────┤
    │Spatial players  │  WRONG axis     │  CORRECT axis   │
    │(KS-Spatial)     │ (time spread    │ (can identify   │
    │                 │  across joints) │  label joints)  │
    └─────────────────┴─────────────────┴─────────────────┘

HYPOTHESIS: wrong player abstraction costs more attribution quality (Top-K recovery,
Spearman rank) than wrong imputer quality — i.e., Axis A is the dominant failure mode.

IMPUTER QUALITY within the correct player abstraction:
  • Off-manifold (Zero) < Weak (Marginal) < Strong (Oracle) on datasets with
    complex manifold structure (Burr, LowRank, Skeleton).
  • Oracle ≈ Zero on Gaussian datasets (manifold is the full space by definition).

METHOD → FRAMEWORK PLACEMENT
==============================

┌──────────────────────────────┬────────────────────┬─────────────────────┐
│ Method                       │ Player abstraction │ Imputer quality     │
├──────────────────────────────┼────────────────────┼─────────────────────┤
│ KernelSHAP-Zero              │ Temporal           │ Off-manifold        │
│ KernelSHAP-Mean              │ Temporal           │ Off-manifold (mean) │
│ KernelSHAP-Marginal          │ Temporal           │ Weak on-manifold    │
│ KernelSHAP-Empirical         │ Temporal           │ Weak on-manifold    │
│ KernelSHAP-Oracle            │ Temporal           │ Strong on-manifold  │
│ KernelSHAP-VAEAC             │ Temporal           │ Strong on-manifold  │
│ KernelSHAP-Flow              │ Temporal           │ Strong on-manifold  │
│ KernelSHAP-Temporal          │ Temporal           │ Off-manifold (zero) │
│ WindowSHAP                   │ Temporal           │ Off-manifold (zero) │
│ KernelSHAP-Zero-Spatial      │ Spatial            │ Off-manifold        │
│ KernelSHAP-Oracle-Spatial    │ Spatial            │ Strong on-manifold  │
│ KernelSHAP-VAEAC-Spatial     │ Spatial            │ Strong on-manifold  │
└──────────────────────────────┴────────────────────┴─────────────────────┘

DATASET → PROBING AXIS
=======================

┌──────────────────────────┬────────────────────────┬────────────────────────┐
│ Dataset                  │ Primary probe axis     │ Secondary              │
├──────────────────────────┼────────────────────────┼────────────────────────┤
│ gaussian_k4              │ Imputer (K=4 windows)  │ None (Gaussian → easy) │
│ gaussian_k8              │ Imputer (K=8 windows)  │ None                   │
│ burr_m5                  │ Imputer (non-Gaussian) │ Temporal               │
│ burr_m10                 │ Imputer (non-Gaussian) │ Temporal               │
│ skeleton_structured      │ Imputer (manifold)     │ Spatial                │
│ gait_periodic            │ Temporal (periodicity) │ Imputer                │
│ low_rank_manifold        │ Imputer (low-rank)     │ None                   │
│ window_label_gaussian    │ Player (temporal)      │ Imputer                │
│ joint_subset_skeleton    │ Player (spatial)       │ Imputer                │
└──────────────────────────┴────────────────────────┴────────────────────────┘

Usage
-----
    python scripts/framework_2x2.py [--results-dir results/synthetic]

Generates:
  - results/framework_2x2_report.md   — full interpretive report
  - results/framework_2x2_table.md    — compact evidence table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Framework definitions
# ---------------------------------------------------------------------------

PLAYER_ABSTRACTION: dict[str, str] = {
    "kernelshap_zero":           "temporal",
    "kernelshap_mean":           "temporal",
    "kernelshap_marginal":       "temporal",
    "kernelshap_empirical":      "temporal",
    "kernelshap_oracle":         "temporal",
    "kernelshap_vaeac":          "temporal",
    "kernelshap_flow":           "temporal",
    "kernelshap_temporal":       "temporal",
    "timeshap":                  "temporal",   # legacy alias
    "windowshap":                "temporal",
    "kernelshap_zero_spatial":   "spatial",
    "kernelshap_oracle_spatial": "spatial",
    "kernelshap_vaeac_spatial":  "spatial",
}

IMPUTER_QUALITY: dict[str, str] = {
    "kernelshap_zero":           "off-manifold",
    "kernelshap_mean":           "off-manifold",
    "kernelshap_marginal":       "weak-on-manifold",
    "kernelshap_empirical":      "weak-on-manifold",
    "kernelshap_oracle":         "strong-on-manifold",
    "kernelshap_vaeac":          "strong-on-manifold",
    "kernelshap_flow":           "strong-on-manifold",
    "kernelshap_temporal":       "off-manifold",
    "timeshap":                  "off-manifold",
    "windowshap":                "off-manifold",
    "kernelshap_zero_spatial":   "off-manifold",
    "kernelshap_oracle_spatial": "strong-on-manifold",
    "kernelshap_vaeac_spatial":  "strong-on-manifold",
}

METHOD_LABELS: dict[str, str] = {
    "kernelshap_zero":           "KS-Zero",
    "kernelshap_mean":           "KS-Mean",
    "kernelshap_marginal":       "KS-Marginal",
    "kernelshap_empirical":      "KS-Empirical",
    "kernelshap_oracle":         "KS-Oracle",
    "kernelshap_vaeac":          "KS-VAEAC",
    "kernelshap_flow":           "KS-Flow",
    "kernelshap_temporal":       "KS-Temporal",
    "timeshap":                  "KS-Temporal",   # legacy
    "windowshap":                "WindowSHAP",
    "kernelshap_zero_spatial":   "KS-Zero-Spat",
    "kernelshap_oracle_spatial": "KS-Oracle-Spat",
    "kernelshap_vaeac_spatial":  "KS-VAEAC-Spat",
}

# For each dataset, which player abstraction is *correct* for the label function
DATASET_CORRECT_PLAYER: dict[str, str] = {
    "gaussian_k4":           "temporal",
    "gaussian_k8":           "temporal",
    "burr_m5":               "temporal",
    "burr_m10":              "temporal",
    "skeleton_structured":   "temporal",
    "gait_periodic":         "temporal",
    "low_rank_manifold":     "temporal",
    "window_label_gaussian": "temporal",
    "joint_subset_skeleton": "spatial",
}

# For each dataset, whether manifold structure matters (non-Gaussian)
DATASET_MANIFOLD_MATTERS: dict[str, bool] = {
    "gaussian_k4":           False,
    "gaussian_k8":           False,
    "burr_m5":               True,
    "burr_m10":              True,
    "skeleton_structured":   True,
    "gait_periodic":         True,
    "low_rank_manifold":     True,
    "window_label_gaussian": False,
    "joint_subset_skeleton": True,
}

# Primary attribution-quality metrics to use (higher is better for all)
PRIMARY_METRICS = ["spearman", "kendall", "topk", "ec1", "ec2", "ec3"]

# Canonical method order for tables
METHOD_ORDER = [
    "kernelshap_zero",
    "kernelshap_mean",
    "kernelshap_marginal",
    "kernelshap_empirical",
    "kernelshap_oracle",
    "kernelshap_vaeac",
    "kernelshap_flow",
    "kernelshap_temporal",
    "windowshap",
    "kernelshap_zero_spatial",
    "kernelshap_oracle_spatial",
    "kernelshap_vaeac_spatial",
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results(results_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Return ``{(dataset, classifier, method): result_dict}``."""
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rf in sorted(results_dir.glob("*/*/*/result.json")):
        parts = rf.parts
        method = parts[-2]
        clf = parts[-3]
        ds = parts[-4]
        try:
            data = json.loads(rf.read_text())
            out[(ds, clf, method)] = data
        except (json.JSONDecodeError, OSError):
            pass
    return out


def aggregate_over_classifiers(
    results: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[tuple[str, str], dict[str, float]]:
    """Average primary metrics over classifiers → ``{(dataset, method): metrics}``."""
    accum: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (ds, clf, method), data in results.items():
        key = (ds, method)
        accum.setdefault(key, []).append(data)

    out: dict[tuple[str, str], dict[str, float]] = {}
    for (ds, method), runs in accum.items():
        merged: dict[str, list[float]] = {}
        for r in runs:
            for k, v in r.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    merged.setdefault(k, []).append(float(v))
        out[(ds, method)] = {k: float(np.mean(vs)) for k, vs in merged.items()}
    return out


# ---------------------------------------------------------------------------
# Framework analysis
# ---------------------------------------------------------------------------


def _axis_a_player_analysis(
    agg: dict[tuple[str, str], dict[str, float]],
) -> str:
    """Axis A: Does correct player abstraction matter more than imputer quality?"""
    lines: list[str] = []
    lines.append("## Axis A — Player Abstraction (Temporal vs. Spatial)\n")
    lines.append(
        "**Hypothesis:** On `window_label_gaussian` (temporally localized label) "
        "temporal players beat spatial; on `joint_subset_skeleton` (spatially "
        "localized label) spatial players beat temporal.  The gap between correct "
        "and wrong player abstraction should exceed the imputer-quality gap within "
        "the correct abstraction.\n"
    )

    probe_datasets = {
        "window_label_gaussian": "temporal",
        "joint_subset_skeleton": "spatial",
    }

    for ds, correct_player in probe_datasets.items():
        lines.append(f"### Dataset: `{ds}` (correct player: **{correct_player}**)\n")
        rows: list[tuple[str, str, str, str]] = []
        for method in METHOD_ORDER:
            key = (ds, method)
            if key not in agg:
                continue
            m = agg[key]
            player = PLAYER_ABSTRACTION.get(method, "?")
            quality = IMPUTER_QUALITY.get(method, "?")
            match = "[ok]" if player == correct_player else "[xx]"
            spear = f"{m.get('spearman', float('nan')):.3f}"
            topk = f"{m.get('topk', float('nan')):.3f}"
            label = METHOD_LABELS.get(method, method)
            rows.append((f"{match} {label}", player, quality, spear, topk))

        if rows:
            lines.append("| Method | Player | Imputer | Spearman↑ | TopK↑ |")
            lines.append("|--------|--------|---------|-----------|-------|")
            for r in rows:
                lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
            lines.append("")

    return "\n".join(lines)


def _axis_b_imputer_analysis(
    agg: dict[tuple[str, str], dict[str, float]],
) -> str:
    """Axis B: Does on-manifold imputation beat off-manifold on structured datasets?"""
    lines: list[str] = []
    lines.append("## Axis B — Imputer Quality (Off-manifold vs. On-manifold)\n")
    lines.append(
        "**Hypothesis:** On datasets where the manifold has nontrivial structure "
        "(Burr, Skeleton, LowRank, GaitPeriodic) the attribution quality ordering is:\n"
        "  Off-manifold (Zero/Mean) < Weak on-manifold (Marginal/Empirical) "
        "< Strong on-manifold (Oracle/VAEAC/Flow)\n"
        "On Gaussian datasets the ordering collapses (Zero ≈ Oracle) because the "
        "zero baseline IS the conditional mean for zero-mean Gaussians.\n"
    )

    temporal_methods = [m for m in METHOD_ORDER if PLAYER_ABSTRACTION.get(m) == "temporal"]

    imputer_order = ["off-manifold", "weak-on-manifold", "strong-on-manifold"]

    for ds, manifold_matters in DATASET_MANIFOLD_MATTERS.items():
        rows: list[tuple[str, str, str]] = []
        for method in temporal_methods:
            key = (ds, method)
            if key not in agg:
                continue
            m = agg[key]
            quality = IMPUTER_QUALITY.get(method, "?")
            spear = f"{m.get('spearman', float('nan')):.3f}"
            label = METHOD_LABELS.get(method, method)
            rows.append((label, quality, spear))
        if not rows:
            continue

        marker = "🌀 manifold matters" if manifold_matters else "📐 Gaussian (flat manifold)"
        lines.append(f"### Dataset: `{ds}` — {marker}\n")
        lines.append("| Method | Imputer quality | Spearman↑ |")
        lines.append("|--------|-----------------|-----------|")
        for label, qual, spear in rows:
            lines.append(f"| {label} | {qual} | {spear} |")
        lines.append("")

    return "\n".join(lines)


def _axis_c_temporal_coherence(
    agg: dict[tuple[str, str], dict[str, float]],
) -> str:
    """Axis C: Do on-manifold imputers preserve temporal coherence?"""
    lines: list[str] = []
    lines.append("## Axis C — Temporal Coherence of On-Manifold Imputers\n")
    lines.append(
        "**Hypothesis:** VAEAC and Flow Matching generate spatially valid poses "
        "at each masked timestep but may impute each frame independently, breaking "
        "temporal continuity.  On `gait_periodic` (where temporal periodicity is "
        "the key label signal) VAEAC/Flow should under-perform the exact Oracle "
        "despite having similar spatial quality, because temporal correlation is "
        "not captured by frame-by-frame generation.\n"
    )

    temporal_datasets = ["gait_periodic", "window_label_gaussian", "burr_m5", "burr_m10"]
    compare_methods = [
        "kernelshap_zero",
        "kernelshap_oracle",
        "kernelshap_vaeac",
        "kernelshap_flow",
    ]

    for ds in temporal_datasets:
        rows: list[tuple[str, str]] = []
        for method in compare_methods:
            key = (ds, method)
            if key not in agg:
                continue
            m = agg[key]
            spear = f"{m.get('spearman', float('nan')):.3f}"
            label = METHOD_LABELS.get(method, method)
            rows.append((label, spear))
        if not rows:
            continue
        lines.append(f"### Dataset: `{ds}`\n")
        lines.append("| Method | Spearman↑ |")
        lines.append("|--------|-----------|")
        for label, spear in rows:
            lines.append(f"| {label} | {spear} |")
        lines.append("")

    return "\n".join(lines)


def _compact_evidence_table(
    agg: dict[tuple[str, str], dict[str, float]],
) -> str:
    """A compact dataset × method table of Spearman rank correlations."""
    datasets = [
        "gaussian_k4", "gaussian_k8",
        "burr_m5", "burr_m10",
        "skeleton_structured", "gait_periodic",
        "low_rank_manifold",
        "window_label_gaussian", "joint_subset_skeleton",
    ]
    methods = [m for m in METHOD_ORDER if any((ds, m) in agg for ds in datasets)]

    lines: list[str] = ["## Compact Evidence Table — Spearman Rank Correlation (↑)\n"]
    header_methods = [METHOD_LABELS.get(m, m) for m in methods]
    lines.append("| Dataset | " + " | ".join(header_methods) + " |")
    lines.append("|---------|" + "|".join(["---"] * len(methods)) + "|")

    for ds in datasets:
        correct = DATASET_CORRECT_PLAYER.get(ds, "temporal")
        manifold = "🌀" if DATASET_MANIFOLD_MATTERS.get(ds, False) else "📐"
        row = [f"{manifold} `{ds}`"]
        for method in methods:
            key = (ds, method)
            if key in agg:
                val = agg[key].get("spearman", float("nan"))
                player_ok = PLAYER_ABSTRACTION.get(method) == correct
                cell = f"{val:.3f}" if not np.isnan(val) else "—"
                if player_ok and IMPUTER_QUALITY.get(method) == "strong-on-manifold":
                    cell = f"**{cell}**"  # bold = both axes correct
                row.append(cell)
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append(
        "_Legend: **bold** = correct player abstraction AND strong on-manifold imputer. "
        "🌀 = manifold structure matters (non-Gaussian). 📐 = Gaussian (flat manifold)._\n"
    )
    return "\n".join(lines)


def _generate_report(
    agg: dict[tuple[str, str], dict[str, float]],
    results_dir: Path,
) -> str:
    n_cells = len(agg)
    datasets = sorted({ds for ds, _ in agg})
    methods = sorted({m for _, m in agg})

    header = f"""# MotionBench-XAI: 2×2 Failure Mode Framework Report

**Generated from:** `{results_dir}`
**Cells with results:** {n_cells} ({len(datasets)} datasets × {len(methods)} methods)

---

## Framework Overview

We decompose SHAP attribution quality for spatiotemporal motion data into two
orthogonal failure axes:

**Axis A — Player Abstraction:** The unit of attribution (temporal windows vs.
body joints).  When the label function is temporally localized (e.g., activity
in a specific phase), temporal players are necessary.  When the label is spatially
localized (e.g., signal in a subset of joints), spatial players are necessary.
**Wrong player abstraction is expected to be the dominant failure mode.**

**Axis B — Imputer Quality:** How masked features are filled during coalition
evaluation.  Off-manifold imputers (Zero, Mean) generate physically implausible
sequences.  On-manifold imputers (Oracle, VAEAC, Flow) sample from the true
conditional distribution.  On datasets with nontrivial manifold structure
(Burr, Skeleton, LowRank), the imputer gap is meaningful.

**Axis C — Temporal Coherence (sub-axis of B):** On-manifold imputers trained
frame-by-frame (VAEAC, Flow) may fail to capture temporal dependencies, producing
spatially valid but temporally incoherent sequences.  This is expected to hurt on
datasets where temporal periodicity carries the label signal (GaitPeriodic).

---

"""
    body = "\n---\n\n".join([
        _compact_evidence_table(agg),
        _axis_a_player_analysis(agg),
        _axis_b_imputer_analysis(agg),
        _axis_c_temporal_coherence(agg),
    ])

    return header + body


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/synthetic"),
        help="Root directory of synthetic evaluation results.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Directory to write the framework report.",
    )
    args = parser.parse_args()

    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} (dataset, classifier, method) cells.")

    agg = aggregate_over_classifiers(results)
    print(f"Aggregated to {len(agg)} (dataset, method) pairs.")

    report = _generate_report(agg, args.results_dir)

    out_report = args.out_dir / "framework_2x2_report.md"
    out_report.write_text(report)
    print(f"Report written to {out_report}")

    # Also print a compact summary to stdout
    print("\n" + "=" * 70)
    print("COMPACT EVIDENCE TABLE (Spearman, averaged over classifiers)")
    print("=" * 70)

    datasets_to_show = [
        "window_label_gaussian", "joint_subset_skeleton",
        "gait_periodic", "burr_m5", "skeleton_structured", "low_rank_manifold",
    ]
    key_methods = [
        "kernelshap_zero", "kernelshap_oracle",
        "kernelshap_temporal", "windowshap",
        "kernelshap_zero_spatial", "kernelshap_oracle_spatial",
        "kernelshap_vaeac",
    ]

    col_w = 12
    header = f"{'Dataset':<28}" + "".join(
        METHOD_LABELS.get(m, m)[:col_w].center(col_w) for m in key_methods
    )
    print(header)
    print("-" * len(header))
    for ds in datasets_to_show:
        row = f"{ds:<28}"
        for method in key_methods:
            key = (ds, method)
            if key in agg:
                val = agg[key].get("spearman", float("nan"))
                row += f"{val:^{col_w}.3f}"
            else:
                row += f"{'—':^{col_w}}"
        print(row)


if __name__ == "__main__":
    main()
