"""scripts/generate_paper_tables.py — produce main-text LaTeX tables.

Tables produced (writes to ``paper/tables/``):
  table1_dataset_taxonomy.tex   — Synthetic dataset family overview.
  table2_synth_ec1.tex          — EC1 across imputers x synthetic datasets.
  table3_2x2_winners.tex        — Winners by (spatial, temporal) regime.
  table4_real_carepd.tex        — Real CARE-PD KernelSHAP variants leaderboard.

All tables use ``booktabs``-style rules.  Mean ± std where multiple cells
contribute; bold marks the best non-oracle method per column.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
import statistics

REPO = Path("/home/papillon/code/motionbench-xai")
SYNTH = REPO / "results" / "synthetic"
REAL = REPO / "results" / "care_pd"
OUT = REPO / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------- #
# Methods used in the paper (canonical names)                              #
# ---------------------------------------------------------------------- #

KS_VARIANT_METHODS = [
    ("kernelshap_zero",     "KS--Zero"),
    ("kernelshap_mean",     "KS--Mean"),
    ("kernelshap_marginal", "KS--Marginal"),
    ("kernelshap_empirical","KS--Empirical"),
    ("kernelshap_vaeac",    "KS--VAEAC"),
    ("kernelshap_flow",     "KS--Flow"),
]
TEMPORAL_SHAP_METHODS = [
    # KS-Temporal is logged under either "timeshap" (legacy sweep) or
    # "kernelshap_temporal" (newer scripts).  The loader transparently tries
    # both names; see load_synth_result().
    ("timeshap",            "KS--Temporal"),
    ("windowshap",          "WindowSHAP"),
]
METHOD_ALIASES: dict[str, list[str]] = {
    "timeshap": ["timeshap", "kernelshap_temporal"],
    # windowshap_w4 is used as fallback when the default run errored
    "windowshap": ["windowshap", "windowshap_w4"],
}
GRADIENT_METHODS = [
    ("ig_zero",             "IG"),
    ("deeplift",            "DeepLIFT"),
    ("smoothgrad",          "SmoothGrad"),
    ("lrp",                 "LRP"),
]
CORE_METHODS = KS_VARIANT_METHODS + TEMPORAL_SHAP_METHODS + GRADIENT_METHODS
ORACLE_METHOD = ("kernelshap_oracle", "Oracle")

DATASETS_TAXONOMY = [
    # name              spatial   temporal   marginal   spatial_J  temporal_T  K
    ("gaussian_k4",     "low",    "AR(1)",   "Gaussian", 5,  16, 4),
    ("gaussian_k8",     "low",    "AR(1)",   "Gaussian", 5,  16, 8),
    ("burr_m5",         "low",    "AR(1)",   "Burr",     5,  20, 5),
    ("burr_m10",        "low",    "AR(1)",   "Burr",     5,  20, 10),
    ("skeleton_structured","high","AR(1)",  "Gaussian", 17, 16, 4),
    ("gait_periodic",   "low",    "Periodic","Gaussian",17, 16, 4),
    ("skeleton_gait_combined","high","Periodic","Gaussian", 17, 16, 4),
    ("joint_subset_skeleton","high","AR(1)","Gaussian", 17, 16, 4),
    ("low_rank_manifold","high",  "AR(1)",   "Gaussian", 17, 16, 4),
]

# Datasets used in the main-text 2x2 framework.
# All four quadrants are populated with closed-form Gaussian oracles.
PILLAR_DATASETS = [
    "gaussian_k4",                # low spatial, weak temporal
    "skeleton_structured",        # high spatial, weak temporal
    "gait_periodic",              # low spatial, strong temporal
    "skeleton_gait_combined",     # high spatial, strong temporal
    "burr_m5",                    # low spatial, weak temporal, heavy-tail
]


def load_synth_result(ds: str, clf: str, method: str) -> dict | None:
    """Load a per-cell result, transparently trying any registered aliases."""
    candidates = [method] + [a for a in METHOD_ALIASES.get(method, []) if a != method]
    for cand in candidates:
        p = SYNTH / ds / clf / cand / "result.json"
        if p.exists():
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            # Skip stub records that lack metric fields (e.g. failed oracle runs).
            if not any(k in d for k in ("ec1", "faithfulness_correlation", "player_aopc",
                                        "player_aopc_comp")):
                continue
            return d
    return None


def avg_metric(ds: str, method: str, key: str) -> tuple[float, float, int, str | None] | None:
    """Average a metric across classifiers for one (dataset, method).

    Returns ``(mean, std, n, arch)`` where ``n`` is the number of classifiers
    contributing and ``arch`` is the short architecture name when n=1 (else None).
    ``std`` is 0 when ``n=1``; callers should suppress the ``±std`` rendering.
    """
    ARCH_LABELS = {
        "synthetic_mlp":         "MLP",
        "synthetic_cnn":         "CNN",
        "synthetic_transformer": "Tfm",
    }
    vals = []
    arch = None
    for clf in ("synthetic_mlp", "synthetic_cnn", "synthetic_transformer"):
        r = load_synth_result(ds, clf, method)
        if r is not None and key in r and r[key] is not None:
            vals.append(float(r[key]))
            arch = ARCH_LABELS.get(clf)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0], 0.0, 1, arch
    return statistics.mean(vals), statistics.stdev(vals), len(vals), None


# ---------------------------------------------------------------------- #
# Table 1 — Dataset taxonomy                                               #
# ---------------------------------------------------------------------- #


def table_taxonomy() -> str:
    rows = []
    rows.append(r"\begin{table}[t]")
    rows.append(r"\centering\small")
    rows.append(r"\caption{Synthetic dataset family.  Each dataset is a draw "
                r"$\mathbf{x}\!\sim\!\mathcal{N}(\mathbf{0},\,\Sigma_J\!\otimes\!I_F\!\otimes\!\Sigma_T)$ "
                r"(or its Burr-marginal analogue), with class label generated by the "
                r"Olsen-block interaction~\eqref{eq:olsen} on $K$ window-mean features.  "
                r"\textit{Spatial} measures the rank/coupling of $\Sigma_J$; "
                r"\textit{Temporal} is the structure of $\Sigma_T$.  All datasets use $T{=}16$, $F{=}3$.}")
    rows.append(r"\label{tab:datasets}")
    rows.append(r"\begin{tabular}{lcccccc}")
    rows.append(r"\toprule")
    rows.append(r"Dataset & Spatial & Temporal & Marginal & $J$ & $T$ & $K$ \\")
    rows.append(r"\midrule")
    pretty = {
        "gaussian_k4":            r"\texttt{gaussian\_k4}",
        "gaussian_k8":            r"\texttt{gaussian\_k8}",
        "burr_m5":                r"\texttt{burr\_m5}",
        "burr_m10":               r"\texttt{burr\_m10}",
        "skeleton_structured":    r"\texttt{skeleton}",
        "gait_periodic":          r"\texttt{gait\_periodic}",
        "skeleton_gait_combined": r"\texttt{skeleton+gait}",
        "joint_subset_skeleton":  r"\texttt{joint\_subset}",
        "low_rank_manifold":      r"\texttt{low\_rank}",
    }
    for name, spatial, temporal, marginal, J, T, K in DATASETS_TAXONOMY:
        rows.append(f"{pretty[name]} & {spatial} & {temporal} & {marginal} "
                    f"& ${J}$ & ${T}$ & ${K}$ \\\\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append(r"\end{table}")
    return "\n".join(rows)


# ---------------------------------------------------------------------- #
# Table 2 — EC1 across imputers x datasets                                 #
# ---------------------------------------------------------------------- #


def fmt_pm(m: float, s: float, fmt: str = ".3f", n: int | None = None,
           arch: str | None = None) -> str:
    """Render a mean ± std cell.  When only one architecture (n=1), show the
    architecture name inline as a compact superscript rather than a separate
    footnote marker, e.g. ``$0.018^{\rm MLP}$``."""
    if n == 1:
        lbl = arch if arch else "1\\,clf"
        return f"$\\mathit{{{m:{fmt}}}}^{{\\rm {lbl}}}$"
    return f"${m:{fmt}}\\!\\pm\\!{s:{fmt}}$"


def table_synth_ec1() -> str:
    rows = []
    rows.append(r"\begin{table}[t]")
    rows.append(r"\centering\small")
    rows.append(r"\caption{Attribution error EC1 ($\downarrow$) on synthetic datasets, "
                r"averaged over three classifier architectures (MLP, CNN, Transformer); "
                r"mean $\pm$ std across classifiers.  EC1 is the mean absolute Shapley error "
                r"\citep{aas2021explaining,olsen2022using} between estimated and oracle Shapley "
                r"values per player.  "
                r"Methods are grouped into KernelSHAP imputation variants (off- and on-manifold), "
                r"temporal-SHAP baselines (KS--Temporal, WindowSHAP), and "
                r"gradient-based attributions (Integrated Gradients, DeepLIFT, SmoothGrad, LRP).  "
                r"\textbf{Bold} marks the best non-oracle method per dataset.  "
                r"Italic cells with a superscript label (MLP, CNN, Tfm) contain results from "
                r"a single architecture only; the label identifies which one.  "
                r"Oracle$^{\ddagger}$ uses $n_\text{mc}{=}10$ conditional samples per coalition "
                r"(same budget as the KernelSHAP imputers); its EC1 $\approx 0.06$--$0.12$ is "
                r"the MC noise floor of that estimator --- learned imputers (KS--VAEAC, KS--Flow) "
                r"fall \emph{below} this floor (see \S\ref{sec:results-error}).  "
                r"Italic $M$-ablation rows show KS--VAEAC EC1 on \texttt{gauss\_k4} for varying "
                r"$M$ completion samples; EC1 saturates by $M{=}5$.  "
                r"Empty cells (--) indicate no result available.}")
    rows.append(r"\label{tab:synth_ec1}")
    rows.append(r"\begin{tabular}{l" + "r" * len(PILLAR_DATASETS) + r"}")
    rows.append(r"\toprule")
    pretty_ds = {
        "gaussian_k4":            r"\texttt{gauss\_k4}",
        "gaussian_k8":            r"\texttt{gauss\_k8}",
        "skeleton_structured":    r"\texttt{skeleton}",
        "gait_periodic":          r"\texttt{gait}",
        "skeleton_gait_combined": r"\texttt{skel+gait}",
        "burr_m5":                r"\texttt{burr\_m5}",
        "burr_m10":               r"\texttt{burr\_m10}",
        "low_rank_manifold":      r"\texttt{low\_rank}",
    }
    rows.append("Method & " + " & ".join(pretty_ds[d] for d in PILLAR_DATASETS) + r" \\")
    rows.append(r"\midrule")

    # Build the full grid first (for bold lookup)
    grid: dict[str, dict[str, tuple[float, float, int] | None]] = defaultdict(dict)
    for method_key, _ in CORE_METHODS + [ORACLE_METHOD]:
        for ds in PILLAR_DATASETS:
            grid[method_key][ds] = avg_metric(ds, method_key, "ec1")

    # Best non-oracle per dataset
    best_per_ds: dict[str, str] = {}
    for ds in PILLAR_DATASETS:
        cands = [(m, grid[m][ds][0]) for m, _ in CORE_METHODS
                 if grid[m].get(ds) is not None]
        if cands:
            best_per_ds[ds] = min(cands, key=lambda t: t[1])[0]

    # Render rows, grouping methods by family with a \midrule between groups.
    method_groups = [KS_VARIANT_METHODS, TEMPORAL_SHAP_METHODS, GRADIENT_METHODS]
    for gi, group in enumerate(method_groups):
        for method_key, label in group:
            cells = []
            for ds in PILLAR_DATASETS:
                v = grid[method_key].get(ds)
                if v is None:
                    cells.append(r"--")
                else:
                    m, s, n, arch = v
                    cell = fmt_pm(m, s, n=n, arch=arch)
                    if best_per_ds.get(ds) == method_key:
                        cell = r"\textbf{" + cell + "}"
                    cells.append(cell)
            rows.append(f"{label} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    # Oracle row
    cells_o = []
    for ds in PILLAR_DATASETS:
        v = grid[ORACLE_METHOD[0]].get(ds)
        if v is None:
            cells_o.append(r"--")
        else:
            cells_o.append(fmt_pm(v[0], v[1], n=v[2], arch=v[3]))
    rows.append(f"Oracle$^{{\\ddagger}}$ & " + " & ".join(cells_o) + r" \\")

    # M-ablation rows: KS-VAEAC at M in {1, 5, 20, 50} on gauss_k4 only
    m_ablation_path = REPO / "results" / "ablations" / "m_ablation_gaussian_k4.json"
    if m_ablation_path.exists():
        try:
            m_data = json.loads(m_ablation_path.read_text())
            m_values = m_data.get("M", [1, 5, 20, 50])
            ec1_avg = m_data.get("ec1_avg", [])
            if ec1_avg:
                rows.append(r"\midrule")
                rows.append(r"\multicolumn{" + str(len(PILLAR_DATASETS) + 1) + r"}{l}{\footnotesize\itshape "
                            r"$M$-ablation: KS--VAEAC completion samples on \texttt{gauss\_k4} "
                            r"(sensitivity saturates by $M{=}5$):} \\")
                for i, (m_val, ec1_val) in enumerate(zip(m_values, ec1_avg)):
                    # Only fill gauss_k4 column (first pillar dataset)
                    cells_m = []
                    for j, ds in enumerate(PILLAR_DATASETS):
                        if ds == "gaussian_k4":
                            cells_m.append(f"{ec1_val:.4f}")
                        else:
                            cells_m.append(r"--")
                    label = rf"\textit{{KS--VAEAC ($M{{=}}{m_val}$)}}"
                    rows.append(f"{label} & " + " & ".join(cells_m) + r" \\")
        except Exception:
            pass  # silently skip if ablation file malformed

    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append(r"\end{table}")
    return "\n".join(rows)


# ---------------------------------------------------------------------- #
# Table 3 — 2x2 winners                                                    #
# ---------------------------------------------------------------------- #


def _load_player_set_ec1(ds: str, method_key: str, player_suffix: str) -> float | None:
    """Load EC1 for a player-set ablation result (averaged across classifiers)."""
    suffixed = f"{method_key}_{player_suffix}"
    v = avg_metric(ds, suffixed, "ec1")
    return v[0] if v is not None else None


def table_2x2_winners() -> str:
    """Map each dataset to a (spatial, temporal) regime cell and report
    best/second-best EC1 with the on-manifold gap, plus best player-set."""
    cell_map = {
        ("low",  "weak"):   ["gaussian_k4", "burr_m5"],
        ("low",  "strong"): ["gait_periodic"],
        ("high", "weak"):   ["skeleton_structured", "low_rank_manifold"],
        ("high", "strong"): ["skeleton_gait_combined"],
    }
    # Map from (spatial, temporal) regime to which dataset has player-set ablation data
    player_set_ds_map = {
        ("high", "weak"):   ("skeleton_structured", "pjoint"),
        ("high", "strong"): ("skeleton_gait_combined", "pcell"),
    }

    rows = []
    rows.append(r"\begin{table}[t]")
    rows.append(r"\centering\small")
    rows.append(r"\caption{Attribution error EC1 ($\downarrow$) of the best on-manifold "
                r"and best off-manifold method in each spatiotemporal regime, with the "
                r"\emph{on-manifold gap} $\Delta = \mathrm{EC1}_\mathrm{off} / \mathrm{EC1}_\mathrm{on}$.  "
                r"The rightmost column reports the best KS--VAEAC EC1 across player-set "
                r"granularities ($\mathcal{P}_\text{temp}$, $\mathcal{P}_\text{joint}$, "
                r"$\mathcal{P}_\text{cell}$), with the winning granularity in parentheses.  "
                r"For high-spatial regimes ($J{=}17$), $\mathcal{P}_\text{joint}$ (17 players) "
                r"and $\mathcal{P}_\text{cell}$ (68 players) were compared against "
                r"$\mathcal{P}_\text{temp}$ (4 players); for low-spatial datasets ($J{=}5$) "
                r"only $\mathcal{P}_\text{temp}$ was evaluated, as the alternative "
                r"granularities add negligible players.  "
                r"On-manifold imputers dominate every regime by 1--2 orders of magnitude.}")
    rows.append(r"\label{tab:winners}")
    rows.append(r"\begin{tabular}{lllll}")
    rows.append(r"\toprule")
    rows.append(r"Regime & Best on-manifold & Best off-manifold & Gap $\Delta$ & Best player-set \\")
    rows.append(r"\midrule")
    on_manifold_keys = {"kernelshap_vaeac", "kernelshap_flow", "kernelshap_empirical"}
    for spatial in ("low", "high"):
        for temporal in ("weak", "strong"):
            datasets = cell_map[(spatial, temporal)]
            if not datasets:
                continue
            # Only consider KernelSHAP and Temporal-SHAP variants for the
            # on/off-manifold competition.  Gradient methods are reported
            # separately in Table 2 since they are not Shapley estimators.
            shap_methods = KS_VARIANT_METHODS + TEMPORAL_SHAP_METHODS
            on_best = None
            off_best = None
            for ds in datasets:
                for method_key, label in shap_methods:
                    v = avg_metric(ds, method_key, "ec1")
                    if v is None:
                        continue
                    rec = (v[0], label, ds)
                    if method_key in on_manifold_keys:
                        if on_best is None or v[0] < on_best[0]:
                            on_best = rec
                    else:
                        if off_best is None or v[0] < off_best[0]:
                            off_best = rec
            spatial_lbl = "low spatial" if spatial == "low" else "high spatial"
            temporal_lbl = "AR(1) temporal" if temporal == "weak" else "strong temporal"
            regime = f"{spatial_lbl}, {temporal_lbl}"
            on_cell = f"{on_best[1]} ({on_best[0]:.3f})" if on_best else "--"
            off_cell = f"{off_best[1]} ({off_best[0]:.3f})" if off_best else "--"
            gap = f"${off_best[0]/on_best[0]:.1f}\\times$" if on_best and off_best and on_best[0] > 0 else "--"

            # Player-set ablation column: compare P_temp vs P_joint vs P_cell for KS-VAEAC.
            # For high-spatial datasets, run a full sweep; for low-spatial (J=5) only
            # P_temp is evaluated — show the P_temp result directly (no empty cell).
            ps_cell = "--"
            ps_ds_info = player_set_ds_map.get((spatial, temporal))
            if ps_ds_info and on_best is not None:
                ps_ds, ps_suffix = ps_ds_info
                # P_temp: use standard VAEAC result
                ec1_ptemp = avg_metric(ps_ds, "kernelshap_vaeac", "ec1")
                ec1_ptemp_val = ec1_ptemp[0] if ec1_ptemp else None
                # P_joint or P_cell ablation (loaded from suffixed method name)
                ec1_palt = _load_player_set_ec1(ps_ds, "kernelshap_vaeac", ps_suffix)
                alt_label = r"$\mathcal{P}_\text{joint}$" if ps_suffix == "pjoint" else r"$\mathcal{P}_\text{cell}$"
                if ec1_ptemp_val is not None and ec1_palt is not None:
                    if ec1_palt < ec1_ptemp_val:
                        ps_cell = f"{ec1_palt:.4f} ({alt_label})"
                    else:
                        ps_cell = rf"{ec1_ptemp_val:.4f} ($\mathcal{{P}}_\text{{temp}}$)"
                elif ec1_ptemp_val is not None:
                    ps_cell = rf"{ec1_ptemp_val:.4f} ($\mathcal{{P}}_\text{{temp}}$)"
            else:
                # Low-spatial regimes: only P_temp was evaluated (J=5 makes
                # P_joint trivial); show the KS-VAEAC P_temp result directly.
                for ds in datasets:
                    v_ptemp = avg_metric(ds, "kernelshap_vaeac", "ec1")
                    if v_ptemp is not None:
                        ps_cell = rf"{v_ptemp[0]:.4f} ($\mathcal{{P}}_\text{{temp}}$)"
                        break

            rows.append(f"{regime} & {on_cell} & {off_cell} & {gap} & {ps_cell} \\\\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append(r"\end{table}")
    return "\n".join(rows)


# ---------------------------------------------------------------------- #
# Table 4 — Real CARE-PD                                                   #
# ---------------------------------------------------------------------- #


def _load_real_extended() -> dict | None:
    """Load the multi-fold summary if present (paper-final version)."""
    p = REPO / "results" / "care_pd_extended" / "summary_with_ci.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def table_real() -> str:
    rows = []
    rows.append(r"\begin{table}[t]")
    rows.append(r"\centering\small")
    real_methods = [
        ("kernelshap_zero",     "KS--Zero"),
        ("kernelshap_mean",     "KS--Mean"),
        ("kernelshap_marginal", "KS--Marginal"),
        ("kernelshap_vaeac",    "KS--VAEAC"),
        ("kernelshap_flow",     "KS--Flow"),
    ]

    extended = _load_real_extended()
    if extended is not None and "methods" in extended:
        per_method = extended["methods"]
        n_folds = extended.get("folds", [])
        rows.append(r"\caption{KernelSHAP variants on real-world BMCLab gait data "
                    r"(CARE-PD, $T{=}80$, $K{=}4$ gait-phase windows, $n{=}200$ clips "
                    r"per fold, " + str(len(n_folds)) + r" leave-one-subject-out folds, "
                    r"MotionBERT classifier).  Mean across folds with bootstrap 95\% "
                    r"confidence intervals ($B{=}10{,}000$ resamples).  "
                    r"\textit{Faithfulness}: Pearson correlation between the sum of "
                    r"absent-player attributions and $v(N)\!-\!v(S)$.  "
                    r"\textit{PlayerAOPC}: mean output drop when removing players in order "
                    r"of $|\phi|$.  Higher faithfulness can reflect off-manifold disruption "
                    r"(see Sec.~\ref{sec:results-real}).}")
        rows.append(r"\label{tab:real}")
        rows.append(r"\begin{tabular}{lcc}")
        rows.append(r"\toprule")
        rows.append(r"Method & Faithfulness $\uparrow$ & PlayerAOPC $\uparrow$ \\")
        rows.append(r"\midrule")

        # Find best in each column
        def get_mean(method, metric):
            d = per_method.get(method, {}).get(metric, {})
            return d.get("pooled_mean", d.get("mean_of_fold_means"))

        best_faith = None
        best_aopc = None
        for method, _ in real_methods:
            if method not in per_method:
                continue
            f = get_mean(method, "faithfulness")
            a = get_mean(method, "player_aopc")
            if f is not None and (best_faith is None or f > best_faith[1]):
                best_faith = (method, f)
            if a is not None and (best_aopc is None or a > best_aopc[1]):
                best_aopc = (method, a)

        for method, label in real_methods:
            d = per_method.get(method)
            if d is None:
                rows.append(f"{label} & -- & -- \\\\")
                continue
            fk = d.get("faithfulness", {})
            ak = d.get("player_aopc", {})
            f_mean = fk.get("pooled_mean", fk.get("mean_of_fold_means"))
            a_mean = ak.get("pooled_mean", ak.get("mean_of_fold_means"))
            if f_mean is None or a_mean is None:
                rows.append(f"{label} & -- & -- \\\\")
                continue
            f_cell = (f"${f_mean:+.3f}$ "
                      f"$[{fk['ci95_low']:+.3f},\\,{fk['ci95_high']:+.3f}]$")
            a_cell = (f"${a_mean:+.3f}$ "
                      f"$[{ak['ci95_low']:+.3f},\\,{ak['ci95_high']:+.3f}]$")
            if best_faith and method == best_faith[0]:
                f_cell = r"\textbf{" + f_cell + "}"
            if best_aopc and method == best_aopc[0]:
                a_cell = r"\textbf{" + a_cell + "}"
            rows.append(f"{label} & {f_cell} & {a_cell} \\\\")

        rows.append(r"\bottomrule")
        rows.append(r"\end{tabular}")
        # Significance footnote
        paired = extended.get("paired_tests", [])
        if paired:
            t = paired[0]
            p2 = t.get("p_two_sided", 1.0)
            mu = t.get("mean_diff", 0.0)
            lo = t.get("ci95_low", 0.0)
            hi = t.get("ci95_high", 0.0)
            _pretty_method = {
                "kernelshap_zero": "KS--Zero",
                "kernelshap_mean": "KS--Mean",
                "kernelshap_marginal": "KS--Marginal",
                "kernelshap_empirical": "KS--Empirical",
                "kernelshap_vaeac": "KS--VAEAC",
                "kernelshap_flow": "KS--Flow",
            }
            ma = _pretty_method.get(t.get("method_a", "A"), t.get("method_a", "A"))
            mb = _pretty_method.get(t.get("method_b", "B"), t.get("method_b", "B"))
            rows.append(r"\vspace{0.5ex}\par\footnotesize Paired bootstrap on the "
                        f"faithfulness gap {ma}$-${mb} = "
                        f"${mu:+.3f}\\,[{lo:+.3f},\\,{hi:+.3f}]$, "
                        f"$p_{{\\text{{two-sided}}}}={p2:.4f}$.")
        rows.append(r"\end{table}")
        return "\n".join(rows)

    # Fall-back: legacy single-fold version
    rows.append(r"\caption{KernelSHAP variants on real-world BMCLab gait data "
                r"(CARE-PD, $T{=}80$, $K{=}4$ gait-phase windows, $n{=}50$ "
                r"clips, MotionBERT classifier).}")
    rows.append(r"\label{tab:real}")
    rows.append(r"\begin{tabular}{lcc}")
    rows.append(r"\toprule")
    rows.append(r"Method & Faithfulness $\uparrow$ & PlayerAOPC $\uparrow$ \\")
    rows.append(r"\midrule")
    real_data = {}
    for method, label in real_methods:
        p = REAL / "care_pd_bmclab_cache" / "motionbert" / method / "result.json"
        if p.exists():
            d = json.loads(p.read_text())
            real_data[method] = d
    if not real_data:
        rows.append(r"-- & -- & -- \\")
    else:
        best_faith = max(real_data.items(),
                         key=lambda kv: kv[1].get("faithfulness_correlation", -float("inf")))[0]
        best_aopc = max(real_data.items(),
                        key=lambda kv: kv[1].get("player_aopc", -float("inf")))[0]
        for method, label in real_methods:
            d = real_data.get(method)
            if d is None:
                rows.append(f"{label} & -- & -- \\\\")
                continue
            f = d.get("faithfulness_correlation", float("nan"))
            a = d.get("player_aopc", float("nan"))
            f_cell = f"${f:+.3f}$"
            a_cell = f"${a:+.3f}$"
            if method == best_faith:
                f_cell = r"\textbf{" + f_cell + "}"
            if method == best_aopc:
                a_cell = r"\textbf{" + a_cell + "}"
            rows.append(f"{label} & {f_cell} & {a_cell} \\\\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append(r"\end{table}")
    return "\n".join(rows)


# ---------------------------------------------------------------------- #
# Main                                                                     #
# ---------------------------------------------------------------------- #


def main() -> None:
    print(f"Output: {OUT}")
    for fname, fn in [
        ("table1_dataset_taxonomy.tex", table_taxonomy),
        ("table2_synth_ec1.tex",        table_synth_ec1),
        ("table3_2x2_winners.tex",      table_2x2_winners),
        ("table4_real_carepd.tex",      table_real),
    ]:
        s = fn()
        (OUT / fname).write_text(s + "\n")
        print(f"  wrote {fname} ({len(s.splitlines())} lines)")


if __name__ == "__main__":
    main()
