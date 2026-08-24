"""scripts/make_paper_tables.py — Regenerate the paper's tables from canonical results.

Every number in the paper's result tables regenerates from the canonical
JSON files in ``results/canonical/`` — no experiment rerun, no other input.
Emits booktabs LaTeX into ``--out`` (default ``results/paper_tables/``):

  tab_temporal / tab_spatial / tab_cells       Tables 2-4 (synthetic EC3)
  tab_faith / tab_aopc                         Tables 5-6 (real data)
  tab_app_ec1_{temporal,spatial,cells}         EC1 companions of Tables 2-4
  tab_app_cross_{temporal,spatial,cells}       cross-graded (all-conditional)
  tab_app_perclf_temporal                      per-classifier EC3 breakdown
  tab_app_clf_gate                             classifier gate (accuracies)
  tab_app_real_cells                           joint x window cells, real data
  tab_app_imputer_gate                         hide-one-recover capability gate
  tab_app_floors                               grading-noise floors

Conventions (ported from the paper workspace's generators):

  - booktabs only, no vertical rules;
  - values as mean {\\scriptsize$\\pm$ sd} via the paper's ``\\msd`` macro;
  - synthetic aggregation over the classifier architectures passing the
    accuracy/generalization gate (``GATED_OUT``; the gate itself ships in
    ``results/canonical/classifier_gate.json``);
  - game-matched grading: each method's EC values are read from its own
    fill family's ground-truth block (``meta.game_assignment``);
  - per-column winner in ``\\textbf`` (near-null cells excluded);
  - near-null cells (periodic data x conditional game) in gray;
  - real-data spread across data folds.

Usage::

    PYTHONPATH=. python scripts/make_paper_tables.py
    PYTHONPATH=. python scripts/make_paper_tables.py --out /tmp/tables
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL = REPO_ROOT / "results" / "canonical"

DS = ["gauss_k4", "skeleton", "burr_m5", "gait", "skel+gait"]
CLF = ["mlp", "cnn", "transformer"]
DS_TEX = {d: d.replace("_", r"\_") for d in DS}

# Classifier gate (results/canonical/classifier_gate.json): a classifier is
# explained only if (G1) held-out accuracy >= chance + 0.05 and (G2)
# train-test gap <= 0.30.  Synthetic: gait/cnn and skel+gait/cnn fail G2.
GATED_OUT = {("gait", "cnn"), ("skel+gait", "cnn")}

# Method rows (label, key) per table family.
T_STD = [
    ("KS-Zero", "ks_zero"),
    ("KS-Mean", "ks_mean"),
    ("KS-Marginal", "ks_marginal"),
    ("WindowSHAP-S", "windowshap_s"),
    ("WindowSHAP-D", "windowshap_d"),
    ("TimeSHAP", "timeshap"),
    ("KS-Empirical", "ks_empirical"),
]
T_CND = [
    ("KS-VAEAC", "ks_vaeac"),
    ("KS-Flow", "ks_flow"),
    ("KS-Gauss (closed form)", "ks_shapr"),
]
MAIN_T_STD = [
    ("KS-Marginal", "ks_marginal"),
    ("WindowSHAP-S", "windowshap_s"),
    ("WindowSHAP-D", "windowshap_d"),
    ("TimeSHAP", "timeshap"),
    ("KS-Empirical", "ks_empirical"),
]
P_STD = [
    ("KS-Zero", "ks_zero"),
    ("KS-Mean", "ks_mean"),
    ("KS-Marginal", "ks_marginal"),
    ("KS-Empirical", "ks_empirical"),
]
MAIN_P_STD = [("KS-Marginal", "ks_marginal"), ("KS-Empirical", "ks_empirical")]
P_CND = T_CND

# Near-null cells: on the periodic datasets the conditional game itself
# carries almost no rankable signal.
NEARNULL = tuple((d, k) for d in ("gait", "skel+gait") for k in ("ks_vaeac", "ks_flow", "ks_shapr"))
NEARNULL_COLS = ("gait", "skel+gait")  # cross tables: the cond target itself

METH_REAL = [
    ("KS-Zero", "kernelshap_zero"),
    ("KS-Mean", "kernelshap_mean"),
    ("KS-Marginal", "kernelshap_marginal"),
    ("KS-VAEAC", "kernelshap_vaeac"),
    ("KS-Flow", "kernelshap_flow"),
]

GATE_NOTE = (
    "mean $\\pm$ spread across the gate-passing classifier "
    "architectures (Appendix~\\ref{app:clf_gate})"
)
CROSS_STD_HDR = "Standard fills, graded against the on-manifold ground truth"
CROSS_CND_HDR = (
    "Generative fills, graded against the on-manifold ground truth (as in the main tables)"
)


def clfs_for(ds):
    """Gate-passing classifier architectures for one synthetic dataset."""
    return [c for c in CLF if (ds, c) not in GATED_OUT]


def cell(v, sd, best=False, gray=False, prec=2):
    """Format one table cell as ``mean \\msd{sd}`` with winner/near-null styling.

    Args:
        v: Cell mean.
        sd: Spread (omitted when ``None``).
        best: Set the mean in ``\\textbf``.
        gray: Wrap the whole cell in ``\\textcolor{gray}``.
        prec: Decimal places.

    Returns:
        LaTeX cell string.
    """
    pm = f" \\msd{{{sd:.{prec}f}}}" if sd is not None else ""
    s = f"\\textbf{{{v:.{prec}f}}}{pm}" if best else f"{v:.{prec}f}{pm}"
    return f"\\textcolor{{gray}}{{{v:.{prec}f}{pm}}}" if gray else s


class Sources:
    """Loaded canonical files plus the derived per-cell accessors.

    Args:
        canonical: Directory holding the ``results/canonical`` JSON files.
    """

    def __init__(self, canonical: Path) -> None:
        """Load every canonical JSON the tables draw from."""
        st = json.loads((canonical / "synthetic_temporal.json").read_text())
        self.cells = st["cells"]
        self.game_of = st["meta"]["game_assignment"]
        self.e10_tables = json.loads((canonical / "synthetic_players.json").read_text())
        self.floors = self.e10_tables["meta"]["grading_floors"]["floors"]
        rt = json.loads((canonical / "real_tracks.json").read_text())
        self.tracks = rt["tracks"]
        self.carepd_gate = rt["carepd_gate"]
        self.players = json.loads((canonical / "real_players.json").read_text())["tracks"]
        self.ptbxl_cells = json.loads((canonical / "real_ptbxl_cells.json").read_text())["tracks"]
        self.clf_gate = json.loads((canonical / "classifier_gate.json").read_text())
        self.imputer_gate = json.loads((canonical / "imputer_gate.json").read_text())["tracks"]

    # -- synthetic temporal (per-classifier cells, gate-aware aggregation) --

    def gm(self, k, ds, game, met):
        """(mean, spread) of one temporal method over gate-passing classifiers.

        Args:
            k: Method key (e.g. ``"ks_marginal"``).
            ds: Dataset name.
            game: Ground-truth block (``"cond"`` / ``"marg"``); pass
                ``self.game_of[k]`` for game-matched grading.
            met: Metric key (``"ec1"`` / ``"ec3"``).

        Returns:
            ``(mean, std-or-None)``.
        """
        v = [self.cells[f"{ds}/{c}"]["methods"][k][game][met] for c in clfs_for(ds)]
        return float(np.mean(v)), (float(np.std(v)) if len(v) > 1 else None)

    # -- synthetic players (spatial / cells), incl. cross-graded blocks --

    def e10(self, ps, k, ds, met, cross=False):
        """(mean, spread) of one player-set method over gate-passing classifiers.

        Args:
            ps: Player set (``"joint"`` / ``"cell"``).
            k: Method key.
            ds: Dataset name.
            met: Metric key.
            cross: Read the record's ``cross`` block (all-conditional grading).

        Returns:
            ``(mean, std-or-None)``.
        """
        recs = [self.e10_tables["tables"][ps]["per_cell"][k][ds][c] for c in clfs_for(ds)]
        v = [(r["cross"][met] if cross else r[met]) for r in recs]
        return float(np.mean(v)), (float(np.std(v)) if len(v) > 1 else None)

    # -- real tracks --

    def real_val(self, src, tk, m, metric):
        """Pooled metric value of one real-data track cell."""
        return src[tk][m][metric]

    def real_sd(self, src, tk, m, metric):
        """Across-fold spread of one real-data track cell."""
        folds = [f[metric] for f in src[tk][m].get("per_fold", {}).values()]
        return float(np.std(folds)) if len(folds) > 1 else 0.0


def synth_style_table(
    out,
    fname,
    caption,
    label,
    std_rows,
    cond_rows,
    valfn,
    oracle_fn,
    nearnull=(),
    nn_cols=(),
    prec=2,
    std_hdr="Standard fills (graded against the standard ground truth)",
    cnd_hdr="Generative fills (graded against the on-manifold ground truth)",
):
    """Emit one synthetic-suite table (methods x datasets, KS-Oracle reference row).

    Args:
        out: Output directory.
        fname: Output file name.
        caption: LaTeX caption.
        label: LaTeX label.
        std_rows: Standard-fill (label, key) rows.
        cond_rows: Generative-fill (label, key) rows.
        valfn: ``(key, ds) -> (mean, std)``.
        oracle_fn: ``(ds) -> (mean, std)`` for the KS-Oracle reference row.
        nearnull: ``(ds, key)`` cells rendered gray and excluded from winners.
        nn_cols: Datasets whose whole column is gray with no winner marked.
        prec: Decimal places.
        std_hdr: Group header over the standard-fill block.
        cnd_hdr: Group header over the generative-fill block.
    """
    keys = [k for _, k in std_rows + cond_rows]
    win = {}
    for ds in DS:
        pool = [k for k in keys if (ds, k) not in nearnull]
        win[ds] = None if ds in nn_cols else min(pool, key=lambda k: valfn(k, ds)[0])
    L = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        "\\begin{tabular}{l" + "c" * len(DS) + "}",
        "\\toprule",
        "Method & " + " & ".join(f"\\texttt{{{DS_TEX[d]}}}" for d in DS) + r" \\",
        "\\midrule",
        rf"\multicolumn{{{len(DS) + 1}}}{{l}}{{\emph{{{std_hdr}}}}} \\",
    ]
    for block, hdr in ((std_rows, None), (cond_rows, cnd_hdr)):
        if hdr is not None:
            L.append("\\addlinespace")
            L.append(rf"\multicolumn{{{len(DS) + 1}}}{{l}}{{\emph{{{hdr}}}}} \\")
        for lab, k in block:
            row = [lab]
            for d in DS:
                v, sd = valfn(k, d)
                row.append(
                    cell(
                        v,
                        sd,
                        best=(win[d] == k),
                        gray=((d, k) in nearnull or d in nn_cols),
                        prec=prec,
                    )
                )
            L.append(" & ".join(row) + r" \\")
    L.append("\\midrule")
    orow = ["KS-Oracle (reference)"]
    for d in DS:
        v, sd = oracle_fn(d)
        orow.append(cell(v, sd, gray=(d in nn_cols), prec=prec))
    L.append(" & ".join(orow) + r" \\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / fname).write_text("\n".join(L) + "\n")
    print("wrote", fname)


def real_table(out, src_cols, fname, metric, caption, label):
    """Emit one real-data table (methods x tracks, fold spread, per-column winner).

    Args:
        out: Output directory.
        src_cols: ``(S, source, track key, header line 1, header line 2)``
            column tuples, temporal tracks first.
        fname: Output file name.
        metric: ``"faith"`` or ``"aopc"``.
        caption: LaTeX caption.
        label: LaTeX label.
    """
    S = src_cols[0][0]
    cols = [c[1:] for c in src_cols]
    n_temporal = 4  # first four columns are temporal players
    win = {}
    for src, tk, *_ in cols:
        win[tk] = max((m for _, m in METH_REAL), key=lambda m: S.real_val(src, tk, m, metric))
    L = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{l" + "c" * len(cols) + "}",
        "\\toprule",
        f" & \\multicolumn{{{n_temporal}}}{{c}}{{\\emph{{Temporal players}}}} "
        f"& \\multicolumn{{{len(cols) - n_temporal}}}{{c}}{{\\emph{{Spatial players}}}} \\\\",
        f"\\cmidrule(lr){{2-{n_temporal + 1}}}\\cmidrule(lr){{{n_temporal + 2}-{len(cols) + 1}}}",
        "Method & "
        + " & ".join(f"\\shortstack{{{c[2]}\\\\{{\\scriptsize {c[3]}}}}}" for c in cols)
        + r" \\",
        "\\midrule",
    ]
    for lab, m in METH_REAL:
        if m == "kernelshap_vaeac":
            L.append("\\addlinespace")
        row = [lab]
        for src, tk, *_ in cols:
            v, sd = S.real_val(src, tk, m, metric), S.real_sd(src, tk, m, metric)
            row.append(cell(v, sd, best=(win[tk] == m)))
        L.append(" & ".join(row) + r" \\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / fname).write_text("\n".join(L) + "\n")
    print("wrote", fname)


def perclf_table(out, S):
    """Emit the per-classifier temporal EC3 breakdown (gated rows dagger-marked)."""
    pc_methods = T_STD + T_CND + [("KS-Oracle", "ks_oracle")]
    L = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\textbf{Per-classifier EC3, temporal players} (game-matched). "
        "The two $\\dagger$ rows fail the accuracy/generalization gate "
        "(Appendix~\\ref{app:clf_gate}) and are excluded from every aggregated "
        "table; they are shown here for completeness. Per-classifier values for "
        "the spatial and cell player sets ship with the canonical result files "
        "in the code release.}",
        "\\label{tab:app-perclf}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{ll" + "c" * len(pc_methods) + "}",
        "\\toprule",
        "Dataset & Classifier & "
        + " & ".join(
            f"\\shortstack{{{lab.replace(' (closed form)', '')}}}" for lab, _ in pc_methods
        )
        + r" \\",
        "\\midrule",
    ]
    for ds in DS:
        for ci, c in enumerate(CLF):
            gated = (ds, c) in GATED_OUT
            name = f"\\texttt{{{DS_TEX[ds]}}}" if ci == 0 else ""
            row = [name, c + ("$^\\dagger$" if gated else "")]
            for _, k in pc_methods:
                v = S.cells[f"{ds}/{c}"]["methods"][k][S.game_of[k]]["ec3"]
                s = f"{v:.2f}"
                row.append(f"\\textcolor{{gray}}{{{s}}}" if gated else s)
            L.append(" & ".join(row) + r" \\")
        if ds != DS[-1]:
            L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "tab_app_perclf_temporal.tex").write_text("\n".join(L) + "\n")
    print("wrote tab_app_perclf_temporal.tex")


def clf_gate_table(out, S):
    """Emit the classifier-gate table from ``classifier_gate.json``."""
    G = S.clf_gate
    L = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\textbf{Classifier gate.} A classifier's explanations enter the "
        "aggregated tables only if (G1) held-out accuracy $\\geq$ chance $+\\,0.05$ "
        "and (G2) train--test gap $\\leq 0.30$. Synthetic: accuracies on the "
        "training / validation / evaluation splits. Real: held-out accuracy pooled "
        "over the three data folds (per-fold values in parentheses); the training "
        "column reports accuracy on the pooled training material of the "
        "cross-validation folds. One rule yields every exclusion: POTR fails G1; "
        "the \\texttt{gait} and \\texttt{skel+gait} CNNs fail G2.}",
        "\\label{tab:app-clf-gate}",
        "\\small",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Dataset & Classifier & Train & Val.\\ & Test & Gap & Gate \\\\",
        "\\midrule",
        "\\multicolumn{7}{l}{\\emph{Synthetic (chance $= 1/3$; gate bar $0.383$)}} \\\\",
    ]
    order = {d: i for i, d in enumerate(DS)}
    for r in sorted(
        G["synthetic"], key=lambda r: (order[r["dataset"]], CLF.index(r["classifier"]))
    ):
        verdict = "\\checkmark" if r["gate_pass"] else "\\ding{55}\\ (G2)"
        L.append(
            f"\\texttt{{{DS_TEX[r['dataset']]}}} & {r['classifier']} & "
            f"{r['train_acc']:.3f} & {r['val_acc']:.3f} & {r['test_acc']:.3f} & "
            f"{r['train_test_gap']:+.3f} & {verdict} \\\\"
        )
    L.append("\\addlinespace")
    L.append(
        "\\multicolumn{7}{l}{\\emph{Real (chance: CARE-PD $1/3$, PTB-XL $1/2$, "
        "ESC-50 $1/50$)}} \\\\"
    )
    cg = G["real"]["carepd"]
    for clfk, label in (("motionbert", "MotionBERT"), ("motionagformer", "MotionAGFormer")):
        r = cg[clfk]
        ev = [r["per_fold_eval_acc"][str(f)] for f in (1, 2, 3)]
        pf = "/".join(f"{v:.2f}" for v in ev)
        L.append(
            f"CARE-PD & {label} & {r['trainpool_acc_mean']:.3f} & --- & "
            f"{r['pooled_acc']:.3f} {{\\scriptsize({pf})}} & "
            f"{r['train_test_gap']:+.3f} & \\checkmark \\\\"
        )
    L.append(
        f"CARE-PD & POTR & --- & --- & {cg['potr']['pooled_acc']:.3f} & --- & "
        f"\\ding{{55}}\\ (G1) \\\\"
    )
    r = G["real"]["ptbxl"]["resnet1d"]
    ev = [r["per_fold_eval_acc"][str(f)] for f in (1, 2, 3)]
    pf = "/".join(f"{v:.2f}" for v in ev)
    L.append(
        f"PTB-XL & ResNet1d & {r['trainpool_acc_mean']:.3f} & --- & "
        f"{r['eval_acc_mean']:.3f} {{\\scriptsize({pf})}} & "
        f"{r['train_test_gap']:+.3f} & \\checkmark \\\\"
    )
    r = G["real"]["esc50"]["ast"]
    te = [r["per_fold_test_acc"][str(f)] for f in (1, 2, 3)]
    pf = "/".join(f"{v:.2f}" for v in te)
    L.append(
        f"ESC-50 & AST (3 fold models) & {r['train_acc_mean']:.3f} & --- & "
        f"{r['test_acc_mean']:.3f} {{\\scriptsize({pf})}} & "
        f"{r['train_test_gap']:+.3f} & \\checkmark \\\\"
    )
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "tab_app_clf_gate.tex").write_text("\n".join(L) + "\n")
    print("wrote tab_app_clf_gate.tex")


def real_cells_table(out, S):
    """Emit the real joint x window / lead x window / band x window cells table."""
    cols = [
        ("CARE-PD A", "$\\np{=}68$", S.players["carepd_cell_motionbert"]),
        ("CARE-PD B", "$\\np{=}68$", S.players["carepd_cell_motionagformer"]),
        ("PTB-XL", "$\\np{=}48$", S.ptbxl_cells["ptbxl_cell"]),
        ("ESC-50", "$\\np{=}16$", S.players["esc50_cell_ast"]),
    ]

    def sd(src, m, met):
        f = [v[met] for v in src[m].get("per_fold", {}).values()]
        return float(np.std(f)) if len(f) > 1 else 0.0

    L = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\textbf{Real-world results, spatiotemporal players} "
        "(joint$\\times$window / lead$\\times$window / band$\\times$window cells). "
        "Faithfulness ($\\uparrow$) and PlayerAOPC ($\\uparrow$), pooled over the "
        "three data folds, $\\pm$ spread across folds; bold: best per column and "
        "metric. Protocol identical to Tables~\\ref{tab:real-faith}--"
        "\\ref{tab:real-aopc} (coalition budget 2048, $N{=}200$ per fold).}",
        "\\label{tab:app-real-cells}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{l" + "c" * 8 + "}",
        "\\toprule",
        " & \\multicolumn{4}{c}{\\emph{Faithfulness}} & "
        "\\multicolumn{4}{c}{\\emph{PlayerAOPC}} \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
        "Method & "
        + " & ".join(f"\\shortstack{{{n}\\\\{{\\scriptsize {m}}}}}" for n, m, _ in cols * 2)
        + r" \\",
        "\\midrule",
    ]
    win = {}
    for met in ("faith", "aopc"):
        for n, _, src in cols:
            win[(met, n)] = max((m for _, m in METH_REAL), key=lambda m: src[m][met])
    for lab, m in METH_REAL:
        if m == "kernelshap_vaeac":
            L.append("\\addlinespace")
        row = [lab]
        for met in ("faith", "aopc"):
            for n, _, src in cols:
                row.append(cell(src[m][met], sd(src, m, met), best=(win[(met, n)] == m)))
        L.append(" & ".join(row) + r" \\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "tab_app_real_cells.tex").write_text("\n".join(L) + "\n")
    print("wrote tab_app_real_cells.tex")


def imputer_gate_table(out, S):
    """Emit the imputer capability-gate table from ``imputer_gate.json``."""
    tracks = [("CARE-PD", "carepd"), ("PTB-XL", "ptbxl"), ("ESC-50", "esc50")]
    fams = ("spatial", "temporal", "cell")
    L = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\textbf{Imputer capability gate}: hide-one-recover correlation "
        "between the imputed conditional mean and the hidden ground truth, per "
        "player-set mask family (mean $\\pm$ spread over masks; 3 folds, 24 "
        "sequences per fold, 8 draws). \\emph{Donor}: copy the hidden entries from "
        "the most-correlated other unit of the same sequence -- the recovery "
        "available from raw redundancy alone. \\emph{Unconditional}: imputer run "
        "with everything hidden. A learned imputer conditions on the visible "
        "context exactly when its recovery clears both controls.}",
        "\\label{tab:app-imputer-gate}",
        "\\small",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Dataset & Imputer & Spatial & Temporal & Cell \\\\",
        "\\midrule",
    ]
    for name, tk in tracks:
        rec = S.imputer_gate[tk]
        rows = []
        for kind, lab in (("vaeac", "VAEAC"), ("flow", "Flow")):
            g = rec[kind]
            rows.append((lab, [f"{g[f]['mean']:.2f} \\msd{{{g[f]['std']:.2f}}}" for f in fams]))
        rows.append(("Donor control", [f"{rec['donor'][f]:.2f}" for f in fams]))
        uc = f"{rec['unconditional']:.2f}"
        rows.append(("Unconditional", [f"\\multicolumn{{3}}{{c}}{{{uc} (all families)}}"]))
        for ri, (lab, vals) in enumerate(rows):
            first = name if ri == 0 else ""
            L.append(f"{first} & {lab} & " + " & ".join(vals) + r" \\")
        if tk != "esc50":
            L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "tab_app_imputer_gate.tex").write_text("\n".join(L) + "\n")
    print("wrote tab_app_imputer_gate.tex")


def floors_table(out, S):
    """Emit the grading-noise floors table from ``meta.grading_floors``."""
    rows = []
    cross = []
    for key, r in S.floors.items():
        ps, ds, tail = key.split("/")
        if tail.startswith("B="):
            rows.append((ps, ds, int(tail[2:]), r))
        elif tail == "cross_budget_8192_vs_32768":
            cross.append(r)
    L = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\textbf{Grading-noise floors} for the sampled hi-budget "
        "targets: EC between the stored grading target and an independent "
        "replicate (same budget, disjoint coalition seed), averaged over the "
        "gate-passing classifier architectures. Method-vs-target differences "
        "below these levels are not interpretable. Exact-enumeration cells "
        "($\\np \\leq 12$) have zero grading noise by construction. On the "
        "periodic datasets the conditional-target floor is itself elevated -- "
        "the near-null phenomenon of \\S\\ref{sec:salient} measured directly: "
        "when the conditional game carries little rankable signal, even two "
        "independent estimates of its ground truth rank cells inconsistently.}",
        "\\label{tab:app-floors}",
        "\\small",
        "\\begin{tabular}{llrrcc}",
        "\\toprule",
        "Player set & Dataset & $\\np$ & Budget & Floor EC3 (cond / marg) & "
        "Floor EC1 (cond / marg) \\\\",
        "\\midrule",
    ]
    PS_LAB = {"joint": "spatial", "cell": "cells"}
    for ps, ds, b, r in sorted(rows, key=lambda t: (t[0], DS.index(t[1]), t[2])):
        L.append(
            f"{PS_LAB[ps]} & \\texttt{{{DS_TEX[ds]}}} & {r['M']} & {b:,} & "
            f"{r['cond_ec3']:.3f} / {r['marg_ec3']:.3f} & "
            f"{r['cond_ec1']:.4f} / {r['marg_ec1']:.4f} \\\\"
        )
    if cross:
        v = sum(r["cond_ec3"] * r["n_clf"] for r in cross) / sum(r["n_clf"] for r in cross)
        L.append("\\addlinespace")
        L.append(
            f"\\multicolumn{{6}}{{l}}{{Cross-budget agreement at $\\np{{=}}68$ "
            f"(cond target, $B{{=}}8{{,}}192$ vs $B{{=}}32{{,}}768$): EC3 {v:.3f}}} \\\\"
        )
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "tab_app_floors.tex").write_text("\n".join(L) + "\n")
    print("wrote tab_app_floors.tex")


def main() -> None:
    """Regenerate every canonical-sourced paper table into ``--out``."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=str, default=str(REPO_ROOT / "results" / "paper_tables"))
    ap.add_argument("--canonical", type=str, default=str(CANONICAL))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    S = Sources(Path(args.canonical))

    # ---------------- main-text synthetic tables (EC3) ----------------
    synth_style_table(
        out,
        "tab_temporal.tex",
        "\\textbf{Synthetic results, temporal players} ($K$ uniform windows). EC3 "
        "(ranking error, $1-r$; lower is better), mean $\\pm$ spread across the classifier "
        "architectures passing the accuracy/generalization gate (Appendix~\\ref{app:clf_gate}). "
        "Each method is graded against the exact ground truth of "
        "its own fill family (game-matched). Gray: near-null cells on periodic data, where "
        "the oracle's own conditional game carries almost no rankable signal "
        "(Sec.~\\ref{sec:salient}); these are excluded from per-column winners (bold). "
        "EC1 tables in Appendix.",
        "tab:synth-temporal",
        MAIN_T_STD,
        T_CND,
        lambda k, d: S.gm(k, d, S.game_of[k], "ec3"),
        lambda d: S.gm("ks_oracle", d, "cond", "ec3"),
        nearnull=NEARNULL,
    )
    synth_style_table(
        out,
        "tab_spatial.tex",
        "\\textbf{Synthetic results, spatial players} (one player per joint; $J{=}5$ or "
        "$17$ by dataset). EC3, "
        "mean $\\pm$ spread across the gate-passing classifier architectures, game-matched "
        "grading. The near-null cells of Table~\\ref{tab:synth-temporal} are gone: spatial "
        "players "
        "cut across the period and restore rankable signal on \\texttt{gait} and "
        "\\texttt{skel+gait}.",
        "tab:synth-spatial",
        MAIN_P_STD,
        P_CND,
        lambda k, d: S.e10("joint", k, d, "ec3"),
        lambda d: S.e10("joint", "ks_oracle", d, "ec3"),
    )
    synth_style_table(
        out,
        "tab_cells.tex",
        "\\textbf{Synthetic results, spatiotemporal players} (joint$\\times$window cells; "
        "$M{=}20$--$68$ by dataset). EC3, mean $\\pm$ spread across the gate-passing "
        "classifier architectures, "
        "game-matched grading. Gray: near-null cells (periodic data). Grading-noise floors "
        "at this granularity are quantified in Appendix.",
        "tab:synth-cells",
        MAIN_P_STD,
        P_CND,
        lambda k, d: S.e10("cell", k, d, "ec3"),
        lambda d: S.e10("cell", "ks_oracle", d, "ec3"),
        nearnull=NEARNULL,
    )

    # ---------------- main-text real tables ----------------
    real_cols = [
        (S, S.tracks, "carepd_motionbert", "CARE-PD", "model A, windows"),
        (S, S.tracks, "carepd_motionagformer", "CARE-PD", "model B, windows"),
        (S, S.tracks, "ptbxl", "PTB-XL", "windows"),
        (S, S.tracks, "esc50", "ESC-50", "windows"),
        (S, S.tracks, "ptbxl_leads", "PTB-XL", "per lead"),
        (S, S.players, "carepd_joint_motionbert", "CARE-PD", "model A, joints"),
        (S, S.players, "carepd_joint_motionagformer", "CARE-PD", "model B, joints"),
        (S, S.tracks, "esc50_freq", "ESC-50", "freq.\\ bands"),
    ]
    real_table(
        out,
        real_cols,
        "tab_faith.tex",
        "faith",
        "\\textbf{Real-world results, faithfulness correlation} ($\\uparrow$): Pearson "
        "correlation between the summed credits of hidden players and the model's prediction "
        "drop, over sampled coalitions. Mean $\\pm$ spread across data folds; bold: best per "
        "column. Temporal players are $K{=}4$ uniform windows; spatial players are the 12 "
        "ECG leads, the 17 skeleton joints, or the audio frequency bands. CARE-PD models A "
        "and B are two classifier architectures. Joint$\\times$window results in Appendix.",
        "tab:real-faith",
    )
    real_table(
        out,
        real_cols,
        "tab_aopc.tex",
        "aopc",
        "\\textbf{Real-world results, PlayerAOPC} ($\\uparrow$): mean prediction drop as "
        "players are removed in decreasing-credit order. Mean $\\pm$ spread across data "
        "folds; bold: best per column. Same tracks as Table~\\ref{tab:real-faith}. AOPC "
        "magnitudes scale with player granularity, so values are comparable within a column "
        "block, not across blocks.",
        "tab:real-aopc",
    )

    # ---------------- appendix: EC1 companions ----------------
    synth_style_table(
        out,
        "tab_app_ec1_temporal.tex",
        "\\textbf{EC1, temporal players} (companion to Table~\\ref{tab:synth-temporal}): "
        "mean absolute attribution error against the exact game-matched ground truth, "
        f"{GATE_NOTE}. EC1 carries the attribution scale of each dataset, so values "
        "compare within a column, not across columns. Gray: near-null cells as in the "
        "main table.",
        "tab:app-ec1-temporal",
        T_STD,
        T_CND,
        lambda k, d: S.gm(k, d, S.game_of[k], "ec1"),
        lambda d: S.gm("ks_oracle", d, "cond", "ec1"),
        nearnull=NEARNULL,
        prec=3,
    )
    synth_style_table(
        out,
        "tab_app_ec1_spatial.tex",
        "\\textbf{EC1, spatial players} (companion to Table~\\ref{tab:synth-spatial}), "
        f"{GATE_NOTE}, game-matched grading.",
        "tab:app-ec1-spatial",
        P_STD,
        P_CND,
        lambda k, d: S.e10("joint", k, d, "ec1"),
        lambda d: S.e10("joint", "ks_oracle", d, "ec1"),
        prec=3,
    )
    synth_style_table(
        out,
        "tab_app_ec1_cells.tex",
        "\\textbf{EC1, spatiotemporal players} (companion to Table~\\ref{tab:synth-cells}), "
        f"{GATE_NOTE}, game-matched grading. Gray: near-null cells.",
        "tab:app-ec1-cells",
        P_STD,
        P_CND,
        lambda k, d: S.e10("cell", k, d, "ec1"),
        lambda d: S.e10("cell", "ks_oracle", d, "ec1"),
        nearnull=NEARNULL,
        prec=3,
    )

    # ---------------- appendix: cross-graded (all-conditional) EC3 ----------------
    synth_style_table(
        out,
        "tab_app_cross_temporal.tex",
        "\\textbf{Cross-graded EC3, temporal players}: every method graded against the "
        "on-manifold (conditional) ground truth, the convention of prior oracle "
        "benchmarks. Comparing the standard-fill block with its game-matched values in "
        "Table~\\ref{tab:synth-temporal} isolates the ground-truth divergence "
        "$\\Delta(\\gtstd,\\gtman)$ from estimator error. Gray columns: on periodic data "
        "the conditional target itself is near null, so no winner is marked.",
        "tab:app-cross-temporal",
        T_STD,
        T_CND,
        lambda k, d: S.gm(k, d, "cond", "ec3"),
        lambda d: S.gm("ks_oracle", d, "cond", "ec3"),
        nn_cols=NEARNULL_COLS,
        std_hdr=CROSS_STD_HDR,
        cnd_hdr=CROSS_CND_HDR,
    )
    synth_style_table(
        out,
        "tab_app_cross_spatial.tex",
        "\\textbf{Cross-graded EC3, spatial players}: all methods against the "
        "on-manifold ground truth (exact conditional target for $\\np \\leq 12$, "
        "hi-budget sampled target above).",
        "tab:app-cross-spatial",
        P_STD,
        P_CND,
        lambda k, d: S.e10("joint", k, d, "ec3", cross=True),
        lambda d: S.e10("joint", "ks_oracle", d, "ec3", cross=True),
        std_hdr=CROSS_STD_HDR,
        cnd_hdr=CROSS_CND_HDR,
    )
    synth_style_table(
        out,
        "tab_app_cross_cells.tex",
        "\\textbf{Cross-graded EC3, spatiotemporal players}: all methods against the "
        "on-manifold ground truth. At $\\np{=}68$ (\\texttt{skel+gait} cells) the "
        "standard-fill methods' cross-graded error is an order of magnitude above their "
        "game-matched error (Table~\\ref{tab:synth-cells}): at fine granularity the two "
        "ground truths rank cells very differently.",
        "tab:app-cross-cells",
        P_STD,
        P_CND,
        lambda k, d: S.e10("cell", k, d, "ec3", cross=True),
        lambda d: S.e10("cell", "ks_oracle", d, "ec3", cross=True),
        std_hdr=CROSS_STD_HDR,
        cnd_hdr=CROSS_CND_HDR,
    )

    # ---------------- remaining appendix tables ----------------
    perclf_table(out, S)
    clf_gate_table(out, S)
    real_cells_table(out, S)
    imputer_gate_table(out, S)
    floors_table(out, S)
    print("done")


if __name__ == "__main__":
    main()
