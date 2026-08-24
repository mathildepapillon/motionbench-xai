"""Tests for scripts/make_paper_tables.py — canonical-file table regeneration.

Runs the generator into a tmpdir against the shipped canonical result files
and asserts that a handful of stable, paper-visible values appear in the
emitted booktabs LaTeX.  No golden full-file diffs: the assertions pin the
numbers the paper quotes (2-3 decimal places), not the surrounding markup,
so caption edits never break this test while any change to the canonical
values or the aggregation conventions (gate-aware averaging, game-matched
extraction, winner bolding) does.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "make_paper_tables.py"

EXPECTED_FILES = [
    "tab_temporal.tex",
    "tab_spatial.tex",
    "tab_cells.tex",
    "tab_faith.tex",
    "tab_aopc.tex",
    "tab_app_ec1_temporal.tex",
    "tab_app_ec1_spatial.tex",
    "tab_app_ec1_cells.tex",
    "tab_app_cross_temporal.tex",
    "tab_app_cross_spatial.tex",
    "tab_app_cross_cells.tex",
    "tab_app_perclf_temporal.tex",
    "tab_app_clf_gate.tex",
    "tab_app_real_cells.tex",
    "tab_app_imputer_gate.tex",
    "tab_app_floors.tex",
]


@pytest.fixture(scope="module")
def tables_dir(tmp_path_factory) -> Path:
    """Run the generator once into a tmpdir and return the output directory."""
    out = tmp_path_factory.mktemp("paper_tables")
    subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out)],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return out


def test_all_tables_emitted(tables_dir):
    """Every paper table with a canonical source is emitted, non-empty, booktabs."""
    for name in EXPECTED_FILES:
        tex = (tables_dir / name).read_text()
        assert "\\toprule" in tex and "\\bottomrule" in tex, name


def test_real_faith_stable_values(tables_dir):
    """Table 5: ESC-50 windows KS-Marginal 0.92 (bold winner), CARE-PD A 0.96."""
    tex = (tables_dir / "tab_faith.tex").read_text()
    marginal_row = next(line for line in tex.splitlines() if line.startswith("KS-Marginal"))
    cols = [c.strip() for c in marginal_row.rstrip("\\").split("&")]
    # Columns: method, CARE-PD A/B windows, PTB-XL windows, ESC-50 windows, ...
    assert cols[1].startswith("\\textbf{0.96}")  # CARE-PD model A, windows
    assert cols[4].startswith("\\textbf{0.92}")  # ESC-50, windows (bold winner)


def test_synthetic_temporal_gate_aware_oracle(tables_dir):
    """Table 2 keeps the KS-Oracle reference row and the near-null gray cells."""
    tex = (tables_dir / "tab_temporal.tex").read_text()
    assert "KS-Oracle (reference)" in tex
    assert "\\textcolor{gray}" in tex  # near-null cells on periodic data


def test_floors_rows(tables_dir):
    """Floors table carries the measured M=68 rows and the cross-budget line."""
    tex = (tables_dir / "tab_app_floors.tex").read_text()
    assert "cells & \\texttt{skel\\_gait}" not in tex  # dataset names not mangled
    assert "cells & \\texttt{skel+gait} & 68 & 32,768 & 0.032 / 0.027" in tex
    assert "cells & \\texttt{skeleton} & 68 & 8,192 & 0.118 / 0.020" in tex
    assert "Cross-budget agreement" in tex
    assert "EC3 0.065" in tex


def test_classifier_gate_values(tables_dir):
    """Classifier gate: ESC-50 AST row and the two G2 exclusions."""
    tex = (tables_dir / "tab_app_clf_gate.tex").read_text()
    assert "0.965 {\\scriptsize(0.95/0.98/0.96)}" in tex  # ESC-50 fold test accs
    assert tex.count("\\ding{55}\\ (G2)") == 2  # gait/cnn and skel+gait/cnn
    assert tex.count("\\ding{55}\\ (G1)") == 1  # POTR


def test_imputer_gate_values(tables_dir):
    """Imputer gate: CARE-PD VAEAC row and the PTB-XL donor control."""
    tex = (tables_dir / "tab_app_imputer_gate.tex").read_text()
    assert "CARE-PD & VAEAC & 1.00 \\msd{0.00} & 0.96 \\msd{0.12} & 1.00 \\msd{0.00}" in tex
    assert " & Donor control & -0.01 & -0.01 & -0.01" in tex  # PTB-XL redundancy floor
