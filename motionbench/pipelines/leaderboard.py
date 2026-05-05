"""motionbench.pipelines.leaderboard — Leaderboard aggregation from sweep results.

Reads all ``result.json`` files produced by
:func:`~motionbench.pipelines.synthetic_eval.run_synthetic_eval` or
:func:`~motionbench.pipelines.real_eval.run_real_eval`, aggregates scores
across datasets and classifiers, and produces a ranked leaderboard table.

Usage
-----
::

    from motionbench.pipelines.leaderboard import build_leaderboard
    lb = build_leaderboard("results/synthetic")
    lb.to_csv("leaderboard.csv", index=False)

The leaderboard table has one row per method and columns for the mean score
of each metric across all dataset × classifier pairs.

References
----------
OpenXAI (NeurIPS D&B 2022) leaderboard format.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["build_leaderboard", "load_results"]


def load_results(results_dir: str | Path) -> pd.DataFrame:
    """Load all ``result.json`` files from a results directory tree.

    Recursively searches ``results_dir`` for files named ``result.json`` and
    loads each into a row of the returned :class:`~pandas.DataFrame`.  Error
    records (``error.json``) are silently skipped.

    Args:
        results_dir: Root directory produced by the evaluation pipeline.

    Returns:
        :class:`pandas.DataFrame` with one row per completed (dataset, classifier,
        method) cell.  Columns include ``dataset``, ``classifier``, ``method``,
        and one column per metric sub-score.  Missing values are ``NaN``.

    Raises:
        FileNotFoundError: If ``results_dir`` does not exist.
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    rows: list[dict[str, Any]] = []
    for result_file in results_dir.rglob("result.json"):
        try:
            data = json.loads(result_file.read_text())
            rows.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not load %s: %s", result_file, exc)

    if not rows:
        log.warning("No result.json files found in %s", results_dir)
        return pd.DataFrame()

    return pd.DataFrame(rows)


def build_leaderboard(
    results_dir: str | Path,
    rank_by: str = "ec1",
    ascending: bool = True,
) -> pd.DataFrame:
    """Build a method-level leaderboard from evaluation results.

    Aggregates all (dataset, classifier) combinations by computing the mean
    of each metric per method, then ranks methods by ``rank_by``.

    Args:
        results_dir: Root directory produced by the evaluation pipeline
            (e.g. ``results/synthetic``).
        rank_by: Metric column name to rank by.  Defaults to ``"ec1"``
            (lower is better).
        ascending: Whether lower values are better for ``rank_by``.
            Defaults to ``True``.

    Returns:
        :class:`pandas.DataFrame` with one row per method, sorted by the
        ``rank_by`` column.  Columns include ``method``, ``rank``, and one
        mean-score column per metric.

    Raises:
        FileNotFoundError: If ``results_dir`` does not exist.
        ValueError: If ``rank_by`` is not a column in the loaded results.
    """
    df = load_results(results_dir)
    if df.empty:
        return df

    # Keep only numeric score columns (drop metadata strings)
    meta_cols = {"dataset", "classifier", "method", "n_sequences", "error"}
    score_cols = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]

    if "method" not in df.columns:
        raise ValueError("Loaded results do not contain a 'method' column.")

    agg = df.groupby("method")[score_cols].mean().reset_index()

    if rank_by not in agg.columns:
        log.warning(
            "rank_by=%r not in leaderboard columns %s; using first numeric column.",
            rank_by,
            list(agg.columns),
        )
        numeric_cols = [c for c in agg.columns if c != "method"]
        rank_by = numeric_cols[0] if numeric_cols else "method"

    agg = agg.sort_values(rank_by, ascending=ascending).reset_index(drop=True)
    agg.insert(0, "rank", agg.index + 1)
    return agg
