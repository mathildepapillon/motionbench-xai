"""motionbench.pipelines.real_eval — CARE-PD BMCLab evaluation pipeline.

Mirrors :mod:`~motionbench.pipelines.synthetic_eval` but operates on the
:class:`~motionbench.data.real.care_pd.BMCLabDataset` real-world dataset.

Key differences from the synthetic pipeline
-------------------------------------------
* No ground-truth oracle — GT metrics (EC1–EC3, TopK, …) are skipped.
* Classifiers are ported CARE-PD encoders loaded from checkpoints.
* Device default: ``"cuda:0"`` for ported classifiers.

Usage
-----
Invoked automatically by :func:`motionbench.cli.run.main` when
``cfg.pipeline == "real"``.

References
----------
Dindorf et al. (2022) CARE-PD dataset and encoder benchmarks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig

import pandas as pd

from motionbench.pipelines.synthetic_eval import (
    _init_wandb,
    _log_to_wandb,
    _run_cell,
)

log = logging.getLogger(__name__)

__all__ = ["run_real_eval"]


def run_real_eval(cfg: DictConfig) -> pd.DataFrame:
    """Run the CARE-PD evaluation sweep.

    Operates identically to :func:`~motionbench.pipelines.synthetic_eval.run_synthetic_eval`
    but uses the real BMCLab dataset and ported CARE-PD classifiers.
    Ground-truth metrics are automatically skipped because the real dataset
    has no oracle.

    Args:
        cfg: Hydra DictConfig for the ``care_pd_sweep`` experiment.

    Returns:
        :class:`pandas.DataFrame` with one row per completed (or cached) cell.
    """
    _init_wandb(cfg)

    datasets: list[str] = list(cfg.datasets)
    methods: list[str] = list(cfg.methods)
    classifiers: list[str] = list(cfg.classifiers)

    cells = [
        (ds, clf, mth)
        for ds in datasets
        for clf in classifiers
        for mth in methods
    ]
    log.info(
        "CARE-PD sweep: %d cells (%d datasets × %d classifiers × %d methods)",
        len(cells),
        len(datasets),
        len(classifiers),
        len(methods),
    )

    results = []
    for ds, clf, mth in cells:
        result = _run_cell(ds, clf, mth, cfg)
        _log_to_wandb(result)
        results.append(result)

    try:
        import wandb  # noqa: PLC0415

        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass

    return pd.DataFrame(results) if results else pd.DataFrame()
