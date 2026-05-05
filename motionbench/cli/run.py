"""motionbench.cli.run — Hydra CLI entry point for MotionBench-XAI evaluations.

Usage
-----
Run the full synthetic benchmark::

    motionbench experiments=full_synthetic_sweep

Tiny smoke-test (10 sequences, no WandB, 1 worker)::

    motionbench run experiments=full_synthetic_sweep \\
        +n_sequences=10 +n_jobs=1 wandb.mode=disabled

CARE-PD sweep (requires BMCLab data paths)::

    motionbench pipeline=real \\
        datasets=[care_pd_bmclab] classifiers=[motionbert]

The ``run`` subcommand word is optional: ``motionbench run ...`` and
``motionbench ...`` are both accepted.

Hydra overrides
---------------
All fields in the loaded experiment config can be overridden on the CLI::

    motionbench n_sequences=10 n_jobs=4 results_dir=my_results wandb.mode=disabled

Config path
-----------
By default the CLI loads ``configs/experiments/full_synthetic_sweep.yaml``
relative to the directory from which ``motionbench`` is invoked.  Override
``config_name`` using the Hydra ``--config-name`` flag.

References
----------
Hydra documentation: https://hydra.cc/docs/
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import hydra

if TYPE_CHECKING:
    from omegaconf import DictConfig

log = logging.getLogger(__name__)

__all__ = ["main"]

# Resolve the config directory relative to this file at import time so that
# Hydra receives an absolute filesystem path regardless of CWD or how the
# package is installed.
_CONFIG_PATH: str = str(Path(__file__).parent.parent.parent / "configs")


@hydra.main(
    config_path=_CONFIG_PATH,
    config_name="config",
    version_base="1.3",
)
def _hydra_main(cfg: DictConfig) -> None:
    """Hydra-decorated main — reads sys.argv for Hydra overrides.

    Routes to the correct pipeline based on ``cfg.pipeline``:

    * ``"real"`` → :func:`~motionbench.pipelines.real_eval.run_real_eval`
    * anything else → :func:`~motionbench.pipelines.synthetic_eval.run_synthetic_eval`

    Args:
        cfg: Merged Hydra DictConfig.
    """
    pipeline: str = str(cfg.get("pipeline", "synthetic"))

    if pipeline == "real":
        from motionbench.pipelines.real_eval import run_real_eval  # noqa: PLC0415

        df = run_real_eval(cfg)
    else:
        from motionbench.pipelines.synthetic_eval import run_synthetic_eval  # noqa: PLC0415

        df = run_synthetic_eval(cfg)

    if not df.empty:
        log.info("Sweep complete.  %d cells finished.", len(df))


def main() -> None:
    """CLI entry point: ``motionbench [run] [overrides...]``.

    Accepts an optional ``run`` subcommand word for ergonomics::

        motionbench run experiments=full_synthetic_sweep ...
        motionbench experiments=full_synthetic_sweep ...  # also valid

    Both forms delegate to :func:`_hydra_main` which processes all remaining
    arguments as Hydra config overrides.
    """
    # Strip the optional "run" subcommand so Hydra sees only overrides
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        del sys.argv[1]
    _hydra_main()
