"""motionbench.pipelines — Evaluation pipeline orchestration."""

from motionbench.pipelines.leaderboard import build_leaderboard, load_results
from motionbench.pipelines.real_eval import run_real_eval
from motionbench.pipelines.synthetic_eval import run_synthetic_eval

__all__ = [
    "run_synthetic_eval",
    "run_real_eval",
    "build_leaderboard",
    "load_results",
]
