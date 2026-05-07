"""scripts/_run_one_cell.py — single-cell entry point for multi-GPU dispatch.

Internal helper used by ``run_xor_sweep_multigpu.py``.  Bypasses the Hydra
defaults-list semantics that prevent CLI overrides on keys whose names
collide with config-group directories (``classifiers/``, ``methods/``,
``data/``).

Constructs a minimal :class:`omegaconf.DictConfig` programmatically, then
calls :func:`motionbench.pipelines.synthetic_eval._run_cell` directly.

Usage::

    CUDA_VISIBLE_DEVICES=3 python scripts/_run_one_cell.py \\
        --dataset xor_label_gaussian \\
        --classifier synthetic_mlp \\
        --method kernelshap_zero \\
        --device cuda:0 \\
        --n-sequences 200 \\
        --results-dir results/synthetic
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from omegaconf import OmegaConf  # noqa: E402

from motionbench.pipelines.synthetic_eval import _run_cell  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--classifier", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-sequences", type=int, default=200)
    p.add_argument("--results-dir", type=Path, default=REPO / "results" / "synthetic")
    p.add_argument("--checkpoint-dir", type=Path,
                   default=REPO / "motionbench" / "classifiers" / "checkpoints" / "synthetic")
    p.add_argument("--metrics-mode", choices=["full", "gt_only", "gt_plus_faith"],
                   default="full",
                   help="full: all metrics (slow, paper-default).  "
                        "gt_only: skip fidelity/stability/sanity (fast restore).  "
                        "gt_plus_faith: GT + faithfulness_correlation only "
                        "(skip pixel_flipping which dominates Flow runtime).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.metrics_mode == "full":
        metrics = {
            "gt": ["ec1", "ec2", "ec3", "topk", "spearman", "kendall",
                   "efficiency_error"],
            "fidelity": ["faithfulness_correlation", "pixel_flipping"],
            "stability": ["max_sensitivity"],
            "sanity": ["model_parameter_randomisation"],
        }
    elif args.metrics_mode == "gt_plus_faith":
        metrics = {
            "gt": ["ec1", "ec2", "ec3", "topk", "spearman", "kendall",
                   "efficiency_error"],
            "fidelity": ["faithfulness_correlation"],
            "stability": [],
            "sanity": [],
        }
    else:  # gt_only
        metrics = {
            "gt": ["ec1", "ec2", "ec3", "topk", "spearman", "kendall",
                   "efficiency_error"],
            "fidelity": [],
            "stability": [],
            "sanity": [],
        }
    cfg_d = {
        "pipeline": "synthetic",
        "results_dir": str(args.results_dir),
        "checkpoint_dir": str(args.checkpoint_dir),
        "device": args.device,
        "n_sequences": int(args.n_sequences),
        "n_jobs": 1,
        "metrics": metrics,
        "wandb": {"mode": "disabled", "project": "motionbench-xai",
                  "entity": None, "tags": []},
    }
    cfg = OmegaConf.create(cfg_d)

    result = _run_cell(args.dataset, args.classifier, args.method, cfg)
    out_path = args.results_dir / args.dataset / args.classifier / args.method / "result.json"
    if out_path.exists():
        print(f"OK {args.dataset}/{args.classifier}/{args.method} -> {out_path}")
    else:
        print(f"DONE (no result.json) for {args.dataset}/{args.classifier}/{args.method}")
        print(json.dumps(result, indent=2)[:400])


if __name__ == "__main__":
    main()
