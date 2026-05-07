"""scripts/run_care_pd_sweep.py — CARE-PD sweep with per-classifier caching.

Loads each (dataset, classifier) pair once, then runs all methods against the
pre-loaded classifier.  This avoids the ~65s MotionBERT re-load cost for each
of the 10 methods.

Usage:
    conda activate motionbench-xai
    CUDA_VISIBLE_DEVICES=2 python scripts/run_care_pd_sweep.py
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)

CAREPD_ROOT = Path(__file__).parents[1]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-override", type=str, default=None,
                        help="If set, run only this single method (for per-GPU parallelism).")
    args, _ = parser.parse_known_args()

    from motionbench.pipelines.synthetic_eval import (
        _build_and_fit_imputer,
        _build_attributor,
        _build_classifier,
        _build_players,
        _collect_metric_names,
        _evaluate_metrics,
        _instantiate_dataset,
        _load_sub_config,
    )
    from motionbench.imputers.off_manifold import ZeroImputer

    with hydra.initialize_config_dir(
        config_dir=str((CAREPD_ROOT / "configs").resolve()),
        version_base="1.3",
    ):
        cfg = hydra.compose(
            config_name="config",
            overrides=["experiments=care_pd_sweep"],
        )

    cfg.wandb.mode = "disabled"
    device: str = str(cfg.get("device", "cuda:0"))
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    datasets: list[str] = list(cfg.datasets)
    classifiers: list[str] = list(cfg.classifiers)
    methods: list[str] = [args.method_override] if args.method_override else list(cfg.methods)
    n_seq: int = int(cfg.get("n_sequences", 100))

    all_results = []

    for ds_name in datasets:
        log.info("Dataset: %s", ds_name)
        dataset_cfg = _load_sub_config("data", ds_name, cfg)
        dataset, K = _instantiate_dataset(dataset_cfg)
        J, F, T = dataset.shape
        n_classes = int(str(dataset.metadata.get("n_classes", 3)))
        n_seq_actual = min(n_seq, len(dataset))
        log.info("  shape (J=%d, F=%d, T=%d), K=%d, n_classes=%d, n_seq=%d",
                 J, F, T, K, n_classes, n_seq_actual)

        for clf_name in classifiers:
            log.info("  Classifier: %s — loading ...", clf_name)
            t_load = time.time()
            clf_cfg = _load_sub_config("classifiers", clf_name, cfg)
            clf = _build_classifier(clf_cfg, J, F, T, K, n_classes)
            clf = clf.to(torch.device(device))
            clf.eval()
            log.info("  Classifier loaded in %.1fs", time.time() - t_load)

            for method_name in methods:
                result_path = results_dir / ds_name / clf_name / method_name / "result.json"
                if result_path.exists():
                    log.info("    [SKIP cached] %s/%s/%s", ds_name, clf_name, method_name)
                    all_results.append(json.loads(result_path.read_text()))
                    continue

                log.info("    Method: %s", method_name)
                t_method = time.time()
                try:
                    method_cfg = _load_sub_config("methods", method_name, cfg)
                    players = _build_players(method_cfg, J, F, T, K)
                    has_imputer = OmegaConf.select(method_cfg, "imputer") is not None
                    imputer = None
                    if has_imputer:
                        imputer = _build_and_fit_imputer(method_cfg, dataset, J, F, T, device)

                    attributor = _build_attributor(method_cfg, clf, imputer, players)

                    clf_device = torch.device(device)
                    phi_list, x_list, target_list = [], [], []
                    cache_path = result_path.parent / "attributions.npz"

                    cache_loaded = False
                    if cache_path.exists():
                        try:
                            _z = np.load(cache_path, allow_pickle=False)
                            if _z["phi"].shape[0] == n_seq_actual:
                                phi_list = [torch.from_numpy(_z["phi"][i]) for i in range(n_seq_actual)]
                                x_list = [torch.from_numpy(_z["x"][i]) for i in range(n_seq_actual)]
                                target_list = [int(t) for t in _z["target"].tolist()]
                                cache_loaded = True
                        except Exception as exc:
                            log.warning("Failed to load attribution cache: %s", exc)

                    # Detect flow imputer → can use batched coalition pre-computation
                    from motionbench.imputers.carepd_imputer import CarepdFlowImputer
                    is_flow = isinstance(imputer, CarepdFlowImputer)
                    flow_n_samples = int(method_cfg.get("n_completion_samples", 1))

                    if not cache_loaded:
                        for idx in range(n_seq_actual):
                            x, _y = dataset[idx]
                            x = x.to(clf_device)

                            # Pre-compute all 2^K temporal coalitions in one ODE run (16x speedup)
                            if is_flow and imputer is not None:
                                imputer.precompute_all_temporal_coalitions(
                                    x.cpu(), K=K, n_samples=flow_n_samples,
                                )

                            with torch.no_grad():
                                logits = clf(x.unsqueeze(0))
                            target_i = int(logits.argmax(dim=-1).item())
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                phi = attributor.attribute(x, players, target=target_i)

                            if is_flow and imputer is not None:
                                imputer.clear_cache()

                            phi_list.append(phi.detach().cpu())
                            x_list.append(x.cpu())
                            target_list.append(target_i)
                        try:
                            cache_path.parent.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(
                                cache_path,
                                phi=np.stack([p.numpy() for p in phi_list]),
                                x=np.stack([t.numpy() for t in x_list]),
                                target=np.asarray(target_list, dtype=np.int64),
                            )
                        except Exception as exc:
                            log.warning("Failed to write attribution cache: %s", exc)

                    metric_names = _collect_metric_names(cfg)
                    fidelity_imputer = imputer if imputer is not None else ZeroImputer().fit(dataset)
                    scores = _evaluate_metrics(
                        phi_list=phi_list,
                        x_list=x_list,
                        target_list=target_list,
                        classifier=clf,
                        players=players,
                        dataset=dataset,
                        imputer=fidelity_imputer,
                        metric_names=metric_names,
                        cfg=cfg,
                        device=device,
                        attributor=attributor,
                    )

                    result = {
                        "dataset": ds_name,
                        "classifier": clf_name,
                        "method": method_name,
                        "n_sequences": n_seq_actual,
                        **scores,
                    }
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    result_path.write_text(json.dumps(result, indent=2))
                    log.info(
                        "    Done %s/%s/%s (%.1fs) — %d metrics",
                        ds_name, clf_name, method_name, time.time() - t_method, len(scores),
                    )

                except Exception as exc:
                    log.warning("    FAILED %s/%s/%s: %s", ds_name, clf_name, method_name, exc)
                    error_path = result_path.parent / "error.json"
                    error_path.parent.mkdir(parents=True, exist_ok=True)
                    error_path.write_text(json.dumps({"error": str(exc), "type": type(exc).__name__}))
                    result = {
                        "dataset": ds_name,
                        "classifier": clf_name,
                        "method": method_name,
                        "error": str(exc),
                    }

                all_results.append(result)

    # Summary
    ok = sum(1 for r in all_results if "error" not in r)
    err = sum(1 for r in all_results if "error" in r)
    log.info("CARE-PD sweep complete: %d OK, %d errors", ok, err)

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    log.info("Results at %s", results_dir)


if __name__ == "__main__":
    main()
