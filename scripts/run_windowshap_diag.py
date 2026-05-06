"""scripts/run_windowshap_diag.py — WindowSHAP window-size sensitivity sweep.

For each window size w ∈ {2, 4, 8, 16}, runs WindowSHAP on ``gaussian_k4``
and ``skeleton_gait_combined`` with the ``synthetic_transformer`` classifier,
computes EC1 against oracle ground truth, and saves results.

Results are written to::

    results/synthetic/{dataset}/synthetic_transformer/windowshap_w{w}/result.json

Usage::

    conda activate motionbench-xai
    CUDA_VISIBLE_DEVICES=3 python scripts/run_windowshap_diag.py
"""
from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO = Path("/home/papillon/code/motionbench-xai")
DEVICE = "cuda:0"   # CUDA_VISIBLE_DEVICES=3 → physical GPU 3, logical 0
N_SEQ = 50
WINDOW_SIZES = [2, 4, 8, 16]
DATASETS = ["gaussian_k4", "skeleton_gait_combined"]
CLF_NAME = "synthetic_transformer"

CKPT_ROOT = REPO / "motionbench/classifiers/checkpoints/synthetic"
RESULTS_ROOT = REPO / "results/synthetic"

# Oracle settings — match the overnight sweep
ORACLE_N_MC = 50
ORACLE_N_COALITIONS = 64

# ---------------------------------------------------------------------------
# Helpers re-used from motionbench.pipelines.synthetic_eval
# ---------------------------------------------------------------------------

from motionbench.pipelines.synthetic_eval import (  # noqa: E402
    _instantiate_dataset,
    _build_classifier,
    _build_players,
)
from motionbench.attribution.windowshap import WindowSHAPAttributor  # noqa: E402
from motionbench.players.temporal_windows import TemporalWindows  # noqa: E402
from motionbench.metrics.ground_truth import EC1Metric  # noqa: E402


def _load_cfg(subdir: str, name: str):
    return OmegaConf.load(REPO / "configs" / subdir / f"{name}.yaml")


def _build_prob_clf(classifier, device):
    """Return a softmax-probability callable matching the oracle game."""
    clf_device = torch.device(device)

    def _prob_clf(x: Tensor) -> Tensor:
        with torch.no_grad():
            logits = classifier(x.to(clf_device))
        return torch.softmax(logits, dim=-1)

    return _prob_clf


# ---------------------------------------------------------------------------
# Single (dataset, window_size) cell
# ---------------------------------------------------------------------------


def run_cell(
    dataset_name: str,
    window_len: int,
    device: str,
) -> dict[str, Any]:
    """Run WindowSHAP with ``window_len`` on ``dataset_name`` and return metrics."""
    method_tag = f"windowshap_w{window_len}"
    result_path = (
        RESULTS_ROOT / dataset_name / CLF_NAME / method_tag / "result.json"
    )

    if result_path.exists():
        log.info("SKIP %s / %s (cached)", dataset_name, method_tag)
        return json.loads(result_path.read_text())

    t0 = time.time()
    log.info("=== %s | w=%d ===", dataset_name, window_len)

    # ---- Dataset -------------------------------------------------------
    ds_cfg = _load_cfg("data", dataset_name)
    dataset, K = _instantiate_dataset(ds_cfg)
    J, F, T = dataset.shape
    n_classes = int(dataset.metadata.get("n_classes", 3))
    n_seq = min(N_SEQ, len(dataset))
    log.info("  dataset: J=%d F=%d T=%d K=%d n_seq=%d", J, F, T, K, n_seq)

    # ---- Classifier ----------------------------------------------------
    clf_cfg = _load_cfg("classifiers", CLF_NAME)
    clf_device = torch.device(device)
    classifier = _build_classifier(clf_cfg, J, F, T, K, n_classes)
    classifier = classifier.to(clf_device)
    classifier.eval()

    ckpt_path = CKPT_ROOT / dataset_name / f"{CLF_NAME}.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=clf_device)
        classifier.load_state_dict(ckpt["model_state_dict"])
        log.info("  loaded checkpoint (val_acc=%.3f)", ckpt.get("val_acc", float("nan")))
    else:
        log.warning("  no checkpoint at %s — random init!", ckpt_path)

    prob_clf = _build_prob_clf(classifier, device)

    # ---- Players -------------------------------------------------------
    players = TemporalWindows(K=K, J=J, F=F, T=T)

    # ---- WindowSHAP attributor -----------------------------------------
    # w=16 with T=16 is invalid (window_len must be < T); clamp to T-1.
    # A window of T-1 frames means nearly the whole sequence is one "super-window"
    # and only a single stride step falls inside the sequence — a degenerate case
    # that is useful diagnostically to show how performance degrades at the limit.
    effective_window_len = min(window_len, T - 1)
    if effective_window_len != window_len:
        log.warning(
            "  window_len=%d >= T=%d; clamped to %d",
            window_len, T, effective_window_len,
        )

    attributor = WindowSHAPAttributor(
        classifier=prob_clf,
        window_len=effective_window_len,
        stride=effective_window_len,   # non-overlapping, same as default
        seed=42,
    )

    # ---- Attribution loop ----------------------------------------------
    phi_list: list[Tensor] = []
    x_list: list[Tensor] = []
    target_list: list[int] = []

    for i in range(n_seq):
        x_i, _ = dataset[i]
        x_i = x_i.float()

        with torch.no_grad():
            logits_i = classifier(x_i.unsqueeze(0).to(clf_device))
        if logits_i.ndim == 2:
            target_i = int(logits_i.argmax(dim=-1).item())
        else:
            target_i = 0

        try:
            phi_i = attributor.attribute(x_i, players, target=target_i)
        except Exception as exc:
            log.warning("  seq %d: attribution error — %s", i, exc)
            continue

        phi_list.append(phi_i.cpu())
        x_list.append(x_i.cpu())
        target_list.append(target_i)

        if (i + 1) % 10 == 0:
            log.info("  attributed %d/%d seqs (%.1fs)", i + 1, n_seq, time.time() - t0)

    log.info("  attributed %d seqs total (%.1fs)", len(phi_list), time.time() - t0)

    # ---- EC1 evaluation ------------------------------------------------
    oracle = getattr(dataset, "oracle", None)
    ec1_vals: list[float] = []

    if oracle is None:
        log.warning("  no oracle — EC1 skipped")
    else:
        metric = EC1Metric(n_mc=ORACLE_N_MC)

        def _make_clf_fn(tgt: int):
            def _fn(b: Tensor) -> Tensor:
                with torch.no_grad():
                    logits = classifier(b.to(clf_device))
                return torch.softmax(logits, dim=-1)[:, tgt]
            return _fn

        for idx, (phi_i, x_i, tgt_i) in enumerate(zip(phi_list, x_list, target_list)):
            try:
                # Pre-compute oracle phi once, then wrap to avoid redundant calls
                phi_oracle = oracle.true_shapley(
                    x_i, _make_clf_fn(tgt_i), players,
                    n_mc=ORACLE_N_MC, n_coalitions=ORACLE_N_COALITIONS,
                )
                result_dict = metric.evaluate(
                    phi=phi_i,
                    x=x_i,
                    classifier=_make_clf_fn(tgt_i),
                    players=players,
                    target=tgt_i,
                    oracle=type(
                        "_CachedOracle",
                        (),
                        {
                            "true_shapley": lambda self, *a, **kw: phi_oracle,
                            "conditional_sample": oracle.conditional_sample,
                            "__getattr__": lambda self, n: getattr(oracle, n),
                        },
                    )(),
                )
                ec1_vals.append(float(result_dict["ec1"]))
            except Exception as exc:
                log.warning("  seq %d: metric error — %s", idx, exc)

    mean_ec1 = float(np.mean(ec1_vals)) if ec1_vals else float("nan")
    log.info(
        "  EC1=%.5f over %d seqs (%.1fs)",
        mean_ec1, len(ec1_vals), time.time() - t0,
    )

    result = {
        "dataset": dataset_name,
        "classifier": CLF_NAME,
        "method": method_tag,
        "window_len": window_len,
        "effective_window_len": effective_window_len,
        "n_sequences": len(phi_list),
        "n_ec1_seqs": len(ec1_vals),
        "ec1": mean_ec1,
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    log.info("  saved → %s", result_path)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t_total = time.time()
    device = DEVICE

    # Summary table: dataset -> window_len -> ec1
    summary: dict[str, dict[int, float]] = {ds: {} for ds in DATASETS}

    for dataset_name in DATASETS:
        for w in WINDOW_SIZES:
            result = run_cell(dataset_name, w, device)
            summary[dataset_name][w] = result.get("ec1", float("nan"))

    log.info("=== DONE (%.1fs) ===", time.time() - t_total)
    log.info("")
    log.info("EC1 summary table:")
    log.info("  %-6s  %-12s  %-12s", "w", "gaussian_k4", "skel+gait")
    for w in WINDOW_SIZES:
        log.info(
            "  w=%-4d  %.5f       %.5f",
            w,
            summary["gaussian_k4"].get(w, float("nan")),
            summary["skeleton_gait_combined"].get(w, float("nan")),
        )


if __name__ == "__main__":
    main()
