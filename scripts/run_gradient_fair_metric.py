"""scripts/run_gradient_fair_metric.py — Evaluate IG, DeepLIFT, SmoothGrad on
CARE-PD real data using non-Shapley metrics only: FaithfulnessCorrelation and
PlayerAOPC.

Addresses reviewer concern C4: gradient-based attribution methods (IG, DeepLIFT,
SmoothGrad) are not Shapley approximations, so EC1 (Shapley ground-truth
correlation) is an inappropriate metric for them.  This script evaluates gradient
methods solely on faithfulness metrics, using the same K=4 temporal window player
abstraction and data split as run_care_pd_extended.py.

Player-level attributions
    phi_k = sum_{j,f,t in window_k} |attr(j,f,t)|

i.e. the sum of absolute coordinate-level attributions within each temporal
window.  This is the standard gradient-saliency → player-set reduction.

Coalition value function
    The perturbation baseline for computing v(S) is zero-imputation (standard
    for gradient saliency faithfulness evaluation).  This is distinct from the
    imputers used by the KernelSHAP variants; all gradient methods share the same
    v(S) to make their faithfulness scores directly comparable to each other.

LRP
    LRP with vanilla backward rules (Captum) on transformer architectures produces
    unreliable attributions; attention-aware LRP (Ali et al. 2022) is deferred to
    the camera-ready.  A note JSON is written instead of numeric results.

Usage::

    conda activate motionbench-xai
    CUDA_VISIBLE_DEVICES=7 python scripts/run_gradient_fair_metric.py --fold 1
    CUDA_VISIBLE_DEVICES=7 python scripts/run_gradient_fair_metric.py --fold 1 \\
        --methods ig smoothgrad
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "care_pd_gradient_fair"
CARE_PD_ROOT = Path(os.environ.get("CARE_PD_ROOT", REPO_ROOT.parent / "CARE-PD"))
CACHE_TEMPLATE = str(
    CARE_PD_ROOT / "cache" / "flow_matching"
    / "BMCLab_h36m_80_classifier23fold_fold{fold}_eval" / "cache.npz"
)
CAREPD_ROOT = REPO_ROOT  # backwards-compat alias used elsewhere in this script
CKPT_TEMPLATE = (
    "motionbench/classifiers/checkpoints/real/"
    "carepd_bmclab_fold{fold}_motionbert.pt"
)

K = 4
DEVICE = "cuda:0"  # CUDA_VISIBLE_DEVICES controls which physical GPU


# ---------------------------------------------------------------------------
# Coalition masks  (identical to run_care_pd_extended.py)
# ---------------------------------------------------------------------------


def build_coalition_masks(K: int, T: int) -> tuple[Tensor, Tensor]:
    """Return (2^K, K) bool coalition indicators and (2^K, T) frame masks."""
    n_coal = 1 << K
    win_size = T // K
    z_bin = np.zeros((n_coal, K), dtype=bool)
    frame_mask = np.zeros((n_coal, T), dtype=bool)
    for ci in range(n_coal):
        for k in range(K):
            if (ci >> k) & 1:
                z_bin[ci, k] = True
                t0 = k * win_size
                t1 = t0 + win_size if k < K - 1 else T
                frame_mask[ci, t0:t1] = True
    return torch.from_numpy(z_bin), torch.from_numpy(frame_mask)


def build_completions_zero(x: Tensor, frame_mask_2k: Tensor) -> Tensor:
    """Zero-impute absent frames.  Returns (2^K, J, F, T)."""
    J, F, T = x.shape
    n_coal = frame_mask_2k.shape[0]
    obs = frame_mask_2k.view(n_coal, 1, 1, T).expand(n_coal, J, F, T)
    x_b = x.view(1, J, F, T).expand(n_coal, J, F, T)
    return torch.where(obs, x_b, torch.zeros_like(x_b)).contiguous()


# ---------------------------------------------------------------------------
# Metrics  (copied verbatim from run_care_pd_extended.py)
# ---------------------------------------------------------------------------


def faithfulness_correlation(z_bin: Tensor, v_vals: Tensor, phi: Tensor) -> float:
    """Pearson correlation between sum-of-attributions-of-removed-players
    and (v(N) - v(S)) over all coalitions."""
    z = z_bin.float()
    not_z = 1.0 - z
    sum_phi_absent = not_z @ phi
    delta = v_vals[-1] - v_vals
    a = sum_phi_absent.numpy()
    b = delta.float().numpy()
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def player_aopc(v_vals: Tensor, z_bin: Tensor, phi: Tensor, K: int) -> float:
    """Area Over Player-removal Curve."""
    order = torch.argsort(phi.abs(), descending=True).tolist()
    v_full = v_vals[-1].item()
    drops: list[float] = []
    cur_z = z_bin[-1].clone()
    for k in order:
        cur_z[k] = False
        idx = int((cur_z.int() * (1 << torch.arange(K))).sum().item())
        drops.append(v_full - v_vals[idx].item())
    if not drops:
        return 0.0
    return float(np.mean(drops))


# ---------------------------------------------------------------------------
# Gradient attribution helpers
# ---------------------------------------------------------------------------


class _TargetWrapper(nn.Module):
    """Wrap classifier to return scalar output for a single target class."""

    def __init__(self, clf: nn.Module, target: int) -> None:
        super().__init__()
        self._clf = clf
        self._target = target

    def forward(self, x: Tensor) -> Tensor:
        out = self._clf(x)
        if out.dim() == 2:
            return out[:, self._target]
        return out


def aggregate_abs_temporal(phi_coords: Tensor, K: int, T: int) -> Tensor:
    """Sum absolute coordinate-level attributions within each temporal window.

    Args:
        phi_coords: ``(J, F, T)`` attribution tensor (may be signed or positive).
        K: Number of temporal windows.
        T: Total time steps.

    Returns:
        ``(K,)`` float tensor — per-window sum of absolute values.
    """
    ws = T // K
    phi = torch.zeros(K, dtype=torch.float32)
    for k in range(K):
        t0 = k * ws
        t1 = (k + 1) * ws if k < K - 1 else T
        phi[k] = phi_coords[:, :, t0:t1].abs().sum()
    return phi


def _compute_attrs_on_device(
    method: str,
    clf: nn.Module,
    x: Tensor,
    target: int,
    device: torch.device,
    *,
    ig_n_steps: int = 50,
    sg_nt_samples: int = 50,
    sg_stdevs: float = 0.1,
) -> Tensor:
    """Run one gradient attribution method on the specified device.

    Returns ``(J, F, T)`` attribution tensor (raw; abs-aggregation happens outside).
    """
    from captum.attr import DeepLift, IntegratedGradients, NoiseTunnel, Saliency

    clf_dev = clf.to(device)
    x_dev = x.to(device)
    wrapped = _TargetWrapper(clf_dev, target)
    x_in = x_dev.unsqueeze(0)

    if method == "ig":
        ig = IntegratedGradients(wrapped)
        baseline = torch.zeros_like(x_in)
        with torch.enable_grad():
            attrs = ig.attribute(x_in, baselines=baseline, n_steps=ig_n_steps)
        return attrs.squeeze(0).detach().cpu()

    if method == "deeplift":
        dl = DeepLift(wrapped)
        baseline = torch.zeros_like(x_in)
        with torch.enable_grad():
            attrs = dl.attribute(x_in, baselines=baseline)
        return attrs.squeeze(0).detach().cpu()

    if method == "smoothgrad":
        nt = NoiseTunnel(Saliency(wrapped))
        with torch.enable_grad():
            attrs = nt.attribute(
                x_in,
                nt_type="smoothgrad",
                nt_samples=sg_nt_samples,
                stdevs=sg_stdevs,
                abs=True,
            )
        return attrs.squeeze(0).detach().cpu()

    raise ValueError(f"Unknown method {method!r}")


def compute_gradient_attrs(
    method: str,
    clf: nn.Module,
    x: Tensor,
    target: int,
    gpu_device: torch.device,
) -> Tensor:
    """Compute gradient attributions with automatic CPU fallback.

    Tries GPU first; falls back to CPU if any exception occurs (e.g. DeepLIFT
    CUDA hook conflicts).  Returns ``(J, F, T)`` attribution tensor.
    """
    try:
        return _compute_attrs_on_device(method, clf, x, target, gpu_device)
    except Exception as exc:
        log.warning(
            "  GPU attribution failed for %s (%s); retrying on CPU.", method, exc
        )
        # Move model back to GPU after CPU fallback so coalition value computation
        # in a subsequent call still works.
        result = _compute_attrs_on_device(method, clf, x, target, torch.device("cpu"))
        clf.to(gpu_device)
        return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--fold", type=int, default=1, choices=list(range(1, 24)),
        help="CARE-PD classifier fold (default: 1).",
    )
    ap.add_argument(
        "--n_seq", type=int, default=200,
        help="Number of validation sequences to evaluate (default: 200).",
    )
    ap.add_argument(
        "--methods", type=str, nargs="+", default=None,
        help="Subset of gradient methods to run; default = all three.",
    )
    ap.add_argument(
        "--results_dir", type=str, default=str(RESULTS_DIR),
        help="Output root directory (fold subdir created automatically).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    fold = args.fold
    N_SEQ = int(args.n_seq)
    results_root = Path(args.results_dir)

    cache_path = Path(CACHE_TEMPLATE.format(fold=fold))
    if not cache_path.exists():
        raise FileNotFoundError(
            f"CARE-PD eval cache not found for fold {fold}: {cache_path}"
        )
    ckpt_path = CAREPD_ROOT / CKPT_TEMPLATE.format(fold=fold)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"MotionBERT checkpoint missing: {ckpt_path}\n"
            f"Extract from CARE-PD/experiment_outs/Hypertune/.../fold{fold}/"
            "latest_epoch.pth.tr."
        )

    t_total = time.time()
    device = torch.device(DEVICE)

    # ------------------------------------------------------------------ data
    log.info("[fold%d] loading BMCLab eval cache from %s", fold, cache_path)
    d = np.load(cache_path, allow_pickle=True)
    x_val = np.transpose(d["x1_val"], (0, 2, 3, 1)).astype(np.float32)
    y_val = np.asarray(d["meta_updrs_gait_val"], dtype=np.int64)
    keep = y_val >= 0
    x_val = x_val[keep][:N_SEQ]
    N, J, F, T = x_val.shape
    log.info(
        "[fold%d] data: N=%d, J=%d, F=%d, T=%d, val_class_counts=%s",
        fold, N, J, F, T,
        np.bincount(y_val[y_val >= 0], minlength=3).tolist(),
    )
    if N < N_SEQ:
        log.warning("[fold%d] only %d valid sequences (asked for %d)", fold, N, N_SEQ)

    # -------------------------------------------------------------- coalitions
    z_bin, frame_mask = build_coalition_masks(K, T)
    n_coal = 1 << K
    log.info("[fold%d] coalitions: %d (K=%d, win_size=%d)", fold, n_coal, K, T // K)

    # ------------------------------------------------------------ classifier
    log.info("[fold%d] loading MotionBERT fold%d checkpoint...", fold, fold)
    t_load = time.time()
    from motionbench.classifiers.ported_care_pd.motionbert import MotionBERTClassifier
    clf = MotionBERTClassifier(
        n_classes=3,
        checkpoint_path=str(ckpt_path),
    ).to(device)
    clf.eval()
    log.info("[fold%d] MotionBERT loaded in %.1fs", fold, time.time() - t_load)

    with torch.no_grad():
        x_val_t = torch.from_numpy(x_val).to(device)
        logits = clf(x_val_t)
    targets = logits.argmax(dim=-1).cpu().numpy()
    log.info(
        "[fold%d] predicted class distribution: %s",
        fold, np.bincount(targets, minlength=3).tolist(),
    )

    # ------------------------------------------ coalition values (zero imputer)
    # Precompute v(S) for every sequence and all 2^K coalitions using zero
    # imputation.  These are shared across all gradient methods.
    log.info(
        "[fold%d] precomputing coalition values (zero imputer) for %d sequences...",
        fold, N,
    )
    t_coal = time.time()
    v_all = np.zeros((N, n_coal), dtype=np.float32)
    for i in range(N):
        x_i = torch.from_numpy(x_val[i])
        comps = build_completions_zero(x_i, frame_mask).to(device)  # (16, J, F, T)
        with torch.no_grad():
            logits_b = clf(comps)
        v_b = torch.softmax(logits_b, dim=-1)[:, int(targets[i])]
        v_all[i] = v_b.cpu().numpy()
        if (i + 1) % 50 == 0 or i == N - 1:
            log.info("  coalition values %d/%d", i + 1, N)
    log.info(
        "[fold%d] coalition values done in %.1fs", fold, time.time() - t_coal
    )

    # ------------------------------------------------- gradient methods loop
    all_methods = ["ig", "deeplift", "smoothgrad"]
    methods = args.methods if args.methods else all_methods
    for m in methods:
        if m not in all_methods:
            raise ValueError(f"Unknown method {m!r}; valid: {all_methods}")

    fold_dir = results_root / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for method in methods:
        method_dir = fold_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        result_path = method_dir / "result.json"

        log.info("=" * 60)
        log.info("[fold%d] Method: %s", fold, method)
        t_method = time.time()

        phis = np.zeros((N, K), dtype=np.float32)
        n_fallback = 0

        for i in range(N):
            x_i = torch.from_numpy(x_val[i])
            target_i = int(targets[i])

            try:
                phi_coords = _compute_attrs_on_device(
                    method, clf, x_i, target_i, device
                )
            except Exception as exc:
                log.warning(
                    "  [fold%d] %s seq %d: GPU failed (%s); retrying CPU.",
                    fold, method, i, exc,
                )
                n_fallback += 1
                phi_coords = _compute_attrs_on_device(
                    method, clf, x_i, target_i, torch.device("cpu")
                )
                clf.to(device)

            phi = aggregate_abs_temporal(phi_coords, K, T)
            phis[i] = phi.numpy()

            if (i + 1) % 25 == 0 or i == N - 1:
                log.info(
                    "  [fold%d] %s  seq %d/%d  (avg %.2fs/seq)",
                    fold, method, i + 1, N,
                    (time.time() - t_method) / (i + 1),
                )

        if n_fallback > 0:
            log.warning(
                "[fold%d] %s: %d/%d sequences fell back to CPU.",
                fold, method, n_fallback, N,
            )

        # ---------------------------------------------------------------- metrics
        faiths: list[float] = []
        aopcs: list[float] = []
        for i in range(N):
            v_i = torch.from_numpy(v_all[i])
            phi_i = torch.from_numpy(phis[i])
            faiths.append(faithfulness_correlation(z_bin, v_i, phi_i))
            aopcs.append(player_aopc(v_i, z_bin, phi_i, K))

        faiths_arr = np.asarray(faiths, dtype=np.float64)
        aopcs_arr = np.asarray(aopcs, dtype=np.float64)
        finite = np.isfinite(faiths_arr)
        n_finite = int(finite.sum())

        faith_mean = float(np.nanmean(faiths_arr))
        faith_std = (
            float(np.nanstd(faiths_arr, ddof=1)) if n_finite > 1 else float("nan")
        )
        aopc_mean = float(np.mean(aopcs_arr))
        aopc_std = float(np.std(aopcs_arr, ddof=1)) if N > 1 else float("nan")

        np.savez_compressed(
            method_dir / "attributions.npz",
            phi=phis,
            x=x_val,
            target=targets,
            v=v_all,
        )

        result = {
            "dataset": "care_pd_bmclab_cache",
            "classifier": "motionbert",
            "fold": int(fold),
            "method": method,
            "imputer_for_coalition_values": "zero",
            "n_sequences": int(N),
            "n_finite_faithfulness": n_finite,
            "n_cpu_fallback": n_fallback,
            "faithfulness_correlation": faith_mean,
            "faithfulness_correlation_std": faith_std,
            "player_aopc": aopc_mean,
            "player_aopc_std": aopc_std,
            "phi_mean": phis.mean(axis=0).tolist(),
            "phi_std": phis.std(axis=0).tolist(),
            # Per-sequence arrays for downstream bootstrap CIs
            "faithfulness_per_seq": faiths_arr.tolist(),
            "player_aopc_per_seq": aopcs_arr.tolist(),
            "targets_per_seq": targets.tolist(),
            "note": (
                f"{method.upper()} gradient attributions aggregated to K={K} "
                "temporal windows by summing absolute coordinate-level values "
                "(phi_k = sum_{j,f,t in window_k} |attr(j,f,t)|).  "
                "Faithfulness evaluated against zero-imputation coalition values.  "
                "EC1 (Shapley ground-truth correlation) intentionally excluded: "
                "gradient methods are not Shapley approximations (reviewer C4)."
            ),
        }
        result_path.write_text(json.dumps(result, indent=2))

        summary_rows.append({
            "method": method,
            "fold": int(fold),
            "n": int(N),
            "n_finite_faithfulness": n_finite,
            "faith_mean": faith_mean,
            "faith_std": faith_std,
            "aopc_mean": aopc_mean,
            "aopc_std": aopc_std,
        })
        log.info(
            "  [fold%d] done %s in %.1fs — faith=%+.3f±%.3f, aopc=%.4f±%.4f "
            "(n_finite=%d)",
            fold, method, time.time() - t_method,
            faith_mean, faith_std, aopc_mean, aopc_std, n_finite,
        )

    # ---------------------------------------------------------------- LRP note
    lrp_dir = fold_dir / "lrp"
    lrp_dir.mkdir(parents=True, exist_ok=True)
    lrp_note = {
        "method": "lrp",
        "note": (
            "LRP with vanilla backward rules (Captum) on transformer architectures "
            "produces unreliable attributions; attention-aware LRP (Ali et al. 2022) "
            "deferred to camera-ready"
        ),
    }
    (lrp_dir / "result.json").write_text(json.dumps(lrp_note, indent=2))
    log.info("[fold%d] LRP note written to %s", fold, lrp_dir / "result.json")

    # --------------------------------------------------------------- summary
    summary_path = fold_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2))

    log.info("=" * 60)
    log.info(
        "[fold%d] ALL DONE in %.1fs.  Results at %s",
        fold, time.time() - t_total, fold_dir,
    )
    for r in summary_rows:
        log.info(
            "  %-12s faith=%+.3f±%.3f  aopc=%+.4f±%.4f  (n=%d, n_finite=%d)",
            r["method"], r["faith_mean"], r["faith_std"],
            r["aopc_mean"], r["aopc_std"], r["n"], r["n_finite_faithfulness"],
        )


if __name__ == "__main__":
    main()
