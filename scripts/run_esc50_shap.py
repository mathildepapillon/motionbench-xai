"""scripts/run_esc50_shap.py — KernelSHAP sweep on ESC-50 audio.

Mirrors run_ptbxl_shap.py for the ESC-50 environmental sound dataset.
Runs KernelSHAP with five imputation strategies (Zero / Mean / Marginal /
VAEAC / Flow) on ESC-50 mel-spectrogram test sequences using the fine-tuned
AST classifier (bioamla/ast-esc50).

Player set
----------
K=4 temporal windows over T=1024 (each window = 256 time-steps).

Results are written to::

    results/esc50/{fold}/{method}/result.json

Usage::

    CUDA_VISIBLE_DEVICES=0 python scripts/run_esc50_shap.py \\
        --fold 1 --method kernelshap_zero --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).parents[1]
SCRIPTS_DIR = Path(__file__).parent

# Import shared KernelSHAP utilities from run_care_pd_multiclf — no duplication.
sys.path.insert(0, str(SCRIPTS_DIR))
from run_care_pd_multiclf import (   # noqa: E402
    build_coalition_masks,
    faithfulness_correlation,
    kernel_shap_exact,
    player_aopc,
    shapley_kernel,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

RESULTS_ROOT = REPO_ROOT / "results" / "esc50"
VAEAC_CKPT = REPO_ROOT / "results" / "esc50_imputers" / "vaeac" / "vaeac_best.pt"
FLOW_CKPT  = REPO_ROOT / "results" / "esc50_imputers" / "flow"  / "flow_best.pt"

K = 4       # temporal windows; each window = T//K = 256 time-steps
DEVICE = "cuda:0"

ALL_METHODS = [
    "kernelshap_zero",
    "kernelshap_mean",
    "kernelshap_marginal",
    "kernelshap_vaeac",
    "kernelshap_flow",
]


def build_completions_offmanifold(
    x: Tensor,
    frame_mask_2k: Tensor,
    kind: str,
    fill_tensor: Tensor | None = None,
) -> Tensor:
    """Build off-manifold completions for all 2^K coalitions.

    Args:
        x: ``(J, F, T)`` float32 single sequence.
        frame_mask_2k: ``(2^K, T)`` bool — True = observed frame.
        kind: ``"zero"`` | ``"mean"`` | ``"marginal"``.
        fill_tensor: ``(J, F, T)`` donor or ``(J, F)`` mean tensor.

    Returns:
        ``(2^K, J, F, T)`` float32 completions.
    """
    J, F, T = x.shape
    n_coal = frame_mask_2k.shape[0]
    obs = frame_mask_2k.view(n_coal, 1, 1, T).expand(n_coal, J, F, T)
    if kind == "zero":
        fill = torch.zeros_like(x)
    elif kind == "mean":
        fill = fill_tensor.view(J, F, 1).expand(J, F, T)
    elif kind == "marginal":
        fill = fill_tensor   # (J, F, T) donor
    else:
        raise ValueError(f"unknown kind {kind!r}")
    x_b    = x.view(1, J, F, T).expand(n_coal, J, F, T)
    fill_b = fill.view(1, J, F, T).expand(n_coal, J, F, T)
    return torch.where(obs, x_b, fill_b).contiguous()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fold", type=int, default=1, choices=[1, 2, 3],
                    help="Fold index (1–3).")
    ap.add_argument("--method", type=str, default=None,
                    help="Single method to run (default: all five).")
    ap.add_argument("--methods", type=str, nargs="+", default=None,
                    help="Subset of methods to run.")
    ap.add_argument("--n_seq", type=int, default=200,
                    help="Number of test sequences to evaluate.")
    ap.add_argument("--results_dir", type=str, default=str(RESULTS_ROOT))
    ap.add_argument("--device", type=str, default=DEVICE)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    fold = args.fold
    N_SEQ = int(args.n_seq)
    results_root = Path(args.results_dir)
    device = torch.device(args.device)

    t_total = time.time()

    # ------------------------------------------------------------------ data
    data_dir = REPO_ROOT / "data" / "esc50"
    test_npz  = data_dir / f"fold{fold}_test.npz"
    train_npz = data_dir / f"fold{fold}_train.npz"

    log.info("[fold%d] loading test data from %s", fold, test_npz)
    test_d  = np.load(test_npz)
    train_d = np.load(train_npz)

    x_test_all = test_d["x_test"]   # (400, 128, 1, 1024)
    y_test_all = test_d["y_test"]   # (400,)
    x_train    = train_d["x_train"] # (1600, 128, 1, 1024)

    # Subsample to N_SEQ (or all if fewer)
    N_avail = x_test_all.shape[0]
    N = min(N_SEQ, N_avail)
    rng = np.random.default_rng(42 + fold)
    if N < N_avail:
        idx = rng.choice(N_avail, size=N, replace=False)
        idx = np.sort(idx)
    else:
        idx = np.arange(N)

    x_val = x_test_all[idx]  # (N, 128, 1, 1024)
    y_val = y_test_all[idx]  # (N,)

    N, J, F, T = x_val.shape
    log.info("[fold%d] N=%d J=%d F=%d T=%d", fold, N, J, F, T)
    log.info("[fold%d] train pool: %d records", fold, x_train.shape[0])

    # ---------------------------------------------------------- coalitions
    z_bin, frame_mask = build_coalition_masks(K, T)
    n_coal = 1 << K

    # --------------------------------------------------------- classifier
    sys.path.insert(0, str(REPO_ROOT))
    from motionbench.classifiers.esc50_classifier import load_esc50_classifier
    clf = load_esc50_classifier(device=device)
    clf.eval()
    log.info("[fold%d] ESC-50 AST classifier loaded", fold)

    # Get predicted targets on the test set
    with torch.no_grad():
        x_val_t = torch.from_numpy(x_val).to(device)
        # Process in batches to avoid OOM
        probs_all = []
        bs = 32
        for start in range(0, N, bs):
            probs_batch = clf(x_val_t[start:start+bs])
            probs_all.append(probs_batch.cpu())
        probs_all = torch.cat(probs_all, dim=0)
    targets = probs_all.argmax(dim=-1).numpy()
    log.info("[fold%d] predicted targets (top-5): %s", fold,
             np.bincount(targets, minlength=50).argsort()[-5:][::-1].tolist())

    # --------------------------------------------------------- imputer setup
    mean_jf = torch.from_numpy(x_train.mean(axis=(0, 3))).float()  # (J, F)
    donor_idx = rng.integers(0, x_train.shape[0], size=N)
    donors = torch.from_numpy(x_train[donor_idx]).float()  # (N, J, F, T)

    vaeac_imputer = None
    flow_imputer  = None

    def get_vaeac():
        nonlocal vaeac_imputer
        if vaeac_imputer is None:
            from motionbench.imputers.vaeac import VAEACImputer
            vaeac_imputer = VAEACImputer.load(VAEAC_CKPT)
            vaeac_imputer = vaeac_imputer.to(device)
            # vaeac does not have .eval() (BaseImputer, not nn.Module)
        return vaeac_imputer

    def get_flow():
        nonlocal flow_imputer
        if flow_imputer is None:
            from motionbench.imputers.flow_matching import FlowMatchingImputer
            flow_imputer = FlowMatchingImputer.load(FLOW_CKPT)
            # FlowMatchingImputer uses _device and _net.to() directly
            flow_imputer._device = device
            flow_imputer._net = flow_imputer._net.to(device)
        return flow_imputer

    # --------------------------------------------------------- methods
    if args.method is not None:
        methods = [args.method]
    elif args.methods is not None:
        methods = args.methods
    else:
        methods = ALL_METHODS

    fold_dir = results_root / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    for method in methods:
        method_dir = fold_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        result_path = method_dir / "result.json"
        if result_path.exists():
            log.info("[fold%d] %s already done, skipping.", fold, method)
            with open(result_path) as fh:
                summary_rows.append(json.load(fh))
            continue

        log.info("=" * 60)
        log.info("[fold%d] Method: %s", fold, method)
        t_method = time.time()

        phis  = np.zeros((N, K), dtype=np.float32)
        v_all = np.zeros((N, n_coal), dtype=np.float32)

        imp = None
        if method == "kernelshap_vaeac":
            try:
                imp = get_vaeac()
                log.info("[fold%d] VAEAC loaded", fold)
            except Exception as exc:
                log.warning("[fold%d] VAEAC unavailable (%s) — skipping method.", fold, exc)
                continue
        elif method == "kernelshap_flow":
            try:
                imp = get_flow()
                log.info("[fold%d] Flow loaded", fold)
            except Exception as exc:
                log.warning("[fold%d] Flow unavailable (%s) — skipping method.", fold, exc)
                continue

        for i in range(N):
            x_i = torch.from_numpy(x_val[i])   # (J, F, T)
            target_i = int(targets[i])

            if method == "kernelshap_zero":
                comps = build_completions_offmanifold(x_i, frame_mask, "zero")
            elif method == "kernelshap_mean":
                comps = build_completions_offmanifold(x_i, frame_mask, "mean", mean_jf)
            elif method == "kernelshap_marginal":
                comps = build_completions_offmanifold(x_i, frame_mask, "marginal", donors[i])
            elif method in ("kernelshap_vaeac", "kernelshap_flow"):
                # Use impute() per coalition (no sample_completions_batched available)
                comps_list = []
                with torch.no_grad():
                    for c_idx in range(n_coal):
                        # Build (J, F, T) mask: True = observed frame
                        t_mask = frame_mask[c_idx]  # (T,) bool
                        mask_jft = t_mask.view(1, 1, T).expand(J, F, T).contiguous()
                        # impute returns (1, J, F, T)
                        comp = imp.impute(x_i.to(device), mask_jft.to(device), n_samples=1)
                        comps_list.append(comp.squeeze(0).cpu())
                comps = torch.stack(comps_list, dim=0).contiguous()  # (n_coal, J, F, T)
            else:
                raise ValueError(f"unknown method {method}")

            # Feed completions through classifier
            with torch.no_grad():
                probs_b = clf(comps.to(device))
            v_b = probs_b[:, target_i].cpu()
            v_all[i] = v_b.numpy()
            phis[i]  = kernel_shap_exact(z_bin, v_b, K).numpy()

            if (i + 1) % 25 == 0 or i == N - 1:
                elapsed = time.time() - t_method
                log.info("  [fold%d] %s  %d/%d  (%.2fs/seq)",
                         fold, method, i + 1, N, elapsed / (i + 1))

        # ------------------------------------------------------ metrics
        faiths, aopcs = [], []
        for i in range(N):
            v_i   = torch.from_numpy(v_all[i])
            phi_i = torch.from_numpy(phis[i])
            faiths.append(faithfulness_correlation(z_bin, v_i, phi_i))
            aopcs.append(player_aopc(v_i, z_bin, phi_i, K))

        faiths_arr = np.asarray(faiths, dtype=np.float64)
        aopcs_arr  = np.asarray(aopcs,  dtype=np.float64)
        n_finite   = int(np.isfinite(faiths_arr).sum())

        faith_mean = float(np.nanmean(faiths_arr))
        faith_std  = float(np.nanstd(faiths_arr, ddof=1)) if n_finite > 1 else float("nan")
        aopc_mean  = float(np.mean(aopcs_arr))
        aopc_std   = float(np.std(aopcs_arr, ddof=1)) if N > 1 else float("nan")

        np.savez_compressed(
            method_dir / "attributions.npz",
            phi=phis, x=x_val, target=targets, v=v_all,
        )
        result = {
            "dataset": "esc50",
            "classifier": "ast",
            "fold": int(fold),
            "method": method,
            "n_sequences": int(N),
            "n_finite_faithfulness": n_finite,
            "faithfulness_correlation": faith_mean,
            "faithfulness_correlation_std": faith_std,
            "player_aopc": aopc_mean,
            "player_aopc_std": aopc_std,
            "phi_mean": phis.mean(axis=0).tolist(),
            "phi_std":  phis.std(axis=0).tolist(),
            "faithfulness_per_seq": faiths_arr.tolist(),
            "player_aopc_per_seq":  aopcs_arr.tolist(),
            "targets_per_seq": targets.tolist(),
        }
        result_path.write_text(json.dumps(result, indent=2))
        summary_rows.append(result)
        log.info("  [fold%d] %s done %.1fs — faith=%+.3f aopc=%+.3f (n_fin=%d)",
                 fold, method, time.time() - t_method,
                 faith_mean, aopc_mean, n_finite)

    (fold_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2))
    log.info("=" * 60)
    log.info("[fold%d] ALL DONE in %.1fs", fold, time.time() - t_total)


if __name__ == "__main__":
    main()
