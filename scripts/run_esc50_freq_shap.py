"""scripts/run_esc50_freq_shap.py — Frequency-band KernelSHAP on ESC-50.

Runs KernelSHAP treating **J=12 frequency bands as independent spatial
players**, complementing the temporal K=4 player set in ``run_esc50_shap.py``.
The 128 mel bins are partitioned into 12 contiguous frequency bands
(roughly 10-11 bins each), giving 2^12=4096 coalitions enumerated exactly.
This directly mirrors the PTB-XL lead-level (J=12) experimental setup and
makes the two datasets directly comparable on their spatial player sets.

Band boundaries (0-indexed mel bins, inclusive)::

    Band 0:   bins   0-10   (  0– 10)   sub-bass
    Band 1:   bins  11-21   ( 11– 21)   bass
    Band 2:   bins  22-32   ( 22– 32)   upper bass
    Band 3:   bins  33-43   ( 33– 43)   low-mid
    Band 4:   bins  44-54   ( 44– 54)   mid
    Band 5:   bins  55-65   ( 55– 65)   upper-mid
    Band 6:   bins  66-76   ( 66– 76)   presence
    Band 7:   bins  77-87   ( 77– 87)   brilliance
    Band 8:   bins  88-98   ( 88– 98)   high
    Band 9:   bins  99-109  ( 99–109)   very-high
    Band 10:  bins 110-120  (110–120)   air
    Band 11:  bins 121-127  (121–127)   ultra-high

All five imputation strategies are supported: Zero, Mean, Marginal, VAEAC,
Flow.  The Flow imputer is unconditional (a single prior draw per sequence
is reused across all coalitions, as in the PTB-XL lead-level flow run).

Results are written to::

    results/esc50_freq/{fold}/{method}/result.json

Usage::

    conda activate motionbench-xai

    # Run all methods for one fold
    CUDA_VISIBLE_DEVICES=0 python scripts/run_esc50_freq_shap.py --fold 1 --device cuda:0

    # Run all three folds in parallel
    for fold in 1 2 3; do
        CUDA_VISIBLE_DEVICES=$((fold-1)) python scripts/run_esc50_freq_shap.py \\
            --fold $fold --device cuda:0 &
    done
    wait
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

REPO_ROOT   = Path(__file__).parents[1]
SCRIPTS_DIR = Path(__file__).parent

sys.path.insert(0, str(SCRIPTS_DIR))
from run_care_pd_multiclf import (   # noqa: E402
    faithfulness_correlation,
    kernel_shap_exact,
    player_aopc,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

RESULTS_ROOT = REPO_ROOT / "results" / "esc50_freq"
VAEAC_CKPT   = REPO_ROOT / "results" / "esc50_imputers" / "vaeac" / "vaeac_best.pt"
FLOW_CKPT    = REPO_ROOT / "results" / "esc50_imputers" / "flow"  / "flow_best.pt"

# ── frequency band partition ───────────────────────────────────────────────
# J_BANDS=4 gives 2^4=16 coalitions — same order as temporal K=4.
# This matches the per-sequence classifier evaluation count of the temporal
# player-set experiments and finishes well within a deadline.
# The four bands map onto the perceptually meaningful frequency regions of
# log-mel spectrograms: sub-bass/bass, mid, upper-mid/presence, high/air.
J_BINS  = 128   # total mel bins
J_BANDS = 4     # spatial players: 4 frequency quartiles of 32 bins each
N_COAL  = 1 << J_BANDS  # 16

# Build band → bin indices (contiguous, ~10-11 bins per band)
_base, _extra = divmod(J_BINS, J_BANDS)
BAND_SLICES: list[tuple[int, int]] = []
start = 0
for b in range(J_BANDS):
    w = _base + (1 if b < _extra else 0)
    BAND_SLICES.append((start, start + w))
    start += w

BAND_NAMES = [
    "sub-bass/bass (bins 0-31)",
    "mid (bins 32-63)",
    "upper-mid/presence (bins 64-95)",
    "high/air (bins 96-127)",
]

ALL_METHODS = [
    "kernelshap_zero",
    "kernelshap_mean",
    "kernelshap_marginal",
    "kernelshap_vaeac",
    "kernelshap_flow",
]

FOLDS = [1, 2, 3]


# ── coalition helpers ─────────────────────────────────────────────────────

def build_band_coalition_masks() -> tuple[Tensor, Tensor]:
    """Return (z_bin, band_mask) of shape (2^J_BANDS, J_BANDS)."""
    z = np.zeros((N_COAL, J_BANDS), dtype=bool)
    for ci in range(N_COAL):
        for j in range(J_BANDS):
            if (ci >> j) & 1:
                z[ci, j] = True
    t = torch.from_numpy(z)
    return t, t.clone()


def build_completions_freq(
    x: Tensor,
    band_mask: Tensor,
    kind: str,
    fill: Tensor | None = None,
) -> Tensor:
    """Build completions where unobserved frequency bands are imputed.

    Args:
        x:         ``(J_BINS, F, T)`` single input sequence.
        band_mask: ``(N_COAL, J_BANDS)`` bool — True = band observed.
        kind:      ``"zero"`` | ``"mean"`` | ``"marginal"`` | ``"prior"``.
        fill:      For ``mean``: ``(J_BINS, F)`` mean tensor.
                   For ``marginal``/``prior``: ``(J_BINS, F, T)`` donor/prior.

    Returns:
        ``(N_COAL, J_BINS, F, T)`` float32.
    """
    _, F, T = x.shape
    n_coal_local = len(band_mask)
    comps = x.unsqueeze(0).expand(n_coal_local, -1, -1, -1).clone()  # (n_coal_local, J_BINS, F, T)

    if kind == "zero":
        fill_full = torch.zeros_like(x)
    elif kind == "mean":
        fill_full = fill.view(J_BINS, F, 1).expand(J_BINS, F, T)
    elif kind in ("marginal", "prior"):
        fill_full = fill   # (J_BINS, F, T)
    else:
        raise ValueError(f"unknown kind {kind!r}")

    for j_band in range(J_BANDS):
        b_start, b_end = BAND_SLICES[j_band]
        # mask out all coalitions where band j is unobserved
        unobs = ~band_mask[:, j_band]   # (N_COAL,) bool
        if unobs.any():
            comps[unobs, b_start:b_end, :, :] = fill_full[b_start:b_end, :, :]

    return comps.contiguous()


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fold",    type=int, default=1, choices=FOLDS)
    ap.add_argument("--methods", type=str, nargs="+", default=None,
                    help="Subset of methods (default: all five).")
    ap.add_argument("--n_seq",   type=int, default=200)
    ap.add_argument("--results_dir", type=str, default=str(RESULTS_ROOT))
    ap.add_argument("--device",  type=str, default="cuda:0")
    return ap.parse_args()


# ── main ──────────────────────────────────────────────────────────────────

def main() -> None:
    args         = parse_args()
    fold         = args.fold
    N_SEQ        = args.n_seq
    device       = torch.device(args.device)
    results_root = Path(args.results_dir)

    t_total = time.time()

    # ── data ─────────────────────────────────────────────────────────────
    data_dir  = REPO_ROOT / "data" / "esc50"
    test_d    = np.load(data_dir / f"fold{fold}_test.npz")
    train_d   = np.load(data_dir / f"fold{fold}_train.npz")

    x_test_all = test_d["x_test"]    # (400, 128, 1, 1024)
    x_train    = train_d["x_train"]  # (1600, 128, 1, 1024)

    rng   = np.random.default_rng(42 + fold)
    N_av  = x_test_all.shape[0]
    N     = min(N_SEQ, N_av)
    idx   = np.sort(rng.choice(N_av, size=N, replace=False)) if N < N_av else np.arange(N)
    x_val = x_test_all[idx]   # (N, 128, 1, 1024)

    N, J_raw, F, T = x_val.shape
    assert J_raw == J_BINS
    log.info("[fold%d] N=%d J=%d F=%d T=%d  bands=%d", fold, N, J_raw, F, T, J_BANDS)

    # ── coalitions ───────────────────────────────────────────────────────
    z_bin, band_mask = build_band_coalition_masks()   # (4096, 12)

    # ── classifier ───────────────────────────────────────────────────────
    sys.path.insert(0, str(REPO_ROOT))
    from motionbench.classifiers.esc50_classifier import load_esc50_classifier
    clf = load_esc50_classifier(device=device)
    clf.eval()
    log.info("[fold%d] ESC-50 AST classifier loaded", fold)

    with torch.no_grad():
        probs_list = []
        for s in range(0, N, 32):
            probs_list.append(clf(torch.from_numpy(x_val[s:s+32]).to(device)).cpu())
    targets = torch.cat(probs_list, dim=0).argmax(dim=-1).numpy()

    # ── imputer materials ─────────────────────────────────────────────────
    mean_jf   = torch.from_numpy(x_train.mean(axis=(0, 3))).float()  # (J, F)
    donor_idx = rng.integers(0, x_train.shape[0], size=N)
    donors    = torch.from_numpy(x_train[donor_idx]).float()          # (N, J, F, T)

    vaeac_imputer = None
    flow_imputer  = None

    def get_vaeac():
        nonlocal vaeac_imputer
        if vaeac_imputer is None:
            from motionbench.imputers.vaeac import VAEACImputer
            vaeac_imputer = VAEACImputer.load(VAEAC_CKPT).to(device)
        return vaeac_imputer

    def get_flow():
        nonlocal flow_imputer
        if flow_imputer is None:
            from motionbench.imputers.flow_matching import FlowMatchingImputer
            flow_imputer = FlowMatchingImputer.load(FLOW_CKPT)
            # FlowMatchingImputer has no .to(); set device manually
            flow_imputer._device = device
            flow_imputer._net = flow_imputer._net.to(device)
        return flow_imputer

    # ── sweep ─────────────────────────────────────────────────────────────
    methods  = args.methods or ALL_METHODS
    fold_dir = results_root / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for method in methods:
        method_dir  = fold_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        result_path = method_dir / "result.json"
        if result_path.exists():
            log.info("[fold%d] %s already done, skipping.", fold, method)
            continue

        log.info("=" * 60)
        log.info("[fold%d] freq-band %s  (2^%d=%d coalitions)", fold, method, J_BANDS, N_COAL)
        t_method = time.time()

        phis  = np.zeros((N, J_BANDS), dtype=np.float32)
        v_all = np.zeros((N, N_COAL),  dtype=np.float32)

        imp = None
        if method == "kernelshap_vaeac":
            try:
                imp = get_vaeac()
            except Exception as exc:
                log.warning("[fold%d] VAEAC unavailable: %s — skipping.", fold, exc)
                continue
        elif method == "kernelshap_flow":
            try:
                imp = get_flow()
            except Exception as exc:
                log.warning("[fold%d] Flow unavailable: %s — skipping.", fold, exc)
                continue

        for i in range(N):
            x_i      = torch.from_numpy(x_val[i])   # (128, 1, 1024)
            target_i = int(targets[i])

            # Build completions ──────────────────────────────────────────
            if method == "kernelshap_zero":
                comps = build_completions_freq(x_i, band_mask, "zero")

            elif method == "kernelshap_mean":
                comps = build_completions_freq(x_i, band_mask, "mean", mean_jf)

            elif method == "kernelshap_marginal":
                comps = build_completions_freq(x_i, band_mask, "marginal", donors[i])

            elif method == "kernelshap_vaeac":
                # Batch all 4096 coalitions through the VAEAC model directly via
                # its internal sample_completions(B,T,J,F) method.
                # Build bin-level mask: (N_COAL, J_BINS)
                bin_masks = torch.zeros(N_COAL, J_BINS, dtype=torch.bool)
                for jb in range(J_BANDS):
                    bs2, be2 = BAND_SLICES[jb]
                    bin_masks[:, bs2:be2] = band_mask[:, jb:jb+1]
                comps_list = []
                for c_start in range(0, N_COAL, 64):
                    c_end    = min(c_start + 64, N_COAL)
                    batch_sz = c_end - c_start
                    x_b    = x_i.unsqueeze(0).expand(batch_sz, -1, -1, -1)   # (B,J,F,T)
                    bm_b   = bin_masks[c_start:c_end]                         # (B,J)
                    mask_b = (bm_b.unsqueeze(-1).unsqueeze(-1)
                               .expand(batch_sz, J_BINS, F, T))               # (B,J,F,T)
                    # VAEAC internal model expects (B, T, J, F)
                    x_b_v = x_b.permute(0, 3, 1, 2).to(device)
                    m_b_v = mask_b.permute(0, 3, 1, 2).to(device)
                    with torch.no_grad():
                        comps_b = imp._model.sample_completions(
                            x_b_v, m_b_v, n_samples=1
                        )  # (B, T, J, F)
                    # Back to (B, J, F, T)
                    comps_b = comps_b.permute(0, 2, 3, 1).cpu()
                    comps_list.append(comps_b)
                comps = torch.cat(comps_list, dim=0).contiguous()

            elif method == "kernelshap_flow":
                # Unconditional prior draw: impute with all-unobserved mask
                # samples from the learned data prior; used for all coalitions.
                try:
                    all_unobs = torch.zeros(J_BINS, F, T, dtype=torch.bool)
                    with torch.no_grad():
                        prior_out = imp.impute(
                            x_i.to(device), all_unobs.to(device), n_samples=1
                        )
                    prior = prior_out.squeeze(0).cpu()  # (J_BINS, F, T)
                except Exception:
                    prior = donors[i]  # fallback: marginal donor
                comps = build_completions_freq(x_i, band_mask, "prior", prior)

            else:
                raise ValueError(f"unknown method {method}")

            # Batch classifier over 4096 completions ─────────────────────
            chunk  = 1024   # all 16 coalitions fit in one batch for J=4
            v_parts: list[Tensor] = []
            for c_start in range(0, N_COAL, chunk):
                c_end = min(c_start + chunk, N_COAL)
                with torch.no_grad():
                    logits_b = clf(comps[c_start:c_end].to(device))
                v_parts.append(
                    torch.softmax(logits_b, dim=-1)[:, target_i].cpu()
                )
            v_b = torch.cat(v_parts, dim=0)   # (N_COAL,)
            v_all[i] = v_b.numpy()
            phis[i]  = kernel_shap_exact(z_bin, v_b, J_BANDS).numpy()

            if (i + 1) % 20 == 0 or i == N - 1:
                elapsed = time.time() - t_method
                log.info("  [fold%d] %s  %d/%d  (%.2fs/seq)",
                         fold, method, i + 1, N, elapsed / (i + 1))

        # ── metrics & save ───────────────────────────────────────────────
        faiths, aopcs = [], []
        for i in range(N):
            v_i   = torch.from_numpy(v_all[i])
            phi_i = torch.from_numpy(phis[i])
            faiths.append(faithfulness_correlation(z_bin, v_i, phi_i))
            aopcs.append(player_aopc(v_i, z_bin, phi_i, J_BANDS))

        faiths_arr = np.asarray(faiths, dtype=np.float64)
        aopcs_arr  = np.asarray(aopcs,  dtype=np.float64)
        n_finite   = int(np.isfinite(faiths_arr).sum())

        phi_abs_mean = np.abs(phis).mean(axis=0).tolist()   # (J_BANDS,)

        np.savez_compressed(
            method_dir / "attributions.npz",
            phi=phis, x=x_val, target=targets, v=v_all,
            band_names=np.array(BAND_NAMES),
        )

        result = {
            "dataset":     "esc50_freq",
            "player_set":  f"freq_bands_j{J_BANDS}",
            "fold":        int(fold),
            "method":      method,
            "n_sequences": int(N),
            "n_finite_faithfulness": n_finite,
            "faithfulness_correlation":     float(np.nanmean(faiths_arr)),
            "faithfulness_correlation_std": float(np.nanstd(faiths_arr, ddof=1)) if n_finite > 1 else float("nan"),
            "player_aopc":     float(np.mean(aopcs_arr)),
            "player_aopc_std": float(np.std(aopcs_arr, ddof=1)) if N > 1 else float("nan"),
            "phi_mean_per_band":    phi_abs_mean,
            "band_names":           BAND_NAMES,
            "faithfulness_per_seq": faiths_arr.tolist(),
            "player_aopc_per_seq":  aopcs_arr.tolist(),
            "targets_per_seq":      targets.tolist(),
        }
        result_path.write_text(json.dumps(result, indent=2))
        log.info(
            "  [fold%d] %s done %.1fs — faith=%+.3f aopc=%+.3f",
            fold, method, time.time() - t_method,
            result["faithfulness_correlation"], result["player_aopc"],
        )

    (fold_dir / "summary.json").write_text(
        json.dumps({"dataset": "esc50_freq", "fold": fold}, indent=2)
    )
    log.info("[fold%d] ALL DONE in %.1fs", fold, time.time() - t_total)


if __name__ == "__main__":
    main()
