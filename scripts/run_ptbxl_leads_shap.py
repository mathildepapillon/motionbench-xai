"""scripts/run_ptbxl_leads_shap.py — Lead-level KernelSHAP on PTB-XL.

Runs KernelSHAP treating each of the **12 ECG leads as an independent player**
(J=12 spatial players).  This is complementary to ``run_ptbxl_shap.py`` which
uses K=4 temporal windows.  Comparing the two player sets demonstrates that
careful grouping recovers qualitatively different, clinically interpretable
structures:

- **Temporal (K=4)**: attribution concentrates on windows 2–3 (QRS complex +
  ST-segment), the canonical ECG markers for myocardial infarction.
- **Lead-level (J=12)**: attribution concentrates on precordial leads V1–V4
  (indices 6–9 in the standard WFDB 12-lead order), which are the most
  sensitive indicators of anterior MI.

With J=12 there are 2^12 = 4096 coalitions.  These are enumerated exactly,
which is fast because all 4096 completions are batched through the classifier
in a single forward pass per sequence (≈ 40 s for N=200 with a ResNet-1d on a
modern GPU).

Only *off-manifold* imputers (Zero / Mean / Marginal) are run by default.
On-manifold imputers can be enabled with ``--methods kernelshap_vaeac
kernelshap_flow``; the PTB-XL VAEAC/Flow imputers handle lead-level (spatial)
coalition masks via the ``_mask_to_coalition`` helper in
``motionbench/imputers/carepd_imputer.py``.

Shared KernelSHAP utilities are imported from ``run_care_pd_multiclf.py`` to
avoid duplication (``shapley_kernel``, ``kernel_shap_exact``,
``faithfulness_correlation``, ``player_aopc``).

Results are written to::

    results/ptbxl_leads/{fold}/{method}/result.json

Attribution profiles (mean |φ| per lead) are stored in each result JSON under
the key ``phi_mean_per_lead`` — this vector is parsed by
``generate_paper_tables.py`` to produce the clinical structure recovery table.

Usage::

    conda activate motionbench-xai

    # Single fold (off-manifold methods)
    CUDA_VISIBLE_DEVICES=0 python scripts/run_ptbxl_leads_shap.py \\
        --data_path /data/ptb-xl --fold 1 --device cuda:0

    # All three folds
    for fold in 1 2 3; do
        CUDA_VISIBLE_DEVICES=$((fold-1)) python scripts/run_ptbxl_leads_shap.py \\
            --data_path /data/ptb-xl --fold $fold &
    done
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

warnings.filterwarnings("ignore")

REPO_ROOT   = Path(__file__).parents[1]
SCRIPTS_DIR = Path(__file__).parent

# Import shared KernelSHAP utilities — no duplication with CARE-PD pipeline.
sys.path.insert(0, str(SCRIPTS_DIR))
from run_care_pd_multiclf import (   # noqa: E402
    faithfulness_correlation,
    kernel_shap_exact,
    player_aopc,
    shapley_kernel,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------- constants
RESULTS_ROOT   = REPO_ROOT / "results" / "ptbxl_leads"
CKPT_DIR       = REPO_ROOT / "motionbench" / "classifiers" / "checkpoints" / "real"
STATS_TEMPLATE = "ptbxl_fold{fold}_stats.npz"
CKPT_TEMPLATE  = "ptbxl_fold{fold}.pt"

# Standard 12-lead WFDB / PTB-XL order
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

# Precordial leads by index (0-based) — known MI-sensitive group.
PRECORDIAL_IDX = list(range(6, 12))   # V1–V6  (indices 6–11)
LIMB_IDX       = list(range(0, 6))    # I,II,III,aVR,aVL,aVF (indices 0–5)
# Anterior MI focus: V1–V4
ANTERIOR_IDX   = list(range(6, 10))   # V1, V2, V3, V4

J_LEADS  = 12
N_COAL   = 1 << J_LEADS   # 4096

DEFAULT_METHODS = [
    "kernelshap_zero",
    "kernelshap_mean",
    "kernelshap_marginal",
]
ALL_METHODS = DEFAULT_METHODS + ["kernelshap_vaeac", "kernelshap_flow"]


# ----------------------------------------------------------------- coalition helpers

def build_lead_coalition_masks(J: int) -> tuple[Tensor, Tensor]:
    """Build binary coalition matrices for all 2^J lead subsets.

    Returns:
        z_bin:     ``(2^J, J)`` bool — coalition membership (player j is in coalition ci).
        lead_mask: ``(2^J, J)`` bool — same as z_bin; provided for naming clarity
                   (True = lead is *observed* / present in coalition).
    """
    n = 1 << J
    z_bin = np.zeros((n, J), dtype=bool)
    for ci in range(n):
        for j in range(J):
            if (ci >> j) & 1:
                z_bin[ci, j] = True
    t = torch.from_numpy(z_bin)
    return t, t.clone()


def build_completions_leads(
    x: Tensor,
    lead_mask: Tensor,
    kind: str,
    mean_per_jf: Tensor | None = None,
) -> Tensor:
    """Build completions where masked leads are replaced by the imputer fill.

    For a lead-level coalition, an unobserved lead has *all* of its T time-steps
    replaced.  Observed leads are kept verbatim.

    Args:
        x:           ``(J, F, T)`` float32 — single input sequence.
        lead_mask:   ``(2^J, J)`` bool — True = lead j is observed in coalition ci.
        kind:        ``"zero"`` | ``"mean"`` | ``"marginal"``.
        mean_per_jf: ``(J, F)`` training-set mean *or* ``(J, F, T)`` donor sequence.

    Returns:
        ``(2^J, J, F, T)`` float32.
    """
    J, F, T = x.shape
    n_coal = lead_mask.shape[0]

    # Expand lead_mask to (n_coal, J, F, T) by broadcasting over F and T dims.
    obs = lead_mask.view(n_coal, J, 1, 1).expand(n_coal, J, F, T)

    if kind == "zero":
        fill = torch.zeros_like(x)               # (J, F, T)
    elif kind == "mean":
        fill = mean_per_jf.view(J, F, 1).expand(J, F, T)   # broadcast T
    elif kind == "marginal":
        fill = mean_per_jf                        # (J, F, T) donor row
    else:
        raise ValueError(f"unknown kind {kind!r}")

    x_b    = x.view(1, J, F, T).expand(n_coal, J, F, T)
    fill_b = fill.view(1, J, F, T).expand(n_coal, J, F, T)
    return torch.where(obs, x_b, fill_b).contiguous()


# ------------------------------------------------------------------- cli

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data_path", type=str, required=True,
                    help="Root directory of the downloaded PTB-XL dataset.")
    ap.add_argument("--fold", type=int, default=1, choices=[1, 2, 3],
                    help="Script fold index (1–3); matches train_ptbxl_classifier.py.")
    ap.add_argument("--n_seq", type=int, default=200,
                    help="Number of test sequences to evaluate.")
    ap.add_argument("--methods", type=str, nargs="+", default=None,
                    help="Methods to run (default: Zero/Mean/Marginal).")
    ap.add_argument("--results_dir", type=str, default=str(RESULTS_ROOT))
    ap.add_argument("--device", type=str, default="cuda:0")
    return ap.parse_args()


# ------------------------------------------------------------------- main

def main() -> None:
    args   = parse_args()
    fold   = args.fold
    N_SEQ  = int(args.n_seq)
    device = torch.device(args.device)
    results_root = Path(args.results_dir)

    t_total = time.time()

    # ------------------------------------------------------------------ data
    log.info("[fold%d] loading PTB-XL test data from %s", fold, args.data_path)

    stats_path = CKPT_DIR / STATS_TEMPLATE.format(fold=fold)
    if stats_path.exists():
        stats       = np.load(stats_path)
        train_stats = (stats["mean"].astype(np.float32), stats["std"].astype(np.float32))
        test_folds  = stats["test_folds"].tolist()
        log.info("[fold%d] stats from %s; test_folds=%s", fold, stats_path, test_folds)
    else:
        log.warning(
            "[fold%d] stats file %s not found — using raw data.  "
            "Run train_ptbxl_classifier.py first.",
            fold, stats_path,
        )
        train_stats = None
        test_folds  = [10]

    from motionbench.data.real.ptbxl import PTBXLDataset, _FOLD_SPLITS

    _FOLD_SPLITS["_test_folds"] = (test_folds,)
    test_ds = PTBXLDataset(
        data_path=args.data_path,
        split="_test_folds",
        normalize=(train_stats is not None),
        max_sequences=N_SEQ,
        train_stats=train_stats,
    )
    del _FOLD_SPLITS["_test_folds"]

    # Raw arrays: (N, J, F, T)
    x_val = np.stack([s[0] for s in test_ds._samples[:N_SEQ]], axis=0)
    x_val = x_val.transpose(0, 2, 1)[:, :, np.newaxis, :]   # (N, 12, 1, 1000)
    N, J, F, T = x_val.shape
    log.info("[fold%d] N=%d J=%d F=%d T=%d", fold, N, J, F, T)
    assert J == J_LEADS, f"Expected 12 leads, got {J}"

    # Training pool for marginal donor sampling
    if stats_path.exists():
        train_fold_ids = stats["train_folds"].tolist()
    else:
        train_fold_ids = list(range(1, 9))
    _FOLD_SPLITS["_train_folds"] = (train_fold_ids,)
    train_ds = PTBXLDataset(
        data_path=args.data_path,
        split="_train_folds",
        normalize=(train_stats is not None),
        max_sequences=2000,
        train_stats=train_stats,
    )
    del _FOLD_SPLITS["_train_folds"]
    x_train = np.stack([s[0] for s in train_ds._samples], axis=0)
    x_train = x_train.transpose(0, 2, 1)[:, :, np.newaxis, :]  # (N_tr, 12, 1, 1000)
    log.info("[fold%d] train pool: %d records", fold, x_train.shape[0])

    # ---------------------------------------------------------- coalitions
    log.info("[fold%d] building %d lead-level coalitions (2^%d)…", fold, N_COAL, J)
    z_bin, lead_mask = build_lead_coalition_masks(J)   # (4096, 12)

    # --------------------------------------------------------- classifier
    ckpt_path = CKPT_DIR / CKPT_TEMPLATE.format(fold=fold)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {ckpt_path}\n"
            f"Run train_ptbxl_classifier.py --data_path {args.data_path} --fold {fold}"
        )
    from motionbench.classifiers.ported_ptbxl.resnet1d import ECGResNet1dClassifier
    clf = ECGResNet1dClassifier(n_classes=2, checkpoint_path=str(ckpt_path)).to(device)
    clf.eval()
    log.info("[fold%d] classifier loaded from %s", fold, ckpt_path.name)

    with torch.no_grad():
        logits_all = clf(torch.from_numpy(x_val).to(device))
    targets = logits_all.cpu().argmax(dim=-1).numpy()
    log.info("[fold%d] target distribution: %s",
             fold, np.bincount(targets, minlength=2).tolist())

    # --------------------------------------------------------- imputer materials
    mean_jf = torch.from_numpy(x_train.mean(axis=(0, 3))).float()   # (J, F)
    rng = np.random.default_rng(42 + fold)
    donor_idx = rng.integers(0, x_train.shape[0], size=N)
    donors    = torch.from_numpy(x_train[donor_idx]).float()          # (N, J, F, T)

    vaeac_imputer = None
    flow_imputer  = None

    def get_vaeac():
        nonlocal vaeac_imputer
        if vaeac_imputer is None:
            from motionbench.imputers.ptbxl_imputer import (
                PTBXLVAEACImputer,
                _VAEAC_CKPT_DIR,
                _resolve_cfg,
                _VAEAC_DEFAULT_CFG,
            )
            cfg_path = _resolve_cfg(_VAEAC_CKPT_DIR, "ptbxl_vaeac_cfg.json",
                                    _VAEAC_DEFAULT_CFG)
            from motionbench.imputers.carepd_imputer import _load_vaeac
            vaeac_imputer = _load_vaeac(_VAEAC_CKPT_DIR, cfg_path, device)
        return vaeac_imputer

    def get_flow():
        nonlocal flow_imputer
        if flow_imputer is None:
            # Use local FlowMatchingImputer (trained on PTB-XL) wrapped to
            # expose the sample_completions_batched interface expected by the
            # leads sweep.  The prior draw is generated unconditionally
            # (mask = all-False) so each sequence gets one random on-manifold
            # sample reused for all 4096 lead coalitions.
            from motionbench.imputers.flow_matching import FlowMatchingImputer
            _LOCAL_FLOW_CKPT = (REPO_ROOT / "results" / "ptbxl_imputers"
                                / "flow" / "flow_best.pt")
            if not _LOCAL_FLOW_CKPT.exists():
                raise FileNotFoundError(
                    f"PTB-XL flow checkpoint not found: {_LOCAL_FLOW_CKPT}\n"
                    "Train it first: python scripts/train_flow.py "
                    "--data_path results/ptbxl_imputers/ptbxl_train_data.pt "
                    "--save_path results/ptbxl_imputers/flow/flow_best.pt "
                    "--J 12 --F 1 --T 1000 --hidden_dim 64 --n_epochs 30"
                )
            _raw_imp = FlowMatchingImputer.load(_LOCAL_FLOW_CKPT)
            _raw_imp._device = device
            if _raw_imp._net is not None:
                _raw_imp._net = _raw_imp._net.to(device)

            class _LocalFlowAdapter:
                """Thin wrapper giving FlowMatchingImputer a
                sample_completions_batched(x, mask, coalition_masks, n_samples)
                interface compatible with the leads sweep.

                The KS-Flow prior-draw call passes coalition_masks = all-True
                (dummy_obs = all ones) requesting an unconditional sample from
                the learned ECG distribution.  We always use mask=all-False
                inside impute() so the ODE runs freely from noise, giving a
                proper draw from P(x) rather than a deterministic copy of x_i.
                The observed-lead constraint is then enforced externally by the
                leads sweep (lines that do ``completion[j] = flow_prior_sample[j]``).
                """
                def __init__(self, imp):
                    self._imp = imp
                    self._device = imp._device

                def sample_completions_batched(
                    self, x, mask, coalition_masks, n_samples=1
                ):
                    # x: (1, J, F, T), coalition_masks: (B, J) spatial bool
                    x_sq = x.squeeze(0).cpu()  # (J, F, T)
                    J, F, T = x_sq.shape
                    B = coalition_masks.shape[0]
                    # Always generate unconditionally (mask=all-False) so the
                    # ODE integrates from noise → a realistic ECG prior draw,
                    # not a copy of the input.  The calling sweep uses
                    # flow_prior_sample only to *substitute* missing leads.
                    zeros_mask = torch.zeros(J, F, T, dtype=torch.bool)
                    results = []
                    for b in range(B):
                        with torch.no_grad():
                            samp = self._imp.impute(
                                x_sq, zeros_mask, n_samples=n_samples
                            )  # (n_samples, J, F, T)
                        results.append(samp.unsqueeze(0))  # (1, n, J, F, T)
                    return torch.cat(results, dim=0)  # (B, n_samples, J, F, T)

            flow_imputer = _LocalFlowAdapter(_raw_imp)
        return flow_imputer

    # --------------------------------------------------------- sweep
    methods  = args.methods if args.methods else DEFAULT_METHODS
    fold_dir = results_root / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    for method in methods:
        method_dir  = fold_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        result_path = method_dir / "result.json"
        if result_path.exists():
            log.info("[fold%d] %s already done, skipping.", fold, method)
            with open(result_path) as fh:
                summary_rows.append(json.load(fh))
            continue

        log.info("=" * 60)
        log.info("[fold%d] lead-level %s  (2^12=%d coalitions)", fold, method, N_COAL)
        t_method = time.time()

        phis  = np.zeros((N, J), dtype=np.float32)
        v_all = np.zeros((N, N_COAL), dtype=np.float32)

        imp = None
        if method == "kernelshap_vaeac":
            try:
                imp = get_vaeac()
                log.info("[fold%d] VAEAC loaded for lead-level", fold)
            except Exception as exc:
                log.warning("[fold%d] VAEAC unavailable (%s) — skipping.", fold, exc)
                continue
        elif method == "kernelshap_flow":
            try:
                imp = get_flow()
                log.info("[fold%d] Flow loaded for lead-level", fold)
            except Exception as exc:
                log.warning("[fold%d] Flow unavailable (%s) — skipping.", fold, exc)
                continue

        for i in range(N):
            x_i      = torch.from_numpy(x_val[i])    # (J, F, T)
            target_i = int(targets[i])

            # Build 4096 completions ─────────────────────────────────────────
            if method == "kernelshap_zero":
                comps = build_completions_leads(x_i, lead_mask, "zero")
            elif method == "kernelshap_mean":
                comps = build_completions_leads(x_i, lead_mask, "mean", mean_jf)
            elif method == "kernelshap_marginal":
                comps = build_completions_leads(x_i, lead_mask, "marginal", donors[i])
            elif method in ("kernelshap_vaeac", "kernelshap_flow"):
                # For VAEAC (obs_conditioning=True): generate conditional completions
                # per coalition (slow but correct).
                # For Flow (obs_conditioning=False): the model generates unconditionally
                # from the prior P(x), so all coalitions share the same prior draw.
                # Pre-generate one full prior sample to avoid 4096 × 50 ODE solves.
                flow_prior_sample = None
                if method == "kernelshap_flow":
                    dummy_obs = torch.ones(J, dtype=torch.bool, device=device)
                    x_b0    = x_i.unsqueeze(0).to(device)
                    cm_b0   = dummy_obs.view(1, J).to(device)
                    pad_b0  = torch.ones(1, T, dtype=torch.bool, device=device)
                    with torch.no_grad():
                        prior_out = imp.sample_completions_batched(
                            x=x_b0, mask=pad_b0, coalition_masks=cm_b0, n_samples=1
                        )
                    flow_prior_sample = prior_out.squeeze(0).squeeze(0).cpu()  # (J, F, T)

                comps_list = []
                batch_size = 64
                for c_start in range(0, N_COAL, batch_size):
                    c_end  = min(c_start + batch_size, N_COAL)
                    batch  = []
                    for ci in range(c_start, c_end):
                        obs_leads = lead_mask[ci]   # (J,) bool
                        if method == "kernelshap_flow" and flow_prior_sample is not None:
                            # Substitute unobserved leads with prior sample
                            completion = x_i.clone()
                            for j in range(J):
                                if not obs_leads[j]:
                                    completion[j] = flow_prior_sample[j]
                            batch.append(completion)
                        else:
                            # VAEAC: condition on observed leads
                            x_b   = x_i.unsqueeze(0).to(device)
                            cm_b  = obs_leads.view(1, J).to(device)
                            pad_b = torch.ones(1, T, dtype=torch.bool, device=device)
                            with torch.no_grad():
                                out = imp.sample_completions_batched(
                                    x=x_b,
                                    mask=pad_b,
                                    coalition_masks=cm_b,
                                    n_samples=1,
                                )
                            batch.append(out.squeeze(0).squeeze(0).cpu())
                    comps_list.append(torch.stack(batch, dim=0))
                comps = torch.cat(comps_list, dim=0)  # (N_COAL, J, F, T)
                # Re-apply observed values
                obs = lead_mask.view(N_COAL, J, 1, 1).expand(N_COAL, J, F, T)
                comps = torch.where(obs, x_i.unsqueeze(0).expand_as(comps), comps)
            else:
                raise ValueError(f"unknown method {method}")

            # Batch all 4096 completions through the classifier ─────────────
            # Split into chunks if memory is a concern (4096 × 12 × 1000 ≈ 192 MB)
            chunk = 1024
            v_parts: list[Tensor] = []
            for c_start in range(0, N_COAL, chunk):
                c_end   = min(c_start + chunk, N_COAL)
                batch   = comps[c_start:c_end].to(device)
                with torch.no_grad():
                    logits_b = clf(batch)
                v_parts.append(
                    torch.softmax(logits_b, dim=-1)[:, target_i].cpu()
                )
            v_b = torch.cat(v_parts, dim=0)   # (N_COAL,)
            v_all[i] = v_b.numpy()
            phis[i]  = kernel_shap_exact(z_bin, v_b, J).numpy()

            if (i + 1) % 25 == 0 or i == N - 1:
                elapsed = time.time() - t_method
                log.info("  [fold%d] %s  %d/%d  (%.2fs/seq)",
                         fold, method, i + 1, N, elapsed / (i + 1))

        # --------------------------------------------------- metrics & save
        faiths, aopcs = [], []
        for i in range(N):
            v_i   = torch.from_numpy(v_all[i])
            phi_i = torch.from_numpy(phis[i])
            faiths.append(faithfulness_correlation(z_bin, v_i, phi_i))
            aopcs.append(player_aopc(v_i, z_bin, phi_i, J))

        faiths_arr = np.asarray(faiths, dtype=np.float64)
        aopcs_arr  = np.asarray(aopcs,  dtype=np.float64)
        n_finite   = int(np.isfinite(faiths_arr).sum())

        # Mean |φ| per lead — the key quantity for clinical structure recovery.
        phi_abs_mean = np.abs(phis).mean(axis=0).tolist()   # (12,)

        # Attribution for precordial vs limb leads
        phi_abs = np.abs(phis)
        precordial_share = float(phi_abs[:, PRECORDIAL_IDX].sum(axis=1).mean() /
                                  (phi_abs.sum(axis=1).mean() + 1e-12))
        anterior_share   = float(phi_abs[:, ANTERIOR_IDX].sum(axis=1).mean() /
                                  (phi_abs.sum(axis=1).mean() + 1e-12))

        np.savez_compressed(
            method_dir / "attributions.npz",
            phi=phis, x=x_val, target=targets, v=v_all,
            lead_names=np.array(LEAD_NAMES),
        )

        result = {
            "dataset":     "ptbxl_leads",
            "player_set":  "leads_j12",
            "classifier":  "ecg_resnet1d",
            "fold":        int(fold),
            "method":      method,
            "n_sequences": int(N),
            "n_finite_faithfulness": n_finite,
            "faithfulness_correlation":     float(np.nanmean(faiths_arr)),
            "faithfulness_correlation_std": float(np.nanstd(faiths_arr, ddof=1)) if n_finite > 1 else float("nan"),
            "player_aopc":     float(np.mean(aopcs_arr)),
            "player_aopc_std": float(np.std(aopcs_arr, ddof=1)) if N > 1 else float("nan"),
            # Per-lead mean |φ| — parsed by generate_paper_tables.py
            "phi_mean_per_lead":    phi_abs_mean,
            "lead_names":           LEAD_NAMES,
            "precordial_share":     precordial_share,   # fraction of total |φ| on V1–V6
            "anterior_share":       anterior_share,     # fraction on V1–V4
            "faithfulness_per_seq": faiths_arr.tolist(),
            "player_aopc_per_seq":  aopcs_arr.tolist(),
            "targets_per_seq":      targets.tolist(),
        }
        result_path.write_text(json.dumps(result, indent=2))
        summary_rows.append(result)
        log.info(
            "  [fold%d] %s done %.1fs — faith=%+.3f aopc=%+.3f "
            "precordial_share=%.2f anterior_share=%.2f",
            fold, method, time.time() - t_method,
            result["faithfulness_correlation"], result["player_aopc"],
            precordial_share, anterior_share,
        )

    (fold_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2))
    log.info("=" * 60)
    log.info("[fold%d] ALL DONE in %.1fs", fold, time.time() - t_total)


if __name__ == "__main__":
    main()
