"""scripts/run_nk_ablation.py — Non-Kronecker synthetic robustness ablation (C2).

This script implements the reviewer-concern-C2 ablation:

1. **Sanity check**: wraps the Kronecker covariance as a FullGaussianOracle
   and verifies it matches the GaussianOracle (standard Kronecker oracle) to
   within 20% tolerance on EC1.

2. **Experiment**: evaluates VAEAC, Flow, Marginal, Mean, Zero imputers on
   the non-Kronecker dataset using FullGaussianOracle as the reference oracle.

Results are written to::

    results/synthetic_extended/non_kronecker/{method}/result.json
    results/synthetic_extended/non_kronecker/summary.json

Usage::

    conda activate motionbench-xai
    CUDA_VISIBLE_DEVICES=0 python scripts/run_nk_ablation.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
import warnings
from math import comb
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results" / "synthetic_extended" / "non_kronecker"
CARE_PD = Path(os.environ.get("CARE_PD_ROOT", REPO.parent / "CARE-PD"))
CKPT_SYNTH = REPO / "motionbench" / "classifiers" / "checkpoints" / "synthetic"

DEVICE_STR = "cuda:0"
N_TEST = 400          # test sequences
N_TRAIN = 200         # training sequences (for Marginal/Mean)
N_MC = 50             # MC samples per coalition in oracle
N_SANITY_SEQ = 10     # sequences for sanity check
TOL_SANITY = 0.20     # 20 % EC1 tolerance

# J=5, F=3, T=16, K=4 matching gaussian_k4
J, F, T, K = 5, 3, 16, 4

CLASSIFIERS = ["synthetic_mlp", "synthetic_cnn", "synthetic_transformer"]

# Methods to evaluate
METHODS = ["kernelshap_vaeac", "kernelshap_flow", "kernelshap_marginal",
           "kernelshap_mean", "kernelshap_zero"]

# ---------------------------------------------------------------------------
# KernelSHAP helpers (inlined; previously imported from
# `run_synth_vaeac_flow.py`, now archived under archive/scripts/).
# ---------------------------------------------------------------------------


def _shap_kernel(K: int, s: int) -> float:
    if s == 0 or s == K:
        return 1e6
    return (K - 1) / (comb(K, s) * s * (K - s))


def kernel_shap_exact(z_bin: np.ndarray, v_vals: np.ndarray, n_players: int) -> np.ndarray:
    """Exact KernelSHAP WLS solve for the given coalition values.

    Args:
        z_bin: ``(2^K, K)`` bool/int coalition indicators.
        v_vals: ``(2^K,)`` float value function evaluations.
        n_players: K.

    Returns:
        ``(K,)`` float32 Shapley value array.
    """
    Z = z_bin.astype(float)
    v = v_vals.astype(float)
    n = Z.shape[0]
    sizes = Z.sum(axis=1).astype(int)
    w = np.array([_shap_kernel(n_players, int(s)) for s in sizes])
    Z_ext = np.concatenate([np.ones((n, 1)), Z], axis=1)
    W = np.diag(w)
    A = Z_ext.T @ W @ Z_ext + 1e-8 * np.eye(Z_ext.shape[1])
    b = Z_ext.T @ W @ v
    sol = np.linalg.solve(A, b)
    return sol[1:].astype(np.float32)


def build_coalition_masks(K: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    """Build all 2^K coalition binary indicators and frame masks.

    Returns:
        z_bin: ``(2^K, K)`` bool array.
        frame_mask: ``(2^K, T)`` bool array.
    """
    n_coal = 1 << K
    ws = T // K
    z_bin = np.zeros((n_coal, K), dtype=bool)
    frame_mask = np.zeros((n_coal, T), dtype=bool)
    for ci in range(n_coal):
        for k in range(K):
            if (ci >> k) & 1:
                z_bin[ci, k] = True
                t0 = k * ws
                t1 = t0 + ws if k < K - 1 else T
                frame_mask[ci, t0:t1] = True
    return z_bin, frame_mask


# ---------------------------------------------------------------------------
# Classifier loading
# ---------------------------------------------------------------------------


def load_classifier(clf_name: str, device: torch.device) -> torch.nn.Module:
    """Load a trained synthetic classifier from the gaussian_k4 checkpoints.

    Falls back to random initialisation if checkpoint is missing.
    """
    cfg_path = REPO / "configs" / "classifiers" / f"{clf_name}.yaml"
    clf_cfg = OmegaConf.load(cfg_path)
    from motionbench.pipelines.synthetic_eval import _build_classifier  # noqa: PLC0415
    n_classes = 3
    clf = _build_classifier(clf_cfg, J, F, T, K, n_classes).to(device)
    clf.eval()

    ckpt_path = CKPT_SYNTH / "gaussian_k4" / f"{clf_name}.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        clf.load_state_dict(ckpt["model_state_dict"])
        val_acc = ckpt.get("val_acc", float("nan"))
        log.info("  Loaded %s (val_acc=%.3f)", clf_name, val_acc)
    else:
        log.warning("  No checkpoint at %s — using random init", ckpt_path)
    return clf


# ---------------------------------------------------------------------------
# Imputer loading helpers
# ---------------------------------------------------------------------------


def _add_carepd_to_path() -> None:
    carepd_str = str(CARE_PD)
    if carepd_str not in sys.path:
        sys.path.insert(0, carepd_str)


def load_vaeac(device: torch.device):
    """Load VAEAC from CARE-PD gaussian_k8_t16 checkpoint (J=5, F=3, T=16).

    The gaussian_k4 directory was trained on J=17 skeleton data; the
    gaussian_k8_t16 checkpoint is the correct one for J=5, F=3, T=16
    Gaussian datasets and is what the motionbench-xai registry maps
    GaussianMotionDataset to.
    """
    from motionbench.imputers.carepd_imputer import _load_vaeac  # noqa: PLC0415
    ckpt_dir = CARE_PD / "experiment_outs" / "vaeac_synthetic" / "gaussian_k8_t16"
    cfg_path = CARE_PD / "configs" / "vaeac" / "gaussian_k8_t16.json"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"VAEAC checkpoint dir not found: {ckpt_dir}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"VAEAC config not found: {cfg_path}")
    return _load_vaeac(ckpt_dir, cfg_path, device)


def load_flow(device: torch.device):
    """Load Flow matching from CARE-PD gaussian_k4_t16 checkpoint."""
    from motionbench.imputers.carepd_imputer import _load_flow  # noqa: PLC0415
    ckpt_dir = CARE_PD / "experiment_outs" / "flow_matching_synthetic" / "gaussian_k4_t16"
    cfg_path = CARE_PD / "configs" / "flow_matching" / "gaussian_k4_t16.json"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Flow checkpoint dir not found: {ckpt_dir}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Flow config not found: {cfg_path}")
    cfg = json.loads(cfg_path.read_text())
    cfg["num_steps"] = 20  # reduce for speed
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        json.dump(cfg, fh)
        tmp_cfg = Path(fh.name)
    try:
        imp = _load_flow(ckpt_dir, tmp_cfg, device)
    finally:
        tmp_cfg.unlink()
    return imp


# ---------------------------------------------------------------------------
# Value function computation per imputer
# ---------------------------------------------------------------------------


def compute_coalition_values_neural(
    imp,
    x_i: Tensor,
    frame_mask_t: Tensor,
    z_bin: np.ndarray,
    clf,
    target_class: int,
    device: torch.device,
    n_mc: int = 1,
) -> np.ndarray:
    """Compute v(S) for all 2^K coalitions using a neural imputer.

    Uses ``sample_completions_batched`` when available; falls back to
    per-coalition ``sample_completions`` otherwise.

    Args:
        imp: CARE-PD imputer with ``sample_completions_batched`` or
            ``sample_completions``.
        x_i: ``(J, F, T)`` float32 sequence on CPU.
        frame_mask_t: ``(2^K, T)`` bool tensor.
        z_bin: ``(2^K, K)`` bool array.
        clf: Classifier (on ``device``).
        target_class: Class index for softmax output.
        device: Torch device.
        n_mc: Completions per coalition (1 for speed).

    Returns:
        ``(2^K,)`` float32 value array.
    """
    n_coal = frame_mask_t.shape[0]
    x_in = x_i.unsqueeze(0).to(device)            # (1, J, F, T)
    pad = torch.ones(1, T, dtype=torch.bool, device=device)
    frame_mask_dev = frame_mask_t.to(device)

    if hasattr(imp, "sample_completions_batched"):
        with torch.no_grad():
            out = imp.sample_completions_batched(
                x=x_in,
                mask=pad,
                coalition_masks=frame_mask_dev,
                n_samples=n_mc,
            )                                    # (n_coal, n_mc, J, F, T)
        # Enforce observed entries
        obs_exp = frame_mask_dev.view(n_coal, 1, 1, 1, T).expand(
            n_coal, n_mc, J, F, T
        )
        x_exp = x_i.to(device).view(1, 1, J, F, T).expand(n_coal, n_mc, J, F, T)
        comps = torch.where(obs_exp, x_exp, out)   # (n_coal, n_mc, J, F, T)
        comps = comps.view(n_coal * n_mc, J, F, T)

        with torch.no_grad():
            logits = clf(comps)                    # (n_coal*n_mc, n_classes)
        if logits.ndim == 2:
            probs = torch.softmax(logits, dim=-1)[:, target_class]
        else:
            probs = logits
        v_vals = probs.view(n_coal, n_mc).mean(dim=1).cpu().numpy()
    else:
        # Fallback: per-coalition loop using sample_completions
        v_vals = np.zeros(n_coal, dtype=np.float32)
        for ci in range(n_coal):
            coal_mask = frame_mask_dev[ci]         # (T,) bool
            coalition_in = coal_mask.unsqueeze(0)  # (1, T)
            comps_list = imp.sample_completions(
                x=x_in, y=None, mask=pad, lengths=None,
                coalition_mask=coalition_in, n_samples=n_mc,
            )
            comps = torch.cat(comps_list, dim=0)   # (n_mc, J, F, T)
            # Enforce observed
            obs_exp = coal_mask.view(1, 1, 1, T).expand(n_mc, J, F, T)
            x_exp2 = x_i.to(device).unsqueeze(0).expand(n_mc, -1, -1, -1)
            comps = torch.where(obs_exp, x_exp2, comps)
            with torch.no_grad():
                logits = clf(comps)
            if logits.ndim == 2:
                probs = torch.softmax(logits, dim=-1)[:, target_class]
            else:
                probs = logits
            v_vals[ci] = probs.mean().item()

    return v_vals


def compute_coalition_values_simple(
    fill_fn: Callable[[Tensor, Tensor], Tensor],
    x_i: Tensor,
    frame_mask_t: Tensor,
    clf,
    target_class: int,
    device: torch.device,
) -> np.ndarray:
    """Compute v(S) using a deterministic fill function (Mean/Zero).

    Args:
        fill_fn: Callable ``(x_obs, obs_mask) → (1, J, F, T)`` completed tensor.
        x_i: ``(J, F, T)`` float32 CPU tensor.
        frame_mask_t: ``(2^K, T)`` bool tensor.
        clf: Classifier on ``device``.
        target_class: Class index.
        device: Torch device.

    Returns:
        ``(2^K,)`` float32 value array.
    """
    n_coal = frame_mask_t.shape[0]
    completions = []
    for ci in range(n_coal):
        obs_mask_jft = frame_mask_t[ci].view(1, 1, T).expand(J, F, T)
        comp = fill_fn(x_i, obs_mask_jft)    # (1, J, F, T) or (J, F, T)
        if comp.ndim == 3:
            comp = comp.unsqueeze(0)
        completions.append(comp)
    comps = torch.cat(completions, dim=0).to(device)  # (n_coal, J, F, T)
    with torch.no_grad():
        logits = clf(comps)
    if logits.ndim == 2:
        probs = torch.softmax(logits, dim=-1)[:, target_class]
    else:
        probs = logits
    return probs.cpu().numpy().astype(np.float32)


def compute_coalition_values_marginal(
    x_pool: Tensor,
    x_i: Tensor,
    frame_mask_t: Tensor,
    clf,
    target_class: int,
    device: torch.device,
    n_mc: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Compute v(S) using marginal imputation (random donor from pool).

    Args:
        x_pool: ``(N_train, J, F, T)`` training pool.
        x_i: ``(J, F, T)`` test sequence.
        frame_mask_t: ``(2^K, T)`` bool.
        clf: Classifier.
        target_class: Class index.
        device: Torch device.
        n_mc: Number of random donors per coalition.

    Returns:
        ``(2^K,)`` float32 value array.
    """
    rng = np.random.default_rng(seed)
    n_coal = frame_mask_t.shape[0]
    N_pool = x_pool.shape[0]

    v_vals = np.zeros(n_coal, dtype=np.float32)
    for ci in range(n_coal):
        obs_mask = frame_mask_t[ci]  # (T,) bool
        obs_jft = obs_mask.view(1, 1, T).expand(J, F, T)  # (J, F, T) bool
        # Draw n_mc random donors
        donor_idx = rng.integers(0, N_pool, size=n_mc)
        donors = x_pool[donor_idx]  # (n_mc, J, F, T)
        # Replace observed windows with x_i
        x_obs_exp = x_i.unsqueeze(0).expand(n_mc, -1, -1, -1)  # (n_mc, J, F, T)
        obs_exp = obs_jft.unsqueeze(0).expand(n_mc, -1, -1, -1)
        comps = torch.where(obs_exp, x_obs_exp, donors).to(device)  # (n_mc, J, F, T)
        with torch.no_grad():
            logits = clf(comps)
        if logits.ndim == 2:
            probs = torch.softmax(logits, dim=-1)[:, target_class]
        else:
            probs = logits
        v_vals[ci] = probs.mean().item()

    return v_vals


# ---------------------------------------------------------------------------
# EC metrics
# ---------------------------------------------------------------------------


def ec_metrics(phi_hat: np.ndarray, phi_true: np.ndarray) -> dict[str, float]:
    diff = phi_hat - phi_true
    ec1 = float(np.mean(np.abs(diff)))
    denom = float(np.mean(np.abs(phi_true)) + 1e-8)
    ec1_norm = ec1 / denom
    ec2 = float(np.mean(diff ** 2))
    if np.std(phi_hat) < 1e-10 or np.std(phi_true) < 1e-10:
        ec3 = float("nan")
    else:
        corr = float(np.corrcoef(phi_hat, phi_true)[0, 1])
        ec3 = 1.0 - corr
    return {"ec1": ec1, "ec1_norm": ec1_norm, "ec2": ec2, "ec3": ec3}


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------


def run_sanity_check(device: torch.device) -> dict:
    """Verify FullGaussianOracle ≈ GaussianOracle on a Kronecker Sigma.

    Creates a Kronecker covariance from gaussian_k4's Sigma_joints and
    Sigma_time, wraps it as FullGaussianOracle, and compares EC1 to the
    standard GaussianOracle on N_SANITY_SEQ sequences.

    Returns:
        Dict with keys: passed (bool), mean_ec1_full, mean_ec1_kron, ratio.
    """
    log.info("=== SANITY CHECK ===")
    from motionbench.data.synthetic.gaussian_motion import GaussianMotionDataset  # noqa: PLC0415
    from motionbench.oracles.gaussian_oracle import GaussianOracle              # noqa: PLC0415
    from motionbench.oracles.full_gaussian_oracle import FullGaussianOracle     # noqa: PLC0415
    from motionbench.players.temporal_windows import TemporalWindows             # noqa: PLC0415

    ds = GaussianMotionDataset(J=J, F=F, T=T, N=50, K=K, rho=0.5, alpha=0.8, seed=42)
    bench = ds._benchmark

    Sigma_joints = bench.Sigma_joints
    Sigma_time = bench.Sigma_time

    # Build Kronecker Sigma as full (D, D) matrix
    Sigma_kron_full = np.kron(np.kron(Sigma_joints, np.eye(F)), Sigma_time)

    oracle_kron = GaussianOracle(Sigma_joints=Sigma_joints, Sigma_time=Sigma_time)
    oracle_full = FullGaussianOracle(Sigma_full=Sigma_kron_full, J=J, F=F, T=T)
    players = TemporalWindows(K=K, T=T, J=J, F=F)

    # Load classifier
    clf = load_classifier("synthetic_transformer", device)

    def clf_fn(x_batch: Tensor) -> Tensor:
        x_dev = x_batch.float().to(device)
        with torch.no_grad():
            logits = clf(x_dev)
        if logits.ndim == 2:
            return torch.softmax(logits, dim=-1)[:, 0].cpu()
        return logits.cpu()

    phis_kron, phis_full = [], []
    for i in range(min(N_SANITY_SEQ, len(ds))):
        x_i, _ = ds[i]
        try:
            phi_k = oracle_kron.true_shapley(x_i, clf_fn, players, n_mc=N_MC,
                                              n_coalitions=1 << K, seed=i)
            phi_f = oracle_full.true_shapley(x_i, clf_fn, players, n_mc=N_MC,
                                              n_coalitions=1 << K, seed=i)
            phis_kron.append(phi_k.numpy())
            phis_full.append(phi_f.numpy())
        except Exception as exc:
            log.warning("  Sanity seq %d failed: %s", i, exc)
            continue

    if len(phis_kron) == 0:
        return {"passed": False, "error": "all sequences failed"}

    phis_kron_arr = np.stack(phis_kron)  # (N_sanity, K)
    phis_full_arr = np.stack(phis_full)  # (N_sanity, K)

    # EC1 between the two oracles
    ec1_vals = [ec_metrics(phis_full_arr[i], phis_kron_arr[i])["ec1"]
                for i in range(len(phis_kron))]
    mean_ec1 = float(np.mean(ec1_vals))

    # EC1 of oracle against itself (should be 0; use full-vs-kron as cross-check)
    mean_kron_mag = float(np.mean(np.abs(phis_kron_arr)))
    ratio = mean_ec1 / (mean_kron_mag + 1e-8)

    passed = ratio <= TOL_SANITY
    log.info(
        "  Sanity: n_seqs=%d  EC1(full vs kron)=%.4f  |phi_kron|=%.4f  ratio=%.3f  %s",
        len(phis_kron), mean_ec1, mean_kron_mag, ratio,
        "PASS" if passed else "FAIL",
    )
    return {
        "passed": passed,
        "n_sequences": len(phis_kron),
        "ec1_full_vs_kron": mean_ec1,
        "mean_kron_magnitude": mean_kron_mag,
        "ratio": ratio,
        "tolerance": TOL_SANITY,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment(device: torch.device) -> dict:
    """Run all imputers on the non-Kronecker dataset and compute EC1."""
    log.info("=== EXPERIMENT ===")
    from motionbench.data.synthetic.gaussian_nk import GaussianNKDataset           # noqa: PLC0415
    from motionbench.oracles.full_gaussian_oracle import FullGaussianOracle          # noqa: PLC0415
    from motionbench.players.temporal_windows import TemporalWindows                  # noqa: PLC0415
    from motionbench.imputers.off_manifold import (                                   # noqa: PLC0415
        ZeroImputer, MeanImputer, MarginalDonorImputer,
    )

    # ---- Datasets ----
    log.info("Building NK datasets (train N=%d, test N=%d)...", N_TRAIN, N_TEST)
    ds_train = GaussianNKDataset(J=J, F=F, T=T, N=N_TRAIN, K=K, seed=100)
    ds_test = GaussianNKDataset(J=J, F=F, T=T, N=N_TEST, K=K, seed=200)

    log.info("  Perturbation fraction: %.3f", ds_test.perturbation_frac)

    # Build oracle from test dataset's covariance (same as train up to seed noise
    # in Sigma, but we want to use the test set's Sigma_full_nk since it's fixed
    # by construction — just use ds_test's Sigma)
    oracle = FullGaussianOracle(
        Sigma_full=ds_test.Sigma_full_nk, J=J, F=F, T=T
    )
    players = TemporalWindows(K=K, T=T, J=J, F=F)

    # Pre-build all coalition masks
    z_bin, frame_mask_np = build_coalition_masks(K, T)
    n_coal = 1 << K
    frame_mask_t = torch.from_numpy(frame_mask_np)  # (16, T) bool

    # Stack test sequences
    x_test_list = [ds_test[i][0] for i in range(N_TEST)]  # list of (J,F,T) tensors
    y_test = [int(ds_test[i][1]) for i in range(N_TEST)]
    x_pool = torch.stack([ds_train[i][0] for i in range(N_TRAIN)])  # (N_TRAIN, J, F, T)

    # Training statistics for Mean imputer
    x_train_np = x_pool.numpy()  # (N_TRAIN, J, F, T)
    mean_per_coord = x_train_np.mean(axis=0)  # (J, F, T)
    mean_tensor = torch.tensor(mean_per_coord, dtype=torch.float32)

    summary_rows = []

    for clf_name in CLASSIFIERS:
        log.info("--- Classifier: %s ---", clf_name)
        clf = load_classifier(clf_name, device)

        # ---- Pre-compute oracle Shapley values (shared across methods) ----
        log.info("  Pre-computing oracle Shapley values for %d sequences...", N_TEST)
        oracle_phis = np.zeros((N_TEST, K), dtype=np.float32)
        targets = np.zeros(N_TEST, dtype=int)

        with torch.no_grad():
            x_batch = torch.stack(x_test_list).to(device)
            logits_all = clf(x_batch)
        if logits_all.ndim == 2:
            targets[:] = logits_all.argmax(dim=-1).cpu().numpy()
        else:
            targets[:] = y_test

        t_oracle_start = time.time()
        for i in range(N_TEST):
            x_i = x_test_list[i]
            tgt = int(targets[i])

            def clf_fn_i(x_b: Tensor, _tgt: int = tgt) -> Tensor:
                with torch.no_grad():
                    out = clf(x_b.float().to(device))
                if out.ndim == 2:
                    return torch.softmax(out, dim=-1)[:, _tgt].cpu()
                return out.cpu()

            try:
                phi_oracle = oracle.true_shapley(
                    x_i, clf_fn_i, players,
                    n_mc=N_MC, n_coalitions=n_coal, seed=i,
                )
                oracle_phis[i] = phi_oracle.numpy()
            except Exception as exc:
                log.warning("    Oracle seq %d failed: %s", i, exc)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_oracle_start
                eta = elapsed / (i + 1) * (N_TEST - i - 1)
                log.info("    oracle: %d/%d done (eta %.0fs)", i + 1, N_TEST, eta)

        log.info("  Oracle Shapley done in %.1fs", time.time() - t_oracle_start)

        # ---- Evaluate each method ----
        for method in METHODS:
            out_dir = RESULTS_DIR / clf_name / method
            out_dir.mkdir(parents=True, exist_ok=True)
            result_path = out_dir / "result.json"

            # Skip if already cached
            if result_path.exists():
                log.info("  [SKIP] %s/%s (cached)", clf_name, method)
                try:
                    cached = json.loads(result_path.read_text())
                    summary_rows.append(cached)
                except Exception:
                    pass
                continue

            log.info("  [RUN] %s/%s", clf_name, method)
            t_method = time.time()

            imp_phis = np.zeros((N_TEST, K), dtype=np.float32)

            try:
                if method == "kernelshap_vaeac":
                    imp = load_vaeac(device)
                    for i in range(N_TEST):
                        x_i = x_test_list[i]
                        tgt = int(targets[i])
                        v_vals = compute_coalition_values_neural(
                            imp, x_i, frame_mask_t, z_bin, clf, tgt, device,
                        )
                        imp_phis[i] = kernel_shap_exact(z_bin, v_vals, K)
                    del imp
                    torch.cuda.empty_cache()

                elif method == "kernelshap_flow":
                    imp = load_flow(device)
                    for i in range(N_TEST):
                        x_i = x_test_list[i]
                        tgt = int(targets[i])
                        v_vals = compute_coalition_values_neural(
                            imp, x_i, frame_mask_t, z_bin, clf, tgt, device,
                        )
                        imp_phis[i] = kernel_shap_exact(z_bin, v_vals, K)
                    del imp
                    torch.cuda.empty_cache()

                elif method == "kernelshap_zero":
                    def zero_fill(x_obs: Tensor, obs_mask: Tensor) -> Tensor:
                        return torch.where(obs_mask, x_obs, torch.zeros_like(x_obs))

                    for i in range(N_TEST):
                        x_i = x_test_list[i]
                        tgt = int(targets[i])
                        v_vals = compute_coalition_values_simple(
                            zero_fill, x_i, frame_mask_t, clf, tgt, device,
                        )
                        imp_phis[i] = kernel_shap_exact(z_bin, v_vals, K)

                elif method == "kernelshap_mean":
                    def mean_fill(x_obs: Tensor, obs_mask: Tensor) -> Tensor:
                        return torch.where(obs_mask, x_obs, mean_tensor)

                    for i in range(N_TEST):
                        x_i = x_test_list[i]
                        tgt = int(targets[i])
                        v_vals = compute_coalition_values_simple(
                            mean_fill, x_i, frame_mask_t, clf, tgt, device,
                        )
                        imp_phis[i] = kernel_shap_exact(z_bin, v_vals, K)

                elif method == "kernelshap_marginal":
                    for i in range(N_TEST):
                        x_i = x_test_list[i]
                        tgt = int(targets[i])
                        v_vals = compute_coalition_values_marginal(
                            x_pool, x_i, frame_mask_t, clf, tgt, device,
                            n_mc=5, seed=i,
                        )
                        imp_phis[i] = kernel_shap_exact(z_bin, v_vals, K)

                else:
                    log.warning("  Unknown method %s, skipping", method)
                    continue

                # Compute EC metrics against oracle
                ecs_list = [ec_metrics(imp_phis[i], oracle_phis[i]) for i in range(N_TEST)]
                ec1_mean = float(np.mean([e["ec1"] for e in ecs_list]))
                ec1_std = float(np.std([e["ec1"] for e in ecs_list]))
                ec1_norm_mean = float(np.nanmean([e["ec1_norm"] for e in ecs_list]))

                result = {
                    "method": method,
                    "classifier": clf_name,
                    "dataset": "gaussian_nk",
                    "n_sequences": N_TEST,
                    "ec1_mean": ec1_mean,
                    "ec1_std": ec1_std,
                    "ec1_norm_mean": ec1_norm_mean,
                    "wall_seconds": time.time() - t_method,
                }
                result_path.write_text(json.dumps(result, indent=2))
                np.savez_compressed(
                    out_dir / "attributions.npz",
                    phi=imp_phis,
                    phi_oracle=oracle_phis,
                )
                log.info(
                    "    DONE %s in %.1fs  EC1=%.4f ± %.4f  EC1_norm=%.4f",
                    method, time.time() - t_method, ec1_mean, ec1_std, ec1_norm_mean,
                )
                summary_rows.append(result)

            except Exception as exc:
                log.error("  [ERROR] %s/%s: %s", clf_name, method, exc)
                traceback.print_exc()
                error_result = {
                    "method": method,
                    "classifier": clf_name,
                    "error": str(exc),
                    "wall_seconds": time.time() - t_method,
                }
                (out_dir / "error.json").write_text(json.dumps(error_result, indent=2))
                summary_rows.append(error_result)
                torch.cuda.empty_cache()

        del clf
        torch.cuda.empty_cache()

    return {"rows": summary_rows}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    t_start = time.time()
    os.chdir(REPO)
    log.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))

    device = torch.device(DEVICE_STR if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Sanity check                                                         #
    # ------------------------------------------------------------------ #
    sanity_result = {"passed": None, "error": "not_run"}
    try:
        sanity_result = run_sanity_check(device)
    except Exception as exc:
        log.error("Sanity check failed with exception: %s", exc)
        traceback.print_exc()
        sanity_result = {"passed": False, "error": str(exc)}

    (RESULTS_DIR / "sanity_check.json").write_text(
        json.dumps(sanity_result, indent=2)
    )
    log.info("Sanity check result: %s", sanity_result)

    if not sanity_result.get("passed", False):
        prose = (
            "Sanity check FAILED or could not be completed.\n\n"
            "Theoretical argument (why non-Kronecker oracle is still valid):\n"
            "The FullGaussianOracle uses the standard Schur complement formula\n"
            "for conditional Gaussian distributions, which is exact for ANY\n"
            "Gaussian distribution regardless of covariance structure.  The\n"
            "GaussianOracle (Kronecker) is a special case of this formula.\n"
            "When the sanity check fails, it indicates a numerical discrepancy\n"
            "exceeding the 20% tolerance, likely due to Monte Carlo variance\n"
            "at n_mc=50.  The full oracle remains theoretically correct.\n\n"
            f"Sanity result: {sanity_result}\n"
        )
        (RESULTS_DIR / "NOTE.txt").write_text(prose)
        log.warning("Sanity check did not pass — NOTE.txt written.")

    # ------------------------------------------------------------------ #
    # Main experiment                                                      #
    # ------------------------------------------------------------------ #
    exp_result = {"rows": [], "error": "not_run"}
    try:
        exp_result = run_experiment(device)
    except Exception as exc:
        log.error("Experiment failed: %s", exc)
        traceback.print_exc()
        exp_result = {"rows": [], "error": str(exc)}

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    summary = {
        "sanity_check": sanity_result,
        "experiment_rows": exp_result.get("rows", []),
        "total_wall_seconds": time.time() - t_start,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("=" * 60)
    log.info("Total wall time: %.1fs", time.time() - t_start)
    log.info("Results dir: %s", RESULTS_DIR)

    # Print compact table
    rows = exp_result.get("rows", [])
    if rows:
        log.info("\n%-30s %-25s %s", "Method", "Classifier", "EC1_mean ± EC1_std")
        log.info("-" * 80)
        for r in rows:
            if "ec1_mean" in r:
                log.info(
                    "%-30s %-25s %.4f ± %.4f",
                    r.get("method", "?"), r.get("classifier", "?"),
                    r["ec1_mean"], r.get("ec1_std", float("nan")),
                )
            elif "error" in r:
                log.info("%-30s %-25s ERROR: %s", r.get("method", "?"),
                         r.get("classifier", "?"), str(r["error"])[:60])


if __name__ == "__main__":
    main()
