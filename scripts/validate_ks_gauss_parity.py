"""scripts/validate_ks_gauss_parity.py — KS-Gauss parity gate vs the validation study.

Reproduces the independent validation study's ``ks_shapr`` cells (the paper's
KS-Gauss rows) end-to-end with **release components only** — dataset,
player sets, shared coalition design, ShaprGaussianImputer, deterministic
conditional targets, constrained WLS solve — and compares the per-sequence
Shapley vectors ``phi`` (and ``phi_star``, EC1, EC3) against the study's
stored ``per_sequence.npz`` files.

RNG protocol replicated exactly: the study threads ONE numpy generator,
``default_rng([seed, sequence_idx])``, through the coalition rows of each
sequence in design order (full-coalition rows consume no draws; the empty
row draws unconditionally).  The release imputer supports this via its
keyword-only ``generator`` argument, so the fills here are bit-identical to
the study's and the residual ``|delta phi|`` measures only classifier
forward-pass differences (e.g. CPU vs GPU kernels).

Default cells: (joint, cell) player sets x gauss_k4 x MLP — the study's
``results/{joint,cell}/gauss_k4/mlp/ks_shapr/``.  The defaults point at the
internal artifact locations; external users need the study's result archive
and classifier checkpoints to run this gate.

Usage::

    python scripts/validate_ks_gauss_parity.py \\
        --ground-truth ~/experiments/exp10_players/results \\
        --ckpt-root /mnt/delicate-frog/artifacts/silico/experiments/_flat/\\
exp_01kzq2axgmfjna5vwmbg8frjvn/checkpoints
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from motionbench.attribution.sampled_coalitions import (  # noqa: E402
    phi_from_values,
    sampled_coalition_set,
)
from motionbench.classifiers.synthetic_mlp import SyntheticMLPClassifier  # noqa: E402
from motionbench.data.synthetic.gaussian_motion import GaussianMotionDataset  # noqa: E402
from motionbench.imputers.shapr_gaussian import ShaprGaussianImputer  # noqa: E402
from motionbench.oracles.deterministic import DeterministicConditionalOracle  # noqa: E402
from motionbench.players.joint_window_cells import JointWindowCells  # noqa: E402
from motionbench.players.spatial_joints import SpatialJoints  # noqa: E402

# gauss_k4 field (configs/data/gaussian_k4.yaml) and the study's protocol pins.
J, F, T, K = 5, 3, 16, 4
N_EVAL, EVAL_SEED = 200, 42
N_FIT, FIT_SEED = 1000, 99  # imputer-training pool (RESOLUTIONS.md section 8)
N_COMPLETION = 5
COALITION_SEED = 7919
BUDGET = 1024


def batched_prob_fn(model, target, device, batch=1024):
    """(n, J, F, T) float32 -> (n,) float64 softmax prob, as in the study."""

    def fn(arr):
        vals = []
        with torch.no_grad():
            for s in range(0, len(arr), batch):
                xb = torch.from_numpy(
                    np.ascontiguousarray(arr[s : s + batch], dtype=np.float32)
                ).to(device)
                vals.append(torch.softmax(model(xb), -1)[:, target].float().cpu().numpy())
        return np.concatenate(vals).astype(np.float64)

    return fn


def study_ec3(phi, phi_star):
    """EC3 with the validation study's clip convention (clip(1 - r, -1, 1))."""
    if np.std(phi) < 1e-10 or np.std(phi_star) < 1e-10:
        return 1.0
    return float(np.clip(1.0 - np.corrcoef(phi, phi_star)[0, 1], -1, 1))


def make_players(playerset):
    """The study's player sets: 'joint' (M = J) and 'cell' (M = J*K)."""
    if playerset == "joint":
        return SpatialJoints(J=J, F=F, T=T)
    if playerset == "cell":
        return JointWindowCells(J=J, K=K, F=F, T=T)
    raise ValueError(playerset)


def run_cell(playerset, dataset, imputer, model, device, n_seq):
    """Recompute one ks_shapr cell; returns (phi, phi_star, ec1, ec3, target) arrays."""
    players = make_players(playerset)
    M = players.n_players
    Z, w = sampled_coalition_set(M, BUDGET, seed=COALITION_SEED)
    masks = [
        players.coalition_mask(torch.as_tensor(z != 0, dtype=torch.bool).clone())
        for z in np.asarray(Z, dtype=np.int64)
    ]
    det = DeterministicConditionalOracle.from_oracle(dataset.oracle, players, Z)

    phis, stars, ec1s, ec3s, targets = [], [], [], [], []
    t0 = time.time()
    for idx in range(n_seq):
        x32_t, _ = dataset[idx]  # (J, F, T) float32 tensor
        x32 = x32_t.numpy()
        x64 = x32.astype(np.float64)
        with torch.no_grad():
            logits = model(x32_t[None].to(device))
        target = int(logits.argmax(dim=-1).item())
        clf_fn = batched_prob_fn(model, target, device)

        # The study's per-sequence stream: ONE generator threaded through the
        # coalition rows in design order (exp10/run_player_cell.method_fills).
        rng = np.random.default_rng([EVAL_SEED, idx])
        fills = np.empty((len(Z), J, F, T), dtype=np.float32)
        for i, z in enumerate(Z):
            zb = z.astype(bool)
            if zb.all():
                fills[i] = x32
            else:
                comps = imputer.impute(x32_t, masks[i], N_COMPLETION, generator=rng)
                fills[i] = comps.numpy().mean(axis=0)
        v = clf_fn(fills)
        phi = phi_from_values(Z, w, v)

        v_star = clf_fn(det.fill_all(x64))
        phi_star = phi_from_values(Z, w, v_star)

        phis.append(phi)
        stars.append(phi_star)
        ec1s.append(float(np.mean(np.abs(phi - phi_star))))
        ec3s.append(study_ec3(phi, phi_star))
        targets.append(target)
        if idx % 50 == 0:
            print(f"  [{playerset}] {idx}/{n_seq} ({time.time() - t0:.0f}s)", flush=True)
    return (
        np.array(phis),
        np.array(stars),
        np.array(ec1s),
        np.array(ec3s),
        np.array(targets),
    )


def hi_budget_regrade(playerset, dataset, model, device, phi, n_seq, gt_dir):
    """Reproduce the study's hi-budget target regrade for a sampled cell.

    The study regraded every sampled-design cell (M > 12) against a
    deterministic conditional target on an independent B=8192, seed=900001
    design (``exp10/regrade_targets.py``); ``analysis_final.json`` reports
    those EC values.  This recomputes the hi target with release components,
    compares it to the stored ``hi_targets.npz``, and recomputes the regraded
    EC1/EC3 of the gate's ``phi`` against the study's ``regrade_hi.json``.
    """
    players = make_players(playerset)
    Z, w = sampled_coalition_set(players.n_players, 8192, seed=900001)
    det = DeterministicConditionalOracle.from_oracle(dataset.oracle, players, Z)
    stars = []
    t0 = time.time()
    for idx in range(n_seq):
        x32_t, _ = dataset[idx]
        x64 = x32_t.numpy().astype(np.float64)
        with torch.no_grad():
            target = int(model(x32_t[None].to(device)).argmax(dim=-1).item())
        clf_fn = batched_prob_fn(model, target, device, batch=2048)
        stars.append(phi_from_values(Z, w, clf_fn(det.fill_all(x64))))
        if idx % 50 == 0:
            print(f"  [{playerset} hi-regrade] {idx}/{n_seq} ({time.time() - t0:.0f}s)", flush=True)
    star_hi = np.array(stars)

    gt_star = np.load(gt_dir / "hi_targets.npz")["star_cond"][:n_seq]
    gt_ec = json.loads((gt_dir / "ks_shapr" / "regrade_hi.json").read_text())
    ec1 = float(np.mean(np.abs(phi - star_hi)))
    ec3 = float(np.mean([study_ec3(phi[i], star_hi[i]) for i in range(n_seq)]))
    return {
        "max_abs_dstar_hi": float(np.max(np.abs(star_hi - gt_star))),
        "regraded_ec1": ec1,
        "regraded_ec3": ec3,
        "abs_dec1_vs_study": float(abs(ec1 - gt_ec["ec1"])),
        "abs_dec3_vs_study": float(abs(ec3 - gt_ec["ec3"])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ground-truth",
        default="~/experiments/exp10_players/results",
        help="Validation-study results root (contains {joint,cell}/gauss_k4/mlp/ks_shapr/)",
    )
    ap.add_argument(
        "--ckpt-root",
        default=(
            "/mnt/delicate-frog/artifacts/silico/experiments/_flat/"
            "exp_01kzq2axgmfjna5vwmbg8frjvn/checkpoints"
        ),
        help="Study checkpoint root (contains classifiers/gauss_k4/mlp.pt)",
    )
    ap.add_argument("--playersets", nargs="+", default=["joint", "cell"])
    ap.add_argument("--n-sequences", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tol", type=float, default=1e-5, help="Gate threshold on max |delta phi|")
    ap.add_argument(
        "--skip-hi-regrade",
        action="store_true",
        help="Skip reproducing the study's B=8192 target regrade for sampled cells",
    )
    args = ap.parse_args()

    gt_root = Path(args.ground_truth).expanduser()
    device = torch.device(args.device)

    dataset = GaussianMotionDataset(
        J=J, F=F, T=T, K=K, N=N_EVAL, rho=0.5, alpha=0.8, seed=EVAL_SEED
    )
    fit_dataset = GaussianMotionDataset(
        J=J, F=F, T=T, K=K, N=N_FIT, rho=0.5, alpha=0.8, seed=FIT_SEED
    )
    imputer = ShaprGaussianImputer(shrinkage="ledoit_wolf").fit(fit_dataset)
    print(
        f"Fitted ShaprGaussianImputer on N={N_FIT} @ seed {FIT_SEED} "
        f"(Ledoit-Wolf shrinkage {imputer.shrinkage_coef:.6f})"
    )

    ckpt_path = Path(args.ckpt_root) / "classifiers" / "gauss_k4" / "mlp.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = SyntheticMLPClassifier(
        J=J, F=F, T=T, K=K, n_classes=3, hidden=64, player_mode="temporal"
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    all_pass = True
    report = {}
    for playerset in args.playersets:
        gt = np.load(gt_root / playerset / "gauss_k4" / "mlp" / "ks_shapr" / "per_sequence.npz")
        n_seq = min(args.n_sequences, len(gt["phi"]))
        phi, star, ec1, ec3, target = run_cell(playerset, dataset, imputer, model, device, n_seq)

        d_phi = float(np.max(np.abs(phi - gt["phi"][:n_seq])))
        d_star = float(np.max(np.abs(star - gt["phi_star"][:n_seq])))
        d_ec1 = float(np.max(np.abs(ec1 - gt["ec1"][:n_seq])))
        d_ec3 = float(np.max(np.abs(ec3 - gt["ec3"][:n_seq])))
        d_mean_ec1 = float(abs(np.mean(ec1) - np.mean(gt["ec1"][:n_seq])))
        n_tgt = int(np.sum(target != gt["target"][:n_seq]))
        ok = d_phi < args.tol and n_tgt == 0
        report[playerset] = {
            "n_sequences": n_seq,
            "max_abs_dphi": d_phi,
            "max_abs_dphi_star": d_star,
            "max_abs_dec1": d_ec1,
            "max_abs_dec3": d_ec3,
            "abs_dmean_ec1": d_mean_ec1,
            "target_mismatches": n_tgt,
            "mean_ec1": float(np.mean(ec1)),
            "mean_ec3": float(np.mean(ec3)),
        }
        print(
            f"[{playerset}/gauss_k4/mlp] max|dphi|={d_phi:.3e}  "
            f"max|dphi*|={d_star:.3e}  max|dEC1|={d_ec1:.3e}  "
            f"max|dEC3|={d_ec3:.3e}  target mismatches={n_tgt}  "
            f"-> {'PASS' if ok else 'FAIL'}"
        )

        # Sampled-design cells were regraded against a hi-budget target for
        # the paper tables; reproduce that regrade too when the study's
        # hi_targets.npz is available.
        gt_dir = gt_root / playerset / "gauss_k4" / "mlp"
        M = make_players(playerset).n_players
        if not args.skip_hi_regrade and M > 12 and (gt_dir / "hi_targets.npz").exists():
            hi = hi_budget_regrade(playerset, dataset, model, device, phi, n_seq, gt_dir)
            ok &= hi["max_abs_dstar_hi"] < args.tol
            report[playerset]["hi_regrade"] = hi
            print(
                f"[{playerset}/gauss_k4/mlp hi-regrade] "
                f"max|dstar_hi|={hi['max_abs_dstar_hi']:.3e}  "
                f"EC1={hi['regraded_ec1']:.6f} (study d={hi['abs_dec1_vs_study']:.3e})  "
                f"EC3={hi['regraded_ec3']:.6f} (study d={hi['abs_dec3_vs_study']:.3e})"
            )
        report[playerset]["pass"] = ok
        all_pass &= ok

    print(json.dumps(report, indent=1))
    if not all_pass:
        sys.exit(1)
    print(f"PARITY GATE PASSED (tol {args.tol:g})")


if __name__ == "__main__":
    main()
