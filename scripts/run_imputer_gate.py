"""scripts/run_imputer_gate.py — Imputer capability gate: hide-one-recover on real tracks.

For each real dataset × learned imputer, measure how well the imputer
reconstructs a fully hidden player from the visible rest, under the three
player-set mask families the benchmark evaluates:

  spatial  : one spatial unit (joint / lead / mel-band quartile), all frames
  temporal : one of K=4 uniform windows, all units
  cell     : one (unit, window) cell (12 sampled per sequence, seeded)

Metric: Pearson correlation between the imputed conditional mean (average of
8 completions) and the ground truth over the hidden entries, per
(sequence, mask), averaged; reported with the spread across masks.

Scope (the canonical protocol behind ``results/canonical/imputer_gate.json``):
all 3 folds, the first 24 sequences of each fold's fingerprinted eval
selection (the per-track sweeps' 200-sequence selection, truncated), imputer
checkpoints identical to the ones behind the paper's tables.  Mask RNG:
cells sampled with ``default_rng([2202, fold])``; completion draws with
``default_rng([2203, fold, seq])`` per (family, fold, sequence), consumed
across that family's masks in order.

``--mode controls`` runs the two capability controls instead:

  unconditional : imputer recovery with *everything* hidden
  donor         : copy the hidden entries from the track's marginal donor
                  sequence — the recovery raw data redundancy alone provides
                  (imputer-independent; identical for vaeac/flow by design)

Data and checkpoints resolve exactly as in the per-track sweeps
(``run_carepd_players_shap.py`` / ``run_esc50_cells_shap.py`` /
``run_ptbxl_cells_shap.py``), whose loaders this script reuses.

Usage::

    PYTHONPATH=. python scripts/run_imputer_gate.py \\
        --mode gate --track ptbxl --imputer flow

Results: ``results/imputer_gate/{gate,controls}_{track}_{imputer}.json``
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_carepd_players_shap import load_carepd_data  # noqa: E402
from run_esc50_cells_shap import load_esc50_data  # noqa: E402
from run_ptbxl_cells_shap import load_ptbxl_data  # noqa: E402

from motionbench.imputers.frame_flow import FrameFlowImputer  # noqa: E402
from motionbench.imputers.frame_vaeac import FrameVAEACImputer  # noqa: E402

RESULTS_ROOT = REPO_ROOT / "results" / "imputer_gate"
IMPUTER_DIR = REPO_ROOT / "checkpoints" / "imputers"

N_SEQ = 24  # sequences per fold (first of the sweeps' fingerprinted selection)
N_DRAWS = 8  # completions averaged per (sequence, mask)
K = 4  # uniform temporal windows
CELL_SAMPLE = 12  # cells per sequence (sampled, seeded) to bound runtime
SELECTION_N = 200  # the per-track sweeps' eval-selection size (then truncated)
FAMILIES = ("spatial", "temporal", "cell")


def spatial_units(track, J):
    """Spatial player units per track: singleton joints/leads, or mel quartiles.

    Args:
        track: ``"carepd"`` / ``"esc50"`` / ``"ptbxl"``.
        J: Number of spatial units (joints / mel bins / leads).

    Returns:
        List of index lists, one per spatial unit.
    """
    if track == "esc50":  # 4 contiguous mel-bin quartiles (release partition)
        q = J // 4
        return [list(range(b * q, (b + 1) * q)) for b in range(4)]
    return [[j] for j in range(J)]


def masks_for_track(track, shape, rng):
    """Build the three mask families for one fold (True = observed).

    Args:
        track: Track name (drives the spatial-unit partition).
        shape: ``(J, F, T)`` sequence shape.
        rng: Numpy Generator for the sampled cell subset
            (``default_rng([2202, fold])`` in the canonical protocol).

    Returns:
        Dict ``{family: [masks]}`` in family order spatial, temporal, cell.
    """
    J, F, T = shape
    W = T // K
    fams = {"spatial": [], "temporal": [], "cell": []}
    for units in spatial_units(track, J):
        m = np.ones((J, F, T), bool)
        m[units, :, :] = False
        fams["spatial"].append(m)
    for w in range(K):
        m = np.ones((J, F, T), bool)
        m[:, :, w * W : (w + 1) * W] = False
        fams["temporal"].append(m)
    su = spatial_units(track, J)
    pairs = [(u, w) for u in range(len(su)) for w in range(K)]
    idx = rng.choice(len(pairs), size=min(CELL_SAMPLE, len(pairs)), replace=False)
    for i in idx:
        u, w = pairs[i]
        m = np.ones((J, F, T), bool)
        m[np.ix_(su[u], range(F), range(w * W, (w + 1) * W))] = False
        fams["cell"].append(m)
    return fams


def recovery(x, imp, mask, rng):
    """Pearson recovery of the hidden entries from the mean of N_DRAWS completions.

    Args:
        x: ``(J, F, T)`` sequence.
        imp: Imputer with the ``impute(x, mask, n, rng)`` protocol.
        mask: ``(J, F, T)`` bool, True = observed.
        rng: Numpy Generator (one torch seed drawn per impute call).

    Returns:
        Pearson correlation over the hidden entries (0.0 if degenerate).
    """
    draws = imp.impute(x, mask, N_DRAWS, rng)  # (n, J, F, T)
    est = np.asarray(draws, np.float64).mean(axis=0)
    hid = ~mask
    a, b = est[hid], np.asarray(x, np.float64)[hid]
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def load_eval(track, fold, args):
    """Load one fold's eval sequences and donors via the sweep loaders.

    The loaders are called at the sweeps' selection size (200) and truncated
    to the first ``N_SEQ`` sequences, so the gate sees exactly the head of
    each fingerprinted eval selection.

    Args:
        track: Track name.
        fold: Data fold (1-3).
        args: Parsed CLI arguments (cache/data locations).

    Returns:
        ``(x_eval (N_SEQ, J, F, T) float32, donors (N_SEQ, J, F, T))``.
    """
    if track == "carepd":
        x, _mean_jf, donors = load_carepd_data(fold, SELECTION_N, args.cache_dir)
    elif track == "esc50":
        x, _mean_jf, donors = load_esc50_data(fold, SELECTION_N, Path(args.data_dir))
    elif track == "ptbxl":
        x, _mean_jf, donors = load_ptbxl_data(fold, SELECTION_N, args.cache_dir, args.data_path)
    else:
        raise ValueError(track)
    return x[:N_SEQ], donors[:N_SEQ]


def run_gate(track, imp, args):
    """Run the hide-one-recover gate over all folds and mask families.

    Args:
        track: Track name.
        imp: Loaded imputer.
        args: Parsed CLI arguments.

    Returns:
        Result dict with per-family mean/std/n recovery.
    """
    out = {"track": track, "imputer": args.imputer, "n_seq": N_SEQ, "n_draws": N_DRAWS}
    per_fam = {fam: [] for fam in FAMILIES}
    for fold in (1, 2, 3):
        x_eval, _donors = load_eval(track, fold, args)
        fams = masks_for_track(track, x_eval.shape[1:], np.random.default_rng([2202, fold]))
        for fam, masks in fams.items():
            for si in range(len(x_eval)):
                rng = np.random.default_rng([2203, fold, si])
                for m in masks:
                    per_fam[fam].append(recovery(x_eval[si], imp, m, rng))
            log.info(
                "%s/%s fold%d %s: running mean %.3f",
                track,
                args.imputer,
                fold,
                fam,
                float(np.mean(per_fam[fam])),
            )
    out["families"] = {
        fam: {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        for fam, vals in per_fam.items()
    }
    return out


def run_controls(track, imp, args):
    """Run the unconditional and donor controls over all folds.

    Args:
        track: Track name.
        imp: Loaded imputer (used for the unconditional control only).
        args: Parsed CLI arguments.

    Returns:
        Result dict ``{"uncond_mean": float, "donor": {family: float}}``.
    """
    uncond = []
    don_fam = {fam: [] for fam in FAMILIES}
    for fold in (1, 2, 3):
        x_eval, donors = load_eval(track, fold, args)
        J, F, T = x_eval.shape[1:]
        all_hidden = np.zeros((J, F, T), bool)
        fams = masks_for_track(track, (J, F, T), np.random.default_rng([2202, fold]))
        for si in range(len(x_eval)):
            rng = np.random.default_rng([2203, fold, si])
            uncond.append(recovery(x_eval[si], imp, all_hidden, rng))
            for fam, masks in fams.items():
                for m in masks:
                    hid = ~m
                    a = donors[si].astype(np.float64)[hid]
                    b = x_eval[si].astype(np.float64)[hid]
                    c = (
                        0.0
                        if (np.std(a) < 1e-10 or np.std(b) < 1e-10)
                        else float(np.corrcoef(a, b)[0, 1])
                    )
                    don_fam[fam].append(c)
        log.info("%s/%s fold%d controls done", track, args.imputer, fold)
    return {
        "uncond_mean": float(np.mean(uncond)),
        "donor": {fam: float(np.mean(v)) for fam, v in don_fam.items()},
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", type=str, default="gate", choices=["gate", "controls"])
    ap.add_argument("--track", type=str, required=True, choices=["carepd", "esc50", "ptbxl"])
    ap.add_argument("--imputer", type=str, required=True, choices=["vaeac", "flow"])
    ap.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Imputer checkpoint (default: checkpoints/imputers/{track}_{imputer}.pt).",
    )
    ap.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Eval-cache dir (CARE-PD / PTB-XL; falls back to $CAREPD_CACHE_DIR / "
        "$PTBXL_CACHE_DIR, then the track defaults).",
    )
    ap.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("ESC50_DATA_DIR", str(REPO_ROOT / "data" / "esc50")),
        help="ESC-50 cache dir (fold{f}_train.npz / fold{f}_test.npz).",
    )
    ap.add_argument(
        "--data_path",
        type=str,
        default=os.environ.get("PTBXL_DATA_ROOT") or None,
        help="Raw PTB-XL root (used when --cache_dir is not given).",
    )
    ap.add_argument("--out", type=str, default=str(RESULTS_ROOT))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    """Run the requested gate or controls cell and write its JSON result."""
    args = parse_args()
    if args.cache_dir is None:
        env = {"carepd": "CAREPD_CACHE_DIR", "ptbxl": "PTBXL_CACHE_DIR"}.get(args.track)
        args.cache_dir = os.environ.get(env) if env else None

    ckpt = args.ckpt or str(IMPUTER_DIR / f"{args.track}_{args.imputer}.pt")
    if args.imputer == "vaeac":
        imp = FrameVAEACImputer.load(ckpt, device=args.device)
    else:
        imp = FrameFlowImputer.load(ckpt, device=args.device)
    log.info("[%s/%s] imputer loaded from %s", args.track, args.imputer, ckpt)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "gate":
        res = run_gate(args.track, imp, args)
        out_path = out_dir / f"gate_{args.track}_{args.imputer}.json"
        log.info("families: %s", json.dumps(res["families"], indent=1))
    else:
        res = run_controls(args.track, imp, args)
        out_path = out_dir / f"controls_{args.track}_{args.imputer}.json"
        log.info("controls: %s", json.dumps(res, indent=1))
    out_path.write_text(json.dumps(res, indent=1))
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
