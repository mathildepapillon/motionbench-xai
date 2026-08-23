"""scripts/run_ptbxl_cells_shap.py — PTB-XL lead × window cell KernelSHAP.

Sampled-coalition KernelSHAP (B=2048) on PTB-XL ECGs, per fold, with the
spatio-temporal cell player set ``JointWindowCells``: 12 leads × K=4
temporal windows of 250 samples, M=48 (cell index ``lead * 4 + window``) —
the finer-granularity companion of ``run_ptbxl_leads_shap.py`` (M=12,
exact enumeration).  Value function and pool conventions match the leads
sweep: fills in z-scored cache coordinates, per-fold 1-D ResNet
(``ECGResNet1dClassifier``), 2000-record training pool with donors
``default_rng(42 + fold).integers(0, len(pool), N)``.  Data resolves via
``--cache_dir``/``$PTBXL_CACHE_DIR``, then ``--data_path``/
``$PTBXL_DATA_ROOT`` (raw PTB-XL through ``PTBXLDataset``); classifier and
imputer checkpoints per ``checkpoints/README.md``.  Shared
coalition/fill/phi/metric protocol: ``scripts/_player_shap_common.py`` and
RESOLUTIONS.md §12.

Usage::

    PYTHONPATH=. python scripts/run_ptbxl_cells_shap.py --fold 1 \\
        --methods kernelshap_zero kernelshap_mean kernelshap_marginal

Results: ``results/ptbxl_cells/fold{fold}/{method}/result.json``
"""

from __future__ import annotations

import argparse
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

from _player_shap_common import (  # noqa: E402
    ALL_METHODS,
    DETERMINISTIC_METHODS,
    run_player_cell,
)

from motionbench.players import JointWindowCells  # noqa: E402

RESULTS_ROOT = REPO_ROOT / "results" / "ptbxl_cells"
CKPT_DIR = REPO_ROOT / "motionbench" / "classifiers" / "checkpoints" / "real"
CKPT_CANDIDATES = [
    "motionbench/classifiers/checkpoints/real/ptbxl_fold{fold}.pt",
    "checkpoints/ptbxl_clf/fold{fold}.pt",
]
IMPUTER_DIR = REPO_ROOT / "checkpoints" / "imputers"

K = 4  # temporal windows of 250 samples
FW_BATCH = 1024  # classifier forward batch (matches the validated runs)
IMP_CHUNK = 192  # imputer rows per GPU pass (matches the validated runs)


def load_ptbxl_classifier(fold: int, ckpt_path: Path, device: torch.device):
    """Build ``ECGResNet1dClassifier`` and strictly load either checkpoint format."""
    from motionbench.classifiers.ported_ptbxl.resnet1d import ECGResNet1dClassifier

    clf = ECGResNet1dClassifier(checkpoint_path=None, n_classes=2)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    if any(k.startswith("block1.") for k in state):
        # Study format: bare block1/2/3 + head layout.
        sd = {}
        for k, v in state.items():
            if k.startswith(("block1.", "block2.", "block3.")):
                sd["backbone." + k] = v
            elif k.startswith("head."):
                sd["cls_head." + k[len("head.") :]] = v
            else:
                sd[k] = v
    else:
        sd = dict(state)
    clf.load_state_dict(sd, strict=True)
    clf = clf.to(device)
    clf.eval()
    log.info("[fold%d] classifier loaded strictly from %s", fold, ckpt_path)
    return clf


class PTBXLValueFn:
    """v(S) evaluator: cache-space completions (n, 12, 1, 1000) -> (n, 2) probs."""

    def __init__(self, clf, device: torch.device) -> None:
        """Initialise with a loaded per-fold classifier."""
        self.clf = clf
        self.device = device

    @torch.no_grad()
    def __call__(self, comps_jft: np.ndarray, batch: int = FW_BATCH) -> np.ndarray:
        out = []
        for s in range(0, len(comps_jft), batch):
            xb = torch.from_numpy(
                np.ascontiguousarray(comps_jft[s : s + batch], dtype=np.float32)
            ).to(self.device)
            logits = self.clf(xb)
            out.append(torch.softmax(logits, -1).float().cpu().numpy())
        return np.concatenate(out)


def load_ptbxl_data(fold: int, n_seq: int, cache_dir: str | None, data_path: str | None):
    """Returns (x_eval (N, 12, 1, 1000) f32, mean_jf (12, 1), donors (N, ...)).

    Pool conventions match ``run_ptbxl_leads_shap.py``: per-(J, F) mean over
    the 2000-record training pool; donors via
    ``default_rng(42 + fold).integers(0, len(pool), N)``.
    """
    if cache_dir:
        x = np.load(Path(cache_dir) / f"fold{fold}_eval.npz")["x"].astype(np.float32)
        pool = np.load(Path(cache_dir) / f"fold{fold}_pool.npz")["x"].astype(np.float32)
    elif data_path:
        stats_path = CKPT_DIR / f"ptbxl_fold{fold}_stats.npz"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Stats file {stats_path} not found — run train_ptbxl_classifier.py first."
            )
        stats = np.load(stats_path)
        train_stats = (stats["mean"].astype(np.float32), stats["std"].astype(np.float32))

        from motionbench.data.real.ptbxl import _FOLD_SPLITS, PTBXLDataset

        _FOLD_SPLITS["_test_folds"] = (stats["test_folds"].tolist(),)
        test_ds = PTBXLDataset(
            data_path=data_path,
            split="_test_folds",
            normalize=True,
            max_sequences=n_seq,
            train_stats=train_stats,
        )
        del _FOLD_SPLITS["_test_folds"]
        x = np.stack([s[0] for s in test_ds._samples[:n_seq]], axis=0)
        x = x.transpose(0, 2, 1)[:, :, np.newaxis, :].astype(np.float32)

        _FOLD_SPLITS["_train_folds"] = (stats["train_folds"].tolist(),)
        train_ds = PTBXLDataset(
            data_path=data_path,
            split="_train_folds",
            normalize=True,
            max_sequences=2000,
            train_stats=train_stats,
        )
        del _FOLD_SPLITS["_train_folds"]
        pool = np.stack([s[0] for s in train_ds._samples], axis=0)
        pool = pool.transpose(0, 2, 1)[:, :, np.newaxis, :].astype(np.float32)
    else:
        raise ValueError("Provide --cache_dir or --data_path (see module docstring).")
    n = min(n_seq, len(x))
    x = x[:n]
    mean_jf = pool.mean(axis=(0, 3))  # (12, 1)
    rng = np.random.default_rng(42 + fold)
    donor_idx = rng.integers(0, pool.shape[0], size=n)
    donors = pool[donor_idx]
    return x, mean_jf, donors


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fold", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--n_seq", type=int, default=200)
    ap.add_argument("--budget", type=int, default=2048, help="Interior sampled-coalition rows.")
    ap.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help=f"Subset of {ALL_METHODS} (default: deterministic methods).",
    )
    ap.add_argument(
        "--cache_dir",
        type=str,
        default=os.environ.get("PTBXL_CACHE_DIR") or None,
        help="Prepared cache dir (fold{f}_eval.npz + fold{f}_pool.npz).",
    )
    ap.add_argument(
        "--data_path",
        type=str,
        default=os.environ.get("PTBXL_DATA_ROOT") or None,
        help="Raw PTB-XL root (used when --cache_dir is not given).",
    )
    ap.add_argument("--ckpt", type=str, default=None, help="Classifier checkpoint override.")
    ap.add_argument("--vaeac_ckpt", type=str, default=str(IMPUTER_DIR / "ptbxl_vaeac.pt"))
    ap.add_argument("--flow_ckpt", type=str, default=str(IMPUTER_DIR / "ptbxl_flow.pt"))
    ap.add_argument("--results_dir", type=str, default=str(RESULTS_ROOT))
    ap.add_argument("--device", type=str, default="cuda:0")
    return ap.parse_args()


def main() -> None:
    """Run the cell player-set sweep for one fold."""
    args = parse_args()
    fold = args.fold
    device = torch.device(args.device)
    methods = args.methods if args.methods else DETERMINISTIC_METHODS

    x_eval, mean_jf, donors = load_ptbxl_data(fold, args.n_seq, args.cache_dir, args.data_path)
    N, J, F, T = x_eval.shape
    log.info("[fold%d] N=%d J=%d F=%d T=%d", fold, N, J, F, T)

    players = JointWindowCells(J, K, F, T)

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        candidates = [REPO_ROOT / c.format(fold=fold) for c in CKPT_CANDIDATES]
        ckpt_path = next((c for c in candidates if c.exists()), candidates[0])
    clf = load_ptbxl_classifier(fold, ckpt_path, device)
    value_fn = PTBXLValueFn(clf, device)

    probs_full = value_fn(x_eval)
    targets = probs_full.argmax(-1)
    log.info("[fold%d] target distribution: %s", fold, np.bincount(targets, minlength=2).tolist())

    for method in methods:
        cell_dir = Path(args.results_dir) / f"fold{fold}" / method
        if (cell_dir / "result.json").exists():
            log.info("[fold%d] %s already done, skipping.", fold, method)
            continue

        imputer = None
        if method == "kernelshap_vaeac":
            from motionbench.imputers.frame_vaeac import FrameVAEACImputer

            imputer = FrameVAEACImputer.load(args.vaeac_ckpt, device=args.device)
        elif method == "kernelshap_flow":
            from motionbench.imputers.frame_flow import FrameFlowImputer

            imputer = FrameFlowImputer.load(args.flow_ckpt, device=args.device)

        log.info("[fold%d] %s (lead x window cells, M=%d)", fold, method, players.n_players)
        run_player_cell(
            x_eval=x_eval,
            mean_jf=mean_jf,
            donors=donors,
            players=players,
            method=method,
            fold=fold,
            probs_fn=value_fn,
            targets=targets,
            imputer=imputer,
            imp_chunk=IMP_CHUNK,
            budget=args.budget,
            cell_dir=cell_dir,
            meta={
                "dataset": "ptbxl",
                "player_set": f"lead_window_cells_m{players.n_players}",
                "classifier": "ecg_resnet1d",
            },
        )


if __name__ == "__main__":
    main()
