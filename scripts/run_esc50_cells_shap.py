"""scripts/run_esc50_cells_shap.py — ESC-50 band × window cell KernelSHAP.

Runs sampled-coalition KernelSHAP on ESC-50 mel-spectrograms with the
spectro-temporal cell player set: 4 contiguous mel-bin quartiles (the same
band partition as ``run_esc50_freq_shap.py``) × K=4 temporal windows of 256
frames, i.e. ``BandWindowCells`` with M=16 players.

M=16 exceeds the exact-enumeration bound (M <= 12), so coalitions come from
the fixed sampled design ``sampled_coalition_set(16, B=2048, seed=7919)``
shared across methods and folds; faithfulness is computed over all B+2
design rows (boundary rows included) and PlayerAOPC over the M explicit
deletion-path coalitions.  See ``scripts/_player_shap_common.py`` and
RESOLUTIONS.md §12 for the full protocol.

Value function, data selection, mean and donor conventions are identical to
the temporal sweep ``run_esc50_shap.py``: fills happen in raw mel space; the
fold's AST classifier (fold-disciplined fine-tune) returns softmax probabilities;
``v(S) = probs[target]`` with target = argmax of the full-clip prediction;
the eval subset is ``sort(default_rng(42 + fold).choice(400, 200))`` and the
marginal donors continue the same stream.

VAEAC/Flow imputer checkpoints default to
``checkpoints/imputers/esc50_{vaeac,flow}.pt`` (the validation-study format
documented in ``checkpoints/README.md``).

Usage::

    PYTHONPATH=. python scripts/run_esc50_cells_shap.py --fold 1 \\
        --methods kernelshap_zero kernelshap_mean kernelshap_marginal

Results are written to::

    results/esc50_cells/fold{fold}/{method}/result.json
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

from motionbench.players import BandWindowCells  # noqa: E402

RESULTS_ROOT = REPO_ROOT / "results" / "esc50_cells"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "esc50"
IMPUTER_DIR = REPO_ROOT / "checkpoints" / "imputers"

N_BANDS = 4  # contiguous mel-bin quartiles (run_esc50_freq_shap partition)
K = 4  # temporal windows of 256 frames
FW_BATCH = 48  # AST forward batch (matches the validated runs)
IMP_CHUNK = 96  # imputer rows per GPU pass (matches the validated runs)


def load_esc50_data(fold: int, n_seq: int, data_dir: Path):
    """Returns (x_eval (N, 128, 1, 1024) f32, mean_jf (128, 1), donors (N, ...)).

    Selection and donor conventions are those of ``run_esc50_shap.py``:
    sorted ``default_rng(42 + fold).choice`` eval subset, per-(J, F) train
    mean over (samples, time), donors drawn by continuing the same stream.
    """
    test_d = np.load(data_dir / f"fold{fold}_test.npz")
    train_d = np.load(data_dir / f"fold{fold}_train.npz")
    x_all = test_d["x_test"].astype(np.float32)  # (400, 128, 1, 1024)
    x_train = train_d["x_train"].astype(np.float32)  # (1600, 128, 1, 1024)
    rng = np.random.default_rng(42 + fold)
    n = min(n_seq, len(x_all))
    idx = np.sort(rng.choice(len(x_all), size=n, replace=False)) if n < len(x_all) else np.arange(n)
    x = x_all[idx]
    mean_jf = x_train.mean(axis=(0, 3))  # (128, 1)
    donor_idx = rng.integers(0, x_train.shape[0], size=n)  # stream continuation
    donors = x_train[donor_idx]
    return x, mean_jf, donors


class ESC50ValueFn:
    """v(S) evaluator: raw mel completions (n, 128, 1, 1024) -> (n, 50) probs."""

    def __init__(self, fold: int, device: torch.device) -> None:
        from motionbench.classifiers.esc50_classifier import load_esc50_classifier

        self.clf = load_esc50_classifier(fold=fold, device=device)
        self.device = device

    @torch.no_grad()
    def __call__(self, comps_jft: np.ndarray, batch: int = FW_BATCH) -> np.ndarray:
        out = []
        for s in range(0, len(comps_jft), batch):
            xb = torch.from_numpy(
                np.ascontiguousarray(comps_jft[s : s + batch], dtype=np.float32)
            ).to(self.device)
            probs = self.clf(xb)  # softmax inside the wrapper
            out.append(probs.float().cpu().numpy())
        return np.concatenate(out)


def parse_args() -> argparse.Namespace:
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
        "--data_dir",
        type=str,
        default=os.environ.get("ESC50_DATA_DIR", str(DEFAULT_DATA_DIR)),
        help="Directory with fold{f}_train.npz / fold{f}_test.npz caches.",
    )
    ap.add_argument("--vaeac_ckpt", type=str, default=str(IMPUTER_DIR / "esc50_vaeac.pt"))
    ap.add_argument("--flow_ckpt", type=str, default=str(IMPUTER_DIR / "esc50_flow.pt"))
    ap.add_argument("--results_dir", type=str, default=str(RESULTS_ROOT))
    ap.add_argument("--device", type=str, default="cuda:0")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    fold = args.fold
    device = torch.device(args.device)
    methods = args.methods if args.methods else DETERMINISTIC_METHODS

    x_eval, mean_jf, donors = load_esc50_data(fold, args.n_seq, Path(args.data_dir))
    N, J, F, T = x_eval.shape
    log.info("[fold%d] N=%d J=%d F=%d T=%d", fold, N, J, F, T)

    players = BandWindowCells(J=J, n_bands=N_BANDS, K=K, F=F, T=T)
    value_fn = ESC50ValueFn(fold, device)

    probs_full = value_fn(x_eval)
    targets = probs_full.argmax(-1)

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

        log.info("[fold%d] %s (band x window cells, M=%d)", fold, method, players.n_players)
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
                "dataset": "esc50",
                "player_set": f"band_window_cells_m{players.n_players}",
                "classifier": "ast",
            },
        )


if __name__ == "__main__":
    main()
