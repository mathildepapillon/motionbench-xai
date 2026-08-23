"""scripts/run_carepd_players_shap.py — CARE-PD spatial / cell player-set KernelSHAP.

Runs sampled-coalition KernelSHAP on the BMCLab CARE-PD gait sequences with
two player sets that go beyond the temporal windows of
``run_care_pd_multiclf.py``:

- ``--playerset joint``: one player per H36M joint across all frames/axes
  (``SpatialJoints``, M=17).
- ``--playerset cell``: joint × temporal-window cells
  (``JointWindowCells``, M=68 = 17 joints × K=4 windows of 20 frames).

Both exceed the exact-enumeration bound (M <= 12), so coalitions come from
the fixed sampled design ``sampled_coalition_set(M, B=2048, seed=7919)``
shared across methods and folds; faithfulness is computed over all B+2
design rows (boundary rows included) and PlayerAOPC over the M explicit
deletion-path coalitions.  See ``scripts/_player_shap_common.py`` and
RESOLUTIONS.md §11 for the full protocol.

Value function (identical to the temporal sweep): coalition fills happen in
RAW cache coordinates; the filled clip goes through crop_scale + confidence
(``crop_scale_and_conf`` semantics, vectorised) and the CARE-PD
``MotionEncoder`` head (valid-frame masked mean over T, flatten joints,
linear head); ``v(S) = softmax(logits)[target]`` with the target = argmax of
the full-clip prediction.  MotionAGFormer clips are zero-padded 80 -> 81
frames BEFORE the transform with valid mask ``[1]*80 + [0]``.

POTR is excluded: it fails the pooled accuracy gate (0.374).

Data layouts (first match wins):

1. ``--cache_dir`` (or ``$CAREPD_CACHE_DIR``): prepared evaluation caches —
   ``fold{f}_eval.npz`` with key ``x`` of shape (n, 80, 17, 3) (raw world
   coordinates, label-filtered) and ``imputer_train.npz`` with key ``x``
   (the fold-1 train pool, used fold-independently for the mean/donor pool).
2. The CARE-PD release layout under ``$CARE_PD_ROOT`` (same cache.npz files
   as ``run_care_pd_multiclf.py``; the fold-1 pool is used whenever the eval
   cache carries no train split).

Checkpoints: ``--ckpt`` accepts either the release slim state dicts
(``motionbench/classifiers/checkpoints/real/carepd_bmclab_fold{f}_{clf}.pt``)
or the validation study's ``{state_dict, backbone, fold}`` dicts documented
in ``checkpoints/README.md`` (``carepd_clf/{clf}_fold{f}.pt``); both load
strictly after key remapping.  VAEAC/Flow imputer checkpoints default to
``checkpoints/imputers/carepd_{vaeac,flow}.pt`` (study format).

Usage::

    PYTHONPATH=. python scripts/run_carepd_players_shap.py \\
        --playerset joint --classifier motionbert --fold 1 \\
        --methods kernelshap_zero kernelshap_mean kernelshap_marginal

Results are written to::

    results/carepd_players/{playerset}/{classifier}/fold{fold}/{method}/result.json
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
from torch import Tensor

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

from motionbench.players import JointWindowCells, SpatialJoints  # noqa: E402

RESULTS_ROOT = REPO_ROOT / "results" / "carepd_players"
CARE_PD_ROOT = Path(os.environ.get("CARE_PD_ROOT", REPO_ROOT.parent / "CARE-PD"))
CACHE_TEMPLATE = str(
    CARE_PD_ROOT
    / "cache"
    / "flow_matching"
    / "BMCLab_h36m_80_classifier23fold_fold{fold}_eval"
    / "cache.npz"
)
TRAIN_POOL_CACHE = CARE_PD_ROOT / "cache" / "flow_matching" / "BMCLab_h36m_80_fold1" / "cache.npz"

# Release slim checkpoints first, then the study-format manifest layout.
CKPT_CANDIDATES = [
    "motionbench/classifiers/checkpoints/real/carepd_bmclab_fold{fold}_{clf}.pt",
    "checkpoints/carepd_clf/{clf}_fold{fold}.pt",
]
IMPUTER_DIR = REPO_ROOT / "checkpoints" / "imputers"

K = 4  # temporal windows for the cell player set
FW_BATCH = 256  # classifier forward batch (matches the validated runs)
IMP_CHUNK = 2048  # imputer rows per GPU pass (matches the validated runs)


# ---------------------------------------------------------------------- #
# Preprocessing — vectorised crop_scale_and_conf (fingerprint-exact)      #
# ---------------------------------------------------------------------- #


def crop_conf_batch(x_btj3: Tensor) -> Tensor:
    """Vectorised ``crop_scale_and_conf`` (release semantics).

    Args:
        x_btj3: (B, T, J, 3) raw world coordinates; padded/hidden frames are
            all-zero (z == 0 marks a padded frame).

    Returns:
        (B, T, J, 3) with (x, y) bbox-normalised to [-1, 1] over valid frames
        and channel 2 replaced by the per-frame confidence (1 real, 0 padded).
    """
    B, T, J, _ = x_btj3.shape
    valid = (x_btj3[..., 2] != 0).any(dim=-1)  # (B, T)
    has = valid.any(dim=1)  # (B,)
    vf = valid[:, :, None].expand(B, T, J)
    big = torch.finfo(x_btj3.dtype).max
    xs_ = torch.where(vf, x_btj3[..., 0], torch.full_like(x_btj3[..., 0], big))
    ys_ = torch.where(vf, x_btj3[..., 1], torch.full_like(x_btj3[..., 1], big))
    xmin = xs_.reshape(B, -1).min(dim=1).values
    ymin = ys_.reshape(B, -1).min(dim=1).values
    xs_ = torch.where(vf, x_btj3[..., 0], torch.full_like(x_btj3[..., 0], -big))
    ys_ = torch.where(vf, x_btj3[..., 1], torch.full_like(x_btj3[..., 1], -big))
    xmax = xs_.reshape(B, -1).max(dim=1).values
    ymax = ys_.reshape(B, -1).max(dim=1).values
    scale = torch.maximum(xmax - xmin, ymax - ymin)
    ok = has & (scale > 0)
    scale_safe = torch.where(ok, scale, torch.ones_like(scale))
    xs0 = (xmin + xmax - scale) / 2
    ys0 = (ymin + ymax - scale) / 2
    nx = ((x_btj3[..., 0] - xs0[:, None, None]) / scale_safe[:, None, None] - 0.5) * 2
    ny = ((x_btj3[..., 1] - ys0[:, None, None]) / scale_safe[:, None, None] - 0.5) * 2
    nx = nx.clamp(-1, 1)
    ny = ny.clamp(-1, 1)
    conf = valid.to(x_btj3.dtype)[:, :, None].expand(B, T, J)
    out = torch.stack([nx, ny, conf], dim=-1)
    return torch.where(ok[:, None, None, None], out, torch.zeros_like(out))


# ---------------------------------------------------------------------- #
# Classifier loading and value function                                   #
# ---------------------------------------------------------------------- #


def load_carepd_classifier(clf_name: str, fold: int, ckpt_path: Path, device: torch.device):
    """Build the ported backbone and strictly load either checkpoint format."""
    if clf_name == "motionbert":
        from motionbench.classifiers.ported_care_pd.motionbert import MotionBERTClassifier

        clf = MotionBERTClassifier(checkpoint_path=None, n_classes=3)
    elif clf_name == "motionagformer":
        from motionbench.classifiers.ported_care_pd.motionagformer import MotionAGFormerClassifier

        clf = MotionAGFormerClassifier(checkpoint_path=None, n_classes=3, n_frames=81)
    else:
        raise ValueError(f"unsupported classifier {clf_name!r} (POTR fails the accuracy gate)")

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw and "backbone" in raw:
        # Study format: {state_dict (encoder.*), backbone, fold}.
        if raw["backbone"] != clf_name or int(raw["fold"]) != int(fold):
            raise ValueError(
                f"checkpoint {ckpt_path} is ({raw['backbone']}, fold {raw['fold']}), "
                f"expected ({clf_name}, fold {fold})"
            )
        state = raw["state_dict"]
    else:
        state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    sd = {}
    for k, v in state.items():
        k2 = k[len("encoder.") :] if k.startswith("encoder.") else k
        if k2.startswith("head.fc_layers.0."):
            k2 = "cls_head." + k2[len("head.fc_layers.0.") :]
        k2 = k2.replace(".layer_scale_1", ".ls1").replace(".layer_scale_2", ".ls2")
        sd[k2] = v
    clf.load_state_dict(sd, strict=True)
    clf = clf.to(device)
    clf.eval()
    log.info("[%s fold%d] classifier loaded strictly from %s", clf_name, fold, ckpt_path)
    return clf


class CarePDValueFn:
    """v(S) evaluator: raw completions (n, J=17, F=3, T=80) -> (n, 3) probs.

    Applies the vectorised crop_scale + confidence transform, the ported
    backbone, and the CARE-PD ``ClassifierHead`` semantics (valid-frame
    masked mean over T, flatten joints, linear head).
    """

    def __init__(self, clf_name: str, clf, device: torch.device) -> None:
        self.clf_name = clf_name
        self.clf = clf
        self.device = device

    @torch.no_grad()
    def __call__(self, comps_jft: np.ndarray, batch: int = FW_BATCH) -> np.ndarray:
        comps_tjc = np.transpose(comps_jft, (0, 3, 1, 2))  # (n, T, J, 3)
        out = []
        n = len(comps_tjc)
        for s in range(0, n, batch):
            xb = torch.from_numpy(
                np.ascontiguousarray(comps_tjc[s : s + batch], dtype=np.float32)
            ).to(self.device)
            B, T = xb.shape[0], xb.shape[1]
            if self.clf_name == "motionagformer":
                xb = torch.cat([xb, torch.zeros(B, 1, 17, 3, device=self.device)], 1)
                vm = torch.cat(
                    [
                        torch.ones(B, T, device=self.device),
                        torch.zeros(B, 1, device=self.device),
                    ],
                    1,
                )
            else:
                vm = torch.ones(B, T, device=self.device)
            xc = crop_conf_batch(xb)
            rep = self.clf.backbone(xc, return_rep=True)  # (B, T', J, C)
            feat = rep.permute(0, 2, 3, 1)  # (B, J, C, T')
            mask = vm.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T')
            feat = (feat * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-6)
            logits = self.clf.cls_head(feat.reshape(feat.shape[0], -1))
            out.append(torch.softmax(logits, -1).float().cpu().numpy())
        return np.concatenate(out)


# ---------------------------------------------------------------------- #
# Data loading                                                            #
# ---------------------------------------------------------------------- #


def load_carepd_data(fold: int, n_seq: int, cache_dir: str | None):
    """Returns (x_eval (N, 17, 3, 80) f32, mean_jf (17, 3), donors (N, 17, 3, 80)).

    Selection, mean and donor conventions match the temporal sweep: first
    min(n_seq, n) label-filtered eval clips; per-(J, F) mean over the fold-1
    train pool; one donor per sequence via
    ``default_rng(42 + fold).integers(0, len(pool), N)``.
    """
    if cache_dir:
        d = np.load(Path(cache_dir) / f"fold{fold}_eval.npz", allow_pickle=True)
        x = np.transpose(d["x"], (0, 2, 3, 1)).astype(np.float32)  # (n, 17, 3, 80)
        pool = np.load(Path(cache_dir) / "imputer_train.npz")["x"]
        pool = np.transpose(pool, (0, 2, 3, 1)).astype(np.float32)
    else:
        cache_path = Path(CACHE_TEMPLATE.format(fold=fold))
        if not cache_path.exists():
            raise FileNotFoundError(
                f"CARE-PD cache not found: {cache_path}\n"
                "Set CARE_PD_ROOT or pass --cache_dir (see module docstring)."
            )
        d = np.load(cache_path, allow_pickle=True)
        x = np.transpose(d["x1_val"], (0, 2, 3, 1)).astype(np.float32)
        y_val = np.asarray(d["meta_updrs_gait_val"], dtype=np.int64)
        x = x[y_val >= 0]
        if d["x1_train"].shape[0] == 0:
            d_train = np.load(TRAIN_POOL_CACHE, allow_pickle=True)
            pool = np.transpose(d_train["x1_train"], (0, 2, 3, 1)).astype(np.float32)
        else:
            pool = np.transpose(d["x1_train"], (0, 2, 3, 1)).astype(np.float32)
    n = min(n_seq, len(x))
    x = x[:n]
    mean_jf = pool.mean(axis=(0, 3))  # (17, 3)
    rng = np.random.default_rng(42 + fold)
    donor_idx = rng.integers(0, pool.shape[0], size=n)
    donors = pool[donor_idx]
    return x, mean_jf, donors


# ---------------------------------------------------------------------- #
# Main                                                                    #
# ---------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--playerset", type=str, required=True, choices=["joint", "cell"])
    ap.add_argument(
        "--classifier",
        type=str,
        default="motionbert",
        choices=["motionbert", "motionagformer"],
    )
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
        default=os.environ.get("CAREPD_CACHE_DIR") or None,
        help="Prepared eval-cache dir (fold{f}_eval.npz + imputer_train.npz).",
    )
    ap.add_argument("--ckpt", type=str, default=None, help="Classifier checkpoint override.")
    ap.add_argument("--vaeac_ckpt", type=str, default=str(IMPUTER_DIR / "carepd_vaeac.pt"))
    ap.add_argument("--flow_ckpt", type=str, default=str(IMPUTER_DIR / "carepd_flow.pt"))
    ap.add_argument("--results_dir", type=str, default=str(RESULTS_ROOT))
    ap.add_argument("--device", type=str, default="cuda:0")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    clf_name, fold = args.classifier, args.fold
    device = torch.device(args.device)
    methods = args.methods if args.methods else DETERMINISTIC_METHODS

    x_eval, mean_jf, donors = load_carepd_data(fold, args.n_seq, args.cache_dir)
    N, J, F, T = x_eval.shape
    log.info("[%s fold%d] N=%d J=%d F=%d T=%d", clf_name, fold, N, J, F, T)

    players = SpatialJoints(J, F, T) if args.playerset == "joint" else JointWindowCells(J, K, F, T)

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        candidates = [REPO_ROOT / c.format(fold=fold, clf=clf_name) for c in CKPT_CANDIDATES]
        ckpt_path = next((c for c in candidates if c.exists()), candidates[0])
    clf = load_carepd_classifier(clf_name, fold, ckpt_path, device)
    value_fn = CarePDValueFn(clf_name, clf, device)

    probs_full = value_fn(x_eval)
    targets = probs_full.argmax(-1)
    log.info(
        "[%s fold%d] target distribution: %s",
        clf_name,
        fold,
        np.bincount(targets, minlength=3).tolist(),
    )

    for method in methods:
        cell_dir = Path(args.results_dir) / args.playerset / clf_name / f"fold{fold}" / method
        if (cell_dir / "result.json").exists():
            log.info("[%s fold%d] %s already done, skipping.", clf_name, fold, method)
            continue

        imputer = None
        if method == "kernelshap_vaeac":
            from motionbench.imputers.frame_vaeac import FrameVAEACImputer

            imputer = FrameVAEACImputer.load(args.vaeac_ckpt, device=args.device)
        elif method == "kernelshap_flow":
            from motionbench.imputers.frame_flow import FrameFlowImputer

            imputer = FrameFlowImputer.load(args.flow_ckpt, device=args.device)

        log.info(
            "[%s fold%d] %s (playerset=%s, M=%d)",
            clf_name,
            fold,
            method,
            args.playerset,
            players.n_players,
        )
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
                "dataset": "care_pd_bmclab_cache",
                "player_set": f"{args.playerset}_m{players.n_players}",
                "classifier": clf_name,
            },
        )


if __name__ == "__main__":
    main()
