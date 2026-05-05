"""Validate that ported CARE-PD classifiers reproduce CARE-PD fold-1 results.

Usage (from the motionbench-xai root):
    conda run -n motionbench-xai python scripts/validate_care_pd_accuracy.py \
        --care-pd-root /path/to/CARE-PD

Or set the environment variable:
    export CARE_PD_ROOT=/path/to/CARE-PD
    conda run -n motionbench-xai python scripts/validate_care_pd_accuracy.py

What this script does
---------------------
1.  Loads the BMCLab NPZ file(s) (3-D world-to-camera or 2-D image-projected
    skeletons, 17 joints).
2.  Builds raw clips in motionbench convention ``(N, J, F, T)`` — **no**
    preprocessing is applied here; each classifier handles its own
    preprocessing inside ``forward()``.
3.  Runs the fold-1 fine-tuned checkpoint through the motionbench classifier.
4.  Compares logits and argmax predictions against the stored reference
    logits from CARE-PD's own eval run.

Reference accuracy numbers (23-fold CV, SUB01 as test fold):
    MotionBERT    accuracy ≈ 0.68   macro-F1 ≈ 0.65
    MotionAGFormer  (reference logits stale; port architecture validated)
    PoseFormerV2  (reference logits compared directly)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------------
# Paths — resolved from CLI args or environment variables
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate ported CARE-PD classifiers against reference logits."
    )
    parser.add_argument(
        "--care-pd-root",
        type=Path,
        default=Path(os.environ.get("CARE_PD_ROOT", "CARE-PD")),
        help="Root of the CARE-PD repository (default: $CARE_PD_ROOT or ./CARE-PD)",
    )
    parser.add_argument(
        "--motionbench-root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Root of motionbench-xai repo (default: parent of scripts/)",
    )
    return parser.parse_args()


_args = _parse_args()
CARE_PD_ROOT = _args.care_pd_root
MOTIONBENCH_ROOT = _args.motionbench_root

# 3-D world-to-camera NPZ (MotionBERT / MotionAGFormer / POTR)
NPZ_3D = CARE_PD_ROOT / (
    "assets/datasets/h36m/BMCLab/"
    "h36m_3d_world2cam_backright_floorXZZplus_30f_or_longer.npz"
)

# 2-D image-projected NPZ (PoseFormerV2)
NPZ_2D = CARE_PD_ROOT / (
    "assets/datasets/h36m/BMCLab/"
    "h36m_3d_world2cam2img_backright_floorXZZplus_30f_or_longer.npz"
)

LABELS_PKL = CARE_PD_ROOT / "assets/datasets/BMCLab.pkl"

CKPT_DIR = MOTIONBENCH_ROOT / "motionbench/classifiers/checkpoints/care_pd"

# fold-1 reference outputs from CARE-PD training
MOTIONBERT_CKPT = CKPT_DIR / "motionbert_bmclab_fold1.pth.tr"
MOTIONBERT_LOGITS_REF = (
    CARE_PD_ROOT
    / "experiment_outs/Hypertune/motionbert_BMCLab/0"
    / "train_BMCLab_test_BMCLab_23fold/logits/logits_last_fold1.json"
)
MOTIONBERT_RESULTS_REF = (
    CARE_PD_ROOT
    / "experiment_outs/Hypertune/motionbert_BMCLab/0"
    / "train_BMCLab_test_BMCLab_23fold/results/results_last_fold1.json"
)

MOTIONAGFORMER_CKPT = CKPT_DIR / "motionagformer_bmclab_fold1.pth.tr"
MOTIONAGFORMER_LOGITS_REF = (
    CARE_PD_ROOT
    / "experiment_outs/Hypertune/motionagformer_BMCLab/0"
    / "train_BMCLab_test_BMCLab_23fold/logits/logits_last_fold1.json"
)

POSEFORMERV2_CKPT = CKPT_DIR / "poseformerv2_bmclab_fold1.pth.tr"
POSEFORMERV2_LOGITS_REF = (
    CARE_PD_ROOT
    / "experiment_outs/Hypertune/poseformerv2_BMCLab/0"
    / "train_BMCLab_test_BMCLab_23fold/logits/logits_last_fold1.json"
)

# Hyper-params from CARE-PD config
N_CLASSES = 3
MOTIONBERT_SEQ_LEN = 90     # source_seq_len for MotionBERT BMCLab training
MOTIONAGFORMER_SEQ_LEN = 81  # source_seq_len for MotionAGFormer BMCLab training
POSEFORMERV2_SEQ_LEN = 81   # source_seq_len for PoseFormerV2 BMCLab training


# ---------------------------------------------------------------------------
# Raw-data batch builder (NO preprocessing — models handle it internally)
# ---------------------------------------------------------------------------


def build_raw_batch(
    npz_path: Path,
    video_names: list[str],
    seq_len: int,
    feature_dim: int,
) -> torch.Tensor:
    """Return ``(N, J, F, T)`` float32 tensor of raw (unprocessed) clips.

    Clips longer than ``seq_len`` are truncated to the first ``seq_len``
    frames.  Shorter clips are zero-padded at the end (padded frames are
    detected inside each classifier's ``_preprocess`` by their all-zero
    coordinates).

    Args:
        npz_path: Path to the ``.npz`` file (3-D or 2-D).
        video_names: Video names from CARE-PD's stored logit file; may
            include a ``_view0`` suffix which is stripped for lookup.
        seq_len: Target number of frames.
        feature_dim: Expected number of coordinate channels in the NPZ
            (3 for 3-D, 2 for 2-D).
    """
    data = np.load(npz_path, allow_pickle=True)
    clips = []

    for vname in video_names:
        key = vname.replace("_view0", "")
        if key not in data:
            raise KeyError(
                f"Key {key!r} not found in NPZ. "
                f"Available sample: {list(data.keys())[:3]}"
            )
        raw = data[key]  # (T, J, F)
        T = raw.shape[0]

        if T >= seq_len:
            clip = raw[:seq_len].copy()
        else:
            pad = np.zeros(
                (seq_len - T, raw.shape[1], raw.shape[2]),
                dtype=raw.dtype,
            )
            clip = np.concatenate([raw, pad], axis=0)

        # motionbench convention: (J, F, T)
        clip_mb = clip.transpose(1, 2, 0)  # (J, F, T)
        clips.append(clip_mb)

    batch = np.stack(clips, axis=0)   # (N, J, F, T)
    return torch.from_numpy(batch.astype(np.float32))


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def load_reference(
    logits_path: Path,
    results_path: Path | None = None,
) -> tuple[list, list, list]:
    """Load CARE-PD stored logits and true labels for fold-1 (SUB01)."""
    with open(logits_path) as f:
        ref_logits_data = json.load(f)
    logits_ref = ref_logits_data["predicted_logits"]
    true_labels = ref_logits_data["true_labels"]
    video_names = ref_logits_data["video_names"]
    return logits_ref, true_labels, video_names


def run_inference(
    model: torch.nn.Module,
    batch: torch.Tensor,
    chunk: int = 64,
) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(batch), chunk):
            logits = model(batch[i : i + chunk].to(device))
            all_logits.append(logits.cpu())
    return torch.cat(all_logits, dim=0)


def print_metrics(
    all_logits_t: torch.Tensor,
    logits_ref: list,
    true_labels: list,
    class_names: list[str] | None = None,
) -> None:
    class_names = class_names or ["UPDRS-0", "UPDRS-1", "UPDRS-2"]
    pred_labels = all_logits_t.argmax(dim=-1).numpy()
    true_arr = np.array(true_labels)
    logits_ref_arr = np.array(logits_ref)
    ref_preds = np.argmax(logits_ref_arr, axis=-1)
    ours = all_logits_t.numpy()

    cos_sim = (
        (ours * logits_ref_arr).sum(axis=-1)
        / (
            np.linalg.norm(ours, axis=-1)
            * np.linalg.norm(logits_ref_arr, axis=-1)
            + 1e-9
        )
    )
    logit_mae = np.abs(ours - logits_ref_arr).mean()
    our_acc = accuracy_score(true_arr, pred_labels)
    ref_acc = accuracy_score(true_arr, ref_preds)

    print(f"\n  Logit cosine similarity  — mean: {cos_sim.mean():.4f}  min: {cos_sim.min():.4f}")
    print(f"  Logit MAE                — mean: {logit_mae:.4f}")
    print(f"\n  CARE-PD reference accuracy  : {ref_acc:.4f}")
    print(f"  Our port accuracy           : {our_acc:.4f}")
    print(f"  Prediction agreement w/ ref : {(pred_labels == ref_preds).mean():.4f}")
    print("\n  Classification report (our port):")
    print(classification_report(true_arr, pred_labels, target_names=class_names,
                                labels=[0, 1, 2]))


# ---------------------------------------------------------------------------
# Per-classifier validation functions
# ---------------------------------------------------------------------------


def validate_motionbert() -> None:
    print("\n" + "=" * 70)
    print("  Validating MotionBERT (fold 1 — SUB01 as test subject)")
    print("=" * 70)
    if not MOTIONBERT_CKPT.exists():
        print(f"  SKIP: checkpoint not found at {MOTIONBERT_CKPT}")
        return
    if not MOTIONBERT_LOGITS_REF.exists():
        print(f"  SKIP: reference logits not found at {MOTIONBERT_LOGITS_REF}")
        return

    sys.path.insert(0, str(MOTIONBENCH_ROOT))
    from motionbench.classifiers.ported_care_pd.motionbert import MotionBERTClassifier

    model = MotionBERTClassifier(
        checkpoint_path=str(MOTIONBERT_CKPT),
        n_classes=N_CLASSES,
        merge_joints=False,
    )
    logits_ref, true_labels, video_names = load_reference(
        MOTIONBERT_LOGITS_REF, MOTIONBERT_RESULTS_REF
    )
    print(f"  Reference: {len(video_names)} clips")

    # Raw 3-D data — preprocessing happens inside model.forward()
    batch = build_raw_batch(
        NPZ_3D, video_names,
        seq_len=MOTIONBERT_SEQ_LEN,
        feature_dim=3,
    )
    print(f"  Batch shape: {tuple(batch.shape)}")

    all_logits_t = run_inference(model, batch)
    print_metrics(all_logits_t, logits_ref, true_labels)


def validate_motionagformer() -> None:
    print("\n" + "=" * 70)
    print("  Validating MotionAGFormer (fold 1 — SUB01 as test subject)")
    print("=" * 70)
    if not MOTIONAGFORMER_CKPT.exists():
        print(f"  SKIP: checkpoint not found at {MOTIONAGFORMER_CKPT}")
        return
    if not MOTIONAGFORMER_LOGITS_REF.exists():
        print(f"  SKIP: reference logits not found at {MOTIONAGFORMER_LOGITS_REF}")
        return

    from motionbench.classifiers.ported_care_pd.motionagformer import MotionAGFormerClassifier

    model = MotionAGFormerClassifier(
        checkpoint_path=str(MOTIONAGFORMER_CKPT),
        n_classes=N_CLASSES,
        n_frames=MOTIONAGFORMER_SEQ_LEN,
        merge_joints=False,
    )
    logits_ref, true_labels, video_names = load_reference(MOTIONAGFORMER_LOGITS_REF)
    print(f"  Reference: {len(video_names)} clips")

    batch = build_raw_batch(
        NPZ_3D, video_names,
        seq_len=MOTIONAGFORMER_SEQ_LEN,
        feature_dim=3,
    )
    print(f"  Batch shape: {tuple(batch.shape)}")

    all_logits_t = run_inference(model, batch)
    print_metrics(all_logits_t, logits_ref, true_labels)


def validate_poseformerv2() -> None:
    print("\n" + "=" * 70)
    print("  Validating PoseFormerV2 (fold 1 — SUB01 as test subject)")
    print("=" * 70)
    if not POSEFORMERV2_CKPT.exists():
        print(f"  SKIP: checkpoint not found at {POSEFORMERV2_CKPT}")
        return
    if not POSEFORMERV2_LOGITS_REF.exists():
        print(f"  SKIP: reference logits not found at {POSEFORMERV2_LOGITS_REF}")
        return

    from motionbench.classifiers.ported_care_pd.poseformerv2 import PoseFormerV2Classifier

    model = PoseFormerV2Classifier(
        checkpoint_path=str(POSEFORMERV2_CKPT),
        n_classes=N_CLASSES,
    )
    logits_ref, true_labels, video_names = load_reference(POSEFORMERV2_LOGITS_REF)
    print(f"  Reference: {len(video_names)} clips")

    # Raw 2-D pixel data — screen normalisation happens inside model.forward()
    batch = build_raw_batch(
        NPZ_2D, video_names,
        seq_len=POSEFORMERV2_SEQ_LEN,
        feature_dim=2,
    )
    print(f"  Batch shape: {tuple(batch.shape)}")

    all_logits_t = run_inference(model, batch)
    print_metrics(all_logits_t, logits_ref, true_labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    validate_motionbert()
    validate_motionagformer()
    validate_poseformerv2()
    print("\nDone.")
