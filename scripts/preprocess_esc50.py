"""scripts/preprocess_esc50.py — Preprocess ESC-50 audio to mel-spectrogram npz arrays.

Reads ESC-50 CSV, loads WAV files with soundfile + scipy resample,
runs ASTFeatureExtractor to produce 128x1024 mel-spectrograms, and saves
per-fold numpy arrays in (N, J=128, F=1, T=1024) format.

Our 3-fold split from ESC-50's 5 folds:
  - fold 1: ESC folds {2,3,4,5} train, ESC fold {1} test
  - fold 2: ESC folds {1,3,4,5} train, ESC fold {2} test
  - fold 3: ESC folds {1,2,4,5} train, ESC fold {3} test

Usage::

    python scripts/preprocess_esc50.py \\
        --esc50_dir data/esc50/ESC-50-master \\
        --output_dir data/esc50
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from math import gcd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--esc50_dir",
        type=str,
        default="data/esc50/ESC-50-master",
        help="Path to ESC-50-master directory",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="data/esc50",
        help="Output directory for .npz files",
    )
    return ap.parse_args()


# Our 3-fold splits: each entry is (train_esc_folds, test_esc_fold)
FOLD_SPLITS = {
    1: ([2, 3, 4, 5], 1),
    2: ([1, 3, 4, 5], 2),
    3: ([1, 2, 4, 5], 3),
}


def resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample waveform using scipy polyphase filter."""
    if orig_sr == target_sr:
        return waveform
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return resample_poly(waveform, up, down).astype(np.float32)


def main() -> None:
    args = parse_args()
    esc50_dir = _REPO_ROOT / args.esc50_dir
    output_dir = _REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load CSV
    csv_path = esc50_dir / "meta" / "esc50.csv"
    audio_dir = esc50_dir / "audio"

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"[esc50] CSV: {len(rows)} rows")

    # Load ASTFeatureExtractor
    print("[esc50] loading ASTFeatureExtractor...")
    from transformers import ASTFeatureExtractor
    fe = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    assert fe.sampling_rate == 16000
    assert fe.num_mel_bins == 128
    assert fe.max_length == 1024
    print(f"[esc50] fe: sr={fe.sampling_rate}, bins={fe.num_mel_bins}, max_len={fe.max_length}")

    TARGET_SR = 16000
    all_specs = []   # list of (128, 1, 1024) float32 np arrays
    all_labels = []  # list of int
    all_folds = []   # list of int (ESC fold 1-5)

    t0 = time.time()
    for idx, row in enumerate(rows):
        filename = row["filename"]
        esc_fold = int(row["fold"])
        target = int(row["target"])

        wav_path = audio_dir / filename
        waveform, sr = sf.read(str(wav_path), dtype="float32")

        # Convert stereo to mono if needed
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)

        # Resample to 16kHz if needed
        if sr != TARGET_SR:
            waveform = resample_audio(waveform, sr, TARGET_SR)

        # Extract mel-spectrogram via ASTFeatureExtractor
        # Returns input_values of shape (1, max_length=1024, num_mel_bins=128)
        out = fe(waveform, sampling_rate=TARGET_SR, return_tensors="pt", padding="max_length")
        input_values = out.input_values  # (1, 1024, 128)

        # Permute to (128, 1024), then unsqueeze to (128, 1, 1024)
        spec = input_values.squeeze(0)  # (1024, 128)
        spec = spec.permute(1, 0)       # (128, 1024)
        spec = spec.unsqueeze(1)        # (128, 1, 1024)
        spec_np = spec.numpy().astype(np.float32)  # (128, 1, 1024)

        all_specs.append(spec_np)
        all_labels.append(target)
        all_folds.append(esc_fold)

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/2000] {elapsed:.1f}s elapsed ({elapsed/(idx+1)*2000:.0f}s est. total)")

    print(f"[esc50] all {len(all_specs)} clips processed in {time.time()-t0:.1f}s")

    all_specs_arr = np.stack(all_specs, axis=0)   # (2000, 128, 1, 1024)
    all_labels_arr = np.array(all_labels, dtype=np.int64)  # (2000,)
    all_folds_arr = np.array(all_folds, dtype=np.int64)    # (2000,)

    print(f"[esc50] stacked array shape: {all_specs_arr.shape}, dtype: {all_specs_arr.dtype}")
    print(f"[esc50] label range: [{all_labels_arr.min()}, {all_labels_arr.max()}], unique folds: {np.unique(all_folds_arr)}")

    # Save per-fold npz files
    for our_fold, (train_esc_folds, test_esc_fold) in FOLD_SPLITS.items():
        train_mask = np.isin(all_folds_arr, train_esc_folds)
        test_mask = (all_folds_arr == test_esc_fold)

        x_train = all_specs_arr[train_mask]
        y_train = all_labels_arr[train_mask]
        x_test = all_specs_arr[test_mask]
        y_test = all_labels_arr[test_mask]

        train_path = output_dir / f"fold{our_fold}_train.npz"
        test_path = output_dir / f"fold{our_fold}_test.npz"

        np.savez_compressed(train_path, x_train=x_train, y_train=y_train)
        np.savez_compressed(test_path, x_test=x_test, y_test=y_test)

        print(f"[fold{our_fold}] train: {x_train.shape}, test: {x_test.shape} — saved to {output_dir}")

    # Also save combined tensor for imputer training
    all_tensor = torch.from_numpy(all_specs_arr)  # (2000, 128, 1, 1024)
    imputer_path = output_dir / "all_train_for_imputer.pt"
    torch.save(all_tensor, imputer_path)
    print(f"[esc50] imputer tensor saved: {all_tensor.shape} → {imputer_path}")

    print("[esc50] preprocessing complete!")


if __name__ == "__main__":
    main()
