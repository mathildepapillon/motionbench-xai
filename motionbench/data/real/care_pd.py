"""motionbench.data.real.care_pd — CARE-PD BMCLab gait dataset loader.

Inference-only port of ``CARE-PD/data/bmclab_datareader.py``.  All training
augmentations (random mirror, rotation, noise injection) are stripped.

The dataset returns ``(J=17, F=3, T)`` float32 tensors paired with integer
UPDRS-gait labels ``{0, 1, 2, 3}``.

Dataset format
--------------
Data is stored as NumPy ``.npz`` archives.  Each archive maps sequence names
to arrays of shape ``(N_views, T, J, 3)`` or ``(T, J, 3)`` (legacy 3-D).
Labels are stored in a ``joblib``-serialized dict keyed by
``subject_id → walk_id → {"UPDRS_GAIT": int, ...}``.

Metadata
--------
``skeleton``
    ``"h36m_17"`` — Human3.6M-17 joint topology.
``frame_rate``
    ``27.0`` Hz (BMCLab recording rate).

Example
-------
>>> ds = BMCLabDataset(joints_paths=["path/to/poses.npz"],
...                    labels_path="path/to/labels.pkl")
>>> x, y = ds[0]
>>> x.shape
torch.Size([17, 3, 81])
>>> y.item()
0

References
----------
* BMCLab dataset: https://neurips2025.care-pd.ca/
* CARE-PD paper: hal-05280110
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from motionbench.oracles.base import Oracle

logger = logging.getLogger(__name__)

__all__ = ["BMCLabDataset"]

_FRAME_RATE: float = 27.0
_SKELETON: str = "h36m_17"


class BMCLabDataset:
    """Inference-only CARE-PD BMCLab gait dataset.

    Satisfies the :class:`motionbench.data.base.BaseDataset` structural
    protocol.  No training augmentations are applied.

    Args:
        joints_paths: List of paths to ``.npz`` archives containing pose
            sequences.  Multiple paths are used for multi-view data; the
            first view of each sequence is taken.
        labels_path: Path to the joblib-serialized labels file (a nested
            dict keyed by ``subject_id → walk_id → {fields...}``).
        clip_len: Number of frames per clip ``T``.  Sequences shorter than
            ``clip_len`` are zero-padded on the right; longer sequences are
            centre-cropped.

    Raises:
        FileNotFoundError: If any of the provided paths do not exist.
        ValueError: If no valid sequences are found in the archives.
    """

    def __init__(
        self,
        joints_paths: Sequence[str | Path],
        labels_path: str | Path,
        clip_len: int = 81,
    ) -> None:
        self._clip_len = clip_len
        self._joints_paths = [Path(p) for p in joints_paths]
        self._labels_path = Path(labels_path)

        for p in self._joints_paths:
            if not p.exists():
                raise FileNotFoundError(f"Joints archive not found: {p}")
        if not self._labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self._labels_path}")

        self._label_df: dict[str, dict[str, dict[str, object]]] = joblib.load(
            self._labels_path
        )
        self._samples: list[tuple[np.ndarray[tuple[int, ...], np.dtype[np.float32]], int]] = (
            self._load_sequences()
        )

        if not self._samples:
            raise ValueError("No valid sequences found in the provided archives.")

        logger.info(
            "BMCLabDataset loaded %d sequences from %d archive(s).",
            len(self._samples),
            len(self._joints_paths),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_label(self, seq_name: str) -> int | None:
        """Return the UPDRS-gait label for *seq_name*, or ``None`` on error.

        Args:
            seq_name: Sequence name in format ``"subject_id__walk_id"``
                (optionally suffixed with ``_down...``).

        Returns:
            Integer UPDRS-gait score in ``{0, 1, 2, 3}``, or ``None`` if
            the key is missing.
        """
        try:
            parts = seq_name.split("__")
            subject_id = parts[0]
            walk_id = parts[1].split("_down")[0]
            return int(str(self._label_df[subject_id][walk_id]["UPDRS_GAIT"]))
        except (KeyError, IndexError, ValueError):
            return None

    def _load_sequences(
        self,
    ) -> list[tuple[np.ndarray[tuple[int, ...], np.dtype[np.float32]], int]]:
        """Read all sequences from the NPZ archives.

        Returns:
            List of ``(joints_array, label)`` tuples where
            ``joints_array`` has shape ``(T, J, 3)`` (raw, not yet
            clipped/padded to ``clip_len``).
        """
        samples: list[tuple[np.ndarray[tuple[int, ...], np.dtype[np.float32]], int]] = []
        for path in self._joints_paths:
            archive = np.load(path, allow_pickle=True)
            for seq_name in archive.files:
                if "trimmed" in seq_name.lower():
                    continue
                joints = archive[seq_name]
                # Some archives store (N_views, T, J, 3); take view 0.
                if joints.ndim == 4:
                    joints = joints[0]
                elif joints.ndim == 2:
                    # Legacy flat format: (T, J*3) — not expected for BMCLab
                    joints = joints.reshape(joints.shape[0], -1, 3)

                label = self._read_label(seq_name)
                if label is None:
                    logger.debug("No label found for sequence %s; skipping.", seq_name)
                    continue
                if joints is None or joints.size == 0:
                    logger.warning("Empty joint array for %s; skipping.", seq_name)
                    continue
                samples.append((joints.astype(np.float32), label))
        return samples

    def _clip_or_pad(
        self,
        joints: np.ndarray[tuple[int, ...], np.dtype[np.float32]],
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
        """Centre-crop or right-pad *joints* to ``clip_len`` frames.

        Args:
            joints: Array of shape ``(T_raw, J, 3)``.

        Returns:
            Array of shape ``(clip_len, J, 3)``.
        """
        T_raw, J, C = joints.shape
        T = self._clip_len
        if T_raw >= T:
            start = (T_raw - T) // 2
            return joints[start : start + T]
        # Pad with zeros on the right.
        padded = np.zeros((T, J, C), dtype=np.float32)
        padded[:T_raw] = joints
        return padded

    # ------------------------------------------------------------------
    # BaseDataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return ``(x, y)`` for sample at index *idx*.

        Args:
            idx: Sample index in ``[0, len(self))``.

        Returns:
            ``x``: Float32 tensor of shape ``(J=17, F=3, T=clip_len)``.
            ``y``: Int64 scalar tensor — UPDRS-gait class label.
        """
        joints_raw, label = self._samples[idx]
        joints = self._clip_or_pad(joints_raw)  # (T, J, 3)
        # Transpose to (J, F=3, T) layout.
        x = torch.from_numpy(joints).permute(1, 2, 0)  # (J, 3, T)
        y = torch.tensor(label, dtype=torch.int64)
        return x, y

    @property
    def shape(self) -> tuple[int, int, int]:
        """Spatial shape of every sample as ``(J, F, T)``."""
        return (17, 3, self._clip_len)

    @property
    def metadata(self) -> dict[str, object]:
        """Dataset-level metadata.

        Returns:
            Dict with at minimum ``"skeleton"`` and ``"frame_rate"`` keys.
        """
        return {
            "skeleton": _SKELETON,
            "frame_rate": _FRAME_RATE,
            "clip_len": self._clip_len,
            "n_sequences": len(self._samples),
        }

    @property
    def oracle(self) -> Oracle | None:
        """Always ``None`` — real data has no closed-form oracle."""
        return None
