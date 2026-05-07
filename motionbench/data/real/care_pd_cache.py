"""motionbench.data.real.care_pd_cache — BMCLab CARE-PD dataset from preprocessed cache.

Loads the same ``cache.npz`` files that were used to train the CARE-PD VAEAC
and Flow Matching imputers, guaranteeing that evaluation samples follow the
distribution the imputers were trained on.

This adapter is preferred over :class:`~motionbench.data.real.care_pd.BMCLabDataset`
for our SHAP evaluation because it sidesteps separate per-encoder preprocessing
pipelines (motionbert / motionagformer / bilstm each have their own normalisation
in the original CARE-PD code).

Cache format (from CARE-PD/scripts/build_flow_cache.py)::

    x1_train             (N_tr, T, J, 3) float32   pose coordinates
    mask_train           (N_tr, T)        bool      valid-frame mask
    x1_val               (N_va, T, J, 3) float32
    mask_val             (N_va, T)        bool
    meta_updrs_gait_val  (N_va,)         int       UPDRS-gait label (0..3, -1 missing)
    seq_len, fold, num_folds, ...

The dataset returns ``(J=17, F=3, T=80)`` float32 tensors paired with integer
UPDRS-gait labels.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = ["BMCLabCacheDataset"]


class BMCLabCacheDataset:
    """BMCLab data loaded directly from a CARE-PD imputer cache.npz file.

    Args:
        cache_path: Path to ``cache.npz`` (e.g.
            ``CARE-PD/cache/flow_matching/BMCLab_h36m_80_fold1/cache.npz``).
        split: ``"val"`` (default; what we evaluate SHAP on) or ``"train"``.
        max_sequences: Optional cap on number of returned sequences (head).
        drop_missing_labels: If True, exclude samples with UPDRS label < 0.

    Returns shape: ``(J=17, F=3, T=80)`` per sample.
    """

    def __init__(
        self,
        cache_path: str | Path,
        split: str = "val",
        max_sequences: int | None = None,
        drop_missing_labels: bool = True,
    ) -> None:
        self._cache_path = Path(cache_path)
        if not self._cache_path.exists():
            raise FileNotFoundError(f"BMCLab cache not found: {self._cache_path}")

        d = np.load(self._cache_path, allow_pickle=True)
        if split == "val":
            x = d["x1_val"]
            y = d["meta_updrs_gait_val"]
        elif split == "train":
            x = d["x1_train"]
            y = d["meta_updrs_gait_train"]
        else:
            raise ValueError(f"split must be 'train' or 'val'; got {split!r}.")

        # (N, T, J, 3) → (N, J, 3, T)
        x = np.transpose(x, (0, 2, 3, 1)).astype(np.float32)
        y = np.asarray(y, dtype=np.int64)

        if drop_missing_labels:
            keep = y >= 0
            x = x[keep]
            y = y[keep]

        if max_sequences is not None:
            x = x[: int(max_sequences)]
            y = y[: int(max_sequences)]

        self._x: Tensor = torch.from_numpy(x)
        self._y: Tensor = torch.from_numpy(y)
        self._N, self._J, self._F, self._T = x.shape

        logger.info(
            "BMCLabCacheDataset: %d sequences (J=%d, F=%d, T=%d) from %s split=%s",
            self._N, self._J, self._F, self._T, self._cache_path, split,
        )

    def __len__(self) -> int:
        return self._N

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self._x[idx], self._y[idx]

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self._J, self._F, self._T)

    @property
    def metadata(self) -> dict[str, object]:
        # CARE-PD BMCLab fold1 contains UPDRS-gait labels {0, 1, 2}; no 3 in this fold.
        return {
            "skeleton": "h36m_17",
            "frame_rate": 27.0,
            "n_classes": 3,
            "split_source": str(self._cache_path),
        }
