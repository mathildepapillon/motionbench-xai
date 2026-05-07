"""motionbench.data.real.ptbxl — PTB-XL 12-lead ECG dataset loader.

Loads the PTB-XL dataset (Wagner et al. 2020, Scientific Data) and exposes
it as a binary classification problem:

    NORM  (class 0) — Normal ECG
    MI    (class 1) — Myocardial Infarction

The dataset returns ``(J=12, F=1, T=1000)`` float32 tensors (12 ECG leads ×
1 voltage channel × 1000 time-steps at 100 Hz = 10 s) paired with integer
binary labels.

Fold convention
---------------
PTB-XL ships with an official 10-fold stratified split (``strat_fold`` column
in ``ptbxl_database.csv``).  We use the canonical Strodthoff et al. split:

    train  = strat_fold ∈ {1, …, 8}   (held-in)
    val    = strat_fold == 9           (validation / hyper-param tuning)
    test   = strat_fold == 10          (held-out test)

For SHAP evaluation we expose ``split="test"`` (fold 10) as the primary
evaluation partition (``split="val"`` for intermediate evaluation).

Manifold note
-------------
The 12 standard ECG leads are not independent: Leads I, II, III satisfy
Einthoven's triangle exactly, and the augmented leads aVR, aVL, aVF are
algebraically derived from the limb leads.  Under the dipole approximation
all 12 leads are projections of a 3-dimensional cardiac electrical vector,
giving intrinsic dimensionality ≪ 12 × T.  This makes PTB-XL an ideal
real-world testbed for manifold-aware imputers.

References
----------
* Wagner et al. (2020). *PTB-XL, a large publicly available
  electrocardiography dataset.* Scientific Data, 7, 154.
  https://doi.org/10.1038/s41597-020-0495-6

* Goldberger et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet.*
  Circulation, 101(23), e215–e220.
  https://doi.org/10.1161/01.CIR.101.23.e215

* Strodthoff et al. (2021). *Deep Learning for ECG Analysis: Benchmarks and
  Insights from PTB-XL.* IEEE Journal of Biomedical and Health Informatics,
  25(5), 1519–1528. https://doi.org/10.1109/jbhi.2020.3022989

Example
-------
>>> ds = PTBXLDataset(data_path="/data/ptb-xl", split="test")
>>> x, y = ds[0]
>>> x.shape
torch.Size([12, 1, 1000])
>>> y.item()
0
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch import Tensor

if TYPE_CHECKING:
    from motionbench.oracles.base import Oracle

logger = logging.getLogger(__name__)

__all__ = ["PTBXLDataset"]

# Sampling rate for the low-resolution records (100 Hz)
_SAMPLING_RATE: int = 100
# Sequence length: 10 s × 100 Hz = 1000 samples
_SEQ_LEN: int = 1000
# Number of ECG leads in the 12-lead standard
_N_LEADS: int = 12

# Fold partitions following Strodthoff et al. (2021)
_FOLD_SPLITS: dict[str, tuple[list[int], ...]] = {
    # (train_folds, eval_folds)
    "train": (list(range(1, 9)),),    # folds 1–8
    "val":   ([9],),                  # fold 9 (validation)
    "test":  ([10],),                 # fold 10 (held-out test)
}

# Superclass label mapping used in this binary task
_LABEL_MAP: dict[str, int] = {"NORM": 0, "MI": 1}


def _parse_scp_codes(raw: str) -> dict[str, float]:
    """Safely parse the ``scp_codes`` column string to a Python dict.

    PTB-XL stores SCP codes as string representations of Python dicts, e.g.
    ``"{'SR': 0.0, 'NORM': 100.0}"``.

    Args:
        raw: Raw string from the ``scp_codes`` CSV column.

    Returns:
        Dict mapping SCP statement code → confidence (0–100).  Returns an
        empty dict on parse failure.
    """
    try:
        return ast.literal_eval(str(raw))
    except (ValueError, SyntaxError):
        return {}


def _dominant_superclass(
    scp_codes: dict[str, float],
    code_to_super: dict[str, str],
) -> str | None:
    """Return the highest-confidence diagnostic superclass for a record.

    Args:
        scp_codes: Dict of ``{scp_code: confidence}`` for one record.
        code_to_super: Mapping from SCP code to diagnostic superclass
            (built from ``scp_statements.csv``).

    Returns:
        The superclass label string (e.g. ``"NORM"``, ``"MI"``), or
        ``None`` if no diagnostic superclass is found.
    """
    best_super: str | None = None
    best_conf: float = -1.0
    for code, conf in scp_codes.items():
        sup = code_to_super.get(code)
        if sup and conf > best_conf:
            best_conf = conf
            best_super = sup
    return best_super


class PTBXLDataset:
    """PTB-XL 12-lead ECG dataset, binary NORM vs. MI classification.

    Satisfies the :class:`motionbench.data.base.BaseDataset` structural
    protocol.  No training augmentations are applied.

    Args:
        data_path: Root directory of the downloaded PTB-XL dataset.  Must
            contain ``ptbxl_database.csv``, ``scp_statements.csv``, and the
            ``records100/`` subdirectory.
        split: One of ``"train"``, ``"val"``, or ``"test"``.  Determines
            which ``strat_fold`` values are included.
        normalize: If ``True`` (default), apply per-lead z-score
            normalisation using statistics computed over the *train* split of
            this ``data_path``.  Pass ``False`` to get raw μV signals.
        max_sequences: Optional cap on number of returned sequences.
        train_stats: Pre-computed ``(mean, std)`` arrays of shape
            ``(J=12,)`` used for normalisation.  When ``None`` and
            ``normalize=True``, statistics are computed from the train split
            of this dataset instance (adds ~5 s to construction).

    Shape
    -----
    ``x`` : ``(J=12, F=1, T=1000)`` float32.
    ``y`` : scalar int64 — ``0`` for NORM, ``1`` for MI.

    Raises:
        FileNotFoundError: If ``data_path`` is not a valid PTB-XL root.
        RuntimeError: If no NORM or MI records are found.
    """

    def __init__(
        self,
        data_path: str | Path,
        split: str = "test",
        normalize: bool = True,
        max_sequences: int | None = None,
        train_stats: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        if split not in _FOLD_SPLITS:
            raise ValueError(f"split must be one of {list(_FOLD_SPLITS)}; got {split!r}.")

        self._data_path = Path(data_path)
        self._split = split
        self._normalize = normalize

        db_path = self._data_path / "ptbxl_database.csv"
        scp_path = self._data_path / "scp_statements.csv"
        if not db_path.exists():
            raise FileNotFoundError(f"ptbxl_database.csv not found in {self._data_path}")
        if not scp_path.exists():
            raise FileNotFoundError(f"scp_statements.csv not found in {self._data_path}")

        # Build SCP-code → diagnostic superclass mapping
        scp_df = pd.read_csv(scp_path, index_col=0)
        # PTB-XL uses 'diagnostic_class' for the 5-way superclass label
        # (NORM, MI, STTC, CD, HYP).
        diagnostic_col = (
            "diagnostic_class"
            if "diagnostic_class" in scp_df.columns
            else next(
                (c for c in scp_df.columns if "class" in c.lower() or "superclass" in c.lower()),
                scp_df.columns[0],
            )
        )
        code_to_super: dict[str, str] = {
            idx: str(row[diagnostic_col])
            for idx, row in scp_df.iterrows()
            if pd.notna(row[diagnostic_col]) and str(row[diagnostic_col]).strip()
        }

        # Load and filter the database CSV
        db = pd.read_csv(db_path)
        db["scp_dict"] = db["scp_codes"].map(_parse_scp_codes)
        db["superclass"] = db["scp_dict"].map(
            lambda d: _dominant_superclass(d, code_to_super)
        )

        # Keep only NORM and MI records
        db = db[db["superclass"].isin(_LABEL_MAP)].copy()

        # Filter to the requested fold(s)
        fold_values = _FOLD_SPLITS[split][0]
        db = db[db["strat_fold"].isin(fold_values)].copy()

        if db.empty:
            raise RuntimeError(
                f"No NORM/MI records found for split={split!r} in {self._data_path}."
            )

        if max_sequences is not None:
            db = db.head(int(max_sequences))

        # Compute or accept normalisation statistics
        if normalize:
            if train_stats is not None:
                self._mean, self._std = train_stats
            else:
                self._mean, self._std = self._compute_train_stats(code_to_super)
        else:
            self._mean = np.zeros(_N_LEADS, dtype=np.float32)
            self._std  = np.ones(_N_LEADS,  dtype=np.float32)

        # Load all waveforms into memory
        self._samples: list[tuple[np.ndarray, int]] = self._load_records(db)

        if not self._samples:
            raise RuntimeError("No waveforms could be loaded from the dataset.")

        logger.info(
            "PTBXLDataset[%s]: %d records (%d NORM, %d MI)",
            split,
            len(self._samples),
            sum(1 for _, y in self._samples if y == 0),
            sum(1 for _, y in self._samples if y == 1),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_waveform(self, filename_lr: str) -> np.ndarray | None:
        """Load a single WFDB record and return a ``(T, J)`` float32 array.

        Args:
            filename_lr: Relative filename from ``ptbxl_database.csv``
                (e.g. ``"records100/00000/00001_lr"``).

        Returns:
            ``(T=1000, J=12)`` float32 array, or ``None`` on load failure.
        """
        try:
            import wfdb  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "PTBXLDataset requires the 'wfdb' package: pip install wfdb"
            ) from e

        record_path = str(self._data_path / filename_lr)
        try:
            record = wfdb.rdsamp(record_path)
            signal = record[0].astype(np.float32)  # (T, 12)
        except Exception as exc:
            logger.debug("Failed to load %s: %s", record_path, exc)
            return None

        T, J = signal.shape
        if J != _N_LEADS:
            logger.debug("Unexpected lead count %d in %s; skipping.", J, record_path)
            return None

        # Clip or zero-pad to exactly _SEQ_LEN
        if T >= _SEQ_LEN:
            signal = signal[:_SEQ_LEN]
        else:
            padded = np.zeros((_SEQ_LEN, _N_LEADS), dtype=np.float32)
            padded[:T] = signal
            signal = padded

        return signal  # (1000, 12)

    def _load_records(
        self, db: "pd.DataFrame"
    ) -> list[tuple[np.ndarray, int]]:
        """Load all waveforms in *db* and return ``(waveform, label)`` pairs.

        Args:
            db: Filtered rows of the PTB-XL database CSV.

        Returns:
            List of ``(signal, label)`` where ``signal`` is ``(T, J)``
            float32 and ``label`` is 0 (NORM) or 1 (MI).
        """
        samples: list[tuple[np.ndarray, int]] = []
        for _, row in db.iterrows():
            signal = self._load_waveform(row["filename_lr"])
            if signal is None:
                continue
            # Per-lead z-score: (T, J) → normalise along T (subtract mean, divide std)
            # Mean and std are per-lead scalars computed over the train set.
            signal = (signal - self._mean[np.newaxis, :]) / (
                self._std[np.newaxis, :] + 1e-8
            )
            label = _LABEL_MAP[row["superclass"]]
            samples.append((signal.astype(np.float32), label))
        return samples

    def _compute_train_stats(
        self, code_to_super: dict[str, str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-lead mean and std over the train split.

        Args:
            code_to_super: SCP-code → superclass mapping.

        Returns:
            ``(mean, std)`` each of shape ``(J=12,)``.
        """
        logger.info(
            "PTBXLDataset: computing normalisation statistics from train split …"
        )
        try:
            db = pd.read_csv(self._data_path / "ptbxl_database.csv")
            db["scp_dict"] = db["scp_codes"].map(_parse_scp_codes)
            db["superclass"] = db["scp_dict"].map(
                lambda d: _dominant_superclass(d, code_to_super)
            )
            db = db[
                db["superclass"].isin(_LABEL_MAP)
                & db["strat_fold"].isin(_FOLD_SPLITS["train"][0])
            ]
        except Exception:
            return (
                np.zeros(_N_LEADS, dtype=np.float32),
                np.ones(_N_LEADS, dtype=np.float32),
            )

        accum: list[np.ndarray] = []
        for _, row in db.iterrows():
            sig = self._load_waveform(row["filename_lr"])
            if sig is not None:
                accum.append(sig)
            if len(accum) >= 2000:
                break   # cap at 2000 records for speed

        if not accum:
            return (
                np.zeros(_N_LEADS, dtype=np.float32),
                np.ones(_N_LEADS, dtype=np.float32),
            )

        all_signals = np.stack(accum, axis=0)   # (N, T, J)
        flat = all_signals.reshape(-1, _N_LEADS)  # (N*T, J)
        mean = flat.mean(axis=0).astype(np.float32)
        std  = flat.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0   # avoid division by zero for flat leads
        logger.info(
            "PTBXLDataset: stats from %d train records — mean range [%.3f, %.3f]",
            len(accum), float(mean.min()), float(mean.max()),
        )
        return mean, std

    # ------------------------------------------------------------------
    # BaseDataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of samples in this split."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return ``(x, y)`` for sample at index *idx*.

        Args:
            idx: Sample index in ``[0, len(self))``.

        Returns:
            ``x``: Float32 tensor of shape ``(J=12, F=1, T=1000)``.
            ``y``: Int64 scalar tensor — ``0`` (NORM) or ``1`` (MI).
        """
        signal, label = self._samples[idx]
        # signal: (T=1000, J=12) → permute to (J=12, T=1000) → unsqueeze F → (J, F=1, T)
        x = torch.from_numpy(signal).T.unsqueeze(1)  # (12, 1, 1000)
        y = torch.tensor(label, dtype=torch.int64)
        return x, y

    @property
    def shape(self) -> tuple[int, int, int]:
        """Spatial shape of every sample as ``(J, F, T)``."""
        return (_N_LEADS, 1, _SEQ_LEN)

    @property
    def metadata(self) -> dict[str, object]:
        """Dataset-level metadata."""
        return {
            "domain": "ecg",
            "dataset": "ptb-xl",
            "sampling_rate_hz": _SAMPLING_RATE,
            "seq_len": _SEQ_LEN,
            "n_leads": _N_LEADS,
            "n_classes": 2,
            "classes": ["NORM", "MI"],
            "split": self._split,
            "n_sequences": len(self._samples),
            "normalize": self._normalize,
        }

    @property
    def train_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-lead normalisation statistics ``(mean, std)``, shape ``(12,)`` each."""
        return self._mean, self._std

    @property
    def oracle(self) -> "Oracle | None":
        """Always ``None`` — real ECG data has no closed-form oracle."""
        return None
