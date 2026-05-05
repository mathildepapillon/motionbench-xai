"""motionbench.classifiers.ported_care_pd.bilstm — Bidirectional LSTM encoder.

Port of ``CARE-PD/model/bilstm/bilstm_encoder.py`` adapted for the motionbench
``(B, J, F, T)`` input convention.

Architecture overview
---------------------
The BiLSTM encoder is a simple bidirectional LSTM baseline that processes
flattened per-frame joint coordinates ``(B, T, J*F)`` and produces a
mean-pooled representation ``(B, 2*hidden_size)`` for classification.

Unlike the transformer-based encoders, the BiLSTM has no publicly available
pre-trained CARE-PD checkpoint.  The backbone_loader.py in CARE-PD returns a
freshly-initialised model for this architecture.  Accordingly
``checkpoint_path`` defaults to ``None`` and the model is always randomly
initialised (no pre-trained weights are loaded).

Shape convention
----------------
Input ``x``:   ``(B, J, F=3, T)``
Output logits: ``(B, n_classes)``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from torch import Tensor

from motionbench.classifiers.base import Classifier

logger = logging.getLogger(__name__)

__all__ = ["BiLSTMClassifier"]


class _BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder.

    Adapted from CARE-PD/model/bilstm/bilstm_encoder.py.

    Args:
        input_dim: Input feature dimension per time step (J * F).
        hidden_size: Number of hidden units per direction.
        num_layers: Number of LSTM layers.
        dropout: Dropout probability between LSTM layers.
    """

    def __init__(
        self,
        input_dim: int = 51,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: ``(B, T, input_dim)``
        Returns:
            ``(B, T, 2 * hidden_size)``
        """
        out, _ = self.bilstm(x)
        return out


class BiLSTMClassifier(Classifier):
    """Bidirectional LSTM encoder with a linear classification head.

    Wraps :class:`_BiLSTMEncoder` and exposes the standard motionbench
    classifier interface ``(B, J, F, T) → (B, n_classes)``.

    Args:
        checkpoint_path: Not used for BiLSTM (no pre-trained weights
            available in CARE-PD).  Accepted for API consistency; always
            ignored.
        n_classes: Number of output logit dimensions (default 4).
        hidden_size: Number of LSTM hidden units per direction (default 256).
        num_layers: Number of LSTM layers (default 2).
        dropout: Dropout probability (default 0.1).
        num_joints: Number of skeletal joints (default 17).
        n_coords: Number of coordinates per joint (default 3 for xyz).

    Note:
        The CARE-PD BiLSTM is always freshly initialised — no pre-trained
        backbone checkpoint exists for this architecture.  The
        ``checkpoint_path`` argument is accepted for interface uniformity
        but no weights are loaded from it.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path, None] = None,
        n_classes: int = 4,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_joints: int = 17,
        n_coords: int = 3,
    ) -> None:
        super().__init__(checkpoint_path=None, n_classes=n_classes)

        if checkpoint_path is not None:
            logger.warning(
                "BiLSTMClassifier: checkpoint_path=%s was provided but BiLSTM "
                "has no pre-trained CARE-PD backbone; ignoring.",
                checkpoint_path,
            )

        input_dim = num_joints * n_coords  # 51 by default
        encoder_dim = 2 * hidden_size     # 512 by default

        self.backbone = _BiLSTMEncoder(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.cls_head = nn.Linear(encoder_dim, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Map a batch of 3D pose sequences to class logits.

        Args:
            x: Float32 tensor of shape ``(B, J, F=3, T)``.

        Returns:
            Float32 tensor of shape ``(B, n_classes)`` — raw logits.
        """
        B, J, F, T = x.shape
        # (B, J, F=3, T) → (B, T, J*F=51)
        x = x.permute(0, 3, 1, 2).reshape(B, T, J * F)
        out = self.backbone(x)    # (B, T, 2 * hidden_size)
        rep = out.mean(dim=1)     # (B, 2 * hidden_size)
        return self.cls_head(rep)
