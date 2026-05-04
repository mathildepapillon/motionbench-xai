"""motionbench.classifiers.synthetic_transformer — Transformer encoder classifier for motion sequences.

Architecture:
    Input (B, J, F, T) → reshape to (B, T, J*F) → project to (B, T, d_model)
    → positional encoding (sinusoidal, fixed)
    → 4-layer TransformerEncoder (d_model=64, nhead=4, dim_feedforward=128, dropout=0.1)
    → mean pool over T → Linear(d_model, n_classes)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from motionbench.classifiers.base import Classifier


class _SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017).

    Args:
        d_model: Model embedding dimension.
        max_len: Maximum sequence length to pre-compute encodings for.
        dropout: Dropout probability applied after adding the encoding.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to x.

        Args:
            x: (B, T, d_model) float32 tensor.

        Returns:
            (B, T, d_model) float32 tensor with positional encoding added.
        """
        pe: Tensor = self.pe  # type: ignore[assignment]
        x = x + pe[: x.size(1)]
        return self.dropout(x)


class SyntheticTransformerClassifier(Classifier):
    """Small Transformer encoder classifier for synthetic motion sequences.

    Architecture::

        (B, J, F, T) → reshape (B, T, J*F) → Linear → (B, T, d_model)
        → sinusoidal positional encoding
        → 4-layer TransformerEncoder (d_model, nhead, dim_ff=128, dropout=0.1)
        → mean pool over T → Linear(d_model, n_classes)

    Args:
        J: Number of joints.
        F: Features per joint.
        n_classes: Number of output classes.
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
    """

    def __init__(
        self,
        J: int = 17,
        F: int = 3,
        n_classes: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.J = J
        self.F = F
        self.n_classes = n_classes
        self.d_model = d_model

        self.input_proj = nn.Linear(J * F, d_model)
        self.pos_enc = _SinusoidalPositionalEncoding(d_model, dropout=0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (B, J, F, T) float32 tensor.

        Returns:
            (B, n_classes) float32 logit tensor.
        """
        B, J, F, T = x.shape
        h = x.permute(0, 3, 1, 2).reshape(B, T, J * F)  # (B, T, J*F)
        h = self.input_proj(h)                            # (B, T, d_model)
        h = self.pos_enc(h)                               # (B, T, d_model)
        h = self.transformer(h)                           # (B, T, d_model)
        h = h.mean(dim=1)                                 # (B, d_model)
        return self.classifier(h)                         # (B, n_classes)
