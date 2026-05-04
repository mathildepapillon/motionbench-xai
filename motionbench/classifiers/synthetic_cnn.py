"""motionbench.classifiers.synthetic_cnn — Small 1D CNN classifier for synthetic motion sequences.

Architecture:
    Input (B, J, F, T) → reshape to (B, J*F, T)
    → Conv1d(J*F, 32, kernel_size=5, padding=2) → ReLU → BatchNorm1d
    → Conv1d(32, 32, kernel_size=5, padding=2) → ReLU → BatchNorm1d
    → Conv1d(32, 64, kernel_size=5, padding=2) → ReLU → BatchNorm1d
    → AdaptiveAvgPool1d(1) → squeeze → Linear(64, n_classes)
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from motionbench.classifiers.base import Classifier


class SyntheticCNNClassifier(Classifier):
    """Small 1D CNN classifier for synthetic motion sequences.

    Architecture::

        (B, J, F, T) → reshape (B, J*F, T)
        → Conv1d(J*F, 32, k=5) → ReLU → BN
        → Conv1d(32, 32, k=5) → ReLU → BN
        → Conv1d(32, 64, k=5) → ReLU → BN
        → AdaptiveAvgPool1d(1) → squeeze → Linear(64, n_classes)

    Args:
        J: Number of joints.
        F: Features per joint.
        n_classes: Number of output classes.
    """

    def __init__(
        self,
        J: int = 17,
        F: int = 3,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.J = J
        self.F = F
        self.n_classes = n_classes
        in_channels = J * F

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (B, J, F, T) float32 tensor.

        Returns:
            (B, n_classes) float32 logit tensor.
        """
        B, J, F, T = x.shape
        h = x.reshape(B, J * F, T)      # (B, J*F, T)
        h = self.conv_layers(h)          # (B, 64, T)
        h = self.pool(h).squeeze(-1)     # (B, 64)
        return self.classifier(h)        # (B, n_classes)
