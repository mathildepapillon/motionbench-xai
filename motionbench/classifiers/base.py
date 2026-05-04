"""motionbench.classifiers.base — Classifier abstract base class.

Uniform interface for all classifiers in the motionbench benchmark.
Classifiers receive (B, J, F, T) batches and return (B, n_classes) logits.

Shape convention:
    Input:  (B, J, F, T) float32 — batch of motion sequences.
    Output: (B, n_classes) float32 logits.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Classifier(ABC, torch.nn.Module):
    """Abstract base class for all motionbench classifiers.

    All classifiers follow PyTorch nn.Module conventions and additionally
    expose predict_proba for use in attribution pipelines.

    Shape convention:
        Input:  (B, J, F, T) float32 — batch of motion sequences.
        Output: (B, n_classes) float32 logits.
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (B, J, F, T) float32 tensor.

        Returns:
            (B, n_classes) float32 logit tensor.
        """

    def predict_proba(self, x: Tensor, class_idx: int = 0) -> Tensor:
        """Return class probability for class_idx.

        Args:
            x: (B, J, F, T) float32 tensor.
            class_idx: Class index to return probability for.

        Returns:
            (B,) float32 probability tensor.
        """
        return torch.softmax(self.forward(x), dim=-1)[:, class_idx]
