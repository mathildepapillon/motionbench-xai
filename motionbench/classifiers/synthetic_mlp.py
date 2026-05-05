"""motionbench.classifiers.synthetic_mlp — MLP classifier for synthetic motion benchmarks.

Ported and generalised from CARE-PD/synthetic/gaussian_motion.py.
Supports two feature-extraction modes:
  - 'temporal': K per-window grand means → input_dim=K
  - 'spatial':  J per-joint grand means  → input_dim=J

References:
    Olsen et al. (JMLR 2022) — temporal feature aggregation design.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from motionbench.classifiers.base import Classifier

PlayerMode = Literal["temporal", "spatial"]


class SyntheticMLPClassifier(Classifier):
    """MLP classifier for synthetic motion benchmarks.

    Two feature-extraction modes:

    - ``'temporal'``: K per-window grand means → input_dim=K. Mirrors the
      Olsen et al. (2022) aggregation: the classifier sees the same K aggregates
      the nonlinear label depends on, making it a genuine black-box for SHAP.
    - ``'spatial'``: J per-joint grand means → input_dim=J. The classifier
      sees every joint individually; the label depends on 4 signal joints but
      the classifier does not know which.

    Architecture::

        input → BatchNorm1d → Linear(hidden) → ReLU → Linear(hidden) → ReLU → Linear(n_classes)

    Args:
        J: Number of joints.
        F: Features per joint.
        T: Frames per clip.
        K: Number of temporal windows (only used in temporal mode).
        n_classes: Number of output classes.
        hidden: Hidden layer width.
        player_mode: ``'temporal'`` or ``'spatial'``.

    Raises:
        ValueError: If ``player_mode`` is not ``'temporal'`` or ``'spatial'``.
    """

    def __init__(
        self,
        J: int = 17,
        F: int = 3,
        T: int = 81,
        K: int = 4,
        n_classes: int = 3,
        hidden: int = 64,
        player_mode: PlayerMode = "temporal",
    ) -> None:
        super().__init__()
        if player_mode not in ("temporal", "spatial"):
            raise ValueError(
                f"player_mode must be 'temporal' or 'spatial'; got {player_mode!r}"
            )
        self.J = J
        self.F = F
        self.T = T
        self.K = K
        self.n_classes = n_classes
        self.player_mode: PlayerMode = player_mode

        if player_mode == "temporal":
            quarter = T // K
            self.window_starts: list[int] = [k * quarter for k in range(K)]
            self.window_ends: list[int] = [
                (k + 1) * quarter if k < K - 1 else T for k in range(K)
            ]
            input_dim = K
        else:
            self.window_starts = []
            self.window_ends = []
            input_dim = J

        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def _extract_features(self, x: Tensor) -> Tensor:
        """Extract fixed-size features from a (B, J, F, T) input tensor.

        Args:
            x: (B, J, F, T) float32 tensor.

        Returns:
            (B, K) grand means in temporal mode, or (B, J) grand means in spatial mode.
        """
        if self.player_mode == "temporal":
            feats = [
                x[:, :, :, s:e].mean(dim=(1, 2, 3))
                for s, e in zip(self.window_starts, self.window_ends, strict=True)
            ]
            return torch.stack(feats, dim=1)  # (B, K)
        return x.mean(dim=(2, 3))  # (B, J)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (B, J, F, T) float32 tensor.

        Returns:
            (B, n_classes) float32 logit tensor.
        """
        return self.net(self._extract_features(x))
