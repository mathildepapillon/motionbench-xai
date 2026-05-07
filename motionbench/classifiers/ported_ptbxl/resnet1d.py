"""motionbench.classifiers.ported_ptbxl.resnet1d — 1D ResNet classifier for PTB-XL.

Architecture: ResNet for time series (``resnet1d_wang``), ported to PyTorch and
adapted to the motionbench ``(B, J, F, T)`` input contract.

Source architecture
-------------------
Wang, Z., Yan, W., & Oates, T. (2017). Time series classification from scratch
with deep neural networks: A strong baseline.  In *Proceedings of the 2017
International Joint Conference on Neural Networks (IJCNN)*, pp. 1578–1585.
https://doi.org/10.1109/ijcnn.2017.7966039

The architecture was benchmarked on PTB-XL in:
Strodthoff, N., Wagner, P., Schaeffter, T., & Samek, W. (2021). Deep learning
for ECG analysis: Benchmarks and insights from PTB-XL. *IEEE Journal of
Biomedical and Health Informatics*, 25(5), 1519–1528.
https://doi.org/10.1109/jbhi.2020.3022989

Reference PyTorch implementation:
https://github.com/helme/ecg_ptbxl_benchmarking (MIT licence)

Adaptation notes
----------------
* Input is ``(B, J=12, F=1, T=1000)`` following the motionbench convention.
  ``_preprocess`` squeezes the F dimension to produce ``(B, 12, 1000)`` for
  the 1-D convolutions.
* Three residual blocks of increasing filter counts (64 → 128 → 128) with
  kernel sizes 8 / 5 / 3, global average pooling, and a linear classifier head.
* Checkpoint is a plain ``state_dict`` saved by :func:`torch.save`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from motionbench.classifiers.base import Classifier

logger = logging.getLogger(__name__)

__all__ = ["ECGResNet1dClassifier"]

_N_LEADS: int = 12  # standard 12-lead ECG; must precede ECGResNet1dClassifier (used as default arg)

# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------
__credits__ = (
    "Architecture: Wang et al. (2017) IJCNN 'resnet1d_wang'; "
    "ECG benchmarking: Strodthoff et al. (2021) IEEE JBHI; "
    "Reference code: https://github.com/helme/ecg_ptbxl_benchmarking (MIT licence)."
)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ResBlock1d(nn.Module):
    """One residual block: three Conv1d layers + skip connection.

    Architecture (Wang et al. 2017):
        conv(out_ch, k=8) → BN → ReLU →
        conv(out_ch, k=5) → BN → ReLU →
        conv(out_ch, k=3) → BN
        shortcut: conv(out_ch, k=1) → BN  (if in_ch ≠ out_ch, else identity)
        output: sum → ReLU

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels,  out_channels, kernel_size=8, padding="same", bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding="same", bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding="same", bias=False)
        self.bn3   = nn.BatchNorm1d(out_channels)

        # Shortcut: 1×1 conv + BN when dimensions change, identity otherwise
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: ``(B, C_in, T)`` float32.

        Returns:
            ``(B, C_out, T)`` float32.
        """
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return F.relu(x + residual)


class _ResNet1dWang(nn.Module):
    """Backbone: three ``_ResBlock1d`` + global average pooling.

    Filter progression: in_channels → 64 → 128 → 128.

    Args:
        in_channels: Number of input channels (12 for standard 12-lead ECG).
    """

    def __init__(self, in_channels: int = 12) -> None:
        super().__init__()
        self.block1 = _ResBlock1d(in_channels, 64)
        self.block2 = _ResBlock1d(64, 128)
        self.block3 = _ResBlock1d(128, 128)
        self.out_dim = 128

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: ``(B, in_channels, T)`` float32.

        Returns:
            ``(B, 128)`` float32 embedding (after global average pooling).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x.mean(dim=-1)  # global average pooling over time


# ---------------------------------------------------------------------------
# Motionbench classifier wrapper
# ---------------------------------------------------------------------------

class ECGResNet1dClassifier(Classifier):
    """12-lead ECG classifier using the ``resnet1d_wang`` architecture.

    Accepts motionbench's ``(B, J=12, F=1, T=1000)`` raw tensor layout.
    ``_preprocess`` squeezes the ``F=1`` channel to produce ``(B, 12, 1000)``
    for the 1-D convolutions.

    Args:
        checkpoint_path: Path to a ``.pt`` state-dict checkpoint, or ``None``
            for random initialisation (for architecture tests only).
        n_classes: Number of output classes (default ``2``: NORM vs. MI).
        in_channels: Number of ECG leads (default ``12``).

    Input / output
    --------------
    ``forward(x)``
        x : ``(B, 12, 1, 1000)`` float32
        returns : ``(B, n_classes)`` float32 logits

    References
    ----------
    Wang et al. (2017) IJCNN — original architecture.
    Strodthoff et al. (2021) IEEE JBHI — ECG benchmarking on PTB-XL.
    """

    #: ECG voltage is 1D per lead; override the default F=3 from Classifier base.
    input_feature_dim: int = 1

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        n_classes: int = 2,
        in_channels: int = _N_LEADS,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, n_classes=n_classes)
        self.backbone = _ResNet1dWang(in_channels=in_channels)
        self.cls_head = nn.Linear(self.backbone.out_dim, n_classes)

        if checkpoint_path is not None:
            matched, discarded = self._load_checkpoint(
                checkpoint_path, self, strict=False
            )
            logger.info(
                "ECGResNet1dClassifier: loaded %d tensors, discarded %d",
                len(matched), len(discarded),
            )
        else:
            logger.info(
                "ECGResNet1dClassifier: random init (no checkpoint)."
            )

    def _preprocess(self, x: Tensor) -> Tensor:
        """Squeeze the F=1 channel to produce ``(B, J=12, T=1000)``.

        Args:
            x: Raw ``(B, J=12, F=1, T=1000)`` float32 tensor.

        Returns:
            ``(B, 12, 1000)`` float32.
        """
        # x: (B, J, F, T) with F=1 → squeeze dim 2 → (B, J, T)
        return x.squeeze(2)

    def forward(self, x: Tensor) -> Tensor:
        """Classify a batch of 12-lead ECG clips.

        Args:
            x: ``(B, J=12, F=1, T=1000)`` float32 in the motionbench layout.

        Returns:
            ``(B, n_classes)`` float32 raw logits.
        """
        x = self._preprocess(x)     # (B, 12, 1000)
        emb = self.backbone(x)       # (B, 128)
        return self.cls_head(emb)    # (B, n_classes)
