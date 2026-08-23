"""motionbench.utils.seeding — Deterministic seed helper.

Provides ``seed_everything``, which locks every random source (Python,
NumPy, PyTorch CPU/CUDA) and enables cuDNN determinism.  The release
trainers use their own narrower seeding (torch/numpy/random only): the
shipped checkpoints were produced without cuDNN determinism flags, so this
helper must not be retro-wired into them.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["seed_everything"]


def seed_everything(seed: int = 42) -> None:
    """Fix all random seeds and enable cuDNN determinism.

    Sets ``PYTHONHASHSEED`` and seeds Python ``random``, NumPy, PyTorch CPU
    and (if available) PyTorch CUDA; on CUDA it also flips cuDNN to
    deterministic, non-benchmarking mode.  Note the cuDNN flags change
    kernel selection: the release trainers deliberately seed only
    torch/numpy/random, so do not add this helper to them.

    Args:
        seed: Integer seed value.  Default ``42``.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
