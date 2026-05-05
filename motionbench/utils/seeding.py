"""motionbench.utils.seeding — Deterministic seed helpers.

All motionbench experiments must be reproducible.  Call ``seed_everything``
at the top of every script or pipeline to lock all random sources.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["seed_everything"]


def seed_everything(seed: int = 42) -> None:
    """Fix all random seeds for full reproducibility.

    Sets seeds for Python ``random``, NumPy, PyTorch CPU and (if available)
    PyTorch CUDA.  Also sets ``PYTHONHASHSEED`` for dict/set ordering
    reproducibility.

    Args:
        seed: Integer seed value.  Default ``42``.

    Example::

        from motionbench.utils.seeding import seed_everything
        seed_everything(0)
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
