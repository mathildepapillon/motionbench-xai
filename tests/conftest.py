"""Shared pytest fixtures for motionbench tests.

All tests use deterministic seeds via ``torch.manual_seed`` / ``np.random.seed``.
The canonical test shapes are:

    J=5, F=3, T=16, M=4

These are small enough that tests run fast on CPU and large enough to catch
shape-regression bugs.

Cross-worktree imports
----------------------
The 2E worktree depends on modules from 2A (imputers) and 1A (oracles, utils).
Because all worktrees share the same ``motionbench-xai`` package name, only the
last ``pip install -e`` registers its source tree with the editable finder.
The block below extends each relevant subpackage's ``__path__`` so that
Python finds modules from sibling worktrees without reinstalling.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Extend motionbench subpackage __path__ to include sibling worktrees.
# This lets `from motionbench.imputers.off_manifold import ZeroImputer`
# and similar cross-worktree imports work regardless of install order.
# ---------------------------------------------------------------------------

_WORKSPACE = Path(__file__).parent.parent.parent  # /home/papillon/code

_CROSS_WORKTREE_PATHS: dict[str, list[str]] = {
    "motionbench.imputers": [
        str(_WORKSPACE / "mbxai-task-2A-offmanifold" / "motionbench" / "imputers"),
    ],
    "motionbench.oracles": [
        str(_WORKSPACE / "mbxai-task-1A-gaussian" / "motionbench" / "oracles"),
    ],
    "motionbench.utils": [
        str(_WORKSPACE / "mbxai-task-1A-gaussian" / "motionbench" / "utils"),
    ],
}


def _extend_cross_worktree_paths() -> None:
    """Extend motionbench subpackage __path__ with sibling worktree sources."""
    import importlib

    for pkg_name, extra_paths in _CROSS_WORKTREE_PATHS.items():
        # Ensure the package is importable (it must already exist in 2E)
        try:
            mod = importlib.import_module(pkg_name)
        except ImportError:
            continue
        for ep in extra_paths:
            if Path(ep).is_dir() and ep not in mod.__path__:  # type: ignore[union-attr]
                mod.__path__.append(ep)  # type: ignore[union-attr]


_extend_cross_worktree_paths()


# ---------------------------------------------------------------------------
# Global seeds
# ---------------------------------------------------------------------------

SEED = 42
J, F, T, M = 5, 3, 16, 4


@pytest.fixture(autouse=True)
def _set_seed() -> None:
    """Fix random seeds for every test."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Shape fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def shape() -> tuple[int, int, int]:
    """Canonical (J, F, T) shape."""
    return J, F, T


@pytest.fixture()
def n_players() -> int:
    """Canonical number of players M."""
    return M


@pytest.fixture()
def x_sample(shape: tuple[int, int, int]) -> torch.Tensor:
    """A single ``(J, F, T)`` float32 sample."""
    j, f, t = shape
    return torch.randn(j, f, t)


@pytest.fixture()
def mask_half(shape: tuple[int, int, int]) -> torch.Tensor:
    """An element mask with the first T//2 time-steps observed."""
    j, f, t = shape
    mask = torch.zeros(j, f, t, dtype=torch.bool)
    mask[:, :, : t // 2] = True
    return mask


@pytest.fixture()
def classifier_fn(shape: tuple[int, int, int]) -> object:
    """Toy classifier that returns mean of input (B, J, F, T) → (B,)."""

    def _clf(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(1, 2, 3))

    return _clf
