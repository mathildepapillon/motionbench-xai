"""Shared pytest fixtures for motionbench tests.

All tests use deterministic seeds via ``torch.manual_seed`` / ``np.random.seed``.
The canonical test shapes are:

    J=5, F=3, T=16, M=4

These are small enough that tests run fast on CPU and large enough to catch
shape-regression bugs.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


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
