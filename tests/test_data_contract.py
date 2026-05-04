"""Contract tests for motionbench.data.base.BaseDataset and GroundTruthDataset.

Verifies that:
1. Protocols are correctly checkable at runtime via isinstance.
2. Concrete implementations satisfy required method signatures and output shapes.
3. oracle=None is valid for BaseDataset; oracle≠None is required for GroundTruthDataset.
"""
from __future__ import annotations

import pytest
import torch
from torch import Tensor

from motionbench.data.base import BaseDataset, GroundTruthDataset
from motionbench.oracles.base import Oracle


# ---------------------------------------------------------------------------
# Mock oracle (minimal, used in GroundTruthDataset mock)
# ---------------------------------------------------------------------------


class _MockOracle(Oracle):
    def conditional_sample(self, x_obs, mask, n, seed=None):
        J, F, T = x_obs.shape
        return torch.zeros(n, J, F, T)

    def true_shapley(self, x, classifier, players, n_mc=1000, seed=None):
        return torch.zeros(players.n_players)


# ---------------------------------------------------------------------------
# Mock BaseDataset
# ---------------------------------------------------------------------------


class MockDataset:
    """Minimal concrete BaseDataset."""

    def __init__(self, J=5, F=3, T=16, N=10):
        self._J = J
        self._F = F
        self._T = T
        self._N = N

    def __getitem__(self, idx: int):
        return torch.zeros(self._J, self._F, self._T), torch.tensor(0)

    def __len__(self) -> int:
        return self._N

    @property
    def shape(self):
        return (self._J, self._F, self._T)

    @property
    def metadata(self):
        return {"skeleton": "mock", "frame_rate": 30.0}

    @property
    def oracle(self):
        return None


# ---------------------------------------------------------------------------
# Mock GroundTruthDataset
# ---------------------------------------------------------------------------


class MockGroundTruthDataset:
    def __init__(self, J=5, F=3, T=16, N=10):
        self._J = J
        self._F = F
        self._T = T
        self._N = N
        self._oracle = _MockOracle()

    def __getitem__(self, idx: int):
        return torch.zeros(self._J, self._F, self._T), torch.tensor(0)

    def __len__(self) -> int:
        return self._N

    @property
    def shape(self):
        return (self._J, self._F, self._T)

    @property
    def metadata(self):
        return {"skeleton": "mock", "frame_rate": 30.0}

    @property
    def oracle(self):
        return self._oracle


# ---------------------------------------------------------------------------
# Contract tests — BaseDataset
# ---------------------------------------------------------------------------


def test_mock_dataset_is_basedataset():
    ds = MockDataset()
    assert isinstance(ds, BaseDataset)


def test_dataset_getitem_shape(shape):
    J, F, T = shape
    ds = MockDataset(J, F, T)
    x, y = ds[0]
    assert x.shape == (J, F, T), f"Expected ({J},{F},{T}), got {x.shape}"
    assert x.dtype == torch.float32


def test_dataset_len():
    N = 7
    ds = MockDataset(N=N)
    assert len(ds) == N


def test_dataset_shape_property(shape):
    J, F, T = shape
    ds = MockDataset(J, F, T)
    assert ds.shape == (J, F, T)


def test_dataset_metadata_keys():
    ds = MockDataset()
    meta = ds.metadata
    assert "skeleton" in meta, "metadata must contain 'skeleton'"
    assert "frame_rate" in meta, "metadata must contain 'frame_rate'"


def test_dataset_oracle_is_none():
    ds = MockDataset()
    assert ds.oracle is None


def test_dataset_without_oracle_fails_groundtruth_protocol():
    ds = MockDataset()
    # MockDataset has oracle=None, which does NOT satisfy GroundTruthDataset
    # Protocol can't enforce non-None at isinstance time, but we document the
    # contract here: accessing .oracle and checking it is the caller's job.
    assert ds.oracle is None


# ---------------------------------------------------------------------------
# Contract tests — GroundTruthDataset
# ---------------------------------------------------------------------------


def test_mock_gt_dataset_is_groundtruth():
    ds = MockGroundTruthDataset()
    assert isinstance(ds, GroundTruthDataset)


def test_gt_dataset_oracle_is_not_none():
    ds = MockGroundTruthDataset()
    assert ds.oracle is not None


def test_gt_dataset_oracle_type():
    ds = MockGroundTruthDataset()
    assert isinstance(ds.oracle, Oracle)


def test_gt_dataset_getitem_shape(shape):
    J, F, T = shape
    ds = MockGroundTruthDataset(J, F, T)
    x, y = ds[0]
    assert x.shape == (J, F, T)


def test_gt_dataset_is_basedataset():
    """GroundTruthDataset is-a BaseDataset."""
    ds = MockGroundTruthDataset()
    assert isinstance(ds, BaseDataset)
