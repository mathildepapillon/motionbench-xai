"""Tests for motionbench.attribution temporal SHAP wrappers (Task 3C).

Covers:
- TimeSHAPAttributor  (timeshap.py)
- WindowSHAPAttributor (windowshap.py)
- ShaTSAttributor      (shats.py)
- GroupSegmentSHAPAttributor (group_segment_shap.py)
- Pure-math helpers in group_segment_shap.py

Fast (non-slow) tests verify instantiation, property contracts, and the
ShaTSAttributor NotImplementedError.  Full attribute() calls are marked
``@pytest.mark.slow`` because they invoke the underlying SHAP solvers.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from motionbench.attribution.group_segment_shap import (
    GroupSegmentSHAPAttributor,
    _shapley_from_v,
    direct_group_shapley,
    shapley_from_value_table,
)
from motionbench.attribution.shats import ShaTSAttributor
from motionbench.attribution.timeshap import TimeSHAPAttributor
from motionbench.attribution.windowshap import WindowSHAPAttributor
from motionbench.imputers.base import BaseImputer
from tests.conftest import F, J, M, T


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


class _TemporalPlayerSet:
    """Partition T time-steps into M equal-width windows (test fixture)."""

    n_players: int = M
    shape: tuple[int, int, int] = (J, F, T)

    def coalition_mask(self, z: Tensor) -> Tensor:
        ws = T // M
        mask = torch.zeros(J, F, T, dtype=torch.bool)
        for k in range(M):
            if z[k]:
                mask[:, :, k * ws : (k + 1) * ws] = True
        return mask

    def aggregate(self, phi_coords: Tensor) -> Tensor:
        ws = T // M
        phi = torch.zeros(M)
        for k in range(M):
            phi[k] = phi_coords[:, :, k * ws : (k + 1) * ws].sum()
        return phi


class _ZeroImputer(BaseImputer):
    """Trivial imputer: fills hidden coordinates with zeros."""

    def fit(self, train_data: object) -> "_ZeroImputer":  # type: ignore[override]
        return self

    def impute(
        self,
        x_obs: Tensor,
        mask: Tensor,
        n_samples: int,
        seed: int | None = None,
    ) -> Tensor:
        out = torch.zeros(n_samples, *x_obs.shape, dtype=torch.float32)
        out[:, mask] = x_obs[mask]
        return out


class _TinyGRU(nn.Module):
    """Tiny 1-layer GRU: (B, J, F, T) → (B,)."""

    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=J * F, hidden_size=4, batch_first=True)
        self.head = nn.Linear(4, 1)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        # (B, J, F, T) → (B, T, J*F)
        x_flat = x.permute(0, 3, 1, 2).reshape(B, T, J * F)
        _, h = self.gru(x_flat)  # h: (1, B, 4)
        out = self.head(h.squeeze(0)).squeeze(-1)  # (B,)
        return torch.sigmoid(out)


def _make_classifier() -> object:
    """Return a deterministic tiny GRU classifier callable."""
    torch.manual_seed(0)
    model = _TinyGRU()
    model.eval()
    return model


def _make_sample() -> Tensor:
    torch.manual_seed(42)
    return torch.randn(J, F, T)


# ---------------------------------------------------------------------------
# Instantiation (non-slow)
# ---------------------------------------------------------------------------


def test_timeshap_instantiates() -> None:
    imputer = _ZeroImputer()
    clf = _make_classifier()
    attr = TimeSHAPAttributor(clf, imputer=imputer, n_coalitions=10, seed=0)
    assert attr is not None


def test_windowshap_instantiates() -> None:
    clf = _make_classifier()
    attr = WindowSHAPAttributor(clf, window_len=8, seed=0)
    assert attr is not None


def test_shats_instantiates() -> None:
    clf = _make_classifier()
    attr = ShaTSAttributor(clf)
    assert attr is not None


def test_group_segment_shap_instantiates() -> None:
    imputer = _ZeroImputer()
    clf = _make_classifier()
    attr = GroupSegmentSHAPAttributor(clf, imputer=imputer, n_coalitions=1, seed=0)
    assert attr is not None


# ---------------------------------------------------------------------------
# Property contracts (non-slow)
# ---------------------------------------------------------------------------


def test_timeshap_name() -> None:
    imputer = _ZeroImputer()
    clf = _make_classifier()
    attr = TimeSHAPAttributor(clf, imputer=imputer)
    assert attr.name == "TimeSHAP"


def test_windowshap_name() -> None:
    clf = _make_classifier()
    attr = WindowSHAPAttributor(clf)
    assert attr.name == "WindowSHAP"


def test_shats_name() -> None:
    clf = _make_classifier()
    attr = ShaTSAttributor(clf)
    assert attr.name == "ShaTS"


def test_group_segment_shap_name() -> None:
    imputer = _ZeroImputer()
    clf = _make_classifier()
    attr = GroupSegmentSHAPAttributor(clf, imputer=imputer)
    assert attr.name == "GroupSegmentSHAP"


def test_timeshap_requires_imputer() -> None:
    imputer = _ZeroImputer()
    clf = _make_classifier()
    attr = TimeSHAPAttributor(clf, imputer=imputer)
    assert attr.requires_imputer is True


def test_group_segment_requires_imputer() -> None:
    imputer = _ZeroImputer()
    clf = _make_classifier()
    attr = GroupSegmentSHAPAttributor(clf, imputer=imputer)
    assert attr.requires_imputer is True


def test_windowshap_not_requires_imputer() -> None:
    clf = _make_classifier()
    attr = WindowSHAPAttributor(clf)
    # WindowSHAP uses the library's internal masking — no BaseImputer needed.
    assert attr.requires_imputer is False


def test_shats_not_requires_imputer() -> None:
    clf = _make_classifier()
    attr = ShaTSAttributor(clf)
    assert attr.requires_imputer is False


# ---------------------------------------------------------------------------
# ShaTSAttributor — NotImplementedError (non-slow)
# ---------------------------------------------------------------------------


def test_shats_attribute_raises_not_implemented() -> None:
    clf = _make_classifier()
    attr = ShaTSAttributor(clf)
    players = _TemporalPlayerSet()
    x = _make_sample()
    with pytest.raises(NotImplementedError, match="shats library not available"):
        attr.attribute(x, players)


def test_shats_error_message_contains_url() -> None:
    clf = _make_classifier()
    attr = ShaTSAttributor(clf)
    players = _TemporalPlayerSet()
    x = _make_sample()
    with pytest.raises(NotImplementedError, match="https://github.com"):
        attr.attribute(x, players)


# ---------------------------------------------------------------------------
# Shapley math helpers (non-slow, pure-NumPy)
# ---------------------------------------------------------------------------


def test_shapley_efficiency_axiom() -> None:
    """Shapley values sum to v(grand_coalition) - v(empty_coalition)."""
    M_test = 3
    rng = np.random.default_rng(7)
    v = rng.standard_normal(1 << M_test)
    v[0] = 0.0
    phi = _shapley_from_v(v, M_test)
    grand = float(v[(1 << M_test) - 1])
    assert abs(phi.sum() - grand) < 1e-10, f"Efficiency violated: {phi.sum()} != {grand}"


def test_shapley_from_value_table_alias() -> None:
    """Public alias matches internal function."""
    M_test = 3
    rng = np.random.default_rng(8)
    v = rng.standard_normal(1 << M_test)
    v[0] = 0.0
    phi_internal = _shapley_from_v(v, M_test)
    phi_public = shapley_from_value_table(v, M_test)
    np.testing.assert_array_equal(phi_internal, phi_public)


def test_direct_group_shapley_matches_shapley() -> None:
    """When each group has size 1, group Shapley == individual Shapley."""
    M_test = 4
    rng = np.random.default_rng(9)
    v = rng.standard_normal(1 << M_test)
    v[0] = 0.0
    groups = [[i] for i in range(M_test)]  # trivial groups
    phi_group = direct_group_shapley(v, M_test, groups)
    phi_indiv = _shapley_from_v(v, M_test)
    np.testing.assert_allclose(phi_group, phi_indiv, atol=1e-10)


def test_direct_group_shapley_efficiency() -> None:
    """Group Shapley values sum to v(all) - v(empty)."""
    M_test = 4
    rng = np.random.default_rng(10)
    v = rng.standard_normal(1 << M_test)
    v[0] = 0.0
    groups = [[0, 1], [2, 3]]
    phi = direct_group_shapley(v, M_test, groups)
    grand = float(v[(1 << M_test) - 1])
    assert abs(phi.sum() - grand) < 1e-10


def test_group_segment_large_m_raises() -> None:
    """GroupSegmentSHAP must reject M > 20."""
    imputer = _ZeroImputer()
    clf = _make_classifier()
    attr = GroupSegmentSHAPAttributor(clf, imputer=imputer)

    class _BigPlayers:
        n_players = 21
        shape = (J, F, T)

        def coalition_mask(self, z: Tensor) -> Tensor:
            return torch.ones(J, F, T, dtype=torch.bool)

        def aggregate(self, phi_coords: Tensor) -> Tensor:
            return torch.zeros(21)

    with pytest.raises(ValueError, match="M ≤ 20"):
        attr.attribute(_make_sample(), _BigPlayers())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TimeSHAPAttributor — output shape / dtype (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_timeshap_attribute_shape_dtype() -> None:
    """TimeSHAP returns (M,) float32."""
    imputer = _ZeroImputer()
    clf = _make_classifier()
    attr = TimeSHAPAttributor(clf, imputer=imputer, n_coalitions=20, seed=0)
    players = _TemporalPlayerSet()
    x = _make_sample()

    phi = attr.attribute(x, players, target=0)

    assert phi.shape == (M,), f"Expected ({M},), got {phi.shape}"
    assert phi.dtype == torch.float32, f"Expected float32, got {phi.dtype}"


@pytest.mark.slow
def test_timeshap_attribute_target_ignored_gracefully() -> None:
    """TimeSHAP accepts target=1 without error for single-output classifier."""
    imputer = _ZeroImputer()
    clf = _make_classifier()
    attr = TimeSHAPAttributor(clf, imputer=imputer, n_coalitions=10, seed=1)
    players = _TemporalPlayerSet()
    x = _make_sample()

    phi = attr.attribute(x, players, target=1)
    assert phi.shape == (M,)


# ---------------------------------------------------------------------------
# WindowSHAPAttributor — output shape / dtype (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_windowshap_attribute_shape_dtype() -> None:
    """WindowSHAP returns (M,) float32."""
    clf = _make_classifier()
    attr = WindowSHAPAttributor(clf, window_len=8, stride=8, seed=0)
    players = _TemporalPlayerSet()
    x = _make_sample()

    phi = attr.attribute(x, players, target=0)

    assert phi.shape == (M,), f"Expected ({M},), got {phi.shape}"
    assert phi.dtype == torch.float32, f"Expected float32, got {phi.dtype}"


@pytest.mark.slow
def test_windowshap_invalid_window_len_raises() -> None:
    """WindowSHAP raises ValueError when window_len >= T."""
    clf = _make_classifier()
    attr = WindowSHAPAttributor(clf, window_len=T, seed=0)
    players = _TemporalPlayerSet()
    x = _make_sample()
    with pytest.raises(ValueError, match="window_len"):
        attr.attribute(x, players)


# ---------------------------------------------------------------------------
# GroupSegmentSHAPAttributor — output shape / dtype (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_group_segment_shap_attribute_shape_dtype() -> None:
    """GroupSegmentSHAP returns (M,) float32."""
    imputer = _ZeroImputer()
    clf = _make_classifier()
    # n_coalitions=1 for speed: only 1 imputer draw per coalition,
    # so 2^M = 16 classifier calls total.
    attr = GroupSegmentSHAPAttributor(clf, imputer=imputer, n_coalitions=1, seed=0)
    players = _TemporalPlayerSet()
    x = _make_sample()

    phi = attr.attribute(x, players, target=0)

    assert phi.shape == (M,), f"Expected ({M},), got {phi.shape}"
    assert phi.dtype == torch.float32, f"Expected float32, got {phi.dtype}"


@pytest.mark.slow
def test_group_segment_shap_efficiency() -> None:
    """GroupSegmentSHAP values sum to v(all_present) - v(all_absent)."""
    imputer = _ZeroImputer()
    clf = _make_classifier()
    attr = GroupSegmentSHAPAttributor(clf, imputer=imputer, n_coalitions=1, seed=0)
    players = _TemporalPlayerSet()
    x = _make_sample()

    phi = attr.attribute(x, players)

    # Compute v(grand) and v(empty) directly.
    all_mask = torch.ones(J, F, T, dtype=torch.bool)
    none_mask = torch.zeros(J, F, T, dtype=torch.bool)

    x_full = imputer.impute(x, all_mask, n_samples=1)[0]
    x_empty = imputer.impute(x, none_mask, n_samples=1)[0]

    with torch.no_grad():
        v_grand = float(clf(x_full.unsqueeze(0))[0].item())
        v_empty = float(clf(x_empty.unsqueeze(0))[0].item())

    expected_sum = v_grand - v_empty
    actual_sum = float(phi.sum().item())
    assert abs(actual_sum - expected_sum) < 1e-4, (
        f"Efficiency violated: sum(phi)={actual_sum:.6f} != "
        f"v(all)-v(none)={expected_sum:.6f}"
    )
