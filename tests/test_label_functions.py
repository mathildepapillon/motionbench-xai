"""tests/test_label_functions — Unit tests for the synthetic label function library.

Tests are organised per label function and cover:

1. ``test_output_shape``          — call returns ``(N,)`` int64 in {0,..,n_classes-1}.
2. ``test_important_players_subset`` — important_players returns set[int] ⊆ valid indices.
3. ``test_irrelevant_players_have_low_gradient`` (``@pytest.mark.slow``, Localized* only) —
   randomising irrelevant coordinates does not change labels.

The ``FakePlayerSet`` fixture provides a minimal concrete PlayerSet backed by
K equal-width temporal windows for shape ``(J, F, T)``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from motionbench.data.synthetic.label_functions import (
    Linear,
    LocalizedSpatial,
    LocalizedSpatiotemporal,
    LocalizedTemporal,
    OlsenInteraction,
    SpatialOlsen,
)
from motionbench.players.base import PlayerSet

# ---------------------------------------------------------------------------
# Test constants (small but non-trivial)
# ---------------------------------------------------------------------------

N, J, F, T, K = 200, 5, 3, 16, 4
SEED = 42


# ---------------------------------------------------------------------------
# Minimal concrete PlayerSet for testing
# ---------------------------------------------------------------------------


class FakeTemporalPlayerSet(PlayerSet):
    """Minimal PlayerSet: K equal-width temporal windows over (J, F, T)."""

    def __init__(self, J: int, F: int, T: int, K: int) -> None:
        self._J = J
        self._F = F
        self._T = T
        self._K = K
        quarter = T // K
        self._windows = [
            list(range(k * quarter, (k + 1) * quarter if k < K - 1 else T))
            for k in range(K)
        ]

    @property
    def n_players(self) -> int:
        return self._K

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self._J, self._F, self._T)

    def coalition_mask(self, z: Tensor) -> Tensor:
        mask = torch.zeros(self._J, self._F, self._T, dtype=torch.bool)
        for k in range(self._K):
            if z[k]:
                for t in self._windows[k]:
                    mask[:, :, t] = True
        return mask

    def aggregate(self, phi_coords: Tensor) -> Tensor:
        result = torch.zeros(self._K, dtype=phi_coords.dtype)
        for k, frames in enumerate(self._windows):
            result[k] = phi_coords[:, :, frames].sum()
        return result


class FakeSpatialPlayerSet(PlayerSet):
    """Minimal PlayerSet: one player per joint over (J, F, T)."""

    def __init__(self, J: int, F: int, T: int) -> None:
        self._J = J
        self._F = F
        self._T = T

    @property
    def n_players(self) -> int:
        return self._J

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self._J, self._F, self._T)

    def coalition_mask(self, z: Tensor) -> Tensor:
        mask = torch.zeros(self._J, self._F, self._T, dtype=torch.bool)
        for j in range(self._J):
            if z[j]:
                mask[j, :, :] = True
        return mask

    def aggregate(self, phi_coords: Tensor) -> Tensor:
        return phi_coords.sum(dim=(1, 2))  # (J,)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture()
def x_batch(rng: np.random.Generator) -> np.ndarray:
    """``(N, J, F, T)`` standard-normal float32 batch."""
    return rng.standard_normal((N, J, F, T)).astype(np.float32)


@pytest.fixture()
def temporal_players() -> FakeTemporalPlayerSet:
    return FakeTemporalPlayerSet(J=J, F=F, T=T, K=K)


@pytest.fixture()
def spatial_players() -> FakeSpatialPlayerSet:
    return FakeSpatialPlayerSet(J=J, F=F, T=T)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _assert_label_array(labels: np.ndarray, n_classes: int = 3) -> None:
    """Assert shape (N,), dtype int64, values in {0,..,n_classes-1}."""
    assert labels.shape == (N,), f"Expected shape ({N},), got {labels.shape}."
    assert labels.dtype == np.int64, f"Expected int64, got {labels.dtype}."
    assert labels.min() >= 0, f"Negative label found: {labels.min()}"
    assert labels.max() < n_classes, (
        f"Label {labels.max()} >= n_classes={n_classes}."
    )


# ===========================================================================
# Linear
# ===========================================================================


class TestLinear:
    def test_output_shape(self, x_batch: np.ndarray) -> None:
        weights = np.ones(J * F * T, dtype=np.float64)
        lf = Linear(weights)
        labels = lf(x_batch)
        _assert_label_array(labels)

    def test_output_shape_binary(self, x_batch: np.ndarray) -> None:
        weights = np.random.default_rng(0).standard_normal(J * F * T)
        lf = Linear(weights, n_classes=2, percentiles=(50.0,))
        labels = lf(x_batch)
        _assert_label_array(labels, n_classes=2)

    def test_important_players_subset(
        self,
        x_batch: np.ndarray,
        temporal_players: FakeTemporalPlayerSet,
    ) -> None:
        # Build weights non-zero only in the coordinate range of player 0.
        # Use coalition_mask to locate those coordinates correctly.
        z0 = torch.zeros(K, dtype=torch.int32)
        z0[0] = 1
        mask_p0 = temporal_players.coalition_mask(z0).numpy().flatten()  # (J*F*T,)
        weights = np.zeros(J * F * T, dtype=np.float64)
        weights[mask_p0] = 1.0

        lf = Linear(weights)
        lf(x_batch)  # trigger any lazy init
        players = lf.important_players(temporal_players)
        assert isinstance(players, set)
        valid = set(range(temporal_players.n_players))
        assert players <= valid, f"{players} is not a subset of {valid}."
        # Player 0 must be included (has non-zero weight); others must not.
        assert 0 in players
        for p in range(1, K):
            assert p not in players

    def test_weights_not_1d_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            Linear(np.ones((J, F * T)))

    def test_n_classes_2_raises_bad_percentiles(self) -> None:
        with pytest.raises(ValueError):
            Linear(np.ones(J * F * T), n_classes=2, percentiles=(33.0, 67.0))


# ===========================================================================
# OlsenInteraction
# ===========================================================================


class TestOlsenInteraction:
    def test_output_shape(self, x_batch: np.ndarray) -> None:
        lf = OlsenInteraction(K=4, seed=0)
        labels = lf(x_batch)
        _assert_label_array(labels)

    def test_output_shape_k8(self, rng: np.random.Generator) -> None:
        x = rng.standard_normal((N, J, F, 32)).astype(np.float32)
        lf = OlsenInteraction(K=8, seed=1)
        labels = lf(x)
        assert labels.shape == (N,)
        assert labels.dtype == np.int64
        assert set(np.unique(labels)) <= {0, 1, 2}

    def test_important_players_subset(
        self,
        x_batch: np.ndarray,
        temporal_players: FakeTemporalPlayerSet,
    ) -> None:
        lf = OlsenInteraction(K=4, seed=0)
        players = lf.important_players(temporal_players)
        assert isinstance(players, set)
        assert players <= set(range(temporal_players.n_players))

    def test_all_windows_important(
        self,
        temporal_players: FakeTemporalPlayerSet,
    ) -> None:
        lf = OlsenInteraction(K=4, seed=0)
        players = lf.important_players(temporal_players)
        assert players == {0, 1, 2, 3}

    def test_calibration_is_deterministic(self, x_batch: np.ndarray) -> None:
        lf1 = OlsenInteraction(K=4, seed=7)
        lf2 = OlsenInteraction(K=4, seed=7)
        labels1 = lf1(x_batch)
        labels2 = lf2(x_batch)
        np.testing.assert_array_equal(labels1, labels2)

    def test_k_not_multiple_of_4_raises(self) -> None:
        with pytest.raises(ValueError, match="multiple of 4"):
            OlsenInteraction(K=3)

    def test_subsequent_calls_use_cached_params(self, x_batch: np.ndarray) -> None:
        lf = OlsenInteraction(K=4, seed=0)
        labels_first = lf(x_batch)
        labels_second = lf(x_batch)
        np.testing.assert_array_equal(labels_first, labels_second)


# ===========================================================================
# SpatialOlsen
# ===========================================================================


class TestSpatialOlsen:
    def test_output_shape(self, x_batch: np.ndarray) -> None:
        lf = SpatialOlsen(signal_joints=[0, 1, 2, 3], seed=0)
        labels = lf(x_batch)
        _assert_label_array(labels)

    def test_important_players_subset(
        self,
        x_batch: np.ndarray,
        spatial_players: FakeSpatialPlayerSet,
    ) -> None:
        signal = [0, 1, 2, 3]
        lf = SpatialOlsen(signal_joints=signal, seed=0)
        players = lf.important_players(spatial_players)
        assert isinstance(players, set)
        valid = set(range(spatial_players.n_players))
        assert players <= valid, f"{players} not a subset of {valid}."

    def test_important_players_equals_signal_joints(
        self,
        spatial_players: FakeSpatialPlayerSet,
    ) -> None:
        signal = [1, 2, 3, 4]
        lf = SpatialOlsen(signal_joints=signal, seed=0)
        assert lf.important_players(spatial_players) == {1, 2, 3, 4}

    def test_wrong_num_signal_joints_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly 4"):
            SpatialOlsen(signal_joints=[0, 1, 2])

    def test_duplicate_signal_joints_raises(self) -> None:
        with pytest.raises(ValueError, match="distinct"):
            SpatialOlsen(signal_joints=[0, 0, 1, 2])

    def test_calibration_is_deterministic(self, x_batch: np.ndarray) -> None:
        lf1 = SpatialOlsen(signal_joints=[0, 1, 2, 3], seed=3)
        lf2 = SpatialOlsen(signal_joints=[0, 1, 2, 3], seed=3)
        np.testing.assert_array_equal(lf1(x_batch), lf2(x_batch))


# ===========================================================================
# LocalizedTemporal
# ===========================================================================


class TestLocalizedTemporal:
    def test_output_shape(self, x_batch: np.ndarray) -> None:
        lf = LocalizedTemporal(window_idx=0, K=K)
        labels = lf(x_batch)
        _assert_label_array(labels)

    def test_important_players_subset(
        self,
        temporal_players: FakeTemporalPlayerSet,
    ) -> None:
        lf = LocalizedTemporal(window_idx=1, K=K)
        players = lf.important_players(temporal_players)
        assert isinstance(players, set)
        assert players <= set(range(temporal_players.n_players))
        assert players == {1}

    def test_window_idx_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            LocalizedTemporal(window_idx=4, K=4)

    @pytest.mark.slow
    def test_irrelevant_players_have_low_gradient(
        self, rng: np.random.Generator
    ) -> None:
        """Randomising non-window-0 frames must not change labels."""
        lf = LocalizedTemporal(window_idx=0, K=K)
        x = rng.standard_normal((N, J, F, T)).astype(np.float32)
        labels_orig = lf(x)

        quarter = T // K
        x_perturbed = x.copy()
        # Replace frames belonging to windows 1, 2, 3 with fresh noise.
        x_perturbed[:, :, :, quarter:] = rng.standard_normal(
            (N, J, F, T - quarter)
        ).astype(np.float32)

        labels_perturbed = lf(x_perturbed)
        # Perfectly localised: labels must be identical.
        np.testing.assert_array_equal(
            labels_orig,
            labels_perturbed,
            err_msg="Randomising irrelevant windows changed labels.",
        )


# ===========================================================================
# LocalizedSpatial
# ===========================================================================


class TestLocalizedSpatial:
    def test_output_shape(self, x_batch: np.ndarray) -> None:
        lf = LocalizedSpatial(joint_idx=0)
        labels = lf(x_batch)
        _assert_label_array(labels)

    def test_important_players_subset(
        self,
        spatial_players: FakeSpatialPlayerSet,
    ) -> None:
        lf = LocalizedSpatial(joint_idx=2)
        players = lf.important_players(spatial_players)
        assert isinstance(players, set)
        assert players <= set(range(spatial_players.n_players))
        assert players == {2}

    def test_negative_joint_idx_raises(self) -> None:
        with pytest.raises(ValueError):
            LocalizedSpatial(joint_idx=-1)

    @pytest.mark.slow
    def test_irrelevant_players_have_low_gradient(
        self, rng: np.random.Generator
    ) -> None:
        """Randomising non-joint-0 data must not change labels."""
        lf = LocalizedSpatial(joint_idx=0)
        x = rng.standard_normal((N, J, F, T)).astype(np.float32)
        labels_orig = lf(x)

        x_perturbed = x.copy()
        # Replace all joints except joint 0 with fresh noise.
        x_perturbed[:, 1:, :, :] = rng.standard_normal(
            (N, J - 1, F, T)
        ).astype(np.float32)

        labels_perturbed = lf(x_perturbed)
        np.testing.assert_array_equal(
            labels_orig,
            labels_perturbed,
            err_msg="Randomising irrelevant joints changed labels.",
        )


# ===========================================================================
# LocalizedSpatiotemporal
# ===========================================================================


class TestLocalizedSpatiotemporal:
    def test_output_shape(self, x_batch: np.ndarray) -> None:
        lf = LocalizedSpatiotemporal(joint_idx=0, window_idx=0, K=K)
        labels = lf(x_batch)
        _assert_label_array(labels)

    def test_important_players_subset(
        self,
        temporal_players: FakeTemporalPlayerSet,
    ) -> None:
        lf = LocalizedSpatiotemporal(joint_idx=1, window_idx=2, K=K)
        players = lf.important_players(temporal_players)
        assert isinstance(players, set)
        valid = set(range(temporal_players.n_players))
        # {1, 2} ⊆ {0,1,2,3}
        assert players <= valid
        assert 1 in players
        assert 2 in players

    def test_same_joint_and_window_idx(
        self,
        temporal_players: FakeTemporalPlayerSet,
    ) -> None:
        lf = LocalizedSpatiotemporal(joint_idx=1, window_idx=1, K=K)
        players = lf.important_players(temporal_players)
        assert players == {1}

    def test_window_idx_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            LocalizedSpatiotemporal(joint_idx=0, window_idx=4, K=4)

    @pytest.mark.slow
    def test_irrelevant_players_have_low_gradient(
        self, rng: np.random.Generator
    ) -> None:
        """Randomising non-(joint0, window0) cells must not change labels."""
        lf = LocalizedSpatiotemporal(joint_idx=0, window_idx=0, K=K)
        x = rng.standard_normal((N, J, F, T)).astype(np.float32)
        labels_orig = lf(x)

        quarter = T // K
        x_perturbed = x.copy()
        # Replace joint 0 windows 1-3 with noise.
        x_perturbed[:, 0, :, quarter:] = rng.standard_normal(
            (N, F, T - quarter)
        ).astype(np.float32)
        # Replace all other joints entirely.
        x_perturbed[:, 1:, :, :] = rng.standard_normal(
            (N, J - 1, F, T)
        ).astype(np.float32)

        labels_perturbed = lf(x_perturbed)
        np.testing.assert_array_equal(
            labels_orig,
            labels_perturbed,
            err_msg="Randomising irrelevant spatiotemporal cells changed labels.",
        )


# ===========================================================================
# LabelFunction ABC validation
# ===========================================================================


class TestLabelFunctionABC:
    def test_n_classes_1_raises(self) -> None:
        with pytest.raises(ValueError, match="n_classes must be >= 2"):
            OlsenInteraction(K=4, n_classes=1)

    def test_mismatched_percentiles_raises(self) -> None:
        with pytest.raises(ValueError, match="len\\(percentiles\\)"):
            OlsenInteraction(K=4, n_classes=3, percentiles=(50.0,))

    def test_binarize_covers_all_classes(self, x_batch: np.ndarray) -> None:
        lf = OlsenInteraction(K=4, seed=0)
        labels = lf(x_batch)
        # With N=200 samples and ternary split, all 3 classes should appear.
        assert set(np.unique(labels)) == {0, 1, 2}


# ===========================================================================
# Custom fn parameter
# ===========================================================================


class TestCustomFn:
    def test_custom_fn_temporal(self, x_batch: np.ndarray) -> None:
        def max_fn(arr: np.ndarray) -> np.ndarray:
            return arr.reshape(arr.shape[0], -1).max(axis=1)

        lf = LocalizedTemporal(window_idx=0, K=K, fn=max_fn)
        labels = lf(x_batch)
        _assert_label_array(labels)

    def test_custom_fn_spatial(self, x_batch: np.ndarray) -> None:
        def std_fn(arr: np.ndarray) -> np.ndarray:
            return arr.reshape(arr.shape[0], -1).std(axis=1)

        lf = LocalizedSpatial(joint_idx=1, fn=std_fn)
        labels = lf(x_batch)
        _assert_label_array(labels)

    def test_custom_fn_spatiotemporal(self, x_batch: np.ndarray) -> None:
        def max_fn(arr: np.ndarray) -> np.ndarray:
            return arr.reshape(arr.shape[0], -1).max(axis=1)

        lf = LocalizedSpatiotemporal(joint_idx=2, window_idx=1, K=K, fn=max_fn)
        labels = lf(x_batch)
        _assert_label_array(labels)
