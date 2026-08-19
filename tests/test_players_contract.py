"""Contract tests for motionbench.players.base.PlayerSet.

These tests verify that any concrete subclass of PlayerSet satisfies the
full interface contract, including output shapes, dtype, and the
indivisible-masking guarantee.

A ``MockPlayerSet`` is defined here; all passing tests confirm that the
ABC itself is correctly wired.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from motionbench.players.base import PlayerSet

# ---------------------------------------------------------------------------
# Mock implementation (minimal concrete subclass)
# ---------------------------------------------------------------------------


class MockPlayerSet(PlayerSet):
    """Temporal windows: M equal-width windows over T time-steps."""

    def __init__(self, J: int, F: int, T: int, M: int) -> None:
        self._J = J
        self._F = F
        self._T = T
        self._M = M
        assert T % M == 0, "T must be divisible by M for this mock"
        self._window_size = T // M

    @property
    def n_players(self) -> int:
        return self._M

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._J, self._F, self._T

    def coalition_mask(self, z: Tensor) -> Tensor:
        if z.shape != (self._M,):
            raise ValueError(f"expected z.shape==({self._M},); got {tuple(z.shape)}")
        mask = torch.zeros(self._J, self._F, self._T, dtype=torch.bool)
        for k in range(self._M):
            if z[k]:
                t_start = k * self._window_size
                t_end = (k + 1) * self._window_size
                mask[:, :, t_start:t_end] = True
        return mask

    def aggregate(self, phi_coords: Tensor) -> Tensor:
        if phi_coords.shape != (self._J, self._F, self._T):
            raise ValueError(
                f"expected phi_coords.shape=={self.shape}; got {tuple(phi_coords.shape)}"
            )
        phi = torch.zeros(self._M)
        for k in range(self._M):
            t_start = k * self._window_size
            t_end = (k + 1) * self._window_size
            phi[k] = phi_coords[:, :, t_start:t_end].sum()
        return phi


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def players(shape, n_players):
    J, F, T = shape
    return MockPlayerSet(J, F, T, n_players)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


def test_abc_cannot_be_instantiated():
    """PlayerSet is abstract; direct instantiation must fail."""
    with pytest.raises(TypeError):
        PlayerSet()  # type: ignore[abstract]


def test_mock_is_concrete(players):
    """MockPlayerSet must not raise — confirms ABC is fully satisfied."""
    assert players is not None


def test_n_players(players, n_players):
    assert players.n_players == n_players


def test_shape(players, shape):
    assert players.shape == shape


def test_coalition_mask_shape(players, shape, n_players):
    z = torch.ones(n_players, dtype=torch.int)
    mask = players.coalition_mask(z)
    assert mask.shape == shape, f"Expected {shape}, got {mask.shape}"
    assert mask.dtype == torch.bool


def test_coalition_mask_all_observed(players, shape, n_players):
    z = torch.ones(n_players, dtype=torch.int)
    mask = players.coalition_mask(z)
    assert mask.all(), "All players observed → all coordinates observed"


def test_coalition_mask_all_hidden(players, shape, n_players):
    z = torch.zeros(n_players, dtype=torch.int)
    mask = players.coalition_mask(z)
    assert not mask.any(), "No players observed → all coordinates hidden"


def test_coalition_mask_indivisible(players, shape, n_players):
    """When player k is hidden, ALL its coordinates must be hidden (indivisible masking)."""
    J, F, T = shape
    ws = T // n_players
    for k in range(n_players):
        z = torch.ones(n_players, dtype=torch.int)
        z[k] = 0
        mask = players.coalition_mask(z)
        # Check: player k's time slice is fully hidden
        t_start = k * ws
        t_end = (k + 1) * ws
        assert not mask[:, :, t_start:t_end].any(), (
            f"Player {k} hidden but some coordinates still observed"
        )
        # Check: all other time slices are fully observed
        other_frames = list(range(0, t_start)) + list(range(t_end, T))
        assert mask[:, :, other_frames].all(), (
            f"Observed players {[i for i in range(n_players) if i != k]} have hidden coords"
        )


def test_coalition_mask_wrong_shape_raises(players, n_players):
    with pytest.raises(ValueError):
        players.coalition_mask(torch.ones(n_players + 1, dtype=torch.int))


def test_aggregate_shape(players, shape, n_players):
    J, F, T = shape
    phi_coords = torch.randn(J, F, T)
    phi = players.aggregate(phi_coords)
    assert phi.shape == (n_players,), f"Expected ({n_players},), got {phi.shape}"


def test_aggregate_wrong_shape_raises(players, shape):
    J, F, T = shape
    with pytest.raises(ValueError):
        players.aggregate(torch.randn(J + 1, F, T))


def test_aggregate_linearity(players, shape, n_players):
    """Aggregate is linear: agg(α·φ + β·ψ) == α·agg(φ) + β·agg(ψ)."""
    J, F, T = shape
    alpha, beta = 2.0, -0.5
    phi = torch.randn(J, F, T)
    psi = torch.randn(J, F, T)
    lhs = players.aggregate(alpha * phi + beta * psi)
    rhs = alpha * players.aggregate(phi) + beta * players.aggregate(psi)
    assert torch.allclose(lhs, rhs, atol=1e-5), "Aggregate must be linear"


def test_batch_coalition_masks_shape(players, shape, n_players):
    N = 8
    Z = torch.randint(0, 2, (N, n_players))
    masks = players.batch_coalition_masks(Z)
    J, F, T = shape
    assert masks.shape == (N, J, F, T)
    assert masks.dtype == torch.bool


def test_repr(players):
    r = repr(players)
    assert "MockPlayerSet" in r
    assert "n_players" in r


# ---------------------------------------------------------------------------
# Concrete player-set contracts (mask/aggregate) — shared parametrised tests
# ---------------------------------------------------------------------------

from motionbench.players.joint_window_cells import JointWindowCells  # noqa: E402
from motionbench.players.spatial_joints import SpatialJoints  # noqa: E402
from motionbench.players.temporal_windows import TemporalWindows  # noqa: E402

_J, _F, _T, _K = 5, 3, 16, 4


def _concrete_player_sets():
    return [
        pytest.param(TemporalWindows(K=_K, T=_T, J=_J, F=_F), id="temporal"),
        pytest.param(SpatialJoints(J=_J, F=_F, T=_T), id="spatial"),
        pytest.param(JointWindowCells(J=_J, K=_K, F=_F, T=_T), id="cells"),
    ]


@pytest.mark.parametrize("ps", _concrete_player_sets())
def test_concrete_singleton_masks_partition(ps):
    """Singleton coalition masks are disjoint and cover all coordinates."""
    M = ps.n_players
    total = torch.zeros(ps.shape, dtype=torch.int32)
    for k in range(M):
        z = torch.zeros(M, dtype=torch.int32)
        z[k] = 1
        mask = ps.coalition_mask(z)
        assert mask.dtype == torch.bool
        assert mask.any(), f"player {k} owns no coordinates"
        total += mask.int()
    assert (total == 1).all(), "player masks must partition (J, F, T) exactly once"


@pytest.mark.parametrize("ps", _concrete_player_sets())
def test_concrete_mask_union_consistency(ps):
    """coalition_mask(z) equals the union of the singleton masks of z."""
    M = ps.n_players
    z = torch.zeros(M, dtype=torch.int32)
    z[:: max(1, M // 3)] = 1
    mask = ps.coalition_mask(z)
    union = torch.zeros(ps.shape, dtype=torch.bool)
    for k in torch.nonzero(z).flatten().tolist():
        zk = torch.zeros(M, dtype=torch.int32)
        zk[k] = 1
        union |= ps.coalition_mask(zk)
    assert torch.equal(mask, union)


@pytest.mark.parametrize("ps", _concrete_player_sets())
def test_concrete_boundary_masks(ps):
    M = ps.n_players
    assert ps.coalition_mask(torch.ones(M, dtype=torch.int32)).all()
    assert not ps.coalition_mask(torch.zeros(M, dtype=torch.int32)).any()


@pytest.mark.parametrize("ps", _concrete_player_sets())
def test_concrete_aggregate_sums_own_coordinates(ps):
    """aggregate(phi)[k] is the sum of phi over player k's coordinates."""
    M = ps.n_players
    phi_coords = torch.randn(*ps.shape)
    agg = ps.aggregate(phi_coords)
    assert agg.shape == (M,)
    for k in range(M):
        zk = torch.zeros(M, dtype=torch.int32)
        zk[k] = 1
        mask = ps.coalition_mask(zk)
        expected = float(phi_coords[mask].sum())
        assert float(agg[k]) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("ps", _concrete_player_sets())
def test_concrete_aggregate_efficiency_preserved(ps):
    """Total attribution mass is preserved by aggregation (partition sets)."""
    phi_coords = torch.randn(*ps.shape)
    agg = ps.aggregate(phi_coords)
    assert float(agg.sum()) == pytest.approx(float(phi_coords.sum()), abs=1e-4)


def test_cells_player_index_layout():
    """JointWindowCells player index is j * K + k."""
    ps = JointWindowCells(J=_J, K=_K, F=_F, T=_T)
    ws = _T // _K
    j, k = 2, 3
    z = torch.zeros(ps.n_players, dtype=torch.int32)
    z[ps.player_index(j, k)] = 1
    mask = ps.coalition_mask(z)
    expected = torch.zeros(_J, _F, _T, dtype=torch.bool)
    expected[j, :, k * ws : (k + 1) * ws] = True
    assert torch.equal(mask, expected)
