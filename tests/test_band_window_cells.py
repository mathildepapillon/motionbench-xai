"""Tests for motionbench.players.band_window_cells.BandWindowCells."""

from __future__ import annotations

import pytest
import torch

from motionbench.players import BandWindowCells


class TestConstruction:
    def test_n_players(self):
        players = BandWindowCells(J=128, n_bands=4, K=4, F=1, T=1024)
        assert players.n_players == 16

    def test_shape(self):
        players = BandWindowCells(J=128, n_bands=4, K=4, F=1, T=1024)
        assert players.shape == (128, 1, 1024)

    def test_band_edges_equal_quartiles(self):
        players = BandWindowCells(J=128, n_bands=4, K=4, F=1, T=1024)
        assert players.band_edges == [(0, 32), (32, 64), (64, 96), (96, 128)]

    def test_band_edges_uneven_split(self):
        # divmod partition: the first J % n_bands bands get one extra bin.
        players = BandWindowCells(J=10, n_bands=3, K=2, F=1, T=8)
        assert players.band_edges == [(0, 4), (4, 7), (7, 10)]

    def test_rejects_indivisible_t(self):
        with pytest.raises(ValueError, match="divisible"):
            BandWindowCells(J=128, n_bands=4, K=3, F=1, T=1024)

    def test_rejects_more_bands_than_bins(self):
        with pytest.raises(ValueError, match="n_bands"):
            BandWindowCells(J=2, n_bands=4, K=2, F=1, T=8)


class TestCoalitionMask:
    def test_empty_and_full(self):
        players = BandWindowCells(J=8, n_bands=4, K=2, F=1, T=16)
        M = players.n_players
        empty = players.coalition_mask(torch.zeros(M, dtype=torch.int))
        full = players.coalition_mask(torch.ones(M, dtype=torch.int))
        assert not empty.any()
        assert full.all()

    def test_single_cell_extent(self):
        players = BandWindowCells(J=8, n_bands=4, K=2, F=2, T=16)
        z = torch.zeros(players.n_players, dtype=torch.int)
        z[players.player_index(1, 1)] = 1  # band 1 (bins 2:4), window 1 (t 8:16)
        mask = players.coalition_mask(z)
        expected = torch.zeros(8, 2, 16, dtype=torch.bool)
        expected[2:4, :, 8:16] = True
        assert torch.equal(mask, expected)

    def test_players_partition_the_grid(self):
        players = BandWindowCells(J=10, n_bands=3, K=4, F=2, T=8)
        M = players.n_players
        total = torch.zeros(10, 2, 8, dtype=torch.int)
        for p in range(M):
            z = torch.zeros(M, dtype=torch.int)
            z[p] = 1
            total += players.coalition_mask(z).int()
        assert torch.equal(total, torch.ones_like(total))  # disjoint + exhaustive

    def test_index_convention_b_times_k_plus_k(self):
        players = BandWindowCells(J=8, n_bands=4, K=2, F=1, T=16)
        z = torch.zeros(players.n_players, dtype=torch.int)
        z[3 * 2 + 0] = 1  # band 3, window 0
        mask = players.coalition_mask(z)
        assert mask[6:8, :, 0:8].all()
        assert mask.sum() == 2 * 1 * 8

    def test_rejects_wrong_shape(self):
        players = BandWindowCells(J=8, n_bands=4, K=2, F=1, T=16)
        with pytest.raises(ValueError, match="Expected z.shape"):
            players.coalition_mask(torch.zeros(4, dtype=torch.int))


class TestAggregate:
    def test_sums_within_cells(self):
        players = BandWindowCells(J=8, n_bands=4, K=2, F=1, T=16)
        phi_coords = torch.ones(8, 1, 16)
        phi = players.aggregate(phi_coords)
        assert phi.shape == (8,)
        assert torch.allclose(phi, torch.full((8,), 2.0 * 8.0))  # 2 bins x 8 steps

    def test_matches_masked_sum(self):
        players = BandWindowCells(J=10, n_bands=3, K=4, F=2, T=8)
        g = torch.Generator().manual_seed(0)
        phi_coords = torch.randn(10, 2, 8, generator=g)
        phi = players.aggregate(phi_coords)
        M = players.n_players
        for p in range(M):
            z = torch.zeros(M, dtype=torch.int)
            z[p] = 1
            mask = players.coalition_mask(z)
            assert torch.allclose(phi[p], phi_coords[mask].sum())

    def test_rejects_wrong_shape(self):
        players = BandWindowCells(J=8, n_bands=4, K=2, F=1, T=16)
        with pytest.raises(ValueError, match="Expected phi_coords"):
            players.aggregate(torch.zeros(8, 1, 8))
