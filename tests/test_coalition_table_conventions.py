"""Freeze tests for the pinned coalition-table conventions.

Golden values were computed from the implementations that produced the
canonical result files (previously embedded in scripts/run_care_pd_multiclf.py
and scripts/_player_shap_common.py) immediately before their verbatim move
into the package; the move itself was verified bit-exact in that environment.
Tolerances here are ULP-scale (rel 1e-12 float64 / 1e-6 float32) — loose
enough to survive BLAS differences across numpy builds, tight enough that any
convention change (dtype quantisation, row order, boundary handling, tie
break) fails by many orders of magnitude.  Treat a failure as a regression,
never as a fixture to update.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from motionbench.attribution.enumerated_kernel_shap import (
    build_coalition_masks,
    kernel_shap_exact,
    shapley_kernel,
)
from motionbench.metrics.coalition_table import (
    aopc_order,
    deletion_path,
    faithfulness_enumerated,
    faithfulness_sampled,
    player_aopc_enumerated,
)


def _sha(*arrays: np.ndarray) -> str:
    return hashlib.sha256(np.concatenate([a.ravel() for a in arrays]).tobytes()).hexdigest()


class TestEnumeratedConventions:
    """K=4, T=243: exercises the remainder-absorbing last window."""

    def test_mask_row_order_and_windows(self) -> None:
        z_bin, frame_mask = build_coalition_masks(4, 243)
        assert z_bin.shape == (16, 4) and frame_mask.shape == (16, 243)
        # little-endian: row 0 all-hidden, row 15 all-visible
        assert not z_bin[0].any() and z_bin[-1].all()
        assert (
            _sha(z_bin.numpy(), frame_mask.numpy())
            == "e5b9891aa1306fb4ad6716501df135ec8051a0b36dfdfd162df42657a8c026ce"
        )

    def test_kernel_boundary_in_place(self) -> None:
        assert [shapley_kernel(4, s) for s in range(5)] == [1e6, 0.25, 0.125, 0.25, 1e6]

    def test_solver_float32_pin(self) -> None:
        z_bin, _ = build_coalition_masks(4, 243)
        v = torch.from_numpy(np.random.default_rng(11).standard_normal(16)).float()
        phi = kernel_shap_exact(z_bin, v, 4)
        assert phi.dtype == torch.float32
        np.testing.assert_allclose(
            phi.numpy(),
            np.array(
                [
                    -0.09675675630569458,
                    0.5292210578918457,
                    -0.1422617882490158,
                    0.13871431350708008,
                ],
                dtype=np.float32,
            ),
            rtol=1e-6,
            atol=0,
        )

    def test_metrics_golden(self) -> None:
        z_bin, _ = build_coalition_masks(4, 243)
        v = torch.from_numpy(np.random.default_rng(11).standard_normal(16)).float()
        phi = kernel_shap_exact(z_bin, v, 4)
        assert faithfulness_enumerated(z_bin, v, phi) == pytest.approx(
            0.28725923812476006, rel=1e-6
        )
        assert player_aopc_enumerated(v, z_bin, phi, 4) == pytest.approx(
            0.6105978721752763, rel=1e-6
        )


class TestSampledConventions:
    def test_faithfulness_golden(self) -> None:
        rng = np.random.default_rng(11)
        rng.standard_normal(16)  # consume the enumerated fixture's draw
        Z = (rng.random((40, 7)) > 0.5).astype(np.int8)
        Z[0] = 0
        Z[1] = 1
        v = rng.standard_normal(40)
        phi = rng.standard_normal(7)
        assert faithfulness_sampled(Z, v, phi, 1) == pytest.approx(0.047036553702306624, rel=1e-12)

    def test_stable_tie_break_by_player_index(self) -> None:
        phi = np.array([0.5, -0.5, 0.25, 0.5, 0.0, -0.25, 0.25])
        order = aopc_order(phi)
        assert order == [0, 1, 3, 2, 5, 6, 4]
        assert (
            hashlib.sha256(deletion_path(order, 7).tobytes()).hexdigest()
            == "afd2ec5e58071b8b954b81d10e0af096e57e78c8369f92aba40625089e109b8d"
        )

    def test_degenerate_returns_nan(self) -> None:
        Z = np.array([[0, 0], [1, 1], [1, 0]], dtype=np.int8)
        v = np.zeros(3)
        assert np.isnan(faithfulness_sampled(Z, v, np.array([1.0, -1.0]), 1))
