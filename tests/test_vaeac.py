"""tests/test_vaeac.py — Unit tests for motionbench.imputers.vaeac.

Tests
-----
1. test_vaeac_shape                — impute returns (n_samples, J, F, T).
2. test_vaeac_observed_preserved   — standard observed-entry contract.
3. test_vaeac_serialization        — save/load; impute output identical (same seed).
4. test_vaeac_smoke                — 2-epoch training; loss decreases.  @slow.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from motionbench.imputers.vaeac import VAEACImputer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

J, F, T = 5, 3, 16


def _make_imputer(fitted: bool = True) -> VAEACImputer:
    """Return a small VAEACImputer (possibly pre-fitted, no actual training)."""
    imp = VAEACImputer(J=J, F=F, T=T, latent_dim=8, hidden_dim=32)
    if fitted:
        # fit() just stores the ref and marks _fitted=True; no training occurs.
        imp._fitted = True
    return imp


def _make_sample() -> tuple[torch.Tensor, torch.Tensor]:
    """Return an (x_obs, mask) pair with ~half the entries observed."""
    x_obs = torch.randn(J, F, T)
    mask = torch.zeros(J, F, T, dtype=torch.bool)
    mask[:, :, : T // 2] = True  # first T//2 frames observed
    return x_obs, mask


# ---------------------------------------------------------------------------
# Test 1 — output shape
# ---------------------------------------------------------------------------


def test_vaeac_shape() -> None:
    """impute(x, mask, n_samples=3) must return (n_samples, J, F, T)."""
    imp = _make_imputer()
    x_obs, mask = _make_sample()

    out = imp.impute(x_obs, mask, n_samples=3, seed=0)

    assert out.shape == (3, J, F, T), f"Expected (3, {J}, {F}, {T}), got {tuple(out.shape)}"
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Test 2 — observed-entry preservation contract
# ---------------------------------------------------------------------------


def test_vaeac_observed_preserved() -> None:
    """output[:, mask] must exactly equal x_obs[mask] for all samples."""
    imp = _make_imputer()
    x_obs, mask = _make_sample()

    n_samples = 5
    out = imp.impute(x_obs, mask, n_samples=n_samples, seed=1)

    assert out.shape == (n_samples, J, F, T)

    obs_vals = x_obs[mask]  # (num_true,)
    for i in range(n_samples):
        sample_obs = out[i][mask]
        max_diff = (sample_obs - obs_vals).abs().max().item()
        assert max_diff == 0.0, (
            f"Observed entry not preserved in sample {i}: max |diff| = {max_diff}"
        )


# ---------------------------------------------------------------------------
# Test 3 — serialisation round-trip
# ---------------------------------------------------------------------------


def test_vaeac_serialization() -> None:
    """save/load must produce bit-exact impute output for the same seed."""
    imp = _make_imputer()
    x_obs, mask = _make_sample()

    out_before = imp.impute(x_obs, mask, n_samples=4, seed=7)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "vaeac_test.pt"
        imp.save(ckpt_path)

        imp_loaded = VAEACImputer.load(ckpt_path)
        out_after = imp_loaded.impute(x_obs, mask, n_samples=4, seed=7)

    assert out_before.shape == out_after.shape
    max_diff = (out_before - out_after).abs().max().item()
    assert max_diff == 0.0, (
        f"save/load round-trip produced different outputs: max |diff| = {max_diff}"
    )


# ---------------------------------------------------------------------------
# Test 4 — smoke (marked slow; not run in CI)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_vaeac_smoke() -> None:
    """2-epoch training on random Gaussian data; ELBO loss must decrease."""
    imp = VAEACImputer(J=J, F=F, T=T, latent_dim=8, hidden_dim=32)

    # Synthetic Gaussian training data: (N, J, F, T)
    torch.manual_seed(42)
    x_data = torch.randn(32, J, F, T)

    losses = imp._fit_epochs(x_data, epochs=2, batch_size=8, lr=1e-2)

    assert len(losses) == 2, f"Expected 2 loss values, got {len(losses)}"
    assert all(isinstance(v, float) for v in losses)

    # After 2 epochs of training on a tiny dataset the loss should go down.
    # We use a generous tolerance: just check final < initial.
    assert losses[-1] < losses[0], (
        f"Training loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    )

    # Verify impute still works after training
    x_obs, mask = _make_sample()
    out = imp.impute(x_obs, mask, n_samples=3, seed=0)
    assert out.shape == (3, J, F, T)

    # Verify observed-entry contract holds after training
    obs_vals = x_obs[mask]
    for i in range(3):
        max_diff = (out[i][mask] - obs_vals).abs().max().item()
        assert max_diff == 0.0, f"Observed entry not preserved post-training in sample {i}"
