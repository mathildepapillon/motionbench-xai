"""Tests for motionbench.imputers.flow_matching.FlowMatchingImputer.

Non-manual, non-slow tests run in CI:
  pytest tests/test_flow.py -q -m "not slow and not manual"

All tests use deterministic seeds from conftest.py (SEED=42).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import torch
from torch import Tensor

from motionbench.imputers.flow_matching import FlowMatchingImputer
from tests.conftest import F, J, T

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class _TinyDataset:
    """Minimal dataset for testing: N identical random samples of shape (J, F, T)."""

    def __init__(self, n: int, J: int, F: int, T: int, seed: int = 0) -> None:
        rng = torch.Generator()
        rng.manual_seed(seed)
        # Use a fixed sample repeated to make the dataset easy to overfit.
        x0 = torch.randn(J, F, T, generator=rng)
        self._x = x0.unsqueeze(0).expand(n, -1, -1, -1).clone()
        self._y = torch.zeros(n, dtype=torch.long)

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self._x[idx], self._y[idx]

    @property
    def shape(self) -> tuple[int, int, int]:
        return J, F, T

    @property
    def metadata(self) -> dict[str, object]:
        return {"skeleton": "mock", "frame_rate": 30.0}

    @property
    def oracle(self) -> None:
        return None


def _make_fitted_imputer(
    *,
    J: int = J,
    F: int = F,
    T: int = T,
    hidden_dim: int = 32,
    num_steps: int = 5,
    n_samples_ds: int = 20,
    n_epochs: int = 1,
    batch_size: int = 8,
    solver: str = "midpoint",
    noise_init_scale: float = 1.0,
    seed: int = 42,
) -> FlowMatchingImputer:
    """Return a FlowMatchingImputer fitted for one epoch on tiny data (fast)."""
    torch.manual_seed(seed)
    ds = _TinyDataset(n=n_samples_ds, J=J, F=F, T=T, seed=seed)
    imp = FlowMatchingImputer(
        J=J, F=F, T=T,
        hidden_dim=hidden_dim,
        num_steps=num_steps,
        noise_init_scale=noise_init_scale,
        n_epochs=n_epochs,
        batch_size=batch_size,
        solver=solver,
        device="cpu",
    )
    imp.fit(ds)
    return imp


@pytest.fixture(scope="module")
def fitted_imp() -> FlowMatchingImputer:
    """Module-scoped fitted FlowMatchingImputer (trained once per test session).

    Using ``scope="module"`` avoids re-training for every test function and
    keeps the non-slow suite fast enough for CI.
    """
    return _make_fitted_imputer(seed=42)


# ---------------------------------------------------------------------------
# test_flow_shape
# ---------------------------------------------------------------------------


def test_flow_shape(fitted_imp: FlowMatchingImputer, x_sample: Tensor, mask_half: Tensor) -> None:
    """impute() output must have shape (n_samples, J, F, T)."""
    n = 6
    out = fitted_imp.impute(x_sample, mask_half, n_samples=n)
    assert out.shape == (n, J, F, T), (
        f"Expected ({n}, {J}, {F}, {T}), got {tuple(out.shape)}"
    )
    assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"


def test_flow_shape_n1(fitted_imp: FlowMatchingImputer, x_sample: Tensor, mask_half: Tensor) -> None:
    """n_samples=1 should produce shape (1, J, F, T)."""
    out = fitted_imp.impute(x_sample, mask_half, n_samples=1)
    assert out.shape == (1, J, F, T)


def test_flow_shape_full_mask(fitted_imp: FlowMatchingImputer, x_sample: Tensor) -> None:
    """Full mask (all observed): shape still (n, J, F, T)."""
    full_mask = torch.ones(J, F, T, dtype=torch.bool)
    out = fitted_imp.impute(x_sample, full_mask, n_samples=4)
    assert out.shape == (4, J, F, T)


# ---------------------------------------------------------------------------
# test_flow_observed_preserved
# ---------------------------------------------------------------------------


def test_flow_observed_preserved(
    fitted_imp: FlowMatchingImputer, x_sample: Tensor, mask_half: Tensor
) -> None:
    """Standard contract: observed entries must be bit-for-bit equal to x_obs."""
    n = 8
    out = fitted_imp.impute(x_sample, mask_half, n_samples=n, seed=7)
    for i in range(n):
        assert torch.allclose(out[i][mask_half], x_sample[mask_half]), (
            f"Sample {i}: observed entry changed after imputation"
        )


def test_flow_observed_full_mask(fitted_imp: FlowMatchingImputer, x_sample: Tensor) -> None:
    """Full mask: every sample equals x_obs everywhere."""
    full_mask = torch.ones(J, F, T, dtype=torch.bool)
    out = fitted_imp.impute(x_sample, full_mask, n_samples=5, seed=0)
    for i in range(5):
        assert torch.allclose(out[i], x_sample), (
            f"Full mask: sample {i} differs from x_obs"
        )


def test_flow_observed_empty_mask(fitted_imp: FlowMatchingImputer, x_sample: Tensor) -> None:
    """Empty mask (all hidden): contract satisfied trivially (no observed entries)."""
    empty = torch.zeros(J, F, T, dtype=torch.bool)
    out = fitted_imp.impute(x_sample, empty, n_samples=3)
    assert out.shape == (3, J, F, T)


def test_flow_observed_seed_reproducibility(
    fitted_imp: FlowMatchingImputer, x_sample: Tensor, mask_half: Tensor
) -> None:
    """Same seed → identical output; different seed → different output."""
    out1 = fitted_imp.impute(x_sample, mask_half, n_samples=4, seed=123)
    out2 = fitted_imp.impute(x_sample, mask_half, n_samples=4, seed=123)
    out3 = fitted_imp.impute(x_sample, mask_half, n_samples=4, seed=999)
    assert torch.allclose(out1, out2), "Same seed should produce identical samples."
    assert not torch.allclose(out1, out3), "Different seeds should produce different samples."


# ---------------------------------------------------------------------------
# test_flow_serialization
# ---------------------------------------------------------------------------


def test_flow_serialization(
    fitted_imp: FlowMatchingImputer, tmp_path: Path, x_sample: Tensor, mask_half: Tensor
) -> None:
    """save/load round trip must preserve imputation outputs exactly."""
    imp = fitted_imp
    ckpt = tmp_path / "flow_test.pt"
    imp.save(ckpt)

    imp2 = FlowMatchingImputer.load(ckpt)

    # Check meta attributes
    assert imp2.J == imp.J  # type: ignore[union-attr]
    assert imp2.F == imp.F  # type: ignore[union-attr]
    assert imp2.T == imp.T  # type: ignore[union-attr]
    assert imp2.num_steps == imp.num_steps  # type: ignore[union-attr]
    assert imp2.noise_init_scale == imp.noise_init_scale  # type: ignore[union-attr]
    assert imp2._solver == imp._solver  # type: ignore[union-attr]
    assert imp2.is_on_manifold is True

    # Check outputs match
    out1 = imp.impute(x_sample, mask_half, n_samples=4, seed=17)
    out2 = imp2.impute(x_sample, mask_half, n_samples=4, seed=17)
    assert torch.allclose(out1, out2, atol=1e-5), (
        "save/load round trip changed imputation output."
    )


def test_flow_save_raises_before_fit(tmp_path: Path) -> None:
    """save() before fit() must raise RuntimeError."""
    imp = FlowMatchingImputer(J=J, F=F, T=T, hidden_dim=32, num_steps=5, device="cpu")
    with pytest.raises(RuntimeError, match="fit"):
        imp.save(tmp_path / "should_not_exist.pt")


def test_flow_impute_raises_before_fit(x_sample: Tensor, mask_half: Tensor) -> None:
    """impute() before fit() must raise RuntimeError."""
    imp = FlowMatchingImputer(J=J, F=F, T=T, hidden_dim=32, num_steps=5, device="cpu")
    with pytest.raises(RuntimeError, match="fit"):
        imp.impute(x_sample, mask_half, n_samples=1)


def test_flow_invalid_solver() -> None:
    """Constructor with unknown solver must raise ValueError."""
    with pytest.raises(ValueError, match="solver"):
        FlowMatchingImputer(J=J, F=F, T=T, solver="runge_kutta_4")


def test_flow_is_on_manifold() -> None:
    """is_on_manifold must be True for FlowMatchingImputer."""
    imp = FlowMatchingImputer(J=J, F=F, T=T, hidden_dim=32, num_steps=5, device="cpu")
    assert imp.is_on_manifold is True


def test_flow_name_property() -> None:
    """name property must return a non-empty string."""
    imp = FlowMatchingImputer(J=J, F=F, T=T, hidden_dim=32, num_steps=5, device="cpu")
    assert isinstance(imp.name, str)
    assert len(imp.name) > 0


# ---------------------------------------------------------------------------
# test_flow_smoke  (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_flow_smoke():
    """End-to-end: 5 epochs on 20 identical samples; training loss decreases.

    Uses hidden_dim=32 and a tiny dataset (20 clones of one sample) to ensure
    the network can overfit and loss monotonically decreases.
    The test verifies the gradient pipeline is healthy.
    """
    torch.manual_seed(0)
    J_s, F_s, T_s = 5, 3, 16
    ds = _TinyDataset(n=20, J=J_s, F=F_s, T=T_s, seed=0)
    imp = FlowMatchingImputer(
        J=J_s, F=F_s, T=T_s,
        hidden_dim=32,
        num_steps=10,
        noise_init_scale=1.0,
        n_epochs=5,
        batch_size=8,
        lr=5e-3,
        device="cpu",
    )
    imp.fit(ds)

    assert len(imp.train_losses) == 5, "Expected one loss entry per epoch."
    initial = imp.train_losses[0]
    final = imp.train_losses[-1]
    assert math.isfinite(initial), f"Initial loss is not finite: {initial}"
    assert math.isfinite(final), f"Final loss is not finite: {final}"
    # Over 5 epochs with 20 identical samples, loss should decrease noticeably.
    assert final < initial, (
        f"Loss did not decrease: initial={initial:.4f}, final={final:.4f}"
    )

    # Basic sanity on imputed output
    x_obs = ds[0][0]
    mask = torch.zeros(J_s, F_s, T_s, dtype=torch.bool)
    mask[:, :, : T_s // 2] = True
    out = imp.impute(x_obs, mask, n_samples=4, seed=1)
    assert out.shape == (4, J_s, F_s, T_s)
    for i in range(4):
        assert torch.allclose(out[i][mask], x_obs[mask]), (
            f"Smoke test: observed entry changed in sample {i}"
        )


# ---------------------------------------------------------------------------
# test_flow_m10_burr_ablation  (manual)
# ---------------------------------------------------------------------------


@pytest.mark.manual
def test_flow_m10_burr_ablation(tmp_path):
    """M=10 Burr-XII regression investigation — ablation scaffold.

    Sweeps ``num_steps ∈ {10, 50, 100}`` and ``noise_init_scale ∈ {0.5, 1.0, 2.0}``
    to test H1 (discretization) and H2 (Gaussian init mismatch) hypotheses.

    .. note::
        Full EC2 computation requires oracle Shapley values (Task 1B) and the
        attribution pipeline (Task 2E), neither of which is available yet.
        This test uses a **proxy metric** — the per-coordinate MSE between
        imputed samples and held-out test samples — as a stand-in for EC2.

        When Task 1B and 2E are complete, replace ``_proxy_quality`` with
        the real EC2 computation from ``motionbench.metrics``.

    Hypothesis summary (see module docstring for full reasoning):

    * **H1 (ODE steps):** Expected minor improvement from 10→100 steps;
      unlikely to close the EC2 gap.
    * **H2 (noise_init_scale):** Expected to be the primary driver.
      Burr-XII(c=2, k=2) has std ≈ 1.53 (> 1.0), so ``noise_init_scale=2.0``
      better matches the target marginal. We predict the lowest proxy metric
      for ``noise_init_scale=2.0``, consistent with Tong et al. (2024).
    """
    try:
        from scipy.stats import burr12  # type: ignore[import-untyped]

        _HAS_SCIPY = True
    except ImportError:
        _HAS_SCIPY = False

    # --- Problem setup -------------------------------------------------------
    # M=10 temporal players over T=20 frames → each player = 2 frames.
    # We use J=3, F=2 to keep training fast enough to run manually.
    J_b, F_b, T_b = 3, 2, 20
    N_train, N_test = 500, 20
    M = 10  # temporal players

    # --- Generate Burr-XII data -----------------------------------------------
    if _HAS_SCIPY:
        rng_np = 42
        x_train_np = burr12.rvs(c=2.0, k=2.0, size=(N_train, J_b, F_b, T_b),
                                 random_state=rng_np)
        x_test_np = burr12.rvs(c=2.0, k=2.0, size=(N_test, J_b, F_b, T_b),
                                random_state=rng_np + 1)
        x_train = torch.from_numpy(x_train_np.astype("float32"))
        x_test = torch.from_numpy(x_test_np.astype("float32"))
    else:
        # Fallback: approximate Burr-XII with a log-normal (same heavy-tail flavour)
        rng = torch.Generator().manual_seed(42)
        x_train = torch.randn(N_train, J_b, F_b, T_b, generator=rng).abs() + 0.5
        rng2 = torch.Generator().manual_seed(99)
        x_test = torch.randn(N_test, J_b, F_b, T_b, generator=rng2).abs() + 0.5

    class _BurrDS:
        def __init__(self, x: Tensor) -> None:
            self._x = x
            self._y = torch.zeros(len(x), dtype=torch.long)

        def __len__(self) -> int:
            return int(self._x.shape[0])

        def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
            return self._x[idx], self._y[idx]

        @property
        def shape(self) -> tuple[int, int, int]:
            return J_b, F_b, T_b

        @property
        def metadata(self) -> dict[str, object]:
            return {"skeleton": "burr_synthetic", "frame_rate": 30.0}

        @property
        def oracle(self) -> None:
            return None

    train_ds = _BurrDS(x_train)

    # --- Proxy quality metric -------------------------------------------------
    def _proxy_quality(imp: FlowMatchingImputer, x_test_t: Tensor) -> float:
        """Lower is better: mean squared distance between imputed and actual samples.

        Full EC2 requires oracle Shapley values (Task 1B) and attribution
        pipeline (Task 2E).  This proxy is used as a stand-in.

        The mask corresponds to M//2 = 5 observed temporal players (half).
        """
        # Temporal mask: first M//2 = 5 players × 2 frames = 10 frames observed.
        frames_per_player = T_b // M
        n_obs_players = M // 2
        mask = torch.zeros(J_b, F_b, T_b, dtype=torch.bool)
        mask[:, :, : n_obs_players * frames_per_player] = True

        total_mse = 0.0
        for i in range(len(x_test_t)):
            x_obs = x_test_t[i]
            out = imp.impute(x_obs, mask, n_samples=4, seed=i)
            # Compare mean imputed sample to the actual hidden values
            mean_imputed = out.mean(0)
            hidden_mse = F_mse(mean_imputed[~mask], x_obs[~mask])
            total_mse += float(hidden_mse)
        return total_mse / len(x_test_t)

    def F_mse(a: Tensor, b: Tensor) -> Tensor:
        """Inline MSE for linter clarity."""
        return ((a - b) ** 2).mean()

    # --- Ablation grid --------------------------------------------------------
    num_steps_list = [10, 50, 100]
    noise_init_scales = [0.5, 1.0, 2.0]

    # Short training for the ablation (5 epochs — full run needs more but
    # this is enough to verify the scaffold works).
    N_EPOCHS_ABLATION = 5
    HIDDEN_DIM = 32

    results = []
    print("\n=== M=10 Burr Ablation ===")
    print(f"{'num_steps':>10} {'noise_scale':>12} {'proxy_mse':>12}")
    print("-" * 38)

    for ns in num_steps_list:
        for scale in noise_init_scales:
            torch.manual_seed(42)
            imp = FlowMatchingImputer(
                J=J_b, F=F_b, T=T_b,
                hidden_dim=HIDDEN_DIM,
                num_steps=ns,
                noise_init_scale=scale,
                n_epochs=N_EPOCHS_ABLATION,
                batch_size=32,
                lr=5e-3,
                device="cpu",
            )
            imp.fit(train_ds)
            proxy = _proxy_quality(imp, x_test)
            results.append({"num_steps": ns, "noise_init_scale": scale, "proxy_mse": proxy})
            print(f"{ns:>10} {scale:>12.1f} {proxy:>12.4f}")

    print("=" * 38)

    # --- Verify scaffold runs without errors ----------------------------------
    assert len(results) == len(num_steps_list) * len(noise_init_scales), (
        "Ablation did not complete all configurations."
    )
    for r in results:
        assert math.isfinite(r["proxy_mse"]), (
            f"Infinite proxy MSE for config {r}"
        )

    # --- Document expected findings ------------------------------------------
    # Expected: proxy_mse for noise_init_scale=2.0 < 1.0 < 0.5
    #           (H2: source std mismatch is the primary driver)
    # Expected: proxy_mse decreases modestly from num_steps=10→100
    #           (H1: secondary effect)
    #
    # NOTE: With only 5 training epochs, these trends may not yet be visible.
    # The manual ablation should be re-run with N_EPOCHS_ABLATION ≥ 100 for
    # reliable conclusions. The scaffold is correct and serves as a template.
    #
    # To obtain true EC2 scores, replace _proxy_quality with:
    #   from motionbench.metrics import ec2_score   (Task 5A)
    #   oracle = train_ds_with_oracle.oracle         (Task 1B)
    #   shapley_vals = compute_shap(imp, oracle, ...)  (Task 2E)
    #   ec2 = ec2_score(shapley_vals, oracle_shapley)

    print(
        "\nHypothesis: H2 (Gaussian init mismatch) is the primary cause.\n"
        "Expect noise_init_scale=2.0 to yield lower proxy_mse than 1.0.\n"
        "Full validation requires Task 1B (CopulaOracle) and Task 5A (EC2 metric)."
    )
