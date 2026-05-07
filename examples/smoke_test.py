"""End-to-end smoke test for motionbench-xai core abstractions.

Verifies that the four base abstractions are correctly wired by running a
complete mock pipeline: Dataset → PlayerSet → Imputer → Attributor → Metric.

This script is run by CI on every PR and must pass in < 10 seconds on CPU.
No heavy dependencies (no captum, shap, quantus, etc.) are used here.

Run:
    python examples/smoke_test.py
"""
from __future__ import annotations

import sys
import time
import traceback

import torch

from motionbench.attribution.base import BaseAttributor
from motionbench.data.base import BaseDataset, GroundTruthDataset
from motionbench.imputers.base import BaseImputer
from motionbench.metrics.base import BaseMetric
from motionbench.oracles.base import Oracle
from motionbench.players.base import PlayerSet

# ---------------------------------------------------------------------------
# Canonical test shape
# ---------------------------------------------------------------------------

J, F, T, M = 5, 3, 16, 4
N = 10
SEED = 42

torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# 1. Mock implementations of each ABC
# ---------------------------------------------------------------------------


class MockPlayers(PlayerSet):
    """M equal temporal windows over T frames."""

    def __init__(self) -> None:
        self._J, self._F, self._T, self._M = J, F, T, M
        self._ws = T // M

    @property
    def n_players(self) -> int:
        return self._M

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._J, self._F, self._T

    def coalition_mask(self, z: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(J, F, T, dtype=torch.bool)
        for k in range(M):
            if z[k]:
                mask[:, :, k * self._ws : (k + 1) * self._ws] = True
        return mask

    def aggregate(self, phi_coords: torch.Tensor) -> torch.Tensor:
        phi = torch.zeros(M)
        for k in range(M):
            phi[k] = phi_coords[:, :, k * self._ws : (k + 1) * self._ws].sum()
        return phi


class MockOracle(Oracle):
    """Trivial oracle: returns zeros for hidden, copies observed."""

    def conditional_sample(
        self,
        x_obs: torch.Tensor,
        mask: torch.Tensor,
        n: int,
        seed: int | None = None,
    ) -> torch.Tensor:
        out = torch.zeros(n, J, F, T)
        out[:, mask] = x_obs[mask]
        return out

    def true_shapley(
        self,
        x: torch.Tensor,
        classifier,  # type: ignore[override]
        players: PlayerSet,
        n_mc: int = 100,
        seed: int | None = None,
    ) -> torch.Tensor:
        return torch.full((players.n_players,), 1.0 / players.n_players)


class MockDataset:
    """Minimal GroundTruthDataset with mock oracle."""

    def __init__(self) -> None:
        self._oracle = MockOracle()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(idx)
        return torch.randn(J, F, T), torch.tensor(0)

    def __len__(self) -> int:
        return N

    @property
    def shape(self) -> tuple[int, int, int]:
        return J, F, T

    @property
    def metadata(self) -> dict[str, object]:
        return {"skeleton": "mock_5j", "frame_rate": 30.0}

    @property
    def oracle(self) -> MockOracle:
        return self._oracle


class MockImputer(BaseImputer):
    """Zero-fill imputer."""

    def fit(self, train_data: BaseDataset) -> "MockImputer":  # type: ignore[override]
        self._fitted = True
        return self

    def impute(
        self,
        x_obs: torch.Tensor,
        mask: torch.Tensor,
        n_samples: int,
        seed: int | None = None,
    ) -> torch.Tensor:
        out = torch.zeros(n_samples, J, F, T)
        out[:, mask] = x_obs[mask]
        return out


class MockAttributor(BaseAttributor):
    """Returns sum of abs(x) per player."""

    def attribute(
        self,
        x: torch.Tensor,
        players: PlayerSet,
        target: int = 0,
    ) -> torch.Tensor:
        return players.aggregate(x.abs())


class MockMetric(BaseMetric):
    """Returns EC1 vs oracle."""

    requires_oracle = True
    requires_imputer = False

    def evaluate(
        self,
        phi: torch.Tensor,
        x: torch.Tensor,
        classifier,  # type: ignore[override]
        players: PlayerSet,
        target: int = 0,
        oracle: Oracle | None = None,
        imputer: BaseImputer | None = None,
    ) -> dict[str, float]:
        self._check_deps(oracle, imputer)
        assert oracle is not None
        phi_true = oracle.true_shapley(x, classifier, players)
        ec1 = float((phi - phi_true).abs().mean().item())
        return {"ec1": ec1}


# ---------------------------------------------------------------------------
# 2. Classifier (tiny, deterministic)
# ---------------------------------------------------------------------------


def mock_classifier(x: torch.Tensor) -> torch.Tensor:
    """(B, J, F, T) → (B,) scalar."""
    return x.mean(dim=(1, 2, 3))


# ---------------------------------------------------------------------------
# 3. Smoke test
# ---------------------------------------------------------------------------


def run_smoke_test() -> None:
    print("=" * 60)
    print("MotionBench-XAI Phase 0 smoke test")
    print("=" * 60)

    t0 = time.time()

    # --- Dataset ---
    print("[1/6] Dataset ... ", end="", flush=True)
    ds = MockDataset()
    assert isinstance(ds, BaseDataset), "MockDataset must satisfy BaseDataset protocol"
    assert isinstance(ds, GroundTruthDataset), "MockDataset must satisfy GroundTruthDataset"
    x, y = ds[0]
    assert x.shape == (J, F, T), f"Expected ({J},{F},{T}), got {x.shape}"
    assert ds.oracle is not None
    print("OK")

    # --- PlayerSet ---
    print("[2/6] PlayerSet ... ", end="", flush=True)
    players = MockPlayers()
    z = torch.ones(M, dtype=torch.int)
    mask = players.coalition_mask(z)
    assert mask.shape == (J, F, T)
    assert mask.all()
    phi_coords = torch.randn(J, F, T)
    phi_agg = players.aggregate(phi_coords)
    assert phi_agg.shape == (M,)
    print("OK")

    # --- Oracle ---
    print("[3/6] Oracle ... ", end="", flush=True)
    oracle = ds.oracle
    half_mask = torch.zeros(J, F, T, dtype=torch.bool)
    half_mask[:, :, : T // 2] = True
    samples = oracle.conditional_sample(x, half_mask, n=5)
    assert samples.shape == (5, J, F, T)
    # Observed entries must match
    for i in range(5):
        assert torch.allclose(samples[i][half_mask], x[half_mask]), (
            f"Sample {i}: oracle changed observed entries"
        )
    phi_oracle = oracle.true_shapley(x, mock_classifier, players)
    assert phi_oracle.shape == (M,)
    print("OK")

    # --- Imputer ---
    print("[4/6] Imputer ... ", end="", flush=True)
    imputer = MockImputer().fit(ds)
    imputations = imputer.impute(x, half_mask, n_samples=8)
    assert imputations.shape == (8, J, F, T)
    for i in range(8):
        assert torch.allclose(imputations[i][half_mask], x[half_mask]), (
            f"Imputer {i}: observed entry changed"
        )
    print("OK")

    # --- Attributor ---
    print("[5/6] Attributor ... ", end="", flush=True)
    attributor = MockAttributor(mock_classifier)
    phi = attributor.attribute(x, players, target=0)
    assert phi.shape == (M,), f"Expected ({M},), got {phi.shape}"
    assert phi.dtype == torch.float32
    print("OK")

    # --- Metric ---
    print("[6/6] Metric ... ", end="", flush=True)
    metric = MockMetric()
    scores = metric.evaluate(phi, x, mock_classifier, players, oracle=oracle)
    assert isinstance(scores, dict)
    assert "ec1" in scores
    assert isinstance(scores["ec1"], float)
    print("OK")

    elapsed = time.time() - t0
    print()
    print(f"All checks passed in {elapsed:.2f}s.")
    print()
    print("Core abstractions are correctly wired.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        run_smoke_test()
    except Exception:
        print("\nSMOKE TEST FAILED")
        traceback.print_exc()
        sys.exit(1)
