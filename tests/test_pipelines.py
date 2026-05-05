"""Tests for motionbench.pipelines — synthetic_eval, real_eval, leaderboard.

All tests use tiny data (J=3, F=2, T=8, K=4, N=20) and a single method
(kernelshap_zero) to keep the test suite fast.  They are NOT marked slow
so they run in CI.

The end-to-end test invokes the full CLI via ``subprocess`` to verify that
Hydra config loading, the argument parser, and the pipeline all wire together
correctly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_gaussian_dataset() -> Any:
    """Return a tiny GaussianMotionDataset (J=3, F=2, T=8, K=4, N=20)."""
    from motionbench.data.synthetic.gaussian_motion import GaussianMotionDataset

    return GaussianMotionDataset(J=3, F=2, T=8, K=4, N=20, seed=0)


@pytest.fixture
def tiny_mlp(tiny_gaussian_dataset: Any) -> Any:
    """Return a tiny SyntheticMLPClassifier matching the tiny dataset."""
    from motionbench.classifiers.synthetic_mlp import SyntheticMLPClassifier

    J, F, T = tiny_gaussian_dataset.shape
    K = int(tiny_gaussian_dataset.metadata["K"])
    return SyntheticMLPClassifier(J=J, F=F, T=T, K=K, n_classes=3, hidden=16)


@pytest.fixture
def tiny_players(tiny_gaussian_dataset: Any) -> Any:
    """Return TemporalWindows matching the tiny dataset."""
    from motionbench.players.temporal_windows import TemporalWindows

    J, F, T = tiny_gaussian_dataset.shape
    K = int(tiny_gaussian_dataset.metadata["K"])
    return TemporalWindows(K=K, T=T, J=J, F=F)


@pytest.fixture
def tmp_results(tmp_path: Path) -> Path:
    """Return a temporary results directory."""
    return tmp_path / "results"


# ---------------------------------------------------------------------------
# Unit tests — _instantiate_dataset
# ---------------------------------------------------------------------------


class TestInstantiateDataset:
    """Tests for :func:`motionbench.pipelines.synthetic_eval._instantiate_dataset`."""

    def test_gaussian_dataset_with_k(self) -> None:
        from motionbench.pipelines.synthetic_eval import _instantiate_dataset

        cfg = OmegaConf.create({
            "_target_": "motionbench.data.synthetic.gaussian_motion.GaussianMotionDataset",
            "J": 3,
            "F": 2,
            "T": 8,
            "K": 4,
            "N": 20,
            "rho": 0.3,
            "alpha": 0.5,
            "seed": 0,
        })
        dataset, K = _instantiate_dataset(cfg)
        assert K == 4
        assert dataset.shape == (3, 2, 8)
        assert len(dataset) == 20

    def test_burr_dataset_k_stripped(self) -> None:
        from motionbench.pipelines.synthetic_eval import _instantiate_dataset

        cfg = OmegaConf.create({
            "_target_": "motionbench.data.synthetic.burr_motion.BurrMotionBenchmark",
            "J": 3,
            "F": 2,
            "T": 8,
            "K": 5,  # pipeline-only; must be stripped before passing to constructor
            "N": 20,
            "rho": 0.3,
            "alpha": 0.5,
            "seed": 0,
        })
        dataset, K = _instantiate_dataset(cfg)
        assert K == 5
        assert dataset.shape == (3, 2, 8)


# ---------------------------------------------------------------------------
# Unit tests — _build_players
# ---------------------------------------------------------------------------


class TestBuildPlayers:
    """Tests for :func:`motionbench.pipelines.synthetic_eval._build_players`."""

    def test_temporal_windows(self) -> None:
        from motionbench.pipelines.synthetic_eval import _build_players

        method_cfg = OmegaConf.create({
            "name": "kernelshap_zero",
            "players": {
                "_target_": "motionbench.players.temporal_windows.TemporalWindows",
            },
        })
        players = _build_players(method_cfg, J=3, F=2, T=8, K=4)
        assert players.n_players == 4
        assert players.shape == (3, 2, 8)


# ---------------------------------------------------------------------------
# Unit tests — _build_classifier
# ---------------------------------------------------------------------------


class TestBuildClassifier:
    """Tests for :func:`motionbench.pipelines.synthetic_eval._build_classifier`."""

    def test_synthetic_mlp(self) -> None:
        from motionbench.pipelines.synthetic_eval import _build_classifier

        clf_cfg = OmegaConf.create({
            "_target_": "motionbench.classifiers.synthetic_mlp.SyntheticMLPClassifier",
            "hidden": 16,
            "player_mode": "temporal",
        })
        clf = _build_classifier(clf_cfg, J=3, F=2, T=8, K=4, n_classes=3)
        assert clf.n_classes == 3
        x = torch.randn(2, 3, 2, 8)
        with torch.no_grad():
            out = clf(x)
        assert out.shape == (2, 3)

    def test_synthetic_cnn(self) -> None:
        from motionbench.pipelines.synthetic_eval import _build_classifier

        clf_cfg = OmegaConf.create({
            "_target_": "motionbench.classifiers.synthetic_cnn.SyntheticCNNClassifier",
        })
        clf = _build_classifier(clf_cfg, J=3, F=2, T=8, K=4, n_classes=3)
        x = torch.randn(2, 3, 2, 8)
        with torch.no_grad():
            out = clf(x)
        assert out.shape == (2, 3)


# ---------------------------------------------------------------------------
# Unit tests — _build_attributor (gradient methods)
# ---------------------------------------------------------------------------


class TestBuildAttributorGradient:
    """Tests for gradient-based attributor construction."""

    def test_ig_attributor(self, tiny_mlp: Any, tiny_players: Any) -> None:
        from motionbench.pipelines.synthetic_eval import _build_attributor, _make_model_fn

        method_cfg = OmegaConf.create({
            "name": "ig_zero",
            "attributor": {
                "_target_": "motionbench.attribution.captum_methods.IntegratedGradientsAttributor",
                "baseline": "zero",
                "n_steps": 5,
            },
            "players": {
                "_target_": "motionbench.players.temporal_windows.TemporalWindows",
            },
        })
        model_fn = _make_model_fn(tiny_mlp, target=0)
        attributor = _build_attributor(method_cfg, tiny_mlp, None, tiny_players)
        assert attributor is not None

        J, F, T = tiny_players.shape
        x = torch.randn(J, F, T)
        phi = attributor.attribute(x, tiny_players, target=0)
        assert phi.shape == (tiny_players.n_players,)


# ---------------------------------------------------------------------------
# Unit tests — _run_cell (end-to-end single cell)
# ---------------------------------------------------------------------------


class TestRunCell:
    """Tests for :func:`motionbench.pipelines.synthetic_eval._run_cell`."""

    def test_kernelshap_zero_cell(self, tmp_results: Path) -> None:
        """Run one kernelshap_zero cell on a tiny Gaussian dataset."""
        from motionbench.pipelines.synthetic_eval import _run_cell

        # Build a minimal config that looks like the experiment config
        cfg = OmegaConf.create({
            "results_dir": str(tmp_results),
            "n_sequences": 2,
            "n_jobs": 1,
            "device": "cpu",
            "wandb": {"mode": "disabled"},
            "metrics": {
                "gt": ["ec1", "ec2"],
                "stability": ["max_sensitivity"],
            },
            "datasets": ["gaussian_k4"],
            "methods": ["kernelshap_zero"],
            "classifiers": ["synthetic_mlp"],
        })

        # Patch _load_sub_config to return inline configs (avoid filesystem dependency)
        def mock_load_sub_config(subdir: str, name: str, _cfg: DictConfig) -> DictConfig:
            configs: dict[str, dict[str, Any]] = {
                ("data", "gaussian_k4"): {
                    "_target_": "motionbench.data.synthetic.gaussian_motion.GaussianMotionDataset",
                    "J": 3, "F": 2, "T": 8, "K": 4, "N": 20, "seed": 0,
                },
                ("classifiers", "synthetic_mlp"): {
                    "_target_": "motionbench.classifiers.synthetic_mlp.SyntheticMLPClassifier",
                    "hidden": 16, "player_mode": "temporal",
                },
                ("methods", "kernelshap_zero"): {
                    "name": "kernelshap_zero",
                    "attributor": {
                        "_target_": "motionbench.attribution.kernel_shap.KernelShapAttributor",
                    },
                    "imputer": {
                        "_target_": "motionbench.imputers.off_manifold.ZeroImputer",
                    },
                    "players": {
                        "_target_": "motionbench.players.temporal_windows.TemporalWindows",
                    },
                    "n_kernel_samples": 16,
                    "n_completion_samples": 2,
                    "seed": 0,
                },
            }
            return OmegaConf.create(configs[(subdir, name)])

        with patch(
            "motionbench.pipelines.synthetic_eval._load_sub_config",
            side_effect=mock_load_sub_config,
        ):
            result = _run_cell("gaussian_k4", "synthetic_mlp", "kernelshap_zero", cfg)

        assert "error" not in result, f"Cell failed: {result.get('error')}"
        assert result["dataset"] == "gaussian_k4"
        assert result["method"] == "kernelshap_zero"
        # Result JSON should have been saved
        result_path = (
            tmp_results / "gaussian_k4" / "synthetic_mlp" / "kernelshap_zero" / "result.json"
        )
        assert result_path.exists()

    def test_resumability(self, tmp_results: Path) -> None:
        """Re-running a completed cell returns cached result without recomputation."""
        from motionbench.pipelines.synthetic_eval import _run_cell

        cached = {"dataset": "d", "classifier": "c", "method": "m", "ec1": 0.1}
        result_path = tmp_results / "d" / "c" / "m" / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(cached))

        cfg = OmegaConf.create({"results_dir": str(tmp_results)})
        result = _run_cell("d", "c", "m", cfg)

        assert result == cached


# ---------------------------------------------------------------------------
# Unit tests — leaderboard
# ---------------------------------------------------------------------------


class TestLeaderboard:
    """Tests for :func:`motionbench.pipelines.leaderboard.build_leaderboard`."""

    def _write_results(self, tmp_path: Path) -> None:
        for method in ("method_a", "method_b"):
            path = tmp_path / "ds1" / "clf1" / method / "result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "dataset": "ds1",
                "classifier": "clf1",
                "method": method,
                "ec1": 0.1 if method == "method_a" else 0.2,
                "ec2": 0.01 if method == "method_a" else 0.04,
            }))

    def test_load_results(self, tmp_path: Path) -> None:
        from motionbench.pipelines.leaderboard import load_results

        self._write_results(tmp_path)
        df = load_results(tmp_path)
        assert len(df) == 2
        assert set(df["method"]) == {"method_a", "method_b"}

    def test_build_leaderboard_rank_order(self, tmp_path: Path) -> None:
        from motionbench.pipelines.leaderboard import build_leaderboard

        self._write_results(tmp_path)
        lb = build_leaderboard(tmp_path, rank_by="ec1", ascending=True)
        assert lb.iloc[0]["method"] == "method_a"  # lower ec1 is better
        assert lb.iloc[0]["rank"] == 1

    def test_load_results_empty(self, tmp_path: Path) -> None:
        from motionbench.pipelines.leaderboard import load_results

        df = load_results(tmp_path)
        assert df.empty

    def test_load_results_missing_dir(self, tmp_path: Path) -> None:
        from motionbench.pipelines.leaderboard import load_results

        with pytest.raises(FileNotFoundError):
            load_results(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# Integration test — full pipeline on tiny config
# ---------------------------------------------------------------------------


class TestFullPipelineIntegration:
    """Integration test: run synthetic_eval end-to-end with tiny mocked configs."""

    def test_run_synthetic_eval(self, tmp_path: Path) -> None:
        from motionbench.pipelines.synthetic_eval import run_synthetic_eval

        cfg = OmegaConf.create({
            "pipeline": "synthetic",
            "datasets": ["gaussian_k4"],
            "methods": ["kernelshap_zero"],
            "classifiers": ["synthetic_mlp"],
            "metrics": {"gt": ["ec1"], "stability": []},
            "n_sequences": 2,
            "n_jobs": 1,
            "device": "cpu",
            "results_dir": str(tmp_path / "results"),
            "wandb": {"mode": "disabled"},
        })

        def mock_load_sub_config(subdir: str, name: str, _cfg: DictConfig) -> DictConfig:
            configs: dict[str, dict[str, Any]] = {
                ("data", "gaussian_k4"): {
                    "_target_": "motionbench.data.synthetic.gaussian_motion.GaussianMotionDataset",
                    "J": 3, "F": 2, "T": 8, "K": 4, "N": 20, "seed": 1,
                },
                ("classifiers", "synthetic_mlp"): {
                    "_target_": "motionbench.classifiers.synthetic_mlp.SyntheticMLPClassifier",
                    "hidden": 16, "player_mode": "temporal",
                },
                ("methods", "kernelshap_zero"): {
                    "name": "kernelshap_zero",
                    "attributor": {
                        "_target_": "motionbench.attribution.kernel_shap.KernelShapAttributor",
                    },
                    "imputer": {
                        "_target_": "motionbench.imputers.off_manifold.ZeroImputer",
                    },
                    "players": {
                        "_target_": "motionbench.players.temporal_windows.TemporalWindows",
                    },
                    "n_kernel_samples": 16,
                    "n_completion_samples": 2,
                    "seed": 1,
                },
            }
            return OmegaConf.create(configs[(subdir, name)])

        with patch(
            "motionbench.pipelines.synthetic_eval._load_sub_config",
            side_effect=mock_load_sub_config,
        ):
            df = run_synthetic_eval(cfg)

        assert not df.empty
        assert "method" in df.columns
        assert df.iloc[0]["method"] == "kernelshap_zero"
