"""Tests for motionbench.pipelines.player_eval.

Uses a small GaussianMotionDataset (via a temporary config tree) so the full
cell — coalition design, imputer fills, WLS solve, deterministic grading —
runs end-to-end on CPU in seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from motionbench.pipelines.player_eval import (
    _build_player_set,
    _ec_metrics,
    _infer_game,
    run_player_eval,
)

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestInferGame:
    def test_explicit_key_wins(self):
        cfg = OmegaConf.create(
            {"game": "cond", "imputer": {"_target_": "x.ZeroImputer"}}
        )
        assert _infer_game(cfg) == "cond"

    def test_inferred_cond_for_oracle(self):
        cfg = OmegaConf.create(
            {"imputer": {"_target_": "motionbench.oracles.gaussian_oracle.GaussianOracle"}}
        )
        assert _infer_game(cfg) == "cond"

    def test_inferred_marg_for_zero_and_empirical(self):
        for target in (
            "motionbench.imputers.off_manifold.ZeroImputer",
            "motionbench.imputers.empirical.EmpiricalConditionalImputer",
        ):
            cfg = OmegaConf.create({"imputer": {"_target_": target}})
            assert _infer_game(cfg) == "marg", target


class TestBuildPlayerSet:
    @pytest.mark.parametrize(
        "target,expected_m",
        [
            ("motionbench.players.temporal_windows.TemporalWindows", 4),
            ("motionbench.players.spatial_joints.SpatialJoints", 5),
            ("motionbench.players.joint_window_cells.JointWindowCells", 20),
        ],
    )
    def test_constructor_kwarg_filtering(self, target, expected_m):
        cfg = OmegaConf.create({"name": "x", "_target_": target})
        players = _build_player_set(cfg, J=5, F=3, T=16, K=4)
        assert players.n_players == expected_m
        assert players.shape == (5, 3, 16)


class TestECMetrics:
    def test_perfect_attribution_scores_zero(self):
        phi = np.array([0.3, -0.1, 0.5])
        ec1, ec3 = _ec_metrics(phi, phi.copy())
        assert ec1 == 0.0
        assert ec3 == pytest.approx(0.0, abs=1e-12)

    def test_constant_vector_gets_ec3_one(self):
        phi = np.zeros(4)
        star = np.array([0.1, 0.2, -0.1, 0.4])
        _ec1, ec3 = _ec_metrics(phi, star)
        assert ec3 == 1.0

    def test_anticorrelated_near_two(self):
        star = np.array([1.0, -1.0, 2.0, -2.0])
        _ec1, ec3 = _ec_metrics(-star, star)
        assert ec3 == pytest.approx(2.0, abs=1e-12)


# ---------------------------------------------------------------------------
# End-to-end cell (temporary config tree; no checkpoint => random init)
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_config_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a minimal configs/ tree in tmp_path and chdir into it."""
    root = tmp_path / "configs"
    (root / "data").mkdir(parents=True)
    (root / "players").mkdir()
    (root / "classifiers").mkdir()
    (root / "methods").mkdir()
    (root / "data" / "tiny_gauss.yaml").write_text(
        "_target_: motionbench.data.synthetic.gaussian_motion.GaussianMotionDataset\n"
        "J: 3\nF: 2\nT: 8\nK: 4\nN: 12\nrho: 0.5\nalpha: 0.8\nseed: 0\n"
    )
    (root / "players" / "spatial.yaml").write_text(
        "name: spatial\n_target_: motionbench.players.spatial_joints.SpatialJoints\n"
    )
    (root / "classifiers" / "synthetic_mlp.yaml").write_text(
        "_target_: motionbench.classifiers.synthetic_mlp.SyntheticMLPClassifier\n"
        "hidden: 16\nplayer_mode: temporal\n"
    )
    (root / "methods" / "kernelshap_zero.yaml").write_text(
        "name: kernelshap_zero\ngame: marg\n"
        "attributor:\n  _target_: motionbench.attribution.kernel_shap.KernelShapAttributor\n"
        "imputer:\n  _target_: motionbench.imputers.off_manifold.ZeroImputer\n"
        "n_completion_samples: 2\nseed: 42\n"
    )
    (root / "methods" / "kernelshap_oracle.yaml").write_text(
        "name: kernelshap_oracle\ngame: cond\n"
        "attributor:\n  _target_: motionbench.attribution.kernel_shap.KernelShapAttributor\n"
        "imputer:\n  _target_: motionbench.oracles.gaussian_oracle.GaussianOracle\n"
        "n_completion_samples: 3\nseed: 42\n"
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _tiny_cfg(tmp_path: Path, methods: list[str]) -> OmegaConf:
    return OmegaConf.create(
        {
            "pipeline": "player_eval",
            "datasets": ["tiny_gauss"],
            "player_sets": ["spatial"],
            "methods": methods,
            "classifiers": ["synthetic_mlp"],
            "coalition_budget": 64,
            "coalition_seed": 7919,
            "value_fn": "f_of_mean",
            "n_sequences": 4,
            "seed": 42,
            "results_dir": str(tmp_path / "results"),
            "checkpoint_dir": str(tmp_path / "ckpts"),  # absent => random init
            "device": "cpu",
            "wandb": {"mode": "disabled"},
        }
    )


def test_end_to_end_zero_scores_zero_on_marginal_game(tiny_config_tree: Path):
    """KS-Zero equals the deterministic marginal target by construction."""
    torch.manual_seed(0)
    df = run_player_eval(_tiny_cfg(tiny_config_tree, ["kernelshap_zero"]))
    assert len(df) == 1
    row = df.iloc[0]
    assert row["game"] == "marg"
    assert row["ec1"] < 1e-6, "KS-Zero must match the marginal target exactly"
    assert row["M"] == 3
    result_path = (
        tiny_config_tree / "results" / "spatial" / "tiny_gauss" / "synthetic_mlp"
        / "kernelshap_zero" / "result.json"
    )
    assert result_path.exists()
    per_seq = np.load(result_path.parent / "per_sequence.npz")
    assert per_seq["phi"].shape == (4, 3)
    assert per_seq["phi_star"].shape == (4, 3)


def test_end_to_end_oracle_conditional_game(tiny_config_tree: Path):
    """Oracle-imputer cell runs the conditional deterministic target."""
    torch.manual_seed(0)
    df = run_player_eval(_tiny_cfg(tiny_config_tree, ["kernelshap_oracle"]))
    row = df.iloc[0]
    assert row["game"] == "cond"
    assert np.isfinite(row["ec1"]) and np.isfinite(row["ec3"])
    assert row["ec1"] >= 0.0


def test_cell_resume_from_cache(tiny_config_tree: Path):
    """A second run returns the cached result without recomputation."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(tiny_config_tree, ["kernelshap_zero"])
    df1 = run_player_eval(cfg)
    result_path = (
        tiny_config_tree / "results" / "spatial" / "tiny_gauss" / "synthetic_mlp"
        / "kernelshap_zero" / "result.json"
    )
    stamped = json.loads(result_path.read_text())
    stamped["ec1"] = 123.0  # sentinel: must be returned untouched
    result_path.write_text(json.dumps(stamped))
    df2 = run_player_eval(cfg)
    assert df2.iloc[0]["ec1"] == 123.0
    assert df1.iloc[0]["dataset"] == "tiny_gauss"
