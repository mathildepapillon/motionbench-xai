"""Tests for motionbench.classifiers — MLP, CNN, and Transformer synthetic classifiers.

All tests use small dimensions (J=5, F=3, T=16, n_classes=3) for fast CPU execution.
Slow tests (checkpoint existence) are marked with @pytest.mark.slow.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from motionbench.classifiers.base import Classifier
from motionbench.classifiers.synthetic_cnn import SyntheticCNNClassifier
from motionbench.classifiers.synthetic_mlp import SyntheticMLPClassifier
from motionbench.classifiers.synthetic_transformer import SyntheticTransformerClassifier

# ---------------------------------------------------------------------------
# Common test configuration
# ---------------------------------------------------------------------------
J, F, T, K, N_CLASSES = 5, 3, 16, 4, 3
CKPT_DIR = Path(__file__).parent.parent / "motionbench" / "classifiers" / "checkpoints"


def _randn_batch(B: int) -> torch.Tensor:
    return torch.randn(B, J, F, T)


# ---------------------------------------------------------------------------
# Forward shape tests
# ---------------------------------------------------------------------------


def test_mlp_forward_shape_temporal():
    model = SyntheticMLPClassifier(J=J, F=F, T=T, K=K, n_classes=N_CLASSES, player_mode="temporal")
    out = model(_randn_batch(2))
    assert out.shape == (2, N_CLASSES), f"Expected (2, {N_CLASSES}), got {out.shape}"


def test_mlp_forward_shape_spatial():
    model = SyntheticMLPClassifier(J=J, F=F, T=T, K=K, n_classes=N_CLASSES, player_mode="spatial")
    out = model(_randn_batch(2))
    assert out.shape == (2, N_CLASSES), f"Expected (2, {N_CLASSES}), got {out.shape}"


def test_cnn_forward_shape():
    model = SyntheticCNNClassifier(J=J, F=F, n_classes=N_CLASSES)
    out = model(_randn_batch(2))
    assert out.shape == (2, N_CLASSES), f"Expected (2, {N_CLASSES}), got {out.shape}"


def test_transformer_forward_shape():
    model = SyntheticTransformerClassifier(J=J, F=F, n_classes=N_CLASSES)
    out = model(_randn_batch(2))
    assert out.shape == (2, N_CLASSES), f"Expected (2, {N_CLASSES}), got {out.shape}"


# ---------------------------------------------------------------------------
# predict_proba tests
# ---------------------------------------------------------------------------


def test_mlp_predict_proba():
    model = SyntheticMLPClassifier(J=J, F=F, T=T, K=K, n_classes=N_CLASSES)
    model.eval()
    with torch.no_grad():
        proba = model.predict_proba(_randn_batch(4), class_idx=0)
    assert proba.shape == (4,), f"Expected (4,), got {proba.shape}"
    assert (proba >= 0).all() and (proba <= 1).all(), "Probabilities must be in [0, 1]"


def test_cnn_predict_proba():
    model = SyntheticCNNClassifier(J=J, F=F, n_classes=N_CLASSES)
    model.eval()
    with torch.no_grad():
        proba = model.predict_proba(_randn_batch(4), class_idx=1)
    assert proba.shape == (4,), f"Expected (4,), got {proba.shape}"
    assert (proba >= 0).all() and (proba <= 1).all(), "Probabilities must be in [0, 1]"


def test_transformer_predict_proba():
    model = SyntheticTransformerClassifier(J=J, F=F, n_classes=N_CLASSES)
    model.eval()
    with torch.no_grad():
        proba = model.predict_proba(_randn_batch(4), class_idx=2)
    assert proba.shape == (4,), f"Expected (4,), got {proba.shape}"
    assert (proba >= 0).all() and (proba <= 1).all(), "Probabilities must be in [0, 1]"


# ---------------------------------------------------------------------------
# Gradient flow tests
# ---------------------------------------------------------------------------


def test_mlp_gradient_flows():
    model = SyntheticMLPClassifier(J=J, F=F, T=T, K=K, n_classes=N_CLASSES)
    x = _randn_batch(2).requires_grad_(True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None, "Gradient did not flow back to input"


def test_cnn_gradient_flows():
    model = SyntheticCNNClassifier(J=J, F=F, n_classes=N_CLASSES)
    x = _randn_batch(2).requires_grad_(True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None, "Gradient did not flow back to input"


def test_transformer_gradient_flows():
    model = SyntheticTransformerClassifier(J=J, F=F, n_classes=N_CLASSES)
    x = _randn_batch(2).requires_grad_(True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None, "Gradient did not flow back to input"


# ---------------------------------------------------------------------------
# Batch size robustness
# ---------------------------------------------------------------------------


def test_mlp_different_batch_sizes():
    model = SyntheticMLPClassifier(J=J, F=F, T=T, K=K, n_classes=N_CLASSES)
    model.eval()
    for B in (1, 4, 16):
        with torch.no_grad():
            out = model(_randn_batch(B))
        assert out.shape == (B, N_CLASSES), f"B={B}: expected ({B}, {N_CLASSES}), got {out.shape}"


# ---------------------------------------------------------------------------
# Abstract base class enforcement
# ---------------------------------------------------------------------------


def test_classifier_base_is_abstract():
    with pytest.raises(TypeError):
        Classifier()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Checkpoint existence (slow — run after training script)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_checkpoints_exist():
    for name in ("synthetic_mlp_k4.pt", "synthetic_cnn.pt", "synthetic_transformer.pt"):
        ckpt = CKPT_DIR / name
        assert ckpt.exists(), f"Checkpoint not found: {ckpt}"
        data = torch.load(ckpt, map_location="cpu", weights_only=True)
        assert "model_state_dict" in data, f"{name}: missing model_state_dict"
        assert "config" in data, f"{name}: missing config"
        assert "val_acc" in data, f"{name}: missing val_acc"
