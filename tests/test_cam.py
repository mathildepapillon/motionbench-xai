"""Tests for GradCAMAttributor and AttentionRolloutAttributor.

Uses tiny inline synthetic models — no CARE-PD classifiers required.
All models are defined within this file so the tests are fully self-contained.

Canonical test shapes (matching conftest.py):
    J=5, F=3, T=16, M=4
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from motionbench.attribution.attention_rollout import AttentionRolloutAttributor
from motionbench.attribution.grad_cam import GradCAMAttributor
from tests.conftest import J, F, M, T


# ---------------------------------------------------------------------------
# Tiny inline models
# ---------------------------------------------------------------------------


class _TinyCNN(nn.Module):
    """J=5, F=3, T=16 → (B, n_classes) logits. Exposes conv1 for GradCAM.

    Flattens (J, F) into channels for Conv1d so the temporal dimension T is
    preserved through the convolutional layer (padding=1, stride=1).
    """

    def __init__(self, n_classes: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(J * F, 8, kernel_size=3, padding=1)  # 15 → 8 channels, T preserved
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(8, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: ``(B, J, F, T)`` float32 input.

        Returns:
            ``(B, n_classes)`` logits.
        """
        B, Jj, Ff, Tt = x.shape
        x = x.view(B, Jj * Ff, Tt)          # (B, 15, T)
        x = torch.relu(self.conv1(x))        # (B, 8, T)
        x = self.pool(x).squeeze(-1)         # (B, 8)
        return self.fc(x)                    # (B, n_classes)


class _TinyTransformer(nn.Module):
    """Single-layer transformer; treats T as sequence length, J*F as embedding.

    Stores attention weights for rollout testing.  The single
    ``nn.MultiheadAttention`` layer captures per-head weights during the
    forward pass and returns them via ``get_attention_weights()``.

    Attributes:
        _attn_cache: Cached attention weights from the most recent forward
            pass; each element has shape ``(B, n_heads, T, T)``.
    """

    def __init__(self, n_classes: int = 3, n_heads: int = 3) -> None:
        super().__init__()
        self._embed_dim = J * F          # 15 — must be divisible by n_heads
        self._attn_cache: list[Tensor] = []
        self.attn = nn.MultiheadAttention(
            self._embed_dim, n_heads, batch_first=True
        )
        self.fc = nn.Linear(self._embed_dim, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: ``(B, J, F, T)`` float32 input.

        Returns:
            ``(B, n_classes)`` logits.
        """
        B, Jj, Ff, Tt = x.shape
        seq = x.view(B, Jj * Ff, Tt).permute(0, 2, 1)  # (B, T, J*F)
        out, w = self.attn(
            seq, seq, seq,
            need_weights=True,
            average_attn_weights=False,  # keep per-head weights → (B, H, T, T)
        )
        self._attn_cache = [w]                           # one layer
        pooled = out.mean(dim=1)                         # (B, J*F)
        return self.fc(pooled)                           # (B, n_classes)

    def get_attention_weights(self) -> list[Tensor]:
        """Return cached per-layer attention weights.

        Returns:
            List with one element of shape ``(B, H, T, T)``.
        """
        return self._attn_cache


class _NoAttnModel(nn.Module):
    """Classifier with no ``get_attention_weights`` method."""

    def forward(self, x: Tensor) -> Tensor:
        return x.mean(dim=(1, 2, 3))


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


class _TemporalPlayers:
    """Simple M-window temporal player set for testing.

    Divides the T time steps into M equal windows.
    """

    n_players = M
    shape = (J, F, T)

    def coalition_mask(self, z: Tensor) -> Tensor:
        ws = T // M
        mask = torch.zeros(J, F, T, dtype=torch.bool)
        for k in range(M):
            if z[k]:
                mask[:, :, k * ws : (k + 1) * ws] = True
        return mask

    def aggregate(self, phi_coords: Tensor) -> Tensor:
        ws = T // M
        phi = torch.zeros(M)
        for k in range(M):
            phi[k] = phi_coords[:, :, k * ws : (k + 1) * ws].sum()
        return phi


@pytest.fixture()
def players() -> _TemporalPlayers:
    """M-window temporal player set."""
    return _TemporalPlayers()


@pytest.fixture()
def x_jft() -> Tensor:
    """Single ``(J, F, T)`` float32 input sample."""
    return torch.randn(J, F, T)


@pytest.fixture()
def cnn() -> _TinyCNN:
    """Tiny CNN with deterministic weights."""
    torch.manual_seed(0)
    return _TinyCNN()


@pytest.fixture()
def transformer() -> _TinyTransformer:
    """Tiny transformer with deterministic weights."""
    torch.manual_seed(0)
    return _TinyTransformer()


# ---------------------------------------------------------------------------
# GradCAMAttributor tests
# ---------------------------------------------------------------------------


def test_gradcam_requires_gradient() -> None:
    """GradCAMAttributor.requires_gradient must be True."""
    assert GradCAMAttributor.requires_gradient is True


def test_gradcam_shape(x_jft: Tensor, cnn: _TinyCNN, players: _TemporalPlayers) -> None:
    """attribute() must return a (M,) shaped tensor."""
    attributor = GradCAMAttributor(cnn, layer=cnn.conv1)
    phi = attributor.attribute(x_jft, players)
    assert phi.shape == (M,), f"Expected ({M},), got {phi.shape}"


def test_gradcam_no_nan(x_jft: Tensor, cnn: _TinyCNN, players: _TemporalPlayers) -> None:
    """GradCAM output must not contain NaN or Inf values."""
    attributor = GradCAMAttributor(cnn, layer=cnn.conv1)
    phi = attributor.attribute(x_jft, players)
    assert torch.isfinite(phi).all(), f"GradCAM attribution contains NaN or Inf: {phi}"


def test_gradcam_interpolate_modes(
    x_jft: Tensor, cnn: _TinyCNN, players: _TemporalPlayers
) -> None:
    """Both interpolate_mode values should produce valid (M,) outputs."""
    for mode in ("nearest", "bilinear"):
        attributor = GradCAMAttributor(cnn, layer=cnn.conv1, interpolate_mode=mode)  # type: ignore[arg-type]
        phi = attributor.attribute(x_jft, players)
        assert phi.shape == (M,), f"mode={mode}: expected ({M},), got {phi.shape}"
        assert torch.isfinite(phi).all(), f"mode={mode}: NaN/Inf in {phi}"


def test_gradcam_different_targets(
    x_jft: Tensor, cnn: _TinyCNN, players: _TemporalPlayers
) -> None:
    """attribute() must accept any target class index without error."""
    attributor = GradCAMAttributor(cnn, layer=cnn.conv1)
    for target in range(3):  # _TinyCNN has n_classes=3
        phi = attributor.attribute(x_jft, players, target=target)
        assert phi.shape == (M,)


# ---------------------------------------------------------------------------
# AttentionRolloutAttributor tests
# ---------------------------------------------------------------------------


def test_attention_rollout_shape(
    x_jft: Tensor, transformer: _TinyTransformer, players: _TemporalPlayers
) -> None:
    """attribute() must return a (M,) shaped tensor."""
    attributor = AttentionRolloutAttributor(transformer)
    phi = attributor.attribute(x_jft, players)
    assert phi.shape == (M,), f"Expected ({M},), got {phi.shape}"


def test_attention_rollout_no_nan(
    x_jft: Tensor, transformer: _TinyTransformer, players: _TemporalPlayers
) -> None:
    """Attention rollout output must not contain NaN or Inf values."""
    attributor = AttentionRolloutAttributor(transformer)
    phi = attributor.attribute(x_jft, players)
    assert torch.isfinite(phi).all(), f"Rollout attribution contains NaN or Inf: {phi}"


def test_attention_rollout_no_attention_weights(
    x_jft: Tensor, players: _TemporalPlayers
) -> None:
    """AttributeError must be raised when model has no get_attention_weights."""
    no_attn_model = _NoAttnModel()
    attributor = AttentionRolloutAttributor(no_attn_model)
    with pytest.raises(AttributeError, match="get_attention_weights"):
        attributor.attribute(x_jft, players)


def test_attention_rollout_requires_gradient() -> None:
    """AttentionRolloutAttributor.requires_gradient must be False."""
    assert AttentionRolloutAttributor.requires_gradient is False


def test_attention_rollout_is_deterministic(
    x_jft: Tensor, transformer: _TinyTransformer, players: _TemporalPlayers
) -> None:
    """Rollout must be deterministic for the same input."""
    attributor = AttentionRolloutAttributor(transformer)
    phi1 = attributor.attribute(x_jft, players)
    phi2 = attributor.attribute(x_jft, players)
    assert torch.allclose(phi1, phi2), "Rollout is not deterministic"
