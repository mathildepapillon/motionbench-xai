"""Tests for motionbench.attribution.lrp.LRPAttributor.

Test suite covers:
- Output shape: attribute() returns (M,) float32.
- All three supported LRP rules run without error.
- LRP conservation: Σ relevance ≈ model output (slow test).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from motionbench.attribution.lrp import LRPAttributor
from tests.conftest import F, J, M, T

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _TinyLinear(nn.Module):
    """Flatten + bias-free linear layer: (B, J, F, T) → (B,)."""

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(J * F * T, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: ``(B, J, F, T)`` float32 input tensor.

        Returns:
            ``(B,)`` float32 output tensor.
        """
        return self.linear(self.flatten(x)).squeeze(-1)


class _MockPlayers:
    """Minimal PlayerSet splitting T frames into M equal-width windows."""

    n_players: int = M
    shape: tuple[int, int, int] = (J, F, T)

    def coalition_mask(self, z: Tensor) -> Tensor:
        """Expand coalition indicator to (J, F, T) bool mask."""
        ws = T // M
        mask = torch.zeros(J, F, T, dtype=torch.bool)
        for k in range(M):
            if z[k]:
                mask[:, :, k * ws : (k + 1) * ws] = True
        return mask

    def aggregate(self, phi_coords: Tensor) -> Tensor:
        """Sum per-coordinate attributions within each temporal window.

        Args:
            phi_coords: ``(J, F, T)`` float tensor.

        Returns:
            ``(M,)`` float tensor.
        """
        ws = T // M
        phi = torch.zeros(M)
        for k in range(M):
            phi[k] = phi_coords[:, :, k * ws : (k + 1) * ws].sum()
        return phi


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_lrp_shape(x_sample: Tensor) -> None:
    """attribute() returns (M,) float32 tensor for a tiny nn.Linear model."""
    torch.manual_seed(0)
    model = _TinyLinear()
    attributor = LRPAttributor(classifier=model, rule="epsilon")
    players = _MockPlayers()

    phi = attributor.attribute(x_sample, players, target=0)

    assert phi.shape == (M,), f"Expected ({M},), got {phi.shape}"
    assert phi.dtype == torch.float32, f"Expected float32, got {phi.dtype}"


def test_lrp_rules(x_sample: Tensor) -> None:
    """All three LRP rules instantiate and produce (M,) output without error."""
    torch.manual_seed(0)
    players = _MockPlayers()

    for rule in ("epsilon", "gamma", "alpha_beta"):
        model = _TinyLinear()
        attributor = LRPAttributor(classifier=model, rule=rule)  # type: ignore[arg-type]
        phi = attributor.attribute(x_sample, players, target=0)
        assert phi.shape == (M,), f"rule={rule!r}: expected ({M},), got {phi.shape}"


def test_lrp_requires_gradient_property() -> None:
    """LRPAttributor.requires_gradient must be True."""
    model = _TinyLinear()
    attributor = LRPAttributor(classifier=model)
    assert attributor.requires_gradient is True


def test_lrp_invalid_rule() -> None:
    """LRPAttributor raises ValueError for unsupported rule strings."""
    model = _TinyLinear()
    with pytest.raises(ValueError, match="rule must be one of"):
        LRPAttributor(classifier=model, rule="invalid")  # type: ignore[arg-type]


def test_lrp_non_module_classifier_raises() -> None:
    """LRPAttributor raises TypeError when given a plain callable."""
    plain_fn = lambda x: x.mean(dim=(1, 2, 3))  # noqa: E731
    with pytest.raises(TypeError, match="nn.Module"):
        LRPAttributor(classifier=plain_fn)  # type: ignore[arg-type]


@pytest.mark.slow
def test_lrp_conservation(x_sample: Tensor) -> None:
    """Σ LRP relevances ≈ model output (LRP conservation property).

    For a bias-free linear layer with the epsilon rule, the relevance sum
    satisfies::

        Σ R_i = f(x)^2 / (f(x) + ε)  ≈  f(x)   when ε ≪ |f(x)|.

    We verify the ratio |Σ R_i − f(x)| / |f(x)| < 1e-3.
    """
    torch.manual_seed(0)
    model = _TinyLinear()
    epsilon = 1e-9  # tiny stabiliser for tight conservation
    attributor = LRPAttributor(classifier=model, rule="epsilon", epsilon=epsilon)
    players = _MockPlayers()

    x_in = x_sample.unsqueeze(0).float()  # (1, J, F, T)
    with torch.no_grad():
        output = model(x_in).item()

    phi = attributor.attribute(x_sample, players, target=0)
    relevance_sum = phi.sum().item()

    if abs(output) > 1e-6:
        rel_error = abs(relevance_sum - output) / abs(output)
        assert rel_error < 1e-3, (
            f"Conservation failed: Σ R={relevance_sum:.6f}, f(x)={output:.6f}, "
            f"rel_error={rel_error:.4e}"
        )
