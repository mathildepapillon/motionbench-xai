"""motionbench.imputers.frame_flow — Frame-token CondOT flow imputer (study format).

Inference-time port of the **independent validation study's** flow-matching
imputer (its ``mbxr.flow``), kept numerically identical so the real-data
checkpoints documented in ``checkpoints/README.md``
(``imputers/{carepd,esc50,ptbxl}_flow.pt``, dicts
``{state_dict, shape, num_steps[, arch]}``) reproduce the study's completions
bit-for-bit given the same random stream.

Velocity network: frame tokens ``[x_frame_flat, time_embedding]`` -> Linear
-> d_model (+ sinusoidal frame PE) -> TransformerEncoder (GELU) -> Linear ->
per-frame velocity.  Inference is a midpoint (RK2) ODE with ``num_steps``
steps and RePaint harmonisation: observed entries are projected onto the
CondOT interpolant ``(1-t) x0 + t x1`` after the midpoint and the full step,
and restored exactly at the end.

The imputer protocol and seeding conventions are identical to
:class:`~motionbench.imputers.frame_vaeac.FrameVAEACImputer`:
``impute`` / ``impute_multi`` with mask True = observed, one
``torch.manual_seed`` drawn from the caller's numpy Generator per call.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["FrameVelocityNet", "FrameFlowImputer"]

_DROPOUT = 0.1


def _sin_embed(t: Tensor, dim: int) -> Tensor:
    """(B,) scalar times -> (B, dim) sinusoidal embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(1000.0) * torch.arange(half, device=t.device).float() / max(half - 1, 1)
    )
    ang = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


def _frame_pe(T: int, d: int, device: torch.device) -> Tensor:
    """Sinusoidal frame positional encoding ``(T, d)``."""
    pos = torch.arange(T, device=device).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class FrameVelocityNet(nn.Module):
    """Frame-token velocity network (study architecture; state-dict compatible).

    Args:
        J: Number of joints / channels.
        F: Features per joint.
        d_model: Trunk width (study default 256).
        nhead: Attention heads (study default 4).
        n_layers: Trunk layers (study default 4).
        ff: Feed-forward width (study default ``4 * d_model``).
        time_dim: Time-embedding width (study default ``d_model // 2``).
    """

    def __init__(
        self,
        J: int,
        F: int,
        d_model: int = 256,
        nhead: int = 4,
        n_layers: int = 4,
        ff: int | None = None,
        time_dim: int | None = None,
    ) -> None:
        """Initialise the velocity network for ``(J, F)`` frame tokens."""
        super().__init__()
        self.J, self.F = J, F
        ff = 4 * d_model if ff is None else ff
        time_dim = d_model // 2 if time_dim is None else time_dim
        self._d_model = d_model
        self._time_dim = time_dim
        d_in = J * F
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.proj = nn.Linear(d_in + time_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=_DROPOUT,
            activation="gelu",
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.out = nn.Linear(d_model, d_in)

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        """Velocity field.

        Args:
            x_t: ``(B, T, J, F)`` interpolant state.
            t: ``(B,)`` flow times.

        Returns:
            Velocity of the same shape as ``x_t``.
        """
        B, T, J, F = x_t.shape
        temb = self.time_mlp(_sin_embed(t, self._time_dim))  # (B, time_dim)
        tok = torch.cat(
            [x_t.reshape(B, T, J * F), temb[:, None, :].expand(B, T, self._time_dim)],
            dim=-1,
        )
        h = self.proj(tok) + _frame_pe(T, self._d_model, x_t.device)
        return self.out(self.enc(h)).reshape(B, T, J, F)


class FrameFlowImputer:
    """Imputer-protocol CondOT + RePaint flow imputer.

    Args:
        J: Number of joints / channels.
        F: Features per joint.
        T: Time steps.
        num_steps: Midpoint ODE steps (study default 20).
        device: Torch device string.
        repaint: Harmonise observed coordinates during integration (study
            convention; ``False`` integrates unconditionally and pastes the
            observed block only at the end).
        **arch: Architecture overrides forwarded to :class:`FrameVelocityNet`.
    """

    def __init__(
        self,
        J: int,
        F: int,
        T: int,
        num_steps: int = 20,
        device: str = "cpu",
        repaint: bool = True,
        **arch: int,
    ) -> None:
        """Initialise the imputer and its velocity network on ``device``."""
        self.J, self.F, self.T = J, F, T
        self.num_steps = num_steps
        self.repaint = bool(repaint)
        self.device = torch.device(device)
        self.net = FrameVelocityNet(J, F, **arch).to(self.device)

    @torch.no_grad()
    def _integrate(self, x1n: Tensor, obsn: Tensor, x0: Tensor) -> Tensor:
        """Midpoint ODE from ``x0`` to the (optionally harmonised) completion."""
        B = x1n.shape[0]

        def harmonize(xk: Tensor, t: float) -> Tensor:
            """Project observed entries onto the CondOT interpolant at time ``t``."""
            if not self.repaint:
                return xk
            path = (1.0 - t) * x0 + t * x1n
            return torch.where(obsn, path, xk)

        dt = 1.0 / self.num_steps
        xk = x0.clone()
        self.net.eval()
        for k in range(self.num_steps):
            tk, tmid, tnext = k * dt, (k + 0.5) * dt, (k + 1) * dt
            v1 = self.net(xk, torch.full((B,), tk, device=self.device))
            xmid = harmonize(xk + 0.5 * dt * v1, tmid)
            v2 = self.net(xmid, torch.full((B,), tmid, device=self.device))
            xk = harmonize(xk + dt * v2, tnext)
        return torch.where(obsn, x1n, xk)

    @torch.no_grad()
    def impute(
        self,
        x: npt.NDArray[np.float32],
        mask: npt.NDArray[np.bool_],
        n: int,
        rng: np.random.Generator,
    ) -> npt.NDArray[np.float32]:
        """Draw ``n`` completions of one sequence for one mask.

        Args:
            x: ``(J, F, T)`` sequence.
            mask: ``(J, F, T)`` bool, True = observed.
            n: Number of completions.
            rng: Numpy Generator; one 31-bit torch seed is drawn per call.

        Returns:
            ``(n, J, F, T)`` float32 completions.
        """
        torch.manual_seed(int(rng.integers(1 << 31)))
        x1 = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self.device)
        obs = torch.from_numpy(np.asarray(mask, dtype=bool)).to(self.device)
        x1n = x1.permute(2, 0, 1).unsqueeze(0).expand(n, -1, -1, -1).contiguous()
        obsn = obs.permute(2, 0, 1).unsqueeze(0).expand(n, -1, -1, -1).contiguous()
        x0 = torch.randn_like(x1n)
        xk = self._integrate(x1n, obsn, x0)
        out = xk.permute(0, 2, 3, 1).cpu().numpy()
        mask_b = np.asarray(mask, dtype=bool)
        out[:, mask_b] = np.asarray(x, dtype=np.float32)[mask_b]
        return out.astype(np.float32)

    @torch.no_grad()
    def impute_multi(
        self,
        x: npt.NDArray[np.float32],
        masks: npt.NDArray[np.bool_],
        n: int,
        rng: np.random.Generator,
        chunk: int = 4096,
    ) -> npt.NDArray[np.float32]:
        """Completions for many coalition masks in chunked ODE integrations.

        Args:
            x: ``(J, F, T)`` sequence.
            masks: ``(n_masks, J, F, T)`` bool, True = observed.
            n: Completions per mask.
            rng: Numpy Generator; one 31-bit torch seed is drawn per call.
            chunk: Approximate rows per GPU pass (``chunk // n`` masks).

        Returns:
            ``(n_masks, n, J, F, T)`` float32; observed entries restored.
        """
        torch.manual_seed(int(rng.integers(1 << 31)))
        masks_b = np.asarray(masks, dtype=bool)
        nM = len(masks_b)
        x1 = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self.device)
        x1f = x1.permute(2, 0, 1)  # (T, J, F)
        mt = torch.from_numpy(masks_b).to(self.device).permute(0, 3, 1, 2)
        out = np.empty((nM, n) + tuple(np.shape(x)), dtype=np.float32)
        per = max(1, chunk // max(n, 1))
        for s in range(0, nM, per):
            mb = mt[s : s + per]
            b = mb.shape[0]
            x1n = x1f[None, None].expand(b, n, -1, -1, -1).reshape(b * n, *x1f.shape)
            obsn = mb[:, None].expand(b, n, -1, -1, -1).reshape(b * n, *x1f.shape)
            x1n = x1n.contiguous()
            obsn = obsn.contiguous()
            x0 = torch.randn_like(x1n)
            xk = self._integrate(x1n, obsn, x0)
            out[s : s + b] = xk.permute(0, 2, 3, 1).cpu().numpy().reshape(b, n, *np.shape(x))
        xnp = np.asarray(x, dtype=np.float32)
        for i in range(nM):
            out[i][:, masks_b[i]] = xnp[masks_b[i]]
        return out

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str = "cpu",
        num_steps: int | None = None,
        repaint: bool = True,
    ) -> FrameFlowImputer:
        """Load a study-format checkpoint ``{state_dict, shape[, num_steps, arch]}``.

        Args:
            path: Checkpoint path.
            device: Torch device string.
            num_steps: Override the checkpoint's ODE step count.
            repaint: RePaint harmonisation flag.

        Returns:
            Imputer in eval mode with weights loaded strictly.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        J, F, T = ckpt["shape"]
        steps = num_steps if num_steps is not None else ckpt.get("num_steps", 20)
        arch_raw = dict(ckpt.get("arch", {}))
        arch = {
            "d_model": arch_raw.get("d_model", 256),
            "nhead": arch_raw.get("nhead", 4),
            "n_layers": arch_raw.get("num_layers", 4),
            "ff": arch_raw.get("ff"),
            "time_dim": arch_raw.get("time_dim"),
        }
        imp = cls(
            J,
            F,
            T,
            num_steps=steps,
            device=device,
            repaint=repaint,
            **{k: v for k, v in arch.items() if v is not None},
        )
        imp.net.load_state_dict(ckpt["state_dict"], strict=True)
        imp.net.eval()
        return imp
