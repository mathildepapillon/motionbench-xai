"""motionbench.imputers.frame_vaeac — Frame-token transformer VAEAC (checkpoint format).

Inference-time implementation matching the released real-data VAEAC
checkpoints documented in ``checkpoints/README.md``
(``imputers/{carepd,esc50,ptbxl}_vaeac.pt``, dicts
``{state_dict, shape, arch}``): given the same random stream it reproduces
the reference training runs' completions bit-for-bit.

Architecture (per-frame tokens): full encoder q(z|x, m), prior encoder
p(z|x_obs, m) and decoder p(x|z, x_obs, m) are each a TransformerEncoder
trunk (default d_model=256, nhead=8, 2 layers, ff=512, dropout 0.1) with
sinusoidal frame positional encoding; per-frame latent (default 64);
Gaussian output head with one global learnable ``log_sigma``.  The internal
layout is frame-major ``(B, T, J, F)``; the public API speaks the
motionbench ``(J, F, T)`` convention.

The imputer protocol::

    impute(x, mask, n, rng)        -> (n, J, F, T) float32
    impute_multi(x, masks, n, rng) -> (n_masks, n, J, F, T) float32

with ``mask`` True = observed.  Each call draws ``torch.manual_seed`` once
from ``rng`` (``rng.integers(1 << 31)``), so a per-sequence
``np.random.default_rng([seed_base, fold, i])`` stream reproduces the
reference draws exactly.  Completions are produced in whatever coordinate
space ``x`` is given in (raw cache space for CARE-PD/ESC-50, z-scored cache
space for PTB-XL).
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

__all__ = ["FrameVAEACModel", "FrameVAEACImputer"]

_DROPOUT = 0.1


def _frame_pe(T: int, d: int, device: torch.device) -> Tensor:
    """Sinusoidal frame positional encoding ``(T, d)``."""
    pos = torch.arange(T, device=device).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class _Trunk(nn.Module):
    """Linear projection + TransformerEncoder over frame tokens."""

    def __init__(self, feat_in: int, d_model: int, nhead: int, n_layers: int, ff: int) -> None:
        """Initialise the input projection and transformer encoder."""
        super().__init__()
        self.proj = nn.Linear(feat_in, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=_DROPOUT,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self._d_model = d_model

    def forward(self, tok: Tensor) -> Tensor:
        """Project, add frame positional encoding, and encode ``(B, T, feat_in)`` tokens."""
        h = self.proj(tok)
        h = h + _frame_pe(h.shape[1], self._d_model, h.device)
        return self.enc(h)


class _EncoderHead(nn.Module):
    """Trunk + (mu, logvar) heads."""

    def __init__(
        self, feat_in: int, d_model: int, d_latent: int, nhead: int, n_layers: int, ff: int
    ) -> None:
        """Initialise the trunk and latent parameter heads."""
        super().__init__()
        self.trunk = _Trunk(feat_in, d_model, nhead, n_layers, ff)
        self.mu = nn.Linear(d_model, d_latent)
        self.logvar = nn.Linear(d_model, d_latent)

    def forward(self, tok: Tensor) -> tuple[Tensor, Tensor]:
        """Return per-frame ``(mu, logvar)`` for ``(B, T, feat_in)`` tokens."""
        h = self.trunk(tok)
        return self.mu(h), self.logvar(h)


class _DecoderHead(nn.Module):
    """Trunk + output head."""

    def __init__(
        self, feat_in: int, out_dim: int, d_model: int, nhead: int, n_layers: int, ff: int
    ) -> None:
        """Initialise the trunk and output projection."""
        super().__init__()
        self.trunk = _Trunk(feat_in, d_model, nhead, n_layers, ff)
        self.out = nn.Linear(d_model, out_dim)

    def forward(self, tok: Tensor) -> Tensor:
        """Decode ``(B, T, feat_in)`` tokens to ``(B, T, out_dim)``."""
        return self.out(self.trunk(tok))


class FrameVAEACModel(nn.Module):
    """Frame-token VAEAC (checkpoint architecture; state-dict compatible).

    Args:
        J: Number of joints / channels.
        F: Features per joint.
        d_model: Trunk width (reference default 256).
        d_latent: Per-frame latent dimension (reference default 64).
        nhead: Attention heads (reference default 8).
        n_layers: Trunk layers (reference default 2).
        ff: Feed-forward width (reference default ``2 * d_model``).
    """

    def __init__(
        self,
        J: int,
        F: int,
        d_model: int = 256,
        d_latent: int = 64,
        nhead: int = 8,
        n_layers: int = 2,
        ff: int | None = None,
    ) -> None:
        """Initialise the encoders, decoder, and global ``log_sigma``."""
        super().__init__()
        self.J, self.F = J, F
        ff = 2 * d_model if ff is None else ff
        d_in = J * F
        self.full_enc = _EncoderHead(2 * d_in, d_model, d_latent, nhead, n_layers, ff)
        self.prior_enc = _EncoderHead(2 * d_in, d_model, d_latent, nhead, n_layers, ff)
        self.decoder = _DecoderHead(d_latent + 2 * d_in, d_in, d_model, nhead, n_layers, ff)
        self.log_sigma = nn.Parameter(torch.zeros(1))

    def _tok_prior(self, x: Tensor, obs: Tensor) -> Tensor:
        B, T = x.shape[:2]
        o = obs.float()
        return torch.cat([(x * o).reshape(B, T, -1), o.reshape(B, T, -1)], -1)

    def _tok_dec(self, z: Tensor, x: Tensor, obs: Tensor) -> Tensor:
        B, T = x.shape[:2]
        o = obs.float()
        return torch.cat([z, (x * o).reshape(B, T, -1), o.reshape(B, T, -1)], -1)

    @torch.no_grad()
    def complete(self, x: Tensor, obs: Tensor, n: int) -> Tensor:
        """Draw ``n`` conditional completions per input row.

        Args:
            x: ``(B, T, J, F)`` inputs (frame-major).
            obs: ``(B, T, J, F)`` bool observation masks (True = observed).
            n: Completions per input row.

        Returns:
            ``(B*n, T, J, F)``; observed entries copied verbatim.
        """
        x_r = x.repeat_interleave(n, dim=0)
        obs_r = obs.repeat_interleave(n, dim=0)
        mu_p, lv_p = self.prior_enc(self._tok_prior(x_r, obs_r))
        z = mu_p + torch.exp(0.5 * lv_p) * torch.randn_like(mu_p)
        x_hat = self.decoder(self._tok_dec(z, x_r, obs_r)).reshape_as(x_r)
        x_hat = x_hat + torch.exp(self.log_sigma) * torch.randn_like(x_hat)
        return torch.where(obs_r, x_r, x_hat)


class FrameVAEACImputer:
    """Imputer-protocol wrapper around :class:`FrameVAEACModel`.

    Args:
        J: Number of joints / channels.
        F: Features per joint.
        T: Time steps.
        device: Torch device string.
        **arch: Architecture overrides forwarded to :class:`FrameVAEACModel`
            (``d_model``, ``d_latent``, ``nhead``, ``n_layers``, ``ff``).
    """

    def __init__(self, J: int, F: int, T: int, device: str = "cpu", **arch: int) -> None:
        """Initialise the model on ``device`` for ``(J, F, T)`` sequences."""
        self.J, self.F, self.T = J, F, T
        self.device = torch.device(device)
        self.model = FrameVAEACModel(J, F, **arch).to(self.device)

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
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self.device)
        mt = torch.from_numpy(np.asarray(mask, dtype=bool)).to(self.device)
        x_f = xt.permute(2, 0, 1).unsqueeze(0)  # (1, T, J, F)
        m_f = mt.permute(2, 0, 1).unsqueeze(0)
        self.model.eval()
        out = self.model.complete(x_f, m_f, n)  # (n, T, J, F)
        out_np = out.permute(0, 2, 3, 1).cpu().numpy()  # (n, J, F, T)
        mask_b = np.asarray(mask, dtype=bool)
        out_np[:, mask_b] = np.asarray(x, dtype=np.float32)[mask_b]
        return out_np.astype(np.float32)

    @torch.no_grad()
    def impute_multi(
        self,
        x: npt.NDArray[np.float32],
        masks: npt.NDArray[np.bool_],
        n: int,
        rng: np.random.Generator,
        chunk: int = 4096,
    ) -> npt.NDArray[np.float32]:
        """Completions for many coalition masks in chunked decoder passes.

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
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self.device)
        x_f = xt.permute(2, 0, 1)  # (T, J, F)
        mt = torch.from_numpy(masks_b).to(self.device).permute(0, 3, 1, 2)
        out = np.empty((nM, n) + tuple(np.shape(x)), dtype=np.float32)
        self.model.eval()
        per = max(1, chunk // max(n, 1))
        for s in range(0, nM, per):
            mb = mt[s : s + per]
            b = mb.shape[0]
            xb = x_f[None].expand(b, -1, -1, -1).contiguous()
            comp = self.model.complete(xb, mb.contiguous(), n)  # (b*n, T, J, F)
            out[s : s + b] = comp.permute(0, 2, 3, 1).cpu().numpy().reshape(b, n, *np.shape(x))
        xnp = np.asarray(x, dtype=np.float32)
        for i in range(nM):
            out[i][:, masks_b[i]] = xnp[masks_b[i]]
        return out

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> FrameVAEACImputer:
        """Load a reference-format checkpoint ``{state_dict, shape[, arch]}``.

        Handles both state-dict naming variants of the checkpoint lineage: the
        player-set sweep's ``decoder.trunk / decoder.out`` and the real-data
        checkpoints' top-level ``dec_trunk / dec_out`` (remapped on load).

        Args:
            path: Checkpoint path.
            device: Torch device string.

        Returns:
            Imputer in eval mode with weights loaded strictly.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        J, F, T = ckpt["shape"]
        arch_raw = dict(ckpt.get("arch", {}))
        arch = {
            "d_model": arch_raw.get("d_model", 256),
            "d_latent": arch_raw.get("d_latent", 64),
            "nhead": arch_raw.get("nhead", 8),
            "n_layers": arch_raw.get("num_layers", 2),
            "ff": arch_raw.get("ff"),
        }
        imp = cls(J, F, T, device=device, **{k: v for k, v in arch.items() if v is not None})
        sd: dict[str, Tensor] = {}
        for k, v in ckpt["state_dict"].items():
            if k.startswith("dec_trunk."):
                sd["decoder.trunk." + k[len("dec_trunk.") :]] = v
            elif k.startswith("dec_out."):
                sd["decoder.out." + k[len("dec_out.") :]] = v
            else:
                sd[k] = v
        imp.model.load_state_dict(sd, strict=True)
        imp.model.eval()
        return imp
