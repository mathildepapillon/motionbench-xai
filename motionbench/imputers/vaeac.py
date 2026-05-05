"""motionbench.imputers.vaeac — VAEAC on-manifold imputer.

Ports the Transformer-backbone VAEAC (Variational Autoencoder with Arbitrary
Conditioning) from the CARE-PD project to the motionbench :class:`BaseImputer`
interface.

Architecture (Ivanov et al. 2019):

* ``full_encoder``  ``q(z | x, mask)`` — training only; sees the full sequence.
* ``prior_encoder`` ``p(z | x_obs, mask)`` — train + inference; sees only observed
  entries and the mask.
* ``decoder``       ``p(x | z, x_obs, mask)`` — reconstructs hidden entries.

Training loss is the per-frame ELBO::

    L = E_q[-log p(x_hid | z)] + KL(q(z|x,mask) || p(z|x_obs,mask))

Shape conventions (motionbench):

* Per-sample input  ``(J, F, T)`` float32.
* Element mask      ``(J, F, T)`` bool — ``True`` = observed.
* Batch output      ``(n_samples, J, F, T)`` float32.

Internally the model uses CARE-PD layout ``(B, T, J, C)`` where ``C = F``.
:func:`sample_training_mask` is exported so ``scripts/train_vaeac.py`` can
import it without reaching into private modules.

References
----------
Ivanov et al. (2019) "VAEAC: Missing Data Imputation with VAEAC."
Olsen et al. (2022) "Using Shapley Values and Variational Autoencoders To
Explain Predictions from Neural Networks for Short-Term Wind Power
Forecasting." JMLR 23(1).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from motionbench.imputers.base import BaseImputer

if TYPE_CHECKING:
    from motionbench.data.base import BaseDataset

__all__ = ["VAEACImputer", "sample_training_mask"]


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------


class _FramePositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding over the frame (time) dimension.

    Registered as a non-persistent buffer so it moves with ``.to(device)``
    but is never saved in checkpoints.

    Args:
        d_model: Embedding dimension.
        max_len: Maximum supported sequence length.
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("_pe", pe, persistent=False)
        self.max_len = max_len

    def forward(self, T: int) -> Tensor:
        """Return positional encoding for the first *T* frames.

        Args:
            T: Number of frames (must be ≤ ``max_len``).

        Returns:
            ``(T, d_model)`` positional encoding.

        Raises:
            ValueError: if ``T > max_len``.
        """
        if self.max_len < T:
            raise ValueError(
                f"Sequence length {T} exceeds FramePositionalEncoding.max_len={self.max_len}"
            )
        pe: Tensor = self._pe  # type: ignore[assignment]
        return pe[:T]


# ---------------------------------------------------------------------------
# Mask sampling (used by training script and smoke test)
# ---------------------------------------------------------------------------


def sample_training_mask(
    B: int,
    T: int,
    J: int,
    C: int,
    device: torch.device,
    *,
    p_temporal: float = 0.40,
    p_spatial: float = 0.40,
    p_element: float = 0.20,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample per-batch-element observation masks of shape ``(B, T, J, C)``.

    For each batch item one of three masking strategies is chosen:

    * *temporal* (prob ``p_temporal``): a uniformly-random fraction of frames
      is observed; observed frames span all joints and coords.
    * *spatial* (prob ``p_spatial``): a uniformly-random fraction of joints
      is observed; observed joints span all frames and coords.
    * *element* (prob ``p_element``): independent per-``(t,j,c)`` Bernoulli
      with a randomly drawn probability.

    The observation fraction is drawn from ``U(0, 1)`` independently per item.

    Args:
        B: Batch size.
        T: Number of frames.
        J: Number of joints.
        C: Number of coordinates per joint (``F`` in motionbench notation).
        device: Torch device.
        p_temporal: Probability of choosing temporal masking.
        p_spatial: Probability of choosing spatial masking.
        p_element: Probability of choosing element-wise masking.
        generator: Optional :class:`torch.Generator` for reproducibility.

    Returns:
        ``(B, T, J, C)`` bool tensor; ``True`` = observed.

    Raises:
        ValueError: if the three probabilities do not sum to 1.
    """
    if abs(p_temporal + p_spatial + p_element - 1.0) > 1e-6:
        raise ValueError("Mask-type probabilities must sum to 1.")
    g = generator
    kind = torch.rand(B, device=device, generator=g)
    frac = torch.rand(B, device=device, generator=g)

    obs = torch.empty(B, T, J, C, device=device, dtype=torch.bool)
    for b in range(B):
        f = float(frac[b].item())
        k = float(kind[b].item())
        if k < p_temporal:
            t_keep = torch.rand(T, device=device, generator=g) < f
            m = t_keep.view(T, 1, 1).expand(T, J, C)
        elif k < p_temporal + p_spatial:
            j_keep = torch.rand(J, device=device, generator=g) < f
            m = j_keep.view(1, J, 1).expand(T, J, C)
        else:
            m = torch.rand(T, J, C, device=device, generator=g) < f
        obs[b] = m
    return obs


# ---------------------------------------------------------------------------
# Transformer trunk
# ---------------------------------------------------------------------------


def _nhead_for(d_model: int) -> int:
    """Return the largest of {8, 4, 2, 1} that evenly divides *d_model*."""
    for nh in (8, 4, 2, 1):
        if d_model % nh == 0:
            return nh
    return 1  # pragma: no cover


class _TransformerTrunk(nn.Module):
    """Shared per-frame transformer encoder used by all three VAEAC subnets.

    Args:
        d_model: Model dimension.
        nhead: Number of attention heads.
        num_layers: Number of :class:`nn.TransformerEncoderLayer` blocks.
        ff_dim: Feed-forward inner dimension.
        dropout: Dropout rate.
        max_len: Maximum sequence length for the positional encoding buffer.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()
        self.pos = _FramePositionalEncoding(d_model, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, tok: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        """Apply positional encoding and transformer layers.

        Args:
            tok: ``(B, T, d_model)`` token tensor.
            pad_mask: Optional ``(B, T)`` bool; ``True`` = real frame.

        Returns:
            ``(B, T, d_model)`` contextual features.
        """
        T = tok.shape[1]
        tok = tok + self.pos(T).to(tok.dtype)
        src_kp = None if pad_mask is None else ~pad_mask
        result: Tensor = self.encoder(tok, src_key_padding_mask=src_kp)
        return result


# ---------------------------------------------------------------------------
# Encoder and decoder heads
# ---------------------------------------------------------------------------


class _EncoderHead(nn.Module):
    """Input projection → transformer → (μ, log σ²) per frame.

    Args:
        input_feat_dim: Flattened feature dimension of the input token.
        d_model: Transformer model dimension.
        nhead: Attention heads.
        num_layers: Transformer layers.
        ff_dim: Feed-forward inner dimension.
        dropout: Dropout rate.
        d_latent: Per-frame latent dimension.
        max_len: Maximum sequence length.
    """

    def __init__(
        self,
        input_feat_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        d_latent: int,
        max_len: int,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(input_feat_dim, d_model)
        self.trunk = _TransformerTrunk(d_model, nhead, num_layers, ff_dim, dropout, max_len)
        self.mu_head = nn.Linear(d_model, d_latent)
        self.logvar_head = nn.Linear(d_model, d_latent)
        self.apply(_init_linear)

    def forward(
        self, feat: Tensor, pad_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Encode token features into latent distribution parameters.

        Args:
            feat: ``(B, T, input_feat_dim)`` token features.
            pad_mask: Optional ``(B, T)`` bool; ``True`` = real frame.

        Returns:
            Tuple of ``(mu, logvar)`` each ``(B, T, d_latent)``.
        """
        h = self.trunk(self.in_proj(feat), pad_mask)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(min=-8.0, max=4.0)
        return mu, logvar


class _Decoder(nn.Module):
    """Input projection → transformer → linear output.

    Args:
        input_feat_dim: Flattened feature dimension of the input token.
        output_dim: Raw output dimension (``n_joints * n_coords`` for the
            scalar-head variant).
        d_model: Transformer model dimension.
        nhead: Attention heads.
        num_layers: Transformer layers.
        ff_dim: Feed-forward inner dimension.
        dropout: Dropout rate.
        max_len: Maximum sequence length.
    """

    def __init__(
        self,
        input_feat_dim: int,
        output_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(input_feat_dim, d_model)
        self.trunk = _TransformerTrunk(d_model, nhead, num_layers, ff_dim, dropout, max_len)
        self.out_proj = nn.Linear(d_model, output_dim)
        self.apply(_init_linear)

    def forward(self, feat: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        """Decode token features into raw predictions.

        Args:
            feat: ``(B, T, input_feat_dim)`` token features.
            pad_mask: Optional ``(B, T)`` bool; ``True`` = real frame.

        Returns:
            ``(B, T, output_dim)`` raw decoder output.
        """
        result: Tensor = self.out_proj(self.trunk(self.in_proj(feat), pad_mask))
        return result


def _init_linear(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Gaussian scalar output head
# ---------------------------------------------------------------------------


class _GaussianScalarHead(nn.Module):
    """Diagonal Gaussian with a single learnable scalar log σ.

    A single log σ is shared across all features.  Simpler than the per-feature
    heteroscedastic head but sufficient for the motionbench benchmark.
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(1))

    def nll(self, x_target: Tensor, x_pred: Tensor, hid_mask: Tensor) -> Tensor:
        """Negative log-likelihood on hidden entries only.

        Args:
            x_target: ``(B, T, J, C)`` ground-truth.
            x_pred: ``(B, T, J, C)`` decoder mean.
            hid_mask: ``(B, T, J, C)`` bool; ``True`` = hidden entry.

        Returns:
            Scalar NLL averaged over hidden entries.
        """
        sigma2 = (2.0 * self.log_sigma).exp()
        nll_per = (
            0.5 * (x_target - x_pred) ** 2 / sigma2
            + self.log_sigma
            + 0.5 * math.log(2.0 * math.pi)
        )
        hid_f = hid_mask.to(nll_per.dtype)
        n_hid = hid_f.sum().clamp(min=1.0)
        return (nll_per * hid_f).sum() / n_hid

    def sample(self, x_pred: Tensor, temperature: float = 1.0) -> Tensor:
        """Sample completions from the decoder distribution.

        Args:
            x_pred: ``(..., J, C)`` decoder mean.
            temperature: Scale applied to the noise; ``0.0`` = deterministic.

        Returns:
            Sampled tensor of the same shape as *x_pred*.
        """
        if temperature == 0.0:
            return x_pred
        sigma = self.log_sigma.exp()
        return x_pred + temperature * sigma * torch.randn_like(x_pred)


# ---------------------------------------------------------------------------
# VAEAC core neural network
# ---------------------------------------------------------------------------


class _VAEAC(nn.Module):
    """Per-frame VAEAC with transformer subnets.

    Token feature layouts:

    * full encoder:  ``[x_flat, mask_flat]``                ``(2 J C)``
    * prior encoder: ``[(x·mask)_flat, mask_flat]``         ``(2 J C)``
    * decoder:       ``[z, (x·mask)_flat, mask_flat]``      ``(d_latent + 2 J C)``

    Args:
        n_joints: Number of skeletal joints.
        n_coords: Number of coordinates per joint.
        d_model: Transformer model dimension (hidden size).
        d_latent: Per-frame latent dimension.
        num_layers: Transformer layers per subnet.
        max_len: Maximum sequence length for the positional encoding buffer.
    """

    def __init__(
        self,
        n_joints: int,
        n_coords: int,
        d_model: int = 256,
        d_latent: int = 64,
        num_layers: int = 2,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.n_joints = n_joints
        self.n_coords = n_coords
        self.input_dim = n_joints * n_coords
        self.d_latent = d_latent

        nhead = _nhead_for(d_model)
        ff_dim = d_model * 2
        dropout = 0.1

        feat_enc = 2 * self.input_dim
        feat_dec = d_latent + 2 * self.input_dim

        self.full_encoder = _EncoderHead(
            feat_enc, d_model, nhead, num_layers, ff_dim, dropout, d_latent, max_len
        )
        self.prior_encoder = _EncoderHead(
            feat_enc, d_model, nhead, num_layers, ff_dim, dropout, d_latent, max_len
        )
        self.decoder = _Decoder(
            feat_dec, self.input_dim, d_model, nhead, num_layers, ff_dim, dropout, max_len
        )
        self.head = _GaussianScalarHead()

    # ------------------------------------------------------------------
    # Tokenisation helpers
    # ------------------------------------------------------------------

    def _tok_full(self, x: Tensor, obs: Tensor) -> Tensor:
        """Build full-encoder token: ``[x_flat, mask_flat]``."""
        B, T = x.shape[:2]
        return torch.cat([x.reshape(B, T, -1), obs.to(x.dtype).reshape(B, T, -1)], dim=-1)

    def _tok_prior(self, x: Tensor, obs: Tensor) -> Tensor:
        """Build prior-encoder token: ``[(x·mask)_flat, mask_flat]``."""
        B, T = x.shape[:2]
        obs_f = obs.to(x.dtype)
        return torch.cat(
            [(x * obs_f).reshape(B, T, -1), obs_f.reshape(B, T, -1)], dim=-1
        )

    def _tok_dec(self, z: Tensor, x: Tensor, obs: Tensor) -> Tensor:
        """Build decoder token: ``[z, (x·mask)_flat, mask_flat]``."""
        B, T = x.shape[:2]
        obs_f = obs.to(x.dtype)
        return torch.cat(
            [z, (x * obs_f).reshape(B, T, -1), obs_f.reshape(B, T, -1)], dim=-1
        )

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def elbo(
        self,
        x: Tensor,
        obs: Tensor,
        pad_mask: Tensor | None = None,
        kl_weight: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute the ELBO loss.

        Args:
            x: ``(B, T, J, C)`` float32 skeleton sequence.
            obs: ``(B, T, J, C)`` bool; ``True`` = observed.
            pad_mask: Optional ``(B, T)`` bool; ``True`` = real frame.
            kl_weight: KL annealing coefficient (default 1.0).

        Returns:
            Tuple ``(loss, recon_nll, kl)`` — all scalar tensors.
        """
        B, T, J, C = x.shape

        mu_q, lv_q = self.full_encoder(self._tok_full(x, obs), pad_mask)
        mu_p, lv_p = self.prior_encoder(self._tok_prior(x, obs), pad_mask)

        z = mu_q + torch.exp(0.5 * lv_q) * torch.randn_like(mu_q)

        dec_flat = self.decoder(self._tok_dec(z, x, obs), pad_mask)
        x_pred = dec_flat.reshape(B, T, J, C)

        hid = ~obs
        if pad_mask is not None:
            hid = hid & pad_mask[:, :, None, None]
        recon_nll = self.head.nll(x, x_pred, hid)

        kl_per = 0.5 * (
            lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p) ** 2) / lv_p.exp() - 1.0
        )
        if pad_mask is not None:
            pm_f = pad_mask.to(kl_per.dtype)[:, :, None]
            kl = (kl_per * pm_f).sum() / pm_f.sum().clamp(min=1.0) / kl_per.shape[-1]
        else:
            kl = kl_per.mean()

        loss = recon_nll + kl_weight * kl
        return loss, recon_nll, kl

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_completions(
        self,
        x: Tensor,
        obs: Tensor,
        n_samples: int = 1,
        temperature: float = 1.0,
    ) -> Tensor:
        """Draw *n_samples* conditional completions from the prior.

        Args:
            x: ``(B, T, J, C)`` float32 sequence (observed entries in VAEAC space).
            obs: ``(B, T, J, C)`` bool; ``True`` = observed.
            n_samples: Number of completions per input.
            temperature: Latent sampling temperature; ``0.0`` = deterministic mean.

        Returns:
            ``(B * n_samples, T, J, C)`` completions.  Observed entries are
            copied verbatim from *x*.
        """
        B, T, J, C = x.shape

        x_r = x.repeat_interleave(n_samples, dim=0)
        obs_r = obs.repeat_interleave(n_samples, dim=0)

        mu_p, lv_p = self.prior_encoder(self._tok_prior(x_r, obs_r))
        if temperature == 0.0:
            z = mu_p
        else:
            z = mu_p + temperature * torch.exp(0.5 * lv_p) * torch.randn_like(mu_p)

        dec_flat = self.decoder(self._tok_dec(z, x_r, obs_r))
        x_samp = self.head.sample(dec_flat.reshape(B * n_samples, T, J, C), temperature)

        obs_f = obs_r.to(x_r.dtype)
        return x_r * obs_f + x_samp * (1.0 - obs_f)


# ---------------------------------------------------------------------------
# Public imputer
# ---------------------------------------------------------------------------


class VAEACImputer(BaseImputer):
    """VAEAC (VAE with Arbitrary Conditioning) on-manifold imputer.

    Amortised conditional imputer backed by a Transformer-based VAEAC model.
    The model is trained via ``scripts/train_vaeac.py`` and loaded with
    :meth:`load`.  Calling :meth:`fit` with a dataset stores the reference;
    no gradient updates occur inside :meth:`fit`.

    Architecture: ``prior_encoder(x_obs, mask) → z → decoder(z, mask) → x_hid``

    Observed entries are **overwritten exactly** in every returned sample,
    satisfying the :class:`~motionbench.imputers.base.BaseImputer`
    observed-preservation contract.

    Args:
        J: Number of skeletal joints.
        F: Number of coordinates per joint.
        T: Number of frames per clip (stored for validation only).
        latent_dim: Per-frame latent dimension.
        hidden_dim: Transformer model dimension.

    Example::

        imputer = VAEACImputer(J=17, F=3, T=81)
        imputer.fit(train_dataset)           # stores dataset ref only
        # … train via scripts/train_vaeac.py, then:
        imputer = VAEACImputer.load("checkpoint.pt")
        completions = imputer.impute(x_obs, mask, n_samples=20)
    """

    def __init__(
        self,
        J: int,
        F: int,
        T: int,
        latent_dim: int = 64,
        hidden_dim: int = 256,
    ) -> None:
        self._J = J
        self._F = F
        self._T = T
        self._latent_dim = latent_dim
        self._hidden_dim = hidden_dim
        self._fitted = False
        self._device = torch.device("cpu")

        max_len = max(T + 16, 256)
        self._model = _VAEAC(
            n_joints=J,
            n_coords=F,
            d_model=hidden_dim,
            d_latent=latent_dim,
            num_layers=2,
            max_len=max_len,
        )

    # ------------------------------------------------------------------
    # BaseImputer interface
    # ------------------------------------------------------------------

    @property
    def is_on_manifold(self) -> bool:
        """VAEAC is an on-manifold learned imputer."""
        return True

    def fit(self, train_data: BaseDataset) -> VAEACImputer:
        """Store dataset reference; actual training is done via the training script.

        Calling :meth:`fit` marks the imputer as ready for inference.  For
        untrained weights this produces noise completions — train the model
        with ``scripts/train_vaeac.py`` and call :meth:`load` instead.

        Args:
            train_data: A :class:`~motionbench.data.base.BaseDataset` whose
                ``shape`` is ``(J, F, T)`` matching this imputer's dimensions.

        Returns:
            ``self`` for method chaining.
        """
        self._train_data = train_data
        self._fitted = True
        return self

    def impute(
        self,
        x_obs: Tensor,
        mask: Tensor,
        n_samples: int,
        seed: int | None = None,
    ) -> Tensor:
        """Draw *n_samples* completions of *x_obs* given the observed mask.

        Observed entries are preserved bit-for-bit via :func:`torch.where`
        after decoding.

        Args:
            x_obs: ``(J, F, T)`` float32 sequence.  Entries where
                ``mask == False`` are ignored.
            mask: ``(J, F, T)`` bool; ``True`` = observed.
            n_samples: Number of completed sequences to return.
            seed: Optional random seed for reproducibility.

        Returns:
            ``(n_samples, J, F, T)`` float32 tensor.

        Raises:
            RuntimeError: if :meth:`fit` has not been called.
            ValueError: if ``x_obs.shape != mask.shape`` or shapes are
                inconsistent with the fitted dimensions.
        """
        if not self._fitted:
            raise RuntimeError(
                "VAEACImputer.fit() must be called before impute(). "
                "Either call fit(dataset) or load a checkpoint with load()."
            )
        if x_obs.shape != mask.shape:
            raise ValueError(
                f"x_obs.shape {tuple(x_obs.shape)} != mask.shape {tuple(mask.shape)}"
            )
        J, F, T = x_obs.shape
        if (J, F, T) != (self._J, self._F, self._T):
            raise ValueError(
                f"Input shape ({J}, {F}, {T}) does not match imputer shape "
                f"({self._J}, {self._F}, {self._T})."
            )

        if seed is not None:
            torch.manual_seed(seed)

        x = x_obs.to(self._device)
        m = mask.to(self._device)

        # (J, F, T) → (1, T, J, F) — VAEAC layout
        x_vaeac = x.permute(2, 0, 1).unsqueeze(0)
        m_vaeac = m.permute(2, 0, 1).unsqueeze(0)

        self._model.eval()
        completions = self._model.sample_completions(
            x_vaeac, m_vaeac, n_samples=n_samples
        )
        # completions: (n_samples, T, J, F) → (n_samples, J, F, T)
        output = completions.permute(0, 2, 3, 1).contiguous()

        # Guarantee exact observed-entry preservation (contract)
        mask_exp = m.unsqueeze(0)  # (1, J, F, T) broadcasts over n_samples
        x_obs_exp = x.unsqueeze(0)  # (1, J, F, T)
        return torch.where(mask_exp, x_obs_exp, output)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the model weights and configuration to *path*.

        Args:
            path: Destination file path (e.g. ``checkpoint.pt``).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "config": {
                    "J": self._J,
                    "F": self._F,
                    "T": self._T,
                    "latent_dim": self._latent_dim,
                    "hidden_dim": self._hidden_dim,
                },
                "fitted": self._fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> VAEACImputer:
        """Load a :class:`VAEACImputer` from a checkpoint created with :meth:`save`.

        Args:
            path: Path to a checkpoint file produced by :meth:`save`.

        Returns:
            A :class:`VAEACImputer` with weights restored.

        Raises:
            FileNotFoundError: if *path* does not exist.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        cfg = ckpt["config"]
        imputer = cls(
            J=int(cfg["J"]),
            F=int(cfg["F"]),
            T=int(cfg["T"]),
            latent_dim=int(cfg["latent_dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
        )
        imputer._model.load_state_dict(ckpt["state_dict"])
        imputer._fitted = bool(ckpt.get("fitted", True))
        return imputer

    # ------------------------------------------------------------------
    # Training utilities (used by scripts/train_vaeac.py and smoke test)
    # ------------------------------------------------------------------

    def _fit_epochs(
        self,
        x_data: Tensor,
        epochs: int = 10,
        batch_size: int = 16,
        lr: float = 1e-3,
    ) -> list[float]:
        """Run a vanilla training loop for *epochs* epochs.

        This is a convenience method for the training script and tests.
        For production training use ``scripts/train_vaeac.py``.

        Args:
            x_data: ``(N, J, F, T)`` float32 training samples in motionbench
                convention.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            lr: Adam learning-rate.

        Returns:
            List of per-epoch average ELBO losses.
        """
        N = x_data.shape[0]
        # (N, J, F, T) → (N, T, J, F) — VAEAC layout
        x_vaeac = x_data.permute(0, 3, 1, 2).contiguous().to(self._device)
        J = x_vaeac.shape[2]
        C = x_vaeac.shape[3]
        T = x_vaeac.shape[1]

        self._model.to(self._device)
        self._model.train()
        opt = torch.optim.Adam(self._model.parameters(), lr=lr)

        epoch_losses: list[float] = []
        for _epoch in range(epochs):
            perm = torch.randperm(N, device=self._device)
            batch_losses: list[float] = []
            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                xb = x_vaeac[idx]  # (B, T, J, C)
                B = xb.shape[0]
                obs = sample_training_mask(B=B, T=T, J=J, C=C, device=self._device)
                loss, _recon, _kl = self._model.elbo(xb, obs)
                opt.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                opt.step()
                batch_losses.append(float(loss.detach()))
            epoch_losses.append(sum(batch_losses) / max(len(batch_losses), 1))
        self._model.eval()
        self._fitted = True
        return epoch_losses

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def to(self, device: torch.device | str) -> VAEACImputer:
        """Move the underlying model to *device*.

        Args:
            device: Target device string or :class:`torch.device`.

        Returns:
            ``self`` for chaining.
        """
        self._device = torch.device(device)
        self._model.to(self._device)
        return self
