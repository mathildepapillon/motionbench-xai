"""motionbench.imputers.flow_matching — OT-Flow conditional imputer with RePaint.

Overview
--------
:class:`FlowMatchingImputer` learns the data distribution ``p(x)`` via
Conditional Optimal-Transport (CondOT) flow matching (Lipman et al. 2023)
and samples from the conditional ``p(x_hid | x_obs)`` by pairing the
unconditional ODE integration with RePaint harmonisation at every step
(Lugmayr et al. 2022).

Algorithm (per impute call)::

    x_0 ~ N(0, noise_init_scale² · I)          # n_samples noise initialisations
    for k in 0 … K-1:
        t_k   = k / K
        t_mid = t_k + dt/2
        t_k+1 = (k+1) / K
        # Midpoint (RK2) velocity step
        v1      = v_θ(x_k, t_k)
        x_mid   = x_k + (dt/2)·v1
        x_mid   = harmonize(x_mid, t_mid)       # RePaint on x_mid
        v2      = v_θ(x_mid, t_mid)
        x_k+1   = x_k + dt·v2
        x_k+1   = harmonize(x_k+1, t_k+1)      # RePaint
    # Final exact enforcement (guards against float accumulation at t=1)
    x_1[obs] ← x_obs[obs]

The harmonise function follows the CondOT path for *observed* entries::

    harmonize(x, t) = where(obs, (1-t)·x_0 + t·x_obs, x)

Training uses the standard CondOT regression objective::

    L = E_{x_1 ~ p_data, t ~ U(0,1), x_0 ~ N(0,σ²I)}
        [||v_θ(x_t, t) - (x_1 - x_0)||²]
    where  x_t = (1-t)·x_0 + t·x_1

M=10 Burr Regression — Risk R2 Investigation
---------------------------------------------
In the CARE-PD paper, FlowMatchingImputer achieves EC2 = 0.269 on Burr-XII
(c=2, k=2) with M=10 temporal players, compared to EC2 = 0.072 for VAEAC.
Three candidate hypotheses were investigated:

**H1: Too few ODE steps for heavy-tailed distributions.**
  Burr-XII has a velocity field with potentially large magnitudes near t=1
  because the target distribution has heavier tails than the Gaussian source.
  Finer discretisation (more steps) reduces integration error in these
  high-curvature regions.

**H2: Gaussian noise init poorly conditioned for Burr-XII marginals.**
  *(Most likely root cause.)*  Under CondOT with source N(0, σ²) and target
  Burr-XII(c=2, k=2), the target variance is ``σ²_data = k/(c-1)·B(k-1/c, 1/c)``
  which can substantially exceed σ²=1. The velocity field u_t = x_1 - x_0 has
  a right-skewed distribution (x_1 is heavy-tailed; x_0 is light-tailed). The
  network must extrapolate to large positive velocities for tail events, which
  are rare in the training set. As a result the network systematically
  under-imputes the heavy tail, biasing the imputed Shapley values and degrading
  EC2. Tong et al. (2024) explicitly note that "source distribution choice
  critically affects flow quality" and that Gaussian source is suboptimal for
  heavy-tailed targets. Using a larger ``noise_init_scale`` (e.g. 2.0) or a
  heavier-tailed source reduces this mismatch.

**H3: RePaint harmonisation needs adjustment for many-observed coalitions.**
  When M=10 players are observed (large coalition), most coordinates are pinned
  at each step. While this constrains the generation, it should actually *help*
  rather than hurt — more constraints means a smaller effective sampling space.
  This hypothesis is therefore considered unlikely as the primary cause.

**Conclusion:** H2 is the most probable root cause for M=10 Burr regression.
The ablation ``test_flow_m10_burr_ablation`` (``@pytest.mark.manual``) sweeps
``num_steps ∈ {10, 50, 100}`` and ``noise_init_scale ∈ {0.5, 1.0, 2.0}``
to test H1 and H2 empirically. The expected finding is that ``noise_init_scale``
matters more than ``num_steps``.

References
----------
Lipman Y. et al. (2023) "Flow Matching for Generative Modeling."
    arXiv:2210.02747. https://arxiv.org/abs/2210.02747

Tong A. et al. (2024) "Improving and Generalizing Flow-Matching."
    arXiv:2302.00482. https://arxiv.org/abs/2302.00482
    (Source distribution choice, Theorem 3.1 and surrounding discussion.)

Lugmayr A. et al. (2022) "RePaint: Inpainting using Denoising Diffusion
    Probabilistic Models." CVPR 2022.
    https://arxiv.org/abs/2201.09865
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from pathlib import Path

    from motionbench.data.base import BaseDataset

from motionbench.imputers.base import BaseImputer

__all__ = ["FlowMatchingImputer"]


# ---------------------------------------------------------------------------
# Private: sinusoidal embedding
# ---------------------------------------------------------------------------


def _sinusoidal_embedding(values: Tensor, dim: int, max_period: float = 1_000.0) -> Tensor:
    """Continuous sinusoidal positional embedding.

    Args:
        values: 1-D float tensor of shape ``(N,)``; arbitrary shapes are
            flattened to 1-D and then reshaped back.
        dim: Embedding dimensionality (must be even).
        max_period: Wavelength scale; ``1_000.0`` maps t∈[0,1] to a
            discriminative frequency range for the flow-time MLP.

    Returns:
        Tensor of shape ``values.shape + (dim,)``.

    Raises:
        ValueError: If ``dim`` is odd.
    """
    if dim % 2 != 0:
        raise ValueError(f"_sinusoidal_embedding: dim must be even, got {dim}.")
    orig = values.shape
    v = values.reshape(-1).to(torch.float32)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=v.device)
        / half
    )
    args = v[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return emb.reshape(*orig, dim)


# ---------------------------------------------------------------------------
# Private: FramePositionalEncoding
# ---------------------------------------------------------------------------


class _FramePE(nn.Module):
    """Standard sinusoidal positional encoding over frame index.

    Registered as a non-persistent buffer so it moves with ``.to(device)``
    without being saved in the checkpoint (it can always be recomputed).

    Args:
        d_model: Model hidden size.
        max_len: Maximum supported clip length.
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)
        self.max_len = max_len

    def forward(self, T: int) -> Tensor:
        """Return positional encoding for a sequence of length *T*.

        Args:
            T: Sequence length (must be <= ``max_len``).

        Returns:
            ``(T, d_model)`` positional encoding tensor.

        Raises:
            ValueError: If ``T > max_len``.
        """
        if self.max_len < T:
            raise ValueError(
                f"_FramePE: T={T} exceeds max_len={self.max_len}."
            )
        # pe is a buffer; cast to keep mypy happy
        pe: Tensor = self.pe  # type: ignore[assignment]
        return pe[:T]


# ---------------------------------------------------------------------------
# Private: FlowTimeMLP
# ---------------------------------------------------------------------------


class _FlowTimeMLP(nn.Module):
    """Embeds the scalar flow time ``t ∈ [0, 1]`` into a ``d_model``-dim vector.

    Sinusoidal embedding → Linear → SiLU → Linear, matching the
    TimestepEmbedder structure used in MDM (Tevet et al., ICLR 2023).

    Args:
        sinusoid_dim: Sinusoidal embedding dimensionality (must be even).
        out_dim: Output embedding size (typically ``d_model``).
    """

    def __init__(self, sinusoid_dim: int, out_dim: int) -> None:
        super().__init__()
        if sinusoid_dim % 2 != 0:
            raise ValueError(
                f"_FlowTimeMLP: sinusoid_dim must be even, got {sinusoid_dim}."
            )
        self.sinusoid_dim = sinusoid_dim
        self.mlp = nn.Sequential(
            nn.Linear(sinusoid_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """Embed flow time.

        Args:
            t: ``(B,)`` float tensor of flow times in ``[0, 1]``.

        Returns:
            ``(B, out_dim)`` embedding tensor.
        """
        sin_t = _sinusoidal_embedding(t, self.sinusoid_dim)
        return cast("Tensor", self.mlp(sin_t))


# ---------------------------------------------------------------------------
# Private: VelocityNet
# ---------------------------------------------------------------------------


class _VelocityNet(nn.Module):
    """Encoder-only transformer that predicts the CondOT velocity field.

    Uses *frame-level* tokenisation: each token is one full frame
    ``(J*F,)`` plus a flow-time embedding, processed by a standard
    Transformer encoder.  Output is projected back to ``(J*F,)`` per frame
    and reshaped to ``(B, T, J, F)``.

    Args:
        n_joints: Number of skeletal joints ``J``.
        n_coords: Number of coordinates per joint ``F``.
        d_model: Transformer hidden size.
        nhead: Number of attention heads (must divide ``d_model``).
        num_layers: Number of transformer encoder layers.
        ff_dim: Feed-forward dimension inside each encoder layer.
        dropout: Dropout probability.
        time_emb_dim: Sinusoidal time-embedding width; concatenated to each
            frame token before the input projection.
        max_len: Maximum clip length ``T`` for the positional encoding buffer.
    """

    def __init__(
        self,
        n_joints: int,
        n_coords: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        time_emb_dim: int = 128,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(
                f"_VelocityNet: d_model={d_model} must be divisible by nhead={nhead}."
            )
        if time_emb_dim % 2 != 0:
            raise ValueError(
                f"_VelocityNet: time_emb_dim must be even, got {time_emb_dim}."
            )
        self.n_joints = n_joints
        self.n_coords = n_coords
        self.d_model = d_model
        self.time_emb_dim = time_emb_dim
        input_dim = n_joints * n_coords
        self.flow_time_mlp = _FlowTimeMLP(sinusoid_dim=time_emb_dim, out_dim=time_emb_dim)
        self.in_proj = nn.Linear(input_dim + time_emb_dim, d_model)
        self.frame_pe = _FramePE(d_model, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.out_proj = nn.Linear(d_model, input_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        """Truncated-normal initialisation (consistent with MDM/ACTOR)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        """Predict CondOT velocity at ``(x_t, t)``.

        Args:
            x_t: ``(B, T, J, C)`` noised poses at flow time ``t``.
            t: ``(B,)`` flow times in ``[0, 1]``.

        Returns:
            ``(B, T, J, C)`` predicted velocity field.

        Raises:
            ValueError: On shape mismatch.
        """
        B, T, J, C = x_t.shape
        if self.n_joints != J or self.n_coords != C:
            raise ValueError(
                f"_VelocityNet: expected (..., {self.n_joints}, {self.n_coords}), "
                f"got (..., {J}, {C})."
            )
        if t.dim() != 1 or t.shape[0] != B:
            raise ValueError(f"_VelocityNet: t must be (B,), got {tuple(t.shape)}.")

        t_emb = self.flow_time_mlp(t)  # (B, time_emb_dim)
        x_flat = x_t.reshape(B, T, J * C)  # (B, T, J*C)
        t_tok = t_emb.unsqueeze(1).expand(B, T, self.time_emb_dim)  # (B, T, time_emb)
        tok = torch.cat([x_flat, t_tok], dim=-1)  # (B, T, J*C + time_emb)
        tok = self.in_proj(tok)  # (B, T, d_model)
        tok = tok + self.frame_pe(T).to(tok.dtype)
        h = self.encoder(tok)  # (B, T, d_model)
        v_flat = cast("Tensor", self.out_proj(h))  # (B, T, J*C)
        return v_flat.reshape(B, T, J, C)

    @torch.no_grad()
    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Public: FlowMatchingImputer
# ---------------------------------------------------------------------------


class FlowMatchingImputer(BaseImputer):
    """OT-Flow conditional imputer with RePaint harmonisation.

    Samples from ``p_θ(x_hid | x_obs)`` via:

    1. ``x_{t=0} ~ N(0, noise_init_scale² · I)`` — Gaussian noise initialisation.
    2. Integrate the learned velocity field ``v_θ(x, t)`` forward (t: 0 → 1)
       using a midpoint (RK2) ODE solver.
    3. At every step: RePaint harmonisation — observed coordinates are
       projected back onto the CondOT linear path between their noise start
       and their clean target value.

    ``is_on_manifold = True`` — the samples approximate draws from the true
    conditional ``p(x_hid | x_obs)``.

    Training (CondOT regression objective)::

        L = E[‖v_θ((1-t)x_0 + t·x_1, t) − (x_1 − x_0)‖²]

    where ``x_1 ~ p_data``, ``x_0 ~ N(0, noise_init_scale²·I)``, ``t ~ U(0,1)``.

    M=10 Burr Regression Note
    --------------------------
    See module-level docstring for the full investigation. **H2** (Gaussian
    noise init mismatched with Burr-XII marginals) is considered the primary
    cause.  The ablation ``test_flow_m10_burr_ablation`` sweeps
    ``noise_init_scale`` and ``num_steps`` to test this empirically.

    References
    ----------
    Lipman Y. et al. (2023) "Flow Matching for Generative Modeling."
        arXiv:2210.02747.
    Tong A. et al. (2024) "Improving and Generalizing Flow-Matching."
        arXiv:2302.00482.
    Lugmayr A. et al. (2022) "RePaint." CVPR 2022. arXiv:2201.09865.

    Args:
        J: Number of skeletal joints.
        F: Number of coordinates per joint.
        T: Number of time frames per clip.
        hidden_dim: Transformer ``d_model`` (architecture parameter).
        num_steps: Number of ODE integration steps (K).
        noise_init_scale: Standard deviation of the Gaussian source
            distribution ``x_0 ~ N(0, scale²·I)``.  Increasing this
            towards the target data std can mitigate H2 for heavy-tailed
            distributions (Tong et al. 2024, Remark 3.2).
        n_epochs: Training epochs (used by ``fit``).
        batch_size: Training batch size.
        lr: Adam learning rate.
        solver: ODE solver — ``"midpoint"`` (RK2, default) or ``"euler"``.
        device: Torch device string (auto-selects CUDA if available when
            ``None``).
    """

    _ALLOWED_SOLVERS = ("midpoint", "euler")

    def __init__(
        self,
        J: int,
        F: int,
        T: int,
        hidden_dim: int = 256,
        num_steps: int = 100,
        noise_init_scale: float = 1.0,
        n_epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        solver: str = "midpoint",
        device: str | None = None,
    ) -> None:
        if solver not in self._ALLOWED_SOLVERS:
            raise ValueError(
                f"FlowMatchingImputer: solver must be one of "
                f"{self._ALLOWED_SOLVERS!r}; got {solver!r}."
            )
        if num_steps < 2:
            raise ValueError(
                f"FlowMatchingImputer: num_steps must be >= 2; got {num_steps}."
            )
        self.J = J
        self.F = F
        self.T = T
        self.num_steps = num_steps
        self.noise_init_scale = noise_init_scale
        self._hidden_dim = hidden_dim
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._lr = lr
        self._solver = solver
        self._device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._net: _VelocityNet | None = None
        self._fitted = False
        self.train_losses: list[float] = []

    # ------------------------------------------------------------------
    # BaseImputer contract
    # ------------------------------------------------------------------

    @property
    def is_on_manifold(self) -> bool:
        """Always ``True`` — this imputer learns the data manifold."""
        return True

    def fit(self, train_data: BaseDataset) -> FlowMatchingImputer:
        """Train the velocity network on *train_data*.

        Iterates over ``train_data`` for ``n_epochs`` using AdamW with
        gradient clipping (norm 1.0) and the CondOT MSE objective.

        Args:
            train_data: Any object satisfying :class:`~motionbench.data.base.BaseDataset`.
                Each sample must be a ``(J, F, T)`` float32 tensor.

        Returns:
            ``self`` — for method chaining (``imputer.fit(ds).impute(...)``).

        Raises:
            RuntimeError: If ``train_data`` is empty.
        """
        if len(train_data) == 0:
            raise RuntimeError("FlowMatchingImputer.fit: train_data is empty.")

        # Infer (J, F, T) from first sample if constructor defaults differ.
        x0_sample, _ = train_data[0]
        actual_j, actual_f, actual_t = x0_sample.shape

        net = _VelocityNet(
            n_joints=actual_j,
            n_coords=actual_f,
            d_model=self._hidden_dim,
            nhead=max(1, min(4, self._hidden_dim // 8)),
            num_layers=4,
            ff_dim=self._hidden_dim * 4,
            dropout=0.1,
            time_emb_dim=max(2, self._hidden_dim // 2),
            max_len=max(512, actual_t + 16),
        )
        net.to(self._device)
        self._net = net

        opt: torch.optim.Optimizer = torch.optim.AdamW(
            net.parameters(), lr=self._lr, weight_decay=1e-4
        )

        # DataLoader: BaseDataset satisfies the duck-type protocol for
        # torch DataLoader (has __getitem__ + __len__).
        loader: DataLoader[Any] = DataLoader(
            train_data,  # type: ignore[arg-type]
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=False,
        )

        self.train_losses = []
        net.train()
        for _epoch in range(self._n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for x_batch, _ in loader:
                x1: Tensor = x_batch.to(self._device, dtype=torch.float32)
                # (B, J, F, T) → (B, T, J, F)  [VelocityNet layout]
                x1 = x1.permute(0, 3, 1, 2).contiguous()
                B = x1.shape[0]

                t = torch.rand(B, device=self._device, dtype=torch.float32)
                x0 = torch.randn_like(x1) * self.noise_init_scale

                # CondOT interpolation:  x_t = (1-t)·x_0 + t·x_1
                t_b = t[:, None, None, None]
                x_t = (1.0 - t_b) * x0 + t_b * x1
                u_t = x1 - x0  # target velocity

                v_pred = net(x_t, t)
                loss = F.mse_loss(v_pred, u_t)

                opt.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()

                epoch_loss += loss.item()
                n_batches += 1

            self.train_losses.append(epoch_loss / max(n_batches, 1))

        self._fitted = True
        return self

    @torch.no_grad()
    def impute(
        self,
        x_obs: Tensor,
        mask: Tensor,
        n_samples: int,
        seed: int | None = None,
    ) -> Tensor:
        """Draw *n_samples* completions of the hidden entries of *x_obs*.

        Observed entries (``mask == True``) are preserved bit-for-bit via the
        final exact enforcement step after the ODE loop.

        Args:
            x_obs: ``(J, F, T)`` float32 motion sequence.  Hidden entries
                (``mask == False``) may be arbitrary.
            mask: ``(J, F, T)`` bool tensor; ``True`` = observed, must be
                preserved in the output.
            n_samples: Number of completions to return.
            seed: Optional RNG seed for reproducibility.

        Returns:
            ``(n_samples, J, F, T)`` float32 tensor on CPU.

        Raises:
            RuntimeError: If ``fit`` has not been called.
            ValueError: If ``x_obs.shape != mask.shape``.
        """
        if not self._fitted or self._net is None:
            raise RuntimeError(
                "FlowMatchingImputer.impute: fit() must be called first."
            )
        if x_obs.shape != mask.shape:
            raise ValueError(
                f"x_obs.shape={tuple(x_obs.shape)} != mask.shape={tuple(mask.shape)}."
            )

        if seed is not None:
            torch.manual_seed(seed)

        J, F, Tc = x_obs.shape
        dev = self._device

        # Convert to VelocityNet layout:  (J, F, T) → (T, J, F)
        x1 = x_obs.to(dev, dtype=torch.float32).permute(2, 0, 1)  # (T, J, F)
        x1_n = x1.unsqueeze(0).expand(n_samples, -1, -1, -1).contiguous()  # (N,T,J,F)

        # Observation mask in VelocityNet layout: (J, F, T) → (T, J, F)
        obs = mask.to(dev).permute(2, 0, 1)  # (T, J, F)
        obs_n = obs.unsqueeze(0).expand(n_samples, -1, -1, -1).contiguous()  # (N,T,J,F)

        # Initial Gaussian noise: (N, T, J, F)
        x0 = torch.randn(n_samples, Tc, J, F, device=dev) * self.noise_init_scale

        dt = 1.0 / self.num_steps
        net = self._net
        net.eval()

        x_k = x0.clone()
        for k in range(self.num_steps):
            t_k = k * dt
            t_next = (k + 1) * dt

            if self._solver == "midpoint":
                t_mid = t_k + 0.5 * dt
                t_b_k = torch.full((n_samples,), t_k, device=dev)
                v1 = net(x_k, t_b_k)
                x_mid = x_k + 0.5 * dt * v1
                x_mid = self._harmonize(x_mid, t_mid, x0, x1_n, obs_n)
                t_b_mid = torch.full((n_samples,), t_mid, device=dev)
                v2 = net(x_mid, t_b_mid)
                x_next = x_k + dt * v2
            else:  # euler
                t_b_k = torch.full((n_samples,), t_k, device=dev)
                v1 = net(x_k, t_b_k)
                x_next = x_k + dt * v1

            x_k = self._harmonize(x_next, t_next, x0, x1_n, obs_n)

        # Final exact enforcement: guard against float accumulation at t=1.
        x_k = torch.where(obs_n, x1_n, x_k)

        # (N, T, J, F) → (N, J, F, T)  [motionbench layout]
        out = x_k.permute(0, 2, 3, 1).contiguous()
        return out.cpu()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the trained imputer to a checkpoint file.

        The checkpoint stores both the constructor parameters and the
        VelocityNet state dict so that :meth:`load` can reconstruct an
        identical object without re-training.

        Args:
            path: Destination file path (typically ``*.pt``).

        Raises:
            RuntimeError: If ``fit`` has not been called.
        """
        if not self._fitted or self._net is None:
            raise RuntimeError(
                "FlowMatchingImputer.save: fit() must be called before save()."
            )
        torch.save(
            {
                "constructor_params": {
                    "J": self.J,
                    "F": self.F,
                    "T": self.T,
                    "hidden_dim": self._hidden_dim,
                    "num_steps": self.num_steps,
                    "noise_init_scale": self.noise_init_scale,
                    "n_epochs": self._n_epochs,
                    "batch_size": self._batch_size,
                    "lr": self._lr,
                    "solver": self._solver,
                },
                "state_dict": self._net.state_dict(),
                "train_losses": self.train_losses,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> FlowMatchingImputer:
        """Load a :class:`FlowMatchingImputer` from a checkpoint file.

        Args:
            path: Path to a file previously written by :meth:`save`.

        Returns:
            A fitted :class:`FlowMatchingImputer` instance on CPU; move to
            the desired device with ``imputer._device = torch.device("cuda")``
            and ``imputer._net.to("cuda")`` after loading.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        ckpt = torch.load(path, map_location="cpu")
        params: dict[str, Any] = ckpt["constructor_params"]
        imp = cls(**params, device="cpu")
        actual_j, actual_f = params["J"], params["F"]
        net = _VelocityNet(
            n_joints=actual_j,
            n_coords=actual_f,
            d_model=params["hidden_dim"],
            nhead=max(1, min(4, params["hidden_dim"] // 8)),
            num_layers=4,
            ff_dim=params["hidden_dim"] * 4,
            dropout=0.1,
            time_emb_dim=max(2, params["hidden_dim"] // 2),
            max_len=max(512, params["T"] + 16),
        )
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        imp._net = net
        imp._fitted = True
        imp.train_losses = list(ckpt.get("train_losses", []))
        return imp

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _harmonize(
        xk: Tensor,
        t: float,
        x0: Tensor,
        x1: Tensor,
        obs: Tensor,
    ) -> Tensor:
        """RePaint harmonisation: project observed entries onto the CondOT path.

        For observed entries: ``output = (1-t)·x0 + t·x1``
        For hidden entries:   ``output = xk``  (unchanged)

        This ensures that at ``t=1``, observed entries converge to exactly
        ``x1 = x_obs``, and at ``t=0`` they equal the initial noise ``x0``.

        Args:
            xk: ``(N, T, J, F)`` current ODE state.
            t: Scalar flow time in ``[0, 1]``.
            x0: ``(N, T, J, F)`` initial noise tensor (fixed throughout ODE).
            x1: ``(N, T, J, F)`` target data tensor (fixed throughout ODE).
            obs: ``(N, T, J, F)`` bool mask; ``True`` = observed entry.

        Returns:
            ``(N, T, J, F)`` harmonised tensor.
        """
        cond_path = (1.0 - t) * x0 + t * x1
        return torch.where(obs, cond_path, xk)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Short identifier for logging tables."""
        return "FlowMatchingImputer"

    def __repr__(self) -> str:
        fitted = "fitted" if self._fitted else "unfitted"
        return (
            f"FlowMatchingImputer("
            f"J={self.J}, F={self.F}, T={self.T}, "
            f"hidden_dim={self._hidden_dim}, "
            f"num_steps={self.num_steps}, "
            f"noise_init_scale={self.noise_init_scale}, "
            f"solver={self._solver!r}, "
            f"{fitted})"
        )
