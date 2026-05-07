"""motionbench.imputers.ptbxl_imputer — On-manifold imputers for PTB-XL ECG.

Thin wrappers that load trained VAEAC and Flow Matching checkpoints for the
PTB-XL dataset and expose them as drop-in BaseImputer implementations.

These classes are structurally parallel to
:class:`motionbench.imputers.carepd_imputer.CarepdVAEACImputer` and
:class:`motionbench.imputers.carepd_imputer.CarepdFlowImputer`.

Code reuse
----------
The low-level checkpoint loaders :func:`~motionbench.imputers.carepd_imputer._load_vaeac`
and :func:`~motionbench.imputers.carepd_imputer._load_flow` are **imported**
from :mod:`motionbench.imputers.carepd_imputer` — they are generic loaders
that accept any directory containing a ``.ckpt`` file and a JSON config.  The
mask-translation helper
:func:`~motionbench.imputers.carepd_imputer._mask_to_coalition` is also
imported.

ECG-specific checkpoint layout
-------------------------------
PTB-XL VAEAC and Flow checkpoints are stored under::

    results/ptbxl_imputers/
        vaeac/
            *best*.ckpt            ← saved by train_vaeac.py
            ptbxl_vaeac_cfg.json   ← Hydra / hand-written config
        flow/
            *best*.ckpt
            ptbxl_flow_cfg.json

The required JSON config keys are:

    n_joints   = 12   (ECG leads)
    n_coords   = 1    (voltage channel)
    d_model    = 128
    nhead      = 4
    num_layers = 4
    ff_dim     = 256
    d_latent   = 32
    seq_len    = 1000
    dropout    = 0.1
    decoder_head = "gaussian_scalar"

For Flow Matching additionally:

    time_emb_dim      = 64
    tokenization      = "frame"
    obs_conditioning  = true
    num_steps         = 50

These configs are written by ``scripts/train_vaeac.py`` / ``train_flow.py``
when run with ``configs/data/ptbxl.yaml``.  A fallback default config is
embedded below so the class can still *attempt* to load a checkpoint even if
the JSON file is missing.

Training the imputers
---------------------
Train the VAEAC and Flow Matching imputers using the existing scripts::

    # VAEAC
    conda activate motionbench-xai
    python scripts/train_vaeac.py data=ptbxl

    # Flow Matching
    python scripts/train_flow.py data=ptbxl

These scripts read ``configs/data/ptbxl.yaml`` and write checkpoints to
``results/ptbxl_imputers/``.

Mask contract
-------------
ECG coalition masks follow the same convention as CARE-PD:

    Temporal windows  → ``(1, T=1000)`` bool
    Lead (spatial)    → ``(1, J=12)``  bool

The :func:`~motionbench.imputers.carepd_imputer._mask_to_coalition` helper
detects which format is appropriate from the ``(J, F, T)`` mask.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import torch
from torch import Tensor

from motionbench.imputers.base import BaseImputer
from motionbench.imputers.carepd_imputer import (
    _load_flow,
    _load_vaeac,
    _mask_to_coalition,
)

log = logging.getLogger(__name__)

__all__ = ["PTBXLVAEACImputer", "PTBXLFlowImputer"]

# ---------------------------------------------------------------------------
# Checkpoint root for PTB-XL imputers
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parents[2]
_PTBXL_IMPUTER_ROOT = REPO_ROOT / "results" / "ptbxl_imputers"

_VAEAC_CKPT_DIR = _PTBXL_IMPUTER_ROOT / "vaeac"
_FLOW_CKPT_DIR  = _PTBXL_IMPUTER_ROOT / "flow"

# ---------------------------------------------------------------------------
# Embedded fallback configs (used if the JSON file is absent)
# ---------------------------------------------------------------------------
_VAEAC_DEFAULT_CFG: dict = {
    "n_joints":      12,
    "n_coords":      1,
    "d_model":       128,
    "nhead":         4,
    "num_layers":    4,
    "ff_dim":        256,
    "d_latent":      32,
    "seq_len":       1000,
    "dropout":       0.1,
    "decoder_head":  "gaussian_scalar",
    "ivanov_min_sigma": 0.01,
    "use_prior_memory": False,
    "prior_reg_sigma_mu": 1e4,
    "prior_reg_sigma_sigma": 1e-4,
}

_FLOW_DEFAULT_CFG: dict = {
    "n_joints":         12,
    "n_coords":         1,
    "d_model":          128,
    "nhead":            4,
    "num_layers":       4,
    "ff_dim":           256,
    "time_emb_dim":     64,
    "seq_len":          1000,
    "dropout":          0.1,
    "tokenization":     "frame",
    "obs_conditioning": True,
    "num_steps":        50,
    "cfg_scale":        0.0,
}


def _resolve_cfg(ckpt_dir: Path, cfg_filename: str, default: dict) -> Path:
    """Return a resolved path to the config JSON for *ckpt_dir*.

    Looks for *cfg_filename* inside *ckpt_dir*.  If it doesn't exist, writes
    the embedded *default* dict to a temporary file and returns its path.

    Note: the temporary file is intentionally **not** deleted here; callers
    are responsible for cleanup if needed.

    Args:
        ckpt_dir: Directory containing the checkpoint.
        cfg_filename: Expected filename for the JSON config.
        default: Dict to write if the JSON file is absent.

    Returns:
        Path to a readable JSON config file.
    """
    cfg_path = ckpt_dir / cfg_filename
    if cfg_path.exists():
        return cfg_path
    # Write fallback config to a temp file
    tf = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="ptbxl_cfg_"
    )
    json.dump(default, tf)
    tf.close()
    log.warning(
        "PTB-XL imputer: %s not found; using embedded default config "
        "(written to %s). Verify hyperparameters match the trained checkpoint.",
        cfg_path, tf.name,
    )
    return Path(tf.name)


class PTBXLVAEACImputer(BaseImputer):
    """PTB-XL VAEAC imputer — on-manifold ECG completion.

    Loads a pre-trained VAEAC checkpoint from
    ``results/ptbxl_imputers/vaeac/`` and wraps it as a
    :class:`motionbench.imputers.base.BaseImputer`.

    The VAEAC model is the same transformer-based architecture used for
    CARE-PD (``model.vaeac.VAEAC``) re-trained on PTB-XL data with
    ``n_joints=12, n_coords=1, seq_len=1000``.

    Args:
        n_completion_samples: Number of VAEAC posterior draws per imputation
            call (default 10).
        device: PyTorch device string (e.g. ``"cuda:0"`` or ``"cpu"``).
    """

    def __init__(
        self,
        n_completion_samples: int = 10,
        device: str = "cpu",
        **_kwargs: object,
    ) -> None:
        self._n_completion_samples = int(n_completion_samples)
        self._device_str = device
        self._imputer = None
        self._fitted = False
        self._skip = False
        self._tmp_cfg: Path | None = None

    def fit(self, train_data: object) -> "PTBXLVAEACImputer":
        """Load the PTB-XL VAEAC checkpoint.

        The dataset class name is *not* used for registry lookup (unlike the
        CARE-PD imputer).  The checkpoint is always loaded from
        ``_VAEAC_CKPT_DIR``.

        Args:
            train_data: Any object satisfying the BaseDataset protocol.
                Used only for a dimension compatibility check.

        Returns:
            ``self``
        """
        if not _VAEAC_CKPT_DIR.exists():
            log.warning(
                "PTBXLVAEACImputer: checkpoint directory %s not found. "
                "Train the VAEAC imputer first: python scripts/train_vaeac.py data=ptbxl. "
                "This dataset will be skipped.",
                _VAEAC_CKPT_DIR,
            )
            self._skip = True
            self._fitted = True
            return self

        device = torch.device(self._device_str)
        cfg_path = _resolve_cfg(_VAEAC_CKPT_DIR, "ptbxl_vaeac_cfg.json", _VAEAC_DEFAULT_CFG)
        self._tmp_cfg = cfg_path  # keep reference for cleanup

        try:
            self._imputer = _load_vaeac(_VAEAC_CKPT_DIR, cfg_path, device)
            # Dimension compatibility check
            if hasattr(train_data, "shape"):
                J_data, _, T_data = train_data.shape
                cfg = json.loads(cfg_path.read_text())
                J_model = int(cfg.get("n_joints", 12))
                T_model = int(cfg.get("seq_len", 1000))
                if J_data != J_model or T_data != T_model:
                    log.warning(
                        "PTBXLVAEACImputer: shape mismatch — "
                        "data (J=%d, T=%d) vs model (J=%d, T=%d). Skipping.",
                        J_data, T_data, J_model, T_model,
                    )
                    self._skip = True
                    self._imputer = None
        except Exception as exc:
            log.warning("PTBXLVAEACImputer: failed to load checkpoint: %s", exc)
            self._skip = True

        self._fitted = True
        return self

    def impute(
        self,
        x_obs: Tensor,
        mask: Tensor,
        n_samples: int,
        seed: int | None = None,
    ) -> Tensor:
        """Impute missing ECG values conditioned on the coalition mask.

        Args:
            x_obs: ``(J=12, F=1, T=1000)`` float32 observed tensor.
            mask:  ``(J=12, F=1, T=1000)`` bool tensor — True = observed.
            n_samples: Number of samples to draw.
            seed: Unused (VAEAC is non-deterministic by design).

        Returns:
            ``(n_samples, J, F, T)`` float32.

        Raises:
            NotImplementedError: If no checkpoint is available.
            RuntimeError: If ``fit()`` has not been called.
        """
        if not self._fitted:
            raise RuntimeError("PTBXLVAEACImputer.fit() must be called first.")
        if self._skip or self._imputer is None:
            raise NotImplementedError(
                "No PTB-XL VAEAC checkpoint available. "
                "Run: python scripts/train_vaeac.py data=ptbxl"
            )

        J, F, T = x_obs.shape
        device = self._imputer._device
        x_in = x_obs.unsqueeze(0).to(device)
        pad = torch.ones(1, T, dtype=torch.bool, device=device)
        coalition_mask, _ = _mask_to_coalition(mask)
        coalition_mask = coalition_mask.to(device)

        completions = self._imputer.sample_completions(
            x=x_in, y=None, mask=pad, lengths=None,
            coalition_mask=coalition_mask, n_samples=n_samples,
        )
        out = torch.cat(completions, dim=0)   # (n_samples, J, F, T)
        out_dev = out.device
        x_dev   = x_obs.to(out_dev)
        mask_dev = mask.to(out_dev)
        obs = mask_dev.unsqueeze(0).expand_as(out)
        out = torch.where(obs, x_dev.unsqueeze(0).expand_as(out), out)
        return out.float().cpu()

    @property
    def is_on_manifold(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "vaeac"


class PTBXLFlowImputer(BaseImputer):
    """PTB-XL Flow Matching imputer — on-manifold ECG completion.

    Loads a pre-trained Flow Matching checkpoint from
    ``results/ptbxl_imputers/flow/`` and wraps it as a
    :class:`motionbench.imputers.base.BaseImputer`.

    The architecture (``model.flow_matching.VelocityNet``) is the same as
    used for CARE-PD, re-trained on PTB-XL with ``n_joints=12, n_coords=1,
    seq_len=1000``.

    Args:
        num_steps: Number of ODE integration steps (default 50).
        device: PyTorch device string.
    """

    def __init__(
        self,
        num_steps: int = 50,
        device: str = "cpu",
        **_kwargs: object,
    ) -> None:
        self._num_steps = int(num_steps)
        self._device_str = device
        self._imputer = None
        self._fitted = False
        self._skip = False
        self._tmp_cfg: Path | None = None

    def fit(self, train_data: object) -> "PTBXLFlowImputer":
        """Load the PTB-XL Flow Matching checkpoint.

        Args:
            train_data: Any object satisfying the BaseDataset protocol.

        Returns:
            ``self``
        """
        if not _FLOW_CKPT_DIR.exists():
            log.warning(
                "PTBXLFlowImputer: checkpoint directory %s not found. "
                "Train the Flow imputer first: python scripts/train_flow.py data=ptbxl. "
                "This dataset will be skipped.",
                _FLOW_CKPT_DIR,
            )
            self._skip = True
            self._fitted = True
            return self

        device = torch.device(self._device_str)

        # Override num_steps in a temporary copy of the config
        cfg_path = _resolve_cfg(_FLOW_CKPT_DIR, "ptbxl_flow_cfg.json", _FLOW_DEFAULT_CFG)
        cfg = json.loads(cfg_path.read_text())
        cfg["num_steps"] = self._num_steps

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="ptbxl_flow_cfg_"
        )
        json.dump(cfg, tmp)
        tmp.close()
        self._tmp_cfg = Path(tmp.name)

        try:
            self._imputer = _load_flow(_FLOW_CKPT_DIR, self._tmp_cfg, device)
            if hasattr(train_data, "shape"):
                J_data, _, T_data = train_data.shape
                J_model = int(cfg.get("n_joints", 12))
                T_model = int(cfg.get("seq_len", 1000))
                if J_data != J_model or T_data != T_model:
                    log.warning(
                        "PTBXLFlowImputer: shape mismatch — "
                        "data (J=%d, T=%d) vs model (J=%d, T=%d). Skipping.",
                        J_data, T_data, J_model, T_model,
                    )
                    self._skip = True
                    self._imputer = None
        except Exception as exc:
            log.warning("PTBXLFlowImputer: failed to load checkpoint: %s", exc)
            self._skip = True
        finally:
            # Clean up temp config file
            if self._tmp_cfg and self._tmp_cfg.exists():
                try:
                    os.unlink(self._tmp_cfg)
                except OSError:
                    pass
                self._tmp_cfg = None

        self._fitted = True
        return self

    def precompute_all_temporal_coalitions(
        self, x_obs: Tensor, K: int, n_samples: int = 1
    ) -> None:
        """Pre-compute completions for all 2^K temporal-window coalitions.

        Uses ``sample_completions_batched`` for a ~K-fold speedup over
        calling ``impute`` 2^K times.  Results are stored in
        ``self._completion_cache``.

        Args:
            x_obs: ``(J, F, T)`` float32 observed tensor.
            K: Number of temporal windows.
            n_samples: Samples per coalition (default 1).
        """
        if self._skip or self._imputer is None:
            return
        if not hasattr(self._imputer, "sample_completions_batched"):
            return

        J, F, T = x_obs.shape
        T_win = T // K
        device = self._imputer._device
        n_coal = 2 ** K

        all_coal_masks = torch.zeros(n_coal, T, dtype=torch.bool)
        for ci in range(n_coal):
            for win in range(K):
                if (ci >> win) & 1:
                    t0 = win * T_win
                    t1 = t0 + T_win if win < K - 1 else T
                    all_coal_masks[ci, t0:t1] = True

        x_in = x_obs.unsqueeze(0)
        pad  = torch.ones(1, T, dtype=torch.bool)

        batched_out = self._imputer.sample_completions_batched(
            x=x_in, mask=pad,
            coalition_masks=all_coal_masks,
            n_samples=n_samples,
        )   # (n_coal, n_samples, J, F, T)
        batched_out = batched_out.cpu()

        self._completion_cache: dict[tuple, Tensor] = {}
        for ci in range(n_coal):
            key = tuple(all_coal_masks[ci].tolist())
            comps = batched_out[ci]   # (n_samples, J, F, T)
            m_k = all_coal_masks[ci].view(1, 1, 1, T).expand(n_samples, J, F, T)
            comps = torch.where(
                m_k,
                x_obs.cpu().unsqueeze(0).expand(n_samples, -1, -1, -1),
                comps,
            )
            self._completion_cache[key] = comps.float()

        self._cached_x_obs = x_obs.cpu()
        log.debug(
            "PTBXLFlowImputer: pre-computed %d coalitions (K=%d, T=%d)",
            n_coal, K, T,
        )

    def clear_cache(self) -> None:
        """Clear the pre-computed coalition cache."""
        self._completion_cache = {}

    def impute(
        self,
        x_obs: Tensor,
        mask: Tensor,
        n_samples: int,
        seed: int | None = None,
    ) -> Tensor:
        """Impute missing ECG values using flow matching.

        Args:
            x_obs: ``(J=12, F=1, T=1000)`` float32 observed tensor.
            mask:  ``(J=12, F=1, T=1000)`` bool tensor — True = observed.
            n_samples: Number of samples to draw.
            seed: Unused.

        Returns:
            ``(n_samples, J, F, T)`` float32.

        Raises:
            NotImplementedError: If no checkpoint is available.
            RuntimeError: If ``fit()`` has not been called.
        """
        if not self._fitted:
            raise RuntimeError("PTBXLFlowImputer.fit() must be called first.")
        if self._skip or self._imputer is None:
            raise NotImplementedError(
                "No PTB-XL Flow checkpoint available. "
                "Run: python scripts/train_flow.py data=ptbxl"
            )

        # Fast path: use pre-computed coalition cache if available
        cache = getattr(self, "_completion_cache", {})
        if cache:
            J, F, T = x_obs.shape
            coalition_mask_1d, kind = _mask_to_coalition(mask)
            if kind == "temporal" and coalition_mask_1d.shape[-1] == T:
                key = tuple(coalition_mask_1d[0].tolist())
                if key in cache:
                    out = cache[key]
                    if n_samples <= out.shape[0]:
                        out = out[:n_samples]
                    return out.float().cpu()

        # Slow path
        J, F, T = x_obs.shape
        device = self._imputer._device
        x_in = x_obs.unsqueeze(0).to(device)
        pad  = torch.ones(1, T, dtype=torch.bool, device=device)
        coalition_mask, _ = _mask_to_coalition(mask)
        coalition_mask = coalition_mask.to(device)

        completions = self._imputer.sample_completions(
            x=x_in, y=None, mask=pad, lengths=None,
            coalition_mask=coalition_mask, n_samples=n_samples,
        )
        out = torch.cat(completions, dim=0)
        out_dev = out.device
        x_dev   = x_obs.to(out_dev)
        mask_dev = mask.to(out_dev)
        obs = mask_dev.unsqueeze(0).expand_as(out)
        out = torch.where(obs, x_dev.unsqueeze(0).expand_as(out), out)
        return out.float().cpu()

    @property
    def is_on_manifold(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "flow_matching"
