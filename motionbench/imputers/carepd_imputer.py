"""carepd_imputer.py — Bridge adapters that wrap CARE-PD's trained
VAEACImputer and FlowImputer behind the motionbench-xai BaseImputer API.

These adapters load pre-trained checkpoints from the CARE-PD project and
expose them as drop-in ``BaseImputer`` implementations that can be wired
into any motionbench-xai KernelSHAP method config.

Mask translation
----------------
CARE-PD imputers accept *player-level* coalition masks:
    ``(1, T)``  — temporal coalition (True = frame-window observed)
    ``(1, J)``  — spatial coalition (True = joint observed)

motionbench-xai passes *element-level* ``(J, F, T)`` masks.  This adapter
detects the structure and produces the matching coalition mask:

* If all (J, F) are uniform per time step  → temporal ``(1, T)``
* If all (F, T) are uniform per joint      → spatial  ``(1, J)``
* Otherwise                                → temporal heuristic (most common)

Dataset ↔ checkpoint registry
-------------------------------
Pre-trained CARE-PD checkpoints are matched to datasets by class name.
Datasets without a matching entry are silently skipped (``impute`` raises
``NotImplementedError`` so the pipeline records a missing-result rather
than crashing).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from motionbench.imputers.base import BaseImputer

if TYPE_CHECKING:
    from motionbench.data.base import BaseDataset


log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CARE-PD root — all checkpoint paths are relative to this.
#
# Resolution order:
#   1. CARE_PD_ROOT environment variable (preferred).
#   2. Sibling directory of this repo (../CARE-PD).
# ---------------------------------------------------------------------------
def _resolve_care_pd_root() -> Path:
    env = os.environ.get("CARE_PD_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2].parent / "CARE-PD"


_CARE_PD_ROOT = _resolve_care_pd_root()

# ---------------------------------------------------------------------------
# Registry: dataset class-name  →  (ckpt_dir, cfg_path)
# Paths are relative to _CARE_PD_ROOT.
# ---------------------------------------------------------------------------
_VAEAC_REGISTRY: dict[str, tuple[str, str]] = {
    # J=17, T=16 — newly trained for motionbench-xai synthetic datasets
    "SkeletonStructuredDataset": (
        "experiment_outs/vaeac_synthetic/skeleton_t16",
        "configs/vaeac/skeleton_t16.json",
    ),
    "JointSubsetSkeletonDataset": (
        "experiment_outs/vaeac_synthetic/joint_subset_skel_t16",
        "configs/vaeac/joint_subset_skel_t16.json",
    ),
    "GaitPeriodicDataset": (
        "experiment_outs/vaeac_synthetic/gait_t16",
        "configs/vaeac/gait_t16.json",
    ),
    # Fourth-quadrant pillar: high spatial coupling × strong periodic temporal (J=17, T=16)
    "SkeletonGaitDataset": (
        "experiment_outs/vaeac_synthetic/skeleton_gait_combined",
        "configs/vaeac/skeleton_gait_combined.json",
    ),
    # J=5, T=16 Gaussian variants
    "GaussianMotionDataset": (
        "experiment_outs/vaeac_synthetic/gaussian_k8_t16",
        "configs/vaeac/gaussian_k8_t16.json",
    ),
    # Low-rank manifold uses same Gaussian structure (J=17, T=16)
    "LowRankManifoldDataset": (
        "experiment_outs/vaeac_synthetic/skeleton_t16",
        "configs/vaeac/skeleton_t16.json",
    ),
    # Burr-marginal datasets — newly retrained for (J=5, F=3, T=20) BurrMotionBenchmark.
    # Same checkpoint covers burr_m5 and burr_m10 (M is a pipeline-time partition only).
    "BurrMotionBenchmark": (
        "experiment_outs/vaeac_synthetic/burr_jft_t20_j5",
        "configs/vaeac/burr_jft_t20_j5.json",
    ),
    # Real CARE-PD BMCLab gait — J=17, T=80
    "BMCLabDataset": (
        "experiment_outs/vaeac_real/bmclab_fold1_real_gait_bm",
        "configs/vaeac/bmclab_fold1_real_gait_bm.json",
    ),
    "BMCLabCacheDataset": (
        "experiment_outs/vaeac_real/bmclab_fold1_real_gait_bm",
        "configs/vaeac/bmclab_fold1_real_gait_bm.json",
    ),
}

_FLOW_REGISTRY: dict[str, tuple[str, str]] = {
    # J=17, T=16
    "SkeletonStructuredDataset": (
        "experiment_outs/flow_matching_synthetic/skeleton_t16",
        "configs/flow_matching/skeleton_t16.json",
    ),
    "JointSubsetSkeletonDataset": (
        "experiment_outs/flow_matching_synthetic/joint_subset_skel_t16",
        "configs/flow_matching/joint_subset_skel_t16.json",
    ),
    "GaitPeriodicDataset": (
        "experiment_outs/flow_matching_synthetic/gait_t16",
        "configs/flow_matching/gait_t16.json",
    ),
    # Fourth-quadrant pillar: high spatial coupling × strong periodic temporal (J=17, T=16)
    "SkeletonGaitDataset": (
        "experiment_outs/flow_matching_synthetic/skeleton_gait_combined",
        "configs/flow_matching/skeleton_gait_combined.json",
    ),
    # J=5, T=16 Gaussian variants
    "GaussianMotionDataset": (
        "experiment_outs/flow_matching_synthetic/gaussian_k8_t16",
        "configs/flow_matching/gaussian_k8_t16.json",
    ),
    # Low-rank manifold uses skeleton structure (J=17, T=16)
    "LowRankManifoldDataset": (
        "experiment_outs/flow_matching_synthetic/skeleton_t16",
        "configs/flow_matching/skeleton_t16.json",
    ),
    # Burr-marginal datasets — newly retrained for (J=5, F=3, T=20) BurrMotionBenchmark.
    "BurrMotionBenchmark": (
        "experiment_outs/flow_matching_synthetic/burr_jft_t20_j5",
        "configs/flow_matching/burr_jft_t20_j5.json",
    ),
    # Real CARE-PD BMCLab gait — J=17, T=80
    "BMCLabDataset": (
        "experiment_outs/flow_matching/bmclab_h36m3d_fold1",
        "configs/flow_matching/bmclab_h36m3d_fold1.json",
    ),
    "BMCLabCacheDataset": (
        "experiment_outs/flow_matching/bmclab_h36m3d_fold1",
        "configs/flow_matching/bmclab_h36m3d_fold1.json",
    ),
}


def _ensure_carepd_on_path() -> None:
    """Lazily add CARE-PD to sys.path for imports."""
    carepd = str(_CARE_PD_ROOT)
    if carepd not in sys.path:
        sys.path.insert(0, carepd)


def _mask_to_coalition(mask: Tensor) -> tuple[Tensor, str]:
    """Convert element-wise ``(J, F, T)`` mask → CARE-PD coalition format.

    Returns:
        coalition: ``(1, T)`` or ``(1, J)`` bool tensor.
        kind:      ``"temporal"`` or ``"spatial"``.
    """
    J, F, T = mask.shape
    # Check temporal structure: for every t, are all (j, f) identical?
    mask_t = mask[:, :, :]          # (J, F, T)
    per_t = mask_t[0, 0, :]        # (T,) — sample from first (j=0, f=0)
    is_temporal = (mask_t == per_t.view(1, 1, T)).all()

    if is_temporal:
        return per_t.unsqueeze(0).contiguous(), "temporal"   # (1, T)

    # Check spatial structure: for every j, are all (f, t) identical?
    per_j = mask_t[:, 0, 0]        # (J,)
    is_spatial = (mask_t == per_j.view(J, 1, 1)).all()

    if is_spatial:
        return per_j.unsqueeze(0).contiguous(), "spatial"    # (1, J)

    # Fallback: reduce to temporal by OR-across-joints
    log.warning(
        "Mask is neither purely temporal nor purely spatial; "
        "falling back to temporal heuristic for CARE-PD imputer."
    )
    temporal_any = mask_t.any(dim=0).any(dim=0)              # (T,)
    return temporal_any.unsqueeze(0).contiguous(), "temporal"


def _load_vaeac(ckpt_dir: Path, cfg_path: Path, device: torch.device):
    """Load a CARE-PD VAEAC checkpoint and return a VAEACImputer."""
    _ensure_carepd_on_path()
    from model.vaeac import VAEAC                        # type: ignore[import]
    from model.vaeac.imputer import VAEACImputer          # type: ignore[import]

    cfg = json.loads(cfg_path.read_text())

    # Resolve best checkpoint — accept both Lightning .ckpt and plain .pt saves.
    def _find_ckpt(directory: Path) -> Path:
        for pattern in ("*best*.ckpt", "*.ckpt", "*best*.pt", "*.pt"):
            hits = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
            if hits:
                return hits[-1]
        raise FileNotFoundError(f"No .ckpt/.pt files in {directory}")

    ckpt_path = _find_ckpt(ckpt_dir)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    raw = ckpt.get("state_dict", ckpt)
    # Strip "model." prefix if present (Lightning checkpoints), else use as-is.
    if any(k.startswith("model.") for k in raw):
        clean = {k[len("model."):]: v for k, v in raw.items() if k.startswith("model.")}
    else:
        clean = dict(raw)
    # Remap keys saved by older VAEAC versions:
    #   trunk.encoder.layers.X  →  trunk.layers.X  (TransformerEncoder wrapper rename)
    #   head.log_sigma           →  head.log_sigma_x
    def _remap_vaeac(k: str) -> str:
        k = k.replace(".trunk.encoder.layers.", ".trunk.layers.")
        k = k.replace("head.log_sigma", "head.log_sigma_x")
        return k
    clean = {_remap_vaeac(k): v for k, v in clean.items()}

    # Infer num_layers from the checkpoint itself (overrides config if mismatched).
    import re as _re
    layer_idxs = set()
    for k in clean:
        m = _re.search(r"\.trunk\.layers\.(\d+)\.", k)
        if m:
            layer_idxs.add(int(m.group(1)))
    if layer_idxs:
        inferred_layers = max(layer_idxs) + 1
        if inferred_layers != int(cfg.get("num_layers", inferred_layers)):
            log.warning(
                "[_load_vaeac] num_layers in config (%s) does not match checkpoint (%d); "
                "using checkpoint value.",
                cfg.get("num_layers"), inferred_layers,
            )
        cfg = dict(cfg)   # don't mutate the caller's dict
        cfg["num_layers"] = inferred_layers

    model = VAEAC(
        n_joints=int(cfg.get("n_joints", 17)),
        n_coords=int(cfg.get("n_coords", 3)),
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        num_layers=int(cfg["num_layers"]),
        ff_dim=int(cfg["ff_dim"]),
        dropout=float(cfg.get("dropout", 0.0)),
        d_latent=int(cfg["d_latent"]),
        max_len=max(int(cfg["seq_len"]) + 16, 256),
        decoder_head=str(cfg.get("decoder_head", "gaussian_scalar")),
        ivanov_min_sigma=float(cfg.get("ivanov_min_sigma", 1e-2)),
        use_prior_memory=bool(cfg.get("use_prior_memory", False)),
        prior_reg_sigma_mu=float(cfg.get("prior_reg_sigma_mu", 1e4)),
        prior_reg_sigma_sigma=float(cfg.get("prior_reg_sigma_sigma", 1e-4)),
    )
    model.load_state_dict(clean, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    log.info("[CarepdVAEAC] loaded from %s", ckpt_path)
    return VAEACImputer(model, device, stats_mean=None, stats_std=None, temperature=1.0)


def _load_flow(ckpt_dir: Path, cfg_path: Path, device: torch.device):
    """Load a CARE-PD flow-matching checkpoint and return a FlowImputer."""
    _ensure_carepd_on_path()
    from model.flow_matching import VelocityNet               # type: ignore[import]
    from model.flow_shap.imputer import FlowImputer           # type: ignore[import]

    cfg = json.loads(cfg_path.read_text())

    best = sorted(ckpt_dir.glob("*best*.ckpt"), key=lambda p: p.stat().st_mtime)
    ckpt_path = best[-1] if best else (ckpt_dir / "last.ckpt")
    if not ckpt_path.exists():
        # Also accept plain .pt saves (e.g. from train_ptbxl_classifier.py)
        for pattern in ("*best*.pt", "*.pt"):
            hits = sorted(ckpt_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
            if hits:
                ckpt_path = hits[-1]
                break
        else:
            ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
            if not ckpts:
                raise FileNotFoundError(f"No .ckpt/.pt files in {ckpt_dir}")
            ckpt_path = ckpts[-1]

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    raw = ckpt.get("state_dict", ckpt)
    # Strip "model." prefix if present (Lightning checkpoints), else use as-is.
    if any(k.startswith("model.") for k in raw):
        clean = {k[len("model."):]: v for k, v in raw.items() if k.startswith("model.")}
    else:
        clean = dict(raw)

    obs_cfg = bool(cfg.get("obs_conditioning", False))
    tok_cfg = str(cfg.get("tokenization", "frame"))

    net = VelocityNet(
        n_joints=int(cfg.get("n_joints", 17)),
        n_coords=int(cfg.get("n_coords", 3)),
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        num_layers=int(cfg["num_layers"]),
        ff_dim=int(cfg["ff_dim"]),
        dropout=float(cfg.get("dropout", 0.0)),
        time_emb_dim=int(cfg["time_emb_dim"]),
        max_len=max(int(cfg["seq_len"]) + 16, 256),
        tokenization=tok_cfg,
        obs_conditioning=obs_cfg,
    )
    net.load_state_dict(clean, strict=False)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    # Load whitening stats if present
    stats_mean = stats_std = None
    wpath = ckpt_dir / "whitening_stats.npz"
    if wpath.exists():
        import numpy as np
        ws = np.load(wpath)
        stats_mean = torch.from_numpy(ws["mean"])
        stats_std = torch.from_numpy(ws["std"])

    log.info("[CarepdFlow] loaded from %s (obs_cond=%s)", ckpt_path, obs_cfg)
    return FlowImputer(
        net, device,
        stats_mean=stats_mean, stats_std=stats_std,
        num_steps=int(cfg.get("num_steps", 50)),
        solver="midpoint",
        cfg_scale=float(cfg.get("cfg_scale", 0.0)),
    )


class CarepdVAEACImputer(BaseImputer):
    """CARE-PD VAEAC wrapped as a motionbench-xai BaseImputer.

    Loads a pre-trained VAEAC checkpoint from the CARE-PD project.
    The dataset class name is used to look up the correct checkpoint
    from ``_VAEAC_REGISTRY``.  Datasets without a registered checkpoint
    are silently skipped (``impute`` raises ``NotImplementedError``).

    Shape compatibility:
        The CARE-PD VAEAC was trained on ``(J=17, F=3, T=81)`` sequences.
        Datasets with different (J, T) require separate checkpoints or
        re-training.  Use ``GaussianOracle`` for Gaussian datasets where
        the exact conditional is available analytically.
    """

    def __init__(
        self,
        n_completion_samples: int = 10,
        device: str = "cpu",
        **_kwargs: object,  # absorb extra pipeline kwargs (J, F, T, etc.)
    ) -> None:
        self._n_completion_samples = int(n_completion_samples)
        self._device_str = device
        self._imputer = None   # set in fit()
        self._fitted = False
        self._skip = False

    def fit(self, train_data: "BaseDataset") -> "CarepdVAEACImputer":
        cls_name = type(train_data).__name__
        if cls_name not in _VAEAC_REGISTRY:
            log.warning(
                "CarepdVAEACImputer: no checkpoint registered for %s — "
                "this dataset will be skipped.",
                cls_name,
            )
            self._skip = True
            self._fitted = True
            return self

        ckpt_rel, cfg_rel = _VAEAC_REGISTRY[cls_name]
        ckpt_dir = _CARE_PD_ROOT / ckpt_rel
        cfg_path = _CARE_PD_ROOT / cfg_rel

        if not ckpt_dir.exists():
            log.warning(
                "CarepdVAEACImputer: checkpoint dir %s not found — skipping %s.",
                ckpt_dir, cls_name,
            )
            self._skip = True
            self._fitted = True
            return self

        device = torch.device(self._device_str)
        try:
            self._imputer = _load_vaeac(ckpt_dir, cfg_path, device)
            # Validate J/T compatibility with dataset
            J_data, _, T_data = train_data.shape
            cfg = json.loads(cfg_path.read_text())
            J_model = int(cfg.get("n_joints", 17))
            T_model = int(cfg["seq_len"])
            if J_data != J_model or T_data != T_model:
                log.warning(
                    "CarepdVAEACImputer: dimension mismatch for %s — "
                    "data (J=%d, T=%d) vs model (J=%d, T=%d). Skipping.",
                    cls_name, J_data, T_data, J_model, T_model,
                )
                self._skip = True
                self._imputer = None
        except Exception as exc:
            log.warning("CarepdVAEACImputer: failed to load checkpoint: %s", exc)
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
        if not self._fitted:
            raise RuntimeError("CarepdVAEACImputer.fit() must be called first.")
        if self._skip or self._imputer is None:
            raise NotImplementedError(
                "No CARE-PD VAEAC checkpoint available for this dataset."
            )

        J, F, T = x_obs.shape
        device = self._imputer._device

        # (J, F, T) → (1, J, F, T)
        x_in = x_obs.unsqueeze(0).to(device)
        pad = torch.ones(1, T, dtype=torch.bool, device=device)

        coalition_mask, kind = _mask_to_coalition(mask)
        coalition_mask = coalition_mask.to(device)

        completions = self._imputer.sample_completions(
            x=x_in,
            y=None,
            mask=pad,
            lengths=None,
            coalition_mask=coalition_mask,
            n_samples=n_samples,
        )                              # list of n_samples × (1, J, F, T)

        # Stack and enforce observed-entry preservation bit-for-bit
        out = torch.cat(completions, dim=0)                   # (n_samples, J, F, T)
        out_dev = out.device
        x_dev = x_obs.to(out_dev)
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


class CarepdFlowImputer(BaseImputer):
    """CARE-PD Flow Matching wrapped as a motionbench-xai BaseImputer.

    Loads a pre-trained FlowImputer checkpoint from the CARE-PD project.
    The dataset class name is used to look up the correct checkpoint
    from ``_FLOW_REGISTRY``.

    The flow ODE runs ``num_steps`` velocity-net evaluations per sample
    using the midpoint solver with RePaint-style harmonisation on observed
    entries.  This is slower than VAEAC (~50–100× cost per completion).
    """

    def __init__(
        self,
        num_steps: int = 50,
        device: str = "cpu",
        **_kwargs: object,  # absorb extra pipeline kwargs (J, F, T, etc.)
    ) -> None:
        self._num_steps = int(num_steps)
        self._device_str = device
        self._imputer = None
        self._fitted = False
        self._skip = False

    def fit(self, train_data: "BaseDataset") -> "CarepdFlowImputer":
        cls_name = type(train_data).__name__
        if cls_name not in _FLOW_REGISTRY:
            log.warning(
                "CarepdFlowImputer: no checkpoint registered for %s — skipping.",
                cls_name,
            )
            self._skip = True
            self._fitted = True
            return self

        ckpt_rel, cfg_rel = _FLOW_REGISTRY[cls_name]
        ckpt_dir = _CARE_PD_ROOT / ckpt_rel
        cfg_path = _CARE_PD_ROOT / cfg_rel

        if not ckpt_dir.exists():
            log.warning(
                "CarepdFlowImputer: checkpoint dir %s not found — skipping %s.",
                ckpt_dir, cls_name,
            )
            self._skip = True
            self._fitted = True
            return self

        device = torch.device(self._device_str)
        try:
            # Override num_steps with our requested value
            cfg = json.loads(cfg_path.read_text())
            cfg["num_steps"] = self._num_steps
            import tempfile, os
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(cfg, f)
                tmp_cfg = Path(f.name)
            try:
                self._imputer = _load_flow(ckpt_dir, tmp_cfg, device)
            finally:
                os.unlink(tmp_cfg)

            # Validate J/T compatibility
            J_data, _, T_data = train_data.shape
            J_model = int(cfg.get("n_joints", 17))
            T_model = int(cfg["seq_len"])
            if J_data != J_model or T_data != T_model:
                log.warning(
                    "CarepdFlowImputer: dimension mismatch for %s — "
                    "data (J=%d, T=%d) vs model (J=%d, T=%d). Skipping.",
                    cls_name, J_data, T_data, J_model, T_model,
                )
                self._skip = True
                self._imputer = None
        except Exception as exc:
            log.warning("CarepdFlowImputer: failed to load checkpoint: %s", exc)
            self._skip = True

        self._fitted = True
        return self

    def precompute_all_temporal_coalitions(
        self,
        x_obs: Tensor,
        K: int,
        n_samples: int = 1,
    ) -> None:
        """Pre-compute completions for all 2^K temporal-window coalitions.

        Uses ``sample_completions_batched`` so all coalitions share a single
        ODE integration — a ~K-fold speedup over calling ``impute`` 2^K times.
        Results are stored in ``self._completion_cache`` as a mapping from
        a tuple of 0/1 window bits → ``(n_samples, J, F, T)`` CPU tensor.

        Call this once per sequence before running KernelSHAP.
        """
        if self._skip or self._imputer is None:
            return
        if not hasattr(self._imputer, "sample_completions_batched"):
            return  # fallback: let impute() handle it call-by-call

        J, F, T = x_obs.shape
        T_win = T // K  # frames per window
        device = self._imputer._device

        # Build all 2^K coalition masks (B, T) — each row is a binary temporal mask
        n_coal = 2 ** K
        all_coal_masks = torch.zeros(n_coal, T, dtype=torch.bool)
        for coal_idx in range(n_coal):
            for win in range(K):
                if (coal_idx >> win) & 1:
                    start = win * T_win
                    end = start + T_win if win < K - 1 else T
                    all_coal_masks[coal_idx, start:end] = True

        x_in = x_obs.unsqueeze(0)           # (1, J, F, T)
        pad = torch.ones(1, T, dtype=torch.bool)

        # Run all 2^K coalitions in one batched ODE integration
        # Returns shape (B, n_samples, J, F, T) in classifier layout
        batched_out = self._imputer.sample_completions_batched(
            x=x_in,
            mask=pad,
            coalition_masks=all_coal_masks,
            n_samples=n_samples,
        )                                    # (B, n_samples, J, F, T)
        batched_out = batched_out.cpu()

        # Store in cache keyed by tuple of T-length 0/1 strings
        self._completion_cache: dict[tuple, Tensor] = {}
        for coal_idx in range(n_coal):
            key = tuple(all_coal_masks[coal_idx].tolist())
            completions = batched_out[coal_idx]               # (n_samples, J, F, T)
            # Overwrite observed entries bit-for-bit
            m_key = all_coal_masks[coal_idx].view(1, 1, 1, T).expand(n_samples, J, F, T)
            completions = torch.where(
                m_key,
                x_obs.cpu().unsqueeze(0).expand(n_samples, -1, -1, -1),
                completions,
            )
            self._completion_cache[key] = completions.float()

        self._cached_x_obs = x_obs.cpu()
        log.debug(
            "CarepdFlowImputer: pre-computed %d temporal coalitions (K=%d, T=%d)",
            n_coal, K, T,
        )

    def clear_cache(self) -> None:
        self._completion_cache = {}

    def impute(
        self,
        x_obs: Tensor,
        mask: Tensor,
        n_samples: int,
        seed: int | None = None,
    ) -> Tensor:
        if not self._fitted:
            raise RuntimeError("CarepdFlowImputer.fit() must be called first.")
        if self._skip or self._imputer is None:
            raise NotImplementedError(
                "No CARE-PD Flow checkpoint available for this dataset."
            )

        # Fast path: use pre-computed coalition cache if available
        cache = getattr(self, "_completion_cache", {})
        if cache:
            J, F, T = x_obs.shape
            coalition_mask_1d, kind = _mask_to_coalition(mask)
            # coalition_mask_1d shape: (1, T) or (1, J)
            if kind == "temporal" and coalition_mask_1d.shape[-1] == T:
                key = tuple(coalition_mask_1d[0].tolist())
                if key in cache:
                    out = cache[key]   # (n_cached_samples, J, F, T)
                    if n_samples <= out.shape[0]:
                        out = out[:n_samples]
                    return out.float().cpu()

        # Slow path: single-coalition ODE integration
        J, F, T = x_obs.shape
        device = self._imputer._device

        x_in = x_obs.unsqueeze(0).to(device)
        pad = torch.ones(1, T, dtype=torch.bool, device=device)

        coalition_mask, kind = _mask_to_coalition(mask)
        coalition_mask = coalition_mask.to(device)

        completions = self._imputer.sample_completions(
            x=x_in,
            y=None,
            mask=pad,
            lengths=None,
            coalition_mask=coalition_mask,
            n_samples=n_samples,
        )                              # list of n_samples × (1, J, F, T)

        out = torch.cat(completions, dim=0)                   # (n_samples, J, F, T)
        out_dev = out.device
        x_dev = x_obs.to(out_dev)
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
