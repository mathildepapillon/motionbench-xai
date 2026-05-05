"""motionbench.classifiers.ported_care_pd.motionbert — MotionBERT encoder.

Faithfully ports ``CARE-PD/model/motionbert/DSTformer.py`` so that the
CARE-PD fine-tuned ``.pth.tr`` checkpoints load without structural
mismatches.

The key property preserved from CARE-PD is the *dual-stream* block design:
each ``Block`` has **separate** spatial (``_s``) and temporal (``_t``)
attention + MLP sub-modules, fused at the backbone level via a per-layer
``ts_attn`` linear.

Shape convention
----------------
Input ``x``:   ``(B, J, F=3, T)``
Output logits: ``(B, n_classes)``

CARE-PD training configuration (BMCLab, motionbert_BMCLab/0)
-------------------------------------------------------------
* ``dim_feat = dim_rep = 512``
* ``depth = 5``,  ``num_heads = 8``,  ``mlp_ratio = 2``  (default=4 in
  original code but CARE-PD uses 2)
* ``maxlen = 243``
* ``num_joints = 17``
* ``merge_joints = False``  → head input dim = 17 × 512 = 8 704
* ``in_data_dim = 2`` + ``simulate_confidence_score = True``  → 3rd feature
  dim = confidence (1.0 for real frames, 0.0 for padded)
* ``source_seq_len = 90`` frames per clip (stride = 90, no overlap)
* preprocessing: ``crop_scale`` (bounding-box normalise x,y to [-1,1])
"""

from __future__ import annotations

import collections
import logging
import math
import warnings
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from torch import Tensor

from motionbench.classifiers.base import Classifier, crop_scale_and_conf

logger = logging.getLogger(__name__)

__all__ = ["MotionBERTClassifier"]


# ---------------------------------------------------------------------------
# Utility — truncated normal init (verbatim from CARE-PD)
# ---------------------------------------------------------------------------


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lo = norm_cdf((a - mean) / std)
        hi = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lo - 1, 2 * hi - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def _trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ---------------------------------------------------------------------------
# DropPath (stochastic depth)
# ---------------------------------------------------------------------------


def _drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class _DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return _drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# MLP sub-module (verbatim from CARE-PD)
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
# Attention sub-module (verbatim from CARE-PD)
# ---------------------------------------------------------------------------


class _Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, st_mode="vanilla"):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.mode = st_mode
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, seqlen=1):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.mode in ("vanilla", "spatial"):
            x = self._forward_spatial(q, k, v)
        elif self.mode == "temporal":
            x = self._forward_temporal(q, k, v, seqlen=seqlen)
        else:
            raise NotImplementedError(self.mode)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _forward_spatial(self, q, k, v):
        B, _, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C * self.num_heads)
        return x

    def _forward_temporal(self, q, k, v, seqlen=8):
        B, _, N, C = q.shape
        qt = q.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)
        kt = k.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)
        vt = v.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)
        attn = (qt @ kt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ vt
        x = x.permute(0, 3, 2, 1, 4).reshape(B, N, C * self.num_heads)
        return x


# ---------------------------------------------------------------------------
# Block — separate spatial + temporal streams (verbatim from CARE-PD)
# ---------------------------------------------------------------------------


class _Block(nn.Module):
    """Dual-stream spatiotemporal transformer block.

    Has **separate** spatial (``_s``) and temporal (``_t``) sub-modules so
    that the CARE-PD checkpoint key names map directly.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, mlp_out_ratio=1.0,
                 qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 st_mode="stage_st", att_fuse=False):
        super().__init__()
        self.st_mode = st_mode
        self.norm1_s = norm_layer(dim)
        self.norm1_t = norm_layer(dim)
        self.attn_s = _Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, st_mode="spatial")
        self.attn_t = _Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, st_mode="temporal")
        self.drop_path = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2_s = norm_layer(dim)
        self.norm2_t = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_out_dim = int(dim * mlp_out_ratio)
        self.mlp_s = _MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                          out_features=mlp_out_dim, act_layer=act_layer, drop=drop)
        self.mlp_t = _MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                          out_features=mlp_out_dim, act_layer=act_layer, drop=drop)
        self.att_fuse = att_fuse
        if self.att_fuse:
            self.ts_attn = nn.Linear(dim * 2, dim * 2)

    def forward(self, x, seqlen=1):
        if self.st_mode == "stage_st":
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
        elif self.st_mode == "stage_ts":
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
        elif self.st_mode == "stage_para":
            x_t = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen))
            x_t = x_t + self.drop_path(self.mlp_t(self.norm2_t(x_t)))
            x_s = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
            x_s = x_s + self.drop_path(self.mlp_s(self.norm2_s(x_s)))
            if self.att_fuse:
                alpha = torch.cat([x_s, x_t], dim=-1)
                BF, J = alpha.shape[:2]
                alpha = self.ts_attn(alpha).reshape(BF, J, -1, 2)
                alpha = alpha.softmax(dim=-1)
                x = x_t * alpha[:, :, :, 1] + x_s * alpha[:, :, :, 0]
            else:
                x = (x_s + x_t) * 0.5
        else:
            raise NotImplementedError(self.st_mode)
        return x


# ---------------------------------------------------------------------------
# DSTformer backbone (verbatim from CARE-PD, except "backbone" key prefix
# is handled by MotionBERTClassifier wrapping it as self.backbone)
# ---------------------------------------------------------------------------


class _DSTformerBackbone(nn.Module):
    """Dual-Stream Spatiotemporal Transformer (DSTformer) backbone.

    Faithfully ports ``CARE-PD/model/motionbert/DSTformer.py``.
    The ``att_fuse=True`` default matches CARE-PD training, producing
    per-layer learnable weights ``ts_attn`` in the backbone's state dict.
    """

    def __init__(
        self,
        dim_in: int = 3,
        dim_out: int = 3,
        dim_feat: int = 256,
        dim_rep: int = 512,
        depth: int = 5,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_joints: int = 17,
        maxlen: int = 243,
        qkv_bias: bool = True,
        qk_scale=None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer=nn.LayerNorm,
        att_fuse: bool = True,
    ) -> None:
        super().__init__()
        self.dim_feat = dim_feat
        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks_st = nn.ModuleList([
            _Block(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, st_mode="stage_st")
            for i in range(depth)
        ])
        self.blocks_ts = nn.ModuleList([
            _Block(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, st_mode="stage_ts")
            for i in range(depth)
        ])

        self.norm = norm_layer(dim_feat)
        if dim_rep:
            self.pre_logits = nn.Sequential(collections.OrderedDict([
                ("fc", nn.Linear(dim_feat, dim_rep)),
                ("act", nn.Tanh()),
            ]))
        else:
            self.pre_logits = nn.Identity()
        self.head = nn.Linear(dim_rep, dim_out) if dim_out > 0 else nn.Identity()

        self.temp_embed = nn.Parameter(torch.zeros(1, maxlen, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        _trunc_normal_(self.temp_embed, std=0.02)
        _trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

        self.att_fuse = att_fuse
        if self.att_fuse:
            self.ts_attn = nn.ModuleList([
                nn.Linear(dim_feat * 2, 2) for _ in range(depth)
            ])
            for lin in self.ts_attn:
                lin.weight.data.fill_(0)
                lin.bias.data.fill_(0.5)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: Tensor, return_rep: bool = True) -> Tensor:
        """Forward pass.

        Args:
            x: ``(B, F, J, dim_in)`` input tensor (F = temporal length).
            return_rep: If ``True``, return ``(B, F, J, dim_rep)`` embedding.
        """
        B, F, J, C = x.shape
        x = x.reshape(-1, J, C)  # (B*F, J, C)
        BF = x.shape[0]
        x = self.joints_embed(x)          # (B*F, J, dim_feat)
        x = x + self.pos_embed
        _, J, C = x.shape
        x = x.reshape(-1, F, J, C) + self.temp_embed[:, :F, :, :]
        x = x.reshape(BF, J, C)
        x = self.pos_drop(x)

        for idx, (blk_st, blk_ts) in enumerate(zip(self.blocks_st, self.blocks_ts)):
            x_st = blk_st(x, F)
            x_ts = blk_ts(x, F)
            if self.att_fuse:
                att = self.ts_attn[idx]
                alpha = torch.cat([x_st, x_ts], dim=-1)
                BF_a, J_a = alpha.shape[:2]
                alpha = att(alpha)          # (B*F, J, 2)
                alpha = alpha.softmax(dim=-1)
                x = x_st * alpha[:, :, 0:1] + x_ts * alpha[:, :, 1:2]
            else:
                x = (x_st + x_ts) * 0.5

        x = self.norm(x)
        x = x.reshape(B, F, J, -1)
        x = self.pre_logits(x)            # (B, F, J, dim_rep)
        if return_rep:
            return x
        x = self.head(x)
        return x


# ---------------------------------------------------------------------------
# motionbench Classifier wrapper
# ---------------------------------------------------------------------------


class MotionBERTClassifier(Classifier):
    """MotionBERT encoder with a linear classification head.

    Wraps :class:`_DSTformerBackbone` and exposes the standard motionbench
    classifier interface ``(B, J, F, T) → (B, n_classes)``.

    Args:
        checkpoint_path: Path to a checkpoint.  Pass a ``.pth.tr`` path to
            load a CARE-PD fine-tuned checkpoint; any other extension loads
            only the backbone (pre-trained 3D-HPE weights).
        n_classes: Number of output logit dimensions (default 3 for
            UPDRS-gait 0/1/2 on BMCLab).
        dim_feat: Feature dimension of the transformer (default 512).
        dim_rep: Representation dimension after pre-logit projection
            (default 512).
        depth: Number of dual-stream transformer layers (default 5).
        num_heads: Number of attention heads (default 8).
        mlp_ratio: MLP hidden-to-input ratio.  CARE-PD used ``2`` for
            BMCLab (not the original MotionBERT default of 4).
        maxlen: Maximum sequence length for temporal positional embedding
            (default 243).
        num_joints: Number of skeletal joints (default 17).
        merge_joints: If ``True``, mean-pool over joints before the
            classifier head (head input dim = ``dim_rep``).  If ``False``
            (default, matches CARE-PD training), keep joints separate and
            flatten (head input dim = ``num_joints * dim_rep``).

    Input preprocessing (for CARE-PD BMCLab checkpoints)
    -----------------------------------------------------
    CARE-PD trained with ``in_data_dim=2`` +
    ``simulate_confidence_score=True``.  This means the 3rd feature
    dimension in each (T, J, 3) clip is a **confidence score** (1.0 for
    real frames, 0.0 for padded frames), NOT the z-coordinate.  x,y
    are bounding-box normalised to [-1,1] via ``crop_scale``.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path, None] = None,
        n_classes: int = 3,
        dim_feat: int = 512,
        dim_rep: int = 512,
        depth: int = 5,
        num_heads: int = 8,
        mlp_ratio: int = 2,
        maxlen: int = 243,
        num_joints: int = 17,
        merge_joints: bool = False,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, n_classes=n_classes)
        self._merge_joints = merge_joints

        self.backbone = _DSTformerBackbone(
            dim_in=3,
            dim_out=3,
            dim_feat=dim_feat,
            dim_rep=dim_rep,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_joints=num_joints,
            maxlen=maxlen,
            drop_path_rate=0.0,
            att_fuse=True,
        )
        head_dim = dim_rep if merge_joints else dim_rep * num_joints
        self.cls_head = nn.Linear(head_dim, n_classes)

        if self._checkpoint_path is not None:
            if str(self._checkpoint_path).endswith(".pth.tr"):
                self._load_care_pd_checkpoint(self._checkpoint_path)
                logger.info("MotionBERT: loaded CARE-PD fine-tuned checkpoint.")
            else:
                raw = torch.load(
                    self._checkpoint_path,
                    map_location=lambda storage, _: storage,
                    weights_only=False,
                )
                # Pre-trained backbone weights are a raw state dict
                state = raw if isinstance(raw, dict) and "model" not in raw else raw.get("model", raw)
                # Add "backbone." prefix if keys lack it
                if not any(k.startswith("backbone.") for k in state):
                    state = {f"backbone.{k}": v for k, v in state.items()}
                result = self.load_state_dict(state, strict=False)
                logger.info(
                    "MotionBERT: pre-trained backbone loaded; missing=%s, unexpected=%s",
                    result.missing_keys[:3],
                    result.unexpected_keys[:3],
                )

    def _preprocess(self, x: Tensor) -> Tensor:
        """Apply CARE-PD preprocessing: crop_scale + confidence channel.

        Matches MotionBERTPreprocessor in CARE-PD/data/dataloaders.py:
        - Bounding-box normalise (x, y) to [-1, 1] using the valid-frame bbox.
        - Replace z with a binary confidence: 1.0 real, 0.0 padded.

        Args:
            x: ``(B, J, 3, T)`` raw 3-D world-to-camera coordinates.
               Padded frames must have all coords == 0.

        Returns:
            ``(B, T, J, 3)`` preprocessed tensor ready for the DSTformer.
        """
        # (B, J, 3, T) → (B, T, J, 3)
        x_btjf = x.permute(0, 3, 1, 2).contiguous()
        return crop_scale_and_conf(x_btjf)  # (B, T, J, 3)

    def forward(self, x: Tensor) -> Tensor:
        """Map a batch of raw 3D pose sequences to class logits.

        Args:
            x: Raw float32 tensor of shape ``(B, J, F=3, T)`` — 3-D
               world-to-camera coordinates.  Padded frames should be
               all-zero so that ``_preprocess`` can detect them.

        Returns:
            Float32 tensor of shape ``(B, n_classes)`` — raw logits.
        """
        # (B, J, 3, T) → (B, T, J, 3), crop_scale + confidence
        x_proc = self._preprocess(x)           # (B, T, J, 3)
        rep = self.backbone(x_proc, return_rep=True)   # (B, T, J, dim_rep)
        rep = rep.mean(dim=1)                      # mean over T: (B, J, dim_rep)
        if self._merge_joints:
            rep = rep.mean(dim=1)                  # mean over J: (B, dim_rep)
        else:
            rep = rep.flatten(1)                   # flatten J: (B, J * dim_rep)
        return self.cls_head(rep)
