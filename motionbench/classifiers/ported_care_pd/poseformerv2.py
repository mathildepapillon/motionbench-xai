"""motionbench.classifiers.ported_care_pd.poseformerv2 — PoseFormerV2 encoder.

Port of ``CARE-PD/model/poseformerv2/model_poseformer.py`` adapted for the
motionbench ``(B, J, F, T)`` input convention.

Architecture overview
---------------------
PoseFormerV2 is a dual-stream spatial-temporal transformer for 3D pose
estimation, introduced in:

    Qitao Zhao et al. "PoseFormerV2: Exploring Frequency Domain for
    Efficient and Robust 3D Human Pose Estimation." CVPR 2023.

The encoder produces a pooled embedding of dimension
``embed_dim_ratio * num_joints * 2``.  A thin :class:`torch.nn.Linear` head
maps this to ``n_classes`` logits.

Checkpoint loading
------------------
Pre-trained backbone weights are available at::

    CARE-PD/assets/Pretrained_checkpoints/poseformerv2/9_81_46.0.bin

loaded under the key ``"model_pos"``.

**Input-channel note:** the original PoseFormerV2 was pre-trained with
``in_chans=2`` (2D projected joints).  motionbench uses 3D xyz coordinates
(``F=3``).  We therefore instantiate the backbone with ``in_chans=3``.
When the pre-trained checkpoint is loaded, the two input embedding layers
(``Joint_embedding`` and ``Freq_embedding``) are discarded (shape mismatch)
and randomly re-initialised.  All transformer blocks transfer cleanly.
The reproducibility gate therefore cannot be run against the CARE-PD paper
numbers without fine-tuned classifier weights (see TASKS.md row 4B).

Shape convention
----------------
Input ``x``:   ``(B, J, F=3, T)``
Output logits: ``(B, n_classes)``
"""

from __future__ import annotations

import logging
import math
from functools import partial
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401  (kept for completeness)
from einops import rearrange
from torch import Tensor

try:
    import torch_dct as dct  # type: ignore[import-untyped]
except ImportError as exc:
    raise ImportError(
        "torch_dct is required for PoseFormerV2.  "
        "Install with: pip install torch-dct"
    ) from exc

try:
    from timm.layers import DropPath  # type: ignore[import-untyped]
except ImportError as exc:
    raise ImportError(
        "timm is required for PoseFormerV2.  Install with: pip install timm"
    ) from exc

from motionbench.classifiers.base import Classifier

logger = logging.getLogger(__name__)

__all__ = ["PoseFormerV2Classifier"]

# ---------------------------------------------------------------------------
# Backbone sub-modules (copied verbatim from CARE-PD source)
# ---------------------------------------------------------------------------


class _Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
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


class _FreqMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        b, f, _ = x.shape
        x = dct.dct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = dct.idct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        return x


class _Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                               qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                        act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class _MixedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                               qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = _Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                         act_layer=act_layer, drop=drop)
        self.mlp2 = _FreqMlp(in_features=dim, hidden_features=mlp_hidden_dim,
                              act_layer=act_layer, drop=drop)

    def forward(self, x):
        b, f, c = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x1 = x[:, :f // 2] + self.drop_path(self.mlp1(self.norm2(x[:, :f // 2])))
        x2 = x[:, f // 2:] + self.drop_path(self.mlp2(self.norm3(x[:, f // 2:])))
        return torch.cat((x1, x2), dim=1)


class _PoseTransformerV2Backbone(nn.Module):
    """PoseFormerV2 backbone (encoder only).

    Accepts ``(B, T, J, in_chans)`` and returns a pooled embedding of shape
    ``(B, 1, embed_dim * 2)`` when ``return_rep=True``.
    """

    def __init__(
        self,
        num_joints: int = 17,
        in_chans: int = 3,
        embed_dim_ratio: int = 32,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 2.,
        qkv_bias: bool = True,
        qk_scale=None,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.2,
        norm_layer=None,
        number_of_kept_frames: int = 9,
        number_of_kept_coeffs: int = 9,
    ) -> None:
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints
        out_dim = num_joints * 3
        self.num_frame_kept = number_of_kept_frames
        self.num_coeff_kept = number_of_kept_coeffs

        self.Joint_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Freq_embedding = nn.Linear(in_chans * num_joints, embed_dim)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, self.num_frame_kept, embed_dim))
        self.Temporal_pos_embed_ = nn.Parameter(torch.zeros(1, self.num_coeff_kept, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.Spatial_blocks = nn.ModuleList([
            _Block(dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio,
                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                   drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                   norm_layer=norm_layer)
            for i in range(depth)
        ])

        self.blocks = nn.ModuleList([
            _MixedBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                        norm_layer=norm_layer)
            for i in range(depth)
        ])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)
        self.weighted_mean = nn.Conv1d(in_channels=self.num_coeff_kept, out_channels=1, kernel_size=1)
        self.weighted_mean_ = nn.Conv1d(in_channels=self.num_frame_kept, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, out_dim),
        )

    def Spatial_forward_features(self, x):
        b, f, p, _ = x.shape
        num_frame_kept = self.num_frame_kept
        index = torch.arange(
            (f - 1) // 2 - num_frame_kept // 2,
            (f - 1) // 2 + num_frame_kept // 2 + 1,
        )
        x = self.Joint_embedding(x[:, index].reshape(b * num_frame_kept, p, -1))
        x = x + self.Spatial_pos_embed
        x = self.pos_drop(x)
        for blk in self.Spatial_blocks:
            x = blk(x)
        x = self.Spatial_norm(x)
        x = rearrange(x, "(b f) p c -> b f (p c)", f=num_frame_kept)
        return x

    def forward_features(self, x, Spatial_feature):
        b, f, p, _ = x.shape
        num_coeff_kept = self.num_coeff_kept
        x = dct.dct(x.permute(0, 2, 3, 1))[:, :, :, :num_coeff_kept]
        x = x.permute(0, 3, 1, 2).contiguous().reshape(b, num_coeff_kept, -1)
        x = self.Freq_embedding(x)
        Spatial_feature = Spatial_feature + self.Temporal_pos_embed
        x = x + self.Temporal_pos_embed_
        x = torch.cat((x, Spatial_feature), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.Temporal_norm(x)
        return x

    def forward(self, x: Tensor, return_rep: bool = True) -> Tensor:
        """Forward pass.

        Args:
            x: ``(B, T, J, in_chans)`` input tensor.
            return_rep: If ``True``, return the pooled embedding
                ``(B, 1, embed_dim*2)``; if ``False``, apply the regression
                head and return ``(B, 1, J, 3)``.

        Returns:
            Pooled embedding or pose predictions.
        """
        b, f, p, _ = x.shape
        x_ = x.clone()
        Spatial_feature = self.Spatial_forward_features(x)
        x = self.forward_features(x_, Spatial_feature)
        x = torch.cat(
            (self.weighted_mean(x[:, :self.num_coeff_kept]),
             self.weighted_mean_(x[:, self.num_coeff_kept:])),
            dim=-1,
        )
        if return_rep:
            return x
        x = self.head(x).reshape(b, 1, p, -1)
        return x


# ---------------------------------------------------------------------------
# motionbench Classifier wrapper
# ---------------------------------------------------------------------------


class PoseFormerV2Classifier(Classifier):
    """PoseFormerV2 encoder with a linear classification head.

    Wraps :class:`_PoseTransformerV2Backbone` and exposes the standard
    motionbench classifier interface ``(B, J, F, T) → (B, n_classes)``.

    Args:
        checkpoint_path: Path to the pre-trained backbone checkpoint
            (``poseformerv2/9_81_46.0.bin``).  If ``None``, the backbone is
            randomly initialised (useful for architecture tests).
        n_classes: Number of output logit dimensions (default 4 for
            UPDRS-gait scores 0–3).
        num_joints: Number of skeletal joints (default 17 for H36M-17).
        embed_dim_ratio: Spatial embedding dimension ratio (default 32).
        depth: Number of transformer layers (default 4).
        number_of_kept_frames: Central frames used in spatial branch
            (default 9).
        number_of_kept_coeffs: DCT coefficients used in frequency branch
            (default 9).

    Note:
        The pre-trained checkpoint was trained with ``in_chans=2`` (2D joints).
        This wrapper uses ``in_chans=3`` (3D xyz) for motionbench
        compatibility.  The input embedding layers
        (``Joint_embedding``, ``Freq_embedding``) are randomly initialised;
        all transformer blocks are loaded from the checkpoint.
    """

    #: Encoder output dimension (embed_dim_ratio * num_joints * 2)
    ENCODER_DIM: int = 32 * 17 * 2  # 1088

    def __init__(
        self,
        checkpoint_path: Union[str, Path, None] = None,
        n_classes: int = 4,
        num_joints: int = 17,
        embed_dim_ratio: int = 32,
        depth: int = 4,
        number_of_kept_frames: int = 9,
        number_of_kept_coeffs: int = 9,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, n_classes=n_classes)

        encoder_dim = embed_dim_ratio * num_joints * 2  # 1088 with defaults
        self.ENCODER_DIM = encoder_dim

        self.backbone = _PoseTransformerV2Backbone(
            num_joints=num_joints,
            in_chans=3,
            embed_dim_ratio=embed_dim_ratio,
            depth=depth,
            number_of_kept_frames=number_of_kept_frames,
            number_of_kept_coeffs=number_of_kept_coeffs,
            drop_path_rate=0.,
        )
        self.cls_head = nn.Linear(encoder_dim, n_classes)

        if self._checkpoint_path is not None:
            if str(self._checkpoint_path).endswith(".pth.tr"):
                self._load_care_pd_checkpoint(self._checkpoint_path)
                logger.info("PoseFormerV2: loaded CARE-PD fine-tuned checkpoint.")
            else:
                matched, discarded = self._load_checkpoint(
                    self._checkpoint_path,
                    self.backbone,
                    ckpt_key="model_pos",
                    strict=False,
                )
                logger.info(
                    "PoseFormerV2: loaded %d layers, discarded %d "
                    "(expected: Joint_embedding/Freq_embedding discarded due to "
                    "in_chans 2→3 mismatch).",
                    len(matched),
                    len(discarded),
                )

    def forward(self, x: Tensor) -> Tensor:
        """Map a batch of 3D pose sequences to class logits.

        Args:
            x: Float32 tensor of shape ``(B, J, F=3, T)``.

        Returns:
            Float32 tensor of shape ``(B, n_classes)`` — raw logits.
        """
        # (B, J, F=3, T) → (B, T, J, F=3)
        x = x.permute(0, 3, 1, 2)
        rep = self.backbone(x, return_rep=True)  # (B, 1, encoder_dim)
        rep = rep.squeeze(1)  # (B, encoder_dim)
        return self.cls_head(rep)
