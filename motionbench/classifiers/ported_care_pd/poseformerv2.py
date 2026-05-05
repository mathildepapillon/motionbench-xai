"""motionbench.classifiers.ported_care_pd.poseformerv2 — PoseFormerV2 encoder.

Faithfully ports ``CARE-PD/model/poseformerv2/model_poseformer.py`` so that
the CARE-PD fine-tuned ``.pth.tr`` checkpoints load without structural
mismatches.

Shape convention
----------------
Raw input ``x``:  ``(B, J, F=2, T)``  — 2-D image-projected pixel coords
Output logits:    ``(B, n_classes)``

CARE-PD training configuration (BMCLab, poseformerv2_BMCLab/0)
---------------------------------------------------------------
* ``embed_dim_ratio = 32``  → embed_dim = 32 * 17 = 544
* ``depth = 4``,  ``num_heads = 8``
* ``number_of_kept_frames = 9``,  ``number_of_kept_coeffs = 9``
* ``source_seq_len = 81`` frames per clip (stride = 81, no overlap)
* ``in_data_dim = 2``  (2-D screen coordinates, NO confidence score)
* ``merge_joints = False``  → head input dim = embed_dim * 2 = 1088
* preprocessing: ``normalize_screen_coordinates(1100, 1100)``
  → ``(xy_pixels / 1100) * 2 − [1, 1]``
* data source: ``h36m_3d_world2cam2img_backright_floorXZZplus_30f_or_longer.npz``
  (already 2-D image-projected, NOT the 3-D world-to-camera NPZ)

Per-classifier preprocessing (inside forward)
----------------------------------------------
This classifier's ``input_feature_dim = 2``.  The raw input is
``(B, J, 2, T)`` 2-D pixel coordinates.  ``_preprocess`` applies screen
normalisation so the backbone sees the same distribution as during training.
Gradients flow through the normalisation transparently, enabling XAI
attributions w.r.t. the raw 2-D pixel coordinates.
"""

from __future__ import annotations

import logging
import math
from functools import partial
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_dct as dct  # type: ignore[import]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torch_dct is required for PoseFormerV2. "
        "Install it with: pip install torch-dct"
    ) from exc

try:
    from einops import rearrange  # type: ignore[import]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "einops is required for PoseFormerV2. "
        "Install it with: pip install einops"
    ) from exc

from timm.layers import DropPath
from torch import Tensor

from motionbench.classifiers.base import Classifier

logger = logging.getLogger(__name__)

__all__ = ["PoseFormerV2Classifier"]


# ---------------------------------------------------------------------------
# Sub-modules (verbatim from CARE-PD/model/poseformerv2/model_poseformer.py)
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
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, C // self.num_heads)
               .permute(2, 0, 3, 1, 4))
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
        self.attn = _Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
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
        self.attn = _Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
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


# ---------------------------------------------------------------------------
# Backbone: PoseTransformerV2
# ---------------------------------------------------------------------------


class _PoseTransformerV2Backbone(nn.Module):
    """Verbatim port of ``PoseTransformerV2`` from CARE-PD.

    Expects ``(B, T, J, C)`` input where C=in_chans (2 for 2-D poses).
    Returns ``(B, 1, embed_dim * 2)`` representation when ``return_rep=True``.
    """

    def __init__(self, num_joints=17, in_chans=2, embed_dim_ratio=2, depth=1,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=None, number_of_kept_frames=1,
                 number_of_kept_coeffs=1):
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
            _Block(dim=embed_dim_ratio, num_heads=num_heads,
                   mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                   qk_scale=qk_scale, drop=drop_rate,
                   attn_drop=attn_drop_rate, drop_path=dpr[i],
                   norm_layer=norm_layer)
            for i in range(depth)
        ])

        self.blocks = nn.ModuleList([
            _MixedBlock(dim=embed_dim, num_heads=num_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        qk_scale=qk_scale, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[i],
                        norm_layer=norm_layer)
            for i in range(depth)
        ])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        self.weighted_mean = nn.Conv1d(in_channels=self.num_coeff_kept,
                                       out_channels=1, kernel_size=1)
        self.weighted_mean_ = nn.Conv1d(in_channels=self.num_frame_kept,
                                        out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, out_dim),
        )

    def _Spatial_forward_features(self, x):
        b, f, p, _ = x.shape
        num_frame_kept = self.num_frame_kept
        index = torch.arange(
            (f - 1) // 2 - num_frame_kept // 2,
            (f - 1) // 2 + num_frame_kept // 2 + 1,
            device=x.device,
        )
        x = self.Joint_embedding(x[:, index].reshape(b * num_frame_kept, p, -1))
        x = x + self.Spatial_pos_embed
        x = self.pos_drop(x)
        for blk in self.Spatial_blocks:
            x = blk(x)
        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) p c -> b f (p c)', f=num_frame_kept)
        return x

    def _forward_features(self, x, Spatial_feature):
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

    def forward(self, x, return_rep=True):
        """``x``: ``(B, T, J, C)`` → returns ``(B, 1, embed_dim*2)``."""
        b, f, p, _ = x.shape
        x_ = x.clone()
        Spatial_feature = self._Spatial_forward_features(x)
        x = self._forward_features(x_, Spatial_feature)
        x = torch.cat(
            (self.weighted_mean(x[:, :self.num_coeff_kept]),
             self.weighted_mean_(x[:, self.num_coeff_kept:])),
            dim=-1,
        )
        if return_rep:
            return x  # (B, 1, embed_dim*2)
        x = self.head(x).reshape(b, 1, p, -1)
        return x


# ---------------------------------------------------------------------------
# Classifier wrapper
# ---------------------------------------------------------------------------


class PoseFormerV2Classifier(Classifier):
    """PoseFormerV2 fine-tuned on BMCLab (ported from CARE-PD).

    Args:
        checkpoint_path: Path to a CARE-PD ``.pth.tr`` fine-tuned checkpoint.
        n_classes: Number of output logit dimensions.
        num_joints: Skeleton joints (17 for H36M-17).
        embed_dim_ratio: Embedding dim per joint (32 in BMCLab training).
        depth: Transformer depth (4 in BMCLab training).
        number_of_kept_frames: Central frames for spatial branch (9).
        number_of_kept_coeffs: DCT coefficients for temporal branch (9).
        merge_joints: If True use mean pooling over joints before the head;
            if False flatten all joints (BMCLab uses False).
        image_resolution: ``(W, H)`` of the camera used during data collection.
            Default ``(1100, 1100)`` matches the BMCLab CARE-PD config.
    """

    #: PoseFormerV2 uses 2-D image-projected pixel coordinates.
    input_feature_dim: int = 2

    def __init__(
        self,
        checkpoint_path: Union[str, Path, None] = None,
        n_classes: int = 3,
        num_joints: int = 17,
        embed_dim_ratio: int = 32,
        depth: int = 4,
        number_of_kept_frames: int = 9,
        number_of_kept_coeffs: int = 9,
        merge_joints: bool = False,
        image_resolution: tuple[int, int] = (1100, 1100),
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, n_classes=n_classes)
        self._merge_joints = merge_joints
        self._img_w = float(image_resolution[0])
        self._img_h = float(image_resolution[1])

        self.backbone = _PoseTransformerV2Backbone(
            num_joints=num_joints,
            in_chans=2,
            embed_dim_ratio=embed_dim_ratio,
            depth=depth,
            number_of_kept_frames=number_of_kept_frames,
            number_of_kept_coeffs=number_of_kept_coeffs,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
        )

        embed_dim = embed_dim_ratio * num_joints  # 544
        head_dim = embed_dim * 2  # 1088 (two branches concatenated)
        self.cls_head = nn.Linear(head_dim, n_classes)

        if self._checkpoint_path is not None:
            self._load_care_pd_checkpoint(self._checkpoint_path)
            logger.info("PoseFormerV2: loaded CARE-PD fine-tuned checkpoint.")

    def _preprocess(self, x: Tensor) -> Tensor:
        """Apply CARE-PD screen normalisation.

        Matches PoseformerV2Preprocessor.normalize_screen_coordinates in
        CARE-PD/data/dataloaders.py::

            (xy_pixels / W) * 2 - [1, H/W]

        Because H == W == 1100 in the BMCLab config this simplifies to
        ``(xy / 1100) * 2 − 1`` for both axes.

        Args:
            x: ``(B, J, 2, T)`` raw 2-D pixel coordinates in ``[0, 1100]``.

        Returns:
            ``(B, T, J, 2)`` normalised coordinates in approximately ``[-1, 1]``.
        """
        # Normalise: (xy / W) * 2 - [1, H/W]
        # For the BMCLab config W == H == 1100 → both offsets are 1.0.
        offset = torch.tensor(
            [1.0, self._img_h / self._img_w],
            dtype=x.dtype, device=x.device,
        )  # (2,)

        # x: (B, J, 2, T) → (B, T, J, 2)
        x_btjf = x.permute(0, 3, 1, 2)
        x_norm = (x_btjf / self._img_w) * 2.0 - offset
        return x_norm  # (B, T, J, 2)

    def forward(self, x: Tensor) -> Tensor:
        """Map a batch of raw 2-D pose sequences to class logits.

        Args:
            x: Raw float32 tensor of shape ``(B, J, F=2, T)`` — 2-D
               image-projected pixel coordinates.

        Returns:
            Float32 tensor of shape ``(B, n_classes)`` — raw logits.
        """
        x_proc = self._preprocess(x)                    # (B, T, J, 2)
        rep = self.backbone(x_proc, return_rep=True)     # (B, 1, embed_dim*2)
        B, _, C = rep.shape
        rep = rep.reshape(B, C)                          # (B, embed_dim*2)
        return self.cls_head(rep)
