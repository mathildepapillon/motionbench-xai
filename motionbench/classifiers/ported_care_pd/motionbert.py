"""motionbench.classifiers.ported_care_pd.motionbert — MotionBERT encoder.

Port of ``CARE-PD/model/motionbert/DSTformer.py`` adapted for the motionbench
``(B, J, F, T)`` input convention.

Architecture overview
---------------------
MotionBERT is a dual-stream spatiotemporal transformer introduced in:

    Wentao Zhu et al. "MotionBERT: A Unified Perspective on Learning Human
    Motion Representations." ICCV 2023.

The encoder produces a sequence representation of shape ``(B, T, J, dim_rep)``.
We mean-pool over the temporal and joint dimensions to obtain a
``(B, dim_rep)`` embedding used by the classification head.

Checkpoint loading
------------------
Pre-trained backbone weights are available at::

    CARE-PD/assets/Pretrained_checkpoints/motionbert/motionbert.bin

The checkpoint is a raw state dict without a wrapping key.

Shape convention
----------------
Input ``x``:   ``(B, J, F=3, T)``
Output logits: ``(B, n_classes)``
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

from motionbench.classifiers.base import Classifier

logger = logging.getLogger(__name__)

__all__ = ["MotionBERTClassifier"]

# ---------------------------------------------------------------------------
# Backbone sub-modules (adapted from CARE-PD/model/motionbert/)
# ---------------------------------------------------------------------------


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in trunc_normal_.",
            stacklevel=2,
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def _trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class _DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output


class _MLP(nn.Module):
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


class _Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., st_mode='vanilla'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.mode = st_mode
        if self.mode == 'parallel':
            self.ts_attn = nn.Linear(dim * 2, dim * 2)
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, seqlen: int = 1):
        B, N, C = x.shape
        if self.mode == 'spatial':
            q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        elif self.mode == 'temporal':
            q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        elif self.mode == 'vanilla':
            q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        elif self.mode == 'parallel':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        elif self.mode == 'stage_st':
            # Spatial then temporal alternating attention
            q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        elif self.mode == 'stage_ts':
            q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        else:
            raise NotImplementedError(self.mode)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, st_mode='vanilla'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                               qk_scale=qk_scale, attn_drop=attn_drop,
                               proj_drop=drop, st_mode=st_mode)
        self.drop_path = _DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                        act_layer=act_layer, drop=drop)

    def forward(self, x, seqlen: int = 1):
        x = x + self.drop_path(self.attn(self.norm1(x), seqlen))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class _DSTformerBackbone(nn.Module):
    """Dual-Stream Spatiotemporal Transformer backbone.

    Adapted from CARE-PD/model/motionbert/DSTformer.py.
    Forward returns ``(B, F, J, dim_rep)`` when ``return_rep=True``.
    """

    def __init__(
        self,
        dim_in: int = 3,
        dim_out: int = 3,
        dim_feat: int = 512,
        dim_rep: int = 512,
        depth: int = 5,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        num_joints: int = 17,
        maxlen: int = 243,
        qkv_bias: bool = True,
        qk_scale=None,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        norm_layer=nn.LayerNorm,
        att_fuse: bool = True,
    ) -> None:
        super().__init__()
        self.dim_out = dim_out
        self.dim_feat = dim_feat
        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks_st = nn.ModuleList([
            _Block(dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio,
                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                   drop=drop_rate, attn_drop=attn_drop_rate,
                   drop_path=dpr[i], norm_layer=norm_layer, st_mode="stage_st")
            for i in range(depth)
        ])
        self.blocks_ts = nn.ModuleList([
            _Block(dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio,
                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                   drop=drop_rate, attn_drop=attn_drop_rate,
                   drop_path=dpr[i], norm_layer=norm_layer, st_mode="stage_ts")
            for i in range(depth)
        ])
        self.norm = norm_layer(dim_feat)
        if dim_rep:
            self.pre_logits = nn.Sequential(collections.OrderedDict([
                ('fc', nn.Linear(dim_feat, dim_rep)),
                ('act', nn.Tanh()),
            ]))
        else:
            self.pre_logits = nn.Identity()
        self.head = nn.Linear(dim_rep, dim_out) if dim_out > 0 else nn.Identity()
        self.temp_embed = nn.Parameter(torch.zeros(1, maxlen, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        _trunc_normal_(self.temp_embed, std=.02)
        _trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
        self.att_fuse = att_fuse
        if self.att_fuse:
            self.ts_attn = nn.ModuleList([nn.Linear(dim_feat * 2, 2) for _ in range(depth)])
            for i in range(depth):
                self.ts_attn[i].weight.data.fill_(0)
                self.ts_attn[i].bias.data.fill_(0.5)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: Tensor, return_rep: bool = True) -> Tensor:
        """Forward pass.

        Args:
            x: ``(B, F, J, dim_in)`` input tensor.
            return_rep: If ``True``, return ``(B, F, J, dim_rep)`` embedding.

        Returns:
            Sequence representation or pose prediction.
        """
        B, F, J, C = x.shape
        x = x.reshape(-1, J, C)
        BF = x.shape[0]
        x = self.joints_embed(x)
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
                alpha = att(alpha).softmax(dim=-1)
                x = x_st * alpha[:, :, 0:1] + x_ts * alpha[:, :, 1:2]
            else:
                x = (x_st + x_ts) * 0.5
        x = self.norm(x)
        x = x.reshape(B, F, J, -1)
        x = self.pre_logits(x)
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
        n_classes: Number of output logit dimensions (default 4).
        dim_feat: Feature dimension of the transformer (default 512).
        dim_rep: Representation dimension after pre-logit projection
            (default 512).
        depth: Number of dual-stream transformer layers (default 5).
        num_heads: Number of attention heads (default 8).
        mlp_ratio: MLP hidden-to-input ratio (default 2).
        maxlen: Maximum sequence length for temporal positional embedding
            (default 243).
        num_joints: Number of skeletal joints (default 17).
        merge_joints: If ``True``, mean-pool over joints before the
            classifier head (head input dim = ``dim_rep``).  If ``False``
            (default, matches CARE-PD training), keep joints separate and
            flatten (head input dim = ``num_joints * dim_rep``).
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path, None] = None,
        n_classes: int = 4,
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
            drop_path_rate=0.,
        )
        head_dim = dim_rep if merge_joints else dim_rep * num_joints
        self.cls_head = nn.Linear(head_dim, n_classes)

        if self._checkpoint_path is not None:
            if str(self._checkpoint_path).endswith(".pth.tr"):
                self._load_care_pd_checkpoint(self._checkpoint_path)
                logger.info("MotionBERT: loaded CARE-PD fine-tuned checkpoint.")
            else:
                matched, discarded = self._load_checkpoint(
                    self._checkpoint_path,
                    self.backbone,
                    ckpt_key=None,
                    strict=False,
                )
                logger.info(
                    "MotionBERT: loaded %d layers, discarded %d.",
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
        rep = self.backbone(x, return_rep=True)  # (B, T, J, dim_rep)
        rep = rep.mean(dim=1)  # mean over T: (B, J, dim_rep)
        if self._merge_joints:
            rep = rep.mean(dim=1)  # mean over J: (B, dim_rep)
        else:
            rep = rep.flatten(1)  # flatten J: (B, J * dim_rep)
        return self.cls_head(rep)
