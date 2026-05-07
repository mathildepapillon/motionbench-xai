"""motionbench.classifiers.ported_care_pd.motionagformer — MotionAGFormer encoder.

Port of ``CARE-PD/model/motionagformer/`` adapted for the motionbench
``(B, J, F, T)`` input convention.

Architecture overview
---------------------
MotionAGFormer extends MotionBERT's dual-stream design by integrating
adaptive graph convolutional networks, introduced in:

    Soroush Mehraban et al. "MotionAGFormer: Enhancing 3D Human Pose
    Estimation with a Transformer-GCNFormer Network." WACV 2024.

The encoder produces a per-frame per-joint representation of shape
``(B, T, J, dim_rep)``.  We mean-pool over T and J to obtain a
``(B, dim_rep)`` embedding used by the classification head.

Checkpoint loading
------------------
Pre-trained backbone weights are available at::

    CARE-PD/assets/Pretrained_checkpoints/motionagformer/
        motionagformer-s-h36m.pth.tr

loaded under the key ``"model"``.

Shape convention
----------------
Input ``x``:   ``(B, J, F=3, T)``
Output logits: ``(B, n_classes)``
"""

from __future__ import annotations

import collections
import logging
import math
from pathlib import Path
from typing import Any, Union

import torch
import torch.nn as nn
from torch import Tensor

try:
    from timm.layers import DropPath  # type: ignore[import-untyped]
except ImportError as exc:
    raise ImportError(
        "timm is required for MotionAGFormer. Install with: pip install timm"
    ) from exc

from motionbench.classifiers.base import Classifier, crop_scale_and_conf

logger = logging.getLogger(__name__)

__all__ = ["MotionAGFormerClassifier"]

# ---------------------------------------------------------------------------
# H36M-17 spatial connectivity (from CARE-PD/model/motionagformer/modules/graph.py)
# ---------------------------------------------------------------------------

_H36M_CONNECTIONS: dict[int, list[int]] = {
    10: [9], 9: [8, 10], 8: [7, 9], 14: [15, 8], 15: [16, 14],
    11: [12, 8], 12: [13, 11], 7: [0, 8], 0: [1, 7], 1: [2, 0],
    2: [3, 1], 4: [5, 0], 5: [6, 4], 16: [15], 13: [12], 3: [2], 6: [5],
}


# ---------------------------------------------------------------------------
# Sub-modules (inlined from CARE-PD/model/motionagformer/modules/)
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _Attention(nn.Module):
    """Spatial or temporal self-attention operating on (B, T, J, C) tensors."""

    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0., mode='spatial'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.mode = mode
        self.qkv = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, J, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, T, J, 3, self.num_heads, C // self.num_heads)
               .permute(3, 0, 4, 1, 2, 5))  # (3, B, H, T, J, C//H)
        q, k, v = qkv.unbind(0)
        if self.mode == 'spatial':
            # attend over joints within each frame
            attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, J, J)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).permute(0, 2, 3, 1, 4).reshape(B, T, J, C)
        else:
            # attend over frames for each joint
            qt = q.transpose(2, 3)  # (B, H, J, T, C//H)
            kt = k.transpose(2, 3)
            vt = v.transpose(2, 3)
            attn = (qt @ kt.transpose(-2, -1)) * self.scale  # (B, H, J, T, T)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            # permute (B, H, J, T, C//H) → (B, T, J, H, C//H) then reshape to (B, T, J, C)
            x = (attn @ vt).permute(0, 3, 2, 1, 4).reshape(B, T, J, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _GCN(nn.Module):
    """Adaptive graph convolutional network for spatial or temporal graphs."""

    def __init__(self, dim_in, dim_out, num_nodes, neighbour_num=4,
                 mode='spatial', use_temporal_similarity=True,
                 temporal_connection_len=1):
        super().__init__()
        assert mode in ('spatial', 'temporal')
        self.mode = mode
        self.neighbour_num = neighbour_num
        self.num_nodes = num_nodes
        self.use_temporal_similarity = use_temporal_similarity
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.relu = nn.ReLU()
        self.U = nn.Linear(dim_in, dim_out)
        self.V = nn.Linear(dim_in, dim_out)
        self.batch_norm = nn.BatchNorm1d(num_nodes)
        self._init_weights()

        if mode == 'spatial':
            self.adj = self._init_spatial_adj()
        elif not use_temporal_similarity:
            self.adj = self._init_temporal_adj(temporal_connection_len)
        else:
            self.adj = None  # computed on-the-fly

    def _init_weights(self):
        self.U.weight.data.normal_(0, math.sqrt(2. / self.dim_in))
        self.V.weight.data.normal_(0, math.sqrt(2. / self.dim_in))
        self.batch_norm.weight.data.fill_(1)
        self.batch_norm.bias.data.zero_()

    def _init_spatial_adj(self) -> Tensor:
        adj = torch.zeros(self.num_nodes, self.num_nodes)
        for i, neighbours in _H36M_CONNECTIONS.items():
            for j in neighbours:
                adj[i, j] = 1.0
        return adj

    def _init_temporal_adj(self, connection_len: int) -> Tensor:
        adj = torch.zeros(self.num_nodes, self.num_nodes)
        for i in range(self.num_nodes):
            for j in range(connection_len + 1):
                if i + j < self.num_nodes:
                    adj[i, i + j] = 1.0
        return adj

    @staticmethod
    def _normalize_digraph(adj: Tensor) -> Tensor:
        b, n, _ = adj.shape
        node_degrees = adj.detach().sum(dim=-1)
        deg_inv_sqrt = node_degrees.pow(-0.5)
        I = torch.eye(n, device=adj.device).unsqueeze(0)
        norm_deg = I * deg_inv_sqrt.unsqueeze(-1)
        return torch.bmm(torch.bmm(norm_deg, adj), norm_deg)

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: ``(B, T, J, C)``
        Returns:
            ``(B, T, J, dim_out)``
        """
        b, t, j, c = x.shape
        dev = x.device
        if self.mode == 'temporal':
            x = x.transpose(1, 2).reshape(-1, t, c)  # (B*J, T, C)
            if self.use_temporal_similarity:
                sim = x @ x.transpose(1, 2)
                thr = sim.topk(k=self.neighbour_num, dim=-1, largest=True)[0][..., -1].unsqueeze(-1)
                adj = (sim >= thr).float()
            else:
                adj = self.adj.to(dev).repeat(b * j, 1, 1)
        else:
            x = x.reshape(-1, j, c)  # (B*T, J, C)
            adj = self.adj.to(dev).repeat(b * t, 1, 1)

        norm_adj = self._normalize_digraph(adj)
        aggregate = norm_adj @ self.V(x)
        if self.dim_in == self.dim_out:
            x = self.relu(x + self.batch_norm(aggregate + self.U(x)))
        else:
            x = self.relu(self.batch_norm(aggregate + self.U(x)))

        if self.mode == 'spatial':
            return x.reshape(b, t, j, self.dim_out)
        else:
            return x.reshape(b, j, t, self.dim_out).transpose(1, 2)


# ---------------------------------------------------------------------------
# AGFormer block (wraps a single mixer)
# ---------------------------------------------------------------------------


class _AGFormerBlock(nn.Module):
    """Single spatial or temporal AGFormer block (attention or graph mixer)."""

    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0.,
                 drop_path=0., num_heads=8, qkv_bias=False, qk_scale=None,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 mode='spatial', mixer_type='attention',
                 use_temporal_similarity=True, temporal_connection_len=1,
                 neighbour_num=4, n_frames=81):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if mixer_type == 'attention':
            self.mixer = _Attention(dim, dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, mode)
        elif mixer_type == 'graph':
            n_nodes = 17 if mode == 'spatial' else n_frames
            self.mixer = _GCN(dim, dim, num_nodes=n_nodes, neighbour_num=neighbour_num,
                              mode=mode, use_temporal_similarity=use_temporal_similarity,
                              temporal_connection_len=temporal_connection_len)
        else:
            raise ValueError(f"Unknown mixer_type: {mixer_type}")
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = _MLP(dim, mlp_hidden, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.ls1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
            self.ls2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        if self.use_layer_scale:
            x = x + self.drop_path(self.ls1 * self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.ls2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# MotionAGFormer dual-stream block
# ---------------------------------------------------------------------------


class _MotionAGFormerBlock(nn.Module):
    """Dual-stream AGFormer block (attention + graph branches)."""

    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0.,
                 drop_path=0., num_heads=8, use_layer_scale=True, qkv_bias=False,
                 qk_scale=None, layer_scale_init_value=1e-5, use_adaptive_fusion=True,
                 use_temporal_similarity=True, temporal_connection_len=1,
                 neighbour_num=4, n_frames=81):
        super().__init__()
        kw = dict(dim=dim, mlp_ratio=mlp_ratio, act_layer=act_layer,
                  attn_drop=attn_drop, drop=drop, drop_path=drop_path,
                  num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  use_layer_scale=use_layer_scale,
                  layer_scale_init_value=layer_scale_init_value,
                  use_temporal_similarity=use_temporal_similarity,
                  temporal_connection_len=temporal_connection_len,
                  neighbour_num=neighbour_num, n_frames=n_frames)
        self.att_spatial = _AGFormerBlock(mode='spatial', mixer_type='attention', **kw)
        self.att_temporal = _AGFormerBlock(mode='temporal', mixer_type='attention', **kw)
        self.graph_spatial = _AGFormerBlock(mode='spatial', mixer_type='graph', **kw)
        self.graph_temporal = _AGFormerBlock(mode='temporal', mixer_type='graph', **kw)

        self.use_adaptive_fusion = use_adaptive_fusion
        if use_adaptive_fusion:
            self.fusion = nn.Linear(dim * 2, 2)
            self.fusion.weight.data.fill_(0)
            self.fusion.bias.data.fill_(0.5)

    def forward(self, x: Tensor) -> Tensor:
        x_attn = self.att_temporal(self.att_spatial(x))
        x_graph = self.graph_temporal(self.graph_spatial(x))
        if self.use_adaptive_fusion:
            alpha = self.fusion(torch.cat((x_attn, x_graph), dim=-1)).softmax(dim=-1)
            return x_attn * alpha[..., 0:1] + x_graph * alpha[..., 1:2]
        return (x_attn + x_graph) * 0.5


# ---------------------------------------------------------------------------
# MotionAGFormer backbone
# ---------------------------------------------------------------------------


class _MotionAGFormerBackbone(nn.Module):
    """MotionAGFormer backbone (encoder only).

    Returns ``(B, T, J, dim_rep)`` when ``return_rep=True``.
    """

    def __init__(
        self,
        n_layers: int = 26,
        dim_in: int = 3,
        dim_feat: int = 64,
        dim_rep: int = 512,
        dim_out: int = 3,
        mlp_ratio: int = 4,
        act_layer=nn.GELU,
        attn_drop: float = 0.,
        drop: float = 0.,
        drop_path: float = 0.,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        use_adaptive_fusion: bool = True,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qkv_scale=None,
        use_temporal_similarity: bool = True,
        temporal_connection_len: int = 1,
        neighbour_num: int = 2,
        num_joints: int = 17,
        n_frames: int = 81,
    ) -> None:
        super().__init__()
        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.norm = nn.LayerNorm(dim_feat)

        block_kw = dict(dim=dim_feat, mlp_ratio=mlp_ratio, act_layer=act_layer,
                        attn_drop=attn_drop, drop=drop, drop_path=drop_path,
                        num_heads=num_heads, use_layer_scale=use_layer_scale,
                        layer_scale_init_value=layer_scale_init_value,
                        qkv_bias=qkv_bias, qk_scale=qkv_scale,
                        use_adaptive_fusion=use_adaptive_fusion,
                        use_temporal_similarity=use_temporal_similarity,
                        temporal_connection_len=temporal_connection_len,
                        neighbour_num=neighbour_num, n_frames=n_frames)
        self.layers = nn.Sequential(*[
            _MotionAGFormerBlock(**block_kw) for _ in range(n_layers)
        ])

        self.rep_logit = nn.Sequential(collections.OrderedDict([
            ('fc', nn.Linear(dim_feat, dim_rep)),
            ('act', nn.Tanh()),
        ]))
        self.head = nn.Linear(dim_rep, dim_out)

    def forward(self, x: Tensor, return_rep: bool = True) -> Tensor:
        """Args:
            x: ``(B, T, J, dim_in)``
            return_rep: if True, return ``(B, T, J, dim_rep)``
        """
        x = self.joints_embed(x)
        x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.rep_logit(x)
        if return_rep:
            return x
        return self.head(x)


# ---------------------------------------------------------------------------
# motionbench Classifier wrapper
# ---------------------------------------------------------------------------


class MotionAGFormerClassifier(Classifier):
    """MotionAGFormer encoder with a linear classification head.

    Wraps :class:`_MotionAGFormerBackbone` and exposes the standard
    motionbench classifier interface ``(B, J, F, T) → (B, n_classes)``.

    Args:
        checkpoint_path: Path to a checkpoint.  Pass a ``.pth.tr`` path to
            load a CARE-PD fine-tuned checkpoint.
        n_classes: Number of output logit dimensions (default 4).
        n_layers: Number of MotionAGFormer blocks (default 26).
        dim_feat: Per-joint feature dimension (default 64).
        dim_rep: Output representation dimension (default 512).
        num_heads: Number of attention heads (default 8).
        num_joints: Number of skeletal joints (default 17).
        n_frames: Number of input frames (default 81 for motionbench).
        merge_joints: If ``True``, mean-pool over joints before the head
            (head input dim = ``dim_rep``).  If ``False`` (default, matches
            CARE-PD training), flatten joints (head input dim =
            ``num_joints * dim_rep``).
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path, None] = None,
        n_classes: int = 4,
        n_layers: int = 26,
        dim_feat: int = 64,
        dim_rep: int = 512,
        num_heads: int = 8,
        num_joints: int = 17,
        n_frames: int = 81,
        merge_joints: bool = False,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, n_classes=n_classes)
        self._merge_joints = merge_joints

        self.backbone = _MotionAGFormerBackbone(
            n_layers=n_layers,
            dim_in=3,
            dim_feat=dim_feat,
            dim_rep=dim_rep,
            dim_out=3,
            mlp_ratio=4,
            num_heads=num_heads,
            use_layer_scale=True,
            layer_scale_init_value=1e-5,
            use_adaptive_fusion=True,
            use_temporal_similarity=True,
            temporal_connection_len=1,
            neighbour_num=2,
            num_joints=num_joints,
            n_frames=n_frames,
        )
        head_dim = dim_rep if merge_joints else dim_rep * num_joints
        self.cls_head = nn.Linear(head_dim, n_classes)

        if self._checkpoint_path is not None:
            if str(self._checkpoint_path).endswith(".pth.tr"):
                self._load_care_pd_checkpoint(self._checkpoint_path)
                logger.info("MotionAGFormer: loaded CARE-PD fine-tuned checkpoint.")
            else:
                raw = torch.load(
                    self._checkpoint_path,
                    map_location=lambda storage, _: storage,
                    weights_only=True,
                )
                state = raw if isinstance(raw, dict) and "model" not in raw else raw.get("model", raw)
                # CARE-PD model-only .pt files: remap head.fc_layers.0 → cls_head
                if any("head.fc_layers" in k for k in state):
                    remapped: dict[str, Any] = {}
                    for k, v in state.items():
                        if k.startswith("module."):
                            k = k[7:]
                        if k.startswith("head.fc_layers.0."):
                            k = "cls_head." + k[len("head.fc_layers.0."):]
                        k = k.replace(".layer_scale_1", ".ls1").replace(".layer_scale_2", ".ls2")
                        remapped[k] = v
                    result = self.load_state_dict(remapped, strict=False)
                    logger.info(
                        "MotionAGFormer: loaded CARE-PD model-only state dict "
                        "(%d tensors, missing=%s, unexpected=%s)",
                        len(remapped), result.missing_keys, result.unexpected_keys,
                    )
                else:
                    # Slim .pt checkpoint: keys already prefixed with 'backbone.'
                    # (and 'cls_head.'). Load directly into self.
                    result2 = self.load_state_dict(state, strict=False)
                    n_loaded = len(state) - len(result2.unexpected_keys)
                    logger.info(
                        "MotionAGFormer: loaded slim checkpoint; loaded=%d, "
                        "missing=%s, unexpected=%s",
                        n_loaded, result2.missing_keys, result2.unexpected_keys,
                    )

    def _preprocess(self, x: Tensor) -> Tensor:
        """Apply CARE-PD preprocessing: crop_scale + confidence channel.

        Mirrors MotionAGFormerPreprocessor in CARE-PD/data/dataloaders.py.

        Args:
            x: ``(B, J, 3, T)`` raw 3-D world-to-camera coordinates.

        Returns:
            ``(B, T, J, 3)`` preprocessed tensor.
        """
        x_btjf = x.permute(0, 3, 1, 2).contiguous()  # (B, T, J, 3)
        return crop_scale_and_conf(x_btjf)

    def forward(self, x: Tensor) -> Tensor:
        """Map a batch of raw 3D pose sequences to class logits.

        Args:
            x: Raw float32 tensor of shape ``(B, J, F=3, T)`` — 3-D
               world-to-camera coordinates.  Padded frames should be all-zero.

        Returns:
            Float32 tensor of shape ``(B, n_classes)`` — raw logits.
        """
        x_proc = self._preprocess(x)              # (B, T, J, 3)
        rep = self.backbone(x_proc, return_rep=True)  # (B, T, J, dim_rep)
        rep = rep.mean(dim=1)  # mean over T: (B, J, dim_rep)
        if self._merge_joints:
            rep = rep.mean(dim=1)  # mean over J: (B, dim_rep)
        else:
            rep = rep.flatten(1)  # flatten J: (B, J * dim_rep)
        return self.cls_head(rep)
