"""motionbench.classifiers.ported_care_pd.potr — POTR encoder.

Port of ``CARE-PD/model/potr/`` adapted for the motionbench
``(B, J, F, T)`` input convention.

Architecture overview
---------------------
POTR (Pose Transformers) is a sequence-to-sequence transformer model for
human motion prediction, introduced in:

    Angel Martínez-González et al. "Pose Transformers (POTR): Human Motion
    Prediction with Non-Autoregressive Transformers." ICCV Workshop 2021.

In CARE-PD, POTR is used as an **encoder-only** backbone.  The encoder
stack processes flattened per-frame joint coordinates ``(T, B, J*3)`` via
a GCN spatial embedding and produces a temporal sequence
``(T, B, model_dim)`` from which a mean-pooled ``(B, model_dim)``
embedding is derived for classification.

Checkpoint loading
------------------
Pre-trained backbone weights are available at::

    CARE-PD/assets/Pretrained_checkpoints/potr/
        pre-trained_NTU_ckpt_epoch_199_enc_80_dec_20.pt

The checkpoint was trained with ``source_seq_len=80`` on the NTU RGB+D
dataset.  We use ``source_seq_len=81`` to match motionbench's ``T=81``;
the positional encoding parameter (``_encoder_pos_encodings``) is
recomputed and the checkpoint is loaded with ``strict=False``.

Shape convention
----------------
Input ``x``:   ``(B, J, F=3, T)``
Output logits: ``(B, n_classes)``
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401
from torch import Tensor
from torch.nn.parameter import Parameter

from motionbench.classifiers.base import Classifier

logger = logging.getLogger(__name__)

__all__ = ["POTRClassifier"]

# ---------------------------------------------------------------------------
# Weight initialisation helpers (from CARE-PD/model/utils.py)
# ---------------------------------------------------------------------------


def _xavier_init_(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            layer.bias.data.zero_()


def _weight_init(module, init_fn_=_xavier_init_):
    for layer in module.children() if hasattr(module, 'children') else [module]:
        init_fn_(layer)
    init_fn_(module)


# ---------------------------------------------------------------------------
# GCN encoder (from CARE-PD/model/potr/PoseGCN.py — SimpleEncoder)
# ---------------------------------------------------------------------------


class _GraphConvolution(nn.Module):
    """Adjacency-weighted graph convolution: σ(A × H × W)."""

    def __init__(self, in_features: int, out_features: int, output_nodes: int = 17) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._output_nodes = output_nodes
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(output_nodes, output_nodes))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: ``(batch, n_nodes, in_features)``
        Returns:
            ``(batch, n_nodes, out_features)``
        """
        support = torch.matmul(x, self.weight)
        return torch.matmul(self.att, support)


class _GC_Block(nn.Module):
    """Residual GCN block."""

    def __init__(self, in_features: int, p_dropout: float, output_nodes: int = 17) -> None:
        super().__init__()
        self.gc1 = _GraphConvolution(in_features, in_features, output_nodes)
        self.bn1 = nn.BatchNorm1d(output_nodes * in_features)
        self.gc2 = _GraphConvolution(in_features, in_features, output_nodes)
        self.bn2 = nn.BatchNorm1d(output_nodes * in_features)
        self.do = nn.Dropout(p_dropout)
        self.act = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.reshape(b, -1)).reshape(b, n, f)
        y = self.act(y)
        y = self.do(y)
        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.reshape(b, -1)).reshape(b, n, f)
        y = self.act(y)
        y = self.do(y)
        return y + x


class _SimpleEncoder(nn.Module):
    """GCN-based per-frame pose encoder.

    Input:  ``(B, T, J * input_features)``  (flattened joints per frame)
    Output: ``(B, T, model_dim)``
    """

    _HIDDEN_DIM = 512

    def __init__(
        self,
        n_nodes: int = 17,
        input_features: int = 3,
        model_dim: int = 128,
        p_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self._input_features = input_features
        self._output_nodes = n_nodes
        self._model_dim = model_dim

        self.gc1 = _GraphConvolution(input_features, self._HIDDEN_DIM, n_nodes)
        self.bn1 = nn.BatchNorm1d(n_nodes * self._HIDDEN_DIM)
        self.gcbs = nn.ModuleList([
            _GC_Block(self._HIDDEN_DIM, p_dropout=p_dropout, output_nodes=n_nodes)
        ])
        self.gc2 = _GraphConvolution(self._HIDDEN_DIM, model_dim, n_nodes)
        self.do = nn.Dropout(p_dropout)
        self.act = nn.Tanh()

        self._back = nn.Sequential(
            nn.Linear(model_dim * n_nodes, model_dim),
            nn.Dropout(p_dropout),
        )
        # Xavier init for back
        nn.init.xavier_uniform_(self._back[0].weight)
        if self._back[0].bias is not None:
            self._back[0].bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: ``(B, T, J * input_features)``
        Returns:
            ``(B, T, model_dim)``
        """
        B, S, D = x.shape
        y = self.gc1(x.reshape(-1, self._output_nodes, self._input_features))
        b, n, f = y.shape
        y = self.bn1(y.reshape(b, -1)).reshape(b, n, f)
        y = self.act(y)
        y = self.do(y)
        for blk in self.gcbs:
            y = blk(y)
        y = self.gc2(y)
        y = self._back(y.reshape(-1, self._model_dim * self._output_nodes))
        return y.reshape(B, S, self._model_dim)


# ---------------------------------------------------------------------------
# Positional encoding (from CARE-PD/model/potr/PositionEncodings.py)
# ---------------------------------------------------------------------------


class _PositionEncodings1D:
    """Sinusoidal 1-D positional encoding."""

    def __init__(
        self,
        num_pos_feats: int = 512,
        temperature: float = 10000.0,
        alpha: float = 1.0,
    ) -> None:
        self._num_pos_feats = num_pos_feats
        self._temperature = temperature
        self._alpha = alpha

    def __call__(self, seq_length: int) -> Tensor:
        pos = np.arange(seq_length)[:, np.newaxis]
        i = np.arange(self._num_pos_feats)[np.newaxis, :]
        angle_rates = 1.0 / np.power(
            self._temperature, (2 * (i // 2)) / float(self._num_pos_feats)
        )
        angle_rads = (self._alpha * pos * angle_rates).astype(np.float32)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return torch.from_numpy(angle_rads[np.newaxis])


# ---------------------------------------------------------------------------
# Transformer encoder (from CARE-PD/model/potr/TransformerEncoder.py)
# ---------------------------------------------------------------------------


class _EncoderLayer(nn.Module):
    """Single Transformer encoder layer with post-normalization."""

    def __init__(
        self,
        model_dim: int = 128,
        num_heads: int = 4,
        dim_ffn: int = 2048,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self._self_attn = nn.MultiheadAttention(model_dim, num_heads, dropout)
        self._relu = nn.ReLU()
        self._dropout_layer = nn.Dropout(dropout)
        self._linear1 = nn.Linear(model_dim, dim_ffn)
        self._linear2 = nn.Linear(dim_ffn, model_dim)
        self._norm1 = nn.LayerNorm(model_dim, eps=1e-5)
        self._norm2 = nn.LayerNorm(model_dim, eps=1e-5)
        nn.init.xavier_uniform_(self._linear1.weight)
        nn.init.xavier_uniform_(self._linear2.weight)
        if self._linear1.bias is not None:
            self._linear1.bias.data.zero_()
        if self._linear2.bias is not None:
            self._linear2.bias.data.zero_()

    def forward(self, source_seq: Tensor, pos_encodings: Tensor) -> tuple[Tensor, Tensor]:
        """Args:
            source_seq: ``(T, B, model_dim)``
            pos_encodings: ``(T, 1, model_dim)``
        Returns:
            output: ``(T, B, model_dim)``
            attn_weights: attention weight tensor
        """
        query = source_seq + pos_encodings
        key = query
        value = source_seq
        attn_output, attn_weights = self._self_attn(query, key, value, need_weights=True)
        norm_attn = self._dropout_layer(attn_output) + source_seq
        norm_attn = self._norm1(norm_attn)
        output = self._linear1(norm_attn)
        output = self._relu(output)
        output = self._dropout_layer(output)
        output = self._linear2(output)
        output = self._dropout_layer(output) + norm_attn
        output = self._norm2(output)
        return output, attn_weights


class _TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 4,
        model_dim: int = 128,
        num_heads: int = 4,
        dim_ffn: int = 2048,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self._encoder_stack = nn.ModuleList([
            _EncoderLayer(model_dim, num_heads, dim_ffn, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, input_sequence: Tensor, pos_encodings: Tensor) -> tuple[Tensor, Tensor]:
        outputs = input_sequence
        attn_weights = None
        for layer in self._encoder_stack:
            outputs, attn_weights = layer(outputs, pos_encodings)
        return outputs, attn_weights


# ---------------------------------------------------------------------------
# POTR backbone
# ---------------------------------------------------------------------------


class _POTRBackbone(nn.Module):
    """POTR encoder backbone.

    Takes ``(B, T, J * 3)`` flattened pose sequences and returns
    ``(T, B, model_dim)`` encoder memory.
    """

    def __init__(
        self,
        pose_dim: int = 51,
        source_seq_length: int = 81,
        model_dim: int = 128,
        num_encoder_layers: int = 4,
        num_heads: int = 4,
        dim_ffn: int = 2048,
        dropout: float = 0.3,
        n_joints: int = 17,
        pos_enc_beta: float = 500.0,
        pos_enc_alpha: float = 10.0,
    ) -> None:
        super().__init__()
        self._source_seq_length = source_seq_length
        self._model_dim = model_dim

        # GCN-based pose embedding (per-frame: B*T, J, 3 → B*T, model_dim)
        self._pose_embedding = _SimpleEncoder(
            n_nodes=n_joints,
            input_features=3,
            model_dim=model_dim,
            p_dropout=dropout,
        )

        self._transformer = _TransformerEncoder(
            num_layers=num_encoder_layers,
            model_dim=model_dim,
            num_heads=num_heads,
            dim_ffn=dim_ffn,
            dropout=dropout,
        )

        pos_encoder = _PositionEncodings1D(
            num_pos_feats=model_dim,
            temperature=pos_enc_beta,
            alpha=pos_enc_alpha,
        )
        encoder_pos_enc = pos_encoder(source_seq_length).reshape(source_seq_length, 1, model_dim)
        self._encoder_pos_encodings = nn.Parameter(encoder_pos_enc, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: ``(B, T, J * 3)`` flattened pose sequence.
        Returns:
            memory: ``(T, B, model_dim)`` encoder output.
        """
        x = self._pose_embedding(x)            # (B, T, model_dim)
        x = x.transpose(0, 1)                  # (T, B, model_dim)
        memory, _ = self._transformer(x, self._encoder_pos_encodings)
        return memory


# ---------------------------------------------------------------------------
# motionbench Classifier wrapper
# ---------------------------------------------------------------------------


class POTRClassifier(Classifier):
    """POTR encoder with a linear classification head.

    Wraps :class:`_POTRBackbone` and exposes the standard motionbench
    classifier interface ``(B, J, F, T) → (B, n_classes)``.

    Args:
        checkpoint_path: Path to the pre-trained backbone checkpoint
            (``potr/pre-trained_NTU_ckpt_epoch_199_enc_80_dec_20.pt``).
            If ``None``, weights are randomly initialised.
        n_classes: Number of output logit dimensions (default 4).
        model_dim: Transformer hidden dimension (default 128).
        num_encoder_layers: Number of transformer encoder layers (default 4).
        num_heads: Number of attention heads (default 4).
        dim_ffn: Feed-forward dimension (default 2048).
        dropout: Dropout probability (default 0.3).
        num_joints: Number of skeletal joints (default 17).
        source_seq_length: Number of input frames (default 81 for motionbench).

    Note:
        The pre-trained checkpoint was trained with ``source_seq_len=80`` on
        NTU RGB+D.  Loading with ``source_seq_len=81`` triggers ``strict=False``
        to handle the positional encoding shape mismatch (80 → 81 frames).
        The positional encodings are fixed sinusoidal (non-learned), so they
        are recomputed deterministically for the new length.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path, None] = None,
        n_classes: int = 4,
        model_dim: int = 128,
        num_encoder_layers: int = 4,
        num_heads: int = 4,
        dim_ffn: int = 2048,
        dropout: float = 0.3,
        num_joints: int = 17,
        source_seq_length: int = 81,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, n_classes=n_classes)

        self.backbone = _POTRBackbone(
            pose_dim=num_joints * 3,
            source_seq_length=source_seq_length,
            model_dim=model_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            dim_ffn=dim_ffn,
            dropout=dropout,
            n_joints=num_joints,
        )
        self.cls_head = nn.Linear(model_dim, n_classes)

        if self._checkpoint_path is not None:
            matched, discarded = self._load_checkpoint(
                self._checkpoint_path,
                self.backbone,
                ckpt_key=None,
                strict=False,
            )
            logger.info(
                "POTR: loaded %d layers, discarded %d "
                "(expected: _encoder_pos_encodings discarded due to "
                "seq_len 80→81 mismatch).",
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
        # (B, J, F=3, T) → (B, T, J, F=3) → (B, T, J*F=51)
        B, J, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, J * F)
        memory = self.backbone(x)           # (T, B, model_dim)
        rep = memory.mean(dim=0)            # (B, model_dim)
        return self.cls_head(rep)
