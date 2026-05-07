"""motionbench.classifiers.base — Abstract base class for all classifiers.

All classifiers in motionbench inherit from :class:`Classifier`.  A
classifier maps a batch of raw motion sequences ``(B, J, F, T)`` to a
batch of class logits ``(B, n_classes)``.

Design notes
------------
* Classifiers are :class:`torch.nn.Module` subclasses so they participate
  in the normal PyTorch gradient graph, enabling gradient-based XAI methods
  (IG, DeepLift, GradCAM, LRP) without any adapters.
* The canonical **raw** input layout is ``(B, J, F, T)`` where ``F``
  equals :attr:`Classifier.input_feature_dim` — either 3 (3-D world-to-
  camera) or 2 (2-D image-projected) depending on the classifier.
* Each subclass overrides :meth:`_preprocess` to apply its model-specific
  normalization (crop_scale, screen-norm, z-score, etc.) **inside**
  ``forward``.  This keeps XAI attributions in raw coordinate space while
  still feeding the backbone its correctly processed input.
* Checkpoints are loaded in :meth:`_load_checkpoint`; subclasses may
  override to handle custom key mappings.

Shape conventions
-----------------
``B``
    Batch size.
``J``
    Number of skeletal joints (e.g. 17 for H36M-17).
``F``
    Raw coordinates per joint (3 for 3-D, 2 for 2-D).
``T``
    Time-steps per clip (e.g. 81 frames at 27 fps = 3 s).
``n_classes``
    Number of output classes.
"""

from __future__ import annotations

import collections
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = ["Classifier", "crop_scale_and_conf"]


# ---------------------------------------------------------------------------
# Shared preprocessing utilities
# ---------------------------------------------------------------------------


def crop_scale_and_conf(x: Tensor) -> Tensor:
    """Bounding-box normalise x,y + simulate confidence score (in-place safe).

    Mirrors CARE-PD's ``MotionBERTPreprocessor`` / ``MotionAGFormerPreprocessor``
    pipeline:

    1. Detect padded frames: frames where *all* coordinates are 0 (z == 0 in
       the raw 3-D world-to-camera representation).
    2. Compute the bounding box over valid (x, y) positions across all valid
       frames and joints.
    3. Normalise x,y into ``[-1, 1]`` (crop_scale, ratio=1 for deterministic
       inference).
    4. Replace the z channel with a binary confidence: 1.0 for real frames,
       0.0 for padded frames.

    Args:
        x: ``(B, T, J, 3)`` float32 tensor.  Padded frames should have
           all coordinates equal to zero.

    Returns:
        Preprocessed tensor of the same shape ``(B, T, J, 3)``.
    """
    B, T, J, _ = x.shape
    result = x.clone()

    for b in range(B):
        sample = x[b]  # (T, J, 3)
        # Valid frames: at least one non-zero z coordinate in that frame
        valid_frame = (sample[..., 2] != 0).any(dim=-1)  # (T,)
        valid_xy = sample[valid_frame][:, :, :2].reshape(-1, 2)  # (N, 2)

        if valid_xy.shape[0] < 4:
            result[b] = 0.0
            continue

        xmin, xmax = valid_xy[:, 0].min(), valid_xy[:, 0].max()
        ymin, ymax = valid_xy[:, 1].min(), valid_xy[:, 1].max()
        scale = torch.maximum(xmax - xmin, ymax - ymin)
        if scale == 0:
            result[b] = 0.0
            continue

        xs = (xmin + xmax - scale) / 2
        ys = (ymin + ymax - scale) / 2

        norm_xy = (sample[:, :, :2] - torch.stack([xs, ys]).to(device=x.device)) / scale
        norm_xy = (norm_xy - 0.5) * 2
        norm_xy = norm_xy.clamp(-1.0, 1.0)

        # Confidence channel: 1 for real frames, 0 for padded
        conf = valid_frame.to(device=x.device).float().unsqueeze(-1).unsqueeze(-1).expand(T, J, 1)

        result[b] = torch.cat([norm_xy, conf], dim=-1)

    return result


class Classifier(nn.Module, ABC):
    """Abstract base class for all motionbench classifiers.

    Subclasses must implement :meth:`forward`.  The optional
    ``checkpoint_path`` argument triggers weight loading from a stored
    checkpoint.

    Args:
        checkpoint_path: Path to a ``.pt`` / ``.bin`` / ``.pth`` checkpoint
            file, or ``None`` for a randomly-initialised model (useful for
            architecture-only tests).
        n_classes: Number of output logit dimensions.

    Class attributes
    ----------------
    input_feature_dim : int
        The number of raw coordinate channels expected by this classifier.
        Most 3-D classifiers use ``3`` (x, y, z world-to-camera); 2-D
        classifiers such as PoseFormerV2 use ``2`` (projected image coords).
        Set this as a class-level attribute in each subclass.

    Design: per-classifier preprocessing
    -------------------------------------
    Each concrete classifier is responsible for transforming the *raw*
    coordinate input into whatever the backbone expects.  This is done by
    overriding :meth:`_preprocess`, which is called at the top of
    :meth:`forward`.  Because ``_preprocess`` is a pure ``torch``
    computation, gradients flow through it naturally, enabling attribution
    methods to produce attributions w.r.t. the raw input space.

    The canonical raw inputs per classifier family:

    * **MotionBERT / MotionAGFormer** — ``(B, J, 3, T)`` 3-D world-to-camera
      coordinates; ``_preprocess`` applies ``crop_scale`` and replaces the
      z-channel with a confidence score (1 = real frame, 0 = padded).
    * **PoseFormerV2** — ``(B, J, 2, T)`` 2-D image-projected pixel
      coordinates; ``_preprocess`` applies screen normalization
      ``(xy / 1100) * 2 − 1``.
    * **POTR** — ``(B, J, 3, T)`` 3-D world-to-camera; ``_preprocess``
      centres around the root joint and applies stored z-score statistics.
    """

    #: Number of raw coordinate channels for this classifier (override in subclass).
    input_feature_dim: int = 3

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        n_classes: int = 4,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None

    # ------------------------------------------------------------------
    # Per-classifier preprocessing (override in subclasses)
    # ------------------------------------------------------------------

    def _preprocess(self, x: Tensor) -> Tensor:
        """Transform raw input coordinates into backbone-ready tensors.

        The default implementation is the identity; subclasses override
        this to apply model-specific normalisation (crop_scale, screen
        normalisation, z-score, etc.).

        Args:
            x: Raw float32 tensor of shape ``(B, J, F, T)`` where F equals
               ``self.input_feature_dim``.

        Returns:
            Preprocessed tensor suitable for passing to the backbone.
            Shape and channel semantics may differ from the raw input.
        """
        return x

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Map a batch of motion sequences to logits.

        Implementations should call ``self._preprocess(x)`` as their first
        step so that XAI methods receive attributions w.r.t. raw coordinates.

        Args:
            x: Raw float32 tensor of shape ``(B, J, F, T)`` where F equals
               ``self.input_feature_dim``.

        Returns:
            Float32 tensor of shape ``(B, n_classes)`` — raw logits
            (before softmax / sigmoid).
        """
        ...

    def predict_proba(self, x: Tensor, class_idx: int | None = None) -> Tensor:
        """Return class probabilities (softmax over logits).

        Args:
            x: Float32 tensor of shape ``(B, J, F, T)``.
            class_idx: If given, return only the probability for this class
                as a ``(B,)`` tensor.  Otherwise return ``(B, n_classes)``.

        Returns:
            Float32 probability tensor.
        """
        logits = self.forward(x)
        proba = torch.softmax(logits, dim=-1)
        if class_idx is not None:
            return proba[:, class_idx]
        return proba

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _load_checkpoint(
        self,
        path: str | Path,
        model: nn.Module,
        ckpt_key: str | None = None,
        strict: bool = False,
    ) -> tuple[list[str], list[str]]:
        """Load *encoder* weights from a checkpoint file.

        Compatible with DataParallel ``module.`` prefixes and optional
        top-level dict keys (e.g. ``{"model_pos": state_dict}``).

        Args:
            path: Filesystem path to the checkpoint.
            model: The :class:`~torch.nn.Module` whose weights are updated
                in-place.
            ckpt_key: If given, index into the raw checkpoint dict with this
                key before extracting the state dict.
            strict: Passed through to :meth:`~torch.nn.Module.load_state_dict`.
                Defaults to ``False`` so that partial matches succeed (the
                classification head is always freshly initialised).

        Returns:
            ``(matched_layers, discarded_layers)`` — two lists of key names
            useful for diagnostic logging.

        Raises:
            RuntimeError: If no layers match (likely a wrong checkpoint file
                or key mapping error).
        """
        raw = torch.load(
            path,
            map_location=lambda storage, _: storage,
            weights_only=False,
        )

        if ckpt_key is not None and isinstance(raw, dict) and ckpt_key in raw:
            state_dict = raw[ckpt_key]
        elif isinstance(raw, dict) and "state_dict" in raw:
            state_dict = raw["state_dict"]
        else:
            state_dict = raw

        model_dict = model.state_dict()
        model_first_key = next(iter(model_dict))
        new_state_dict: dict[str, Tensor] = collections.OrderedDict()
        matched, discarded = [], []

        for k, v in state_dict.items():
            # Strip DataParallel prefix if the model itself is not wrapped.
            if "module." not in model_first_key and k.startswith("module."):
                k = k[7:]
            if k in model_dict and model_dict[k].shape == v.shape:
                new_state_dict[k] = v
                matched.append(k)
            else:
                discarded.append(k)

        if not matched:
            raise RuntimeError(
                f"[Classifier._load_checkpoint] No layers matched. "
                f"First checkpoint key: {next(iter(state_dict))}. "
                f"First model key: {model_first_key}."
            )

        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict, strict=strict)
        return matched, discarded

    def _load_care_pd_checkpoint(self, path: str | Path) -> None:
        """Load a CARE-PD fine-tuned ``.pth.tr`` checkpoint into ``self``.

        CARE-PD Hypertune checkpoints have the structure::

            {
                "epoch": int,
                "lr": float,
                "optimizer": ...,
                "model": {
                    "backbone.<layer>.*": Tensor,   # encoder weights
                    "head.fc_layers.0.weight": Tensor,  # classifier head
                    "head.fc_layers.0.bias":   Tensor,
                },
            }

        Some checkpoints are wrapped in DataParallel (``module.`` prefix).
        This method:

        1. Extracts ``ckpt["model"]``.
        2. Strips any ``module.`` DataParallel prefix.
        3. Remaps ``head.fc_layers.0.*`` → ``cls_head.*``.
        4. Calls ``self.load_state_dict(..., strict=False)``.

        Args:
            path: Path to the ``.pth.tr`` checkpoint file.
        """
        raw = torch.load(
            path,
            map_location=lambda storage, _: storage,
            weights_only=False,
        )
        state_dict: dict[str, Tensor] = raw["model"]

        remapped: dict[str, Tensor] = {}
        for k, v in state_dict.items():
            # Strip DataParallel prefix
            if k.startswith("module."):
                k = k[7:]
            # Remap CARE-PD head key to the motionbench cls_head attribute
            if k.startswith("head.fc_layers.0."):
                k = "cls_head." + k[len("head.fc_layers.0."):]
            # Remap MotionAGFormer layer-scale names (checkpoint uses layer_scale_N,
            # our port uses the shorter ls1/ls2 attribute names)
            k = k.replace(".layer_scale_1", ".ls1").replace(".layer_scale_2", ".ls2")
            remapped[k] = v

        result = self.load_state_dict(remapped, strict=False)
        matched = [k for k in remapped if k not in result.missing_keys]
        logger.info(
            "_load_care_pd_checkpoint: loaded %d tensors, missing=%s, unexpected=%s",
            len(matched),
            result.missing_keys,
            result.unexpected_keys,
        )

