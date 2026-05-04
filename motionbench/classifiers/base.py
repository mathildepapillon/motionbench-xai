"""motionbench.classifiers.base — Abstract base class for all classifiers.

All classifiers in motionbench inherit from :class:`Classifier`.  A
classifier maps a batch of motion sequences ``(B, J, F, T)`` to a batch
of class logits ``(B, n_classes)``.

Design notes
------------
* Classifiers are :class:`torch.nn.Module` subclasses so they participate
  in the normal PyTorch gradient graph, enabling gradient-based XAI methods
  (IG, DeepLift, GradCAM, LRP) without any adapters.
* The canonical input layout is ``(B, J, F, T)`` — **not** the
  ``(B, T, J, C)`` layout used by many source models.  Subclasses are
  responsible for the permute in their ``forward`` method.
* Checkpoints are loaded in :meth:`_load_checkpoint`; subclasses may
  override to handle custom key mappings.

Shape conventions
-----------------
``B``
    Batch size.
``J``
    Number of skeletal joints (e.g. 17 for H36M-17).
``F``
    Coordinates per joint (e.g. 3 for xyz).
``T``
    Time-steps per clip (e.g. 81 frames at 27 fps = 3 s).
``n_classes``
    Number of output classes.
"""

from __future__ import annotations

import collections
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["Classifier"]


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
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        n_classes: int = 4,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Map a batch of motion sequences to logits.

        Args:
            x: Float32 tensor of shape ``(B, J, F, T)``.

        Returns:
            Float32 tensor of shape ``(B, n_classes)`` — raw logits
            (before softmax / sigmoid).
        """
        ...

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
