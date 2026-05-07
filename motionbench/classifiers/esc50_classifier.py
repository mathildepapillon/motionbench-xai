"""motionbench.classifiers.esc50_classifier — AST-based ESC-50 classifier wrapper.

Wraps ``bioamla/ast-esc50`` (Audio Spectrogram Transformer fine-tuned on
ESC-50) to accept tensors in the motionbench ``(B, J=128, F=1, T=1024)``
format and return per-class softmax probabilities of shape ``(B, 50)``.

Usage::

    from motionbench.classifiers.esc50_classifier import load_esc50_classifier

    clf = load_esc50_classifier(device="cuda:0")
    # x: (B, 128, 1, 1024) float32 tensor
    probs = clf(x)  # (B, 50)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ESC50ASTClassifier(nn.Module):
    """Wraps ``bioamla/ast-esc50`` for the motionbench (B, J, F, T) format.

    Args:
        model_name: HuggingFace model ID to load (default: ``"bioamla/ast-esc50"``).
        device: Torch device to place the model on.
    """

    def __init__(self, model_name: str = "bioamla/ast-esc50", device: str | torch.device = "cpu") -> None:
        super().__init__()
        from transformers import ASTForAudioClassification
        self._model = ASTForAudioClassification.from_pretrained(
            model_name, ignore_mismatched_sizes=True
        )
        self._model.eval()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: ``(B, 128, 1, 1024)`` float32 tensor — motionbench ``(B, J, F, T)`` format.

        Returns:
            ``(B, 50)`` float32 softmax probability tensor on the same device as ``x``.
        """
        # x: (B, 128, 1, 1024)
        B = x.shape[0]
        # Squeeze F dim: (B, 128, 1024)
        x2 = x.squeeze(2)          # (B, 128, 1024)
        # Permute to (B, 1024, 128) = (B, time_steps, num_mel_bins) as AST expects
        x2 = x2.permute(0, 2, 1)   # (B, 1024, 128)

        out = self._model(input_values=x2, output_hidden_states=False)
        logits = out.logits  # (B, 50)
        return torch.softmax(logits, dim=-1)

    def to(self, *args, **kwargs):
        self._model = self._model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def eval(self):
        self._model.eval()
        return super().eval()

    def train(self, mode: bool = True):
        # Keep model in eval mode for inference wrapper
        self._model.eval()
        return super().train(False)


def load_esc50_classifier(device: str | torch.device = "cpu") -> ESC50ASTClassifier:
    """Load and return the ESC-50 AST classifier.

    Args:
        device: Torch device to place the model on.

    Returns:
        :class:`ESC50ASTClassifier` in eval mode on the requested device.
    """
    device = torch.device(device)
    clf = ESC50ASTClassifier(model_name="bioamla/ast-esc50", device=device)
    clf = clf.to(device)
    clf.eval()
    return clf
