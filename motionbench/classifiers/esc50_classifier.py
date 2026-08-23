"""motionbench.classifiers.esc50_classifier — Per-fold AST ESC-50 classifiers.

Loads the fold-disciplined AST fine-tunes behind the paper's ESC-50 tables:
one Audio Spectrogram Transformer per data fold, fine-tuned from
``MIT/ast-finetuned-audioset-10-10-0.4593`` on that fold's ESC-50 training
split only (see ``checkpoints/README.md`` for digests and the training
recipe in the paper appendix).  Accepts tensors in the motionbench
``(B, J=128, F=1, T=1024)`` mel format and returns per-class softmax
probabilities of shape ``(B, 50)``.

Label convention: head index ``k`` is the ESC-50 canonical target id ``k``
(the models are trained directly on the cache labels; this is **not** the
alphabetical class ordering some third-party ESC-50 checkpoints use).

Usage::

    from motionbench.classifiers.esc50_classifier import load_esc50_classifier

    clf = load_esc50_classifier(fold=1, device="cuda:0")
    # x: (B, 128, 1, 1024) float32 tensor
    probs = clf(x)  # (B, 50)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Candidate checkpoint locations, tried in order (relative to the repo root).
CHECKPOINT_CANDIDATES = (
    "motionbench/classifiers/checkpoints/real/esc50_ast_fold{fold}.pt",
    "checkpoints/esc50_clf/ast_fold{fold}.pt",
)

#: ASTConfig does not accept these checkpoint-metadata keys.
_NON_CONFIG_KEYS = ("transformers_version", "torch_dtype", "dtype", "architectures")


def _resolve_checkpoint(fold: int, checkpoint: str | Path | None) -> Path:
    if checkpoint is not None:
        p = Path(checkpoint)
        if p.exists():
            return p
        raise FileNotFoundError(f"ESC-50 AST checkpoint not found: {p}")
    tried = []
    for cand in CHECKPOINT_CANDIDATES:
        p = _REPO_ROOT / cand.format(fold=fold)
        if p.exists():
            return p
        tried.append(str(p))
    raise FileNotFoundError(
        "No ESC-50 AST checkpoint for fold "
        f"{fold}; tried:\n  " + "\n  ".join(tried) + "\n"
        "Download the esc50 checkpoint archive (scripts/download_checkpoints.sh"
        " esc50) or retrain (see REPRODUCIBILITY.md)."
    )


class ESC50ASTClassifier(nn.Module):
    """Per-fold AST fine-tune wrapped for the motionbench (B, J, F, T) format.

    Args:
        fold: ESC-50 data fold (1-3) whose classifier to load.
        checkpoint: Optional explicit checkpoint path; overrides the default
            per-fold lookup in :data:`CHECKPOINT_CANDIDATES`.
    """

    def __init__(self, fold: int = 1, checkpoint: str | Path | None = None) -> None:
        """Initialise the AST model for ``fold`` and load its fine-tuned checkpoint."""
        super().__init__()
        from transformers import ASTConfig, ASTForAudioClassification

        path = _resolve_checkpoint(fold, checkpoint)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        cfg = ASTConfig(
            **{k: v for k, v in ckpt["arch"]["config"].items() if k not in _NON_CONFIG_KEYS}
        )
        self._model = ASTForAudioClassification(cfg)
        self._model.load_state_dict(ckpt["state_dict"], strict=True)
        self._model.eval()
        self.fold = fold
        self.checkpoint_path = str(path)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: ``(B, 128, 1, 1024)`` float32 tensor — motionbench ``(B, J, F, T)`` format.

        Returns:
            ``(B, 50)`` float32 softmax probability tensor on the same device as ``x``.
        """
        # (B, 128, 1, 1024) -> (B, 1024, 128) = (B, time_steps, num_mel_bins)
        x2 = x.squeeze(2).permute(0, 2, 1)
        out = self._model(input_values=x2, output_hidden_states=False)
        return torch.softmax(out.logits, dim=-1)

    def to(self, *args: object, **kwargs: object) -> ESC50ASTClassifier:
        """Move the wrapped module to a device; returns self."""
        self._model = self._model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def eval(self) -> ESC50ASTClassifier:
        """Set the wrapped module to eval mode; returns self."""
        self._model.eval()
        return super().eval()

    def train(self, mode: bool = True) -> ESC50ASTClassifier:
        """Inference wrapper: the wrapped model always stays in eval mode."""
        self._model.eval()
        return super().train(False)


def load_esc50_classifier(
    fold: int = 1,
    device: str | torch.device = "cpu",
    checkpoint: str | Path | None = None,
) -> ESC50ASTClassifier:
    """Load the fold's ESC-50 AST classifier.

    Args:
        fold: ESC-50 data fold (1-3).
        device: Torch device to place the model on.
        checkpoint: Optional explicit checkpoint path.

    Returns:
        :class:`ESC50ASTClassifier` in eval mode on the requested device.
    """
    device = torch.device(device)
    clf = ESC50ASTClassifier(fold=fold, checkpoint=checkpoint)
    clf = clf.to(device)
    clf.eval()
    return clf
