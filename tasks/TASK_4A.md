# TASK 4A — Synthetic classifiers

**Phase:** 4 | **Tag:** [mechanical] | **PR title:** `[4A] Synthetic classifiers`

## Worktree setup

```bash
git worktree add ../mbxai-task-4A-synthetic-clf -b task/4A-synthetic-classifiers
```

## Files to create

```
motionbench/classifiers/synthetic_mlp.py
motionbench/classifiers/synthetic_cnn.py
motionbench/classifiers/synthetic_transformer.py
motionbench/classifiers/base.py
tests/test_synthetic_classifiers.py
```

## Spec

### `motionbench.classifiers.base`

```python
class Classifier(ABC):
    """Uniform interface for all classifiers."""

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """(B, J, F, T) → (B, n_classes) logits."""

    def predict_proba(self, x: Tensor, class_idx: int = 0) -> Tensor:
        """(B, J, F, T) → (B,) probability for class_idx."""
        return torch.softmax(self.forward(x), dim=-1)[:, class_idx]

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
```

### `SyntheticMLPClassifier`

Extract and generalise `SyntheticMLPClassifier` from `CARE-PD/synthetic/gaussian_motion.py`.
Temporal mode (K windows, grand means) and spatial mode (J joints, grand means).

### `SyntheticCNNClassifier`

Small 1D CNN: 3 × Conv1d(32, kernel=5) + AdaptiveAvgPool → Linear.
Input: `(B, J*F, T)` reshape, output: `(B, n_classes)`.

### `SyntheticTransformerClassifier`

4-layer Transformer encoder: d_model=64, nhead=4, dim_ff=128.
Input: `(T, B, J*F)` format, output: `(B, n_classes)` via mean pooling over T.

All three must come with pre-trained weights for the canonical synthetic Gaussian
benchmark (J=17, F=3, T=81, K=4). Train with `scripts/train_synthetic_clf.py` (create
this script) and commit the weights (they are small, < 500 KB each).

### Tests

- Forward pass shapes: `(B, n_classes)`.
- `predict_proba` shape: `(B,)`.
- Gradient flows (requires_grad on input, backward pass doesn't error).

## Definition of done

- [ ] Tests pass
- [ ] Pre-trained weights committed for canonical synthetic benchmark
- [ ] ruff + mypy pass
- [ ] TASKS.md row 4A: done, notes
- [ ] PR: `[4A] Synthetic classifiers`
