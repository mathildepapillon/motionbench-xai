# TASK 2A — Off-manifold imputers

**Phase:** 2 | **Tag:** [mechanical] | **PR title:** `[2A] Off-manifold imputers`

## Worktree setup

```bash
git worktree add ../mbxai-task-2A-offmanifold -b task/2A-off-manifold-imputers
```

## Files to create

```
motionbench/imputers/off_manifold.py
motionbench/utils/masking.py
tests/test_off_manifold_imputers.py
```

## Source mapping

- **Port from:** `CARE-PD/shap_facade/imputers.py` — `ZeroImputer`, `MeanImputer`, `MarginalImputer`
- **Extract to `masking.py`:** `_coalition_to_element_mask`, `_assert_layout`

## Spec

### `motionbench.imputers.off_manifold`

Implement four imputers, all conforming to `BaseImputer`:

```python
class ZeroImputer(BaseImputer):
    """Fill hidden entries with zeros. fit() is a no-op."""

class MeanImputer(BaseImputer):
    """Fill hidden entries with per-coordinate training mean.
    fit() computes mean from train_data."""

class MarginalDonorImputer(BaseImputer):
    """Fill hidden entries by sampling a random training sequence.
    Implements Aas 2021 'independence' imputer / shap.maskers.Independent."""

class GaussianNoiseImputer(BaseImputer):
    """Fill hidden entries with training mean ± Gaussian noise."""
    def __init__(self, scale: float = 1.0): ...
```

Each is ~30–60 lines. All must:
- Return `(n_samples, J, F, T)` from `impute`.
- Preserve observed entries bit-for-bit.
- Implement `is_on_manifold = False`.

### `motionbench.utils.masking`

```python
def coalition_to_element_mask(z: Tensor, player_set) -> Tensor:
    """(M,) binary → (J, F, T) bool via player_set.coalition_mask."""

def assert_mask_shape(mask: Tensor, J: int, F: int, T: int) -> None:
    """Raise ValueError if mask.shape != (J, F, T)."""
```

### Tests

- Shape: `impute` returns `(n_samples, J, F, T)`.
- Observed preservation: all imputers.
- Mean imputer: after `fit`, `impute` mean of large N samples converges to training mean for hidden coords.
- Marginal: distribution of hidden completions matches training marginal (KS test, `@pytest.mark.slow`).
- `n_samples=1`, `n_samples=100` both work.

## Definition of done

- [ ] Tests pass
- [ ] ruff + mypy pass
- [ ] TASKS.md row 2A: `status: done`, notes
- [ ] PR: `[2A] Off-manifold imputers`
