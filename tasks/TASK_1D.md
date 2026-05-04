# TASK 1D — Label function library

**Phase:** 1 | **Tag:** [mechanical] | **PR title:** `[1D] Label function library`

## Worktree setup

```bash
git worktree add ../mbxai-task-1D-labels -b task/1D-labels
```

## Files to create

```
motionbench/data/synthetic/label_functions.py
tests/test_label_functions.py
```

## Source mapping

- **Extract from:** `CARE-PD/synthetic/gaussian_motion.py` — `nonlinear_olsen_score`, `spatial_olsen_score`, `_olsen_term`, `GaussianMotionBenchmark.setup_label_fn`, `GaussianMotionBenchmark.canonical_label_fn`

## Spec

### `LabelFunction` ABC

```python
class LabelFunction(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x: (N, J, F, T) → (N,) int64 class labels (ternary 0/1/2 by default)."""

    @abstractmethod
    def important_players(self, player_set) -> set[int]:
        """Return indices of players that drive the label by construction.
        Used by TopKRecovery metric as ground-truth top-k set."""
```

### Concrete implementations

1. **`Linear(weights)`** — `score = Σ_i w_i · x_mean_i`. Linear baseline.
2. **`OlsenInteraction(K, seed)`** — port `nonlinear_olsen_score`. Works for temporal players (K windows).
3. **`SpatialOlsen(signal_joints, seed)`** — port `spatial_olsen_score`. Works for spatial players.
4. **`LocalizedTemporal(window_idx, fn)`** — label depends only on one temporal window. `fn: (np.ndarray) → np.ndarray`.
5. **`LocalizedSpatial(joint_idx, fn)`** — label depends only on one joint.
6. **`LocalizedSpatiotemporal(joint_idx, window_idx, fn)`** — both.

All label functions must binarize scores into ternary classes {0,1,2} via 33/67 percentile cutoffs by default (configurable via `n_classes` and `percentiles` args).

### Tests

For each label function:
- Verify `important_players(players)` returns correct subset.
- Verify gradient of `score` w.r.t. non-important players is < ε (1e-3) on average across N=1000 samples from `GaussianMotionBenchmark`.

Use `@pytest.mark.slow` for gradient checks.

## Definition of done

- [ ] Tests pass
- [ ] ruff + mypy pass
- [ ] TASKS.md row 1D: `status: done`, notes
- [ ] PR: `[1D] Label function library`
