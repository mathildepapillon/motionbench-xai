# TASK 5B — Fidelity metrics with on/off-manifold variants

**Phase:** 5 | **Tag:** [needs thinking] | **Depends on:** 2A, 2B | **PR title:** `[5B] Fidelity metrics, on/off-manifold variants`

## Worktree setup

```bash
git worktree add ../mbxai-task-5B-fidelity -b task/5B-fidelity
```

## Files to create

```
motionbench/metrics/fidelity.py
tests/test_fidelity.py
```

## Spec

Use **Quantus** as the underlying engine wherever possible. **Read Quantus source** to confirm `perturb_func` signatures before writing integration code.

For each Quantus metric, expose two variants:
- `_OffManifold(imputer=ZeroImputer())` — Quantus default behaviour.
- `_OnManifold(imputer=VAEACImputer.load(...))` — pass our `BaseImputer` as `perturb_func`.

```python
class FaithfulnessCorrelationMetric(BaseMetric):
    requires_imputer = True

class MonotonicityCorrelationMetric(BaseMetric):
    requires_imputer = True

class PixelFlippingMetric(BaseMetric):
    """Deletion / PGU equivalent. Adapted for time-series (flip temporal windows)."""
    requires_imputer = True

class SelectivityMetric(BaseMetric):
    """Insertion / PGI."""
    requires_imputer = True
```

### Quantus integration point

Quantus's `perturb_func` is a callable `(arr, **kwargs) -> arr`. The integration:
```python
def _make_perturb_func(imputer: BaseImputer):
    def perturb_func(arr, indices, indexed_axes, **kwargs):
        # arr: numpy (J, F, T); indices: flattened indices of features to perturb
        # Build mask, call imputer.impute, return single completion
        ...
    return perturb_func
```
**Read `quantus.helpers.perturb_func` and `quantus.metrics.faithfulness.pixel_flipping` source
to confirm the exact expected signature.**

### Tests

1. `test_off_manifold_matches_quantus_default` — our `_OffManifold` variant gives the same number as raw `quantus.FaithfulnessCorrelation` on the same explanation.
2. `test_on_manifold_uses_imputer` — verify imputer is called (mock imputer with a call counter).

## CRITICAL WARNING

If Quantus's API does not cleanly accept arbitrary imputers, do NOT invent a workaround.
Instead: set `status: blocked` in TASKS.md, add a BACKLOG entry, and document exactly
what breaks. The plan explicitly flags this as Risk R3.

## References

- Hedström et al. (2023) "Quantus." JMLR 24(34).
- Quantus source: https://github.com/understandable-machine-intelligence-lab/Quantus

## Definition of done

- [ ] Quantus integration spike confirmed (read source first)
- [ ] Tests pass (or task blocked with clear documentation)
- [ ] ruff + mypy pass
- [ ] TASKS.md row 5B: done or blocked, notes
- [ ] PR: `[5B] Fidelity metrics, on/off-manifold variants`
