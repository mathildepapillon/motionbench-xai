# TASK 5C — Stability and sanity-check metrics

**Phase:** 5 | **Tag:** [mechanical] | **PR title:** `[5C] Stability and sanity-check metrics`

## Worktree setup

```bash
git worktree add ../mbxai-task-5C-stability -b task/5C-stability-sanity
```

## Files to create

```
motionbench/metrics/stability.py
motionbench/metrics/sanity_checks.py
tests/test_stability_sanity.py
```

## Spec

Wrap Quantus metrics:

### `stability.py`
- `MaxSensitivityMetric` — `quantus.MaxSensitivity`
- `ContinuityMetric` — `quantus.Continuity`
- `LipschitzEstimateMetric` — `quantus.RelativeInputStability`

### `sanity_checks.py`
- `ModelParameterRandomisationMetric` — `quantus.ModelParameterRandomisation` (Adebayo et al. 2018)
- `RandomLogitMetric` — `quantus.RandomLogit`

All have `requires_oracle = False`, `requires_imputer = False`.

### Tests

- Shape: `evaluate` returns `dict[str, float]`.
- Sanity: `ModelParameterRandomisation` returns lower correlation for randomized model than original.

## References

- Adebayo et al. (2018) "Sanity Checks for Saliency Maps." NeurIPS.
- Hedström et al. (2023) "Quantus."

## Definition of done

- [ ] Tests pass
- [ ] ruff + mypy pass
- [ ] TASKS.md row 5C: done, notes
- [ ] PR: `[5C] Stability and sanity-check metrics`
