# TASK 3C — Time-series SHAP variants

**Phase:** 3 | **Tag:** [mechanical] | **PR title:** `[3C] Temporal SHAP method wrappers`

## Worktree setup

```bash
git worktree add ../mbxai-task-3C-temporal-shap -b task/3C-temporal-shap
```

## Files to create

```
motionbench/attribution/timeshap.py
motionbench/attribution/windowshap.py
motionbench/attribution/shats.py
motionbench/attribution/group_segment_shap.py
tests/test_temporal_shap.py
```

## Spec

Each is a thin `BaseAttributor` wrapper. All return `(M,)` by aggregating to player level.

- **`TimeSHAPAttributor`** — wraps `timeshap` library (Bento et al. 2020).
- **`WindowSHAPAttributor`** — wraps `windowshap` library (Nayebi et al. 2023).
- **`ShaTS Attributor`** — wraps `shats` library (López et al.).
- **`GroupSegmentSHAPAttributor`** — port from CARE-PD/model/empirical/group_baselines.py; if no Python implementation exists for the paper version, port from paper pseudocode and add a BACKLOG.md entry.

### Tests

For each: call `attribute` on a tiny RNN/LSTM with a (J=5, F=3, T=16) input; verify output shape `(M,)` and no errors.

## References

- Bento et al. (2020) "TimeSHAP." ICDM.
- Nayebi et al. (2023) "WindowSHAP."
- López et al. — ShaTS paper.

## Definition of done

- [ ] All 4 wrappers pass shape tests (or are documented as BACKLOG if library unavailable)
- [ ] ruff + mypy pass
- [ ] TASKS.md row 3C: done, notes
- [ ] PR: `[3C] Temporal SHAP method wrappers`
