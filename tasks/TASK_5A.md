# TASK 5A — Ground-truth attribution metrics

**Phase:** 5 | **Tag:** [mechanical] | **PR title:** `[5A] Ground-truth attribution metrics`

## Worktree setup

```bash
git worktree add ../mbxai-task-5A-gt-metrics -b task/5A-gt-metrics
```

## Files to create

```
motionbench/metrics/ground_truth.py
tests/test_gt_metrics.py
```

## Source mapping

- **Port from:** `CARE-PD/scripts/compute_attribution_quality_metrics.py`
  (EC1, EC2, EC3, EC1_norm, sign_agree, top1, topk_overlap, kendall)

## Spec

All metrics have `requires_oracle = True`. All conform to `BaseMetric.evaluate`.

```python
class EC1Metric(BaseMetric):
    """Mean absolute error vs oracle: mean |φ_m - φ_oracle|."""
    requires_oracle = True

class EC2Metric(BaseMetric):
    """MSE vs oracle: mean (φ_m - φ_oracle)²."""
    requires_oracle = True

class EC3Metric(BaseMetric):
    """1 - Pearson(φ_m, φ_oracle). Range [0, 2]."""
    requires_oracle = True

class TopKRecovery(BaseMetric):
    """Fraction of true top-k players recovered.
    k defaults to ceil(M/2). Uses important_players from LabelFunction if available."""
    requires_oracle = True

class SpearmanRankMetric(BaseMetric):
    """Spearman rank correlation with oracle φ."""
    requires_oracle = True

class KendallRankMetric(BaseMetric):
    """Kendall tau rank correlation with oracle φ."""
    requires_oracle = True

class EfficiencyErrorMetric(BaseMetric):
    """|Σφ - (v(N) - v(∅))| / |v(N) - v(∅)|. Should be < 1e-3 for KernelSHAP."""
    requires_oracle = True
```

### Tests

1. For each metric: verify it raises `ValueError` when oracle=None.
2. `test_perfect_attributions_give_zero_ec1` — φ = oracle.true_shapley → EC1 = 0.
3. `test_zero_attributions_give_ec1_norm_one` — φ = 0 → EC1_norm ≈ 1.
4. `test_topk_recovers_all_important` — φ = oracle Shapley → TopK = 1.0.
5. `test_efficiency_error_kernel_shap` — KernelSHAP with oracle imputer → EfficiencyError < 1e-3. Mark `@pytest.mark.slow`.

## Definition of done

- [ ] Tests pass
- [ ] ruff + mypy pass
- [ ] TASKS.md row 5A: done, notes
- [ ] PR: `[5A] Ground-truth attribution metrics`
