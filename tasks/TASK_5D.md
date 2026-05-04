# TASK 5D — Cross-protocol ranking agreement

**Phase:** 5 | **Tag:** [needs thinking] | **Depends on:** 5A, 5B, 5C | **PR title:** `[5D] Cross-protocol ranking agreement`

## Worktree setup

```bash
git worktree add ../mbxai-task-5D-ranking -b task/5D-ranking-agreement
```

## Files to create

```
motionbench/metrics/ranking_agreement.py
tests/test_ranking_agreement.py
```

## Spec

```python
class RankingAgreementMetric:
    """Cross-protocol Spearman correlation matrix (meta-metric).

    Given a results table indexed by (method, metric), compute pairwise
    Spearman rank correlations between method-rankings under different metrics.
    Output: M×M correlation matrix (M = number of metrics).
    Add bootstrap CIs (95%, n_bootstrap=1000).

    This is Table 3 of the paper.
    """

    def compute(
        self,
        results: dict[str, dict[str, float]],
        # results[method_name][metric_name] = score
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> dict:
        """Return:
        {"correlation_matrix": (M, M) ndarray,
         "ci_lower": (M, M) ndarray,
         "ci_upper": (M, M) ndarray,
         "metric_names": list[str]}
        """
```

### Tests

1. `test_identical_rankings_give_correlation_one` — two metrics that rank methods identically → Spearman = 1.0.
2. `test_reversed_rankings_give_minus_one` — reversed ranking → -1.0.
3. `test_bootstrap_ci_width` — CI width decreases with more methods.

## Definition of done

- [ ] Tests pass
- [ ] ruff + mypy pass
- [ ] TASKS.md row 5D: done, notes
- [ ] PR: `[5D] Cross-protocol ranking agreement`
