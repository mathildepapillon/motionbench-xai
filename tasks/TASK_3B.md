# TASK 3B — LRP via Zennit

**Phase:** 3 | **Tag:** [mechanical] | **PR title:** `[3B] LRP via Zennit`

## Worktree setup

```bash
git worktree add ../mbxai-task-3B-lrp -b task/3B-lrp
```

## Files to create

```
motionbench/attribution/lrp.py
tests/test_lrp.py
```

## Spec

```python
class LRPAttributor(BaseAttributor):
    """Layer-wise Relevance Propagation via Zennit.

    Supported rules: "epsilon", "gamma", "alpha_beta".
    Reference rule choices from gait XAI literature:
    - Slijepcevic et al. (2022) "Explainability of Vision-Based Autonomous Driving Systems."
    - Horst et al. (2019) "Explainability of Deep Neural Networks for MoCap Gait Analysis."
    Cite specific rule choices in docstrings.
    """
    requires_gradient = True

    def __init__(self, classifier, rule="epsilon", epsilon=1e-6, gamma=0.25, **kwargs): ...
```

Aggregate coordinate-level relevance to player level via `players.aggregate`.

### Tests

- Instantiate with a tiny linear model, call `attribute`, verify `(M,)` output.
- Verify conservation: Σ relevance ≈ model output. Mark `@pytest.mark.slow`.

## References

- Zennit: https://github.com/chr5tphr/zennit
- Bach et al. (2015) "On pixel-wise explanations for non-linear classifier decisions."
- Slijepcevic et al. (2022) — cite rule choices.

## Definition of done

- [ ] Tests pass
- [ ] ruff + mypy pass
- [ ] TASKS.md row 3B: done, notes
- [ ] PR: `[3B] LRP via Zennit`
