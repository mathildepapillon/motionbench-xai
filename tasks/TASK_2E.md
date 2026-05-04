# TASK 2E — KernelSHAP attributor wrapping shap library

**Phase:** 2 | **Tag:** [needs thinking] | **Depends on:** 2A | **PR title:** `[2E] KernelSHAP with pluggable imputer`

## Worktree setup

```bash
git worktree add ../mbxai-task-2E-kernelshap -b task/2E-kernelshap
```

## Files to create

```
motionbench/attribution/kernel_shap.py
tests/test_kernel_shap.py
```

## Spec

This is the **linchpin task**. Read `CARE-PD/shap_facade/explainers.py` for context,
but do NOT port it — implement a proper `shap.KernelExplainer`-backed version.

### `KernelShapAttributor(BaseAttributor)`

```python
class KernelShapAttributor(BaseAttributor):
    """KernelSHAP attributor with a pluggable BaseImputer masker.

    Wraps shap.KernelExplainer (or shap.Explainer with algorithm="permutation")
    with a custom shap.maskers.Masker subclass that delegates to BaseImputer.impute.

    Player aggregation: Shapley values are computed at the player level directly,
    not at the coordinate level. The masker receives a (M,) coalition vector and
    expands it via players.coalition_mask before calling imputer.impute.
    """

    requires_imputer = True

    def __init__(
        self,
        classifier: Callable,
        imputer: BaseImputer,
        n_samples: int = 2**11,
        seed: int = 42,
        algorithm: str = "kernel",  # "kernel" or "permutation"
    ): ...

    def attribute(self, x: Tensor, players: PlayerSet, target: int = 0) -> Tensor:
        """Return (M,) Shapley values. x: (J, F, T) single sample."""
```

### Implementation guidance

1. Create `_MotionBenchMasker(shap.maskers.Masker)`:
   - `__call__(mask, x)` receives a single-row coalition vector and the data.
   - Calls `imputer.impute(x_obs, element_mask, n_samples=n_completion_samples)`.
   - Returns the mean of completions (as `shap.KernelExplainer` expects a single "masked" input).
   - Read `shap.maskers.Independent.__call__` source as the reference pattern.

2. Benchmark `algorithm="kernel"` vs `algorithm="permutation"` on a J=17, T=81 sequence.
   Pick the faster one; note the choice in the docstring.

3. Player aggregation is handled by the masker (coalitions operate at player level),
   so no post-hoc aggregation step is needed.

### Tests

1. `test_kernelshap_matches_oracle` — with `GaussianOracle` as the imputer, `KernelShapAttributor` Shapley values match `oracle.true_shapley` within MC tolerance (3σ for K=4). Mark `@pytest.mark.slow`.
2. `test_kernelshap_shape` — output `(M,)` for K=4, K=8 player sets.
3. `test_kernelshap_efficiency` — Σφ ≈ v(N) − v(∅).
4. `test_kernelshap_vs_scratch` — wall-clock: wrapping `shap` is at least as fast as the CARE-PD `_solve_shapley_wls` reference. Mark `@pytest.mark.slow`.

## References

- Read `shap.maskers.Independent` source (< 50 lines) before writing the masker.
- Lundberg & Lee (2017) "SHAP." NeurIPS.
- Aas et al. (2021) §2.3 — WLS Shapley solve.

## Definition of done

- [ ] Tests pass (non-slow)
- [ ] Oracle-matching test passes `@pytest.mark.slow`
- [ ] ruff + mypy pass
- [ ] TASKS.md row 2E: `status: done`, notes
- [ ] PR: `[2E] KernelSHAP with pluggable imputer`
