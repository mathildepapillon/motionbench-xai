# TASK 2B — Empirical / classical-conditional imputers

**Phase:** 2 | **Tag:** [verify against literature] | **PR title:** `[2B] Empirical/copula imputers`

## Worktree setup

```bash
git worktree add ../mbxai-task-2B-empirical -b task/2B-empirical-imputers
```

## Files to create

```
motionbench/imputers/empirical.py
tests/test_empirical_imputers.py
```

## Source mapping

- **Port from:** `CARE-PD/model/empirical/imputer.py` (`EmpiricalImputer`)

## Spec

### `KNNConditionalImputer(k=20, distance="euclidean_observed")`

For each query `x_obs`:
1. Find k nearest training sequences using *only observed coordinates*.
2. Sample completions for hidden coordinates from the k neighbors.

Adapts the kNN portion of the CARE-PD `EmpiricalImputer`. Uses Ledoit-Wolf
shrinkage for the distance metric (see source). **Cite Aas 2021 §3.3 Algorithm 2.**

### `EmpiricalConditionalImputer(bandwidth="auto", shrinkage="ledoit_wolf")`

Full Aas et al. (2021) §3.3 empirical conditional:
1. Mahalanobis distance using Ledoit-Wolf shrunk covariance of observed sub-block.
2. Gaussian kernel weights `w_n = exp(-d²/(2σ²))`.
3. η-truncation (keep top K rows with cumulative weight ≥ η = 0.95).
4. Sample from the surviving rows.

**Must cite specific equation numbers from Aas 2021 in docstrings.**

### `VineCopulaImputer` (if pyvinecopulib installs cleanly)

Wrap `pyvinecopulib` for Gaussian-copula and non-parametric conditional sampling.
If install is painful, document the dependency and add `BACKLOG.md` entry.

### Tests

1. Shape and observed-preservation (all imputers).
2. `test_empirical_matches_gaussian_oracle` — on Gaussian data with N_train → large (N=5000), `EmpiricalConditionalImputer` Shapley values converge to `GaussianOracle` Shapley values within MC noise. Mark `@pytest.mark.slow`.

## References

- **Read and cite:** Aas, Jullum & Løland (2021) §3.3, Algorithm 2 and Table 1.
- `shapr` R package defaults: `fixed_sigma=0.1`, `eta=0.95`.

## Definition of done

- [ ] Tests pass
- [ ] ruff + mypy pass
- [ ] TASKS.md row 2B: `status: done`, notes
- [ ] PR: `[2B] Empirical/copula imputers`
