# TASK 1A — Port Gaussian generator + GaussianOracle

**Phase:** 1 | **Tag:** [needs thinking] | **PR title:** `[1A] Gaussian motion + closed-form oracle`

## Worktree setup

```bash
git worktree add ../mbxai-task-1A-gaussian -b task/1A-gaussian-motion
```

## Files to create

```
motionbench/data/synthetic/gaussian_motion.py
motionbench/oracles/gaussian_oracle.py
motionbench/utils/coalitions.py
tests/test_gaussian_oracle.py
tests/test_coalitions.py
```

## Source mapping (read SOURCE_MAP.md for details)

- **Port from:** `CARE-PD/synthetic/gaussian_motion.py` (`GaussianMotionBenchmark` class)
- **Oracle extract:** `conditional_sample`, `conditional_sample_spatial`, `conditional_sample_spatiotemporal`, `compute_v_true_all_coalitions`, `compute_true_shapley*` → `GaussianOracle`
- **Utilities to `coalitions.py`:** `_ar1_cov`, `_equicorr`, `_shapley_kernel_weight`, `_enumerate_temporal_coalitions`, `_sample_kernelshap_coalitions`, `_solve_shapley_wls`
- **Do NOT port:** `SyntheticMLPClassifier` — that goes in Task 4A.
- **Do NOT port:** label functions — those go in Task 1D.

## Spec

### 1. `motionbench/data/synthetic/gaussian_motion.py`

Port `GaussianMotionBenchmark` from the source. The class:
- Conforms to `GroundTruthDataset` protocol (add `__getitem__`, `__len__`, `shape`, `metadata`, `oracle` properties).
- Returns `(x, y)` from `__getitem__`: x is `(J, F, T)` float32, y is scalar int64 label.
- Stores `GaussianOracle` at `.oracle`.

Add the following `Sigma_joints` variants (currently the source only has `equicorrelated`):

```python
class SigmaJointsFactory:
    @staticmethod
    def equicorrelated(J, rho) -> np.ndarray: ...

    @staticmethod
    def skeleton_adjacency(J, skeleton="h36m_17", decay=0.7) -> np.ndarray:
        """Off-diagonal = decay ** graph_distance in the kinematic tree.
        Load skeleton adjacency from CARE-PD/data/skeleton_covariance/bmclab_h36m17/."""

    @staticmethod
    def block_diagonal(J, left_right=True) -> np.ndarray:
        """Block-diagonal: left/right body halves are independent."""

    @staticmethod
    def data_driven(care_pd_subset: np.ndarray) -> np.ndarray:
        """Empirical Sigma_joints from CARE-PD subset. Cache to disk."""
```

Add the following `Sigma_time` variants:

```python
class SigmaTimeFactory:
    @staticmethod
    def ar1(T, alpha) -> np.ndarray: ...
    @staticmethod
    def ar_p(T, alphas: list[float]) -> np.ndarray: ...
    @staticmethod
    def gait_periodic(T, period, n_harmonics=3) -> np.ndarray:
        """Sum-of-cosines kernel: k(t,t') = Σ_h cos(2πh|t-t'|/period)."""
```

### 2. `motionbench/oracles/gaussian_oracle.py`

Create `GaussianOracle(Oracle)` that wraps the benchmark's conditional sampling:

```python
class GaussianOracle(Oracle):
    def conditional_sample(self, x_obs, mask, n, seed=None) -> Tensor:
        """Closed-form Gaussian conditional. mask: (J,F,T) bool."""
        # mask must be compatible with player structure;
        # support temporal, spatial, and spatiotemporal masks.

    def true_shapley(self, x, classifier, players, n_mc=1000, seed=None) -> Tensor:
        """Enumerate all 2^M coalitions if M <= 12; raise NotImplementedError otherwise."""
```

`GaussianOracle` must also satisfy the `BaseImputer` interface (implement `fit` and `impute`):
- `fit(train_data)` — no-op (oracle needs no training), returns self.
- `impute(x_obs, mask, n_samples, seed)` — delegates to `conditional_sample`.

### 3. `motionbench/utils/coalitions.py`

Extract the coalition utilities (see SOURCE_MAP §4 and §16). Expose as module-level functions with full docstrings. Include:
- `ar1_cov`, `equicorr` (covariance helpers)
- `shapley_kernel_weight`, `enumerate_coalitions`, `sample_kernelshap_coalitions`
- `solve_shapley_wls`

### 4. Tests (`tests/test_gaussian_oracle.py`)

**Required tests:**
1. `test_conditional_sample_vs_rejection_sampling` — for a small (J=3, T=8, K=2) case, verify `conditional_sample` matches large-N rejection sampling within 3σ tolerance.
2. `test_true_shapley_efficiency` — verify `Σφ_i ≈ v(N) − v(∅)` to within 1e-4.
3. `test_diagonal_sigma_marginal_equals_conditional` — with `Sigma_joints = I`, `Sigma_time = I` (independent), verify the oracle's Shapley values equal those from `MarginalImputer` Shapley values (up to MC noise).
4. `test_coalition_shape_variants` — temporal mask, spatial mask, spatiotemporal mask all produce correct `(n, J, F, T)` output.
5. `test_oracle_satisfies_imputer_interface` — `GaussianOracle` passes `BaseImputer` contract (call `impute`, check shapes, check observed entries preserved).

**Mark all as `@pytest.mark.slow` except shapes test.**

## References

- Aas, Jullum & Løland (2021) "Explaining individual predictions when features are dependent" — §3 conditional Gaussian formula. **Cite equation numbers in docstrings.**
- `shap` library's `MaskedModel` for KernelSHAP interface.
- Source code: `CARE-PD/synthetic/gaussian_motion.py` (read it; do not copy verbatim — refactor for the new ABC).

## Definition of done

- [ ] All required tests pass locally (`pytest tests/test_gaussian_oracle.py -m "not slow"`)
- [ ] `ruff check .` passes
- [ ] `mypy motionbench/data/synthetic/gaussian_motion.py motionbench/oracles/gaussian_oracle.py` passes
- [ ] `TASKS.md` row 1A updated: `status: done`, 3-line notes
- [ ] PR opened: `[1A] Gaussian motion + closed-form oracle`
