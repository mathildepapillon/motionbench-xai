# TASK 1C — Skeleton-structured and gait-periodic synthetics

**Phase:** 1 | **Tag:** [needs thinking] | **Depends on:** 1A | **PR title:** `[1C] Skeleton-structured and gait-periodic synthetics`

## Worktree setup

```bash
git worktree add ../mbxai-task-1C-skeleton -b task/1C-skeleton-gait
```

## Files to create

```
motionbench/data/synthetic/skeleton_structured.py
motionbench/data/synthetic/gait_periodic.py
tests/test_structured_synthetics.py
```

## Source mapping

- **Port from:** `CARE-PD/synthetic/diagnostic_motion.py` (Fourier gait, diagnostic joints)
- **Use from Task 1A:** `GaussianOracle`, `SigmaJointsFactory.skeleton_adjacency`, `SigmaTimeFactory.gait_periodic`

## Spec

### 1. `motionbench.data.synthetic.skeleton_structured`

```python
class SkeletonStructuredDataset(GroundTruthDataset):
    """Gaussian motion with skeleton-adjacency Σ_joint and AR(1) Σ_time.

    Post-processing: rescale per-joint coordinate variances to match an
    empirical bone-length distribution from CARE-PD.
    Load the cached fixture from data/skeleton_covariance/bmclab_h36m17/.
    """
```

Uses `GaussianMotionBenchmark` with `SigmaJointsFactory.skeleton_adjacency` and
`SigmaTimeFactory.ar1`. Exposes `GaussianOracle` at `.oracle`.

### 2. `motionbench.data.synthetic.gait_periodic`

```python
class GaitPeriodicDataset(GroundTruthDataset):
    """Gaussian motion with gait-periodic Toeplitz Σ_time.

    Stride period drawn from a configurable distribution. Label functions
    exploit periodicity (e.g. signal lives in the second stride).
    """
    def __init__(self, period_mean: float, period_std: float, n_harmonics: int = 3, ...): ...
```

Uses `SigmaTimeFactory.gait_periodic`. Label function: configurable, default is
`LocalizedTemporal` from Task 1D pointing at the second stride.

### 3. Tests

1. `test_skeleton_adjacency_structure` — verify Sigma_joints satisfies adjacency decay property: `Sigma[i,j] ≈ decay ** d(i,j)` where `d` is kinematic tree distance.
2. `test_gait_periodic_autocorrelation` — verify sample autocorrelation is approximately periodic at configured period.
3. `test_end_to_end_shapley` — instantiate, sample, compute true Shapley via oracle, verify efficiency axiom.

## Definition of done

- [ ] Tests pass
- [ ] ruff + mypy pass
- [ ] TASKS.md row 1C: `status: done`, notes
- [ ] PR: `[1C] Skeleton-structured and gait-periodic synthetics`
