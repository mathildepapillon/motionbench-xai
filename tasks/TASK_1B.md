# TASK 1B — Burr / Gaussian-copula generator + CopulaOracle

**Phase:** 1 | **Tag:** [needs thinking] | **PR title:** `[1B] Burr/copula generator + oracle`

## Worktree setup

```bash
git worktree add ../mbxai-task-1B-burr -b task/1B-burr-motion
```

## Files to create

```
motionbench/data/synthetic/burr_motion.py
motionbench/oracles/copula_oracle.py
tests/test_copula_oracle.py
```

## Source mapping

- **Port from:** `CARE-PD/synthetic/burr_motion.py` (`BurrMotionBenchmark`)

## Spec

### 1. `motionbench.data.synthetic.burr_motion`

Port `BurrMotionBenchmark`. Add a `Marginal` ABC with a strategy pattern:

```python
class Marginal(ABC):
    @abstractmethod
    def cdf(self, x: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def quantile(self, u: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def pdf(self, x: np.ndarray) -> np.ndarray: ...

class BurrXII(Marginal):
    def __init__(self, c: float = 2.0, k: float = 2.0): ...

class StudentT(Marginal):
    def __init__(self, df: float): ...

class MixtureOfGaussians(Marginal):
    def __init__(self, weights, means, scales): ...

class SkewNormal(Marginal):
    def __init__(self, alpha: float): ...
```

`BurrMotionBenchmark.__init__` accepts `marginal: Marginal = BurrXII(2, 2)`.

The benchmark conforms to `GroundTruthDataset` (same as Task 1A). Oracle is `CopulaOracle`.

### 2. `motionbench.oracles.copula_oracle`

```python
class CopulaOracle(Oracle):
    """Gaussian copula oracle with pluggable marginals.

    Algorithm (Aas et al. 2021, copula section):
    1. Transform observed x_obs → latent z_obs via:
          z_obs[i] = Φ⁻¹(F_i(x_obs[i]))     (inverse Gaussian CDF of marginal CDF)
    2. Compute Gaussian conditional: z_hid ~ N(μ_{hid|obs}, Σ_{hid|obs})
       using the same Kronecker formula as GaussianOracle.
    3. Back-transform: x_hid = F_i⁻¹(Φ(z_hid[i]))   (marginal quantile of Gaussian CDF)
    """
```

`CopulaOracle` also satisfies `BaseImputer` (same pattern as `GaussianOracle`).

### 3. Tests (`tests/test_copula_oracle.py`)

**Required tests:**
1. `test_marginal_round_trip` — for each `Marginal` subclass: `F⁻¹(F(x)) ≈ x` to 1e-6.
2. `test_gaussian_marginals_match_gaussian_oracle` — with `Marginal = GaussianMarginal` (or `StudentT(df=∞)`), `CopulaOracle` and `GaussianOracle` give Shapley values within 1e-5.
3. `test_efficiency_axiom_burr` — Σφ ≈ v(N) − v(∅) on Burr data.
4. `test_conditional_sample_preserves_observed` — standard contract test.

**Mark compute-heavy tests as `@pytest.mark.slow`.**

## References

- Joe (2014) *Dependence Modeling with Copulas* — copula transform identity.
- Aas et al. (2021), copula-based conditional expectation section. **Cite equations.**
- `pyvinecopulib` source for Gaussian copula sampling reference.

## Definition of done

- [ ] Tests pass locally
- [ ] ruff + mypy pass
- [ ] TASKS.md row 1B: `status: done`, notes
- [ ] PR: `[1B] Burr/copula generator + oracle`
