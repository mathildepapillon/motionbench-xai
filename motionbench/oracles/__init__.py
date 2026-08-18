"""motionbench.oracles — Ground-truth conditional samplers and exact targets.

- :class:`~motionbench.oracles.gaussian_oracle.GaussianOracle` — exact
  Kronecker-Gaussian conditional sampler / perfect imputer.
- :class:`~motionbench.oracles.copula_oracle.CopulaOracle` — Gaussian copula
  with pluggable marginals (Burr XII, StudentT, ...).
- :class:`~motionbench.oracles.full_gaussian_oracle.FullGaussianOracle` —
  dense (non-Kronecker) covariance variant.
- :class:`~motionbench.oracles.deterministic.DeterministicConditionalOracle`
  — deterministic conditional-mean fills (exact f-of-mean grading targets).
"""

from motionbench.oracles.base import Oracle
from motionbench.oracles.copula_oracle import CopulaOracle
from motionbench.oracles.deterministic import (
    DeterministicConditionalOracle,
    marginal_fill,
)
from motionbench.oracles.full_gaussian_oracle import FullGaussianOracle
from motionbench.oracles.gaussian_oracle import GaussianOracle

__all__ = [
    "Oracle",
    "GaussianOracle",
    "CopulaOracle",
    "FullGaussianOracle",
    "DeterministicConditionalOracle",
    "marginal_fill",
]
