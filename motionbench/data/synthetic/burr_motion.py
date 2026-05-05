"""motionbench.data.synthetic.burr_motion — Gaussian copula benchmark with pluggable marginals.

Ported and generalised from ``CARE-PD/synthetic/burr_motion.py``.

DATA MODEL
----------
    Latent:   z ~ N(0, Σ_joints ⊗ I_F ⊗ Σ_time)   (correlated standard-normal marginals)
    Observed: x[j, f, t] = F⁻¹(Φ(z[j, f, t]))      (Gaussian copula push-forward)

where F is the chosen :class:`Marginal` (Burr XII by default).

The Kronecker structure requires that Σ_joints and Σ_time are *correlation*
matrices (diagonal = 1) so that z has unit marginal variance and Φ(z) ~ U(0,1)
element-wise.  Equicorrelation and AR(1) matrices satisfy this by construction.

CONFORMS TO
-----------
:class:`~motionbench.data.base.GroundTruthDataset` protocol (structural typing,
no inheritance):

* ``__getitem__(idx)`` → ``(x, y)`` with ``x: (J, F, T) float32``,
  ``y: scalar int64``.
* ``__len__()`` → ``N``.
* ``shape`` → ``(J, F, T)``.
* ``metadata`` → dict with required keys.
* ``oracle`` → :class:`~motionbench.oracles.copula_oracle.CopulaOracle`.

MARGINALS
---------
A :class:`Marginal` defines the per-coordinate marginal distribution ``F``
via three primitives:

* ``cdf(x)``      — ``F(x)``, the CDF evaluated at ``x``.
* ``quantile(u)`` — ``F⁻¹(u)``, the quantile (inverse CDF).
* ``pdf(x)``      — ``f(x)``, the probability density.

Provided implementations: :class:`BurrXII` (symmetric, heavy-tailed),
:class:`StudentT`, :class:`GaussianMarginal`, :class:`MixtureOfGaussians`,
:class:`SkewNormal`.

Numerical-stability notes
-------------------------
* All probabilities passed to ``ndtri`` (Φ⁻¹) are clipped to ``[ε, 1−ε]``
  with ``ε = 1e-9`` to prevent divergence at the boundary.
* The BurrXII quantile clips its input ``u`` to the same range.
* MixtureOfGaussians quantile uses ``scipy.optimize.brentq`` with a wide
  initial bracket derived from the component extremes.

References
----------
Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press.
    Copula transform identity (Chapter 2).

Aas, K., Jullum, M., & Løland, A. (2021).
    "Explaining individual predictions when features are dependent:
    More accurate approximations to Shapley values." arXiv:1903.10464.
    Copula-based conditional expectation, §3.4.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import scipy.optimize
import torch
from scipy import stats
from scipy.special import ndtr, ndtri  # Φ, Φ⁻¹
from torch import Tensor

from motionbench.utils.coalitions import ar1_cov, equicorr

if TYPE_CHECKING:
    from motionbench.oracles.copula_oracle import CopulaOracle

__all__ = [
    "Marginal",
    "BurrXII",
    "StudentT",
    "GaussianMarginal",
    "MixtureOfGaussians",
    "SkewNormal",
    "BurrMotionBenchmark",
]

# Probability clip to avoid Φ⁻¹(0) and Φ⁻¹(1) blow-up.
_EPS: float = 1e-9


# ---------------------------------------------------------------------------
# Marginal ABC
# ---------------------------------------------------------------------------


class Marginal(ABC):
    """Abstract base class for a univariate marginal distribution.

    A :class:`Marginal` defines the per-coordinate distribution ``F`` used
    by the Gaussian copula.  All three primitives must cover the full real
    line (or the distribution's support) consistently.

    The copula forward transform is ``z = Φ⁻¹(F(x))`` and the inverse is
    ``x = F⁻¹(Φ(z))``.  Numerical stability requires ``F(x) ∈ [ε, 1−ε]``
    before applying ``Φ⁻¹``.
    """

    @abstractmethod
    def cdf(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the CDF F(x).

        Args:
            x: Input values, any shape.

        Returns:
            Probabilities in ``(0, 1)``, same shape as ``x``.
        """

    @abstractmethod
    def quantile(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the quantile function F⁻¹(u).

        Args:
            u: Probability values in ``(0, 1)``, any shape.

        Returns:
            Quantiles, same shape as ``u``.
        """

    @abstractmethod
    def pdf(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the probability density function f(x).

        Args:
            x: Input values, any shape.

        Returns:
            Density values (≥ 0), same shape as ``x``.
        """


# ---------------------------------------------------------------------------
# BurrXII — symmetric, heavy-tailed
# ---------------------------------------------------------------------------


class BurrXII(Marginal):
    """Symmetric Burr XII marginal defined over the full real line.

    The standard Burr XII distribution has support [0, ∞) with CDF::

        F_Burr(x; c, k) = 1 − (1 + x^c)^{−k},   x ≥ 0.

    We symmetrise it to cover ℝ by folding around 0::

        F(x) = (1 + F_Burr(|x|; c, k)) / 2  for x ≥ 0
        F(x) = (1 − F_Burr(|x|; c, k)) / 2  for x < 0

    Equivalently, the Gaussian copula transform is (Aas et al. 2021)::

        z = sign(x) · Φ⁻¹((F_Burr(|x|; c, k) + 1) / 2)

    Default parameters ``c=2, k=2`` give mean 0, unit variance, and tail
    index 4 (polynomial tails |x|^{−5}, heavier than Gaussian, similar to t₄).

    Args:
        c: Shape parameter c > 0.  Controls tail shape.
        k: Shape parameter k > 0.  Controls tail weight (lower k → heavier).
    """

    def __init__(self, c: float = 2.0, k: float = 2.0) -> None:
        if c <= 0.0:
            raise ValueError(f"BurrXII c must be > 0; got {c}.")
        if k <= 0.0:
            raise ValueError(f"BurrXII k must be > 0; got {k}.")
        self.c = float(c)
        self.k = float(k)

    def _burr_cdf(self, x_pos: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Standard Burr XII CDF for x ≥ 0.

        Args:
            x_pos: Non-negative values.

        Returns:
            CDF values in [0, 1).
        """
        return 1.0 - (1.0 + x_pos**self.c) ** (-self.k)

    def _burr_quantile(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Standard Burr XII quantile for u ∈ (0, 1).

        Formula: Q(u; c, k) = ((1 − u)^{−1/k} − 1)^{1/c}.

        Args:
            u: Probabilities in (0, 1).

        Returns:
            Quantiles ≥ 0.
        """
        u_c = np.clip(u, _EPS, 1.0 - _EPS)
        return ((1.0 - u_c) ** (-1.0 / self.k) - 1.0) ** (1.0 / self.c)

    def _burr_pdf(self, x_pos: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Standard Burr XII PDF for x ≥ 0.

        Formula: f(x; c, k) = c k x^{c−1} (1 + x^c)^{−(k+1)}.

        Args:
            x_pos: Non-negative values.

        Returns:
            Density values ≥ 0.
        """
        xc = x_pos**self.c
        return self.c * self.k * x_pos ** (self.c - 1.0) * (1.0 + xc) ** (-(self.k + 1.0))

    def cdf(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Symmetric Burr XII CDF over ℝ.

        Defined as (1 ± F_Burr(|x|)) / 2 so that F(0) = 0.5.

        Args:
            x: Input values, any shape.

        Returns:
            CDF values in (0, 1), same shape as ``x``.
        """
        x = np.asarray(x, dtype=np.float64)
        abs_x = np.abs(x)
        f_pos = self._burr_cdf(abs_x)
        result = np.where(x >= 0, (1.0 + f_pos) / 2.0, (1.0 - f_pos) / 2.0)
        return np.clip(result, _EPS, 1.0 - _EPS)

    def quantile(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Symmetric Burr XII quantile over (0, 1) → ℝ.

        For u > 0.5: Q(u) = Q_Burr(2u − 1).
        For u < 0.5: Q(u) = −Q_Burr(1 − 2u).
        At u = 0.5:  Q(u) = 0.

        Args:
            u: Probabilities in (0, 1), any shape.

        Returns:
            Quantiles in ℝ, same shape as ``u``.
        """
        u = np.asarray(u, dtype=np.float64)
        u_c = np.clip(u, _EPS, 1.0 - _EPS)
        pos_mask = u_c > 0.5
        q = np.zeros_like(u_c)
        q[pos_mask] = self._burr_quantile(2.0 * u_c[pos_mask] - 1.0)
        neg_mask = u_c < 0.5
        q[neg_mask] = -self._burr_quantile(1.0 - 2.0 * u_c[neg_mask])
        return q

    def pdf(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Symmetric Burr XII PDF over ℝ.

        The density is f_Burr(|x|; c, k) / 2 by the folding formula,
        with f(0) = 0 when c > 1.

        Args:
            x: Input values, any shape.

        Returns:
            Density values ≥ 0, same shape as ``x``.
        """
        x = np.asarray(x, dtype=np.float64)
        return self._burr_pdf(np.abs(x)) / 2.0

    def __repr__(self) -> str:
        return f"BurrXII(c={self.c}, k={self.k})"


# ---------------------------------------------------------------------------
# StudentT
# ---------------------------------------------------------------------------


class StudentT(Marginal):
    """Student-t marginal distribution.

    Uses ``scipy.stats.t(df=df)`` for all three primitives.  The distribution
    has mean 0 for df > 1 and variance df/(df−2) for df > 2.  As df → ∞ it
    converges to N(0, 1).

    Args:
        df: Degrees of freedom (> 0).
    """

    def __init__(self, df: float) -> None:
        if df <= 0.0:
            raise ValueError(f"StudentT df must be > 0; got {df}.")
        self.df = float(df)
        self._dist = stats.t(df=self.df)

    def cdf(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Student-t CDF.

        Args:
            x: Input values, any shape.

        Returns:
            CDF values clipped to ``[ε, 1−ε]``, same shape as ``x``.
        """
        return np.asarray(
            np.clip(self._dist.cdf(np.asarray(x, dtype=np.float64)), _EPS, 1.0 - _EPS),
            dtype=np.float64,
        )

    def quantile(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Student-t quantile (percent-point function).

        Args:
            u: Probabilities in (0, 1), any shape.

        Returns:
            Quantiles in ℝ, same shape as ``u``.
        """
        return np.asarray(
            self._dist.ppf(np.clip(np.asarray(u, dtype=np.float64), _EPS, 1.0 - _EPS)),
            dtype=np.float64,
        )

    def pdf(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Student-t PDF.

        Args:
            x: Input values, any shape.

        Returns:
            Density values ≥ 0, same shape as ``x``.
        """
        return np.asarray(self._dist.pdf(np.asarray(x, dtype=np.float64)), dtype=np.float64)

    def __repr__(self) -> str:
        return f"StudentT(df={self.df})"


# ---------------------------------------------------------------------------
# GaussianMarginal
# ---------------------------------------------------------------------------


class GaussianMarginal(Marginal):
    """Standard normal N(0, 1) marginal.

    The Gaussian copula transform with this marginal is the identity:
    z = Φ⁻¹(Φ(x)) = x and x = Φ(Φ⁻¹(z)) = z.  Therefore
    :class:`~motionbench.oracles.copula_oracle.CopulaOracle` with
    ``GaussianMarginal`` reduces exactly to a Gaussian oracle.
    """

    def cdf(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Standard normal CDF Φ(x).

        Args:
            x: Input values, any shape.

        Returns:
            CDF values clipped to ``[ε, 1−ε]``, same shape as ``x``.
        """
        return np.asarray(
            np.clip(ndtr(np.asarray(x, dtype=np.float64)), _EPS, 1.0 - _EPS),
            dtype=np.float64,
        )

    def quantile(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Standard normal quantile Φ⁻¹(u).

        Args:
            u: Probabilities in (0, 1), any shape.

        Returns:
            Quantiles in ℝ, same shape as ``u``.
        """
        return np.asarray(
            ndtri(np.clip(np.asarray(u, dtype=np.float64), _EPS, 1.0 - _EPS)),
            dtype=np.float64,
        )

    def pdf(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Standard normal PDF φ(x).

        Args:
            x: Input values, any shape.

        Returns:
            Density values ≥ 0, same shape as ``x``.
        """
        x = np.asarray(x, dtype=np.float64)
        return np.asarray(np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi), dtype=np.float64)

    def __repr__(self) -> str:
        return "GaussianMarginal()"


# ---------------------------------------------------------------------------
# MixtureOfGaussians
# ---------------------------------------------------------------------------


class MixtureOfGaussians(Marginal):
    """Mixture of Gaussian marginal.

    CDF and PDF are analytic sums of weighted normal CDFs / PDFs.
    The quantile is computed numerically via ``scipy.optimize.brentq``
    applied to the CDF.

    Args:
        weights: Mixture weights (positive, summing to 1).
        means: Component means.
        scales: Component standard deviations (positive).

    Raises:
        ValueError: if lengths are inconsistent, weights are non-positive,
            or scales are non-positive.
    """

    def __init__(
        self,
        weights: list[float],
        means: list[float],
        scales: list[float],
    ) -> None:
        w = np.asarray(weights, dtype=np.float64)
        m = np.asarray(means, dtype=np.float64)
        s = np.asarray(scales, dtype=np.float64)
        if w.shape != m.shape or w.shape != s.shape:
            raise ValueError("weights, means, and scales must have the same length.")
        if np.any(w <= 0):
            raise ValueError("All weights must be positive.")
        if np.any(s <= 0):
            raise ValueError("All scales must be positive.")
        self._weights: npt.NDArray[np.float64] = w / w.sum()
        self._means: npt.NDArray[np.float64] = m
        self._scales: npt.NDArray[np.float64] = s
        # Bracket bounds: go far enough to cover all components.
        self._lo = float(m.min() - 8.0 * s.max())
        self._hi = float(m.max() + 8.0 * s.max())

    def cdf(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Mixture CDF: weighted sum of normal CDFs.

        Args:
            x: Input values, any shape.

        Returns:
            CDF values clipped to ``[ε, 1−ε]``, same shape as ``x``.
        """
        x = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x)
        for w, mu, sigma in zip(self._weights, self._means, self._scales, strict=True):
            result += w * ndtr((x - mu) / sigma)
        return np.clip(result, _EPS, 1.0 - _EPS)

    def quantile(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Mixture quantile via numerical inversion of the CDF.

        Uses ``scipy.optimize.brentq`` per element.  The bracket
        ``[lo, hi]`` is set to cover ±8σ_max from the extremes.

        Args:
            u: Probabilities in (0, 1), any shape.

        Returns:
            Quantiles in ℝ, same shape as ``u``.
        """
        u = np.clip(np.asarray(u, dtype=np.float64), _EPS, 1.0 - _EPS)
        flat = u.ravel()
        out = np.empty_like(flat)
        for i, ui in enumerate(flat):
            out[i] = scipy.optimize.brentq(
                lambda x, target=float(ui): float(self.cdf(np.array([x]))[0]) - target,
                self._lo,
                self._hi,
                xtol=1e-10,
            )
        return out.reshape(u.shape)

    def pdf(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Mixture PDF: weighted sum of normal PDFs.

        Args:
            x: Input values, any shape.

        Returns:
            Density values ≥ 0, same shape as ``x``.
        """
        x = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x)
        for w, mu, sigma in zip(self._weights, self._means, self._scales, strict=True):
            result += w * stats.norm.pdf(x, loc=mu, scale=sigma)
        return result

    def __repr__(self) -> str:
        return f"MixtureOfGaussians(weights={self._weights.tolist()}, means={self._means.tolist()}, scales={self._scales.tolist()})"


# ---------------------------------------------------------------------------
# SkewNormal
# ---------------------------------------------------------------------------


class SkewNormal(Marginal):
    """Skew-normal marginal.

    Uses ``scipy.stats.skewnorm(a=alpha)`` for all three primitives.
    The distribution degenerates to N(0, 1) as alpha → 0.

    Args:
        alpha: Shape parameter.  Positive values give right-skewed
            distributions; negative values give left-skewed.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self._dist = stats.skewnorm(a=self.alpha)

    def cdf(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Skew-normal CDF.

        Args:
            x: Input values, any shape.

        Returns:
            CDF values clipped to ``[ε, 1−ε]``, same shape as ``x``.
        """
        return np.asarray(
            np.clip(self._dist.cdf(np.asarray(x, dtype=np.float64)), _EPS, 1.0 - _EPS),
            dtype=np.float64,
        )

    def quantile(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Skew-normal quantile (percent-point function).

        Args:
            u: Probabilities in (0, 1), any shape.

        Returns:
            Quantiles in ℝ, same shape as ``u``.
        """
        return np.asarray(
            self._dist.ppf(np.clip(np.asarray(u, dtype=np.float64), _EPS, 1.0 - _EPS)),
            dtype=np.float64,
        )

    def pdf(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Skew-normal PDF.

        Args:
            x: Input values, any shape.

        Returns:
            Density values ≥ 0, same shape as ``x``.
        """
        return np.asarray(self._dist.pdf(np.asarray(x, dtype=np.float64)), dtype=np.float64)

    def __repr__(self) -> str:
        return f"SkewNormal(alpha={self.alpha})"


# ---------------------------------------------------------------------------
# BurrMotionBenchmark
# ---------------------------------------------------------------------------


class BurrMotionBenchmark:
    """Gaussian copula motion benchmark with pluggable marginals.

    Conforms to :class:`~motionbench.data.base.GroundTruthDataset` via
    structural typing (no inheritance required).

    The data model is::

        z ~ N(0, Σ_joints ⊗ I_F ⊗ Σ_time)
        x[j, f, t] = F⁻¹(Φ(z[j, f, t]))

    where F is the chosen :class:`Marginal`.  The Kronecker structure requires
    that Σ_joints and Σ_time are *correlation* matrices (diagonal = 1) so that
    z has unit marginal variance and Φ(z) ~ U(0, 1) element-wise.

    Args:
        J: Number of skeletal joints.
        F: Number of coordinates per joint (e.g. 3 for xyz).
        T: Number of frames per clip.
        N: Number of sequences to pre-generate at construction time.
        marginal: Marginal distribution for the copula.  Defaults to
            ``BurrXII(c=2, k=2)`` (unit-variance symmetric heavy-tailed).
        rho: Off-diagonal equicorrelation for the default joint covariance.
            Ignored when ``sigma_joints`` is provided.
        alpha: AR(1) temporal autocorrelation for the default temporal
            covariance.  Ignored when ``sigma_time`` is provided.
        sigma_joints: Custom ``(J, J)`` correlation matrix.  If ``None``
            uses ``equicorr(J, rho)``.
        sigma_time: Custom ``(T, T)`` correlation matrix.  If ``None``
            uses ``ar1_cov(T, alpha)``.
        sigma_joints_source: Provenance tag for ``sigma_joints``.
        sigma_time_source: Provenance tag for ``sigma_time``.
        seed: Random seed for pre-generating sequences.
    """

    def __init__(
        self,
        J: int = 17,
        F: int = 3,
        T: int = 81,
        N: int = 1000,
        marginal: Marginal | None = None,
        rho: float = 0.5,
        alpha: float = 0.8,
        sigma_joints: npt.NDArray[np.float64] | None = None,
        sigma_time: npt.NDArray[np.float64] | None = None,
        sigma_joints_source: str | None = None,
        sigma_time_source: str | None = None,
        seed: int | None = None,
    ) -> None:
        self.J = J
        self.F = F
        self.T = T
        self._N = N
        self.marginal: Marginal = marginal if marginal is not None else BurrXII(2.0, 2.0)

        # Joint covariance.
        if sigma_joints is None:
            self.Sigma_joints: npt.NDArray[np.float64] = equicorr(J, rho)
            self.sigma_joints_source: str = sigma_joints_source or "equicorr"
        else:
            sj = np.asarray(sigma_joints, dtype=np.float64)
            if sj.shape != (J, J):
                raise ValueError(f"sigma_joints shape {sj.shape} does not match J={J}.")
            sj = 0.5 * (sj + sj.T)
            eig_min = float(np.linalg.eigvalsh(sj).min())
            if eig_min < -1e-6:
                raise ValueError(
                    f"sigma_joints is not PSD (min eigenvalue {eig_min:.3e})."
                )
            self.Sigma_joints = sj
            self.sigma_joints_source = sigma_joints_source or "custom"

        # Temporal covariance.
        if sigma_time is None:
            self.Sigma_time: npt.NDArray[np.float64] = ar1_cov(T, alpha)
            self.sigma_time_source: str = sigma_time_source or "ar1"
        else:
            st = np.asarray(sigma_time, dtype=np.float64)
            if st.shape != (T, T):
                raise ValueError(f"sigma_time shape {st.shape} does not match T={T}.")
            st = 0.5 * (st + st.T)
            eig_min = float(np.linalg.eigvalsh(st).min())
            if eig_min < -1e-6:
                raise ValueError(
                    f"sigma_time is not PSD (min eigenvalue {eig_min:.3e})."
                )
            self.Sigma_time = st
            self.sigma_time_source = sigma_time_source or "custom"

        # Cholesky factors for unconditional sampling.
        self._L_joints: npt.NDArray[np.float64] = np.asarray(
            np.linalg.cholesky(self.Sigma_joints + 1e-8 * np.eye(J)), dtype=np.float64
        )
        self._L_time: npt.NDArray[np.float64] = np.asarray(
            np.linalg.cholesky(self.Sigma_time + 1e-8 * np.eye(T)), dtype=np.float64
        )

        # Lazily initialised oracle.
        self._oracle: CopulaOracle | None = None

        # Pre-generate sequences.
        x_np = self.sample(N, seed=seed)  # (N, J, F, T)

        # Simple proxy label: quantile-bin of joint-0 grand mean.
        score = x_np[:, 0, :, :].mean(axis=(1, 2))
        q33, q67 = np.percentile(score, [33.0, 67.0])
        y_np = np.where(score < q33, 0, np.where(score < q67, 1, 2)).astype(np.int64)

        self._x: Tensor = torch.tensor(x_np, dtype=torch.float32)
        self._y: Tensor = torch.tensor(y_np, dtype=torch.int64)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, N: int, seed: int | None = None) -> npt.NDArray[np.float32]:
        """Sample N sequences from the Gaussian copula distribution.

        Algorithm:
            1. Draw z ~ N(0, Σ_joints ⊗ I_F ⊗ Σ_time) using Cholesky.
            2. Apply element-wise copula: x[j,f,t] = F⁻¹(Φ(z[j,f,t])).

        Requires Σ_joints and Σ_time to be correlation matrices (diagonal = 1)
        so that z[j,f,t] ~ N(0,1) marginally and Φ(z) ~ U(0,1).

        Args:
            N: Number of sequences to draw.
            seed: Optional random seed.

        Returns:
            ``(N, J, F, T)`` float32 array.
        """
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((N, self.J, self.F, self.T)).astype(np.float64)
        # Apply temporal Cholesky: correlate over time.
        z = np.einsum("tT,njfT->njft", self._L_time, z)
        # Apply joint Cholesky: correlate over joints.
        z = np.einsum("jJ,nJft->njft", self._L_joints, z)
        # Copula transform: z → U(0,1) → x via marginal quantile.
        u = np.clip(ndtr(z), _EPS, 1.0 - _EPS)
        x = self.marginal.quantile(u)
        return x.astype(np.float32)

    # ------------------------------------------------------------------
    # GroundTruthDataset protocol
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return the ``idx``-th ``(x, y)`` pair.

        Args:
            idx: Sample index in ``[0, N)``.

        Returns:
            x: ``(J, F, T)`` float32 Tensor.
            y: Scalar int64 label Tensor.
        """
        return self._x[idx], self._y[idx]

    def __len__(self) -> int:
        """Number of pre-sampled sequences.

        Returns:
            N.
        """
        return self._N

    @property
    def shape(self) -> tuple[int, int, int]:
        """Spatial shape of every sample.

        Returns:
            ``(J, F, T)`` tuple.
        """
        return (self.J, self.F, self.T)

    @property
    def metadata(self) -> dict[str, object]:
        """Dataset-level metadata.

        Returns:
            Dict with keys ``"skeleton"``, ``"frame_rate"``, ``"marginal"``,
            ``"sigma_joints_source"``, ``"sigma_time_source"``.
        """
        return {
            "skeleton": "synthetic_burr_copula",
            "frame_rate": 27.0,
            "marginal": repr(self.marginal),
            "sigma_joints_source": self.sigma_joints_source,
            "sigma_time_source": self.sigma_time_source,
        }

    @property
    def oracle(self) -> CopulaOracle:
        """Ground-truth :class:`~motionbench.oracles.copula_oracle.CopulaOracle`.

        Lazily constructed on first access.

        Returns:
            The oracle for this dataset.
        """
        if self._oracle is None:
            from motionbench.oracles.copula_oracle import CopulaOracle  # noqa: PLC0415

            self._oracle = CopulaOracle(
                Sigma_joints=self.Sigma_joints,
                Sigma_time=self.Sigma_time,
                marginal=self.marginal,
            )
        return self._oracle

    def __repr__(self) -> str:
        return (
            f"BurrMotionBenchmark("
            f"J={self.J}, F={self.F}, T={self.T}, N={self._N}, "
            f"marginal={self.marginal!r}, "
            f"sigma_joints_source={self.sigma_joints_source!r}, "
            f"sigma_time_source={self.sigma_time_source!r})"
        )
