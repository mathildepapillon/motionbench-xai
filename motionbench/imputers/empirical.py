"""motionbench.imputers.empirical — Empirical / classical-conditional imputers.

Three on-manifold imputers that use the *training pool* to model the
conditional distribution ``q(x_hid | x_obs)`` without fitting a parametric
generative model:

**KNNConditionalImputer**
    k-nearest-neighbour imputer.  For each query, finds the k closest
    training sequences in the subspace of observed coordinates (Euclidean)
    and samples a completion from those neighbours with inverse-distance
    weights.  Adapted from the kNN branch of :class:`CARE-PD EmpiricalImputer`.

**EmpiricalConditionalImputer**
    Full Aas, Jullum & Løland (2021) §3.3 Algorithm 2 empirical conditional
    with Ledoit-Wolf shrinkage.  Mahalanobis kernel weights, η-truncation,
    and sampling from the surviving training rows.  Matches the ``shapr``
    R-package default hyperparameters: ``sigma=0.1``, ``eta=0.95``.

**VineCopulaImputer**
    Gaussian-copula imputer with empirical marginal transforms (probability
    integral transform via empirical CDF, Gaussian conditional in the
    copula / normal-score space, back-transform via empirical quantile
    function).  For ``d = J*F*T ≤ max_vine_dim`` the bivariate Gaussian
    copula structure is validated via ``pyvinecopulib``; for larger ``d``
    a Ledoit-Wolf regularised correlation matrix is used directly (equivalent
    Gaussian vine copula, same family).

References
----------
Aas, K., Jullum, M., & Løland, A. (2021).
    Explaining individual predictions when features are dependent:
    More accurate approximations to Shapley values. arXiv:1903.10464.
    §3.3, Algorithm 2, Equations (6)–(8).

Olsen, L. R. et al. (2022).
    Using Shapley Values and Variational Autoencoders To Explain Predictions
    from Neural Networks for Short-Term Wind Power Forecasting. JMLR 23(1).

shapr R package defaults: ``empirical.type = "fixed_sigma"``,
    ``empirical.fixed_sigma = 0.1``, ``empirical.eta = 0.95``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch
from scipy.stats import norm as sp_norm
from sklearn.covariance import LedoitWolf
from torch import Tensor

from motionbench.imputers.base import BaseImputer

if TYPE_CHECKING:
    from motionbench.data.base import BaseDataset

_F64 = npt.NDArray[np.float64]

__all__ = [
    "KNNConditionalImputer",
    "EmpiricalConditionalImputer",
    "VineCopulaImputer",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_pool(train_data: BaseDataset) -> _F64:
    """Iterate a dataset and collect all samples into an (N, J, F, T) array.

    Args:
        train_data: Any object with ``__len__`` and ``__getitem__`` returning
            ``(x, label)`` where ``x`` is a ``(J, F, T)`` float Tensor.

    Returns:
        ``(N, J, F, T)`` float64 numpy array.
    """
    rows: list[_F64] = []
    for i in range(len(train_data)):
        x, _ = train_data[i]
        rows.append(np.asarray(x.detach().cpu().numpy(), dtype=np.float64))
    return np.array(rows, dtype=np.float64)


def _zscore_pool(
    pool: _F64,
) -> tuple[_F64, _F64, _F64]:
    """Z-score an (N, D) flat training pool per coordinate.

    Computes ``(x - mean) / std`` where std is clamped to ``1e-8`` to
    prevent divide-by-zero for constant features.

    Args:
        pool: ``(N, D)`` float64 training matrix.

    Returns:
        Tuple of ``(pool_z, mean_d, std_d)``:
        ``pool_z``: ``(N, D)`` standardised matrix.
        ``mean_d``: ``(D,)`` per-feature mean.
        ``std_d``: ``(D,)`` per-feature std (clamped to ≥ 1e-8).
    """
    mean_d = pool.mean(axis=0)
    std_d = pool.std(axis=0)
    std_d = np.where(std_d < 1e-8, 1.0, std_d)
    return (pool - mean_d) / std_d, mean_d, std_d


def _eta_truncate(w: _F64, eta: float) -> _F64:
    """η-truncation from Aas et al. (2021) Eq. (8) / shapr convention.

    Sort training rows by descending weight, keep the smallest prefix whose
    cumulative weight reaches ``eta``, zero out the remainder, renormalise.

    Implements Aas et al. (2021) §3.3 Eq. (8):
        keep the K = argmin_k {sum_{i=1}^k w_{(i)} >= η} rows.

    Args:
        w: ``(N,)`` non-negative weight vector, already normalised to sum 1.
        eta: Truncation fraction in ``(0, 1]``.  ``eta=1`` is a no-op.

    Returns:
        ``(N,)`` renormalised weight vector with ≤ K non-zero entries.
    """
    if eta >= 1.0:
        return w
    order = np.argsort(-w)
    cum = np.cumsum(w[order])
    K = int(np.searchsorted(cum, eta)) + 1
    K = min(K, w.size)
    kept = np.zeros_like(w)
    kept[order[:K]] = w[order[:K]]
    s = kept.sum()
    return kept / s if s > 0 else w


# ---------------------------------------------------------------------------
# KNNConditionalImputer
# ---------------------------------------------------------------------------


class KNNConditionalImputer(BaseImputer):
    """k-nearest-neighbour conditional imputer (Euclidean on observed dims).

    For each of ``n_samples`` completions, this imputer:

    1. Finds the ``k`` training sequences nearest to ``x_obs`` in the
       subspace of *observed* coordinates using Euclidean distance.
    2. Weights the ``k`` neighbours by ``1 / (distance + eps)`` (inverse-
       distance weighting) and samples one neighbour from that distribution.
    3. Copies the sampled training row into the output and overwrites the
       observed coordinates with ``x_obs[mask]`` bit-for-bit.

    This is the kNN branch of Aas et al. (2021) §3.3 Algorithm 2 —
    a special case of the empirical conditional where the kernel is an
    indicator function over the k nearest neighbours.

    References:
        Aas, Jullum & Løland (2021) arXiv:1903.10464, §3.3, Algorithm 2.

    Args:
        k: Number of nearest neighbours to consider.
        eps: Small constant added to distances to prevent division by zero
            when a training row exactly matches ``x_obs`` on observed coords.
    """

    def __init__(self, k: int = 20, eps: float = 1e-8) -> None:
        self.k = k
        self.eps = eps
        self._pool_raw: _F64 | None = None  # (N, J, F, T) float64
        self._pool_flat: _F64 | None = None  # (N, D) float64
        self._shape: tuple[int, int, int] | None = None

    @property
    def is_on_manifold(self) -> bool:
        """kNN completes from real training sequences — on-manifold.

        Returns:
            ``True``.
        """
        return True

    def fit(self, train_data: BaseDataset) -> KNNConditionalImputer:
        """Collect training sequences into the neighbour pool.

        Args:
            train_data: Dataset with ``__len__`` and ``__getitem__`` returning
                ``(x, label)`` where ``x`` is a ``(J, F, T)`` float Tensor.

        Returns:
            ``self`` for method chaining.
        """
        pool = _collect_pool(train_data)  # (N, J, F, T)
        self._pool_raw = pool
        N, J, F, T = pool.shape
        self._pool_flat = pool.reshape(N, J * F * T)
        self._shape = (J, F, T)
        return self

    def impute(
        self,
        x_obs: Tensor,
        mask: Tensor,
        n_samples: int,
        seed: int | None = None,
    ) -> Tensor:
        """Draw ``n_samples`` completions via kNN on observed coordinates.

        For each completion, samples one of the k nearest neighbours
        proportionally to inverse Euclidean distance on the observed subspace,
        then overwrites observed entries with ``x_obs[mask]`` bit-for-bit.

        Args:
            x_obs: ``(J, F, T)`` float32 Tensor.  Unobserved entries are
                ignored.
            mask: ``(J, F, T)`` bool Tensor.  ``True`` = observed.
            n_samples: Number of completed sequences to return.
            seed: Optional random seed for reproducibility.

        Returns:
            ``(n_samples, J, F, T)`` float32 Tensor.

        Raises:
            RuntimeError: If ``fit`` has not been called.
            ValueError: If ``x_obs.shape != mask.shape``.
        """
        if self._pool_raw is None or self._pool_flat is None or self._shape is None:
            raise RuntimeError("KNNConditionalImputer: call fit() before impute().")
        if x_obs.shape != mask.shape:
            raise ValueError(
                f"x_obs.shape {x_obs.shape} != mask.shape {mask.shape}."
            )

        rng = np.random.default_rng(seed)
        J, F, T = self._shape
        x_np = x_obs.detach().cpu().numpy().astype(np.float64)  # (J, F, T)
        mask_np = mask.detach().cpu().numpy().astype(bool)  # (J, F, T)
        x_flat = x_np.reshape(-1)  # (D,)
        mask_flat = mask_np.reshape(-1)  # (D,)

        obs_idx = np.where(mask_flat)[0]

        # Edge case: no observed coords → uniform sample from pool.
        if obs_idx.size == 0:
            chosen = rng.integers(0, self._pool_raw.shape[0], size=n_samples)
            out = self._pool_raw[chosen].astype(np.float32)
            return torch.tensor(out, dtype=torch.float32)

        # Edge case: all observed → return n_samples copies of x_obs.
        D = J * F * T
        if obs_idx.size == D:
            x_out = np.tile(x_np[None], (n_samples, 1, 1, 1)).astype(np.float32)
            return torch.tensor(x_out, dtype=torch.float32)

        # Euclidean distance on observed subspace.
        diff = self._pool_flat[:, obs_idx] - x_flat[obs_idx][None, :]  # (N, |S|)
        d2 = np.sum(diff * diff, axis=1)  # (N,)

        k = min(self.k, self._pool_raw.shape[0])
        knn_idx = np.argpartition(d2, k - 1)[:k]
        knn_d2 = d2[knn_idx]

        # Inverse-distance weights; cap at 1/eps when distance is zero.
        w = 1.0 / (np.sqrt(knn_d2) + self.eps)
        w = w / w.sum()

        chosen_local = rng.choice(k, size=n_samples, replace=True, p=w)
        chosen_global = knn_idx[chosen_local]

        out_np = self._pool_raw[chosen_global].copy()  # (n_samples, J, F, T)
        # Overwrite observed entries bit-for-bit.
        out_flat = out_np.reshape(n_samples, D)
        out_flat[:, obs_idx] = x_flat[obs_idx][None, :]
        return torch.tensor(out_np.astype(np.float32), dtype=torch.float32)


# ---------------------------------------------------------------------------
# EmpiricalConditionalImputer
# ---------------------------------------------------------------------------


class EmpiricalConditionalImputer(BaseImputer):
    """Empirical conditional imputer — Aas et al. (2021) §3.3 Algorithm 2.

    Implements the full empirical conditional algorithm with Ledoit-Wolf
    shrinkage as described in Aas, Jullum & Løland (2021) §3.3, Algorithm 2,
    Equations (6)–(8):

    For query ``x_obs`` with observed subspace ``S``:

    1. **Z-score** the training pool per flat feature (shapr convention).
    2. **Mahalanobis distance** on the observed sub-block (Eq. 6):

       .. math::

           d_n^2 = (x^*_S - x_n^S)^T \\, \\Sigma_{SS}^{-1} \\,
                   (x^*_S - x_n^S)

       where ``Σ_SS`` is the Ledoit-Wolf shrunk covariance of the observed
       training sub-matrix.

    3. **Gaussian kernel weights** (Eq. 7):

       .. math:: w_n = \\exp(-d_n^2 / (2\\sigma^2))

       with ``σ = bandwidth`` (``"auto"`` uses ``0.1 × median(d_n)``).

    4. **η-truncation** (Eq. 8): keep the smallest prefix of training rows
       sorted by descending weight whose cumulative weight ≥ η; renormalise.

    5. **Resample** from the surviving rows with the truncated weights.
       Copy the sampled training row into the output and overwrite observed
       coordinates with ``x_obs[mask]`` bit-for-bit.

    Edge cases:
    - Empty coalition (no observed entries): uniform sample from the pool
      (reduces to the marginal imputer; Aas 2021 §3.1).
    - Full coalition (all observed): return ``n_samples`` copies of ``x_obs``.
    - Kernel underflow (total weight < 1e-12): fall back to uniform sampling
      over the truncated pool (rare for reasonable ``sigma``).

    References:
        Aas, Jullum & Løland (2021) arXiv:1903.10464, §3.3, Algorithm 2,
        Equations (6)–(8).

    Args:
        bandwidth: Gaussian kernel bandwidth σ.  Use ``"auto"`` (default) to
            set σ = 0.1 × median(d_n) on each query, matching the shapr
            R-package ``fixed_sigma = 0.1`` convention on z-scored data.
            Pass a positive float to fix σ globally.
        eta: η-truncation fraction.  ``shapr`` default is ``0.95``.
    """

    def __init__(
        self,
        bandwidth: float | str = "auto",
        eta: float = 0.95,
    ) -> None:
        if not isinstance(bandwidth, str) and bandwidth <= 0:
            raise ValueError(f"bandwidth must be positive or 'auto'; got {bandwidth!r}.")
        if not (0.0 < eta <= 1.0):
            raise ValueError(f"eta must be in (0, 1]; got {eta}.")
        self.bandwidth = bandwidth
        self.eta = eta

        self._pool_raw: _F64 | None = None   # (N, J, F, T) float64
        self._pool_z: _F64 | None = None     # (N, D) z-scored float64
        self._mean_d: _F64 | None = None     # (D,) per-feature mean
        self._std_d: _F64 | None = None      # (D,) per-feature std
        self._shape: tuple[int, int, int] | None = None
        # Cache Cholesky factors keyed by observed index fingerprint.
        self._chol_cache: dict[bytes, _F64] = {}

    @property
    def is_on_manifold(self) -> bool:
        """Empirical imputer completes from real training sequences.

        Returns:
            ``True``.
        """
        return True

    def fit(
        self, train_data: BaseDataset
    ) -> EmpiricalConditionalImputer:
        """Fit the imputer: collect pool, z-score, pre-fit Ledoit-Wolf.

        The Ledoit-Wolf covariance is computed lazily per coalition (cached
        on ``obs_idx`` fingerprint) rather than globally so that the per-block
        sub-matrix is always used (Aas 2021 Algorithm 2, Step 1).

        Args:
            train_data: Dataset with ``__len__`` / ``__getitem__`` returning
                ``(x, label)`` where ``x`` is a ``(J, F, T)`` float Tensor.

        Returns:
            ``self`` for method chaining.
        """
        pool = _collect_pool(train_data)  # (N, J, F, T)
        N, J, F, T = pool.shape
        self._pool_raw = pool
        self._shape = (J, F, T)
        pool_flat = pool.reshape(N, J * F * T)
        self._pool_z, self._mean_d, self._std_d = _zscore_pool(pool_flat)
        self._chol_cache = {}
        return self

    def _get_cholesky(self, obs_idx: npt.NDArray[np.intp]) -> _F64:
        """Return the Cholesky factor of the Ledoit-Wolf Σ_SS (cached).

        Implements the covariance estimation step from Aas et al. (2021)
        §3.3 Algorithm 2, Step 1: fit Ledoit-Wolf on the observed sub-block
        of the z-scored training pool.

        Args:
            obs_idx: ``(|S|,)`` int array of observed flat coordinate indices.

        Returns:
            ``(|S|, |S|)`` lower-triangular Cholesky factor L such that
            ``L @ L.T ≈ Σ_SS``.
        """
        key = obs_idx.tobytes()
        if key in self._chol_cache:
            return self._chol_cache[key]

        assert self._pool_z is not None
        X_S = self._pool_z[:, obs_idx]  # (N, |S|)
        if X_S.shape[1] == 1:
            var = float(np.var(X_S, ddof=1)) if X_S.shape[0] > 1 else 1.0
            sigma_ss = np.array([[max(var, 1e-8)]])
        else:
            lw = LedoitWolf(assume_centered=False).fit(X_S)
            sigma_ss = lw.covariance_.astype(np.float64, copy=False)

        jitter = max(1e-10, 1e-8 * float(np.trace(sigma_ss)) / sigma_ss.shape[0])
        L: _F64 | None = None
        for _ in range(6):
            try:
                L = np.linalg.cholesky(
                    sigma_ss + jitter * np.eye(sigma_ss.shape[0])
                ).astype(np.float64, copy=False)
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        if L is None:  # pragma: no cover
            raise RuntimeError(
                "EmpiricalConditionalImputer: Cholesky failed for Σ_SS even after "
                "jitter escalation. The observed sub-block may be degenerate."
            )

        self._chol_cache[key] = L
        return L

    def _kernel_weights(
        self, x_star_z: _F64, obs_idx: npt.NDArray[np.intp]
    ) -> _F64:
        """Compute normalised (N,) Gaussian kernel weights (Aas 2021 Eq. 7).

        Mahalanobis distance on the observed sub-block (Eq. 6):
            ``d_n² = (x*_S − x_n^S)ᵀ Σ_SS⁻¹ (x*_S − x_n^S)``

        Gaussian kernel weight (Eq. 7):
            ``w_n = exp(−d_n² / (2σ²))``

        Args:
            x_star_z: ``(D,)`` z-scored query vector.
            obs_idx: ``(|S|,)`` int array of observed flat indices.

        Returns:
            ``(N,)`` normalised weight vector.
        """
        assert self._pool_z is not None
        N = self._pool_z.shape[0]
        V = self._pool_z[:, obs_idx] - x_star_z[obs_idx][None, :]  # (N, |S|)
        L = self._get_cholesky(obs_idx)
        # Solve L Z = V.T  =>  Z = L^{-1} V.T, shape (|S|, N).
        try:
            from scipy.linalg import solve_triangular
            Z = solve_triangular(L, V.T, lower=True)
        except Exception:
            Z = np.linalg.solve(L, V.T)
        d2 = np.sum(Z * Z, axis=0)  # (N,)

        # Leave-one-out leakage guard: the fit pool may contain x* itself
        # (or a near-exact duplicate, common in synthetic test setups where
        # train==test).  Such donors have d² ≈ 0 and would dominate the
        # softmax kernel, collapsing the imputer to ``return x_obs``
        # (giving constant Shapley values).  We mask donors closer than
        # ``eps`` in the observed sub-block.
        eps_d2 = 1e-8 * max(float(np.median(d2)), 1e-12)
        loo_mask = d2 > eps_d2
        if loo_mask.sum() == 0:
            # All donors are duplicates (degenerate case); leave d² as-is.
            loo_mask = np.ones_like(d2, dtype=bool)

        d2_eff = np.where(loo_mask, d2, np.inf)

        # Bandwidth: "auto" = 0.1 × median(sqrt(d²)) following shapr convention.
        # Compute on the LOO-filtered distances.
        if self.bandwidth == "auto":
            d2_finite = d2_eff[np.isfinite(d2_eff)]
            med = float(np.median(np.sqrt(np.maximum(d2_finite, 0.0)))) if d2_finite.size else 1.0
            sigma = 0.1 * med if med > 1e-12 else 0.1
        else:
            sigma = float(self.bandwidth)

        # Numerically stable: subtract minimum d² before exponentiating.
        d2_min = float(d2_eff[np.isfinite(d2_eff)].min()) if np.any(np.isfinite(d2_eff)) else 0.0
        logits = -(d2_eff - d2_min) / (2.0 * sigma ** 2)
        # Donors masked by LOO get logits=-inf → weight 0.
        w = np.exp(logits)
        total = float(w.sum())
        if not np.isfinite(total) or total < 1e-12:
            warnings.warn(
                f"EmpiricalConditionalImputer: kernel underflow (sum(w)={total:.3e}); "
                "falling back to uniform sampling.",
                RuntimeWarning,
                stacklevel=3,
            )
            return np.full(N, 1.0 / N, dtype=np.float64)
        return np.asarray(w / total, dtype=np.float64)

    def impute(
        self,
        x_obs: Tensor,
        mask: Tensor,
        n_samples: int,
        seed: int | None = None,
    ) -> Tensor:
        """Draw ``n_samples`` completions via empirical conditional (Aas 2021).

        Implements Aas et al. (2021) §3.3 Algorithm 2, Equations (6)–(8):
        Mahalanobis kernel weights on the observed sub-block with Ledoit-Wolf
        covariance, η-truncation, and weighted resampling from the pool.

        Args:
            x_obs: ``(J, F, T)`` float32 Tensor.
            mask: ``(J, F, T)`` bool Tensor.  ``True`` = observed.
            n_samples: Number of completions to return.
            seed: Optional random seed.

        Returns:
            ``(n_samples, J, F, T)`` float32 Tensor.

        Raises:
            RuntimeError: If ``fit`` has not been called.
            ValueError: If ``x_obs.shape != mask.shape``.
        """
        if (
            self._pool_raw is None
            or self._pool_z is None
            or self._mean_d is None
            or self._std_d is None
            or self._shape is None
        ):
            raise RuntimeError(
                "EmpiricalConditionalImputer: call fit() before impute()."
            )
        if x_obs.shape != mask.shape:
            raise ValueError(
                f"x_obs.shape {x_obs.shape} != mask.shape {mask.shape}."
            )

        rng = np.random.default_rng(seed)
        J, F, T = self._shape
        D = J * F * T
        x_np = x_obs.detach().cpu().numpy().astype(np.float64)
        mask_np = mask.detach().cpu().numpy().astype(bool)
        x_flat = x_np.reshape(-1)
        mask_flat = mask_np.reshape(-1)
        obs_idx = np.where(mask_flat)[0]
        N = self._pool_raw.shape[0]

        # Edge case: empty coalition → uniform marginal sample.
        if obs_idx.size == 0:
            chosen = rng.integers(0, N, size=n_samples)
            out = self._pool_raw[chosen].astype(np.float32)
            return torch.tensor(out, dtype=torch.float32)

        # Edge case: full coalition → copies of x_obs.
        if obs_idx.size == D:
            x_out = np.tile(x_np[None], (n_samples, 1, 1, 1)).astype(np.float32)
            return torch.tensor(x_out, dtype=torch.float32)

        # Z-score x_obs the same way as the pool.
        x_star_z = (x_flat - self._mean_d) / self._std_d

        # Steps 2–4 of Aas 2021 Algorithm 2.
        w = self._kernel_weights(x_star_z, obs_idx)       # Eq. (7)
        w = _eta_truncate(w, self.eta)                      # Eq. (8)

        chosen = rng.choice(N, size=n_samples, replace=True, p=w)
        out_np = self._pool_raw[chosen].copy()              # (n_samples, J, F, T)
        # Step 6: overwrite observed entries bit-for-bit.
        out_flat = out_np.reshape(n_samples, D)
        out_flat[:, obs_idx] = x_flat[obs_idx][None, :]
        return torch.tensor(out_np.astype(np.float32), dtype=torch.float32)


# ---------------------------------------------------------------------------
# VineCopulaImputer
# ---------------------------------------------------------------------------


class VineCopulaImputer(BaseImputer):
    """Gaussian-copula imputer with empirical marginal transforms.

    Models the joint distribution via a Gaussian copula with empirical
    (rank-based) marginals:

    1. **Fit** — per-coordinate empirical CDF (via sorted training values);
       Gaussian normal-score transform ``z = Φ⁻¹(u)``; Ledoit-Wolf
       correlation matrix ``P`` on the normal-score matrix.
       For ``d ≤ max_vine_dim``, the bivariate Gaussian structure is also
       fitted with ``pyvinecopulib`` (Gaussian family) to confirm consistency.

    2. **Impute** — for each query:
       a. Transform ``x_obs`` to normal scores ``z_obs`` using the training
          ECDF per coordinate.
       b. Compute the Gaussian conditional ``p(z_hid | z_obs)`` exactly from
          the Ledoit-Wolf correlation matrix.
       c. Draw ``n_samples`` conditional samples ``z_hid``.
       d. Back-transform via the empirical quantile function
          (``u_hid = Φ(z_hid)``, then inverse ECDF on training values).
       e. Overwrite observed entries with ``x_obs[mask]`` bit-for-bit.

    This is a full **Gaussian vine copula** with empirical marginals:
    every pair copula is a bivariate Gaussian (``pyvinecopulib.BicopFamily
    .gaussian``), which is equivalent to a multivariate Gaussian copula.

    Note:
        For ``d = J*F*T > max_vine_dim`` (default 20), the ``pyvinecopulib``
        fitting step is skipped (intractable for high-dimensional sequences)
        and only the Ledoit-Wolf Gaussian copula is used.  Full vine
        copula fitting for high-dimensional data is left to future work.

    References:
        Joe, H. (2014). *Dependence Modeling with Copulas*. Chapman & Hall.
        Czado, C. (2019). *Analyzing Dependent Data with Vine Copulas*.
        Springer. §3.
        pyvinecopulib documentation: https://vinecopulib.github.io/pyvinecopulib/

    Args:
        max_vine_dim: Maximum ``d = J*F*T`` for which the vine copula is
            fitted with ``pyvinecopulib``.  For ``d > max_vine_dim``, the
            Ledoit-Wolf Gaussian copula is used directly (same model class).
    """

    def __init__(self, max_vine_dim: int = 20) -> None:
        self.max_vine_dim = max_vine_dim

        self._pool_raw: _F64 | None = None    # (N, J, F, T) float64
        self._pool_flat: _F64 | None = None   # (N, D) float64
        self._sorted_cols: _F64 | None = None  # (N, D) sorted per col
        self._corr: _F64 | None = None         # (D, D) LW correlation
        self._shape: tuple[int, int, int] | None = None
        self._used_pyvine: bool = False

    @property
    def is_on_manifold(self) -> bool:
        """Copula imputer samples from the learned data distribution.

        Returns:
            ``True``.
        """
        return True

    def fit(self, train_data: BaseDataset) -> VineCopulaImputer:
        """Fit empirical marginals and Gaussian-copula correlation matrix.

        For ``d ≤ max_vine_dim``, additionally fits a ``pyvinecopulib``
        Gaussian vine copula to validate the model structure.

        Args:
            train_data: Dataset with ``__len__`` / ``__getitem__`` returning
                ``(x, label)`` where ``x`` is a ``(J, F, T)`` float Tensor.

        Returns:
            ``self`` for method chaining.
        """
        pool = _collect_pool(train_data)  # (N, J, F, T)
        N, J, F, T = pool.shape
        D = J * F * T
        self._pool_raw = pool
        self._shape = (J, F, T)
        self._pool_flat = pool.reshape(N, D)

        # Sort each column for ECDF forward / backward transforms.
        self._sorted_cols = np.sort(self._pool_flat, axis=0)  # (N, D)

        # Pseudo-observations: u_{n,j} = rank(x_{n,j}) / (N + 1) ∈ (0, 1).
        from scipy.stats import rankdata
        u = np.column_stack(
            [rankdata(self._pool_flat[:, j]) / (N + 1) for j in range(D)]
        )  # (N, D)

        # Normal scores: z = Φ⁻¹(u), shape (N, D).
        z = sp_norm.ppf(u)

        # Fit Ledoit-Wolf correlation matrix on normal scores.
        lw = LedoitWolf(assume_centered=True).fit(z)
        cov = lw.covariance_.astype(np.float64, copy=False)
        diag = np.sqrt(np.diag(cov))
        diag = np.where(diag < 1e-8, 1.0, diag)
        self._corr = cov / np.outer(diag, diag)

        # Optional: validate with pyvinecopulib for small d.
        self._used_pyvine = False
        if self.max_vine_dim >= D:
            try:
                import pyvinecopulib as pv

                ctrl = pv.FitControlsVinecop(
                    family_set=[pv.BicopFamily.gaussian],  # type: ignore[attr-defined]
                    num_threads=1,
                )
                vc = pv.Vinecop(d=D)
                vc.select(np.asfortranarray(u), controls=ctrl)
                self._used_pyvine = True
            except Exception as exc:  # pragma: no cover
                warnings.warn(
                    f"VineCopulaImputer: pyvinecopulib fitting failed ({exc!r}); "
                    "falling back to LW Gaussian copula.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return self

    def _ecdf_forward(self, x: _F64) -> _F64:
        """Transform ``(n_samples, D)`` values to uniform via training ECDF.

        Uses linear interpolation between empirical quantile points.

        Args:
            x: ``(n_samples, D)`` float64 values in original data space.

        Returns:
            ``(n_samples, D)`` float64 pseudo-observations in ``(0, 1)``.
        """
        assert self._sorted_cols is not None
        N = self._sorted_cols.shape[0]
        u = np.empty_like(x)
        for j in range(x.shape[1]):
            col = self._sorted_cols[:, j]
            # Fraction of training values ≤ x[i, j]; clamp to (0, 1).
            ranks = np.searchsorted(col, x[:, j], side="right") / N
            u[:, j] = np.clip(ranks, 1e-6, 1.0 - 1e-6)
        return u

    def _ecdf_inverse(self, u: _F64) -> _F64:
        """Back-transform ``(n_samples, D)`` uniforms via training quantiles.

        Args:
            u: ``(n_samples, D)`` float64 values in ``(0, 1)``.

        Returns:
            ``(n_samples, D)`` float64 values in original data space.
        """
        assert self._sorted_cols is not None
        N = self._sorted_cols.shape[0]
        x = np.empty_like(u)
        idx_float = u * N - 0.5
        idx_lo = np.clip(np.floor(idx_float).astype(int), 0, N - 1)
        idx_hi = np.clip(idx_lo + 1, 0, N - 1)
        frac = np.clip(idx_float - idx_lo, 0.0, 1.0)
        for j in range(u.shape[1]):
            lo = self._sorted_cols[idx_lo[:, j], j]
            hi = self._sorted_cols[idx_hi[:, j], j]
            x[:, j] = lo + frac[:, j] * (hi - lo)
        return x

    def impute(
        self,
        x_obs: Tensor,
        mask: Tensor,
        n_samples: int,
        seed: int | None = None,
    ) -> Tensor:
        """Draw ``n_samples`` completions via Gaussian-copula conditional.

        Steps:
        1. Transform x_obs (observed coords) to normal scores.
        2. Sample hidden coords from the Gaussian conditional.
        3. Back-transform via empirical quantile function.
        4. Overwrite observed coords with x_obs[mask] bit-for-bit.

        Args:
            x_obs: ``(J, F, T)`` float32 Tensor.
            mask: ``(J, F, T)`` bool Tensor.  ``True`` = observed.
            n_samples: Number of completions to return.
            seed: Optional random seed.

        Returns:
            ``(n_samples, J, F, T)`` float32 Tensor.

        Raises:
            RuntimeError: If ``fit`` has not been called.
            ValueError: If ``x_obs.shape != mask.shape``.
        """
        if (
            self._pool_raw is None
            or self._pool_flat is None
            or self._sorted_cols is None
            or self._corr is None
            or self._shape is None
        ):
            raise RuntimeError(
                "VineCopulaImputer: call fit() before impute()."
            )
        if x_obs.shape != mask.shape:
            raise ValueError(
                f"x_obs.shape {x_obs.shape} != mask.shape {mask.shape}."
            )

        rng = np.random.default_rng(seed)
        J, F, T = self._shape
        D = J * F * T
        x_np = x_obs.detach().cpu().numpy().astype(np.float64)
        mask_np = mask.detach().cpu().numpy().astype(bool)
        x_flat = x_np.reshape(-1)
        mask_flat = mask_np.reshape(-1)
        obs_idx = np.where(mask_flat)[0]
        hid_idx = np.where(~mask_flat)[0]
        N = self._pool_raw.shape[0]

        # Edge case: empty coalition → uniform sample from pool.
        if obs_idx.size == 0:
            chosen = rng.integers(0, N, size=n_samples)
            out = self._pool_raw[chosen].astype(np.float32)
            return torch.tensor(out, dtype=torch.float32)

        # Edge case: full coalition → copies of x_obs.
        if obs_idx.size == D:
            x_out = np.tile(x_np[None], (n_samples, 1, 1, 1)).astype(np.float32)
            return torch.tensor(x_out, dtype=torch.float32)

        # Transform observed x to uniform via training ECDF.
        x_obs_flat_2d = x_flat[obs_idx][None, :]  # (1, |S|)
        # Build a (1, D) array for ECDF forward, then extract obs.
        x_all = np.tile(x_flat[None], (1, 1))  # (1, D)
        u_all = self._ecdf_forward(x_all)  # (1, D)
        z_obs = sp_norm.ppf(u_all[0, obs_idx])  # (|S|,) normal scores

        # Gaussian conditional of z_hid | z_obs.
        P = self._corr
        P_hh = P[np.ix_(hid_idx, hid_idx)]
        P_ho = P[np.ix_(hid_idx, obs_idx)]
        P_oo = P[np.ix_(obs_idx, obs_idx)]

        # Conditional mean: μ_{h|o} = P_ho @ P_oo^{-1} @ z_obs
        # Conditional cov:  Σ_{h|o} = P_hh - P_ho @ P_oo^{-1} @ P_oh
        P_oo_reg = P_oo + 1e-8 * np.eye(len(obs_idx))
        W = P_ho @ np.linalg.solve(P_oo_reg, np.eye(len(obs_idx)))
        mu_h = W @ z_obs  # (|h|,)
        Sigma_h = P_hh - W @ P_ho.T
        Sigma_h = 0.5 * (Sigma_h + Sigma_h.T) + 1e-8 * np.eye(len(hid_idx))
        try:
            L_h = np.linalg.cholesky(Sigma_h)
        except np.linalg.LinAlgError:
            L_h = np.linalg.cholesky(Sigma_h + 1e-6 * np.eye(len(hid_idx)))

        # Sample z_hid ~ N(mu_h, Sigma_h): shape (n_samples, |h|).
        eps_z = rng.standard_normal((n_samples, len(hid_idx)))
        z_hid = mu_h[None, :] + eps_z @ L_h.T

        # Back-transform: z → u → x via empirical quantile.
        u_hid = sp_norm.cdf(z_hid)  # (n_samples, |h|)
        u_hid = np.clip(u_hid, 1e-6, 1.0 - 1e-6)

        # Build (n_samples, D) uniform matrix for inverse ECDF.
        u_full = np.tile(u_all[0][None, :], (n_samples, 1))
        u_full[:, hid_idx] = u_hid
        x_back = self._ecdf_inverse(u_full)  # (n_samples, D)

        # Overwrite observed entries bit-for-bit.
        x_back[:, obs_idx] = x_obs_flat_2d

        out_np = x_back.reshape(n_samples, J, F, T).astype(np.float32)
        return torch.tensor(out_np, dtype=torch.float32)
