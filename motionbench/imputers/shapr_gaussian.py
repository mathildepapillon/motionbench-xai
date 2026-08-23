"""motionbench.imputers.shapr_gaussian — Fitted Gaussian conditional imputer.

The parametric Gaussian approach of Aas, Jullum & Løland (2021) §3.2, as
implemented by the ``shapr`` R package: assume the *flattened* feature vector
``x ∈ R^D`` (``D = J*F*T``) is jointly Gaussian, estimate its mean and
covariance from a training pool, and draw the hidden block of each coalition
from the resulting analytic conditional

.. math::

    x_{hid} \\mid x_{obs} \\sim N\\bigl(\\mu_h + W (x_{obs} - \\mu_o),\\;
    \\Sigma_{hh} - W \\Sigma_{oh}\\bigr),
    \\qquad W = \\Sigma_{ho} \\Sigma_{oo}^{-1}.

This is the release's **KS-Gauss** method: a closed-form conditional imputer
with no learned components.  It sits between the benchmark's oracle imputer
(exact conditional with the true Sigma) and its nonparametric rows (empirical
donors, VAEAC, flow) — the parametric form is correct for the Gaussian
families and wrong for the Burr families, and in both cases the covariance is
*estimated* rather than known.

Deviation from the paper's recipe, deliberate and recorded: the covariance is
a Ledoit-Wolf shrinkage estimate rather than the plain maximum-likelihood one.
The imputer-training pool (N = 1000 per family, RESOLUTIONS.md §8) is smaller
than the flattened dimension ``D = J*F*T`` (240 to 816), so the ML estimate is
singular and the analytic conditional is undefined.  Ledoit-Wolf is the same
shrinkage :class:`~motionbench.imputers.empirical.EmpiricalConditionalImputer`
already uses for its per-coalition Mahalanobis metric, so the choice adds no
new machinery.

Protocol (pinned to the independent validation study's ``ks_shapr`` row,
which the paper reports as KS-Gauss): fit on the family's imputer-training
pool (N = 1000 fresh draws at seed 99 — *not* the evaluation set), 5
completions per coalition, conditional grading game.  See
``configs/methods/kernelshap_gauss.yaml``.

References
----------
Aas, K., Jullum, M., & Løland, A. (2021).
    Explaining individual predictions when features are dependent:
    More accurate approximations to Shapley values.
    Artificial Intelligence 298, 103502 (arXiv:1903.10464), §3.2.

Ledoit, O., & Wolf, M. (2004).
    A well-conditioned estimator for large-dimensional covariance matrices.
    Journal of Multivariate Analysis 88(2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import torch
from sklearn.covariance import LedoitWolf
from torch import Tensor

from motionbench.imputers.base import BaseImputer
from motionbench.imputers.empirical import _collect_pool

if TYPE_CHECKING:
    from motionbench.data.base import BaseDataset

_F64 = npt.NDArray[np.float64]
_CondParams = tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], _F64, _F64]

__all__ = ["ShaprGaussianImputer"]


class ShaprGaussianImputer(BaseImputer):
    """Analytic Gaussian conditional with an estimated mean and covariance.

    Implements the ``shapr``-style parametric Gaussian imputer (Aas et al.
    2021 §3.2) on the flattened ``D = J*F*T`` feature vector:

    1. **Fit** — estimate ``mu`` (sample mean) and ``Sigma`` (Ledoit-Wolf
       shrinkage covariance by default) from the training pool.
    2. **Impute** — for each coalition, form the exact Gaussian conditional
       of the hidden block given the observed block and draw ``n_samples``
       completions from it.  The conditional parameters ``(W, chol(Sigma_c))``
       are computed once per observation pattern and cached.

    Edge cases: a fully-observed mask returns copies of ``x_obs``; a
    fully-hidden mask draws from the unconditional ``N(mu, Sigma)``.
    Observed entries are preserved bit-for-bit in every returned sample.

    Args:
        shrinkage: Covariance estimator — ``"ledoit_wolf"`` (default; required
            whenever the pool size is below ``D``) or ``"ml"`` (plain sample
            covariance, matching the paper's recipe exactly).

    Raises:
        ValueError: if ``shrinkage`` is not one of the two estimators.
    """

    def __init__(self, shrinkage: str = "ledoit_wolf") -> None:
        """Initialise with the covariance shrinkage estimator."""
        if shrinkage not in ("ledoit_wolf", "ml"):
            raise ValueError(f"shrinkage must be 'ledoit_wolf' or 'ml'; got {shrinkage!r}.")
        self.shrinkage = shrinkage
        self.shrinkage_coef: float = 0.0
        self._mu: _F64 | None = None
        self._cov: _F64 | None = None
        self._l_uncond: _F64 | None = None
        self._shape: tuple[int, int, int] | None = None
        # Conditional params cached per observation pattern (packed mask bytes).
        self._cache: dict[bytes, _CondParams] = {}

    @property
    def is_on_manifold(self) -> bool:
        """Parametric conditional model of the data distribution.

        Returns:
            ``True``.
        """
        return True

    def fit(self, train_data: BaseDataset) -> ShaprGaussianImputer:
        """Fit the flattened Gaussian ``N(mu, Sigma)`` on a training dataset.

        The KS-Gauss protocol fits on the family's *imputer-training* pool
        (N = 1000 fresh draws at seed 99), not on the evaluation set; the
        ``player_eval`` pipeline arranges this via the method config's
        ``fit_data`` override (see ``configs/methods/kernelshap_gauss.yaml``).

        Args:
            train_data: Dataset with ``__len__`` / ``__getitem__`` returning
                ``(x, label)`` where ``x`` is a ``(J, F, T)`` float Tensor.

        Returns:
            ``self`` for method chaining.
        """
        return self.fit_pool(_collect_pool(train_data))

    def fit_pool(self, pool: npt.NDArray[np.floating[Any]]) -> ShaprGaussianImputer:
        """Fit the flattened Gaussian ``N(mu, Sigma)`` on a raw sample pool.

        Args:
            pool: ``(N, J, F, T)`` array of training sequences.

        Returns:
            ``self`` for method chaining.

        Raises:
            ValueError: if ``pool`` is not 4-dimensional.
        """
        pool = np.asarray(pool, dtype=np.float64)
        if pool.ndim != 4:
            raise ValueError(f"pool must be (N, J, F, T); got shape {pool.shape}.")
        N, J, F, T = pool.shape
        flat = pool.reshape(N, -1)
        self._mu = flat.mean(axis=0)
        if self.shrinkage == "ledoit_wolf":
            lw = LedoitWolf(assume_centered=False).fit(flat)
            cov = np.asarray(lw.covariance_, dtype=np.float64)
            self.shrinkage_coef = float(lw.shrinkage_)
        else:
            cov = np.cov(flat, rowvar=False)
            self.shrinkage_coef = 0.0
        self._cov = 0.5 * (cov + cov.T)
        self._l_uncond = None
        self._shape = (J, F, T)
        self._cache = {}
        return self

    def _conditional_params(self, mask_flat: npt.NDArray[np.bool_]) -> _CondParams:
        """Conditional-Gaussian parameters for one observation pattern (cached).

        Args:
            mask_flat: ``(D,)`` bool array, ``True`` = observed.  Must have at
                least one observed and one hidden coordinate.

        Returns:
            ``(obs, hid, W, L)`` — observed / hidden flat indices, the
            conditional-mean weight matrix ``W = Sigma_ho Sigma_oo^{-1}`` and
            the Cholesky factor ``L`` of the (jittered) conditional covariance.

        Raises:
            RuntimeError: if the conditional covariance stays non-PD after
                jitter escalation.
        """
        assert self._cov is not None
        key = np.packbits(mask_flat).tobytes()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        obs = np.flatnonzero(mask_flat)
        hid = np.flatnonzero(~mask_flat)
        Soo = self._cov[np.ix_(obs, obs)]
        Sho = self._cov[np.ix_(hid, obs)]
        Shh = self._cov[np.ix_(hid, hid)]
        jit = 1e-10 * max(float(np.trace(Soo)) / max(len(obs), 1), 1e-12)
        W = np.linalg.solve(Soo + jit * np.eye(len(obs)), Sho.T).T
        Sc = Shh - W @ Sho.T
        Sc = 0.5 * (Sc + Sc.T)
        jit2 = max(1e-10, 1e-8 * float(np.trace(Sc)) / max(len(hid), 1))
        L: _F64 | None = None
        for _ in range(8):
            try:
                L = np.linalg.cholesky(Sc + jit2 * np.eye(len(hid)))
                break
            except np.linalg.LinAlgError:
                jit2 *= 10.0
        if L is None:  # pragma: no cover - PD by construction after shrinkage
            raise RuntimeError(
                "ShaprGaussianImputer: Cholesky failed for the conditional "
                "covariance even after jitter escalation."
            )
        out = (obs, hid, W, L)
        self._cache[key] = out
        return out

    def impute(
        self,
        x_obs: Tensor,
        mask: Tensor,
        n_samples: int,
        seed: int | None = None,
        *,
        generator: np.random.Generator | None = None,
    ) -> Tensor:
        """Draw ``n_samples`` completions from the fitted Gaussian conditional.

        Args:
            x_obs: ``(J, F, T)`` float32 Tensor.  Hidden entries are ignored.
            mask: ``(J, F, T)`` bool Tensor.  ``True`` = observed.
            n_samples: Number of completions to return.
            seed: Optional random seed (ignored when ``generator`` is given).
            generator: Optional numpy Generator advanced *in place* — lets a
                caller thread one stream across successive calls, exactly
                reproducing the validation study's per-sequence RNG protocol.

        Returns:
            ``(n_samples, J, F, T)`` float32 Tensor.  Observed entries are
            preserved bit-for-bit.

        Raises:
            RuntimeError: if ``fit`` has not been called.
            ValueError: if ``x_obs.shape != mask.shape`` or the shape does not
                match the fitted pool.
        """
        if self._mu is None or self._cov is None or self._shape is None:
            raise RuntimeError("ShaprGaussianImputer: call fit() before impute().")
        if x_obs.shape != mask.shape:
            raise ValueError(f"x_obs.shape {x_obs.shape} != mask.shape {mask.shape}.")
        if tuple(x_obs.shape) != self._shape:
            raise ValueError(f"x_obs.shape {tuple(x_obs.shape)} != fitted shape {self._shape}.")

        rng = generator if generator is not None else np.random.default_rng(seed)
        J, F, T = self._shape
        x_np = x_obs.detach().cpu().numpy().astype(np.float64)
        mask_np = mask.detach().cpu().numpy().astype(bool)
        xf = x_np.reshape(-1)
        mf = mask_np.reshape(-1)

        if mf.all():
            out_np = np.tile(x_np[None], (n_samples, 1, 1, 1)).astype(np.float32)
            return torch.tensor(out_np, dtype=torch.float32)
        if not mf.any():
            if self._l_uncond is None:
                self._l_uncond = np.linalg.cholesky(self._cov + 1e-8 * np.eye(len(self._cov)))
            eps = rng.standard_normal((n_samples, len(self._mu))) @ self._l_uncond.T
            draws = self._mu[None] + eps
            return torch.tensor(draws.reshape(n_samples, J, F, T).astype(np.float32))

        obs, hid, W, L = self._conditional_params(mf)
        mu_h = self._mu[hid] + W @ (xf[obs] - self._mu[obs])
        eps = rng.standard_normal((n_samples, len(hid))) @ L.T
        out = np.tile(xf[None], (n_samples, 1))
        out[:, hid] = mu_h[None] + eps
        return torch.tensor(out.reshape(n_samples, J, F, T).astype(np.float32))
