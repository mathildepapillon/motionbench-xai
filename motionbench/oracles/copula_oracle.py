"""motionbench.oracles.copula_oracle — Gaussian copula oracle with pluggable marginals.

This oracle provides exact conditional sampling and ground-truth Shapley
values for the :class:`~motionbench.data.synthetic.burr_motion.BurrMotionBenchmark`
generative model.

ALGORITHM
---------
Given the Gaussian copula data model::

    z ~ N(0, Σ_joints ⊗ I_F ⊗ Σ_time)
    x[j, f, t] = F⁻¹(Φ(z[j, f, t]))

the exact conditional ``p(x_hid | x_obs)`` is sampled in three steps
(Aas et al. 2021, §3.4, copula section):

1. **To latent space** (Eq. copula-forward)::

       z_obs[i] = Φ⁻¹(F_i(x_obs[i]))

   This is the copula's Rosenblatt transform; for each observed coordinate
   it maps the observed value to a standard normal latent value.

2. **Gaussian conditional** (Aas et al. 2021, §3 Eq. 3–4)::

       z_hid ~ N(μ_{hid|obs},  Σ_{hid|obs})
       μ_{hid|obs}   = Σ_{hid,obs} Σ_{obs,obs}^{-1} z_obs
       Σ_{hid|obs}   = Σ_{hid,hid} − Σ_{hid,obs} Σ_{obs,obs}^{-1} Σ_{obs,hid}

   The Kronecker structure (Σ_joints ⊗ I_F ⊗ Σ_time) is exploited for
   efficient temporal-only and spatial-only mask patterns.

3. **Back to original marginals** (Eq. copula-inverse)::

       x_hid[i] = F_i⁻¹(Φ(z_hid[i]))

4. **Restore observed** — observed entries are copied bit-for-bit from
   ``x_obs`` to remove any round-trip numerical drift.

Satisfies both Oracle and BaseImputer ABCs
------------------------------------------
``CopulaOracle`` implements :class:`~motionbench.oracles.base.Oracle` for
ground-truth Shapley computation and :class:`~motionbench.imputers.base.BaseImputer`
for use as a "perfect imputer" in EC1–EC3 evaluation.

NUMERICAL STABILITY
-------------------
* All probabilities are clipped to [1e-9, 1−1e-9] before ``ndtri`` (Φ⁻¹).
* Conditional covariances are symmetrised and ridge-regularised (1e-8 I).
* The Cholesky decomposition is used throughout for numerical stability.
* Observed entries are always restored from x_obs after sampling to prevent
  round-trip drift.

SHAPLEY VALUES
--------------
For M ≤ 12 players, all 2^M coalitions are enumerated exactly.
For M > 12, paired KernelSHAP sampling (Covert & Lee 2021) is used with
``n_coalitions`` complementary pairs.  In both cases the value function is
computed via this oracle's exact conditional sampling.

References
----------
Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press.
    §2 — Copula transform identity (Sklar's theorem, Rosenblatt transform).

Aas, K., Jullum, M., & Løland, A. (2021).
    "Explaining individual predictions when features are dependent."
    arXiv:1903.10464.  §3 Eq. (3)–(4) — Gaussian conditional; §3.4 —
    copula-based conditional expectation for KernelSHAP.

Covert, I., & Lee, S.-I. (2021).
    "Improving KernelSHAP: Practical streamlined Monte Carlo with
    application to economic networks." AISTATS.
    — Paired complementary coalition sampling.

Lundberg, S. M., & Lee, S.-I. (2017).
    "A unified approach to interpreting model predictions." NeurIPS 30.
    — KernelSHAP WLS solve, Shapley kernel weight.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch
from scipy.special import ndtr, ndtri  # Φ, Φ⁻¹
from torch import Tensor

from motionbench.imputers.base import BaseImputer
from motionbench.oracles.base import Oracle
from motionbench.utils.coalitions import (
    enumerate_coalitions,
    sample_kernelshap_coalitions,
    solve_shapley_wls,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from motionbench.data.base import BaseDataset
    from motionbench.data.synthetic.burr_motion import Marginal
    from motionbench.players.base import PlayerSet

__all__ = ["CopulaOracle"]

# Clip bound for probabilities fed to Φ⁻¹.
_EPS: float = 1e-9


class CopulaOracle(Oracle, BaseImputer):
    """Gaussian copula oracle with pluggable marginals.

    Provides exact conditional sampling and ground-truth Shapley values for
    the Gaussian copula data model::

        x[j, f, t] = F⁻¹(Φ(z[j, f, t])),   z ~ N(0, Σ_joints ⊗ I_F ⊗ Σ_time)

    where ``F`` is the pluggable :class:`~motionbench.data.synthetic.burr_motion.Marginal`.

    Satisfies both the :class:`~motionbench.oracles.base.Oracle` ABC (for
    ground-truth Shapley computation) and the
    :class:`~motionbench.imputers.base.BaseImputer` ABC (for use as a
    "perfect imputer" in EC1–EC3 pipeline evaluation).

    Args:
        Sigma_joints: ``(J, J)`` PSD joint correlation matrix.
            Must have unit diagonal (correlation matrix) so that z has
            unit marginal variance and Φ(z) ~ U(0, 1) element-wise.
        Sigma_time: ``(T, T)`` PSD temporal correlation matrix.
            Same requirement as ``Sigma_joints``.
        marginal: Univariate marginal distribution shared by all coordinates.
            Defaults to ``BurrXII(c=2, k=2)`` if ``None``.
    """

    def __init__(
        self,
        Sigma_joints: npt.NDArray[np.float64],
        Sigma_time: npt.NDArray[np.float64],
        marginal: Marginal | None = None,
    ) -> None:
        from motionbench.data.synthetic.burr_motion import BurrXII  # noqa: PLC0415

        self.Sigma_joints: npt.NDArray[np.float64] = np.asarray(Sigma_joints, dtype=np.float64)
        self.Sigma_time: npt.NDArray[np.float64] = np.asarray(Sigma_time, dtype=np.float64)
        self.marginal: Marginal = marginal if marginal is not None else BurrXII(2.0, 2.0)
        self._J: int = self.Sigma_joints.shape[0]
        self._T: int = self.Sigma_time.shape[0]

        # Cholesky for unconditional sampling (empty coalition edge case).
        self._L_joints: npt.NDArray[np.float64] = np.asarray(np.linalg.cholesky(
            self.Sigma_joints + 1e-8 * np.eye(self._J)), dtype=np.float64
        )
        self._L_time: npt.NDArray[np.float64] = np.asarray(np.linalg.cholesky(
            self.Sigma_time + 1e-8 * np.eye(self._T)), dtype=np.float64
        )

        # Cache for conditional parameters keyed by mask pattern.
        self._cond_cache: dict[tuple[object, ...], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}

    # ------------------------------------------------------------------
    # Copula transforms
    # ------------------------------------------------------------------

    def _x_to_z(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Forward copula transform: x → z (observed space → latent Gaussian).

        Implements Aas et al. (2021) §3.4 Eq. (copula-forward)::

            z[i] = Φ⁻¹(F_i(x[i]))

        The probability F_i(x[i]) is clipped to [ε, 1−ε] before Φ⁻¹ to
        prevent divergence at the distribution tails.

        Args:
            x: Array of any shape in the observed (marginal) space.

        Returns:
            Latent Gaussian values, same shape as ``x``.
        """
        u = np.clip(self.marginal.cdf(np.asarray(x, dtype=np.float64)), _EPS, 1.0 - _EPS)
        return np.asarray(ndtri(u), dtype=np.float64)

    def _z_to_x(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Inverse copula transform: z → x (latent Gaussian → observed space).

        Implements Aas et al. (2021) §3.4 Eq. (copula-inverse)::

            x[i] = F_i⁻¹(Φ(z[i]))

        The Gaussian CDF Φ(z) is clipped to [ε, 1−ε] before the marginal
        quantile to prevent extreme quantile values.

        Args:
            z: Latent Gaussian values, any shape.

        Returns:
            Observed-space values, same shape as ``z``.
        """
        u = np.clip(ndtr(np.asarray(z, dtype=np.float64)), _EPS, 1.0 - _EPS)
        return self.marginal.quantile(u)

    # ------------------------------------------------------------------
    # Oracle ABC — conditional sampling
    # ------------------------------------------------------------------

    def conditional_sample(
        self,
        x_obs: Tensor,
        mask: Tensor,
        n: int,
        seed: int | None = None,
    ) -> Tensor:
        """Draw *n* samples from ``p(x_hid | x_obs)`` via copula inversion.

        The three-step algorithm (Aas et al. 2021, §3.4):

        1. ``z_obs = Φ⁻¹(F(x_obs))`` — forward copula transform.
        2. ``z_hid ~ N(μ_{hid|obs}, Σ_{hid|obs})`` — exact Gaussian conditional.
        3. ``x_hid = F⁻¹(Φ(z_hid))`` — inverse copula transform.
        4. Restore observed entries bit-for-bit from ``x_obs``.

        Observed entries (``mask == True``) are guaranteed to be preserved
        exactly in every returned sample (step 4 overrides any round-trip drift).

        Args:
            x_obs: ``(J, F, T)`` float32 Tensor.  Entries where
                ``mask == False`` may be arbitrary; they are ignored.
            mask: ``(J, F, T)`` bool Tensor.  ``True`` = observed.
            n: Number of conditional samples to draw.
            seed: Optional random seed.

        Returns:
            ``(n, J, F, T)`` float32 Tensor.

        Raises:
            ValueError: if ``x_obs.shape != mask.shape``.
        """
        rng = np.random.default_rng(seed)
        x_np = x_obs.detach().cpu().numpy().astype(np.float64)
        mask_np = mask.detach().cpu().numpy().astype(bool)
        J, F, T = x_np.shape

        # Broadcast mask to the full (J, F, T) shape if the player set returned
        # a smaller mask (e.g. (1, 1, T) for temporal players).
        if mask_np.shape != x_np.shape:
            mask_np = np.broadcast_to(mask_np, x_np.shape).copy()

        if J != self._J or T != self._T:
            raise ValueError(
                f"Expected (J={self._J}, *, T={self._T}); got x_obs.shape={x_obs.shape}."
            )

        out_np = self._conditional_sample_np(x_np, mask_np, n, rng)
        return torch.tensor(out_np, dtype=torch.float32)

    def _conditional_sample_np(
        self,
        x: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool_],
        n_samples: int,
        rng: np.random.Generator,
    ) -> npt.NDArray[np.float32]:
        """Internal numpy-level copula conditional sampling.

        Args:
            x: ``(J, F, T)`` float64 conditioning sequence.
            mask: ``(J, F, T)`` bool mask.
            n_samples: Number of samples.
            rng: Numpy random Generator.

        Returns:
            ``(n_samples, J, F, T)`` float32 array.
        """
        # Expand mask to the full data shape (J, F, T) if a player set returned
        # a broadcastable mask (e.g. (1, 1, T) from temporal players).
        if mask.shape != x.shape:
            mask = np.broadcast_to(mask, x.shape).copy()

        # Step 1: forward copula transform to latent Gaussian.
        z = self._x_to_z(x)  # (J, F, T)

        # Step 2: Gaussian conditional in latent space.
        is_temporal = _mask_is_temporal(mask)
        is_spatial = _mask_is_spatial(mask)

        if is_temporal:
            z_completions = self._sample_temporal(z, mask, n_samples, rng)
        elif is_spatial:
            z_completions = self._sample_spatial(z, mask, n_samples, rng)
        else:
            z_completions = self._sample_spatiotemporal(z, mask, n_samples, rng)
        # z_completions: (n_samples, J, F, T) latent Gaussian

        # Step 3: inverse copula transform back to observed space.
        x_completions = self._z_to_x(z_completions).astype(np.float32)

        # Step 4: restore observed entries bit-for-bit from x_obs to remove
        # any round-trip numerical drift from the x → z → x transform.
        x_completions[:, mask] = x[mask].astype(np.float32)

        return x_completions

    # ------------------------------------------------------------------
    # Gaussian conditional helpers (same Kronecker structure as GaussianOracle)
    # ------------------------------------------------------------------

    def _sample_temporal(
        self,
        z: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool_],
        n_samples: int,
        rng: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        """Kronecker-efficient temporal conditional sampling in latent space.

        For a pure temporal mask (same pattern across all j, f), the
        conditional parameters factor across joints and features:

            W             = Σ_time[hid, obs] Σ_time[obs, obs]^{-1}
            μ[j,f,hid]    = W @ z[j, f, obs]   (same W for all j, f)
            Σ_{hid|obs}   = Σ_joints ⊗ I_F ⊗ (Σ_time[hid,hid] − W Σ_time[obs,hid])

        Args:
            z: ``(J, F, T)`` float64 latent Gaussian.
            mask: ``(J, F, T)`` bool, uniform across j and f.
            n_samples: Number of samples.
            rng: Numpy random Generator.

        Returns:
            ``(n_samples, J, F, T)`` float64 array.
        """
        J, F, T = z.shape
        t_obs = np.flatnonzero(mask[0, 0, :])
        t_hid = np.flatnonzero(~mask[0, 0, :])
        n_hid = len(t_hid)

        if n_hid == 0:
            return np.tile(z[None], (n_samples, 1, 1, 1))
        if len(t_obs) == 0:
            return self._sample_unconditional_latent(n_samples, J, F, T, rng)

        L_cond, W_mean = self._get_temporal_cond(t_obs, t_hid)
        mu = np.einsum("ht,jft->jfh", W_mean, z[:, :, t_obs])

        # Noise with covariance Σ_joints ⊗ I_F ⊗ Σ_{hid|obs}(temporal).
        noise = rng.standard_normal((n_samples, J, F, n_hid))
        noise = np.einsum("tT,njfT->njft", L_cond, noise)
        noise = np.einsum("jJ,nJft->njft", self._L_joints, noise)

        out = np.tile(z[None], (n_samples, 1, 1, 1)).astype(np.float64)
        out[:, :, :, t_hid] = mu[None] + noise
        return out

    def _sample_spatial(
        self,
        z: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool_],
        n_samples: int,
        rng: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        """Kronecker-efficient spatial conditional sampling in latent space.

        For a pure spatial mask (same pattern across all f, t):

            W             = Σ_joints[hid, obs] Σ_joints[obs, obs]^{-1}
            μ[hid,f,t]    = W @ z[obs, f, t]  (same W for all f, t)
            Σ_{hid|obs}   = (Σ_joints[hid,hid] − W Σ_joints[obs,hid]) ⊗ I_F ⊗ Σ_time

        Args:
            z: ``(J, F, T)`` float64 latent Gaussian.
            mask: ``(J, F, T)`` bool, uniform across f and t.
            n_samples: Number of samples.
            rng: Numpy random Generator.

        Returns:
            ``(n_samples, J, F, T)`` float64 array.
        """
        J, F, T = z.shape
        j_obs = tuple(int(j) for j in np.flatnonzero(mask[:, 0, 0]))
        j_hid = tuple(int(j) for j in np.flatnonzero(~mask[:, 0, 0]))
        n_hid = len(j_hid)

        if n_hid == 0:
            return np.tile(z[None], (n_samples, 1, 1, 1))
        if len(j_obs) == 0:
            return self._sample_unconditional_latent(n_samples, J, F, T, rng)

        L_cond_j, W = self._get_spatial_cond(j_obs, j_hid)
        j_obs_a = np.asarray(j_obs, dtype=int)
        j_hid_a = np.asarray(j_hid, dtype=int)
        mu = np.einsum("ho,oft->hft", W, z[j_obs_a, :, :])

        noise = rng.standard_normal((n_samples, n_hid, F, T))
        noise = np.einsum("tT,nhfT->nhft", self._L_time, noise)
        noise = np.einsum("hH,nHft->nhft", L_cond_j, noise)

        out = np.tile(z[None], (n_samples, 1, 1, 1)).astype(np.float64)
        out[:, j_hid_a, :, :] = mu[None] + noise
        return out

    def _sample_spatiotemporal(
        self,
        z: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool_],
        n_samples: int,
        rng: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        """General spatiotemporal conditional sampling in latent space.

        For arbitrary (J, F, T) masks.  Each F-slice is conditionally
        independent (by the I_F term) but shares the same conditional
        parameters from the (J, T) pattern.

        The (J×T)×(J×T) covariance is formed element-wise from the Hadamard
        product of Σ_joints and Σ_time entries (Kronecker structure):

            Σ[(j,t),(j',t')] = Σ_joints[j,j'] * Σ_time[t,t']

        Args:
            z: ``(J, F, T)`` float64 latent Gaussian.
            mask: ``(J, F, T)`` bool — arbitrary pattern.
            n_samples: Number of samples.
            rng: Numpy random Generator.

        Returns:
            ``(n_samples, J, F, T)`` float64 array.
        """
        J, F, T = z.shape
        jt_mask = mask.all(axis=1)  # (J, T)

        flat = jt_mask.reshape(-1)
        obs_lin = np.flatnonzero(flat)
        hid_lin = np.flatnonzero(~flat)
        n_obs = int(obs_lin.size)
        n_hid = int(hid_lin.size)

        if n_hid == 0:
            return np.tile(z[None], (n_samples, 1, 1, 1))
        if n_obs == 0:
            return self._sample_unconditional_latent(n_samples, J, F, T, rng)

        j_obs = (obs_lin // T).astype(int)
        t_obs = (obs_lin % T).astype(int)
        j_hid = (hid_lin // T).astype(int)
        t_hid = (hid_lin % T).astype(int)

        # Hadamard product of Kronecker sub-blocks.
        Sigma_oo = (
            self.Sigma_joints[j_obs[:, None], j_obs[None, :]]
            * self.Sigma_time[t_obs[:, None], t_obs[None, :]]
        )
        Sigma_hh = (
            self.Sigma_joints[j_hid[:, None], j_hid[None, :]]
            * self.Sigma_time[t_hid[:, None], t_hid[None, :]]
        )
        Sigma_ho = (
            self.Sigma_joints[j_hid[:, None], j_obs[None, :]]
            * self.Sigma_time[t_hid[:, None], t_obs[None, :]]
        )
        W = Sigma_ho @ np.linalg.solve(Sigma_oo + 1e-10 * np.eye(n_obs), np.eye(n_obs))
        Sigma_cond = Sigma_hh - W @ Sigma_ho.T
        Sigma_cond = 0.5 * (Sigma_cond + Sigma_cond.T)
        Sigma_cond += 1e-8 * np.eye(n_hid)
        L_cond = np.linalg.cholesky(Sigma_cond)

        out = np.tile(z[None], (n_samples, 1, 1, 1)).astype(np.float64)
        for f in range(F):
            z_obs_vals = z[j_obs, f, t_obs]
            mu = W @ z_obs_vals
            eps = rng.standard_normal((n_samples, n_hid))
            out[:, j_hid, f, t_hid] = mu[None, :] + eps @ L_cond.T
        return out

    def _sample_unconditional_latent(
        self,
        n: int,
        J: int,
        F: int,
        T: int,
        rng: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        """Draw n unconditional latent Gaussian samples.

        Draws z ~ N(0, Σ_joints ⊗ I_F ⊗ Σ_time) using the pre-computed
        Cholesky factors.

        Args:
            n: Number of samples.
            J: Joints.
            F: Features.
            T: Time-steps.
            rng: Numpy random Generator.

        Returns:
            ``(n, J, F, T)`` float64 array.
        """
        eps = rng.standard_normal((n, J, F, T))
        z = np.asarray(np.einsum("tT,njfT->njft", self._L_time, eps), dtype=np.float64)
        z = np.asarray(np.einsum("jJ,nJft->njft", self._L_joints, z), dtype=np.float64)
        return z

    # ------------------------------------------------------------------
    # Cached conditional parameter computation
    # ------------------------------------------------------------------

    def _get_temporal_cond(
        self,
        t_obs: npt.NDArray[np.intp],
        t_hid: npt.NDArray[np.intp],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute (L_cond, W_mean) for temporal conditioning, with caching.

        Temporal conditional parameters (Aas et al. 2021, §3 Eq. 3–4):

            W             = Σ_time[hid, obs] Σ_time[obs, obs]^{-1}
            Σ_{hid|obs}   = Σ_time[hid,hid] − W Σ_time[obs,hid]

        Args:
            t_obs: ``(n_obs,)`` observed time indices.
            t_hid: ``(n_hid,)`` hidden time indices.

        Returns:
            L_cond: ``(n_hid, n_hid)`` lower-triangular Cholesky factor.
            W_mean: ``(n_hid, n_obs)`` conditional-mean weight matrix.
        """
        key: tuple[object, ...] = ("t", tuple(t_obs.tolist()), tuple(t_hid.tolist()))
        if key not in self._cond_cache:
            Soo = self.Sigma_time[np.ix_(t_obs, t_obs)]
            Shh = self.Sigma_time[np.ix_(t_hid, t_hid)]
            Sho = self.Sigma_time[np.ix_(t_hid, t_obs)]
            W = Sho @ np.linalg.solve(Soo + 1e-10 * np.eye(len(t_obs)), np.eye(len(t_obs)))
            Sc = Shh - W @ Sho.T
            Sc = 0.5 * (Sc + Sc.T) + 1e-8 * np.eye(len(t_hid))
            self._cond_cache[key] = (np.linalg.cholesky(Sc), W)
        return self._cond_cache[key]

    def _get_spatial_cond(
        self,
        j_obs: tuple[int, ...],
        j_hid: tuple[int, ...],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute (L_cond, W_mean) for spatial conditioning, with caching.

        Spatial conditional parameters (Aas et al. 2021, §3 Eq. 3–4):

            W             = Σ_joints[hid, obs] Σ_joints[obs, obs]^{-1}
            Σ_{hid|obs}   = Σ_joints[hid,hid] − W Σ_joints[obs,hid]

        Args:
            j_obs: Tuple of observed joint indices.
            j_hid: Tuple of hidden joint indices.

        Returns:
            L_cond: ``(n_hid, n_hid)`` lower-triangular Cholesky factor.
            W_mean: ``(n_hid, n_obs)`` conditional-mean weight matrix.
        """
        key: tuple[object, ...] = ("j", j_obs, j_hid)
        if key not in self._cond_cache:
            j_obs_a = np.asarray(j_obs, dtype=int)
            j_hid_a = np.asarray(j_hid, dtype=int)
            Soo = self.Sigma_joints[np.ix_(j_obs_a, j_obs_a)]
            Shh = self.Sigma_joints[np.ix_(j_hid_a, j_hid_a)]
            Sho = self.Sigma_joints[np.ix_(j_hid_a, j_obs_a)]
            W = Sho @ np.linalg.solve(
                Soo + 1e-10 * np.eye(len(j_obs_a)), np.eye(len(j_obs_a))
            )
            Sc = Shh - W @ Sho.T
            Sc = 0.5 * (Sc + Sc.T) + 1e-8 * np.eye(len(j_hid_a))
            self._cond_cache[key] = (np.linalg.cholesky(Sc), W)
        return self._cond_cache[key]

    # ------------------------------------------------------------------
    # Oracle ABC — true Shapley values
    # ------------------------------------------------------------------

    def true_shapley(  # type: ignore[override]
        self,
        x: Tensor,
        classifier: Callable[[Tensor], Tensor],
        players: PlayerSet,
        n_mc: int = 1000,
        n_coalitions: int = 2000,
        seed: int | None = None,
    ) -> Tensor:
        """Compute ground-truth Shapley values via copula conditional sampling.

        For M ≤ 12, enumerates all 2^M coalitions exactly.
        For M > 12, uses paired KernelSHAP sampling (Covert & Lee 2021) with
        ``n_coalitions // 2`` complementary pairs.

        In both cases, the value function::

            v(S) = E_{x_{\\bar{S}} ~ p(·|x_S)}[f(x_S, x_{\\bar{S}})]

        is estimated by drawing ``n_mc`` samples from the exact copula
        conditional and averaging the classifier output.

        The efficiency axiom ``Σφ ≈ v(N) − v(∅)`` is enforced by the WLS
        solve with high-weight boundary constraints.

        Args:
            x: ``(J, F, T)`` or ``(1, J, F, T)`` float32 sequence.
            classifier: Callable ``(B, J, F, T) → (B,)`` scalar output
                (e.g. class probability for a fixed class).
            players: :class:`~motionbench.players.base.PlayerSet` defining
                the M players and coalition-to-mask expansion.
            n_mc: Monte Carlo samples per coalition for estimating v(S).
            n_coalitions: Number of paired coalition samples when M > 12.
                Ignored for M ≤ 12.  Rounded down to nearest even number.
            seed: Optional random seed.

        Returns:
            ``(M,)`` float32 Tensor of Shapley values.

        Raises:
            ValueError: if ``x`` has unexpected shape.
        """
        x = x.squeeze(0) if x.ndim == 4 and x.shape[0] == 1 else x
        if x.ndim != 3:
            raise ValueError(f"x must be (J, F, T) or (1, J, F, T); got {x.shape}.")
        M = players.n_players

        rng = np.random.default_rng(seed)

        if M <= 12:
            coalitions, weights = enumerate_coalitions(M)
        else:
            n_pairs = max(1, n_coalitions // 2)
            inner_coalitions, inner_weights = sample_kernelshap_coalitions(M, n_pairs, rng)
            boundary_z = np.array([[0] * M, [1] * M], dtype=int)
            boundary_w = np.zeros(2, dtype=np.float64)
            coalitions = np.vstack([boundary_z, inner_coalitions])
            weights = np.concatenate([boundary_w, inner_weights])

        values = np.zeros(len(coalitions), dtype=np.float64)

        x_np = x.detach().cpu().numpy().astype(np.float64)
        J, F, T = x_np.shape

        for i, z_row in enumerate(coalitions):
            z_t = torch.tensor(z_row, dtype=torch.int32)
            mask = players.coalition_mask(z_t)  # (J, F, T) bool

            if int(z_row.sum()) == M:
                with torch.no_grad():
                    val = float(_eval_classifier(classifier, x.unsqueeze(0)).mean().item())
            elif int(z_row.sum()) == 0:
                z_lat = self._sample_unconditional_latent(
                    n_mc, J, F, T, np.random.default_rng(int(rng.integers(1 << 31)))
                )
                x_marg = self._z_to_x(z_lat).astype(np.float32)
                x_marg_t = torch.tensor(x_marg, dtype=torch.float32)
                with torch.no_grad():
                    val = float(_eval_classifier(classifier, x_marg_t).mean().item())
            else:
                samps_np = self._conditional_sample_np(
                    x_np,
                    mask.detach().cpu().numpy().astype(bool),
                    n_mc,
                    np.random.default_rng(int(rng.integers(1 << 31))),
                )
                samps_t = torch.tensor(samps_np, dtype=torch.float32)
                with torch.no_grad():
                    val = float(_eval_classifier(classifier, samps_t).mean().item())

            values[i] = val

        v_empty = float(values[np.all(coalitions == 0, axis=1)][0])
        v_full = float(values[np.all(coalitions == 1, axis=1)][0])

        phi = solve_shapley_wls(coalitions, values, weights, v_empty, v_full)
        return torch.tensor(phi, dtype=torch.float32)

    # ------------------------------------------------------------------
    # BaseImputer ABC — fit and impute
    # ------------------------------------------------------------------

    def fit(self, train_data: BaseDataset) -> CopulaOracle:
        """No-op: the oracle requires no training.

        Args:
            train_data: Ignored.

        Returns:
            ``self`` for method chaining.
        """
        return self

    def impute(
        self,
        x_obs: Tensor,
        mask: Tensor,
        n_samples: int,
        seed: int | None = None,
    ) -> Tensor:
        """Draw *n_samples* completions via the exact copula conditional.

        Delegates to :meth:`conditional_sample`.  Satisfies the
        :class:`~motionbench.imputers.base.BaseImputer` contract:
        observed entries are preserved bit-for-bit.

        Args:
            x_obs: ``(J, F, T)`` float32 Tensor.
            mask: ``(J, F, T)`` bool Tensor.  ``True`` = observed.
            n_samples: Number of completions to draw.
            seed: Optional random seed.

        Returns:
            ``(n_samples, J, F, T)`` float32 Tensor.
        """
        return self.conditional_sample(x_obs, mask, n_samples, seed=seed)

    # ------------------------------------------------------------------
    # BaseImputer optional overrides
    # ------------------------------------------------------------------

    @property
    def is_on_manifold(self) -> bool:
        """Always ``True``: the oracle samples from the true data manifold.

        Returns:
            ``True``.
        """
        return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mask_is_temporal(mask: npt.NDArray[np.bool_]) -> bool:
    """Return True if the mask pattern is uniform across all joints and features.

    A temporal mask has the same True/False time pattern at every (j, f)
    position, i.e. the observed/hidden status depends only on the time axis.

    Args:
        mask: ``(J, F, T)`` bool array.

    Returns:
        ``True`` if all (j, f) slices have identical time patterns.
    """
    return bool((mask == mask[0:1, 0:1, :]).all())


def _mask_is_spatial(mask: npt.NDArray[np.bool_]) -> bool:
    """Return True if the mask pattern is uniform across all features and times.

    A spatial mask has the same True/False pattern at every (f, t) position,
    i.e. the observed/hidden status depends only on the joint axis.

    Args:
        mask: ``(J, F, T)`` bool array.

    Returns:
        ``True`` if all (f, t) slices have identical joint patterns.
    """
    return bool((mask == mask[:, 0:1, 0:1]).all())


def _eval_classifier(
    classifier_fn: Callable[[Tensor], Tensor],
    x: Tensor,
    chunk: int = 512,
) -> Tensor:
    """Run classifier in chunks to avoid OOM; returns ``(B,)`` float tensor.

    Args:
        classifier_fn: Callable ``(B, J, F, T) → (B,)`` scalar output.
        x: ``(B, J, F, T)`` float32 input batch.
        chunk: Maximum batch size per forward pass.

    Returns:
        ``(B,)`` float32 Tensor.

    Raises:
        ValueError: if ``classifier_fn`` returns a multi-dimensional output.
    """
    results = []
    for i in range(0, len(x), chunk):
        out = classifier_fn(x[i : i + chunk])
        if out.ndim > 1:
            raise ValueError(
                "classifier_fn must return a 1-D tensor of scalars (e.g. a "
                "class probability), not logits."
            )
        results.append(out.float())
    return torch.cat(results)
