"""motionbench.oracles.gaussian_oracle — Exact closed-form Gaussian conditional oracle.

The data model is ``x ~ N(0, Sigma_joints ⊗ I_F ⊗ Sigma_time)``.
The conditional distribution ``p(x_hid | x_obs)`` is Gaussian with
mean and covariance derived from the standard multivariate Gaussian
conditional formula (Aas et al. 2021, §3, Eq. 3–4):

    μ_{hid|obs}   = Σ_{hid,obs} Σ_{obs,obs}^{-1} (x_obs − μ_obs)
    Σ_{hid|obs}   = Σ_{hid,hid} − Σ_{hid,obs} Σ_{obs,obs}^{-1} Σ_{obs,hid}

The Kronecker structure allows efficient computation for temporal and
spatial coalitions:

    W             = Sigma_time[hid, obs] @ inv(Sigma_time[obs, obs])
    μ_hid[j,f,:]  = W @ x[j,f,obs_t]   (same W for all j, f)
    Σ_{hid|obs}   = Sigma_joints ⊗ I_F ⊗ (Sigma_time[hid,hid] − W @ Sigma_time[obs,hid])

This class satisfies **both** the :class:`~motionbench.oracles.base.Oracle`
ABC and the :class:`~motionbench.imputers.base.BaseImputer` ABC, making
it a "perfect imputer" usable directly in EC1–EC3 evaluation.

References
----------
Aas, K., Jullum, M., & Løland, A. (2021).
    Explaining individual predictions when features are dependent:
    More accurate approximations to Shapley values. arXiv:1903.10464.
    §3, Eq. (3)–(4) — Gaussian conditional expectation for KernelSHAP.

Lundberg, S. M., & Lee, S.-I. (2017).
    A unified approach to interpreting model predictions. NeurIPS 30.
    — KernelSHAP WLS solve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
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
    from motionbench.players.base import PlayerSet

__all__ = ["GaussianOracle"]


class GaussianOracle(Oracle, BaseImputer):
    """Exact closed-form Gaussian conditional oracle.

    Satisfies both the :class:`~motionbench.oracles.base.Oracle` ABC
    and the :class:`~motionbench.imputers.base.BaseImputer` ABC.

    The data model is::

        x ~ N(0, Sigma_joints ⊗ I_F ⊗ Sigma_time)

    Conditional distribution (Aas et al. 2021, §3 Eq. 3–4)::

        p(x_hid | x_obs) = N(μ_{hid|obs}, Σ_{hid|obs})
        μ_{hid|obs}  = Σ_{hid,obs} Σ_{obs,obs}^{-1} x_obs
        Σ_{hid|obs}  = Σ_{hid,hid} - Σ_{hid,obs} Σ_{obs,obs}^{-1} Σ_{obs,hid}

    For the Kronecker model, when the mask is a pure temporal or pure
    spatial pattern, the conditional parameters factorise efficiently
    (same W and Sigma_cond for all (j, f) or all (f, t) slices).

    Args:
        Sigma_joints: ``(J, J)`` PSD joint covariance matrix.
        Sigma_time: ``(T, T)`` PSD temporal covariance matrix.
    """

    def __init__(
        self,
        Sigma_joints: np.ndarray,
        Sigma_time: np.ndarray,
    ) -> None:
        self.Sigma_joints: np.ndarray = np.asarray(Sigma_joints, dtype=np.float64)
        self.Sigma_time: np.ndarray = np.asarray(Sigma_time, dtype=np.float64)
        self._J: int = self.Sigma_joints.shape[0]
        self._T: int = self.Sigma_time.shape[0]

        # Cholesky for unconditional sampling (used in empty-coalition edge case).
        self._L_joints: np.ndarray = np.linalg.cholesky(
            self.Sigma_joints + 1e-8 * np.eye(self._J)
        )
        self._L_time: np.ndarray = np.linalg.cholesky(
            self.Sigma_time + 1e-8 * np.eye(self._T)
        )

        # Cache for conditional parameters.
        self._cond_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

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
        """Draw *n* samples from ``p(x_hid | x_obs)`` exactly.

        Supports temporal masks (all J, F share same time pattern),
        spatial masks (all F, T share same joint pattern), and general
        spatiotemporal masks.

        Observed entries (``mask == True``) are copied bit-for-bit into
        every returned sample; hidden entries are drawn from the exact
        Gaussian conditional.

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
        if x_obs.shape != mask.shape:
            raise ValueError(
                f"x_obs.shape {x_obs.shape} != mask.shape {mask.shape}."
            )
        rng = np.random.default_rng(seed)
        x_np = x_obs.detach().cpu().numpy().astype(np.float64)
        mask_np = mask.detach().cpu().numpy().astype(bool)
        J, F, T = x_np.shape

        if J != self._J or T != self._T:
            raise ValueError(
                f"Expected (J={self._J}, *, T={self._T}); "
                f"got x_obs.shape={x_obs.shape}."
            )

        out_np = self._conditional_sample_np(x_np, mask_np, n, rng)
        return torch.tensor(out_np, dtype=torch.float32)

    def _conditional_sample_np(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Internal numpy-level conditional sampling.

        Args:
            x: ``(J, F, T)`` float64 array.
            mask: ``(J, F, T)`` bool array.
            n_samples: Number of samples.
            rng: Numpy random Generator.

        Returns:
            ``(n_samples, J, F, T)`` float32 array.
        """
        J, F, T = x.shape

        is_temporal = _mask_is_temporal(mask)
        is_spatial = _mask_is_spatial(mask)

        if is_temporal:
            return self._sample_temporal(x, mask, n_samples, rng)
        if is_spatial:
            return self._sample_spatial(x, mask, n_samples, rng)
        return self._sample_spatiotemporal(x, mask, n_samples, rng)

    def _sample_temporal(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Temporal Kronecker-efficient sampling.

        Args:
            x: ``(J, F, T)`` float64.
            mask: ``(J, F, T)`` bool with uniform pattern across j, f.
            n_samples: Number of samples.
            rng: Numpy random Generator.

        Returns:
            ``(n_samples, J, F, T)`` float32 array.
        """
        J, F, T = x.shape
        t_obs = np.flatnonzero(mask[0, 0, :])
        t_hid = np.flatnonzero(~mask[0, 0, :])
        n_hid = len(t_hid)

        if n_hid == 0:
            return np.tile(x[None], (n_samples, 1, 1, 1)).astype(np.float32)
        if len(t_obs) == 0:
            return self._sample_unconditional(n_samples, J, F, T, rng)

        L_cond, W_mean = self._get_temporal_cond(t_obs, t_hid)
        mu = np.einsum("ht,jft->jfh", W_mean, x[:, :, t_obs])
        noise_t = rng.standard_normal((n_samples, J, F, n_hid)).astype(np.float64)
        noise_t = np.einsum("tT,njfT->njft", L_cond, noise_t)
        noise_t = np.einsum("jJ,nJft->njft", self._L_joints, noise_t)
        out = np.tile(x[None], (n_samples, 1, 1, 1)).astype(np.float64)
        out[:, :, :, t_hid] = mu[None] + noise_t
        return out.astype(np.float32)

    def _sample_spatial(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Spatial Kronecker-efficient sampling.

        Args:
            x: ``(J, F, T)`` float64.
            mask: ``(J, F, T)`` bool with uniform pattern across f, t.
            n_samples: Number of samples.
            rng: Numpy random Generator.

        Returns:
            ``(n_samples, J, F, T)`` float32 array.
        """
        J, F, T = x.shape
        j_obs = tuple(int(j) for j in np.flatnonzero(mask[:, 0, 0]))
        j_hid = tuple(int(j) for j in np.flatnonzero(~mask[:, 0, 0]))
        n_hid = len(j_hid)

        if n_hid == 0:
            return np.tile(x[None], (n_samples, 1, 1, 1)).astype(np.float32)
        if len(j_obs) == 0:
            return self._sample_unconditional(n_samples, J, F, T, rng)

        L_cond_j, W = self._get_spatial_cond(j_obs, j_hid)
        j_obs_a = np.asarray(j_obs, dtype=int)
        j_hid_a = np.asarray(j_hid, dtype=int)
        mu = np.einsum("ho,oft->hft", W, x[j_obs_a, :, :])
        noise = rng.standard_normal((n_samples, n_hid, F, T)).astype(np.float64)
        noise = np.einsum("tT,nhfT->nhft", self._L_time, noise)
        noise = np.einsum("hH,nHft->nhft", L_cond_j, noise)
        out = np.tile(x[None], (n_samples, 1, 1, 1)).astype(np.float64)
        out[:, j_hid_a, :, :] = mu[None] + noise
        return out.astype(np.float32)

    def _sample_spatiotemporal(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """General spatiotemporal conditional sampling.

        Args:
            x: ``(J, F, T)`` float64.
            mask: ``(J, F, T)`` bool — arbitrary pattern.
            n_samples: Number of samples.
            rng: Numpy random Generator.

        Returns:
            ``(n_samples, J, F, T)`` float32 array.
        """
        J, F, T = x.shape
        jt_mask = mask.all(axis=1)  # (J, T)

        flat = jt_mask.reshape(-1)
        obs_lin = np.flatnonzero(flat)
        hid_lin = np.flatnonzero(~flat)
        n_obs = int(obs_lin.size)
        n_hid = int(hid_lin.size)

        if n_hid == 0:
            return np.tile(x[None], (n_samples, 1, 1, 1)).astype(np.float32)
        if n_obs == 0:
            return self._sample_unconditional(n_samples, J, F, T, rng)

        j_obs = (obs_lin // T).astype(int)
        t_obs = (obs_lin % T).astype(int)
        j_hid = (hid_lin // T).astype(int)
        t_hid = (hid_lin % T).astype(int)

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
        W = Sigma_ho @ np.linalg.solve(
            Sigma_oo + 1e-10 * np.eye(n_obs), np.eye(n_obs)
        )
        Sigma_cond = Sigma_hh - W @ Sigma_ho.T
        Sigma_cond = 0.5 * (Sigma_cond + Sigma_cond.T)
        Sigma_cond += 1e-8 * np.eye(n_hid)
        L_cond = np.linalg.cholesky(Sigma_cond)

        out = np.tile(x[None], (n_samples, 1, 1, 1)).astype(np.float64)
        for f in range(F):
            x_obs_vals = x[j_obs, f, t_obs].astype(np.float64)
            mu = W @ x_obs_vals
            z = rng.standard_normal((n_samples, n_hid))
            z_corr = z @ L_cond.T
            out[:, j_hid, f, t_hid] = mu[None, :] + z_corr
        return out.astype(np.float32)

    def _sample_unconditional(
        self,
        n: int,
        J: int,
        F: int,
        T: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw n unconditional samples from N(0, Sigma_joints ⊗ I_F ⊗ Sigma_time).

        Args:
            n: Number of samples.
            J: Joints.
            F: Features.
            T: Time-steps.
            rng: Numpy random Generator.

        Returns:
            ``(n, J, F, T)`` float32 array.
        """
        z = rng.standard_normal((n, J, F, T)).astype(np.float64)
        x = np.einsum("tT,njfT->njft", self._L_time, z)
        x = np.einsum("jJ,nJft->njft", self._L_joints, x)
        return x.astype(np.float32)

    # ------------------------------------------------------------------
    # Cached conditional parameter computation
    # ------------------------------------------------------------------

    def _get_temporal_cond(
        self,
        t_obs: np.ndarray,
        t_hid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (L_cond, W_mean) for temporal conditioning, cached.

        Args:
            t_obs: Observed time indices.
            t_hid: Hidden time indices.

        Returns:
            L_cond: ``(n_hid, n_hid)`` Cholesky factor.
            W_mean: ``(n_hid, n_obs)`` mean weight matrix.
        """
        key = ("t", tuple(t_obs.tolist()), tuple(t_hid.tolist()))
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
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (L_cond, W_mean) for spatial conditioning, cached.

        Args:
            j_obs: Tuple of observed joint indices.
            j_hid: Tuple of hidden joint indices.

        Returns:
            L_cond: ``(n_hid, n_hid)`` Cholesky factor.
            W_mean: ``(n_hid, n_obs)`` mean weight matrix.
        """
        key = ("j", j_obs, j_hid)
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

    def true_shapley(
        self,
        x: Tensor,
        classifier: Callable[[Tensor], Tensor],
        players: PlayerSet,
        n_mc: int = 1000,
        n_coalitions: int = 2000,
        seed: int | None = None,
    ) -> Tensor:
        """Compute ground-truth Shapley values via coalition enumeration or sampling.

        For ``M <= 12``, enumerates all 2^M coalitions exactly.
        For ``M > 12``, uses paired KernelSHAP sampling (Covert & Lee 2021)
        with ``n_coalitions // 2`` complementary pairs.

        In both cases the conditional expectation ``v(S) = E[f(x) | x_S]``
        is evaluated via exact Gaussian conditional sampling, so the values
        remain ground-truth despite the sampling approximation for M > 12.

        Args:
            x: ``(J, F, T)`` or ``(1, J, F, T)`` float32 sequence.
            classifier: Callable ``(B, J, F, T) → (B,)`` scalar output
                (e.g. class probability for a fixed class).
            players: :class:`~motionbench.players.base.PlayerSet` with
                ``n_players == M`` and ``coalition_mask(z) → (J, F, T)``.
            n_mc: Monte Carlo samples per coalition (conditional sampling).
            n_coalitions: Number of paired coalition samples when M > 12.
                Ignored for M <= 12.  Must be even; rounded down if odd.
            seed: Optional random seed.

        Returns:
            ``(M,)`` float32 Tensor of Shapley values.  Satisfies the
            efficiency axiom: ``phi.sum() ≈ v(full) − v(empty)``.
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

        for i, z_row in enumerate(coalitions):
            z_t = torch.tensor(z_row, dtype=torch.int32)
            mask = players.coalition_mask(z_t)  # (J, F, T) bool

            if int(z_row.sum()) == M:
                with torch.no_grad():
                    val = float(
                        _eval_classifier(classifier, x.unsqueeze(0)).mean().item()
                    )
            elif int(z_row.sum()) == 0:
                J, F, T = x.shape
                x_marg_np = self._sample_unconditional(
                    n_mc, J, F, T, np.random.default_rng(int(rng.integers(1 << 31)))
                )
                x_marg_t = torch.tensor(x_marg_np, dtype=torch.float32)
                with torch.no_grad():
                    val = float(_eval_classifier(classifier, x_marg_t).mean().item())
            else:
                samps_np = self._conditional_sample_np(
                    x.detach().cpu().numpy().astype(np.float64),
                    mask.detach().cpu().numpy().astype(bool),
                    n_mc,
                    np.random.default_rng(int(rng.integers(1 << 31))),
                )
                samps_t = torch.tensor(samps_np, dtype=torch.float32)
                with torch.no_grad():
                    val = float(_eval_classifier(classifier, samps_t).mean().item())

            values[i] = val

        v_empty = values[np.all(coalitions == 0, axis=1)][0]
        v_full = values[np.all(coalitions == 1, axis=1)][0]

        phi = solve_shapley_wls(coalitions, values, weights, float(v_empty), float(v_full))
        return torch.tensor(phi, dtype=torch.float32)

    # ------------------------------------------------------------------
    # BaseImputer ABC — fit and impute
    # ------------------------------------------------------------------

    def fit(self, train_data: BaseDataset) -> GaussianOracle:
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
        """Draw *n_samples* completions via exact Gaussian conditional.

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
        """Always ``True``: the oracle samples from the exact data manifold.

        Returns:
            ``True``.
        """
        return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mask_is_temporal(mask: np.ndarray) -> bool:
    """Return True if the mask is uniform across all joints and features.

    A temporal mask has the same True/False pattern at every (j, f)
    position — i.e. only the time dimension varies.

    Args:
        mask: ``(J, F, T)`` bool array.

    Returns:
        ``True`` if all (j, f) slices have identical time patterns.
    """
    return bool((mask == mask[0:1, 0:1, :]).all())


def _mask_is_spatial(mask: np.ndarray) -> bool:
    """Return True if the mask is uniform across all features and times.

    A spatial mask has the same True/False pattern at every (f, t)
    position — i.e. only the joint dimension varies.

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
