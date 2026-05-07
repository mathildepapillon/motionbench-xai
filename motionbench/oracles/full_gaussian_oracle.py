"""motionbench.oracles.full_gaussian_oracle — Dense full-covariance Gaussian oracle.

Unlike :class:`~motionbench.oracles.gaussian_oracle.GaussianOracle`, which
exploits Kronecker separability to compute conditional distributions
efficiently, this oracle operates on an **arbitrary** (possibly
non-Kronecker) full ``(J*F*T) × (J*F*T)`` covariance matrix.

The conditional distribution ``p(x_hid | x_obs)`` is computed from the
exact Schur complement:

    μ_{hid|obs}  = Σ_{hid,obs} Σ_{obs,obs}^{-1} x_obs
    Σ_{hid|obs}  = Σ_{hid,hid} - Σ_{hid,obs} Σ_{obs,obs}^{-1} Σ_{obs,hid}

The flat index ordering follows C-order: for a ``(J, F, T)`` array,
element ``(j, f, t)`` maps to flat index ``j * F * T + f * T + t``.

Schur complement parameters are cached by the frozenset of observed flat
indices so repeated calls with the same coalition mask avoid redundant
linear solves.

References
----------
Aas, K., Jullum, M., & Løland, A. (2021).
    Explaining individual predictions when features are dependent.
    arXiv:1903.10464.  §3 Eq. (3)–(4).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from torch import Tensor

from motionbench.oracles.base import Oracle
from motionbench.imputers.base import BaseImputer
from motionbench.utils.coalitions import (
    enumerate_coalitions,
    sample_kernelshap_coalitions,
    solve_shapley_wls,
)

if TYPE_CHECKING:
    from motionbench.data.base import BaseDataset
    from motionbench.players.base import PlayerSet

__all__ = ["FullGaussianOracle"]


class FullGaussianOracle(Oracle, BaseImputer):
    """Exact Gaussian oracle for an arbitrary full covariance matrix.

    Works for any data distribution ``x ~ N(0, Sigma_full)`` where
    ``Sigma_full`` is a ``(D, D)`` PSD matrix with ``D = J * F * T``.
    Unlike :class:`GaussianOracle`, there is no assumption of Kronecker
    separability — cross-joint-time interaction terms are handled correctly.

    Conditional sampling uses the standard Schur complement formula.
    Results for each unique coalition mask (set of observed flat indices)
    are cached so the expensive linear solve is performed at most once per
    unique mask pattern.

    Args:
        Sigma_full: ``(D, D)`` PSD covariance matrix with ``D = J * F * T``.
        J: Number of joints.
        F: Number of coordinates per joint.
        T: Number of time steps.
    """

    def __init__(
        self,
        Sigma_full: np.ndarray,
        J: int,
        F: int,
        T: int,
    ) -> None:
        D = J * F * T
        Sigma_full = np.asarray(Sigma_full, dtype=np.float64)
        if Sigma_full.shape != (D, D):
            raise ValueError(
                f"Sigma_full shape {Sigma_full.shape} does not match "
                f"J*F*T = {J}*{F}*{T} = {D}."
            )

        # Symmetrise and check PSD
        Sigma_full = 0.5 * (Sigma_full + Sigma_full.T)
        eig_min = float(np.linalg.eigvalsh(Sigma_full).min())
        if eig_min < -1e-5:
            raise ValueError(
                f"Sigma_full is not PSD (min eigenvalue {eig_min:.3e})."
            )

        self.Sigma_full: np.ndarray = Sigma_full
        self._J = J
        self._F = F
        self._T = T
        self._D = D

        # Cholesky for unconditional sampling (empty-coalition edge case)
        self._L_full: np.ndarray = np.linalg.cholesky(
            Sigma_full + 1e-8 * np.eye(D)
        )

        # Cache: obs_indices_key → (W, L_cond, hid_idx)
        self._cond_cache: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------ #
    # Oracle ABC — conditional sampling                                    #
    # ------------------------------------------------------------------ #

    def conditional_sample(
        self,
        x_obs: Tensor,
        mask: Tensor,
        n: int,
        seed: int | None = None,
    ) -> Tensor:
        """Draw *n* samples from ``p(x_hid | x_obs)`` exactly.

        Args:
            x_obs: ``(J, F, T)`` float32 Tensor.
            mask: ``(J, F, T)`` bool Tensor.  ``True`` = observed.
            n: Number of conditional samples.
            seed: Optional random seed.

        Returns:
            ``(n, J, F, T)`` float32 Tensor.
        """
        if x_obs.shape != mask.shape:
            raise ValueError(
                f"x_obs.shape {x_obs.shape} != mask.shape {mask.shape}."
            )
        rng = np.random.default_rng(seed)
        x_np = x_obs.detach().cpu().numpy().astype(np.float64)
        mask_np = mask.detach().cpu().numpy().astype(bool)
        out_np = self._conditional_sample_np(x_np, mask_np, n, rng)
        return torch.tensor(out_np, dtype=torch.float32)

    def _conditional_sample_np(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Numpy-level conditional sampling using full covariance Schur complement.

        Args:
            x: ``(J, F, T)`` float64 conditioning sequence.
            mask: ``(J, F, T)`` bool array.  ``True`` = observed.
            n_samples: Number of conditional draws.
            rng: Numpy random Generator.

        Returns:
            ``(n_samples, J, F, T)`` float32 array.
        """
        J, F, T = self._J, self._F, self._T

        # Flatten in C-order: flat_idx = j * F * T + f * T + t
        flat_mask = mask.reshape(-1)
        obs_idx = np.flatnonzero(flat_mask)
        hid_idx = np.flatnonzero(~flat_mask)
        n_obs = len(obs_idx)
        n_hid = len(hid_idx)

        if n_hid == 0:
            return np.tile(x[None], (n_samples, 1, 1, 1)).astype(np.float32)
        if n_obs == 0:
            return self._sample_unconditional(n_samples, J, F, T, rng)

        # Retrieve or compute Schur complement parameters
        W, L_cond = self._get_cond_params(obs_idx, hid_idx)

        x_flat = x.reshape(-1).astype(np.float64)
        x_obs_vals = x_flat[obs_idx]
        mu = W @ x_obs_vals  # (n_hid,)

        z = rng.standard_normal((n_samples, n_hid))
        samples_hid = mu[None, :] + z @ L_cond.T  # (n_samples, n_hid)

        # Build output: copy observed entries, fill hidden entries
        out = np.tile(x_flat[None, :], (n_samples, 1)).astype(np.float64)
        out[:, hid_idx] = samples_hid
        return out.reshape(n_samples, J, F, T).astype(np.float32)

    def _get_cond_params(
        self,
        obs_idx: np.ndarray,
        hid_idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return cached (W, L_cond) for the given observation pattern.

        Args:
            obs_idx: 1-D int array of observed flat indices.
            hid_idx: 1-D int array of hidden flat indices.

        Returns:
            W: ``(n_hid, n_obs)`` conditional mean weight matrix.
            L_cond: ``(n_hid, n_hid)`` Cholesky of conditional covariance.
        """
        key = tuple(obs_idx.tolist())
        if key not in self._cond_cache:
            n_obs = len(obs_idx)
            n_hid = len(hid_idx)
            Soo = self.Sigma_full[np.ix_(obs_idx, obs_idx)]
            Shh = self.Sigma_full[np.ix_(hid_idx, hid_idx)]
            Sho = self.Sigma_full[np.ix_(hid_idx, obs_idx)]
            W = Sho @ np.linalg.solve(
                Soo + 1e-10 * np.eye(n_obs), np.eye(n_obs)
            )
            Sc = Shh - W @ Sho.T
            Sc = 0.5 * (Sc + Sc.T) + 1e-8 * np.eye(n_hid)
            L_cond = np.linalg.cholesky(Sc)
            self._cond_cache[key] = (W, L_cond)
        return self._cond_cache[key]

    def _sample_unconditional(
        self,
        n: int,
        J: int,
        F: int,
        T: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw *n* unconditional samples from ``N(0, Sigma_full)``.

        Args:
            n: Number of samples.
            J, F, T: Shape parameters (unused here; stored in self).
            rng: Numpy random Generator.

        Returns:
            ``(n, J, F, T)`` float32 array.
        """
        z = rng.standard_normal((n, self._D)).astype(np.float64)
        x = (z @ self._L_full.T).reshape(n, J, F, T)
        return x.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Oracle ABC — true Shapley values                                     #
    # ------------------------------------------------------------------ #

    def true_shapley(
        self,
        x: Tensor,
        classifier: Callable[[Tensor], Tensor],
        players: "PlayerSet",
        n_mc: int = 50,
        n_coalitions: int = 2000,
        seed: int | None = None,
    ) -> Tensor:
        """Compute ground-truth Shapley values via exact or sampled enumeration.

        For ``M <= 12`` enumerates all 2^M coalitions exactly.  Otherwise
        uses paired KernelSHAP sampling.

        Args:
            x: ``(J, F, T)`` or ``(1, J, F, T)`` float32 sequence.
            classifier: Callable ``(B, J, F, T) → (B,)`` scalar output.
            players: :class:`~motionbench.players.base.PlayerSet` with
                ``n_players == M``.
            n_mc: Monte Carlo samples per coalition.
            n_coalitions: Number of paired coalition samples when M > 12.
            seed: Optional random seed.

        Returns:
            ``(M,)`` float32 Tensor of Shapley values.
        """
        x = x.squeeze(0) if x.ndim == 4 and x.shape[0] == 1 else x
        if x.ndim != 3:
            raise ValueError(f"x must be (J, F, T) or (1, J, F, T); got {x.shape}.")
        M = players.n_players
        rng = np.random.default_rng(seed)

        n_exact = 2 ** M
        use_exact = n_exact <= n_coalitions
        if use_exact:
            coalitions, weights = enumerate_coalitions(M)
        else:
            n_pairs = max(1, n_coalitions // 2)
            inner_coalitions, inner_weights = sample_kernelshap_coalitions(M, n_pairs, rng)
            boundary_z = np.array([[0] * M, [1] * M], dtype=int)
            boundary_w = np.zeros(2, dtype=np.float64)
            coalitions = np.vstack([boundary_z, inner_coalitions])
            weights = np.concatenate([boundary_w, inner_weights])

        J, F, T = self._J, self._F, self._T
        x_np = x.detach().cpu().numpy().astype(np.float64)
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
                x_marg_np = self._sample_unconditional(
                    n_mc, J, F, T,
                    np.random.default_rng(int(rng.integers(1 << 31))),
                )
                x_marg_t = torch.tensor(x_marg_np, dtype=torch.float32)
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

        v_empty = values[np.all(coalitions == 0, axis=1)][0]
        v_full = values[np.all(coalitions == 1, axis=1)][0]

        phi = solve_shapley_wls(coalitions, values, weights, float(v_empty), float(v_full))
        return torch.tensor(phi, dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # BaseImputer ABC                                                      #
    # ------------------------------------------------------------------ #

    def fit(self, train_data: "BaseDataset") -> "FullGaussianOracle":
        """No-op: oracle requires no training."""
        return self

    def impute(
        self,
        x_obs: Tensor,
        mask: Tensor,
        n_samples: int,
        seed: int | None = None,
    ) -> Tensor:
        """Draw *n_samples* completions via exact Gaussian conditional."""
        return self.conditional_sample(x_obs, mask, n_samples, seed=seed)

    @property
    def is_on_manifold(self) -> bool:
        """Always ``True``: samples from exact data manifold."""
        return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _eval_classifier(
    classifier_fn: Callable[[Tensor], Tensor],
    x: Tensor,
    chunk: int = 512,
) -> Tensor:
    """Run classifier in chunks; returns ``(B,)`` float tensor."""
    results = []
    for i in range(0, len(x), chunk):
        out = classifier_fn(x[i : i + chunk])
        if out.ndim > 1:
            raise ValueError(
                "classifier_fn must return a 1-D tensor of scalars."
            )
        results.append(out.float())
    return torch.cat(results)
