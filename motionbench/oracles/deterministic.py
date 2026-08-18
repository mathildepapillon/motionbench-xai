"""motionbench.oracles.deterministic — Deterministic conditional-mean oracles.

Exact grading targets for the f-of-mean value function.  Under the executed
(release) semantics, the estimated game value is ``v(S) = f(mean of R
completions)``.  As ``R -> infinity`` the completion mean converges to the
exact conditional mean, so the game value converges to the deterministic

    conditional game:  v*(S) = f(x_S joined with E[x_hid | x_obs])
    marginal game:     v*(S) = f(x_S joined with E[x])

Both fills are closed-form for the synthetic families in this package and are
computed here **without Monte Carlo**, so a method graded against them incurs
zero grading noise (a perfect attribution scores EC1 = EC3 = 0 exactly).

Conditional fill
----------------
*Gaussian Kronecker* (``x ~ N(0, Sigma_J (x) I_F (x) Sigma_T)``): the F-slices
are independent under ``I_F``, and every player set in this package masks
whole ``(j, t)`` columns (the mask is constant across F).  For each coalition
the conditional mean is the linear map ``W_S = Sigma[hid, obs] @
inv(Sigma[obs, obs])`` on the flattened ``(J*T)`` index with ``Sigma =
kron(Sigma_J, Sigma_T)``; ``W_S`` is precomputed **once per coalition** and
applied to every sequence.

*Gaussian copula (Burr XII and friends)*: the same conditional applies in the
latent z-space per coordinate — each hidden z-coordinate given the observed
columns is univariate normal ``N(mu_z, sd^2)`` — and the observed-space
conditional mean is the 1-D integral of ``quantile(Phi(z))`` against that
normal.  The symmetric Burr XII quantile has a square-root cusp at ``u = 1/2``
(i.e. at ``t0 = -mu/sd``) which caps any global quadrature rule at an
algebraic rate, so the integral is split at the cusp and evaluated with the
substitution ``t = t0 -/+ y^2`` on each side, restoring spectral accuracy for
Gauss-Legendre (:func:`cusp_split_nodes`).

Marginal fill
-------------
``E[x] = 0`` exactly for every family here: the Gaussian field is centered and
the symmetric marginals are symmetric about 0.  The marginal deterministic
fill is therefore the zero fill (:func:`marginal_fill`), identical to
KS-Zero's fill by construction.

Ported from the independent validation study's implementation
(``rb.detoracle`` temporal-mask reference and the player-set sweep's
general-mask ``DetCondFill``); kept numerically identical so the two produce
bit-identical fills on the same inputs.

Example
-------
>>> from motionbench.attribution.sampled_coalitions import sampled_coalition_set
>>> Z, w = sampled_coalition_set(players.n_players, budget=1024)
>>> det = DeterministicConditionalOracle.from_oracle(dataset.oracle, players, Z)
>>> fills = det.fill_all(x_np)          # (len(Z), J, F, T) — no randomness
>>> v_star = clf_fn(fills)              # deterministic conditional game values
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch
from scipy.special import ndtr, ndtri

if TYPE_CHECKING:
    from motionbench.data.synthetic.burr_motion import Marginal
    from motionbench.players.base import PlayerSet

__all__ = [
    "GL_NODES",
    "cusp_split_nodes",
    "DeterministicConditionalOracle",
    "marginal_fill",
]

# Quadrature defaults for the copula fill: E[q(Phi(m + s t))] against the
# standard normal in t, truncated at |t| = 13 (tail mass ~1e-38 while the
# clipped quantile is ~1e3, so the truncation error is negligible).
GL_NODES: int = 64  # per side; 2 * GL_NODES evaluations per coordinate
_GL_HALF_WIDTH: float = 13.0

# Probability clip fed to ndtri, matching the samplers' convention.
_EPS: float = 1e-9


def cusp_split_nodes(
    mu: npt.NDArray[np.float64],
    sd: npt.NDArray[np.float64],
    n: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-coordinate quadrature nodes/weights for ``E[g(mu + sd * t)]``.

    The integrand ``g`` is assumed to have a (square-root) cusp at
    ``t0 = -mu / sd`` — the pre-image of the symmetric marginal's median —
    which would cap any global rule at an algebraic convergence rate.  The
    integral against the standard normal density over ``|t| <= 13`` is split
    at ``t0`` and each side is mapped through ``t = t0 -/+ y^2``, which
    removes the cusp and restores Gauss-Legendre's spectral accuracy.

    Ported from the independent validation study's implementation
    (``rb.detoracle._cusp_split_nodes``).

    Args:
        mu: Conditional means; any shape (broadcast-compatible with ``sd``).
        sd: Conditional standard deviations, same shape as ``mu``.
        n: Nodes per side (default :data:`GL_NODES`).

    Returns:
        ``(t, w)`` of shape ``mu.shape + (2n,)``: ``sum_i g(mu + sd * t_i) w_i``
        integrates ``g(mu + sd * t)`` against the standard normal density.
    """
    y, wy = np.polynomial.legendre.leggauss(n or GL_NODES)
    t0 = np.clip(-mu / np.maximum(sd, 1e-300), -_GL_HALF_WIDTH, _GL_HALF_WIDTH)[..., None]
    ts, ws = [], []
    for sign, half in ((-1.0, t0 + _GL_HALF_WIDTH), (1.0, _GL_HALF_WIDTH - t0)):
        c = np.sqrt(np.maximum(half, 0.0))  # y in [0, sqrt(width)]
        yy = 0.5 * c * (y + 1.0)  # map [-1, 1] -> [0, c]
        t = t0 + sign * yy * yy
        w = (0.5 * c * wy) * 2.0 * yy * np.exp(-0.5 * t * t) / np.sqrt(2.0 * np.pi)
        ts.append(t)
        ws.append(w)
    return np.concatenate(ts, axis=-1), np.concatenate(ws, axis=-1)


def marginal_fill(
    x: npt.NDArray[np.float64], mask: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float32]:
    """Deterministic marginal fill: observed entries verbatim, hidden -> E[x] = 0.

    Valid for every synthetic family in this package (centered Gaussian field;
    symmetric marginals), where the unconditional mean is exactly zero.

    Args:
        x: ``(J, F, T)`` sequence.
        mask: ``(J, F, T)`` bool, ``True`` = observed.

    Returns:
        ``(J, F, T)`` float32 fill.
    """
    return np.where(mask, x, 0.0).astype(np.float32)


class DeterministicConditionalOracle:
    """Per-coalition deterministic conditional-mean fills, precomputed once.

    For a fixed coalition design ``Z`` over a :class:`~motionbench.players.base.PlayerSet`,
    precomputes the conditional-mean operator of every coalition at
    construction time; :meth:`fill_all` then applies all operators to one
    sequence with no further linear algebra (and no randomness).

    Supports **general masks** (any player set whose mask is constant across
    the F axis — true for every player set in this package): the operators act
    on the flattened ``(J*T)`` index of ``Sigma = kron(Sigma_J, Sigma_T)``,
    exact per F-slice under the Kronecker model's ``I_F`` factor.

    Two families:

    * ``marginal is None`` — Gaussian Kronecker field.  Hidden coordinates
      get ``W_S @ x_obs`` directly.
    * ``marginal`` given — Gaussian-copula pushforward with that univariate
      marginal (e.g. :class:`~motionbench.data.synthetic.burr_motion.BurrXII`).
      Hidden coordinates get the exact 1-D integral
      ``E[quantile(Phi(z))]``, ``z ~ N(mu_z, sd^2)``, via cusp-split
      Gauss-Legendre quadrature (:func:`cusp_split_nodes`).

    Ported from the independent validation study's implementation (player-set
    sweep ``DetCondFill``).

    Args:
        sigma_joints: ``(J, J)`` joint covariance of the (latent) Gaussian field.
        sigma_time: ``(T, T)`` temporal covariance of the (latent) Gaussian field.
        players: Player set defining the coalition -> mask expansion.
        Z: ``(n_coalitions, M)`` binary coalition matrix (rows = designs to
            precompute; typically from
            :func:`~motionbench.attribution.sampled_coalitions.sampled_coalition_set`).
        marginal: Optional univariate marginal for the copula family; ``None``
            selects the plain Gaussian family.

    Raises:
        ValueError: if a coalition mask is not constant across the F axis.
    """

    def __init__(
        self,
        sigma_joints: npt.NDArray[np.float64],
        sigma_time: npt.NDArray[np.float64],
        players: PlayerSet,
        Z: npt.NDArray[np.integer],
        marginal: Marginal | None = None,
    ) -> None:
        self.players = players
        self.marginal = marginal
        J, F, T = players.shape
        self._J, self._F, self._T = J, F, T
        sigma_joints = np.asarray(sigma_joints, dtype=np.float64)
        sigma_time = np.asarray(sigma_time, dtype=np.float64)
        Sig = np.kron(sigma_joints, sigma_time)  # (J*T, J*T), F-independent
        self.ops: list[
            tuple[
                npt.NDArray[np.intp],
                npt.NDArray[np.intp],
                npt.NDArray[np.float64] | None,
                npt.NDArray[np.float64] | None,
            ]
        ] = []
        for z in np.asarray(Z):
            z_t = torch.as_tensor(np.asarray(z, dtype=np.int64) != 0, dtype=torch.bool)
            mask3 = players.coalition_mask(z_t).cpu().numpy()  # (J, F, T)
            if not bool((mask3 == mask3[:, 0:1, :]).all()):
                raise ValueError(
                    "DeterministicConditionalOracle requires masks constant "
                    "across the F axis (whole (j, t) columns per player)."
                )
            m = mask3[:, 0, :].reshape(J * T)  # F-agnostic
            obs, hid = np.flatnonzero(m), np.flatnonzero(~m)
            if len(hid) == 0 or len(obs) == 0:
                self.ops.append((obs, hid, None, None))
                continue
            Soo = Sig[np.ix_(obs, obs)] + 1e-8 * np.eye(len(obs))
            W = Sig[np.ix_(hid, obs)] @ np.linalg.inv(Soo)
            sd = None
            if marginal is not None:
                Shh = Sig[np.ix_(hid, hid)]
                sd = np.sqrt(
                    np.clip(
                        np.diag(Shh) - np.einsum("ij,ji->i", W, Sig[np.ix_(obs, hid)]),
                        1e-12,
                        None,
                    )
                )
            self.ops.append((obs, hid, W, sd))

    @classmethod
    def from_oracle(
        cls,
        oracle: object,
        players: PlayerSet,
        Z: npt.NDArray[np.integer],
    ) -> DeterministicConditionalOracle:
        """Build from a fitted sampling oracle's covariance structure.

        Accepts a :class:`~motionbench.oracles.gaussian_oracle.GaussianOracle`
        (plain Gaussian family) or a
        :class:`~motionbench.oracles.copula_oracle.CopulaOracle` (copula
        family; its ``marginal`` is reused), e.g. ``dataset.oracle``.

        Args:
            oracle: Object with ``Sigma_joints`` and ``Sigma_time`` attributes
                and, for the copula family, a ``marginal``.
            players: Player set defining the coalition -> mask expansion.
            Z: ``(n_coalitions, M)`` binary coalition matrix.

        Returns:
            A ready-to-use :class:`DeterministicConditionalOracle`.

        Raises:
            TypeError: if ``oracle`` lacks the covariance attributes.
        """
        sigma_joints = getattr(oracle, "Sigma_joints", None)
        sigma_time = getattr(oracle, "Sigma_time", None)
        if sigma_joints is None or sigma_time is None:
            raise TypeError(
                f"{type(oracle).__name__} has no Sigma_joints/Sigma_time; "
                "the deterministic oracle supports Gaussian-Kronecker and "
                "Gaussian-copula families only."
            )
        marginal = getattr(oracle, "marginal", None)
        return cls(sigma_joints, sigma_time, players, Z, marginal=marginal)

    def fill_all(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        """Deterministic conditional-mean fills of ``x`` for every coalition.

        Observed entries are copied verbatim; hidden entries get the exact
        conditional mean ``E[x_hid | x_obs]`` (fully-hidden coalition:
        ``E[x] = 0``).  No randomness is involved.

        Args:
            x: ``(J, F, T)`` sequence (float64 recommended).

        Returns:
            ``(n_coalitions, J, F, T)`` float32 fills, rows aligned with the
            ``Z`` passed at construction.
        """
        J, F, T = x.shape
        x = np.asarray(x, dtype=np.float64)
        is_copula = self.marginal is not None
        if is_copula:
            z = self._x_to_z(x)  # latent z-space
            zf = z.transpose(1, 0, 2).reshape(F, J * T)
        xf = x.transpose(1, 0, 2).reshape(F, J * T)  # (F, J*T)
        out = np.empty((len(self.ops), F, J * T), dtype=np.float64)
        for i, (obs, hid, W, sd) in enumerate(self.ops):
            fill = xf.copy()
            if len(hid) == 0:
                pass
            elif len(obs) == 0:
                fill[:, :] = xf
                fill[:, hid] = 0.0  # E[x] = 0 for both families
            elif not is_copula:
                fill[:, hid] = (W @ xf[:, obs].T).T
            else:
                mu_z = (W @ zf[:, obs].T).T  # (F, n_hid)
                fill[:, hid] = self._copula_mean(mu_z, sd)
            out[i] = fill
        return (
            out.reshape(len(self.ops), F, J, T).transpose(0, 2, 1, 3).astype(np.float32)
        )

    def _x_to_z(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Forward copula transform ``z = Phi^-1(F(x))`` with probability clip."""
        assert self.marginal is not None
        u = np.clip(self.marginal.cdf(x), _EPS, 1.0 - _EPS)
        return np.asarray(ndtri(u), dtype=np.float64)

    def _copula_mean(
        self, mu_z: npt.NDArray[np.float64], sd: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """``E[q(Phi(mu + sd * t))]`` via cusp-split Gauss-Legendre, vectorised."""
        assert self.marginal is not None
        sd_b = np.broadcast_to(sd[None, :], mu_z.shape)
        t, w = cusp_split_nodes(mu_z, sd_b)
        zz = mu_z[..., None] + sd_b[..., None] * t
        return np.asarray(np.sum(self.marginal.quantile(ndtr(zz)) * w, axis=-1))

    def __repr__(self) -> str:
        fam = "copula" if self.marginal is not None else "gaussian"
        return (
            f"{self.__class__.__name__}(family={fam!r}, "
            f"n_coalitions={len(self.ops)}, players={self.players!r})"
        )
