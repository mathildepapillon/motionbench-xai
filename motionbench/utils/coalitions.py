"""motionbench.utils.coalitions — Coalition sampling and Shapley WLS utilities.

This module provides shared helpers for Shapley-value computation used by
:class:`~motionbench.oracles.gaussian_oracle.GaussianOracle` and the
KernelSHAP attributor.  All functions are stateless and side-effect-free.

The Shapley kernel weight, coalition enumeration, and WLS solver follow:
    Lundberg & Lee (2017) "A unified approach to interpreting model
    predictions" (KernelSHAP, Appendix A).

The paired KernelSHAP sampling strategy follows:
    Covert & Lee (2021) "Improving KernelSHAP: Practical Streamlined
    Monte Carlo with Application to Economic Networks."

References
----------
Lundberg, S. M., & Lee, S.-I. (2017).
    A unified approach to interpreting model predictions. NeurIPS 30.
Covert, I., & Lee, S.-I. (2021).
    Improving KernelSHAP: Practical streamlined Monte Carlo with
    application to economic networks. AISTATS.
Aas, K., Jullum, M., & Løland, A. (2021).
    Explaining individual predictions when features are dependent:
    More accurate approximations to Shapley values. arXiv:1903.10464.
"""

from __future__ import annotations

import itertools
from math import comb

import numpy as np

__all__ = [
    "ar1_cov",
    "equicorr",
    "shapley_kernel_weight",
    "enumerate_coalitions",
    "sample_kernelshap_coalitions",
    "solve_shapley_wls",
]


def ar1_cov(T: int, alpha: float) -> np.ndarray:
    """Compute the T×T AR(1) covariance matrix.

    Entry ``C[t, t'] = alpha^|t-t'|``.  This is the temporal covariance
    implied by a stationary AR(1) process with autocorrelation ``alpha``.

    Args:
        T: Number of time-steps (matrix dimension).
        alpha: AR(1) autocorrelation coefficient.  Must satisfy
            ``0 <= alpha < 1`` for stationarity, but the function does
            not enforce this constraint.

    Returns:
        ``(T, T)`` float64 array.

    Examples:
        >>> ar1_cov(3, 0.5)
        array([[1.  , 0.5 , 0.25],
               [0.5 , 1.  , 0.5 ],
               [0.25, 0.5 , 1.  ]])
    """
    t = np.arange(T, dtype=np.float64)
    return alpha ** np.abs(t[:, None] - t[None, :])


def equicorr(J: int, rho: float) -> np.ndarray:
    """Compute the J×J equicorrelation matrix.

    ``C[j, j] = 1`` and ``C[j, j'] = rho`` for ``j != j'``.

    Args:
        J: Matrix dimension (number of joints).
        rho: Off-diagonal correlation.  Must satisfy
            ``-1/(J-1) <= rho <= 1`` for PSD, but not enforced here.

    Returns:
        ``(J, J)`` float64 array.

    Examples:
        >>> equicorr(3, 0.5)
        array([[1. , 0.5, 0.5],
               [0.5, 1. , 0.5],
               [0.5, 0.5, 1. ]])
    """
    return rho * np.ones((J, J), dtype=np.float64) + (1.0 - rho) * np.eye(J, dtype=np.float64)


def shapley_kernel_weight(s: int, M: int) -> float:
    """Compute the KernelSHAP weight for a coalition of size *s* from *M* players.

    The Shapley kernel weight (Lundberg & Lee 2017, Appendix A) is::

        w(s, M) = (M - 1) / (C(M, s) * s * (M - s))

    Boundary coalitions (empty ``s=0`` and full ``s=M``) receive weight 0
    because they are handled as hard constraints in the WLS solve.

    Args:
        s: Coalition size (number of players present).
        M: Total number of players.

    Returns:
        Non-negative float.  Returns 0.0 for ``s == 0`` or ``s == M``.

    Raises:
        ValueError: if ``s < 0`` or ``s > M``.
    """
    if s < 0 or s > M:
        raise ValueError(f"Coalition size s={s} out of range [0, M={M}].")
    if s == 0 or s == M:
        return 0.0
    return (M - 1) / (comb(M, s) * s * (M - s))


def enumerate_coalitions(M: int) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate all 2^M binary coalition vectors with their Shapley weights.

    Returns all binary vectors of length M and their corresponding
    KernelSHAP kernel weights.  Boundary coalitions (all-zeros and
    all-ones) receive weight 0 and are used as WLS constraints.

    Args:
        M: Number of players.  Must satisfy ``M <= 20`` to avoid
            memory exhaustion (2^20 ≈ 1 million rows).

    Returns:
        coalitions: ``(2^M, M)`` int array.  Each row is a binary
            coalition indicator (1 = player present).
        weights: ``(2^M,)`` float64 array of Shapley kernel weights.

    Raises:
        ValueError: if ``M > 20``.
    """
    if M > 20:
        raise ValueError(
            f"M={M} would enumerate 2^{M} = {2**M:,} coalitions; "
            "refusing.  Use sample_kernelshap_coalitions instead."
        )
    coalitions = np.array(list(itertools.product([0, 1], repeat=M)), dtype=int)
    weights = np.array(
        [shapley_kernel_weight(int(row.sum()), M) for row in coalitions],
        dtype=np.float64,
    )
    return coalitions, weights


def sample_kernelshap_coalitions(
    M: int,
    n_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample paired KernelSHAP coalitions with their kernel weights.

    Paired sampling (Covert & Lee 2021) draws a subset size ``s`` from
    the SHAP kernel size-distribution and emits **both** ``z`` and its
    complement ``1 - z`` to halve variance.  Boundary coalitions (empty
    and full) are emitted separately by the caller as hard constraints
    and are NOT included in the returned arrays.

    Args:
        M: Number of players.
        n_pairs: Number of complementary pairs to draw.  The returned
            arrays contain ``2 * n_pairs`` rows.
        rng: Numpy random Generator for reproducibility.

    Returns:
        coalitions: ``(2*n_pairs, M)`` int array of binary coalition
            indicators (all rows have ``0 < sum < M``).
        weights: ``(2*n_pairs,)`` float64 array of Shapley kernel weights
            (all values are strictly positive by construction).

    Raises:
        ValueError: if ``M < 2`` (no valid coalition sizes exist).
    """
    if M < 2:
        raise ValueError(f"M must be >= 2 for paired sampling; got M={M}.")
    sizes = np.arange(1, M, dtype=int)
    size_weights = np.array(
        [shapley_kernel_weight(int(s), M) * comb(M, int(s)) for s in sizes],
        dtype=np.float64,
    )
    p = size_weights / size_weights.sum()

    coalitions = np.zeros((2 * n_pairs, M), dtype=int)
    weights = np.zeros(2 * n_pairs, dtype=np.float64)

    for i in range(n_pairs):
        s = int(rng.choice(sizes, p=p))
        idx = rng.choice(M, size=s, replace=False)
        z = np.zeros(M, dtype=int)
        z[idx] = 1
        coalitions[2 * i] = z
        coalitions[2 * i + 1] = 1 - z
        weights[2 * i] = shapley_kernel_weight(int(z.sum()), M)
        weights[2 * i + 1] = shapley_kernel_weight(int((1 - z).sum()), M)

    return coalitions, weights


def solve_shapley_wls(
    coalitions: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    v_empty: float,
    v_full: float,
) -> np.ndarray:
    """Solve for Shapley values via constrained weighted least squares.

    Implements the KernelSHAP WLS solve from Lundberg & Lee (2017).  The
    empty (``v_empty``) and full (``v_full``) coalition values are added
    as hard constraints with large weight to enforce the efficiency axiom::

        sum(phi) = v_full - v_empty.

    The regression problem is::

        min_{phi_0, phi} sum_i w_i (phi_0 + z_i · phi - v_i)^2

    with ``z_i`` the ``i``-th coalition indicator, ``phi_0`` an intercept
    (absorbed), and boundary coalitions added as high-weight rows.

    Args:
        coalitions: ``(N, M)`` int array of binary coalition indicators.
        values: ``(N,)`` float array of value-function evaluations v(S).
        weights: ``(N,)`` float array of Shapley kernel weights.
        v_empty: Value of the empty coalition v(∅).
        v_full: Value of the full coalition v(N).

    Returns:
        ``(M,)`` float64 array of Shapley values φ.  Satisfies
        ``phi.sum() ≈ v_full - v_empty`` to numerical precision.

    Raises:
        ValueError: if ``coalitions.shape[1]`` does not equal ``M`` or
            shapes are inconsistent.
    """
    M = coalitions.shape[1]
    if values.shape[0] != coalitions.shape[0]:
        raise ValueError(
            f"values length {values.shape[0]} != coalitions rows {coalitions.shape[0]}."
        )
    if weights.shape[0] != coalitions.shape[0]:
        raise ValueError(
            f"weights length {weights.shape[0]} != coalitions rows {coalitions.shape[0]}."
        )

    _BIG = 1e6
    boundary_z = np.array([[0] * M, [1] * M], dtype=int)
    boundary_v = np.array([v_empty, v_full], dtype=np.float64)
    boundary_w = np.full(2, _BIG, dtype=np.float64)

    c_all = np.vstack([coalitions, boundary_z])
    v_all = np.concatenate([values, boundary_v])
    w_all = np.concatenate([weights, boundary_w])

    keep = w_all > 0
    Z = np.column_stack([np.ones(keep.sum(), dtype=np.float64), c_all[keep].astype(np.float64)])
    sq = np.sqrt(w_all[keep])[:, None]
    A = (Z * sq).T @ (Z * sq) + 1e-8 * np.eye(M + 1)
    b = (Z * sq).T @ (v_all[keep] * sq[:, 0])
    theta = np.linalg.solve(A, b)
    return theta[1:]
