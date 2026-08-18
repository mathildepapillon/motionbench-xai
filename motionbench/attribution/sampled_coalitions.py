"""motionbench.attribution.sampled_coalitions — Fixed sampled coalition designs.

This module builds the **fixed coalition design** ``(Z, w)`` used by the
player-generic evaluation pipeline and solves the constrained KernelSHAP WLS
system on it.  Unlike :func:`~motionbench.utils.coalitions.sample_kernelshap_coalitions`
(paired i.i.d. sampling, fresh draws per call), the design here is *shared*:
one coalition set per (player set, budget) is reused for every method and for
the deterministic grading targets, so that estimator error — not coalition
noise — is what the EC1/EC3 metrics measure.

Scheme (shap-style, importance-corrected)
-----------------------------------------
For ``M <= exact_max_m`` all ``2^M`` coalitions are enumerated with their exact
Shapley kernel weights.  Otherwise:

1. **Boundary rows pinned.**  The empty and full coalitions are always rows
   0 and 1, carrying weight ``1e6`` (they additionally enter the WLS solve as
   hard constraints).
2. **Complete size pairs enumerated in kernel-mass order.**  Coalition sizes
   are processed as complementary pairs ``(s, M - s)`` in order of ascending
   ``min(s, M - s)`` — i.e. descending total kernel mass
   ``C(M, s) * k(s) = (M-1) / (s (M-s))``.  A pair is fully enumerated only if
   *all* its coalitions fit in the remaining budget; enumerated rows carry
   their exact kernel weight ``k(s) = (M-1) / (C(M,s) * s * (M-s))``.
3. **Importance-corrected sampling of the remainder.**  The remaining budget
   is filled with distinct coalitions sampled from the leftover sizes with
   probability proportional to kernel mass.  Each sampled row carries the
   *uniform importance weight* ``(total leftover kernel mass) / n_sampled``,
   so the sampled block's total weight equals the kernel mass it stands in
   for.  This is the correction that keeps enumerated and sampled strata on a
   common scale; weighting sampled rows by their raw kernel weight instead
   would double-count the size distribution they were drawn from.
4. **Log-space weights.**  All weights are computed in log space (no
   ``comb`` overflow for large M) and rescaled so the largest non-boundary
   weight is 1.

Ported from the independent validation study's implementation
(player-set sweep, ``coalition_set``); kept call-for-call identical so the
two produce bit-identical designs for the same ``(M, budget, seed)``.

Example
-------
>>> Z, w = sampled_coalition_set(M=17, budget=1024)
>>> Z.shape[1]
17
>>> phi = phi_from_values(Z, w, v)   # v: (len(Z),) value-function evaluations
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import numpy.typing as npt

from motionbench.utils.coalitions import enumerate_coalitions, solve_shapley_wls

__all__ = [
    "DEFAULT_COALITION_SEED",
    "EXACT_MAX_M",
    "sampled_coalition_set",
    "phi_from_values",
]

# Enumerate all 2^M coalitions exactly up to this many players (2^12 = 4096 rows).
EXACT_MAX_M: int = 12

# Default seed for the shared coalition design.  The design is a *fixed*
# experimental artifact: every method evaluated against it must use the same
# (M, budget, seed) triple, so this constant should only be changed for
# deliberate sensitivity studies.
DEFAULT_COALITION_SEED: int = 7919


def sampled_coalition_set(
    M: int,
    budget: int,
    seed: int = DEFAULT_COALITION_SEED,
    exact_max_m: int = EXACT_MAX_M,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.float64]]:
    """Build the fixed coalition design ``(Z, w)`` for ``M`` players.

    Exact enumeration if ``M <= exact_max_m``; otherwise the shap-style
    scheme documented in the module docstring: enumerate complete size pairs
    ``(s, M-s)`` in kernel-mass order while they fit in the budget (rows carry
    their exact kernel weight), then sample the remaining budget from leftover
    sizes proportional to kernel mass (those rows carry importance-corrected
    uniform weights).  Boundary rows are pinned at rows 0 (empty) and 1 (full)
    with weight ``1e6``.

    Args:
        M: Number of players.
        budget: Number of *interior* coalition rows in the sampled design
            (the two boundary rows are added on top).  Ignored when
            ``M <= exact_max_m``.
        seed: Seed for the sampled remainder.  The full seeding key is
            ``[seed, M, budget]``, so designs for different sizes or budgets
            are independent streams.
        exact_max_m: Threshold below which all ``2^M`` coalitions are
            enumerated (with boundary rows at weight 0, as
            :func:`~motionbench.utils.coalitions.enumerate_coalitions` returns).

    Returns:
        Z: ``(n_rows, M)`` binary coalition matrix (1 = player observed).
        w: ``(n_rows,)`` float64 row weights for the WLS solve.

    Raises:
        ValueError: if ``M < 2``, or ``budget < 1`` in the sampled regime.
        RuntimeError: if the sampler cannot find enough distinct coalitions
            (budget larger than the number of available leftover coalitions).

    Determinism: for a fixed ``(M, budget, seed)`` the returned arrays are
    bit-identical across calls and platforms (single ``default_rng`` stream).
    """
    if M < 2:
        raise ValueError(f"Need at least 2 players; got M={M}.")
    if M <= exact_max_m:
        return enumerate_coalitions(M)
    if budget < 1:
        raise ValueError(f"budget must be >= 1 in the sampled regime; got {budget}.")

    rng = np.random.default_rng([seed, M, budget])

    def log_kernel(s: int) -> float:
        # log of the per-row kernel weight (M-1) / (C(M,s) * s * (M-s)).
        return (
            math.log(M - 1)
            - (math.lgamma(M + 1) - math.lgamma(s + 1) - math.lgamma(M - s + 1))
            - math.log(s)
            - math.log(M - s)
        )

    def log_mass(s: int) -> float:
        # log of the per-size total mass C(M,s) * kernel(s) = (M-1)/(s*(M-s)).
        return math.log(M - 1) - math.log(s) - math.log(M - s)

    def n_of(s: int) -> int:
        return math.comb(M, s)

    sizes = list(range(1, M))
    pairs: list[list[int]] = []
    seen_p: set[frozenset[int]] = set()
    for s_ in sizes:
        key = frozenset((s_, M - s_))
        if key not in seen_p:
            seen_p.add(key)
            pairs.append(sorted(key))
    # kernel-mass order = ascending min(s, M - s); `pairs` is already sorted.
    rows: list[npt.NDArray[np.int8]] = [np.zeros(M, np.int8), np.ones(M, np.int8)]
    wts: list[float | None] = [None, None]
    enumerated: set[int] = set()
    remaining = budget
    for pr in pairs:
        cnt = sum(n_of(s_) for s_ in set(pr))
        if cnt > remaining:
            break
        for s_ in set(pr):
            lw = log_kernel(s_)
            for comb_idx in combinations(range(M), s_):
                z = np.zeros(M, np.int8)
                z[list(comb_idx)] = 1
                rows.append(z)
                wts.append(lw)
            enumerated.add(s_)
        remaining -= cnt
    left = [s_ for s_ in sizes if s_ not in enumerated]
    if left and remaining > 0:
        lm = np.array([log_mass(s_) for s_ in left])
        p = np.exp(lm - lm.max())
        p /= p.sum()
        total_left_logmass = lm.max() + math.log(np.exp(lm - lm.max()).sum())
        picked: set[bytes] = set()
        draws = 0
        # Rejection guard: never triggered at sane budgets, but prevents an
        # unbounded loop when the caller asks for more distinct coalitions
        # than the leftover sizes contain.
        attempts, max_attempts = 0, 1000 * remaining
        while draws < remaining:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"Could not draw {remaining} distinct coalitions from the "
                    f"leftover sizes {left} (M={M}); lower the budget."
                )
            s_ = int(rng.choice(left, p=p))
            z = np.zeros(M, np.int8)
            z[rng.choice(M, size=s_, replace=False)] = 1
            key_b = z.tobytes()
            if key_b in picked:
                continue
            picked.add(key_b)
            rows.append(z)
            wts.append(None)  # fill below: uniform importance weight
            draws += 1
        lw_samp = total_left_logmass - math.log(draws)
        wts = [lw_samp if w is None and i >= 2 else w for i, w in enumerate(wts)]
    # Rescale: max non-boundary log-weight -> 0; boundary rows -> 1e6.
    lws = np.array([w for w in wts[2:]], dtype=np.float64)
    lws -= lws.max()
    w_arr = np.concatenate([[1e6, 1e6], np.exp(lws)])
    return np.stack(rows), w_arr


def phi_from_values(
    Z: npt.NDArray[np.integer],
    w: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Constrained WLS Shapley solve over a fixed coalition design.

    Boundary values are taken from ``v`` itself: the rows of ``Z`` that are
    all-zeros / all-ones supply ``v(empty)`` / ``v(full)``, which enter the
    solve as hard constraints (weight ``1e6``) enforcing the efficiency axiom
    ``sum(phi) = v(full) - v(empty)``.

    Args:
        Z: ``(n_rows, M)`` binary coalition matrix.  Must contain at least one
            all-zeros row and one all-ones row (both
            :func:`sampled_coalition_set` and
            :func:`~motionbench.utils.coalitions.enumerate_coalitions`
            guarantee this).
        w: ``(n_rows,)`` row weights aligned with ``Z``.
        v: ``(n_rows,)`` value-function evaluations ``v(S)`` aligned with ``Z``.

    Returns:
        ``(M,)`` float64 Shapley values.

    Raises:
        IndexError: if ``Z`` lacks a boundary row.
    """
    empty = int(np.flatnonzero(Z.sum(axis=1) == 0)[0])
    full = int(np.flatnonzero(Z.sum(axis=1) == Z.shape[1])[0])
    return solve_shapley_wls(
        np.asarray(Z, dtype=np.intp), np.asarray(v, dtype=np.float64),
        np.asarray(w, dtype=np.float64), float(v[empty]), float(v[full]),
    )
