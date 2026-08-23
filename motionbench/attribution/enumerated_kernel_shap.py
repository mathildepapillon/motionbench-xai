"""motionbench.attribution.enumerated_kernel_shap — Exact KernelSHAP on enumerated designs.

The pinned solver behind every temporal (K uniform windows) and small-M
spatial real-data track.  All three functions are numerically load-bearing:
the canonical result files in ``results/canonical/`` were produced with
exactly these conventions, and the release's bit-exact reproduction gates
(RESOLUTIONS.md) hold only if they stay fixed.

Pinned conventions (do not "fix"):

- **Row order** is little-endian: row ``ci`` contains player ``k`` iff
  ``(ci >> k) & 1``.  The all-hidden row is index 0 and the all-visible row
  is index ``2**K - 1`` (last).  Stored ``v`` tables in ``result.json`` /
  ``attributions.npz`` are indexed by this order, and
  :func:`~motionbench.metrics.coalition_table.player_aopc_enumerated`
  recovers deletion-path rows by the same bit arithmetic.
- **Windows**: ``T`` need not divide by ``K``; the last window absorbs the
  remainder frames.
- **float32 quantisation**: values and returned attributions are cast to
  float32 around a float64 normal-equations solve — the precision of the
  archived runs.
- **Boundary rows** carry kernel weight ``1e6`` in place (contrast
  :func:`motionbench.utils.coalitions.shapley_kernel_weight`, which returns
  0 there and lets the caller append boundary rows).

For sampled designs (``M > 12``) use
:mod:`motionbench.attribution.sampled_coalitions` instead.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

__all__ = ["build_coalition_masks", "kernel_shap_exact", "shapley_kernel"]


def build_coalition_masks(K: int, T: int) -> tuple[Tensor, Tensor]:
    """Enumerate all ``2**K`` window coalitions with their frame masks.

    Args:
        K: Number of uniform temporal windows (players).
        T: Number of frames; the last window absorbs ``T % K``.

    Returns:
        ``(z_bin, frame_mask)``: bool tensors of shape ``(2**K, K)`` and
        ``(2**K, T)`` in the pinned little-endian row order (True = visible).
    """
    n_coal = 1 << K
    win_size = T // K
    z_bin = np.zeros((n_coal, K), dtype=bool)
    frame_mask = np.zeros((n_coal, T), dtype=bool)
    for ci in range(n_coal):
        for k in range(K):
            if (ci >> k) & 1:
                z_bin[ci, k] = True
                t0 = k * win_size
                t1 = t0 + win_size if k < K - 1 else T
                frame_mask[ci, t0:t1] = True
    return torch.from_numpy(z_bin), torch.from_numpy(frame_mask)


def shapley_kernel(K: int, s: int) -> float:
    """Shapley kernel weight for coalition size ``s`` of ``K`` players.

    Boundary sizes (``s in {0, K}``) return the pinned in-place weight
    ``1e6`` rather than 0.
    """
    if s == 0 or s == K:
        return 1e6
    from math import comb

    return (K - 1) / (comb(K, s) * s * (K - s))


def kernel_shap_exact(z_bin: Tensor, v_vals: Tensor, K: int) -> Tensor:
    """Weighted-least-squares Shapley values on a fully enumerated design.

    Args:
        z_bin: ``(2**K, K)`` coalition rows from :func:`build_coalition_masks`.
        v_vals: ``(2**K,)`` game values in the same row order.
        K: Number of players.

    Returns:
        ``(K,)`` float32 attributions (float64 solve, float32 in/out —
        the pinned precision of the archived runs).
    """
    Z = z_bin.float().numpy()
    v = v_vals.float().numpy()
    n = Z.shape[0]
    sizes = Z.sum(axis=1).astype(int)
    w = np.array([shapley_kernel(K, int(s)) for s in sizes])
    Z_ext = np.concatenate([np.ones((n, 1)), Z], axis=1)
    W = np.diag(w)
    A = Z_ext.T @ W @ Z_ext
    b = Z_ext.T @ W @ v
    A += 1e-8 * np.eye(A.shape[0])
    sol = np.linalg.solve(A, b)
    return torch.from_numpy(sol[1:]).float()
