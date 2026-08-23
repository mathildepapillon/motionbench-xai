"""motionbench.metrics.coalition_table — Faithfulness and PlayerAOPC over coalition tables.

The two no-oracle metrics behind every real-data table, in their two pinned
variants:

- **Enumerated** (temporal / small-M tracks): torch/float32 inputs over the
  full ``2**K`` table from
  :func:`motionbench.attribution.enumerated_kernel_shap.build_coalition_masks`;
  ``v(full)`` is the last row; the AOPC deletion path is read *out of the
  table* by the little-endian bit index; deletion order uses torch's default
  (unstable) ``argsort``.
- **Sampled** (player-set tracks, ``M > 12``): numpy/float64 over a sampled
  design with explicit boundary indices; the deletion path is evaluated
  explicitly (its coalitions are generally not in the design); deletion
  order uses a stable sort — ties broken by player index.

Both variants and their differences are pinned by the release's bit-exact
reproduction gates (RESOLUTIONS.md); the canonical result files were
produced with exactly these conventions.  Faithfulness always includes the
boundary rows.  Degenerate inputs (zero spread) return ``nan``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "aopc_order",
    "deletion_path",
    "faithfulness_enumerated",
    "faithfulness_sampled",
    "player_aopc_enumerated",
]


# --------------------------------------------------------------------- #
# Enumerated designs (torch, float32, full 2**K table)                   #
# --------------------------------------------------------------------- #


def faithfulness_enumerated(z_bin: Tensor, v_vals: Tensor, phi: Tensor) -> float:
    """Pearson corr of ``sum_{i not in S} phi_i`` with ``v(full) - v(S)``.

    Args:
        z_bin: ``(2**K, K)`` coalition rows; all-visible row last.
        v_vals: ``(2**K,)`` game values in the same row order.
        phi: ``(K,)`` attributions.

    Returns:
        Correlation over all rows (boundaries included); ``nan`` when either
        side has spread below ``1e-10``.
    """
    z = z_bin.float()
    not_z = 1.0 - z
    sum_phi_absent = not_z @ phi
    delta = v_vals[-1] - v_vals
    a, b = sum_phi_absent.numpy(), delta.float().numpy()
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def player_aopc_enumerated(v_vals: Tensor, z_bin: Tensor, phi: Tensor, K: int) -> float:
    """Mean drop ``v(full) - v(S_step)`` along the |phi|-descending deletion path.

    Path values are read out of the enumerated table by the pinned
    little-endian bit index; deletion order is torch's default ``argsort``
    (unstable ties — pinned).

    Args:
        v_vals: ``(2**K,)`` game values in enumerated row order.
        z_bin: ``(2**K, K)`` coalition rows (only the all-visible row is used).
        phi: ``(K,)`` attributions.
        K: Number of players.
    """
    order = torch.argsort(phi.abs(), descending=True).tolist()
    v_full = v_vals[-1].item()
    drops: list[float] = []
    cur_z = z_bin[-1].clone()
    for k in order:
        cur_z[k] = False
        idx = int((cur_z.int() * (1 << torch.arange(K))).sum().item())
        drops.append(v_full - v_vals[idx].item())
    return float(np.mean(drops)) if drops else 0.0


# --------------------------------------------------------------------- #
# Sampled designs (numpy, float64, explicit boundary / deletion path)    #
# --------------------------------------------------------------------- #


def faithfulness_sampled(Z: np.ndarray, v: np.ndarray, phi: np.ndarray, i_full: int) -> float:
    """Pearson corr of ``sum_{i not in S} phi_i`` with ``v(full) - v(S)``, all rows.

    Args:
        Z: ``(B, M)`` 0/1 design rows (boundaries included).
        v: ``(B,)`` game values.
        phi: ``(M,)`` attributions.
        i_full: Row index of the all-visible coalition.
    """
    sum_absent = (1.0 - Z.astype(np.float64)) @ phi
    delta = v[i_full] - v
    if np.std(sum_absent) < 1e-10 or np.std(delta) < 1e-10:
        return float("nan")
    return float(np.corrcoef(sum_absent, delta)[0, 1])


def aopc_order(phi: np.ndarray) -> list[int]:
    """Deletion order: decreasing ``|phi|``, ties broken by player index."""
    return np.argsort(-np.abs(phi), kind="stable").tolist()


def deletion_path(order: list[int], M: int) -> np.ndarray:
    """``(M, M)`` int8 coalition rows of the cumulative-deletion path."""
    path_Z = np.ones((M, M), np.int8)
    cur = np.ones(M, np.int8)
    for step, p_idx in enumerate(order):
        cur[p_idx] = 0
        path_Z[step] = cur
    return path_Z
