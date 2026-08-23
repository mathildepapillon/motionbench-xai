"""scripts/_player_shap_common.py — shared machinery for the real-data player-set sweeps.

Common code for the sampled-coalition KernelSHAP entry points
``run_carepd_players_shap.py`` (SpatialJoints M=17 / JointWindowCells M=68),
``run_esc50_cells_shap.py`` (BandWindowCells M=16) and
``run_ptbxl_cells_shap.py`` (JointWindowCells M=48).

Protocol (identical across the three tracks; validated against the
independent validation study's real-data player-set runs — see
RESOLUTIONS.md §12):

* **Coalitions**: one fixed design per player count — the shap-style
  enumerate-then-importance-sample scheme of
  ``motionbench.attribution.sampled_coalitions.sampled_coalition_set``
  at B=2048 interior rows, coalition seed 7919, boundary rows pinned at
  rows 0/1 with weight 1e6.  The design is shared across methods and folds.
* **Fills**: zero / per-(J,F) train mean / one marginal donor per sequence
  (deterministic, single fill); VAEAC / Flow draw one completion per
  coalition (n=1) from the per-sequence stream
  ``np.random.default_rng([1104, fold, seq_idx])``.
* **phi**: constrained WLS with intercept, boundary constraints at 1e6 and
  ridge 1e-8 (``motionbench.attribution.sampled_coalitions.phi_from_values``),
  interior rows carrying the design's importance weights.
* **Faithfulness**: Pearson correlation of ``sum_{i not in S} phi_i`` with
  ``v(full) - v(S)`` over ALL rows of the design including the two boundary
  rows (mirrors the temporal protocol, which uses all enumerated rows).
* **PlayerAOPC**: the M cumulative-deletion coalitions (players removed in
  decreasing |phi|) are evaluated explicitly — they are generally not in the
  sampled design.  Deterministic fills are exact; VAEAC/Flow path fills are
  fresh draws from the same per-sequence stream.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Protocol

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

from motionbench.attribution.sampled_coalitions import (
    DEFAULT_COALITION_SEED,
    phi_from_values,
    sampled_coalition_set,
)
from motionbench.metrics.coalition_table import (
    aopc_order,
    deletion_path,
    faithfulness_sampled,
)

SEED_BASE = 1104  # per-sequence stochastic-imputer stream: rng([1104, fold, i])

DETERMINISTIC_METHODS = ["kernelshap_zero", "kernelshap_mean", "kernelshap_marginal"]
ALL_METHODS = [*DETERMINISTIC_METHODS, "kernelshap_vaeac", "kernelshap_flow"]


class _Imputer(Protocol):
    def impute_multi(
        self,
        x: np.ndarray,
        masks: np.ndarray,
        n: int,
        rng: np.random.Generator,
        chunk: int = ...,
    ) -> np.ndarray: ...


def build_sampled_design(M: int, budget: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Fixed sampled coalition design for M > 12 players.

    Returns (Z, w, i_empty, i_full) with the boundary rows pinned at rows
    0 (empty) / 1 (full).  For M <= 12 the temporal scripts enumerate
    exactly; this helper deliberately refuses that regime.
    """
    if M <= 12:
        raise ValueError(
            f"M={M} <= 12 is the exact-enumeration regime; use the temporal / "
            "leads scripts (run_care_pd_multiclf.py, run_ptbxl_leads_shap.py)."
        )
    Z, w = sampled_coalition_set(M, budget, seed=DEFAULT_COALITION_SEED)
    return Z, w, 0, 1


def coalition_masks(players, Z: np.ndarray) -> np.ndarray:
    """Expand every design row to a (J, F, T) bool observation mask."""
    rows = []
    for z in Z:
        zt = torch.from_numpy(np.asarray(z, dtype=np.int64))
        rows.append(players.coalition_mask(zt).numpy())
    return np.stack(rows)


def fills_deterministic(
    x: np.ndarray,
    masks: np.ndarray,
    method: str,
    mean_jf: np.ndarray,
    donor: np.ndarray,
) -> np.ndarray:
    """(n_masks, J, F, T) zero / mean / marginal completions (single fill)."""
    J, F, T = x.shape
    if method == "kernelshap_zero":
        fill = np.zeros((J, F, T), np.float32)
    elif method == "kernelshap_mean":
        fill = np.broadcast_to(mean_jf[:, :, None], (J, F, T)).astype(np.float32)
    elif method == "kernelshap_marginal":
        fill = donor.astype(np.float32)
    else:
        raise ValueError(f"unknown deterministic method {method!r}")
    out = np.where(masks, x[None], fill[None])
    return np.ascontiguousarray(out, dtype=np.float32)


def fills_stochastic(
    x: np.ndarray,
    masks: np.ndarray,
    imputer: _Imputer,
    rng: np.random.Generator,
    chunk: int,
) -> np.ndarray:
    """(n_masks, J, F, T): one completion per coalition (n=1 convention)."""
    out = imputer.impute_multi(x, masks, 1, rng, chunk=chunk)  # (nM, 1, J, F, T)
    return np.ascontiguousarray(out[:, 0], dtype=np.float32)


def run_player_cell(
    *,
    x_eval: np.ndarray,
    mean_jf: np.ndarray,
    donors: np.ndarray,
    players,
    method: str,
    fold: int,
    probs_fn: Callable[[np.ndarray], np.ndarray],
    targets: np.ndarray,
    imputer: _Imputer | None,
    imp_chunk: int,
    budget: int,
    cell_dir: Path,
    meta: dict,
    log_every: int = 20,
) -> dict:
    """Run one (track, player set, fold, method) cell and write its results.

    Args:
        x_eval: (N, J, F, T) float32 evaluation sequences.
        mean_jf: (J, F) training-pool mean for the mean fill.
        donors: (N, J, F, T) marginal-donor sequences (one per eval sequence).
        players: PlayerSet defining the coalition -> mask expansion.
        method: One of ALL_METHODS.
        fold: Fold index (seeds the per-sequence imputer stream).
        probs_fn: (n, J, F, T) float32 -> (n, n_classes) softmax probabilities.
        targets: (N,) predicted class per sequence (argmax of the full clip).
        imputer: Frame imputer for kernelshap_vaeac / kernelshap_flow.
        imp_chunk: Rows per imputer GPU pass.
        budget: Interior rows of the sampled coalition design.
        cell_dir: Output directory (result.json + attributions.npz).
        meta: Extra key/values merged into result.json (dataset, classifier, ...).
        log_every: Progress print frequency (sequences).

    Returns:
        The result dict written to ``cell_dir/result.json``.
    """
    N = len(x_eval)
    M = players.n_players
    Z, w, i_empty, i_full = build_sampled_design(M, budget)
    coal_masks = coalition_masks(players, Z)
    nZ = len(Z)

    cell_dir.mkdir(parents=True, exist_ok=True)
    phis = np.zeros((N, M), np.float32)
    v_all = np.zeros((N, nZ), np.float32)
    faiths = np.zeros(N)
    aopcs = np.zeros(N)
    t0 = time.time()

    for i in range(N):
        x = x_eval[i]
        tg = int(targets[i])
        rng = np.random.default_rng([SEED_BASE, fold, i])

        if method in ("kernelshap_vaeac", "kernelshap_flow"):
            assert imputer is not None
            comps = fills_stochastic(x, coal_masks, imputer, rng, imp_chunk)
        else:
            comps = fills_deterministic(x, coal_masks, method, mean_jf, donors[i])

        v = probs_fn(comps)[:, tg].astype(np.float64)
        v_all[i] = v

        phi = phi_from_values(Z, w, v)
        phis[i] = phi
        faiths[i] = faithfulness_sampled(Z, v, phi, i_full)

        # Explicit deletion-path AOPC (path coalitions are generally not in Z).
        path_Z = deletion_path(aopc_order(phi), M)
        path_masks = coalition_masks(players, path_Z)
        if method in ("kernelshap_vaeac", "kernelshap_flow"):
            assert imputer is not None
            pcomps = fills_stochastic(x, path_masks, imputer, rng, imp_chunk)
        else:
            pcomps = fills_deterministic(x, path_masks, method, mean_jf, donors[i])
        v_path = probs_fn(pcomps)[:, tg].astype(np.float64)
        aopcs[i] = float(np.mean(float(v[i_full]) - v_path))

        if (i + 1) % log_every == 0 or i == N - 1:
            print(
                f"  [{i + 1}/{N}] faith={faiths[i]:+.3f} aopc={aopcs[i]:+.3f} "
                f"({(time.time() - t0) / (i + 1):.2f}s/seq)",
                flush=True,
            )

    n_finite = int(np.isfinite(faiths).sum())
    result = {
        **meta,
        "fold": int(fold),
        "method": method,
        "n_sequences": int(N),
        "n_players": int(M),
        "n_coalitions": int(nZ),
        "budget": int(budget),
        "coalition_seed": DEFAULT_COALITION_SEED,
        "seed_base": SEED_BASE,
        "n_finite_faithfulness": n_finite,
        "faithfulness_correlation": float(np.nanmean(faiths)),
        "faithfulness_correlation_std": float(np.nanstd(faiths, ddof=1))
        if n_finite > 1
        else float("nan"),
        "player_aopc": float(np.mean(aopcs)),
        "player_aopc_std": float(np.std(aopcs, ddof=1)) if N > 1 else float("nan"),
        "phi_abs_mean_per_player": np.abs(phis).mean(axis=0).tolist(),
        "faithfulness_per_seq": faiths.tolist(),
        "player_aopc_per_seq": aopcs.tolist(),
        "targets_per_seq": targets[:N].tolist(),
        "elapsed_s": time.time() - t0,
    }
    np.savez_compressed(cell_dir / "attributions.npz", phi=phis, v=v_all, target=targets[:N])
    (cell_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("fold", "method", "faithfulness_correlation", "player_aopc", "elapsed_s")
            }
        )
    )
    return result
