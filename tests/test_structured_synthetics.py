"""Tests for skeleton_structured and gait_periodic synthetic datasets.

Test plan
---------
1. ``test_skeleton_adjacency_structure``
   Verify ``Sigma_joints[i, j] ≈ decay ** d(i, j)`` for selected (i,j) pairs
   using the known H36M-17 kinematic tree distances.

2. ``test_gait_periodic_autocorrelation``
   Draw N=50 samples, compute mean sample autocorrelation across j, f; verify
   the dominant lag (peak of autocorrelation in [1, T//2]) is within ±3 frames
   of the configured ``period_mean``.

3. ``test_end_to_end_shapley``
   Instantiate each dataset, sample one sequence, compute
   ``oracle.true_shapley`` with K=4 temporal players and ``n_mc=50``.
   Verify efficiency axiom: ``|Σφ − (v_full − v_empty)| < 0.5``.
   Marked ``@pytest.mark.slow``.

4. ``test_shapes``
   ``__getitem__`` returns ``(J, F, T)`` shaped first element.

5. ``test_metadata_keys``
   Metadata dict has required keys ``"skeleton"`` and ``"frame_rate"``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from motionbench.data.synthetic.gait_periodic import GaitPeriodicDataset
from motionbench.data.synthetic.skeleton_structured import SkeletonStructuredDataset

# ---------------------------------------------------------------------------
# Minimal temporal PlayerSet stub (identical to the 1A test helper)
# ---------------------------------------------------------------------------


class _TemporalPlayers:
    """Minimal K-window temporal PlayerSet for Shapley tests.

    Not a formal subclass of PlayerSet ABC; satisfies it structurally.
    """

    def __init__(self, K: int, J: int, F: int, T: int) -> None:
        self._K = K
        self._J = J
        self._F = F
        self._T = T
        quarter = T // K
        self._windows: list[list[int]] = [
            list(range(k * quarter, (k + 1) * quarter if k < K - 1 else T))
            for k in range(K)
        ]

    @property
    def n_players(self) -> int:
        """Number of temporal windows."""
        return self._K

    @property
    def shape(self) -> tuple[int, int, int]:
        """(J, F, T) element-space shape."""
        return (self._J, self._F, self._T)

    def coalition_mask(self, z: Tensor) -> Tensor:
        """Expand coalition vector to ``(J, F, T)`` boolean mask."""
        mask = torch.zeros(self._J, self._F, self._T, dtype=torch.bool)
        for k in range(self._K):
            if int(z[k].item()) == 1:
                for t in self._windows[k]:
                    mask[:, :, t] = True
        return mask

    def aggregate(self, phi_coords: Tensor) -> Tensor:
        """Sum per-window attributions into ``(K,)`` vector."""
        out = torch.zeros(self._K)
        for k in range(self._K):
            for t in self._windows[k]:
                out[k] += phi_coords[:, :, t].sum()
        return out


# ---------------------------------------------------------------------------
# Kinematic-tree distance helper (for test assertions)
# ---------------------------------------------------------------------------

#: H36M-17 kinematic tree edges (same as in SigmaJointsFactory.skeleton_adjacency).
_H36M_EDGES = [
    (0, 1), (1, 2), (2, 3),           # right leg
    (0, 4), (4, 5), (5, 6),           # left leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # spine / neck / head
    (8, 11), (11, 12), (12, 13),       # left arm
    (8, 14), (14, 15), (15, 16),       # right arm
]


def _bfs_distance(J: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Compute all-pairs BFS distance matrix for the kinematic tree.

    Args:
        J: Number of joints.
        edges: Undirected edge list.

    Returns:
        ``(J, J)`` float64 distance matrix.
    """
    adj: list[list[int]] = [[] for _ in range(J)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    dist = np.full((J, J), fill_value=np.inf, dtype=np.float64)
    for start in range(J):
        dist[start, start] = 0.0
        queue = [start]
        while queue:
            node = queue.pop(0)
            for nb in adj[node]:
                if dist[start, nb] == np.inf:
                    dist[start, nb] = dist[start, node] + 1.0
                    queue.append(nb)
    return dist


_DIST_H36M = _bfs_distance(17, _H36M_EDGES)

# Selected (i, j) pairs with known expected BFS distances.
_ADJACENCY_CHECK_PAIRS = [
    (0, 0, 0),    # diagonal: d=0 → decay^0 = 1
    (0, 1, 1),    # pelvis → rhip: d=1
    (1, 2, 1),    # rhip → rknee: d=1
    (0, 2, 2),    # pelvis → rknee: d=2
    (0, 3, 3),    # pelvis → rankle: d=3
    (0, 10, 4),   # pelvis → head: d=4  (0→7→8→9→10)
    (3, 6, 6),    # rankle → lankle: d=6 (3→2→1→0→4→5→6)
    (1, 10, 5),   # rhip → head: d=5 (1→0→7→8→9→10)
    (13, 16, 6),  # lwrist → rwrist: d=6 (13→12→11→8→14→15→16)
]


# ---------------------------------------------------------------------------
# Test 1: Skeleton adjacency structure
# ---------------------------------------------------------------------------


def test_skeleton_adjacency_structure() -> None:
    """Sigma_joints[i,j] ≈ decay ** BFS_distance(i,j) for all test pairs."""
    decay = 0.6
    ds = SkeletonStructuredDataset(J=17, T=16, N=10, decay=decay, seed=0)
    Sj = ds.oracle.Sigma_joints  # (17, 17) float64

    for i, j, expected_d in _ADJACENCY_CHECK_PAIRS:
        # After symmetrisation the matrix should equal decay^d at each entry.
        expected_val = decay ** float(expected_d)
        actual_val = float(Sj[i, j])
        assert abs(actual_val - expected_val) < 1e-9, (
            f"Sigma_joints[{i},{j}]: expected decay^{expected_d}={expected_val:.6f}, "
            f"got {actual_val:.6f}"
        )

    # Cross-check against our own BFS distance matrix.
    for i in range(17):
        for j_idx in range(17):
            d_ij = float(_DIST_H36M[i, j_idx])
            expected = decay ** d_ij
            assert abs(float(Sj[i, j_idx]) - expected) < 1e-9, (
                f"Mismatch at [{i},{j_idx}]: d={d_ij}, expected={expected:.6f}, "
                f"got {float(Sj[i, j_idx]):.6f}"
            )


# ---------------------------------------------------------------------------
# Test 2: Gait-periodic autocorrelation
# ---------------------------------------------------------------------------


def test_gait_periodic_autocorrelation() -> None:
    """Dominant sample autocorrelation lag matches period_mean within ±3 frames.

    Method: for each sequence n and channel (j, f), compute the sample
    autocorrelation r[τ] = mean_t(x[n,j,f,t] * x[n,j,f,t+τ]).  Average
    across all n, j, f.  Find the argmax in [1, T//2] and assert it is
    within ±3 of period_mean.
    """
    period_mean = 27.0
    T = 81
    N = 50
    ds = GaitPeriodicDataset(
        J=5,
        T=T,
        N=N,
        period_mean=period_mean,
        period_std=2.0,
        n_harmonics=3,
        seed=1,
    )

    # Collect all sequences as numpy array (N, J, F, T).
    x_all = np.stack([ds[i][0].numpy() for i in range(N)], axis=0)  # (N, J, F, T)
    N_data, J, F, _T = x_all.shape

    # Compute mean autocorrelation function across n, j, f.
    max_lag = T // 2
    acf = np.zeros(max_lag + 1, dtype=np.float64)
    for tau in range(max_lag + 1):
        # Compute for all (n, j, f) channels simultaneously.
        # acf[tau] = mean_n,j,f( mean_t( x[..., t] * x[..., t+tau] ) )
        prod = x_all[:, :, :, : T - tau] * x_all[:, :, :, tau:]  # (N, J, F, T-tau)
        acf[tau] = float(prod.mean())

    # Normalise by acf[0] to get autocorrelation coefficient.
    acf_norm = acf / acf[0] if acf[0] > 0 else acf.copy()

    # Find dominant lag in [1, max_lag].
    dominant_lag = int(np.argmax(acf_norm[1:]) + 1)

    assert abs(dominant_lag - period_mean) <= 3, (
        f"Dominant autocorrelation lag {dominant_lag} is not within ±3 of "
        f"period_mean={period_mean}.  Full acf[1:max_lag+1]:\n"
        f"{acf_norm[1:]}"
    )


# ---------------------------------------------------------------------------
# Test 3: End-to-end Shapley efficiency axiom
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "dataset_cls,kwargs",
    [
        (
            SkeletonStructuredDataset,
            {"J": 17, "T": 32, "N": 20, "alpha_time": 0.8, "decay": 0.5, "seed": 7},
        ),
        (
            GaitPeriodicDataset,
            {"J": 5, "T": 32, "N": 20, "period_mean": 10.0, "n_harmonics": 2, "seed": 7},
        ),
    ],
)
def test_end_to_end_shapley(dataset_cls: type, kwargs: dict) -> None:
    """Oracle.true_shapley satisfies efficiency axiom for K=4 temporal players.

    Efficiency: ``|Σφ − (v_full − v_empty)| < 0.5``.
    """
    ds = dataset_cls(**kwargs)
    x, _y = ds[0]  # (J, F, T)
    J, F, T = x.shape

    players = _TemporalPlayers(K=4, J=J, F=F, T=T)
    oracle = ds.oracle

    def classifier(batch: torch.Tensor) -> torch.Tensor:
        """Toy classifier: mean over all coordinates → (B,) scalar."""
        return batch.mean(dim=(1, 2, 3))

    phi = oracle.true_shapley(
        x=x,
        classifier=classifier,
        players=players,
        n_mc=50,
        seed=42,
    )  # (K,)

    assert phi.shape == (4,), f"Expected phi shape (4,); got {phi.shape}"

    # Compute v_full and v_empty manually for the efficiency check.
    full_mask = torch.ones(J, F, T, dtype=torch.bool)
    empty_mask = torch.zeros(J, F, T, dtype=torch.bool)

    samps_full = oracle.conditional_sample(x, full_mask, n=50, seed=1)
    samps_empty = oracle.conditional_sample(x, empty_mask, n=50, seed=2)

    v_full = float(classifier(samps_full).mean().item())
    v_empty = float(classifier(samps_empty).mean().item())

    phi_sum = float(phi.sum().item())
    efficiency_error = abs(phi_sum - (v_full - v_empty))

    assert efficiency_error < 0.5, (
        f"Efficiency axiom violated: |Σφ − (v_full − v_empty)| = {efficiency_error:.4f} "
        f"(threshold 0.5).  Σφ={phi_sum:.4f}, v_full={v_full:.4f}, v_empty={v_empty:.4f}."
    )


# ---------------------------------------------------------------------------
# Test 4: Shape contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dataset_cls,kwargs",
    [
        (
            SkeletonStructuredDataset,
            {"J": 17, "F": 3, "T": 16, "N": 5, "seed": 0},
        ),
        (
            GaitPeriodicDataset,
            {"J": 5, "F": 2, "T": 32, "N": 5, "period_mean": 10.0, "seed": 0},
        ),
    ],
)
def test_shapes(dataset_cls: type, kwargs: dict) -> None:
    """__getitem__ returns x with the correct (J, F, T) shape."""
    ds = dataset_cls(**kwargs)
    J = kwargs["J"]
    F = kwargs["F"]
    T = kwargs["T"]

    x, y = ds[0]
    assert x.shape == (J, F, T), f"Expected x.shape ({J},{F},{T}); got {x.shape}"
    assert x.dtype == torch.float32, f"Expected float32; got {x.dtype}"
    assert y.dtype == torch.int64, f"Expected int64 label; got {y.dtype}"
    assert y.ndim == 0, f"Expected scalar label; got shape {y.shape}"

    # Also verify __len__ and shape property.
    assert len(ds) == kwargs["N"]
    assert ds.shape == (J, F, T)


# ---------------------------------------------------------------------------
# Test 5: Metadata keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dataset_cls,kwargs",
    [
        (SkeletonStructuredDataset, {"J": 17, "T": 16, "N": 5, "seed": 0}),
        (GaitPeriodicDataset, {"J": 5, "T": 32, "N": 5, "period_mean": 10.0, "seed": 0}),
    ],
)
def test_metadata_keys(dataset_cls: type, kwargs: dict) -> None:
    """Metadata dict contains required keys ``skeleton`` and ``frame_rate``."""
    ds = dataset_cls(**kwargs)
    meta = ds.metadata

    required_keys = {"skeleton", "frame_rate"}
    missing = required_keys - set(meta.keys())
    assert not missing, f"Missing metadata keys: {missing}"

    assert isinstance(meta["skeleton"], str), "metadata['skeleton'] must be a str"
    assert isinstance(meta["frame_rate"], (int, float)), (
        "metadata['frame_rate'] must be numeric"
    )
