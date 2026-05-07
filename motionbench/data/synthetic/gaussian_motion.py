"""motionbench.data.synthetic.gaussian_motion — Gaussian motion generator.

Ported and refactored from ``CARE-PD/synthetic/gaussian_motion.py``.
The MLP classifier (``SyntheticMLPClassifier``) and label functions
(``setup_label_fn``, ``canonical_label_fn``) are NOT included here;
they belong to Tasks 4A and 1D respectively.

DATA MODEL
----------
    x ~ N(0, Sigma_joints ⊗ I_F ⊗ Sigma_time)

where the Kronecker product expresses independent, identically structured
variability across the F coordinate channels.

CONFORMS TO
-----------
:class:`~motionbench.data.base.GroundTruthDataset` protocol (structural,
no inheritance required):

* ``__getitem__(idx)`` → ``(x, y)`` with ``x: (J, F, T) float32``,
  ``y: scalar int64``.
* ``__len__()`` → ``N`` (number of pre-sampled sequences).
* ``shape`` property → ``(J, F, T)``.
* ``metadata`` property → dict with required keys plus Sigma provenance.
* ``oracle`` property → :class:`GaussianOracle` instance.

COVARIANCE FACTORIES
--------------------
:class:`SigmaJointsFactory` — spatial (joint×joint) covariance matrices.
:class:`SigmaTimeFactory` — temporal (T×T) covariance matrices.

References
----------
Olsen, L. R., Glad, I. K., Hjort, N. L., & Tveten, M. (2022).
    Using Shapley Values and Variational Autoencoders To Explain
    Predictions from Neural Networks for Short-Term Wind Power
    Forecasting.  JMLR 23(1), 1–38.
Aas, K., Jullum, M., & Løland, A. (2021).
    Explaining individual predictions when features are dependent.
    arXiv:1903.10464.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from motionbench.utils.coalitions import ar1_cov, equicorr

if TYPE_CHECKING:
    from motionbench.oracles.gaussian_oracle import GaussianOracle

__all__ = [
    "GaussianMotionBenchmark",
    "GaussianMotionDataset",
    "SigmaJointsFactory",
    "SigmaTimeFactory",
]


# ---------------------------------------------------------------------------
# Covariance factories
# ---------------------------------------------------------------------------


class SigmaJointsFactory:
    """Factory for joint-joint (J×J) covariance matrices.

    Every method returns a PSD matrix suitable for use as ``Sigma_joints``
    in :class:`GaussianMotionBenchmark`.
    """

    @staticmethod
    def equicorrelated(J: int, rho: float) -> np.ndarray:
        """Build a J×J equicorrelation matrix.

        ``C[j, j] = 1``, ``C[j, j'] = rho`` for ``j != j'``.

        Args:
            J: Number of joints.
            rho: Off-diagonal correlation coefficient.

        Returns:
            ``(J, J)`` float64 PSD array.
        """
        return equicorr(J, rho)

    @staticmethod
    def skeleton_adjacency(
        J: int,
        skeleton: str = "h36m_17",
        decay: float = 0.7,
    ) -> np.ndarray:
        """Build a J×J covariance from kinematic-tree graph distance.

        Off-diagonal entry ``C[j, j'] = decay ** BFS_distance(j, j')``
        where BFS distance is computed on the kinematic tree.  For the
        ``"h36m_17"`` skeleton, the 17-joint H36M tree is hardcoded.
        The diagonal is always 1.

        Args:
            J: Number of joints.  For ``skeleton="h36m_17"`` this must
                equal 17; otherwise a ValueError is raised.
            skeleton: Skeleton identifier.  Only ``"h36m_17"`` is
                supported; other values raise ``ValueError``.
            decay: Correlation decay per graph hop.  ``decay=1`` → all-ones
                matrix; ``decay=0`` → identity.

        Returns:
            ``(J, J)`` float64 symmetric PSD array.

        Raises:
            ValueError: if ``skeleton`` is not recognised or ``J`` does
                not match the skeleton's joint count.
        """
        if skeleton != "h36m_17":
            raise ValueError(
                f"Unsupported skeleton {skeleton!r}.  Only 'h36m_17' is hardcoded."
            )
        if J != 17:
            raise ValueError(
                f"h36m_17 skeleton has 17 joints; got J={J}."
            )
        # H36M-17 kinematic tree edges (parent → child).
        # Joint indices: 0=pelvis,1=r_hip,2=r_knee,3=r_ankle,
        # 4=l_hip,5=l_knee,6=l_ankle,7=spine,8=thorax,9=neck,
        # 10=head,11=l_shoulder,12=l_elbow,13=l_wrist,
        # 14=r_shoulder,15=r_elbow,16=r_wrist
        edges = [
            (0, 1), (1, 2), (2, 3),   # right leg
            (0, 4), (4, 5), (5, 6),   # left leg
            (0, 7), (7, 8), (8, 9), (9, 10),  # spine/neck/head
            (8, 11), (11, 12), (12, 13),  # left arm
            (8, 14), (14, 15), (15, 16),  # right arm
        ]
        # BFS distance matrix.
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

        C = decay ** dist
        # Symmetrise for numerical cleanliness.
        return 0.5 * (C + C.T)

    @staticmethod
    def block_diagonal(J: int, left_right: bool = True) -> np.ndarray:
        """Build a block-diagonal joint covariance.

        Splits the J joints into two halves (left body / right body).
        Within each half, off-diagonal correlation = 0.5.  Between halves,
        correlation = 0.  Diagonal = 1.

        Args:
            J: Number of joints.
            left_right: If ``True`` (default), the two halves are
                ``joints[:J//2]`` and ``joints[J//2:]``.  The flag is
                reserved for future non-symmetric splits but is currently
                ignored.

        Returns:
            ``(J, J)`` float64 block-diagonal PSD array.
        """
        C = np.zeros((J, J), dtype=np.float64)
        half = J // 2
        # First block.
        C[:half, :half] = 0.5
        np.fill_diagonal(C[:half, :half], 1.0)
        # Second block.
        C[half:, half:] = 0.5
        np.fill_diagonal(C[half:, half:], 1.0)
        # Remainder (if J is odd).
        if J % 2 != 0:
            C[J - 1, J - 1] = 1.0
        return C

    @staticmethod
    def low_rank(
        J: int,
        rank: int,
        eps: float = 1e-2,
        seed: int = 0,
    ) -> np.ndarray:
        """Build a rank-deficient PSD covariance with effective rank ``rank``.

        ``Sigma = U U^T + eps * I`` with ``U ∈ R^{J×rank}`` having i.i.d.
        standard-normal entries scaled so the diagonal of ``U U^T`` is ~1.
        The data lives near a ``rank``-dimensional linear subspace of ``R^J``;
        ``eps`` controls the noise floor orthogonal to that subspace.

        This factory is used to construct datasets whose joint covariance has
        intrinsic dimension < J, exposing the on-/off-manifold distinction
        for SHAP imputers (Frye et al. 2021; Chen et al. 2023).

        Args:
            J: Number of joints.
            rank: Effective rank of the joint covariance. Must satisfy
                ``1 <= rank <= J``.
            eps: Isotropic noise floor added to ``U U^T`` for PSD stability
                and to model thin off-manifold dispersion.
            seed: Random seed for ``U``.

        Returns:
            ``(J, J)`` float64 symmetric PSD array of rank ``J`` (full rank
            with thin spectrum below ``rank``).

        Raises:
            ValueError: if ``rank < 1`` or ``rank > J``.
        """
        if rank < 1 or rank > J:
            raise ValueError(f"rank must be in [1, J={J}]; got {rank}.")
        rng = np.random.default_rng(seed)
        U = rng.standard_normal((J, rank)) / np.sqrt(rank)
        Sigma = U @ U.T + eps * np.eye(J)
        return 0.5 * (Sigma + Sigma.T)

    @staticmethod
    def data_driven(X_subset: np.ndarray) -> np.ndarray:
        """Estimate Sigma_joints from data via Ledoit-Wolf shrinkage.

        Computes the per-sample per-joint grand mean (averaged over F and T),
        then fits a Ledoit-Wolf shrinkage estimator to the resulting (N, J)
        matrix.

        Args:
            X_subset: ``(N, J)`` array of per-joint per-sample means, or
                ``(N, J, F, T)`` array (grand mean over F, T is taken
                automatically).

        Returns:
            ``(J, J)`` float64 Ledoit-Wolf covariance matrix.

        Raises:
            ImportError: if ``scikit-learn`` is not installed.
        """
        from sklearn.covariance import LedoitWolf  # noqa: PLC0415

        arr = np.asarray(X_subset, dtype=np.float64)
        if arr.ndim == 4:
            arr = arr.mean(axis=(2, 3))  # (N, J)
        if arr.ndim != 2:
            raise ValueError(
                f"X_subset must be (N, J) or (N, J, F, T); got shape {arr.shape}."
            )
        lw = LedoitWolf(assume_centered=False)
        lw.fit(arr)
        return lw.covariance_


class SigmaTimeFactory:
    """Factory for temporal (T×T) covariance matrices.

    Every method returns a PSD matrix suitable for use as ``Sigma_time``
    in :class:`GaussianMotionBenchmark`.
    """

    @staticmethod
    def ar1(T: int, alpha: float) -> np.ndarray:
        """Build a T×T AR(1) covariance matrix.

        ``C[t, t'] = alpha^|t-t'|``.

        Args:
            T: Number of time-steps.
            alpha: AR(1) autocorrelation coefficient.

        Returns:
            ``(T, T)`` float64 PSD array.
        """
        return ar1_cov(T, alpha)

    @staticmethod
    def ar_p(T: int, alphas: list[float]) -> np.ndarray:
        """Build a T×T AR(p) covariance matrix.

        Computes ``C = sum_{k=1}^{p} alpha_k * J_k`` where ``J_k`` is the
        symmetric Toeplitz matrix with 1 on the k-th off-diagonal and 0
        elsewhere.  The result is symmetrised and a small diagonal ridge
        ``1e-6 * I`` is added for PSD stability.

        Args:
            T: Number of time-steps.
            alphas: List of p lag coefficients ``[alpha_1, ..., alpha_p]``.

        Returns:
            ``(T, T)`` float64 PSD-ish array.

        Raises:
            ValueError: if ``alphas`` is empty.
        """
        if not alphas:
            raise ValueError("alphas must be a non-empty list.")
        C = np.eye(T, dtype=np.float64)
        for k, ak in enumerate(alphas, start=1):
            if k >= T:
                break
            # k-th off-diagonal Toeplitz contribution.
            diag = np.ones(T - k, dtype=np.float64)
            J_k = np.diag(diag, k) + np.diag(diag, -k)
            C = C + ak * J_k
        # Symmetrise and regularise.
        C = 0.5 * (C + C.T)
        C += 1e-6 * np.eye(T, dtype=np.float64)
        return C

    @staticmethod
    def gait_periodic(T: int, period: float, n_harmonics: int = 3) -> np.ndarray:
        """Build a T×T sum-of-cosines (gait-periodic) kernel matrix.

        ``k(t, t') = sum_{h=1}^{n_harmonics} cos(2*pi*h*|t-t'| / period)``.
        The result is symmetrised and a small ridge ``1e-6 * I`` is added
        for numerical PSD stability.

        Args:
            T: Number of time-steps.
            period: Gait cycle period in frames.
            n_harmonics: Number of cosine harmonics to sum.

        Returns:
            ``(T, T)`` float64 PSD array.

        Raises:
            ValueError: if ``period <= 0`` or ``n_harmonics < 1``.
        """
        if period <= 0:
            raise ValueError(f"period must be positive; got {period}.")
        if n_harmonics < 1:
            raise ValueError(f"n_harmonics must be >= 1; got {n_harmonics}.")
        t = np.arange(T, dtype=np.float64)
        diff = np.abs(t[:, None] - t[None, :])  # (T, T)
        C = np.zeros((T, T), dtype=np.float64)
        for h in range(1, n_harmonics + 1):
            C += np.cos(2.0 * np.pi * h * diff / period)
        C = 0.5 * (C + C.T)
        C += 1e-6 * np.eye(T, dtype=np.float64)
        return C


# ---------------------------------------------------------------------------
# Core benchmark class (sampling + conditional sampling)
# ---------------------------------------------------------------------------


class GaussianMotionBenchmark:
    """Gaussian motion generative model with exact conditional sampling.

    Data model: ``x ~ N(0, Sigma_joints ⊗ I_F ⊗ Sigma_time)``.

    This class handles all covariance logic and sampling.  The label
    function and classifier live in Tasks 1D and 4A respectively.

    Args:
        J: Number of skeletal joints.
        F: Number of coordinates per joint (e.g. 3 for xyz).
        T: Number of frames per sequence.
        rho: Off-diagonal equicorrelation for the default joint covariance.
            Ignored when ``sigma_joints`` is provided.
        alpha: AR(1) temporal autocorrelation for the default temporal
            covariance.  Ignored when ``sigma_time`` is provided.
        K: Number of temporal windows for conditional-covariance
            precomputation.  Only affects caching; does not restrict usage.
        sigma_joints: Custom ``(J, J)`` PSD joint covariance.  If ``None``
            (default), uses ``SigmaJointsFactory.equicorrelated(J, rho)``.
        sigma_time: Custom ``(T, T)`` PSD temporal covariance.  If ``None``
            (default), uses ``SigmaTimeFactory.ar1(T, alpha)``.
        sigma_joints_source: Human-readable provenance tag for ``sigma_joints``.
        sigma_time_source: Human-readable provenance tag for ``sigma_time``.
    """

    def __init__(
        self,
        J: int = 17,
        F: int = 3,
        T: int = 81,
        rho: float = 0.5,
        alpha: float = 0.8,
        K: int = 4,
        sigma_joints: np.ndarray | None = None,
        sigma_time: np.ndarray | None = None,
        sigma_joints_source: str | None = None,
        sigma_time_source: str | None = None,
    ) -> None:
        if K <= 0:
            raise ValueError(f"K must be positive; got {K}.")

        self.J = J
        self.F = F
        self.T = T
        self.rho = rho
        self.alpha = alpha
        self.K = K

        # Joint covariance.
        if sigma_joints is None:
            self.Sigma_joints: np.ndarray = equicorr(J, rho)
            self.sigma_joints_source: str = sigma_joints_source or "equicorr"
        else:
            sj = np.asarray(sigma_joints, dtype=np.float64)
            if sj.shape != (J, J):
                raise ValueError(
                    f"sigma_joints shape {sj.shape} does not match J={J}."
                )
            sj = 0.5 * (sj + sj.T)
            eig_min = float(np.linalg.eigvalsh(sj).min())
            if eig_min < -1e-6:
                raise ValueError(
                    f"sigma_joints is not PSD (min eigenvalue {eig_min:.3e})."
                )
            self.Sigma_joints = sj
            self.sigma_joints_source = sigma_joints_source or "custom"

        # Temporal covariance.
        if sigma_time is None:
            self.Sigma_time: np.ndarray = ar1_cov(T, alpha)
            self.sigma_time_source: str = sigma_time_source or "ar1"
        else:
            st = np.asarray(sigma_time, dtype=np.float64)
            if st.shape != (T, T):
                raise ValueError(
                    f"sigma_time shape {st.shape} does not match T={T}."
                )
            st = 0.5 * (st + st.T)
            eig_min = float(np.linalg.eigvalsh(st).min())
            if eig_min < -1e-6:
                raise ValueError(
                    f"sigma_time is not PSD (min eigenvalue {eig_min:.3e})."
                )
            self.Sigma_time = st
            self.sigma_time_source = sigma_time_source or "custom"

        # Cholesky factors for unconditional sampling.
        self.L_joints: np.ndarray = np.linalg.cholesky(
            self.Sigma_joints + 1e-8 * np.eye(J)
        )
        self.L_time: np.ndarray = np.linalg.cholesky(
            self.Sigma_time + 1e-8 * np.eye(T)
        )

        # Window assignments for K equal-width windows.
        quarter = T // K
        self.window_assignments: list[list[int]] = [
            list(range(k * quarter, (k + 1) * quarter if k < K - 1 else T))
            for k in range(K)
        ]

        # Cache for conditional parameters (keyed by observation pattern).
        self._cond_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Unconditional sampling
    # ------------------------------------------------------------------

    def sample(self, N: int, seed: int | None = None) -> np.ndarray:
        """Sample N sequences from ``N(0, Sigma_joints ⊗ I_F ⊗ Sigma_time)``.

        Args:
            N: Number of sequences to draw.
            seed: Optional random seed.

        Returns:
            ``(N, J, F, T)`` float32 array.
        """
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((N, self.J, self.F, self.T)).astype(np.float64)
        # Apply temporal Cholesky: x_t = L_time @ z_t (per j, f).
        x = np.einsum("tT,njfT->njft", self.L_time, z)
        # Apply joint Cholesky: x_j = L_joints @ x_J (per f, t).
        x = np.einsum("jJ,nJft->njft", self.L_joints, x)
        return x.astype(np.float32)

    # ------------------------------------------------------------------
    # Conditional parameter computation (cached)
    # ------------------------------------------------------------------

    def _window_frames(self, window_indices: list[int]) -> np.ndarray:
        """Return concatenated frame indices for the given windows.

        Args:
            window_indices: List of window indices (0-based).

        Returns:
            1-D int array of frame indices.
        """
        return np.concatenate([self.window_assignments[k] for k in window_indices])

    def _compute_cond_params_temporal(
        self,
        t_obs: np.ndarray,
        t_hid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute temporal conditional Gaussian parameters.

        Under the Kronecker model, the temporal conditional is the same
        for all (j, f) slices:

            W             = Sigma_time[hid, obs] @ inv(Sigma_time[obs, obs])
            Sigma_hid_cond = Sigma_time[hid, hid] - W @ Sigma_time[obs, hid]

        Args:
            t_obs: ``(n_obs,)`` int array of observed time indices.
            t_hid: ``(n_hid,)`` int array of hidden time indices.

        Returns:
            L_cond: ``(n_hid, n_hid)`` Cholesky of the conditional
                temporal covariance.
            W_mean: ``(n_hid, n_obs)`` conditional-mean weight matrix.
        """
        Soo = self.Sigma_time[np.ix_(t_obs, t_obs)]
        Shh = self.Sigma_time[np.ix_(t_hid, t_hid)]
        Sho = self.Sigma_time[np.ix_(t_hid, t_obs)]
        W = Sho @ np.linalg.solve(Soo + 1e-10 * np.eye(len(t_obs)), np.eye(len(t_obs)))
        Sigma_cond = Shh - W @ Sho.T
        Sigma_cond = 0.5 * (Sigma_cond + Sigma_cond.T)
        Sigma_cond += 1e-8 * np.eye(len(t_hid))
        L_cond = np.linalg.cholesky(Sigma_cond)
        return L_cond, W

    def _compute_cond_params_spatial(
        self,
        j_obs: tuple[int, ...],
        j_hid: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute spatial (joint-level) conditional Gaussian parameters.

        Under the Kronecker model, the joint conditional is the same for
        all (f, t) slices:

            W             = Sigma_joints[hid, obs] @ inv(Sigma_joints[obs, obs])
            Sigma_hid_cond = Sigma_joints[hid, hid] - W @ Sigma_joints[obs, hid]

        Args:
            j_obs: Tuple of observed joint indices.
            j_hid: Tuple of hidden joint indices.

        Returns:
            L_cond: ``(n_hid, n_hid)`` Cholesky of the conditional joint
                covariance.
            W_mean: ``(n_hid, n_obs)`` conditional-mean weight matrix.
        """
        j_obs_a = np.asarray(j_obs, dtype=int)
        j_hid_a = np.asarray(j_hid, dtype=int)
        Soo = self.Sigma_joints[np.ix_(j_obs_a, j_obs_a)]
        Shh = self.Sigma_joints[np.ix_(j_hid_a, j_hid_a)]
        Sho = self.Sigma_joints[np.ix_(j_hid_a, j_obs_a)]
        W = Sho @ np.linalg.solve(
            Soo + 1e-10 * np.eye(len(j_obs_a)), np.eye(len(j_obs_a))
        )
        Sigma_cond = Shh - W @ Sho.T
        Sigma_cond = 0.5 * (Sigma_cond + Sigma_cond.T)
        Sigma_cond += 1e-8 * np.eye(len(j_hid_a))
        L_cond = np.linalg.cholesky(Sigma_cond)
        return L_cond, W

    # ------------------------------------------------------------------
    # Conditional sampling — general (J, F, T) mask
    # ------------------------------------------------------------------

    def conditional_sample_from_mask(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        n_samples: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw n_samples from the exact conditional p(x_hid | x_obs).

        Dispatches to the efficient Kronecker formula for pure-temporal
        or pure-spatial masks; falls back to the general spatiotemporal
        formula otherwise.

        Args:
            x: ``(J, F, T)`` conditioning sequence.
            mask: ``(J, F, T)`` bool — ``True`` = observed.
            n_samples: Number of samples to draw.
            rng: Numpy random Generator.

        Returns:
            ``(n_samples, J, F, T)`` float32 array.

        Raises:
            ValueError: if ``x.shape`` or ``mask.shape`` are inconsistent
                with ``(J, F, T)``.
        """
        if rng is None:
            rng = np.random.default_rng()
        J, F, T = self.J, self.F, self.T
        x = np.asarray(x, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        if x.shape != (J, F, T):
            raise ValueError(f"x.shape {x.shape} != (J={J}, F={F}, T={T}).")
        if mask.shape != (J, F, T):
            raise ValueError(f"mask.shape {mask.shape} != (J={J}, F={F}, T={T}).")

        # Detect mask structure.
        mask_jt = mask.all(axis=1)  # (J, T) — True if all F share same obs status
        is_temporal = (mask_jt == mask_jt[0:1, :]).all() and (mask.all(axis=0) == mask[0]).all()
        is_spatial = (mask == mask[:, 0:1, 0:1]).all()

        if is_temporal:
            return self._conditional_sample_temporal_mask(x, mask, n_samples, rng)
        if is_spatial:
            return self._conditional_sample_spatial_mask(x, mask, n_samples, rng)
        return self._conditional_sample_spatiotemporal(x, mask, n_samples, rng)

    def _conditional_sample_temporal_mask(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Temporal-only conditional sample (Kronecker-efficient).

        Args:
            x: ``(J, F, T)`` float64 conditioning sequence.
            mask: ``(J, F, T)`` bool with uniform pattern across all j, f.
            n_samples: Number of samples.
            rng: Numpy random Generator.

        Returns:
            ``(n_samples, J, F, T)`` float32 array.
        """
        J, F, _T = self.J, self.F, self.T
        t_obs = np.flatnonzero(mask[0, 0, :])
        t_hid = np.flatnonzero(~mask[0, 0, :])
        n_hid = len(t_hid)

        if n_hid == 0:
            return np.tile(x[None], (n_samples, 1, 1, 1)).astype(np.float32)
        if len(t_obs) == 0:
            return self.sample(n_samples, seed=int(rng.integers(1 << 31)))

        key = ("temporal", tuple(t_obs.tolist()), tuple(t_hid.tolist()))
        if key not in self._cond_cache:
            self._cond_cache[key] = self._compute_cond_params_temporal(t_obs, t_hid)
        L_cond, W_mean = self._cond_cache[key]

        # Conditional mean: (J, F, n_hid).
        mu = np.einsum("ht,jft->jfh", W_mean, x[:, :, t_obs])

        # Noise with covariance Sigma_joints ⊗ I_F ⊗ Sigma_hid_cond.
        noise_t = rng.standard_normal((n_samples, J, F, n_hid)).astype(np.float64)
        noise_t = np.einsum("tT,njfT->njft", L_cond, noise_t)
        noise_t = np.einsum("jJ,nJft->njft", self.L_joints, noise_t)

        out = np.tile(x[None], (n_samples, 1, 1, 1)).astype(np.float64)
        out[:, :, :, t_hid] = mu[None] + noise_t
        return out.astype(np.float32)

    def _conditional_sample_spatial_mask(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Spatial-only conditional sample (Kronecker-efficient).

        Args:
            x: ``(J, F, T)`` float64 conditioning sequence.
            mask: ``(J, F, T)`` bool with uniform pattern across all f, t.
            n_samples: Number of samples.
            rng: Numpy random Generator.

        Returns:
            ``(n_samples, J, F, T)`` float32 array.
        """
        _J, F, T = self.J, self.F, self.T
        j_obs = tuple(int(j) for j in np.flatnonzero(mask[:, 0, 0]))
        j_hid = tuple(int(j) for j in np.flatnonzero(~mask[:, 0, 0]))
        n_hid = len(j_hid)

        if n_hid == 0:
            return np.tile(x[None], (n_samples, 1, 1, 1)).astype(np.float32)
        if len(j_obs) == 0:
            return self.sample(n_samples, seed=int(rng.integers(1 << 31)))

        key = ("spatial", j_obs, j_hid)
        if key not in self._cond_cache:
            self._cond_cache[key] = self._compute_cond_params_spatial(j_obs, j_hid)
        L_cond_j, W = self._cond_cache[key]

        j_obs_a = np.asarray(j_obs, dtype=int)
        j_hid_a = np.asarray(j_hid, dtype=int)

        # Conditional mean: (n_hid, F, T).
        mu = np.einsum("ho,oft->hft", W, x[j_obs_a, :, :])

        # Noise with covariance Sigma_hid_cond ⊗ I_F ⊗ Sigma_time.
        noise = rng.standard_normal((n_samples, n_hid, F, T)).astype(np.float64)
        noise = np.einsum("tT,nhfT->nhft", self.L_time, noise)
        noise = np.einsum("hH,nHft->nhft", L_cond_j, noise)

        out = np.tile(x[None], (n_samples, 1, 1, 1)).astype(np.float64)
        out[:, j_hid_a, :, :] = mu[None] + noise
        return out.astype(np.float32)

    def _conditional_sample_spatiotemporal(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """General spatiotemporal conditional sample.

        For arbitrary (J, F, T) masks.  The F dimension is i.i.d. by
        ``I_F`` so each F-slice is conditionally independent; a single
        set of (W, L_cond) is computed from the (J, T) pattern shared
        across all F slices (the F-axis must share the same observation
        pattern at each (j, t) for this to be exact).

        Args:
            x: ``(J, F, T)`` float64 conditioning sequence.
            mask: ``(J, F, T)`` bool mask.
            n_samples: Number of samples.
            rng: Numpy random Generator.

        Returns:
            ``(n_samples, J, F, T)`` float32 array.
        """
        _J, F, T = self.J, self.F, self.T

        # Use the (J, T) pattern from the first F-slice (same for all f).
        jt_mask = mask.all(axis=1)  # (J, T) — True if all F agree on observed
        # Fall back to OR across F if pattern is not uniform.
        if not (mask == mask[:, 0:1, :]).all():
            jt_mask = mask.any(axis=1)

        flat = jt_mask.reshape(-1)
        obs_lin = np.flatnonzero(flat)
        hid_lin = np.flatnonzero(~flat)
        n_obs = int(obs_lin.size)
        n_hid = int(hid_lin.size)

        if n_hid == 0:
            return np.tile(x[None], (n_samples, 1, 1, 1)).astype(np.float32)
        if n_obs == 0:
            return self.sample(n_samples, seed=int(rng.integers(1 << 31)))

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


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------


class GaussianMotionDataset:
    """Pre-sampled Gaussian motion dataset conforming to GroundTruthDataset.

    Wraps :class:`GaussianMotionBenchmark` and pre-generates ``N`` sequences
    at construction time.  Labels are simple quantile-bins of the first
    joint's grand mean (a deterministic proxy label; Task 1D provides the
    Olsen-style nonlinear labels).

    Conforms to :class:`~motionbench.data.base.GroundTruthDataset` (structural
    subtyping — no inheritance required).

    Args:
        J: Number of skeletal joints.
        F: Number of coordinates per joint.
        T: Number of frames per sequence.
        N: Number of sequences to pre-generate.
        rho: Off-diagonal equicorrelation for the default joint covariance.
        alpha: AR(1) temporal autocorrelation for the default temporal covariance.
        K: Number of temporal windows (stored in benchmark; affects caching).
        sigma_joints: Custom ``(J, J)`` joint covariance.  If ``None`` uses
            ``SigmaJointsFactory.equicorrelated(J, rho)``.
        sigma_time: Custom ``(T, T)`` temporal covariance.  If ``None`` uses
            ``SigmaTimeFactory.ar1(T, alpha)``.
        seed: Random seed for sequence generation.
    """

    def __init__(
        self,
        J: int = 17,
        F: int = 3,
        T: int = 81,
        N: int = 1000,
        rho: float = 0.5,
        alpha: float = 0.8,
        K: int = 4,
        sigma_joints: np.ndarray | None = None,
        sigma_time: np.ndarray | None = None,
        seed: int | None = None,
        label_fn: object | None = None,
    ) -> None:
        self._benchmark = GaussianMotionBenchmark(
            J=J,
            F=F,
            T=T,
            rho=rho,
            alpha=alpha,
            K=K,
            sigma_joints=sigma_joints,
            sigma_time=sigma_time,
        )
        x_np = self._benchmark.sample(N, seed=seed)

        if label_fn is not None:
            y_np = np.asarray(label_fn(x_np), dtype=np.int64)
            if y_np.shape != (N,):
                raise ValueError(
                    f"label_fn returned shape {y_np.shape}; expected ({N},)."
                )
        else:
            score = x_np[:, 0, :, :].mean(axis=(1, 2))
            q33, q67 = np.percentile(score, [33.0, 67.0])
            y_np = np.where(score < q33, 0, np.where(score < q67, 1, 2)).astype(np.int64)

        self._x: Tensor = torch.tensor(x_np, dtype=torch.float32)  # (N, J, F, T)
        self._y: Tensor = torch.tensor(y_np, dtype=torch.int64)    # (N,)
        self._N = N
        self._J = J
        self._F = F
        self._T = T

        # Lazily import to avoid circular dependency at module level.
        from motionbench.oracles.gaussian_oracle import GaussianOracle  # noqa: PLC0415

        self._oracle: GaussianOracle = GaussianOracle(
            Sigma_joints=self._benchmark.Sigma_joints,
            Sigma_time=self._benchmark.Sigma_time,
        )

    # ------------------------------------------------------------------
    # GroundTruthDataset protocol
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return the ``idx``-th ``(x, y)`` pair.

        Args:
            idx: Sample index in ``[0, N)``.

        Returns:
            x: ``(J, F, T)`` float32 Tensor.
            y: scalar int64 Tensor.
        """
        return self._x[idx], self._y[idx]

    def __len__(self) -> int:
        """Number of pre-sampled sequences.

        Returns:
            N.
        """
        return self._N

    @property
    def shape(self) -> tuple[int, int, int]:
        """Spatial shape of every sample.

        Returns:
            ``(J, F, T)`` tuple.
        """
        return (self._J, self._F, self._T)

    @property
    def metadata(self) -> dict[str, object]:
        """Dataset-level metadata.

        Returns:
            Dict with keys ``skeleton``, ``frame_rate``, ``rho``, ``alpha``,
            ``K``, ``sigma_joints_source``, ``sigma_time_source``.
        """
        bench = self._benchmark
        return {
            "skeleton": "synthetic_gaussian",
            "frame_rate": 27.0,
            "rho": bench.rho,
            "alpha": bench.alpha,
            "K": bench.K,
            "sigma_joints_source": bench.sigma_joints_source,
            "sigma_time_source": bench.sigma_time_source,
        }

    @property
    def oracle(self) -> GaussianOracle:
        """Ground-truth :class:`~motionbench.oracles.gaussian_oracle.GaussianOracle`.

        Returns:
            The oracle instance for this dataset.
        """
        return self._oracle
