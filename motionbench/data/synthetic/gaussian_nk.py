"""motionbench.data.synthetic.gaussian_nk — Non-Kronecker Gaussian motion dataset.

Reviewer concern C2 ablation: the standard ``GaussianMotionDataset`` uses a
Kronecker-separable covariance ``Sigma_joints ⊗ I_F ⊗ Sigma_time``.  This
module generates a **perturbed** covariance that breaks Kronecker separability
via a low-rank random perturbation:

    Sigma_kron    = Sigma_joints ⊗ I_F ⊗ Sigma_time   (same J=5, F=3, T=16)
    R_low_rank    = U @ U.T  where  U ~ N(0, 1),  shape (J*F*T, r=3)
    lambda        = 0.3 * trace(Sigma_kron) / trace(R_low_rank)
    Sigma_full_nk = Sigma_kron + lambda * R_low_rank   (symmetrised + ridge)

The resulting distribution has *cross-joint-time interaction terms* absent
from any Kronecker product, providing a direct test of Kronecker-structure
sensitivity.

DATA MODEL
----------
    x ~ N(0, Sigma_full_nk),   x ∈ R^{J × F × T}
    flat layout: C-order (j * F * T + f * T + t)

CONFORMS TO
-----------
:class:`~motionbench.data.base.GroundTruthDataset` protocol (structural,
no inheritance required).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from motionbench.utils.coalitions import ar1_cov, equicorr

__all__ = ["GaussianNKDataset"]


class GaussianNKDataset:
    """Pre-sampled non-Kronecker Gaussian motion dataset.

    Covariance is a Kronecker base plus a normalised low-rank perturbation
    (``lambda * U @ U.T``) that introduces cross-joint-time dependencies
    absent from the baseline ``GaussianMotionDataset``.

    Args:
        J: Number of skeletal joints.
        F: Number of coordinates per joint.
        T: Number of frames per sequence.
        N: Number of sequences to pre-generate.
        K: Number of temporal windows (players).
        rho: Off-diagonal equicorrelation for ``Sigma_joints``.
        alpha: AR(1) autocorrelation for ``Sigma_time``.
        r: Rank of the low-rank perturbation.
        lam_frac: Fraction of ``trace(Sigma_kron)`` used to scale the
            perturbation (0.3 = 30 % perturbation).
        seed: Random seed for both the perturbation and dataset sampling.
        label_fn: Optional callable ``(N, J, F, T) → (N,)`` int64.
            If ``None``, uses quantile-bin labels on joint-0 grand mean
            (same default as ``GaussianMotionDataset``).
    """

    def __init__(
        self,
        J: int = 5,
        F: int = 3,
        T: int = 16,
        N: int = 400,
        K: int = 4,
        rho: float = 0.5,
        alpha: float = 0.8,
        r: int = 3,
        lam_frac: float = 0.3,
        seed: int = 42,
        label_fn: object | None = None,
    ) -> None:
        self._J = J
        self._F = F
        self._T = T
        self._K = K
        D = J * F * T

        # ------------------------------------------------------------------ #
        # Build non-Kronecker covariance                                       #
        # ------------------------------------------------------------------ #
        rng = np.random.default_rng(seed)

        Sigma_joints: np.ndarray = equicorr(J, rho)
        Sigma_time: np.ndarray = ar1_cov(T, alpha)

        # Kronecker base: Sigma_joints ⊗ I_F ⊗ Sigma_time
        Sigma_kron: np.ndarray = np.kron(np.kron(Sigma_joints, np.eye(F)), Sigma_time)

        # Low-rank perturbation: U ~ N(0, 1), shape (D, r)
        U = rng.standard_normal((D, r))
        R_lr = U @ U.T  # (D, D)  — positive semidefinite
        lam = lam_frac * float(np.trace(Sigma_kron)) / (float(np.trace(R_lr)) + 1e-10)
        Sigma_full_nk = Sigma_kron + lam * R_lr

        # Ensure PSD: symmetrise + tiny ridge
        Sigma_full_nk = 0.5 * (Sigma_full_nk + Sigma_full_nk.T)
        eps = 1e-6
        Sigma_full_nk += eps * np.eye(D)

        # Verify PSD (warn if not, but do not crash)
        eig_min = float(np.linalg.eigvalsh(Sigma_full_nk).min())
        if eig_min < -1e-5:
            # Increase ridge to fix
            Sigma_full_nk += (-eig_min + 1e-4) * np.eye(D)

        # ------------------------------------------------------------------ #
        # Sample sequences                                                     #
        # ------------------------------------------------------------------ #
        L_full = np.linalg.cholesky(Sigma_full_nk)
        z = rng.standard_normal((N, D))
        x_np = (z @ L_full.T).reshape(N, J, F, T).astype(np.float32)

        # ------------------------------------------------------------------ #
        # Labels                                                               #
        # ------------------------------------------------------------------ #
        if label_fn is not None:
            y_np = np.asarray(label_fn(x_np), dtype=np.int64)
            if y_np.shape != (N,):
                raise ValueError(
                    f"label_fn returned shape {y_np.shape}; expected ({N},)."
                )
        else:
            score = x_np[:, 0, :, :].mean(axis=(1, 2))
            q33, q67 = np.percentile(score, [33.0, 67.0])
            y_np = np.where(
                score < q33, 0, np.where(score < q67, 1, 2)
            ).astype(np.int64)

        self._x: Tensor = torch.tensor(x_np, dtype=torch.float32)
        self._y: Tensor = torch.tensor(y_np, dtype=torch.int64)
        self._N = N

        # Store covariance matrices for oracle construction and sanity checks
        self.Sigma_full_nk: np.ndarray = Sigma_full_nk
        self.Sigma_kron: np.ndarray = Sigma_kron
        self.Sigma_joints: np.ndarray = Sigma_joints
        self.Sigma_time: np.ndarray = Sigma_time
        self.lam: float = float(lam)
        self.perturbation_frac: float = float(lam * np.trace(R_lr) / np.trace(Sigma_kron))

    # ------------------------------------------------------------------ #
    # GroundTruthDataset protocol                                         #
    # ------------------------------------------------------------------ #

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self._x[idx], self._y[idx]

    def __len__(self) -> int:
        return self._N

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self._J, self._F, self._T)

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "skeleton": "synthetic_gaussian_nk",
            "frame_rate": 27.0,
            "K": self._K,
            "n_classes": 3,
            "covariance": "non_kronecker",
            "perturbation_frac": self.perturbation_frac,
        }

    @property
    def oracle(self) -> object:
        """Ground-truth :class:`~motionbench.oracles.full_gaussian_oracle.FullGaussianOracle`."""
        from motionbench.oracles.full_gaussian_oracle import FullGaussianOracle  # noqa: PLC0415
        return FullGaussianOracle(
            Sigma_full=self.Sigma_full_nk,
            J=self._J,
            F=self._F,
            T=self._T,
        )
