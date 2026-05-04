"""motionbench.imputers.base — BaseImputer abstract base class.

An *imputer* (or *completion model*) defines the distribution
``q(x_hid | x_obs)`` used to estimate the KernelSHAP conditional-expectation
value function:

.. math::

    v(S) = \\mathbb{E}_{x_{\\bar{S}} \\sim q(\\cdot\\,|\\,x_S)}
           \\bigl[ f(x_S,\\, x_{\\bar{S}}) \\bigr]

The quality of Shapley attributions depends heavily on how well ``q``
approximates the true conditional ``p(x_hid | x_obs)``.  The motionbench
benchmark explicitly measures this via the ground-truth metrics EC1–EC3.

Imputer taxonomy (Aas et al. 2021, Table 1)
--------------------------------------------
Off-manifold (fast, biased):
    ``ZeroImputer``             — fill with zeros.
    ``MeanImputer``             — fill with per-coordinate training mean.
    ``MarginalDonorImputer``    — draw from unconditional marginal (intervention).
    ``GaussianNoiseImputer``    — fill with mean + Gaussian noise.

Classical conditional:
    ``KNNConditionalImputer``   — k-NN over observed coordinates.
    ``EmpiricalConditionalImputer`` — Gaussian-kernel weighting (shapr default).
    ``VineCopulaImputer``       — Gaussian / non-parametric copula.

Learned on-manifold:
    ``VAEACImputer``            — amortised variational (Ivanov et al. 2019).
    ``FlowMatchingImputer``     — OT-flow with RePaint harmonisation.

Oracle (perfect, ground-truth):
    :class:`~motionbench.oracles.base.Oracle` — implements ``impute`` directly
    for use as a "perfect imputer" in EC1–EC3 evaluation.

API contract
------------
*Observed entries are always preserved bit-for-bit.*  Every implementation
must guarantee that for all ``i`` in ``[0, n_samples)`` and all coordinates
where ``mask == True``:  ``output[i, mask] == x_obs[mask]``.

References
----------
Aas, Jullum & Løland (2021) "Explaining individual predictions when features
are dependent: More accurate approximations to Shapley values."
arXiv:1903.10464.

Olsen et al. (2022) "Using Shapley Values and Variational Autoencoders To
Explain Predictions from Neural Networks for Short-Term Wind Power
Forecasting." JMLR 23(1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    from motionbench.data.base import BaseDataset


__all__ = ["BaseImputer"]


class BaseImputer(ABC):
    """Abstract base class for all imputers / completion models.

    Shape conventions:

    * Per-sample input:    ``(J, F, T)`` float32 Tensor.
    * Element-level mask:  ``(J, F, T)`` bool Tensor.  ``True`` = observed.
    * Batch output:        ``(n_samples, J, F, T)`` float32 Tensor.

    Fitting
    -------
    ``fit`` must be called before ``impute``.  Off-manifold imputers may
    implement ``fit`` as a no-op (just returning ``self``).  Learned
    imputers store trained weights; the training loop lives in
    ``scripts/train_vaeac.py`` / ``scripts/train_flow.py`` and calls
    ``fit`` on a ``BaseDataset``.

    Serialisation
    -------------
    All implementations that require a training phase must also implement
    ``save(path)`` and ``load(path)`` class methods so that checkpoints can
    be committed and loaded without re-running training.  This is not
    enforced by the ABC to avoid over-specifying the checkpoint format.
    """

    @abstractmethod
    def fit(self, train_data: "BaseDataset") -> "BaseImputer":
        """Fit the imputer to a training dataset.

        Off-manifold imputers (Zero, Mean, Marginal) should compute any
        required statistics (e.g. per-coordinate mean) here and return
        ``self``.  Learned imputers (VAEAC, Flow) run their training loop
        from the corresponding ``scripts/`` entry point and call this method
        to wire the result.

        Args:
            train_data: A :class:`~motionbench.data.base.BaseDataset`.

        Returns:
            ``self`` — for method chaining (``imputer.fit(ds).impute(…)``).
        """

    @abstractmethod
    def impute(
        self,
        x_obs: Tensor,
        mask: Tensor,
        n_samples: int,
        seed: int | None = None,
    ) -> Tensor:
        """Draw *n_samples* completions of ``x_obs`` given the observed mask.

        The *observed entries are preserved bit-for-bit* in every returned
        sample.  Implementations must guarantee:

            ``output[:, mask] == x_obs[mask]``  (within floating-point equality)

        Args:
            x_obs: ``(J, F, T)`` float32 sequence.  Entries where
                ``mask == False`` may be arbitrary; implementations must
                ignore them.
            mask: ``(J, F, T)`` bool tensor.  ``True`` = coordinate is
                observed and must be preserved in the output.
            n_samples: Number of completed sequences to return.
            seed: Optional random seed for reproducibility.

        Returns:
            ``(n_samples, J, F, T)`` float32 tensor.

        Raises:
            RuntimeError: if ``fit`` has not been called.
            ValueError: if ``x_obs.shape != mask.shape`` or shapes are
                inconsistent with the fitted imputer.
        """

    # ------------------------------------------------------------------
    # Introspection helpers (optional overrides)
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Short string identifier for logging and leaderboard tables."""
        return self.__class__.__name__

    @property
    def is_on_manifold(self) -> bool:
        """Whether this imputer attempts to sample from the data manifold.

        ``True`` for learned imputers (VAEAC, Flow, KNN, Empirical);
        ``False`` for off-manifold baselines (Zero, Mean, Marginal, Noise).
        """
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
