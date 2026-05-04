"""motionbench.oracles.base — Oracle abstract base class.

An *oracle* is a ground-truth model of the data distribution ``p(x)``.
On synthetic datasets with known generative models (Gaussian, copula,
Fourier-gait), the oracle provides:

1. **Exact conditional sampling** — ``p(x_hid | x_obs)`` in closed form.
2. **True Shapley values** — computed from the exact value function
   ``v(S) = E_{x_{\\bar{S}} \\sim p(·|x_S)}[f(x_S, x_{\\bar{S}})]``
   via full coalition enumeration (M ≤ 12) or estimation.

An oracle also satisfies the :class:`~motionbench.imputers.base.BaseImputer`
interface: ``conditional_sample`` has the same signature as
``BaseImputer.impute``, making the oracle a "perfect imputer" for
ground-truth comparisons.

Implementations
---------------
- :class:`~motionbench.oracles.gaussian_oracle.GaussianOracle` — exact
  closed-form conditional via the Gaussian conditional formula
  μ_{a|b} = μ_a + Σ_{ab} Σ_{bb}⁻¹ (x_b − μ_b).
- :class:`~motionbench.oracles.copula_oracle.CopulaOracle` — Gaussian
  copula with pluggable marginals (Burr XII, StudentT, SkewNormal, MoG).
- :class:`~motionbench.oracles.monte_carlo_oracle.MonteCarloOracle` —
  black-box oracle that estimates v(S) via large-N rejection sampling;
  use only for verification / unit tests.

References
----------
Aas, Jullum & Løland (2021) "Explaining individual predictions when
features are dependent" — §3 conditional expectation game.

Lundberg & Lee (2017) "A unified approach to interpreting model predictions"
— KernelSHAP, Shapley axioms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

from torch import Tensor

if TYPE_CHECKING:
    from motionbench.players.base import PlayerSet


__all__ = ["Oracle"]


class Oracle(ABC):
    """Abstract base class for all ground-truth oracles.

    Shape conventions (same as throughout motionbench):

    * Per-sample coordinate layout: ``(J, F, T)`` — joints, features, time.
    * Batch layout: ``(N, J, F, T)``.
    * Boolean mask: ``(J, F, T)`` — ``True`` = coordinate is *observed*.
    * Per-player output: ``(M,)`` Shapley values.
    """

    @abstractmethod
    def conditional_sample(
        self,
        x_obs: Tensor,
        mask: Tensor,
        n: int,
        seed: int | None = None,
    ) -> Tensor:
        """Draw *n* samples from the exact conditional ``p(x_hid | x_obs)``.

        Observed coordinates (``mask == True``) are copied bit-for-bit into
        every returned sample; hidden coordinates (``mask == False``) are drawn
        from the true conditional.

        This method has the same signature as
        :py:meth:`~motionbench.imputers.base.BaseImputer.impute`, making
        the oracle directly usable as a "perfect imputer" in attribution
        pipelines.

        Args:
            x_obs: ``(J, F, T)`` float32 sequence to condition on.
                Hidden entries (where ``mask == False``) may contain
                arbitrary values; they are ignored.
            mask: ``(J, F, T)`` bool tensor.  ``True`` = observed.
            n: Number of conditional samples to draw.
            seed: Optional random seed for reproducibility.

        Returns:
            ``(n, J, F, T)`` float32 tensor.  Row ``i`` is one draw from
            ``p(x_hid | x_obs)``.

        Raises:
            ValueError: if shapes of ``x_obs`` and ``mask`` are inconsistent.
            NotImplementedError: if ``mask`` pattern is not supported by this
                oracle (e.g. spatiotemporal masks for a temporal-only oracle).
        """

    @abstractmethod
    def true_shapley(
        self,
        x: Tensor,
        classifier: Callable[[Tensor], Tensor],
        players: "PlayerSet",
        n_mc: int = 1000,
        seed: int | None = None,
    ) -> Tensor:
        """Compute ground-truth Shapley values for a single sequence.

        The *true* Shapley values are defined with respect to the exact
        conditional value function:

        .. math::

            v(S) = \\mathbb{E}_{x_{\\bar{S}} \\sim p(\\cdot\\,|\\,x_S)}
                   \\bigl[ f(x_S,\\, x_{\\bar{S}}) \\bigr]

        and satisfy all four Shapley axioms (efficiency, symmetry,
        dummy, linearity) by construction.

        For M ≤ 12 players, implementations should enumerate all 2^M
        coalitions exactly.  For larger M, implementations should raise
        :py:exc:`NotImplementedError` rather than silently use an
        approximation.

        Args:
            x: ``(1, J, F, T)`` or ``(J, F, T)`` float32 sequence for
                which to compute Shapley values.  Squeezed to ``(J, F, T)``
                internally.
            classifier: Callable that maps ``(B, J, F, T)`` → ``(B,)``
                scalar targets (e.g. class probability for a fixed class).
            players: :class:`~motionbench.players.base.PlayerSet` defining
                the M players and coalition→mask expansion.
            n_mc: Monte Carlo samples per coalition for estimating ``v(S)``.
            seed: Optional random seed.

        Returns:
            ``(M,)`` float32 Tensor of Shapley values.  Satisfies the
            *efficiency axiom*: ``φ.sum() ≈ v(full) − v(empty)`` to
            numerical precision.

        Raises:
            NotImplementedError: if M > 12 (enumeration intractable) and
                the implementation does not support estimation.
        """
