"""motionbench.imputers.off_manifold — Off-manifold baseline imputers.

This module provides four off-manifold baseline imputers that fill hidden
coordinates without modelling the conditional distribution
``p(x_hid | x_obs)``.  They are fast, closed-form, and intentionally biased:
useful as diagnostic lower bounds and for ablation studies.

Taxonomy (Aas, Jullum & Løland 2021, Table 1):

    ZeroImputer          — constant-zero fill.
    MeanImputer          — per-coordinate training mean fill.
    MarginalDonorImputer — random donor from training pool (interventional).
    GaussianNoiseImputer — training mean + calibrated Gaussian noise.

All four imputers guarantee that *observed entries are preserved bit-for-bit*
in every returned completion:  ``output[:, mask] == x_obs[mask]``.

References
----------
Sundararajan & Najmi (2020) "The many Shapley values for model explanation."
arXiv:1908.08474.

Aas, Jullum & Løland (2021) "Explaining individual predictions when features
are dependent: More accurate approximations to Shapley values."
arXiv:1903.10464.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from motionbench.imputers.base import BaseImputer

if TYPE_CHECKING:
    from motionbench.data.base import BaseDataset


__all__ = [
    "ZeroImputer",
    "MeanImputer",
    "MarginalDonorImputer",
    "GaussianNoiseImputer",
]


# ---------------------------------------------------------------------------
# ZeroImputer
# ---------------------------------------------------------------------------


class ZeroImputer(BaseImputer):
    """Off-manifold baseline: fill hidden entries with zeros.

    ``fit()`` is a no-op.  ``is_on_manifold = False``.

    The zero baseline is appropriate when 0 is a neutral/uninformative value
    for the model (rarely true for motion, but useful as a diagnostic lower
    bound).  It corresponds to the implicit ``shap.maskers.Fixed(zeros)``
    choice used in most GNN/vision XAI code.

    Reference:
        Sundararajan & Najmi (2020) "The many Shapley values for model
        explanation." — Eq. (2) zero baseline.
    """

    is_on_manifold: bool = False

    def fit(self, train_data: "BaseDataset") -> "ZeroImputer":
        """No-op fit — ZeroImputer requires no training statistics.

        Args:
            train_data: Ignored. Present for API compatibility.

        Returns:
            ``self`` for method chaining.
        """
        return self

    @torch.no_grad()
    def impute(
        self,
        x_obs: Tensor,
        mask: Tensor,
        n_samples: int,
        seed: int | None = None,
    ) -> Tensor:
        """Return *n_samples* completions with hidden entries set to zero.

        Args:
            x_obs: ``(J, F, T)`` float32 sequence.  Entries where
                ``mask == False`` are ignored.
            mask: ``(J, F, T)`` bool tensor.  ``True`` = observed.
            n_samples: Number of completed sequences to return.
            seed: Ignored — ZeroImputer is deterministic.

        Returns:
            ``(n_samples, J, F, T)`` float32 tensor.  Observed entries
            equal ``x_obs``; hidden entries are exactly 0.0.
        """
        x_obs = x_obs.to(dtype=torch.float32)
        mask = mask.to(device=x_obs.device)
        completed = torch.where(mask, x_obs, torch.zeros_like(x_obs))
        return completed.unsqueeze(0).expand(n_samples, -1, -1, -1).contiguous()


# ---------------------------------------------------------------------------
# MeanImputer
# ---------------------------------------------------------------------------


class MeanImputer(BaseImputer):
    """Off-manifold baseline: fill hidden entries with per-coordinate training mean.

    ``fit()`` computes and stores the per-coordinate mean from ``train_data``.
    Observed entries are always preserved bit-for-bit.

    Corresponds to ``shap.maskers.Fixed(mean)``.

    Reference:
        Sundararajan & Najmi (2020) "The many Shapley values for model
        explanation." — mean baseline.
    """

    is_on_manifold: bool = False

    def fit(self, train_data: "BaseDataset") -> "MeanImputer":
        """Compute per-coordinate mean over the training dataset.

        Iterates over ``train_data`` once, accumulating a running sum.
        Each item yielded by ``train_data`` should be a tuple
        ``(x, label)`` where ``x`` is a ``(J, F, T)`` float32 tensor,
        or a bare ``(J, F, T)`` tensor.

        Args:
            train_data: Dataset whose items are ``(J, F, T)`` tensors
                (possibly wrapped in a tuple with a label).

        Returns:
            ``self`` for method chaining.

        Raises:
            ValueError: if ``train_data`` is empty.
        """
        running_sum: Tensor | None = None
        n = 0
        for item in train_data:
            x = item[0] if isinstance(item, (tuple, list)) else item
            x = x.to(dtype=torch.float32)
            if running_sum is None:
                running_sum = x.clone()
            else:
                running_sum = running_sum + x
            n += 1
        if running_sum is None or n == 0:
            raise ValueError("train_data is empty; cannot compute mean.")
        self._mean: Tensor = running_sum / n
        return self

    @torch.no_grad()
    def impute(
        self,
        x_obs: Tensor,
        mask: Tensor,
        n_samples: int,
        seed: int | None = None,
    ) -> Tensor:
        """Return *n_samples* completions with hidden entries set to the training mean.

        Args:
            x_obs: ``(J, F, T)`` float32 sequence.
            mask: ``(J, F, T)`` bool tensor.  ``True`` = observed.
            n_samples: Number of completed sequences to return.
            seed: Ignored — MeanImputer is deterministic.

        Returns:
            ``(n_samples, J, F, T)`` float32 tensor.

        Raises:
            RuntimeError: if ``fit()`` has not been called.
        """
        if not hasattr(self, "_mean"):
            raise RuntimeError(
                "MeanImputer.impute() called before fit(). "
                "Call fit(train_data) first."
            )
        x_obs = x_obs.to(dtype=torch.float32)
        mask = mask.to(device=x_obs.device)
        mean = self._mean.to(device=x_obs.device, dtype=torch.float32)
        completed = torch.where(mask, x_obs, mean)
        return completed.unsqueeze(0).expand(n_samples, -1, -1, -1).contiguous()


# ---------------------------------------------------------------------------
# MarginalDonorImputer
# ---------------------------------------------------------------------------


class MarginalDonorImputer(BaseImputer):
    """Off-manifold baseline: fill hidden coords by sampling a random training sequence.

    For each of the *n_samples* completions, a random training sequence (a
    "donor") is selected and its values are used at the hidden coordinates.

    This implements the *independence* imputer of Aas et al. (2021) and is
    equivalent to ``shap.maskers.Independent``.  Axiomatically it is an
    *interventional* imputer — it samples from ``p(x_hid)`` rather than
    ``p(x_hid | x_obs)``, so it does not respect conditional structure.

    Reference:
        Aas, Jullum & Løland (2021) "Explaining individual predictions when
        features are dependent." arXiv:1903.10464, Table 1 row 'independence'.
    """

    is_on_manifold: bool = False

    def fit(self, train_data: "BaseDataset") -> "MarginalDonorImputer":
        """Store all training sequences as the donor pool.

        Args:
            train_data: Dataset whose items are ``(J, F, T)`` tensors
                (possibly wrapped in a tuple with a label).

        Returns:
            ``self`` for method chaining.

        Raises:
            ValueError: if ``train_data`` is empty.
        """
        sequences: list[Tensor] = []
        for item in train_data:
            x = item[0] if isinstance(item, (tuple, list)) else item
            sequences.append(x.to(dtype=torch.float32))
        if not sequences:
            raise ValueError("train_data is empty; cannot build donor pool.")
        self._pool: Tensor = torch.stack(sequences, dim=0)
        return self

    @torch.no_grad()
    def impute(
        self,
        x_obs: Tensor,
        mask: Tensor,
        n_samples: int,
        seed: int | None = None,
    ) -> Tensor:
        """Return *n_samples* completions by sampling donors from the training pool.

        For each sample ``i``, a random training sequence is chosen as donor;
        hidden coordinates take that donor's values.

        Args:
            x_obs: ``(J, F, T)`` float32 sequence.
            mask: ``(J, F, T)`` bool tensor.  ``True`` = observed.
            n_samples: Number of completed sequences to return.
            seed: Optional integer seed for reproducibility.

        Returns:
            ``(n_samples, J, F, T)`` float32 tensor.

        Raises:
            RuntimeError: if ``fit()`` has not been called.
        """
        if not hasattr(self, "_pool"):
            raise RuntimeError(
                "MarginalDonorImputer.impute() called before fit(). "
                "Call fit(train_data) first."
            )
        x_obs = x_obs.to(dtype=torch.float32)
        n_train = self._pool.shape[0]
        generator: torch.Generator | None = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        idxs = torch.randint(
            0, n_train, (n_samples,), generator=generator
        )
        donors = self._pool[idxs].to(device=x_obs.device)
        out = torch.where(mask.to(device=x_obs.device).unsqueeze(0), x_obs.unsqueeze(0), donors)
        return out.contiguous()


# ---------------------------------------------------------------------------
# GaussianNoiseImputer
# ---------------------------------------------------------------------------


class GaussianNoiseImputer(BaseImputer):
    """Off-manifold baseline: fill hidden entries with training mean ± Gaussian noise.

    Useful as a stochastic version of :class:`MeanImputer` that produces
    variance-calibrated completions without learning any conditional structure.
    Each hidden coordinate is filled as::

        hidden[j, f, t] = mean[j, f, t] + scale * std[j, f, t] * N(0, 1)

    Args:
        scale: Standard deviation multiplier applied to the per-coordinate
            training standard deviation.  Default ``1.0``.
    """

    is_on_manifold: bool = False

    def __init__(self, scale: float = 1.0) -> None:
        """Initialise with a noise scale factor.

        Args:
            scale: Multiplier for the per-coordinate training std.
        """
        self.scale = scale

    def fit(self, train_data: "BaseDataset") -> "GaussianNoiseImputer":
        """Compute per-coordinate mean and standard deviation from training data.

        Iterates over ``train_data`` twice (one pass for mean, one for std)
        to keep memory usage bounded.

        Args:
            train_data: Dataset whose items are ``(J, F, T)`` tensors
                (possibly wrapped in a tuple with a label).

        Returns:
            ``self`` for method chaining.

        Raises:
            ValueError: if ``train_data`` is empty.
        """
        # First pass: compute mean
        running_sum: Tensor | None = None
        n = 0
        for item in train_data:
            x = item[0] if isinstance(item, (tuple, list)) else item
            x = x.to(dtype=torch.float32)
            if running_sum is None:
                running_sum = x.clone()
            else:
                running_sum = running_sum + x
            n += 1
        if running_sum is None or n == 0:
            raise ValueError("train_data is empty; cannot compute statistics.")
        mean = running_sum / n

        # Second pass: compute variance
        running_sq: Tensor | None = None
        for item in train_data:
            x = item[0] if isinstance(item, (tuple, list)) else item
            x = x.to(dtype=torch.float32)
            diff = x - mean
            if running_sq is None:
                running_sq = diff * diff
            else:
                running_sq = running_sq + diff * diff
        # running_sq should not be None if first pass succeeded
        assert running_sq is not None
        var = running_sq / n
        std = torch.sqrt(var.clamp(min=0.0))

        self._mean: Tensor = mean
        self._std: Tensor = std
        return self

    @torch.no_grad()
    def impute(
        self,
        x_obs: Tensor,
        mask: Tensor,
        n_samples: int,
        seed: int | None = None,
    ) -> Tensor:
        """Return *n_samples* noisy completions.

        Hidden entries are filled as ``mean + scale * std * randn``.
        Observed entries are preserved bit-for-bit.

        Args:
            x_obs: ``(J, F, T)`` float32 sequence.
            mask: ``(J, F, T)`` bool tensor.  ``True`` = observed.
            n_samples: Number of completed sequences to return.
            seed: Optional integer seed for reproducibility.

        Returns:
            ``(n_samples, J, F, T)`` float32 tensor.

        Raises:
            RuntimeError: if ``fit()`` has not been called.
        """
        if not hasattr(self, "_mean"):
            raise RuntimeError(
                "GaussianNoiseImputer.impute() called before fit(). "
                "Call fit(train_data) first."
            )
        x_obs = x_obs.to(dtype=torch.float32)
        mean = self._mean.to(device=x_obs.device, dtype=torch.float32)
        std = self._std.to(device=x_obs.device, dtype=torch.float32)

        generator: torch.Generator | None = None
        if seed is not None:
            generator = torch.Generator(device=x_obs.device)
            generator.manual_seed(seed)

        J, F, T = x_obs.shape
        noise = torch.randn(
            n_samples, J, F, T,
            dtype=torch.float32,
            device=x_obs.device,
            generator=generator,
        )
        hidden_fill = mean.unsqueeze(0) + self.scale * std.unsqueeze(0) * noise
        out = torch.where(mask.to(device=x_obs.device).unsqueeze(0), x_obs.unsqueeze(0), hidden_fill)
        return out.contiguous()
