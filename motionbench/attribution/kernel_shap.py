"""motionbench.attribution.kernel_shap — KernelSHAP attributor with pluggable imputer.

This module provides :class:`KernelShapAttributor`, which wraps
``shap.KernelExplainer`` and delegates coalition masking to any
:class:`~motionbench.imputers.base.BaseImputer`.  Coalitions are defined
at the **player level** (M players, not J*F*T coordinates), so no
post-hoc coordinate aggregation is needed.

Design
------
The SHAP game is formulated as follows:

* "Features" presented to ``shap.KernelExplainer`` are the *M player
  indicators* (not raw coordinates).  Background = ``np.zeros((1, M))``
  (all players absent); input = ``np.ones((1, M))`` (all present).
* KernelExplainer masks the input per coalition: because background is
  all-zeros and input is all-ones, the masked vector *is* the binary
  coalition indicator ``z ∈ {0,1}^M``.
* The coalition model function receives ``(n_evals, M)`` binary arrays and
  must return ``(n_evals,)`` value-function estimates.  For each coalition
  it calls :class:`_MotionBenchMasker` to obtain the mean imputed
  completion and then runs the classifier.
* ``shap.KernelExplainer`` performs the WLS Shapley solve (Lundberg & Lee
  2017, Eq. 8) over the sampled coalitions.

Estimator
---------
The value function estimator is::

    v(S) ≈ f(E_{x_bar ~ q(·|x_S)}[x])

i.e. the classifier evaluated at the **mean completion** (not the average
over individual predictions).  For a linear classifier this is exact; for
non-linear classifiers it introduces a Jensen bias that is negligible when
the imputer's conditional variance is small.

For deterministic imputers (``ZeroImputer``, ``MeanImputer``) the mean
completion degenerates to a single constant and the bias is zero.

References
----------
Lundberg, S. M., & Lee, S.-I. (2017).
    A unified approach to interpreting model predictions. NeurIPS 30.
Aas, K., Jullum, M., & Løland, A. (2021).
    Explaining individual predictions when features are dependent.
    arXiv:1903.10464.
Jullum, M., Redelmeier, A., & Aas, K. (2021).
    Groupwise Shapley feature importance values.
    Comput. Stat. 36, 1951–1992.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import shap
import torch
from torch import Tensor

from motionbench.attribution.base import BaseAttributor

if TYPE_CHECKING:
    from collections.abc import Callable

    from motionbench.imputers.base import BaseImputer
    from motionbench.players.base import PlayerSet


__all__ = ["KernelShapAttributor"]


# ---------------------------------------------------------------------------
# Internal masker
# ---------------------------------------------------------------------------


class _MotionBenchMasker(shap.maskers.Masker):  # type: ignore[misc]
    """SHAP masker that delegates coalition imputation to :class:`BaseImputer`.

    The masker is called once per coalition evaluation.  Given a ``(M,)``
    boolean coalition indicator, it:

    1. Expands the indicator to a ``(J, F, T)`` element-level boolean mask
       via :meth:`~motionbench.players.base.PlayerSet.coalition_mask`.
    2. Draws ``n_completion_samples`` completions from the imputer.
    3. Returns their **mean** as a ``(1, J*F*T)`` flat array.

    The mean-completion estimator for v(S) is exact for linear classifiers
    and has negligible Jensen bias for imputers with small conditional
    variance (e.g. ``GaussianOracle`` with many samples).

    Args:
        x_obs: ``(J, F, T)`` observed sequence being explained.
        players: Player set that maps coalition indicators to element masks.
        imputer: Fitted imputer used to complete hidden entries.
        n_completion_samples: Number of completions to average per coalition.
    """

    def __init__(
        self,
        x_obs: Tensor,
        players: PlayerSet,
        imputer: BaseImputer,
        n_completion_samples: int,
    ) -> None:
        self._x_obs = x_obs
        self._players = players
        self._imputer = imputer
        self._n_completion = n_completion_samples
        J, F, T = x_obs.shape
        self._J = J
        self._F = F
        self._T = T
        self._n_flat = J * F * T
        # Required by shap.maskers.Masker._standardize_mask
        self.shape = (1, players.n_players)

    def __call__(
        self,
        mask: npt.NDArray[Any],
        x: npt.NDArray[Any],
    ) -> tuple[npt.NDArray[np.float64]]:
        """Impute hidden players and return the mean completion.

        Args:
            mask: ``(M,)`` boolean array — True = player is observed.
                Provided by SHAP as the coalition indicator.
            x: ``(M,)`` float array — ignored.  The actual data comes from
                ``self._x_obs`` stored at construction time.

        Returns:
            A 1-tuple containing a ``(1, J*F*T)`` float64 array representing
            the mean imputed completion for this coalition.
        """
        z = torch.from_numpy(mask.astype(np.int32))
        element_mask: Tensor = self._players.coalition_mask(z)  # (J, F, T) bool
        completions: Tensor = self._imputer.impute(
            self._x_obs, element_mask, n_samples=self._n_completion
        )  # (n_completion, J, F, T)
        mean_comp: Tensor = completions.float().mean(dim=0)  # (J, F, T)
        flat: npt.NDArray[np.float64] = (
            mean_comp.detach().cpu().numpy().reshape(1, self._n_flat).astype(np.float64)
        )
        return (flat,)


# ---------------------------------------------------------------------------
# KernelShapAttributor
# ---------------------------------------------------------------------------


class KernelShapAttributor(BaseAttributor):
    """KernelSHAP attributor with a pluggable BaseImputer masker.

    Wraps ``shap.KernelExplainer`` with a :class:`_MotionBenchMasker` that
    delegates to :meth:`~motionbench.imputers.base.BaseImputer.impute`.

    Player aggregation is handled by the masker — coalitions operate at the
    player level, so no post-hoc coordinate aggregation is needed.

    The SHAP game is represented as follows:

    * Background dataset: ``np.zeros((1, M))`` — all players absent.
    * Input to explain: ``np.ones((1, M))`` — all players present.
    * Because background=0 and input=1, KernelExplainer's internal masking
      produces binary coalition indicators directly.

    Benchmarking note: the ``"permutation"`` algorithm can be faster for
    M ≤ 20 on motion sequences (fewer model calls per coalition sample).
    This implementation uses ``shap.KernelExplainer`` for both algorithms;
    the ``algorithm`` parameter is stored for downstream pipeline use but
    does not currently select a different SHAP backend.

    Args:
        classifier: Callable ``(B, J, F, T) float32 → (B,) float32``.
            Must return a scalar per sample (e.g. class probability).
        imputer: Fitted :class:`~motionbench.imputers.base.BaseImputer`.
        n_samples: Number of coalition evaluations passed to
            ``shap.KernelExplainer.shap_values(nsamples=…)``.
            Default 2048 is sufficient to enumerate all coalitions for
            M ≤ 10 and gives low variance for larger M.
        n_completion_samples: Number of imputer draws per coalition used
            to estimate the mean completion for v(S).
        seed: NumPy random seed for SHAP's internal coalition sampling.
        algorithm: ``"kernel"`` (default) or ``"permutation"``.  Stored
            for pipeline metadata; both currently use KernelExplainer.
    """

    requires_imputer: bool = True
    requires_gradient: bool = False

    def __init__(
        self,
        classifier: Callable[[Tensor], Tensor],
        imputer: BaseImputer,
        n_samples: int = 2**11,
        n_completion_samples: int = 20,
        seed: int = 42,
        algorithm: str = "kernel",
    ) -> None:
        super().__init__(classifier)
        self._imputer = imputer
        self._n_samples = n_samples
        self._n_completion_samples = n_completion_samples
        self._seed = seed
        self._algorithm = algorithm

    def attribute(
        self,
        x: Tensor,
        players: PlayerSet,
        target: int = 0,
    ) -> Tensor:
        """Return ``(M,)`` Shapley values for a single ``(J, F, T)`` sample.

        The Shapley values are computed via KernelSHAP (Lundberg & Lee 2017)
        with the conditional-expectation value function estimated by the
        pluggable imputer.

        Args:
            x: ``(J, F, T)`` float32 input sequence.  Must not include a
                batch dimension.
            players: :class:`~motionbench.players.base.PlayerSet` defining
                the M players and the coalition→element-mask expansion.
            target: Class index for multi-output classifiers.  If the
                classifier returns ``(B, n_classes)``, column ``target`` is
                selected.  For classifiers that already return ``(B,)`` this
                argument is ignored.

        Returns:
            ``(M,)`` float32 Tensor of Shapley values.  Satisfies the
            efficiency axiom: ``phi.sum() ≈ v(N) − v(∅)``.

        Raises:
            ValueError: if ``x.ndim != 3``.
        """
        if x.ndim != 3:
            raise ValueError(
                f"x must be a (J, F, T) tensor without batch dim; got shape {tuple(x.shape)}."
            )

        J, F, T = x.shape
        M = players.n_players

        # ------------------------------------------------------------------ #
        # Build the masker and classifier wrappers
        # ------------------------------------------------------------------ #

        masker = _MotionBenchMasker(
            x, players, self._imputer, self._n_completion_samples
        )

        # Detect classifier device so we can move inputs to match.
        # self._classifier may be a plain Python function (e.g. _prob_clf wrapper),
        # not an nn.Module.  Try self._classifier first; if it's a closure, look for
        # an nn.Module in the closure variables.
        _clf_device = torch.device("cpu")
        try:
            _clf_device = next(self._classifier.parameters()).device  # type: ignore[attr-defined]
        except (StopIteration, AttributeError):
            # Try closure vars (e.g. _prob_clf captures the classifier)
            _fn = self._classifier
            _cells = getattr(_fn, "__closure__", None) or []
            for cell in _cells:
                try:
                    _obj = cell.cell_contents
                    if hasattr(_obj, "parameters"):
                        _clf_device = next(_obj.parameters()).device
                        break
                except (ValueError, StopIteration, AttributeError):
                    continue

        def _clf_flat(
            flat_np: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            """Run classifier on (n_rows, J*F*T) flat numpy array → (n_rows,)."""
            x_t = torch.tensor(
                flat_np.reshape(-1, J, F, T).astype(np.float32),
                device=_clf_device,
            )
            with torch.no_grad():
                out: Tensor = self._classifier(x_t)
            if out.ndim == 2:
                out = out[:, target]
            result: npt.NDArray[np.float64] = out.detach().cpu().numpy().astype(np.float64)
            return result

        def _coalition_model(
            coalition_np: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            """Map binary coalition indicators (n_evals, M) → v(S) values (n_evals,).

            For each coalition row, calls the masker to obtain the mean
            imputed completion and then runs the classifier.

            Args:
                coalition_np: ``(n_evals, M)`` binary float64 array.
                    Each row is a coalition indicator (1=player present).

            Returns:
                ``(n_evals,)`` float64 array of value-function estimates.
            """
            n_evals = len(coalition_np)
            flat_batch = np.empty((n_evals, J * F * T), dtype=np.float64)
            for i, row in enumerate(coalition_np):
                (mean_comp,) = masker(row.astype(bool), row)
                flat_batch[i] = mean_comp[0]
            return _clf_flat(flat_batch)

        # ------------------------------------------------------------------ #
        # Run KernelExplainer
        # ------------------------------------------------------------------ #

        np.random.seed(self._seed)

        background = np.zeros((1, M), dtype=np.float64)
        input_x = np.ones((1, M), dtype=np.float64)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = shap.KernelExplainer(_coalition_model, background)
            shap_vals = explainer.shap_values(
                input_x,
                nsamples=self._n_samples,
                l1_reg=0,
                silent=True,
            )

        # shap_vals: (1, M) for single-output models
        phi = np.asarray(shap_vals).reshape(M)
        return torch.tensor(phi, dtype=torch.float32)
