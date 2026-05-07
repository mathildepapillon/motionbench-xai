"""motionbench.pipelines.synthetic_eval — Full (method × dataset × classifier) grid for synthetic data.

This module implements :func:`run_synthetic_eval`, which orchestrates:

1. Dataset instantiation from Hydra configs.
2. Classifier construction (freshly initialised — no checkpoint loading for
   synthetic classifiers).
3. Imputer construction and fitting for SHAP-based methods.
4. Attribution computation on ``cfg.n_sequences`` samples per cell.
5. Metric evaluation (ground-truth, fidelity, stability, sanity).
6. Result serialisation to ``cfg.results_dir/{dataset}/{classifier}/{method}/``.
7. WandB logging.

Resumability
------------
Before computing any cell ``(dataset, classifier, method)``, the pipeline
checks for ``results_dir/{dataset}/{clf}/{method}/result.json``.  If the file
exists, the cell is skipped and the cached result is used.  This allows
interrupted sweeps to be resumed without re-running completed cells.

Parallelism
-----------
Cells are dispatched with ``joblib.Parallel(n_jobs=cfg.n_jobs)``.  For
GPU-based imputers (VAEAC, FlowMatching) set ``n_jobs=1`` in the config and
use ``device=cuda:0``.

Error handling
--------------
Each cell is wrapped in a try/except block.  On failure, an error record is
written to ``results_dir/{dataset}/{clf}/{method}/error.json`` and the sweep
continues.

References
----------
OpenXAI (NeurIPS D&B 2022) ``experiments/`` layout — pipeline structure.
Aas et al. (2021) arXiv:1903.10464 — conditional-expectation Shapley game.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from torch import Tensor

from motionbench.imputers.base import BaseImputer
from motionbench.imputers.off_manifold import ZeroImputer
from motionbench.metrics.ground_truth import (
    EC1Metric,
    EC2Metric,
    EC3Metric,
    EfficiencyErrorMetric,
    KendallRankMetric,
    SpearmanRankMetric,
    TopKRecovery,
)
from motionbench.metrics.sanity_checks import (
    ModelParameterRandomisationMetric,
)
from motionbench.metrics.stability import MaxSensitivityMetric

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from motionbench.attribution.base import BaseAttributor
    from motionbench.classifiers.base import Classifier
    from motionbench.data.base import BaseDataset
    from motionbench.players.base import PlayerSet

log = logging.getLogger(__name__)

__all__ = ["run_synthetic_eval"]

# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

_GT_METRICS: dict[str, Any] = {
    "ec1": EC1Metric,
    "ec2": EC2Metric,
    "ec3": EC3Metric,
    "topk": TopKRecovery,
    "spearman": SpearmanRankMetric,
    "kendall": KendallRankMetric,
    "efficiency_error": EfficiencyErrorMetric,
}

_FIDELITY_METRICS: dict[str, Any] = {}
try:
    from motionbench.metrics.fidelity import (  # noqa: PLC0415
        FaithfulnessCorrelationMetric,
        PixelFlippingMetric,
        PlayerDeletionMetric,
    )

    _FIDELITY_METRICS = {
        "faithfulness_correlation": FaithfulnessCorrelationMetric,
        "pixel_flipping": PixelFlippingMetric,
        "player_deletion": PlayerDeletionMetric,
    }
except ImportError:
    log.debug("motionbench.metrics.fidelity not available — fidelity metrics skipped.")

_MANIFOLD_METRICS: dict[str, Any] = {}
try:
    from motionbench.metrics.fidelity import (  # noqa: PLC0415
        CrossGranularityFaithfulnessMetric,
        ManifoldFidelityGapMetric,
    )

    _MANIFOLD_METRICS = {
        "faith_gap": ManifoldFidelityGapMetric,
        "cgfs": CrossGranularityFaithfulnessMetric,
    }
except ImportError:
    log.debug("motionbench.metrics.fidelity not available — manifold metrics skipped.")

_STABILITY_METRICS: dict[str, Any] = {
    "max_sensitivity": MaxSensitivityMetric,
}

_SANITY_METRICS: dict[str, Any] = {
    "model_parameter_randomisation": ModelParameterRandomisationMetric,
}

# These metrics need direct access to the nn.Module (not a clf_fn wrapper) because
# they require gradient flow or parameter enumeration.
_NEEDS_MODULE: frozenset[str] = frozenset({
    "max_sensitivity",
    "model_parameter_randomisation",
})

_ALL_METRICS: dict[str, Any] = {
    **_GT_METRICS,
    **_FIDELITY_METRICS,
    **_MANIFOLD_METRICS,
    **_STABILITY_METRICS,
    **_SANITY_METRICS,
}


# ---------------------------------------------------------------------------
# Helpers — config loading
# ---------------------------------------------------------------------------


def _load_sub_config(subdir: str, name: str, cfg: DictConfig) -> DictConfig:
    """Load a config YAML from ``configs/{subdir}/{name}.yaml``.

    The path is resolved relative to the Hydra original working directory so
    the pipeline works regardless of Hydra's output directory.

    Args:
        subdir: Sub-directory under ``configs/`` (e.g. ``"data"``, ``"methods"``).
        name: Config file stem (e.g. ``"gaussian_k4"``).
        cfg: Root experiment config (used to resolve the config root path).

    Returns:
        Loaded :class:`omegaconf.DictConfig`.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    try:
        from hydra.utils import get_original_cwd  # noqa: PLC0415

        config_root = Path(get_original_cwd()) / "configs"
    except Exception:
        # Fall back to cwd (useful in tests where Hydra is not initialised)
        config_root = Path.cwd() / "configs"

    config_path = config_root / subdir / f"{name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}. "
            f"Run from the motionbench-xai repo root."
        )
    return OmegaConf.load(config_path)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Helpers — dataset instantiation
# ---------------------------------------------------------------------------


def _instantiate_dataset(
    dataset_cfg: DictConfig,
) -> tuple[BaseDataset, int]:
    """Instantiate a dataset and extract the number of temporal windows K.

    ``K`` is a pipeline-level concept (number of players / temporal windows)
    that may or may not be accepted by the dataset constructor.
    :class:`~motionbench.data.synthetic.gaussian_motion.GaussianMotionDataset`
    accepts ``K``; :class:`~motionbench.data.synthetic.burr_motion.BurrMotionBenchmark`
    does not.  The pipeline strips ``K`` from the constructor kwargs for
    datasets that don't accept it.

    Args:
        dataset_cfg: Hydra DictConfig with ``_target_`` and constructor kwargs.

    Returns:
        ``(dataset, K)`` — the instantiated dataset and the number of players.
    """
    cfg_dict: dict[str, Any] = OmegaConf.to_container(dataset_cfg, resolve=True)  # type: ignore[assignment]
    K = int(cfg_dict.get("K", 4))  # default 4 temporal windows

    # GaussianMotionDataset's constructor accepts K; pop it for other datasets
    target_str = str(cfg_dict.get("_target_", ""))
    if "GaussianMotionDataset" not in target_str:
        cfg_dict.pop("K", None)

    # T is also a pipeline-only param for some real-data adapters that derive
    # T from cache files (e.g. BMCLabCacheDataset). Strip it for those.
    if "BMCLabCacheDataset" in target_str:
        cfg_dict.pop("T", None)

    dataset = cast("BaseDataset", instantiate(OmegaConf.create(cfg_dict)))
    return dataset, K


# ---------------------------------------------------------------------------
# Helpers — player instantiation
# ---------------------------------------------------------------------------


def _build_players(
    method_cfg: DictConfig,
    J: int,
    F: int,
    T: int,
    K: int,
) -> PlayerSet:
    """Instantiate the PlayerSet for a method config.

    Fills in the shape arguments ``K``, ``J``, ``F``, ``T`` which are dataset-
    dependent and not known at config-write time.

    Args:
        method_cfg: Method DictConfig (must have a ``players`` sub-config).
        J: Number of skeletal joints.
        F: Number of coordinates per joint.
        T: Number of frames.
        K: Number of temporal-window players.

    Returns:
        An instantiated :class:`~motionbench.players.base.PlayerSet`.
    """
    players_cfg: dict[str, Any] = OmegaConf.to_container(
        method_cfg.players, resolve=True
    )  # type: ignore[assignment]
    # Fill in shape args — all standard player types accept these kwargs
    players_cfg.update({"K": K, "J": J, "F": F, "T": T})
    return cast("PlayerSet", instantiate(OmegaConf.create(players_cfg)))


# ---------------------------------------------------------------------------
# Helpers — imputer instantiation
# ---------------------------------------------------------------------------


def _build_and_fit_imputer(
    method_cfg: DictConfig,
    dataset: BaseDataset,
    J: int,
    F: int,
    T: int,
    device: str = "cpu",
) -> BaseImputer:
    """Instantiate and fit the imputer described in a method config.

    Fills in ``J``, ``F``, ``T`` for learned imputers (VAEAC, FlowMatching)
    whose constructors require them.

    Args:
        method_cfg: Method DictConfig with an ``imputer`` sub-config.
        dataset: Training dataset to fit the imputer on.
        J: Number of joints.
        F: Number of coordinates per joint.
        T: Number of frames.
        device: Torch device string (e.g. ``"cpu"`` or ``"cuda:0"``).

    Returns:
        A fitted :class:`~motionbench.imputers.base.BaseImputer`.
    """
    imputer_cfg: dict[str, Any] = OmegaConf.to_container(
        method_cfg.imputer, resolve=True
    )  # type: ignore[assignment]

    target_str = str(imputer_cfg.get("_target_", ""))

    # GaussianOracle requires Sigma_joints / Sigma_time from the dataset.
    if "GaussianOracle" in target_str or "CopulaOracle" in target_str:
        oracle = getattr(dataset, "oracle", None)
        if oracle is None:
            raise ValueError(
                f"Dataset {dataset!r} has no 'oracle' attribute; "
                "cannot use GaussianOracle/CopulaOracle as an imputer."
            )
        if not isinstance(oracle, BaseImputer):
            raise ValueError(
                f"Dataset oracle {oracle!r} does not implement BaseImputer."
            )
        return oracle  # already fitted — Sigma matrices were set at construction

    # Learned imputers need J, F, T at construction time
    if any(cls in target_str for cls in ("VAEACImputer", "FlowMatchingImputer")):
        imputer_cfg.update({"J": J, "F": F, "T": T})

    # Also pass train_epochs if specified in method config (for VAEAC/Flow)
    train_epochs = int(method_cfg.get("train_epochs", 0))

    imputer: BaseImputer = cast(
        "BaseImputer", instantiate(OmegaConf.create(imputer_cfg))
    )

    # Learned imputers may expose a device attribute
    if hasattr(imputer, "_device"):
        cast("Any", imputer)._device = torch.device(device)

    if train_epochs > 0 and hasattr(imputer, "_fit_epochs"):
        # VAEACImputer._fit_epochs takes a (N, J, F, T) Tensor, not a dataset
        import torch as _torch  # noqa: PLC0415

        xs = _torch.stack([dataset[i][0] for i in range(len(dataset))])
        cast("Any", imputer)._fit_epochs(xs, epochs=train_epochs)
        cast("Any", imputer)._fitted = True
    else:
        imputer.fit(dataset)

    return imputer


# ---------------------------------------------------------------------------
# Helpers — attributor instantiation
# ---------------------------------------------------------------------------


def _build_attributor(
    method_cfg: DictConfig,
    classifier: Classifier,
    imputer: BaseImputer | None,
    players: PlayerSet,
) -> BaseAttributor:
    """Instantiate the attributor for a method config.

    Dispatches based on which additional arguments the constructor requires:

    * SHAP-based methods: ``classifier + imputer``
    * Gradient-based methods: ``classifier`` only
    * GradCAM: ``classifier + layer``

    **Game function convention:**
    SHAP-based methods (KernelSHAP, TimeSHAP, WindowSHAP, GroupSegmentSHAP) use
    ``softmax(classifier(x))[:, target]`` as the value function, matching the
    oracle game.  Gradient-based methods use raw logits (Aumann-Shapley axioms;
    different game from the oracle; difference is measured by EC1 and reported
    as "game mismatch" in the paper).

    Args:
        method_cfg: Method DictConfig with an ``attributor`` sub-config.
        classifier: The model to explain.
        imputer: Fitted imputer (``None`` for gradient methods).
        players: Player set (used to find GradCAM target layer).

    Returns:
        An instantiated :class:`~motionbench.attribution.base.BaseAttributor`.

    Raises:
        RuntimeError: If a required dependency (imputer / layer) is missing.
    """
    # Probability wrapper for SHAP methods: ensures the value function
    # v(S) = softmax(f(x))[target] matches the oracle's clf_fn exactly.
    def _prob_clf(x: Tensor) -> Tensor:
        with torch.no_grad():
            logits = classifier(x)
        return torch.softmax(logits, dim=-1)  # (B, n_classes)
    attr_cfg: dict[str, Any] = OmegaConf.to_container(
        method_cfg.attributor, resolve=True
    )  # type: ignore[assignment]
    target_str = str(attr_cfg.get("_target_", ""))

    # ---- GradCAM: needs a convolutional layer from the classifier ----
    if "GradCAMAttributor" in target_str:
        from motionbench.attribution.grad_cam import GradCAMAttributor  # noqa: PLC0415
        from motionbench.classifiers.synthetic_cnn import SyntheticCNNClassifier  # noqa: PLC0415

        if not isinstance(classifier, SyntheticCNNClassifier):
            raise TypeError(
                "GradCAMAttributor requires a convolutional classifier "
                f"(SyntheticCNNClassifier); got {type(classifier).__name__}."
            )
        layer = classifier.conv_layers[0]  # first Conv1d layer
        return GradCAMAttributor(
            classifier,
            layer=layer,
            interpolate_mode=attr_cfg.get("interpolate_mode", "nearest"),
        )

    # ---- KernelSHAP variants (including temporal-player): need imputer ----
    if "KernelShapAttributor" in target_str:
        if imputer is None:
            raise RuntimeError(
                f"{target_str} requires an imputer but none was built."
            )
        n_samples = int(method_cfg.get("n_kernel_samples", 256))
        n_compl = int(method_cfg.get("n_completion_samples", 20))
        seed = int(method_cfg.get("seed", 42))
        return cast(
            "BaseAttributor",
            instantiate(
                OmegaConf.create(attr_cfg),
                classifier=_prob_clf,  # v(S) = prob[target], matches oracle
                imputer=imputer,
                n_samples=n_samples,
                n_completion_samples=n_compl,
                seed=seed,
            ),
        )

    # NOTE: The "TimeSHAPAttributor" substring deliberately excludes
    # "RealTimeSHAPAttributor" (the actual ``timeshap`` pip-package wrapper),
    # which has its own dispatch branch below.  ``TimeSHAPAttributor`` here
    # refers exclusively to the backward-compat alias for
    # ``KernelSHAPTemporalAttributor`` (our hand-rolled KernelSHAP-over-
    # temporal-windows surrogate, kept for old result-file deserialisation).
    if (
        "KernelSHAPTemporalAttributor" in target_str
        or target_str.endswith(".TimeSHAPAttributor")
    ):
        if imputer is None:
            raise RuntimeError(
                f"{target_str} requires an imputer but none was built."
            )
        n_coalitions = int(method_cfg.get("n_coalitions", 100))
        seed = int(method_cfg.get("seed", 42))
        return cast(
            "BaseAttributor",
            instantiate(
                OmegaConf.create(attr_cfg),
                classifier=_prob_clf,  # v(S) = prob[target], matches oracle
                imputer=imputer,
                n_coalitions=n_coalitions,
                seed=seed,
            ),
        )

    if "WindowSHAPAttributor" in target_str or "RealTimeSHAPAttributor" in target_str:
        # WindowSHAP family + the real ``timeshap`` pip-package wrapper:
        # no imputer (these methods have their own perturbation strategies),
        # but still uses the probability game so v(S) = softmax(f(x))[target]
        # matches the oracle exactly (KernelSHAP rows are evaluated on the
        # same value function).
        return cast(
            "BaseAttributor",
            instantiate(OmegaConf.create(attr_cfg), classifier=_prob_clf),
        )

    # ---- Gradient-based / all other methods ----
    # instantiate() reads constructor kwargs from attr_cfg; we only need to
    # inject 'classifier' which is not in the YAML (it's passed at runtime).
    return cast(
        "BaseAttributor",
        instantiate(OmegaConf.create(attr_cfg), classifier=classifier),
    )


# ---------------------------------------------------------------------------
# Helpers — classifier instantiation
# ---------------------------------------------------------------------------


def _build_classifier(
    clf_cfg: DictConfig,
    J: int,
    F: int,
    T: int,
    K: int,
    n_classes: int = 3,
) -> Classifier:
    """Instantiate a classifier, filling in shape args from the dataset.

    Args:
        clf_cfg: Classifier DictConfig with ``_target_`` and optional kwargs.
        J: Number of joints.
        F: Number of coordinates per joint.
        T: Number of frames per sequence.
        K: Number of temporal windows.
        n_classes: Number of output classes.

    Returns:
        An instantiated :class:`~motionbench.classifiers.base.Classifier`.
    """
    cfg_dict: dict[str, Any] = OmegaConf.to_container(clf_cfg, resolve=True)  # type: ignore[assignment]
    target_str = str(cfg_dict.get("_target_", ""))

    if "SyntheticMLPClassifier" in target_str:
        cfg_dict.update({"J": J, "F": F, "T": T, "K": K, "n_classes": n_classes})
    elif "SyntheticCNNClassifier" in target_str or "SyntheticTransformerClassifier" in target_str:
        cfg_dict.update({"J": J, "F": F, "n_classes": n_classes})

    return cast("Classifier", instantiate(OmegaConf.create(cfg_dict)))


# ---------------------------------------------------------------------------
# Helpers — metric evaluation
# ---------------------------------------------------------------------------


def _collect_metric_names(cfg: DictConfig) -> list[str]:
    """Flatten the hierarchical metrics config into a list of metric names.

    Args:
        cfg: Root experiment config.

    Returns:
        Deduplicated list of metric name strings.
    """
    names: list[str] = []
    if hasattr(cfg, "metrics"):
        for group in OmegaConf.to_container(cfg.metrics, resolve=True).values():  # type: ignore[union-attr]
            if isinstance(group, list):
                names.extend(group)
    return list(dict.fromkeys(names))  # deduplicate preserving order


class _SampleCachedOracle:
    """Thin oracle wrapper that short-circuits ``true_shapley`` with a pre-computed vector.

    Used inside :func:`_evaluate_metrics` so that multiple GT metrics (EC1, EC2,
    EC3, Spearman, Kendall, TopK, EfficiencyError) can all share a single oracle
    ``true_shapley`` call per sample instead of each calling it independently.

    ``conditional_sample`` and all other attributes are forwarded to the real oracle.
    """

    def __init__(self, base_oracle: Any, cached_phi: Tensor) -> None:
        self._base = base_oracle
        self._phi = cached_phi

    def true_shapley(self, *args: Any, **kwargs: Any) -> Tensor:  # noqa: ARG002
        """Return the pre-computed Shapley vector (no oracle re-evaluation)."""
        return self._phi

    def conditional_sample(self, *args: Any, **kwargs: Any) -> Any:
        return self._base.conditional_sample(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def _evaluate_metrics(
    phi_list: list[Tensor],
    x_list: list[Tensor],
    target_list: list[int],
    classifier: Classifier,
    players: PlayerSet,
    dataset: BaseDataset,
    imputer: BaseImputer | None,
    metric_names: list[str],
    cfg: Any,
    device: str = "cpu",
    attributor: Any = None,
) -> dict[str, float]:
    """Evaluate all requested metrics averaged over all (phi, x) pairs.

    Metrics are evaluated on every sample in ``phi_list``/``x_list`` and the
    resulting scalar scores are averaged.  This gives statistically reliable
    estimates instead of relying on a single sequence.

    Each sample uses the model's **predicted class** (argmax) as its target,
    matching the field standard (OpenXAI, SHAP paper, SHAPIQ) of explaining
    actual model decisions.  Oracle-based GT metrics (EC1, EC2, Spearman,
    etc.) build a per-sample ``clf_fn_i`` so the oracle game exactly matches
    the attribution game for every sample — there is no inconsistency.

    Oracle-based GT metrics use ``n_mc`` samples per coalition, controlled
    by ``cfg.metric_oracle_n_mc`` (default 50).  Stability/sanity metrics
    use the first ``cfg.metric_stability_n_samples`` sequences (default 5)
    to keep runtime manageable.

    Args:
        phi_list: List of ``(M,)`` attribution tensors (one per sequence).
        x_list: List of ``(J, F, T)`` input tensors (matching order).
        target_list: Per-sample predicted class indices (argmax of logits).
            Must have the same length as ``phi_list`` and ``x_list``.
        classifier: Model used to produce attributions.
        players: Player set used to produce attributions.
        dataset: Source dataset (provides oracle for GT metrics).
        imputer: Fitted imputer (for fidelity metrics; may be ``None``).
        metric_names: Names of metrics to evaluate.
        cfg: Hydra DictConfig with experiment settings.
        device: Torch device string.
        attributor: The :class:`~motionbench.attribution.base.BaseAttributor`
            instance.  Required for stability/sanity metrics; ``None`` or
            SHAP-based attributors will skip those metrics.

    Returns:
        Dict mapping metric sub-score names to float values (averaged).
    """
    import numpy as _np
    from collections import Counter  # noqa: PLC0415

    oracle = getattr(dataset, "oracle", None)  # None for real datasets
    clf_device = torch.device(device)
    classifier = classifier.to(clf_device)
    classifier.eval()

    def _make_clf_fn(tgt: int) -> Any:
        """Return a softmax-probability callable for a specific target class."""
        def clf_fn(b: Tensor) -> Tensor:
            with torch.no_grad():
                logits = classifier(b.to(clf_device))
            proba = torch.softmax(logits, dim=-1)
            if proba.ndim == 2:
                return proba[:, tgt]
            return proba
        return clf_fn

    # Quantus explain_func is built once; use the most common predicted class
    # so the func remains valid for the small stability/sanity subset.
    dominant_target: int = Counter(target_list).most_common(1)[0][0]
    method_explain_func = None
    if attributor is not None and hasattr(attributor, "build_quantus_explain_func"):
        method_explain_func = attributor.build_quantus_explain_func(
            players, dominant_target, device
        )

    # Config knobs
    oracle_n_mc: int = int(cfg.get("metric_oracle_n_mc", 50))
    # oracle_n_coalitions controls the coalition budget passed to true_shapley.
    # When 2^K <= oracle_n_coalitions, all coalitions are enumerated exactly.
    # When 2^K > oracle_n_coalitions, KernelSHAP-style paired sampling is used.
    # Default 2000 keeps exact enumeration for K <= 10 (2^10=1024) while
    # falling back to sampling for K > 10.  Set to e.g. 64 in the overnight
    # config to force sampling for K >= 7 (2^7=128) — a ~4x speedup for K=8,
    # ~16x for K=10.
    oracle_n_coalitions: int = int(cfg.get("metric_oracle_n_coalitions", 2000))
    stability_n: int = int(cfg.get("metric_stability_n_samples", 5))
    max_sensitivity_nr: int = int(cfg.get("quantus_max_sensitivity_nr_samples", 20))

    # ------------------------------------------------------------------
    # Pre-compute oracle Shapley values once per sample.
    #
    # Without this, each of the 7 GT metrics (ec1, ec2, ec3, topk,
    # spearman, kendall, efficiency_error) would call oracle.true_shapley
    # independently — a 7× redundancy.  For K=8 that is 7 × 256 × n_mc
    # oracle evaluations instead of 256 × n_mc.
    # ------------------------------------------------------------------
    oracle_phi_cache: list[Tensor | None] = [None] * len(phi_list)
    if oracle is not None:
        has_gt = any(
            getattr(_ALL_METRICS.get(n, type(None)), "requires_oracle", False)
            for n in metric_names
            if n in _ALL_METRICS
        )
        if has_gt:
            log.debug(
                "Pre-computing oracle Shapley values for %d samples (n_mc=%d).",
                len(phi_list),
                oracle_n_mc,
            )
            for idx, (x_i, target_i) in enumerate(zip(x_list, target_list)):
                try:
                    oracle_phi_cache[idx] = oracle.true_shapley(
                        x_i, _make_clf_fn(target_i), players,
                        n_mc=oracle_n_mc, n_coalitions=oracle_n_coalitions,
                    )
                except Exception as exc:
                    log.warning("Oracle pre-computation failed for sample %d: %s", idx, exc)

    # Accumulate per-sample scores for later averaging.
    all_scores: dict[str, list[float]] = {}

    for name in metric_names:
        if name not in _ALL_METRICS:
            log.debug("Unknown metric %r — skipped.", name)
            continue
        metric_cls = _ALL_METRICS[name]

        requires_imputer = getattr(metric_cls, "requires_imputer", False)
        requires_oracle = getattr(metric_cls, "requires_oracle", False)

        if requires_oracle and oracle is None:
            log.debug("Metric %r skipped — oracle not available.", name)
            continue
        if requires_imputer and imputer is None:
            log.debug("Metric %r skipped — no imputer for this method.", name)
            continue

        if name in _NEEDS_MODULE and method_explain_func is None:
            log.debug(
                "Metric %r skipped — attributor provides no explain_func "
                "(SHAP-based methods cannot re-attribute per perturbation step).",
                name,
            )
            continue

        # Build metric instance (once per metric, shared across samples).
        metric_kwargs: dict[str, Any] = {}
        if name == "max_sensitivity":
            metric_kwargs["nr_samples"] = max_sensitivity_nr
        if requires_oracle:
            # n_mc is irrelevant here because oracle.true_shapley is bypassed
            # by _SampleCachedOracle — the cached phi is returned directly.
            metric_kwargs["n_mc"] = oracle_n_mc
        metric = (
            metric_cls(imputer=imputer, **metric_kwargs)
            if requires_imputer
            else metric_cls(**metric_kwargs)
        )

        extra_kwargs: dict[str, Any] = {}
        if name in _NEEDS_MODULE:
            extra_kwargs["explain_func"] = method_explain_func

        # Stability/sanity use a limited subset; all others use the full list.
        # Track sample index so we can look up the pre-computed oracle phi.
        indexed_triples = list(enumerate(zip(phi_list, x_list, target_list)))
        if name in _NEEDS_MODULE:
            indexed_triples = indexed_triples[:stability_n]

        for sample_idx, (phi_i, x_i, target_i) in indexed_triples:
            # GT metrics receive a per-sample clf_fn so the oracle game exactly
            # matches the game used to produce phi_i (same target class).
            # Stability/sanity receive the raw module for gradient/param access.
            clf_arg: Any = classifier if name in _NEEDS_MODULE else _make_clf_fn(target_i)

            # Wrap the oracle with the pre-computed phi so each GT metric call
            # immediately returns the cached vector (no re-evaluation).
            cached_phi = oracle_phi_cache[sample_idx]
            oracle_arg: Any = (
                _SampleCachedOracle(oracle, cached_phi)
                if (oracle is not None and cached_phi is not None)
                else oracle
            )

            try:
                result = metric.evaluate(
                    phi=phi_i,
                    x=x_i,
                    classifier=clf_arg,
                    players=players,
                    target=target_i,
                    oracle=oracle_arg,
                    imputer=imputer,
                    **extra_kwargs,
                )
                for k, v in result.items():
                    all_scores.setdefault(k, []).append(v)
            except Exception as exc:
                log.warning("Metric %r on sample failed: %s", name, exc)

    scores: dict[str, float] = {}
    for k, vals in all_scores.items():
        finite = [v for v in vals if v == v]  # filter NaN
        scores[k] = float(_np.mean(finite)) if finite else float("nan")
    return scores


# ---------------------------------------------------------------------------
# Cell runner (single (dataset, clf, method) triple)
# ---------------------------------------------------------------------------


def _run_cell(
    dataset_name: str,
    clf_name: str,
    method_name: str,
    cfg: DictConfig,
) -> dict[str, Any]:
    """Run one evaluation cell and return a result dict.

    Saves the result JSON to disk.  If the file already exists, the cached
    result is returned without recomputation (resumability).

    Args:
        dataset_name: Dataset config name (e.g. ``"gaussian_k4"``).
        clf_name: Classifier config name (e.g. ``"synthetic_mlp"``).
        method_name: Method config name (e.g. ``"kernelshap_zero"``).
        cfg: Root experiment DictConfig.

    Returns:
        Result dict with keys ``dataset``, ``classifier``, ``method``, and
        one entry per metric sub-score.
    """
    results_dir = Path(cfg.results_dir)
    result_path = results_dir / dataset_name / clf_name / method_name / "result.json"
    error_path = result_path.parent / "error.json"

    if result_path.exists():
        log.info("Skip %s/%s/%s (cached)", dataset_name, clf_name, method_name)
        return cast("dict[str, Any]", json.loads(result_path.read_text()))

    try:
        # ---- Dataset ----
        dataset_cfg = _load_sub_config("data", dataset_name, cfg)
        dataset, K = _instantiate_dataset(dataset_cfg)
        J, F, T = dataset.shape

        # ---- Classifier ----
        clf_cfg = _load_sub_config("classifiers", clf_name, cfg)
        n_classes = int(str(dataset.metadata.get("n_classes", 3)))
        classifier = _build_classifier(clf_cfg, J, F, T, K, n_classes)
        device: str = str(cfg.get("device", "cpu"))
        classifier = classifier.to(torch.device(device))
        classifier.eval()  # required: BatchNorm uses running stats in eval mode

        # Load per-dataset checkpoint if available
        _ckpt_root = Path(
            cfg.get("checkpoint_dir", "motionbench/classifiers/checkpoints/synthetic")
        )
        _ckpt_path = _ckpt_root / dataset_name / f"{clf_name}.pt"
        if _ckpt_path.exists():
            _ckpt = torch.load(_ckpt_path, map_location=torch.device(device))
            classifier.load_state_dict(_ckpt["model_state_dict"])
            log.info(
                "Loaded checkpoint %s (val_acc=%.3f)",
                _ckpt_path,
                _ckpt.get("val_acc", float("nan")),
            )
        else:
            log.warning(
                "No checkpoint found at %s — using random initialisation.", _ckpt_path
            )

        # ---- Method ----
        method_cfg = _load_sub_config("methods", method_name, cfg)
        players = _build_players(method_cfg, J, F, T, K)

        # Build imputer if the method config specifies one
        has_imputer = OmegaConf.select(method_cfg, "imputer") is not None
        imputer: BaseImputer | None = None
        if has_imputer:
            imputer = _build_and_fit_imputer(method_cfg, dataset, J, F, T, device)

        attributor = _build_attributor(method_cfg, classifier, imputer, players)

        # ---- Attribution loop (with .npz cache) ----
        n_seq = int(cfg.get("n_sequences", 100))
        n_seq = min(n_seq, len(dataset))
        clf_device = torch.device(device)

        cache_path = result_path.parent / "attributions.npz"
        phi_list: list[Tensor] = []
        x_list: list[Tensor] = []
        target_list: list[int] = []

        # Try cache first — keyed on n_seq and dataset/clf/method (path is unique).
        cache_loaded = False
        if cache_path.exists():
            try:
                _z = np.load(cache_path, allow_pickle=False)
                _phi = _z["phi"]
                _x = _z["x"]
                _tgt = _z["target"]
                if _phi.shape[0] == n_seq:
                    phi_list = [torch.from_numpy(_phi[i]) for i in range(n_seq)]
                    x_list = [torch.from_numpy(_x[i]) for i in range(n_seq)]
                    target_list = [int(t) for t in _tgt.tolist()]
                    cache_loaded = True
                    log.info(
                        "Loaded cached attributions for %s/%s/%s (n=%d)",
                        dataset_name, clf_name, method_name, n_seq,
                    )
            except Exception as exc:  # pragma: no cover — cache is optional
                log.warning("Failed to load attribution cache %s: %s", cache_path, exc)

        if not cache_loaded:
            # Use the model's predicted class (argmax) as the target for each
            # sample — the field standard (OpenXAI, SHAP, SHAPIQ) for
            # explaining actual decisions. Per-sample targets are oracle-
            # consistent because _evaluate_metrics builds a per-sample clf_fn_i
            # for each oracle call.
            for idx in range(n_seq):
                x, _y = dataset[idx]
                x = x.to(clf_device)
                with torch.no_grad():
                    logits = classifier(x.unsqueeze(0))
                target_i = int(logits.argmax(dim=-1).item())
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    phi = attributor.attribute(x, players, target=target_i)
                phi_list.append(phi.detach().cpu())
                x_list.append(x.cpu())
                target_list.append(target_i)
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache_path,
                    phi=np.stack([p.numpy() for p in phi_list], axis=0),
                    x=np.stack([t.numpy() for t in x_list], axis=0),
                    target=np.asarray(target_list, dtype=np.int64),
                )
            except Exception as exc:  # pragma: no cover — cache is optional
                log.warning("Failed to write attribution cache %s: %s", cache_path, exc)

        # ---- Metric evaluation (averaged over all sequences) ----
        metric_names = _collect_metric_names(cfg)
        # For fidelity metrics: use method's imputer if available, else ZeroImputer.
        fidelity_imputer = imputer if imputer is not None else ZeroImputer().fit(dataset)
        scores = _evaluate_metrics(
            phi_list=phi_list,
            x_list=x_list,
            target_list=target_list,
            classifier=classifier,
            players=players,
            dataset=dataset,
            imputer=fidelity_imputer,
            metric_names=metric_names,
            cfg=cfg,
            device=device,
            attributor=attributor,
        )

        result: dict[str, Any] = {
            "dataset": dataset_name,
            "classifier": clf_name,
            "method": method_name,
            "n_sequences": n_seq,
            **scores,
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2))
        log.info(
            "Done %s/%s/%s — %d metrics", dataset_name, clf_name, method_name, len(scores)
        )
        return result

    except Exception as exc:
        log.warning("Cell %s/%s/%s failed: %s", dataset_name, clf_name, method_name, exc)
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(json.dumps({"error": str(exc), "type": type(exc).__name__}))
        return {
            "dataset": dataset_name,
            "classifier": clf_name,
            "method": method_name,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# WandB helpers
# ---------------------------------------------------------------------------


def _init_wandb(cfg: DictConfig) -> None:
    """Initialise a WandB run.

    Args:
        cfg: Root experiment DictConfig.
    """
    mode = str(OmegaConf.select(cfg, "wandb.mode", default="disabled"))
    if mode == "disabled":
        return
    try:
        import wandb  # noqa: PLC0415

        raw_cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=str(OmegaConf.select(cfg, "wandb.project", default="motionbench-xai")),
            entity=cast("str | None", OmegaConf.select(cfg, "wandb.entity")),
            tags=list(OmegaConf.select(cfg, "wandb.tags", default=[])),
            config=cast("dict[str, Any]", raw_cfg),
            mode=mode,  # type: ignore[arg-type]
        )
    except ImportError:
        log.warning("wandb not installed — WandB logging disabled.")


def _log_to_wandb(result: dict[str, Any]) -> None:
    """Log a result dict to WandB if a run is active.

    Args:
        result: Result dict from :func:`_run_cell`.
    """
    try:
        import wandb  # noqa: PLC0415

        if wandb.run is not None:
            wandb.log(result)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_synthetic_eval(cfg: DictConfig) -> pd.DataFrame:
    """Run the full (method × dataset × classifier) grid for synthetic data.

    Reads all dataset/method/classifier configs by name from ``configs/``,
    instantiates objects, runs attributions, evaluates metrics, and saves
    results to ``cfg.results_dir``.

    Args:
        cfg: Hydra DictConfig for the experiment (e.g. ``full_synthetic_sweep``).

    Returns:
        :class:`pandas.DataFrame` with one row per completed (or cached) cell.
        Columns: ``dataset``, ``classifier``, ``method``, one column per
        metric sub-score.
    """
    _init_wandb(cfg)

    datasets: list[str] = list(cfg.datasets)
    methods: list[str] = list(cfg.methods)
    classifiers: list[str] = list(cfg.classifiers)
    n_jobs = int(cfg.get("n_jobs", 1))

    cells = [
        (ds, clf, mth)
        for ds in datasets
        for clf in classifiers
        for mth in methods
    ]
    log.info(
        "Launching %d cells (%d datasets × %d classifiers × %d methods) "
        "with n_jobs=%d",
        len(cells),
        len(datasets),
        len(classifiers),
        len(methods),
        n_jobs,
    )

    def _run(ds: str, clf: str, mth: str) -> dict[str, Any]:
        result = _run_cell(ds, clf, mth, cfg)
        _log_to_wandb(result)
        return result

    if n_jobs == 1:
        results = [_run(ds, clf, mth) for ds, clf, mth in cells]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_run)(ds, clf, mth) for ds, clf, mth in cells
        )

    try:
        import wandb  # noqa: PLC0415

        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass

    return pd.DataFrame(results) if results else pd.DataFrame()
