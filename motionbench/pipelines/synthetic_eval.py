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

import pandas as pd
import torch
from hydra.utils import instantiate
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from torch import Tensor

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
    from motionbench.imputers.base import BaseImputer
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
    )

    _FIDELITY_METRICS = {
        "faithfulness_correlation": FaithfulnessCorrelationMetric,
        "pixel_flipping": PixelFlippingMetric,
    }
except ImportError:
    log.debug("motionbench.metrics.fidelity not available — fidelity metrics skipped.")

_STABILITY_METRICS: dict[str, Any] = {
    "max_sensitivity": MaxSensitivityMetric,
}

_SANITY_METRICS: dict[str, Any] = {
    "model_parameter_randomisation": ModelParameterRandomisationMetric,
}

_ALL_METRICS: dict[str, Any] = {
    **_GT_METRICS,
    **_FIDELITY_METRICS,
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

    # Learned imputers need J, F, T at construction time
    target_str = str(imputer_cfg.get("_target_", ""))
    if any(cls in target_str for cls in ("VAEACImputer", "FlowMatchingImputer")):
        imputer_cfg.update({"J": J, "F": F, "T": T})

    # Also pass train_epochs if specified in method config (for VAEAC/Flow)
    train_epochs = int(method_cfg.get("train_epochs", 0))

    imputer: BaseImputer = cast(
        "BaseImputer", instantiate(OmegaConf.create(imputer_cfg))
    )

    # Learned imputers may expose a device attribute
    if hasattr(imputer, "_device"):
        cast(Any, imputer)._device = torch.device(device)

    if train_epochs > 0 and hasattr(imputer, "_fit_epochs"):
        # VAEACImputer._fit_epochs takes a (N, J, F, T) Tensor, not a dataset
        import torch as _torch  # noqa: PLC0415

        xs = _torch.stack([dataset[i][0] for i in range(len(dataset))])
        cast(Any, imputer)._fit_epochs(xs, epochs=train_epochs)
        cast(Any, imputer)._fitted = True
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

    # ---- KernelSHAP / TimeSHAP: needs imputer ----
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
                classifier=classifier,
                imputer=imputer,
                n_samples=n_samples,
                n_completion_samples=n_compl,
                seed=seed,
            ),
        )

    if "TimeSHAPAttributor" in target_str:
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
                classifier=classifier,
                imputer=imputer,
                n_coalitions=n_coalitions,
                seed=seed,
            ),
        )

    if "WindowSHAPAttributor" in target_str:
        # WindowSHAP takes window_len/stride/seed from attr_cfg; no imputer
        return cast(
            "BaseAttributor",
            instantiate(OmegaConf.create(attr_cfg), classifier=classifier),
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
# Helpers — classifier wrapper for attributors
# ---------------------------------------------------------------------------


def _make_model_fn(
    classifier: Classifier,
    target: int,
    device: str = "cpu",
) -> Any:
    """Return a ``(B, J, F, T) → (B,)`` callable extracting class ``target``.

    Args:
        classifier: Any :class:`~motionbench.classifiers.base.Classifier`.
        target: Class index to extract from softmax probabilities.
        device: Torch device to move inputs to.

    Returns:
        A callable suitable for use as the ``classifier`` argument in
        :class:`~motionbench.attribution.base.BaseAttributor`.
    """
    clf_device = torch.device(device)
    classifier = classifier.to(clf_device)
    classifier.eval()

    def model_fn(x: Tensor) -> Tensor:
        with torch.no_grad():
            x = x.to(clf_device)
            logits = classifier(x)
        proba = torch.softmax(logits, dim=-1)
        if proba.ndim == 2:  # (B, n_classes)
            return proba[:, target]
        return proba  # already (B,)

    return model_fn


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


def _evaluate_metrics(
    phi: Tensor,
    x: Tensor,
    target: int,
    classifier: Classifier,
    players: PlayerSet,
    dataset: BaseDataset,
    imputer: BaseImputer | None,
    metric_names: list[str],
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate all requested metrics on a single (phi, x) pair.

    Args:
        phi: ``(M,)`` attribution vector.
        x: ``(J, F, T)`` input sequence.
        target: Class index used to produce ``phi``.
        classifier: Model used to produce ``phi``.
        players: Player set used to produce ``phi``.
        dataset: Source dataset (provides oracle for GT metrics).
        imputer: Fitted imputer (for fidelity metrics; may be ``None``).
        metric_names: Names of metrics to evaluate.
        device: Torch device string.

    Returns:
        Dict mapping metric sub-score names to float values.
    """
    oracle = dataset.oracle  # None for real datasets
    clf_device = torch.device(device)
    classifier = classifier.to(clf_device)
    classifier.eval()

    # clf_fn: efficient no_grad wrapper for GT and fidelity metrics.
    def clf_fn(b: Tensor) -> Tensor:
        with torch.no_grad():
            logits = classifier(b.to(clf_device))
        proba = torch.softmax(logits, dim=-1)
        if proba.ndim == 2:
            return proba[:, target]
        return proba

    # Metrics that need gradient flow or parameter enumeration (stability /
    # sanity) must receive the raw nn.Module so that:
    #   (a) _gradient_explain_func can call loss.backward(), and
    #   (b) Quantus MPRT can enumerate / randomise model parameters.
    _NEEDS_MODULE: frozenset[str] = frozenset(_STABILITY_METRICS) | frozenset(_SANITY_METRICS)

    scores: dict[str, float] = {}
    for name in metric_names:
        if name not in _ALL_METRICS:
            log.debug("Unknown metric %r — skipped.", name)
            continue
        metric_cls = _ALL_METRICS[name]

        # Some metric classes (FaithfulnessCorrelation, PixelFlipping) take
        # imputer as a required constructor argument.
        requires_imputer = getattr(metric_cls, "requires_imputer", False)
        requires_oracle = getattr(metric_cls, "requires_oracle", False)

        if requires_oracle and oracle is None:
            log.debug("Metric %r skipped — oracle not available.", name)
            continue
        if requires_imputer and imputer is None:
            log.debug("Metric %r skipped — no imputer for this method.", name)
            continue

        if requires_imputer:
            metric = metric_cls(imputer=imputer)
        else:
            metric = metric_cls()

        # Stability/sanity metrics receive the raw module; others get clf_fn.
        clf_arg = classifier if name in _NEEDS_MODULE else clf_fn

        try:
            result = metric.evaluate(
                phi=phi,
                x=x,
                classifier=clf_arg,
                players=players,
                target=target,
                oracle=oracle,
                imputer=imputer,
            )
            scores.update(result)
        except Exception as exc:
            log.warning("Metric %r failed: %s", name, exc)
            scores[name] = float("nan")

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
        return cast(dict[str, Any], json.loads(result_path.read_text()))

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
        target_class: int = 0  # always explain class 0 for synthetic experiments

        # ---- Method ----
        method_cfg = _load_sub_config("methods", method_name, cfg)
        players = _build_players(method_cfg, J, F, T, K)

        # Build imputer if the method config specifies one
        has_imputer = OmegaConf.select(method_cfg, "imputer") is not None
        imputer: BaseImputer | None = None
        if has_imputer:
            imputer = _build_and_fit_imputer(method_cfg, dataset, J, F, T, device)

        attributor = _build_attributor(method_cfg, classifier, imputer, players)

        # ---- Attribution loop ----
        n_seq = int(cfg.get("n_sequences", 100))
        n_seq = min(n_seq, len(dataset))

        phi_list: list[Tensor] = []
        x_list: list[Tensor] = []
        for idx in range(n_seq):
            x, _y = dataset[idx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                phi = attributor.attribute(x, players, target=target_class)
            phi_list.append(phi.detach().cpu())
            x_list.append(x.cpu())

        # ---- Metric evaluation (on first sample for speed) ----
        metric_names = _collect_metric_names(cfg)
        # Use imputer that targets the ZeroImputer for fidelity (cheapest)
        fidelity_imputer = imputer if imputer is not None else ZeroImputer().fit(dataset)
        scores = _evaluate_metrics(
            phi=phi_list[0],
            x=x_list[0],
            target=target_class,
            classifier=classifier,
            players=players,
            dataset=dataset,
            imputer=fidelity_imputer,
            metric_names=metric_names,
            device=device,
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
