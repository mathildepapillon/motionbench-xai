"""motionbench.pipelines.player_eval — Player-generic exact-grading evaluation.

Runs any :class:`~motionbench.players.base.PlayerSet` × any
:class:`~motionbench.imputers.base.BaseImputer` combination over a **shared
fixed coalition design** and grades the resulting Shapley attributions with
EC1/EC3 against the **deterministic** f-of-mean targets — no Monte-Carlo
grading noise (a perfect attribution scores exactly 0).

Protocol (per cell = dataset × player set × classifier × method)
----------------------------------------------------------------
1. Build the coalition design ``(Z, w)`` once per (player set, budget) via
   :func:`~motionbench.attribution.sampled_coalitions.sampled_coalition_set`
   (exact enumeration for M <= 12, shap-style importance-corrected sampling
   above).  Every method and the grading target share this design, so
   coalition noise cancels in the comparison.
2. For each evaluation sequence: the explained scalar is the classifier's
   softmax probability of its own argmax class.  The method's game value is

   * ``value_fn = "f_of_mean"`` (default; the executed release semantics):
     ``v(S) = f(mean of n_completion_samples imputer draws)``
   * ``value_fn = "mean_of_f"`` (paper Eq. 5):
     ``v(S) = mean over draws of f(completion)``

3. Attributions come from the constrained WLS solve
   (:func:`~motionbench.attribution.sampled_coalitions.phi_from_values`).
4. The grading target is the *deterministic* game the method's imputer
   declares (``game`` key in the method config, else inferred):

   * ``cond`` — ``v*(S) = f(x_S joined with E[x_hid | x_obs])`` via
     :class:`~motionbench.oracles.deterministic.DeterministicConditionalOracle`.
   * ``marg`` — ``v*(S) = f(x_S joined with E[x])`` with ``E[x] = 0`` exactly
     for the synthetic families
     (:func:`~motionbench.oracles.deterministic.marginal_fill`).

5. Per-sequence EC1 = mean|phi - phi*| and EC3 = 1 - Pearson(phi, phi*) are
   averaged over sequences; per-sequence arrays are stored in
   ``per_sequence.npz`` next to ``result.json``.

The runner protocol (shared coalition design, deterministic f-of-mean
targets, per-method game assignment) follows the independent validation
study's player-set sweep.

Hydra usage
-----------
::

    python -m motionbench.cli.run experiments=player_set_eval
    python -m motionbench.cli.run experiments=player_set_eval \\
        player_sets='[spatial]' coalition_budget=2048 value_fn=mean_of_f
"""

from __future__ import annotations

import inspect
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from motionbench.attribution.sampled_coalitions import (
    DEFAULT_COALITION_SEED,
    phi_from_values,
    sampled_coalition_set,
)
from motionbench.oracles.deterministic import (
    DeterministicConditionalOracle,
    marginal_fill,
)
from motionbench.pipelines.synthetic_eval import (
    _build_and_fit_imputer,
    _build_classifier,
    _init_wandb,
    _instantiate_dataset,
    _load_sub_config,
    _log_to_wandb,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from omegaconf import DictConfig
    from torch import Tensor

    from motionbench.imputers.base import BaseImputer
    from motionbench.players.base import PlayerSet

log = logging.getLogger(__name__)

__all__ = ["run_player_eval"]

# Methods whose imputer defines the conditional game (per the paper's game
# table; note KS-Empirical is assigned the *marginal* game there).  Used only
# when the method config does not carry an explicit ``game`` key.
_COND_IMPUTER_MARKERS = ("Oracle", "VAEAC", "Flow")


def _build_player_set(players_cfg: DictConfig, J: int, F: int, T: int, K: int) -> PlayerSet:
    """Instantiate a PlayerSet from a ``configs/players/*.yaml`` config.

    Injects ``J``/``F``/``T``/``K`` from the dataset, filtered to the
    parameters the target class constructor actually accepts (player-set
    constructors differ: e.g. ``SpatialJoints(J, F, T)`` has no ``K``).

    Args:
        players_cfg: Config with ``_target_`` and optional extra kwargs.
        J: Number of joints.
        F: Features per joint.
        T: Time steps.
        K: Number of temporal windows.

    Returns:
        Instantiated :class:`~motionbench.players.base.PlayerSet`.
    """
    cfg_dict = dict(OmegaConf.to_container(players_cfg, resolve=True))  # type: ignore[arg-type]
    cfg_dict.pop("name", None)
    target = str(cfg_dict["_target_"])
    module_name, _, cls_name = target.rpartition(".")
    cls = getattr(__import__(module_name, fromlist=[cls_name]), cls_name)
    accepted = set(inspect.signature(cls.__init__).parameters)
    for key, val in {"J": J, "F": F, "T": T, "K": K}.items():
        if key in accepted:
            cfg_dict[key] = val
    return instantiate(OmegaConf.create(cfg_dict))  # type: ignore[no-any-return]


def _infer_game(method_cfg: DictConfig) -> str:
    """Return the method's declared game (``"cond"`` or ``"marg"``).

    Reads the explicit ``game`` key when present; otherwise infers from the
    imputer target: conditional imputers (oracle, VAEAC, flow, empirical,
    KNN, copula) play the conditional game, off-manifold fills the marginal
    game.

    Args:
        method_cfg: Method config (``configs/methods/*.yaml``).

    Returns:
        ``"cond"`` or ``"marg"``.
    """
    game = str(method_cfg.get("game", "")).strip()
    if game in ("cond", "marg"):
        return game
    imputer_target = str(OmegaConf.select(method_cfg, "imputer._target_") or "")
    if any(marker in imputer_target for marker in _COND_IMPUTER_MARKERS):
        return "cond"
    return "marg"


def _batched_prob_fn(
    classifier: torch.nn.Module, target: int, device: torch.device, batch: int = 1024
) -> Any:
    """Wrap a classifier as ``(n, J, F, T) float32 -> (n,) float64`` softmax prob.

    Args:
        classifier: Torch module mapping ``(B, J, F, T)`` to ``(B, n_classes)``
            logits; must already be in eval mode on ``device``.
        target: Class index whose softmax probability is returned.
        device: Device to run the forward passes on.
        batch: Maximum forward batch size.

    Returns:
        Callable evaluating the scalar game payoff on numpy batches.
    """

    def fn(arr: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        vals = []
        with torch.no_grad():
            for s in range(0, len(arr), batch):
                xb = torch.from_numpy(
                    np.ascontiguousarray(arr[s : s + batch], dtype=np.float32)
                ).to(device)
                vals.append(torch.softmax(classifier(xb), -1)[:, target].float().cpu().numpy())
        out: npt.NDArray[np.float64] = np.concatenate(vals).astype(np.float64)
        return out

    return fn


def _method_values(
    clf_fn: Any,
    x: npt.NDArray[np.float64],
    imputer: BaseImputer,
    masks: list[Tensor],
    Z: npt.NDArray[np.integer[Any]],
    n_completion: int,
    value_fn: str,
    seq_seed: list[int],
) -> npt.NDArray[np.float64]:
    """Evaluate the method's game value ``v(S)`` on every coalition row.

    Args:
        clf_fn: ``(n, J, F, T) float32 -> (n,) float64`` payoff function.
        x: ``(J, F, T)`` float64 sequence being explained.
        imputer: Fitted imputer defining the completion distribution.
        masks: Precomputed element-level masks, aligned with ``Z`` rows.
        Z: ``(n_rows, M)`` binary coalition matrix.
        n_completion: Completion draws per coalition.
        value_fn: ``"f_of_mean"`` (release) or ``"mean_of_f"`` (paper Eq. 5).
        seq_seed: Seed-sequence prefix; coalition ``i`` draws with seed
            ``seq_seed + [i]`` so draws are independent across coalitions but
            fully reproducible.

    Returns:
        ``(n_rows,)`` float64 game values.

    Raises:
        ValueError: for an unknown ``value_fn``.
    """
    if value_fn not in ("f_of_mean", "mean_of_f"):
        raise ValueError(f"Unknown value_fn {value_fn!r}; use 'f_of_mean' or 'mean_of_f'.")
    J, F, T = x.shape
    x_t = torch.from_numpy(x.astype(np.float32))
    fills = np.empty((len(Z), J, F, T), dtype=np.float32)
    v_mean_f = np.empty(len(Z), dtype=np.float64)
    for i, z in enumerate(Z):
        if bool(z.astype(bool).all()):
            comps_np = x.astype(np.float32)[None]
        else:
            seed_i = int(np.random.default_rng(seq_seed + [i]).integers(2**31))
            comps = imputer.impute(x_t, masks[i], n_completion, seed=seed_i)
            comps_np = comps.detach().cpu().numpy().astype(np.float32)
        if value_fn == "f_of_mean":
            fills[i] = comps_np.mean(axis=0)
        else:
            v_mean_f[i] = float(np.mean(clf_fn(comps_np)))
    if value_fn == "f_of_mean":
        return np.asarray(clf_fn(fills), dtype=np.float64)
    return v_mean_f


def _ec_metrics(
    phi: npt.NDArray[np.float64], phi_star: npt.NDArray[np.float64]
) -> tuple[float, float]:
    """Per-sequence (EC1, EC3) against the deterministic target.

    EC1 = mean|phi - phi*|; EC3 = 1 - Pearson(phi, phi*) with the Pearson
    coefficient clamped to [-1, 1] (so EC3 in [0, 2]) and a constant vector
    treated as correlation 0 (EC3 = 1) — the same conventions as
    :class:`~motionbench.metrics.ground_truth.EC3Metric`.

    Args:
        phi: ``(M,)`` method attribution.
        phi_star: ``(M,)`` deterministic target attribution.

    Returns:
        ``(ec1, ec3)`` floats.
    """
    ec1 = float(np.mean(np.abs(phi - phi_star)))
    if np.std(phi) < 1e-10 or np.std(phi_star) < 1e-10:
        return ec1, 1.0
    pearson = float(np.clip(np.corrcoef(phi, phi_star)[0, 1], -1.0, 1.0))
    return ec1, 1.0 - pearson


def _run_player_cell(
    dataset_name: str,
    players_name: str,
    clf_name: str,
    method_name: str,
    cfg: DictConfig,
) -> dict[str, Any]:
    """Run one (dataset, player set, classifier, method) evaluation cell.

    Writes ``result.json`` and ``per_sequence.npz`` under
    ``{results_dir}/{players_name}/{dataset_name}/{clf_name}/{method_name}/``
    and skips cells whose ``result.json`` already exists (resumable sweeps).

    Args:
        dataset_name: Name of a ``configs/data/*.yaml`` config.
        players_name: Name of a ``configs/players/*.yaml`` config.
        clf_name: Name of a ``configs/classifiers/*.yaml`` config.
        method_name: Name of a ``configs/methods/*.yaml`` config (must define
            an imputer).
        cfg: Root experiment config.

    Returns:
        Flat result dict (one leaderboard row), or an ``error`` dict if the
        cell failed.
    """
    results_dir = Path(str(cfg.get("results_dir", "results/player_eval")))
    cell_dir = results_dir / players_name / dataset_name / clf_name / method_name
    result_path = cell_dir / "result.json"
    if result_path.exists():
        log.info(
            "[%s/%s/%s/%s] cached result found — skipping.",
            players_name,
            dataset_name,
            clf_name,
            method_name,
        )
        return dict(json.loads(result_path.read_text()))

    try:
        device = torch.device(str(cfg.get("device", "cpu")))
        n_seq = int(cfg.get("n_sequences", 200))
        budget = int(cfg.get("coalition_budget", 1024))
        coalition_seed = int(cfg.get("coalition_seed", DEFAULT_COALITION_SEED))
        value_fn = str(cfg.get("value_fn", "f_of_mean"))
        seed = int(cfg.get("seed", 42))

        dataset_cfg = _load_sub_config("data", dataset_name, cfg)
        dataset, K = _instantiate_dataset(dataset_cfg)
        J, F, T = dataset.shape
        n_seq = min(n_seq, len(dataset))

        players_cfg = _load_sub_config("players", players_name, cfg)
        players = _build_player_set(players_cfg, J=J, F=F, T=T, K=K)
        M = players.n_players
        Z, w = sampled_coalition_set(M, budget, seed=coalition_seed)
        masks = [
            players.coalition_mask(torch.as_tensor(z != 0, dtype=torch.bool).clone())
            for z in np.asarray(Z, dtype=np.int64)
        ]

        clf_cfg = _load_sub_config("classifiers", clf_name, cfg)
        n_classes = int(str(dataset.metadata.get("n_classes", 3)))
        classifier = _build_classifier(clf_cfg, J=J, F=F, T=T, K=K, n_classes=n_classes)
        ckpt_dir = Path(
            str(cfg.get("checkpoint_dir", "motionbench/classifiers/checkpoints/synthetic"))
        )
        ckpt_path = ckpt_dir / dataset_name / f"{clf_name}.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            classifier.load_state_dict(state)
        else:
            log.warning("No checkpoint at %s — using random initialisation.", ckpt_path)
        classifier.to(device).eval()

        method_cfg = _load_sub_config("methods", method_name, cfg)
        if OmegaConf.select(method_cfg, "imputer") is None:
            raise ValueError(
                f"Method {method_name!r} defines no imputer; the player-set "
                "pipeline evaluates imputer-based KernelSHAP methods only."
            )
        imputer = _build_and_fit_imputer(method_cfg, dataset, J=J, F=F, T=T, device=str(device))
        n_completion = int(method_cfg.get("n_completion_samples", 5))
        game = _infer_game(method_cfg)

        det: DeterministicConditionalOracle | None = None
        if game == "cond":
            oracle = getattr(dataset, "oracle", None)
            if oracle is None:
                raise ValueError(
                    f"Dataset {dataset_name!r} exposes no oracle; the "
                    "conditional deterministic target needs the generative "
                    "covariances."
                )
            det = DeterministicConditionalOracle.from_oracle(oracle, players, Z)
        masks_np = [m.cpu().numpy() for m in masks]

        cell_dir.mkdir(parents=True, exist_ok=True)
        phis, stars, ec1s, ec3s, targets = [], [], [], [], []
        t0 = time.time()
        for idx in range(n_seq):
            x_t, _y = dataset[idx]
            x64 = x_t.detach().cpu().numpy().astype(np.float64)
            with torch.no_grad():
                logits = classifier(x_t[None].to(device))
            target = int(logits.argmax(dim=-1).item())
            clf_fn = _batched_prob_fn(classifier, target, device)

            v = _method_values(
                clf_fn, x64, imputer, masks, Z, n_completion, value_fn, seq_seed=[seed, idx]
            )
            phi = phi_from_values(Z, w, v)

            if det is not None:
                v_star = np.asarray(clf_fn(det.fill_all(x64)), dtype=np.float64)
            else:
                zfills = np.stack([marginal_fill(x64, m) for m in masks_np])
                v_star = np.asarray(clf_fn(zfills), dtype=np.float64)
            phi_star = phi_from_values(Z, w, v_star)

            ec1, ec3 = _ec_metrics(phi, phi_star)
            phis.append(phi)
            stars.append(phi_star)
            ec1s.append(ec1)
            ec3s.append(ec3)
            targets.append(target)
            if idx % 20 == 0:
                log.info(
                    "[%s/%s/%s/%s] %d/%d ec1=%.4f ec3=%.4f (%.0fs)",
                    players_name,
                    dataset_name,
                    clf_name,
                    method_name,
                    idx,
                    n_seq,
                    ec1,
                    ec3,
                    time.time() - t0,
                )

        np.savez_compressed(
            cell_dir / "per_sequence.npz",
            phi=np.array(phis),
            phi_star=np.array(stars),
            ec1=np.array(ec1s),
            ec3=np.array(ec3s),
            target=np.array(targets),
        )
        result: dict[str, Any] = {
            "dataset": dataset_name,
            "players": players_name,
            "classifier": clf_name,
            "method": method_name,
            "game": game,
            "M": int(M),
            "n_coalitions": int(len(Z)),
            "exact_coalitions": bool(len(Z) >= 2**M),
            "coalition_seed": coalition_seed,
            "coalition_budget": budget,
            "value_fn": value_fn,
            "n_sequences": int(n_seq),
            "seed": seed,
            "ec1": float(np.mean(ec1s)),
            "ec3": float(np.mean(ec3s)),
            "elapsed_s": time.time() - t0,
        }
        result_path.write_text(json.dumps(result, indent=1))
        return result
    except Exception as exc:  # noqa: BLE001 - sweep must survive cell failures
        log.exception(
            "[%s/%s/%s/%s] cell failed", players_name, dataset_name, clf_name, method_name
        )
        cell_dir.mkdir(parents=True, exist_ok=True)
        error: dict[str, Any] = {
            "dataset": dataset_name,
            "players": players_name,
            "classifier": clf_name,
            "method": method_name,
            "error": str(exc),
        }
        (cell_dir / "error.json").write_text(json.dumps(error, indent=1))
        return error


def run_player_eval(cfg: DictConfig) -> pd.DataFrame:
    """Run the player-set evaluation sweep defined by ``cfg``.

    Iterates the full product of ``cfg.datasets`` × ``cfg.player_sets`` ×
    ``cfg.classifiers`` × ``cfg.methods`` serially (cells are resumable via
    their ``result.json``).

    Args:
        cfg: Hydra config; see ``configs/experiments/player_set_eval.yaml``.

    Returns:
        One row per completed cell (columns: dataset, players, classifier,
        method, game, ec1, ec3, ...).
    """
    _init_wandb(cfg)
    datasets = [str(d) for d in cfg.datasets]
    player_sets = [str(p) for p in cfg.get("player_sets", ["temporal"])]
    classifiers = [str(c) for c in cfg.classifiers]
    methods = [str(m) for m in cfg.methods]

    cells = [
        (ds, ps, clf, mth)
        for ds in datasets
        for ps in player_sets
        for clf in classifiers
        for mth in methods
    ]
    log.info(
        "Player-set sweep: %d cells (%d datasets x %d player sets x %d classifiers x %d methods)",
        len(cells),
        len(datasets),
        len(player_sets),
        len(classifiers),
        len(methods),
    )

    results = []
    for ds, ps, clf, mth in cells:
        result = _run_player_cell(ds, ps, clf, mth, cfg)
        _log_to_wandb(result)
        results.append(result)

    try:
        import wandb  # noqa: PLC0415

        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass

    return pd.DataFrame(results) if results else pd.DataFrame()
