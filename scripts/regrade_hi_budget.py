"""scripts/regrade_hi_budget.py — Hi-budget grading targets for sampled player-set cells.

For a (dataset, player set, classifier) triple whose player count exceeds the
exact-enumeration limit (M > 12), the stored grading target of the player-set
sweep is itself a sampled-coalition estimate.  This script rebuilds that
target at a *high* budget and regrades stored per-sequence attributions
against it — the protocol behind the hi-budget columns of
``results/canonical/synthetic_players.json`` (``meta.grading`` /
``meta.grading_floors``):

1. Build the hi-budget coalition design ``(Z, w)`` via
   :func:`~motionbench.attribution.sampled_coalitions.sampled_coalition_set`
   with a seed disjoint from every method design (methods use seed 7919).
   Canonical stored targets: budget 8192 with seed 900001 (all sampled cells);
   budget 32768 with seed 910001 (the M=68 datasets); floor replicates used
   seeds 910001 / 920001.
2. Compute the deterministic f-of-mean targets on that design through the
   given classifier: the conditional game via
   :class:`~motionbench.oracles.deterministic.DeterministicConditionalOracle`
   (closed-form ``E[x_hid | x_obs]``) and the marginal game via
   :func:`~motionbench.oracles.deterministic.marginal_fill` (zero fill, since
   ``E[x] = 0`` exactly for every synthetic family).
3. Save ``star_cond`` / ``star_marg`` to a targets ``.npz`` (resumable: an
   existing file is loaded, not recomputed).
4. Regrade every stored per-sequence attribution found under
   ``--attributions`` (one sub-directory per method, each holding the
   sweep's ``per_sequence.npz``) against the game-matched hi target, writing
   ``regrade_hi.json`` (``regrade_hi_b{budget}.json`` for budgets != 8192)
   next to each ``per_sequence.npz``.

Attributions themselves are untouched — only the grading target moves.

Usage::

    PYTHONPATH=. python scripts/regrade_hi_budget.py \\
        --dataset skeleton_structured --playerset cell \\
        --classifier synthetic_mlp \\
        --attributions results/player_eval/cells/skeleton_structured/synthetic_mlp

    # M=68 canonical target: --budget 32768 --seed 910001
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motionbench.attribution.sampled_coalitions import (  # noqa: E402
    phi_from_values,
    sampled_coalition_set,
)
from motionbench.oracles.deterministic import (  # noqa: E402
    DeterministicConditionalOracle,
    marginal_fill,
)
from motionbench.pipelines.synthetic_eval import (  # noqa: E402
    build_classifier,
    instantiate_dataset,
)
from motionbench.players import JointWindowCells, SpatialJoints  # noqa: E402

# Canonical hi-target seed (disjoint from the method designs' seed 7919).
HI_SEED = 900001

# Game assignment per method (the paper's table; KS-Empirical plays the
# marginal game).  Both the release method names and the canonical files'
# short names are accepted, so stored result trees regrade unchanged.
GAME = {
    "kernelshap_zero": "marg",
    "kernelshap_mean": "marg",
    "kernelshap_marginal": "marg",
    "kernelshap_empirical": "marg",
    "kernelshap_vaeac": "cond",
    "kernelshap_flow": "cond",
    "kernelshap_gauss": "cond",
    "kernelshap_oracle": "cond",
    "ks_zero": "marg",
    "ks_mean": "marg",
    "ks_marginal": "marg",
    "ks_empirical": "marg",
    "ks_vaeac": "cond",
    "ks_flow": "cond",
    "ks_shapr": "cond",
    "ks_oracle": "cond",
}


def batched_prob_fn(classifier, target, device, batch=2048):
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

    def fn(arr):
        """Batched softmax-probability evaluation of ``(n, J, F, T)`` numpy input."""
        vals = []
        with torch.no_grad():
            for s in range(0, len(arr), batch):
                xb = torch.from_numpy(np.ascontiguousarray(arr[s : s + batch], dtype=np.float32))
                probs = torch.softmax(classifier(xb.to(device)), -1)
                vals.append(probs[:, target].float().cpu().numpy())
        return np.concatenate(vals).astype(np.float64)

    return fn


def build_targets(args, dataset, players, K, Z, w, device):
    """Compute ``(star_cond, star_marg)`` deterministic hi-budget targets.

    Args:
        args: Parsed CLI arguments (classifier / checkpoint / sizes).
        dataset: Instantiated synthetic dataset (must expose ``oracle``).
        players: PlayerSet defining the coalition -> mask expansion.
        K: Number of temporal windows (classifier shape argument).
        Z: ``(n_rows, M)`` binary coalition design.
        w: ``(n_rows,)`` design weights.
        device: Torch device for the classifier forwards.

    Returns:
        ``(star_cond, star_marg)`` float64 arrays of shape ``(n_seq, M)``.
    """
    J, F, T = dataset.shape
    n_classes = int(str(dataset.metadata.get("n_classes", 3)))
    clf_cfg = OmegaConf.load(REPO_ROOT / "configs" / "classifiers" / f"{args.classifier}.yaml")
    classifier = build_classifier(clf_cfg, J=J, F=F, T=T, K=K, n_classes=n_classes)
    ckpt_path = Path(args.checkpoint_dir) / args.dataset / f"{args.classifier}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    classifier.load_state_dict(state)
    classifier.to(device).eval()
    log.info("classifier loaded strictly from %s", ckpt_path)

    det = DeterministicConditionalOracle.from_oracle(dataset.oracle, players, Z)
    masks_np = [
        players.coalition_mask(torch.as_tensor(z != 0, dtype=torch.bool).clone()).cpu().numpy()
        for z in np.asarray(Z, dtype=np.int64)
    ]

    n_seq = min(args.n_sequences, len(dataset))
    M = players.n_players
    star_cond = np.empty((n_seq, M))
    star_marg = np.empty((n_seq, M))
    t0 = time.time()
    for idx in range(n_seq):
        x_t, _y = dataset[idx]
        x64 = x_t.detach().cpu().numpy().astype(np.float64)
        with torch.no_grad():
            logits = classifier(x_t[None].to(device))
        target = int(logits.argmax(dim=-1).item())
        fn = batched_prob_fn(classifier, target, device)
        star_cond[idx] = phi_from_values(Z, w, fn(det.fill_all(x64)))
        zfills = np.stack([marginal_fill(x64, m) for m in masks_np])
        star_marg[idx] = phi_from_values(Z, w, fn(zfills))
        if idx % 20 == 0:
            log.info(
                "[%s/%s/%s] %d/%d (%.0fs)",
                args.playerset,
                args.dataset,
                args.classifier,
                idx,
                n_seq,
                time.time() - t0,
            )
    return star_cond, star_marg


def regrade(attr_dir, star_cond, star_marg, budget, seed):
    """Regrade every stored method attribution under ``attr_dir``.

    Args:
        attr_dir: Directory with one sub-directory per method, each holding
            the sweep's ``per_sequence.npz`` (``phi`` array).
        star_cond: ``(n, M)`` conditional hi-budget target.
        star_marg: ``(n, M)`` marginal hi-budget target.
        budget: Target budget (names the output json).
        seed: Target coalition seed (recorded in the output json).
    """
    suffix = "regrade_hi.json" if budget == 8192 else f"regrade_hi_b{budget}.json"
    for cell in sorted(p for p in attr_dir.iterdir() if p.is_dir()):
        method = cell.name
        if method not in GAME or not (cell / "per_sequence.npz").exists():
            continue
        game = GAME[method]
        phi = np.load(cell / "per_sequence.npz")["phi"]
        star = star_cond if game == "cond" else star_marg
        n = min(len(phi), len(star))
        ec1 = np.mean(np.abs(phi[:n] - star[:n]), axis=1)
        ec3 = np.empty(n)
        for i in range(n):
            if np.std(phi[i]) < 1e-10 or np.std(star[i]) < 1e-10:
                ec3[i] = 1.0
            else:
                ec3[i] = np.clip(1 - np.corrcoef(phi[i], star[i])[0, 1], -1, 1)
        out = {
            "ec1": float(ec1.mean()),
            "ec3": float(ec3.mean()),
            "target_budget": budget,
            "target_seed": seed,
            "game": game,
            "n": int(n),
        }
        (cell / suffix).write_text(json.dumps(out))
        log.info("regraded %s: ec1=%.6f ec3=%.6f (%s)", method, out["ec1"], out["ec3"], game)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, help="configs/data/<name>.yaml config name.")
    ap.add_argument("--playerset", required=True, choices=["joint", "cell"])
    ap.add_argument(
        "--classifier", required=True, help="configs/classifiers/<name>.yaml config name."
    )
    ap.add_argument(
        "--budget",
        type=int,
        default=8192,
        help="Interior coalition rows (canonical: 8192; 32768 for M=68 with --seed 910001).",
    )
    ap.add_argument(
        "--seed", type=int, default=HI_SEED, help="Coalition seed of the hi-budget design."
    )
    ap.add_argument("--n_sequences", type=int, default=200)
    ap.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(REPO_ROOT / "motionbench" / "classifiers" / "checkpoints" / "synthetic"),
        help="Manifest layout: <dir>/<dataset>/<classifier>.pt (checkpoints/README.md).",
    )
    ap.add_argument(
        "--attributions",
        type=str,
        default=None,
        help="Result dir with per-method sub-dirs (per_sequence.npz) to regrade.",
    )
    ap.add_argument(
        "--targets",
        type=str,
        default=None,
        help="Targets npz path (default: <attributions>/hi_targets[_b{budget}].npz).",
    )
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    """Build (or load) the hi-budget targets and regrade stored attributions."""
    args = parse_args()
    if args.targets is None and args.attributions is None:
        raise SystemExit("Provide --attributions and/or --targets (see module docstring).")
    tgt_name = "hi_targets.npz" if args.budget == 8192 else f"hi_targets_b{args.budget}.npz"
    tgt_path = Path(args.targets) if args.targets else Path(args.attributions) / tgt_name

    dataset_cfg = OmegaConf.load(REPO_ROOT / "configs" / "data" / f"{args.dataset}.yaml")
    dataset, K = instantiate_dataset(dataset_cfg)
    J, F, T = dataset.shape
    players = SpatialJoints(J, F, T) if args.playerset == "joint" else JointWindowCells(J, K, F, T)
    M = players.n_players
    Z, w = sampled_coalition_set(M, args.budget, seed=args.seed)
    log.info(
        "%s/%s/%s: M=%d, design %d rows (budget %d, seed %d)",
        args.playerset,
        args.dataset,
        args.classifier,
        M,
        len(Z),
        args.budget,
        args.seed,
    )

    if tgt_path.exists():
        d = np.load(tgt_path)
        star_cond, star_marg = d["star_cond"], d["star_marg"]
        log.info("loaded stored targets from %s (n=%d)", tgt_path, len(star_cond))
    else:
        device = torch.device(args.device)
        star_cond, star_marg = build_targets(args, dataset, players, K, Z, w, device)
        tgt_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            tgt_path, star_cond=star_cond, star_marg=star_marg, budget=args.budget, seed=args.seed
        )
        log.info("saved targets to %s", tgt_path)

    if args.attributions:
        regrade(Path(args.attributions), star_cond, star_marg, args.budget, args.seed)


if __name__ == "__main__":
    main()
