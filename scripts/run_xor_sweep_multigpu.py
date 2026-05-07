"""scripts/run_xor_sweep_multigpu.py — End-to-end multi-GPU XOR sweep runner.

This script drives the full evaluation pipeline for the new
``xor_label_gaussian`` synthetic dataset, distributing work across all
available CUDA devices with a configurable number of jobs per GPU.

Pipeline stages
---------------
1.  **Train classifiers.**  Three architectures (MLP, CNN, Transformer) are
    trained in parallel — one per GPU — by delegating to
    :mod:`scripts.train_synthetic_clf`.

2.  **Verify imputers.**  ``xor_label_gaussian`` re-uses the same
    :class:`GaussianMotionDataset` class with identical (J=5, F=3, T=16,
    rho=0.5, alpha=0.8) covariance as the existing ``gaussian_k4`` baseline,
    so the pre-trained VAEAC and Flow checkpoints registered for
    ``GaussianMotionDataset`` already match this distribution.  No imputer
    retraining is required.

3.  **Run SHAP / IG / gradient sweep.**  All ``len(METHODS) × len(CLASSIFIERS)``
    cells are dispatched as independent subprocesses, scheduled across
    ``len(gpus) × jobs_per_gpu`` worker slots.  Each worker pins itself to a
    single GPU via ``CUDA_VISIBLE_DEVICES`` and invokes the existing
    ``motionbench run`` Hydra entry point with single-cell overrides.

Why subprocess-per-cell?
------------------------
* It re-uses the existing, well-tested pipeline code path verbatim.
* Each subprocess sets ``CUDA_VISIBLE_DEVICES`` *before* importing torch, so
  GPU pinning is exact.  Joblib worker initialisers cannot guarantee this
  ordering for already-loaded torch processes.
* Failures are isolated — one bad cell does not corrupt other workers.

Usage
-----
::

    # Use all 8 GPUs, 2 jobs per GPU (16 concurrent cells)
    python scripts/run_xor_sweep_multigpu.py

    # Override GPU set / parallelism
    python scripts/run_xor_sweep_multigpu.py --gpus 0 1 2 3 --jobs-per-gpu 4

    # Skip the classifier-training step (already done)
    python scripts/run_xor_sweep_multigpu.py --skip-train

    # Run only a subset of methods (debugging)
    python scripts/run_xor_sweep_multigpu.py --methods kernelshap_zero kernelshap_vaeac
"""
from __future__ import annotations

import argparse
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[1]

DATASET = "xor_label_gaussian"

CLASSIFIERS: list[str] = [
    "synthetic_mlp",
    "synthetic_cnn",
    "synthetic_transformer",
]

METHODS: list[str] = [
    "kernelshap_zero",
    "kernelshap_mean",
    "kernelshap_marginal",
    "kernelshap_empirical",
    "kernelshap_vaeac",
    "kernelshap_flow",
    "ig_zero",
    "ig_mean",
    "deeplift",
    "gradientshap",
    "smoothgrad",
    "lrp",
    "kernelshap_temporal",
    "windowshap",
    "shats",
    "gradcam",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    """One unit of SHAP work: (dataset, classifier, method)."""

    dataset: str
    classifier: str
    method: str
    gpu: int = -1
    elapsed_s: float = 0.0
    rc: int = -1
    log_path: Path | None = None

    @property
    def label(self) -> str:
        return f"{self.dataset}/{self.classifier}/{self.method}"


def _detect_gpus() -> list[int]:
    """Return the list of visible CUDA devices, or [] if none."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        )
        return [int(x) for x in out.decode().strip().splitlines()]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


def _result_exists(cell: Cell, results_dir: Path) -> bool:
    return (results_dir / cell.dataset / cell.classifier / cell.method
            / "result.json").exists()


def _run_cell(cell: Cell, gpu: int, results_dir: Path, n_sequences: int,
              log_dir: Path) -> Cell:
    """Run a single cell as a subprocess pinned to ``gpu``.

    The subprocess invokes the standard ``motionbench`` Hydra entry point
    with overrides that restrict the sweep to exactly one (dataset, clf,
    method) triple.  ``CUDA_VISIBLE_DEVICES`` is set in the child env so
    that the child sees the assigned GPU as ``cuda:0``.
    """
    cell.gpu = gpu
    log_path = log_dir / f"{cell.classifier}__{cell.method}.log"
    cell.log_path = log_path

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["HYDRA_FULL_ERROR"] = "1"
    # Cap intra-op threads — by default torch grabs all 128 logical cores per
    # process, so N concurrent cells × 128 threads massively over-subscribes
    # the CPU and starves on-manifold imputers.  4 threads per process keeps
    # blas/conv responsive without contention at 16 concurrent slots.
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    env.setdefault("NUMEXPR_NUM_THREADS", "4")
    env.setdefault("TORCH_NUM_THREADS", "4")

    # Hydra CLI overrides on the keys ``datasets``/``classifiers``/``methods``
    # collide with the config-group directories of the same name, so we use
    # a thin in-process helper that builds the cfg programmatically.
    cmd = [
        sys.executable, str(REPO / "scripts" / "_run_one_cell.py"),
        "--dataset", cell.dataset,
        "--classifier", cell.classifier,
        "--method", cell.method,
        "--device", "cuda:0",
        "--n-sequences", str(n_sequences),
        "--results-dir", str(results_dir),
    ]

    t0 = time.time()
    with log_path.open("wb") as fh:
        proc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                              cwd=REPO)
    cell.elapsed_s = time.time() - t0
    cell.rc = proc.returncode
    return cell


def _worker(slot_id: int, gpu: int, work_q: "queue.Queue[Cell | None]",
            done_q: "queue.Queue[Cell]", results_dir: Path,
            n_sequences: int, log_dir: Path) -> None:
    """Pull cells from ``work_q`` and execute them on ``gpu`` until None."""
    while True:
        cell = work_q.get()
        if cell is None:
            work_q.task_done()
            return
        try:
            _run_cell(cell, gpu, results_dir, n_sequences, log_dir)
        except Exception as exc:  # pragma: no cover — defensive
            cell.rc = -99
            cell.elapsed_s = 0.0
            print(f"  [slot{slot_id} gpu{gpu}] {cell.label} CRASH: {exc}",
                  flush=True)
        else:
            tag = "OK " if cell.rc == 0 else "FAIL"
            print(
                f"  [slot{slot_id} gpu{gpu}] {tag} {cell.label}  "
                f"({cell.elapsed_s:.1f}s, rc={cell.rc})",
                flush=True,
            )
        done_q.put(cell)
        work_q.task_done()


def _train_classifiers(gpus: list[int]) -> None:
    """Dispatch synthetic-classifier training across the first 3 GPUs."""
    train_gpus = gpus[: min(3, len(gpus))]
    cmd = [
        sys.executable, str(REPO / "scripts" / "train_synthetic_clf.py"),
        "--datasets", DATASET,
        "--classifiers", *CLASSIFIERS,
        "--gpus", *map(str, train_gpus),
    ]
    print(f"\n[STAGE 1] Training classifiers on GPUs {train_gpus}: "
          f"{shlex.join(cmd)}\n", flush=True)
    rc = subprocess.call(cmd, cwd=REPO)
    if rc != 0:
        raise SystemExit(f"Classifier training failed (rc={rc}).")


def _verify_imputers() -> None:
    """Verify that the GaussianMotionDataset imputer registry resolves."""
    from motionbench.imputers.carepd_imputer import (  # noqa: PLC0415
        _CARE_PD_ROOT, _FLOW_REGISTRY, _VAEAC_REGISTRY,
    )
    cls_key = "GaussianMotionDataset"
    missing: list[str] = []
    for name, registry in [("VAEAC", _VAEAC_REGISTRY), ("Flow", _FLOW_REGISTRY)]:
        if cls_key not in registry:
            missing.append(f"{name}: registry has no entry for {cls_key}")
            continue
        ckpt_rel, cfg_rel = registry[cls_key]
        ckpt_dir = _CARE_PD_ROOT / ckpt_rel
        cfg_path = _CARE_PD_ROOT / cfg_rel
        if not ckpt_dir.exists():
            missing.append(f"{name}: ckpt dir missing → {ckpt_dir}")
        if not cfg_path.exists():
            missing.append(f"{name}: cfg file missing → {cfg_path}")
    if missing:
        msg = "\n".join("    - " + m for m in missing)
        raise SystemExit(
            "Imputer verification failed for "
            f"{cls_key}:\n{msg}\n"
            "  Train them via CARE-PD or update the registry first."
        )
    print("[STAGE 2] Imputer registry OK — re-using GaussianMotionDataset "
          "VAEAC + Flow checkpoints.", flush=True)


def _run_sweep(gpus: list[int], jobs_per_gpu: int, methods: list[str],
               classifiers: list[str], results_dir: Path,
               n_sequences: int, log_dir: Path,
               force: bool) -> list[Cell]:
    """Dispatch one cell per subprocess across ``gpus × jobs_per_gpu`` slots."""
    log_dir.mkdir(parents=True, exist_ok=True)

    cells_all = [Cell(DATASET, c, m) for c in classifiers for m in methods]
    if force:
        pending = cells_all
        cached = []
    else:
        pending = [c for c in cells_all if not _result_exists(c, results_dir)]
        cached = [c for c in cells_all if _result_exists(c, results_dir)]

    print(f"\n[STAGE 3] Sweep: {len(cells_all)} cells "
          f"({len(cached)} cached, {len(pending)} to run).", flush=True)
    if not pending:
        return cached

    work_q: queue.Queue[Cell | None] = queue.Queue()
    done_q: queue.Queue[Cell] = queue.Queue()
    for c in pending:
        work_q.put(c)

    n_slots = max(1, len(gpus) * jobs_per_gpu)
    print(f"  Workers: {n_slots} slots ({len(gpus)} GPUs × {jobs_per_gpu} "
          f"jobs/GPU)", flush=True)
    print(f"  Logs   : {log_dir}", flush=True)

    threads: list[threading.Thread] = []
    for slot_id in range(n_slots):
        gpu = gpus[slot_id % len(gpus)]
        work_q.put(None)  # one sentinel per worker
        t = threading.Thread(
            target=_worker,
            args=(slot_id, gpu, work_q, done_q, results_dir, n_sequences,
                  log_dir),
            daemon=True,
            name=f"slot{slot_id}-gpu{gpu}",
        )
        t.start()
        threads.append(t)

    work_q.join()
    for t in threads:
        t.join()

    completed: list[Cell] = []
    while not done_q.empty():
        completed.append(done_q.get())
    return cached + completed


def _print_summary(cells: list[Cell], total_s: float) -> None:
    print("\n" + "=" * 78)
    print(f"{'CLASSIFIER':<26} {'METHOD':<22} {'GPU':>4} {'TIME':>8} {'STATUS':>8}")
    print("=" * 78)
    n_ok = n_fail = 0
    for c in sorted(cells, key=lambda x: (x.classifier, x.method)):
        status = "OK" if c.rc == 0 else (f"rc={c.rc}" if c.rc >= 0 else "?")
        if c.rc == 0:
            n_ok += 1
        else:
            n_fail += 1
        gpu = "-" if c.gpu < 0 else str(c.gpu)
        print(f"{c.classifier:<26} {c.method:<22} {gpu:>4} "
              f"{c.elapsed_s:>7.1f}s {status:>8}")
    print("=" * 78)
    print(f"  {n_ok} OK / {n_fail} failed / {len(cells)} total cells")
    print(f"  Total wall time: {total_s:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    detected = _detect_gpus()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gpus", nargs="+", type=int, default=detected,
                        help=f"CUDA device indices to use (default: detected "
                             f"{detected}).")
    parser.add_argument("--jobs-per-gpu", type=int, default=2,
                        help="Concurrent cells per GPU (default: 2).")
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        choices=METHODS, metavar="METHOD",
                        help="Subset of methods to run (default: all 16).")
    parser.add_argument("--classifiers", nargs="+", default=CLASSIFIERS,
                        choices=CLASSIFIERS, metavar="CLF",
                        help="Subset of classifiers (default: all 3).")
    parser.add_argument("--n-sequences", type=int, default=200,
                        help="Sequences per cell (default: 200 — matches "
                             "configs/data/xor_label_gaussian.yaml N).")
    parser.add_argument("--results-dir", type=Path,
                        default=REPO / "results" / "synthetic",
                        help="Where to write per-cell result.json files.")
    parser.add_argument("--log-dir", type=Path,
                        default=REPO / "outputs" / "xor_sweep_logs",
                        help="Per-cell stdout/stderr log directory.")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip the classifier-training stage.")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip the imputer-registry verification stage.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all cells, even if result.json exists.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.gpus:
        raise SystemExit("No CUDA devices detected — aborting.  "
                         "Install drivers or pass --gpus 0 [1 ...].")

    print(f"motionbench-xai — XOR sweep multi-GPU runner")
    print(f"  Repo            : {REPO}")
    print(f"  Dataset         : {DATASET}")
    print(f"  GPUs            : {args.gpus}")
    print(f"  Jobs / GPU      : {args.jobs_per_gpu}")
    print(f"  Concurrent slots: {len(args.gpus) * args.jobs_per_gpu}")
    print(f"  Methods         : {len(args.methods)}")
    print(f"  Classifiers     : {len(args.classifiers)}")
    print(f"  n_sequences     : {args.n_sequences}")
    print(f"  results_dir     : {args.results_dir}")

    t_total = time.time()

    if not args.skip_train:
        _train_classifiers(args.gpus)

    if not args.skip_verify:
        _verify_imputers()

    cells = _run_sweep(
        gpus=args.gpus,
        jobs_per_gpu=args.jobs_per_gpu,
        methods=args.methods,
        classifiers=args.classifiers,
        results_dir=args.results_dir,
        n_sequences=args.n_sequences,
        log_dir=args.log_dir,
        force=args.force,
    )

    _print_summary(cells, time.time() - t_total)


if __name__ == "__main__":
    main()
