"""scripts/run_temporal_shap_multigpu.py — N=200 sweep for the temporal-SHAP baselines.

Re-runs the three pip-package temporal-SHAP baselines
(``windowshap_stationary``, ``windowshap_dynamic``, ``timeshap``) at the
paper's full ``N=200`` sequence budget across all eight synthetic datasets
and all three classifier architectures.  Earlier results in the
``results/synthetic`` tree for these methods were at ``N=15``--``50`` and
were not directly comparable with the ``N=200`` KernelSHAP rows in
Table~\\ref{tab:synth_ec1}.

Mirrors the dispatch pattern of ``scripts/restore_contaminated_n50.py``:
shells out to ``scripts/_run_one_cell.py`` once per cell with
``CUDA_VISIBLE_DEVICES`` and CPU-thread caps set in the child env.

Cells already on disk with the requested ``--n-sequences`` are skipped;
older / lower-N results can be wiped first with ``--wipe-stale`` so the
re-run starts clean.

Usage::

    python scripts/run_temporal_shap_multigpu.py \\
        --gpus 0 1 2 3 4 5 6 7 \\
        --jobs-per-gpu 4 \\
        --omp-threads 2 \\
        --metrics-mode gt_only \\
        --n-sequences 200 \\
        --wipe-stale
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DATASETS = [
    "gaussian_k4",
    "gaussian_k8",
    "skeleton_structured",
    "gait_periodic",
    "burr_m5",
    "burr_m10",
    "low_rank_manifold",
    "skeleton_gait_combined",
]
CLASSIFIERS = ["synthetic_mlp", "synthetic_cnn", "synthetic_transformer"]
METHODS = ["windowshap_stationary", "windowshap_dynamic", "timeshap"]


@dataclass
class Cell:
    dataset: str
    classifier: str
    method: str
    gpu: int = -1
    elapsed_s: float = 0.0
    rc: int = -1

    @property
    def label(self) -> str:
        return f"{self.dataset}/{self.classifier}/{self.method}"


def _wipe_stale(results_dir: Path, cells: list[Cell],
                target_n: int) -> int:
    """Delete result.json files whose ``n_sequences`` is below the target.

    A "stale" file is one that already exists but was generated at a
    smaller N; leaving it on disk would cause the pipeline cache check
    to skip the re-run.  Files at the target N (or higher) are kept.
    """
    n = 0
    for c in cells:
        rp = results_dir / c.dataset / c.classifier / c.method / "result.json"
        if not rp.exists():
            continue
        try:
            existing = json.loads(rp.read_text()).get("n_sequences", 0)
        except Exception:
            existing = 0
        if existing < target_n:
            rp.unlink()
            ep = rp.parent / "error.json"
            if ep.exists():
                ep.unlink()
            ap = rp.parent / "attributions.npz"
            if ap.exists():
                ap.unlink()
            n += 1
    return n


def _filter_pending(results_dir: Path, cells: list[Cell],
                    target_n: int) -> list[Cell]:
    """Skip cells that already have a result.json at >= target_n sequences."""
    pending = []
    for c in cells:
        rp = results_dir / c.dataset / c.classifier / c.method / "result.json"
        if rp.exists():
            try:
                if json.loads(rp.read_text()).get("n_sequences", 0) >= target_n:
                    continue
            except Exception:
                pass
        pending.append(c)
    return pending


def _run_cell(cell: Cell, gpu: int, results_dir: Path, n_sequences: int,
              log_dir: Path, metrics_mode: str = "gt_only",
              omp_threads: int = 2) -> Cell:
    cell.gpu = gpu
    log_path = log_dir / f"{cell.dataset}__{cell.classifier}__{cell.method}.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["HYDRA_FULL_ERROR"] = "1"
    env["OMP_NUM_THREADS"] = str(omp_threads)
    env["MKL_NUM_THREADS"] = str(omp_threads)
    env["OPENBLAS_NUM_THREADS"] = str(omp_threads)
    env["NUMEXPR_NUM_THREADS"] = str(omp_threads)
    env["TORCH_NUM_THREADS"] = str(omp_threads)

    cmd = [
        sys.executable, str(REPO / "scripts" / "_run_one_cell.py"),
        "--dataset", cell.dataset,
        "--classifier", cell.classifier,
        "--method", cell.method,
        "--device", "cuda:0",
        "--n-sequences", str(n_sequences),
        "--results-dir", str(results_dir),
        "--metrics-mode", metrics_mode,
    ]

    t0 = time.time()
    with log_path.open("w") as logf:
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=logf,
                              stderr=subprocess.STDOUT)
    cell.elapsed_s = time.time() - t0
    cell.rc = proc.returncode
    return cell


def _worker(slot_id: int, gpu: int, work_q: queue.Queue,
            done_q: queue.Queue, results_dir: Path, n_sequences: int,
            log_dir: Path, metrics_mode: str = "gt_only",
            omp_threads: int = 2) -> None:
    while True:
        cell = work_q.get()
        if cell is None:
            work_q.task_done()
            return
        try:
            _run_cell(cell, gpu, results_dir, n_sequences, log_dir,
                      metrics_mode=metrics_mode, omp_threads=omp_threads)
            tag = "OK " if cell.rc == 0 else f"rc{cell.rc}"
        except Exception as exc:  # pragma: no cover
            cell.rc = -2
            tag = f"EXC {type(exc).__name__}"
        finally:
            print(f"  [slot{slot_id} gpu{gpu}] {tag} {cell.label}  "
                  f"({cell.elapsed_s:.1f}s)", flush=True)
        done_q.put(cell)
        work_q.task_done()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", nargs="+", type=int,
                   default=[0, 1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--jobs-per-gpu", type=int, default=4)
    p.add_argument("--omp-threads", type=int, default=2)
    p.add_argument("--metrics-mode",
                   choices=["full", "gt_only", "gt_plus_faith"],
                   default="gt_only",
                   help="Default 'gt_only' since the temporal-SHAP baselines "
                        "are reported only in EC1/EC3 ground-truth tables.")
    p.add_argument("--n-sequences", type=int, default=200)
    p.add_argument("--results-dir", type=Path,
                   default=REPO / "results" / "synthetic")
    p.add_argument("--log-dir", type=Path,
                   default=REPO / "outputs" / "temporal_shap_n200_logs")
    p.add_argument("--wipe-stale", action="store_true",
                   help="Delete pre-existing result.json files whose "
                        "n_sequences is below --n-sequences before dispatch.")
    p.add_argument("--methods", nargs="+", default=METHODS,
                   choices=METHODS,
                   help="Subset of temporal-SHAP methods to run.")
    p.add_argument("--datasets", nargs="+", default=DATASETS,
                   choices=DATASETS)
    p.add_argument("--classifiers", nargs="+", default=CLASSIFIERS,
                   choices=CLASSIFIERS)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    all_cells = [Cell(d, c, m) for d in args.datasets
                 for c in args.classifiers for m in args.methods]
    print(f"Total target cells: {len(all_cells)}")
    if args.dry_run:
        for c in all_cells:
            print(f"  {c.label}")
        return

    if args.wipe_stale:
        deleted = _wipe_stale(args.results_dir, all_cells, args.n_sequences)
        print(f"Wiped {deleted} result.json files with n_sequences < "
              f"{args.n_sequences}")

    cells = _filter_pending(args.results_dir, all_cells, args.n_sequences)
    print(f"Pending (no result.json at N>={args.n_sequences} yet): "
          f"{len(cells)}/{len(all_cells)}\n")

    if not cells:
        print(f"Nothing to do; every cell has result.json with N>={args.n_sequences}.")
        return

    work_q: queue.Queue = queue.Queue()
    done_q: queue.Queue = queue.Queue()
    for c in cells:
        work_q.put(c)

    n_slots = max(1, len(args.gpus) * args.jobs_per_gpu)
    print(f"Dispatching {len(cells)} cells across {n_slots} slots "
          f"({len(args.gpus)} GPUs x {args.jobs_per_gpu} jobs/GPU, "
          f"{args.omp_threads} threads/proc, metrics={args.metrics_mode})")
    print(f"Per-cell logs: {args.log_dir}\n")

    threads: list[threading.Thread] = []
    for slot_id in range(n_slots):
        gpu = args.gpus[slot_id % len(args.gpus)]
        work_q.put(None)
        t = threading.Thread(
            target=_worker,
            args=(slot_id, gpu, work_q, done_q, args.results_dir,
                  args.n_sequences, args.log_dir, args.metrics_mode,
                  args.omp_threads),
            daemon=True, name=f"slot{slot_id}-gpu{gpu}",
        )
        t.start()
        threads.append(t)

    t0 = time.time()
    work_q.join()
    for t in threads:
        t.join()
    total_s = time.time() - t0

    completed: list[Cell] = []
    while not done_q.empty():
        completed.append(done_q.get())

    n_ok = sum(1 for c in completed if c.rc == 0)
    n_fail = len(completed) - n_ok
    print("\n" + "=" * 72)
    print(f"DONE  {n_ok} OK  /  {n_fail} failed  /  {len(completed)} total")
    print(f"Total wall time: {total_s:.1f}s")
    if n_fail:
        print("\nFailed cells:")
        for c in completed:
            if c.rc != 0:
                print(f"  rc={c.rc}  {c.label}  (gpu{c.gpu}, {c.elapsed_s:.1f}s)")
        sys.exit(1)


if __name__ == "__main__":
    main()
