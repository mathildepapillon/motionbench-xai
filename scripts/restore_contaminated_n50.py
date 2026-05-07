"""scripts/restore_contaminated_n50.py — emergency N=200 restoration.

Re-runs the 48 ``kernelshap_vaeac`` / ``kernelshap_flow`` cells that were
overwritten with N=50 results by ``run_synth_vaeac_flow.py``.  Targets all
nine synthetic datasets except ``xor_label_gaussian`` (which still has clean
N=200 results) and ``joint_subset_skeleton`` (which was never touched).

Pattern mirrors ``run_xor_sweep_multigpu.py``:
- Deletes the contaminated result.json files first (so the pipeline cache
  check does not skip them).
- Dispatches ``len(GPUS) * JOBS_PER_GPU`` workers; each worker pulls cells
  from a queue and shells out to ``scripts/_run_one_cell.py`` with
  ``CUDA_VISIBLE_DEVICES`` and CPU-thread caps set in the child env.
"""
from __future__ import annotations

import argparse
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
METHODS = ["kernelshap_vaeac", "kernelshap_flow"]


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


def _delete_contaminated_n50(results_dir: Path, cells: list[Cell],
                             cutoff_mtime: float | None = None) -> int:
    """Delete only result.json files that are still N=50-contaminated.

    A result is "contaminated" if it parses as JSON with n_sequences == 50.
    A cutoff_mtime, if given, additionally requires the file is older than
    that timestamp (defensive: do not touch anything written after the
    restoration began)."""
    import json as _json
    n = 0
    for c in cells:
        rp = results_dir / c.dataset / c.classifier / c.method / "result.json"
        if not rp.exists():
            continue
        if cutoff_mtime is not None and rp.stat().st_mtime > cutoff_mtime:
            continue
        try:
            d = _json.loads(rp.read_text())
        except Exception:
            d = None
        if d is not None and d.get("n_sequences") == 50:
            rp.unlink()
            ep = rp.parent / "error.json"
            if ep.exists():
                ep.unlink()
            n += 1
    return n


def _filter_pending(results_dir: Path, cells: list[Cell]) -> list[Cell]:
    """Skip cells whose result.json already exists (i.e., already restored)."""
    pending = []
    for c in cells:
        rp = results_dir / c.dataset / c.classifier / c.method / "result.json"
        if rp.exists():
            continue
        pending.append(c)
    return pending


def _run_cell(cell: Cell, gpu: int, results_dir: Path, n_sequences: int,
              log_dir: Path, metrics_mode: str = "full",
              omp_threads: int = 4) -> Cell:
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
            log_dir: Path, metrics_mode: str = "full",
            omp_threads: int = 4) -> None:
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
    p.add_argument("--jobs-per-gpu", type=int, default=2)
    p.add_argument("--omp-threads", type=int, default=4,
                   help="Per-process BLAS/OMP thread cap.  Default 4 with "
                        "16 slots = 64 cores.  For 32 slots use 2.")
    p.add_argument("--metrics-mode",
                   choices=["full", "gt_only", "gt_plus_faith"],
                   default="full",
                   help="Which metric suite to compute (forwarded to "
                        "_run_one_cell.py).  Default 'full' matches paper.")
    p.add_argument("--n-sequences", type=int, default=200)
    p.add_argument("--results-dir", type=Path,
                   default=REPO / "results" / "synthetic")
    p.add_argument("--log-dir", type=Path,
                   default=REPO / "outputs" / "restore_n50_logs")
    p.add_argument("--skip-deletion", action="store_true",
                   help="Do not auto-delete N=50 contaminated result.json "
                        "files (use when caller already cleaned up).")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    all_cells = [Cell(d, c, m) for d in DATASETS for c in CLASSIFIERS
                 for m in METHODS]
    print(f"Total target cells: {len(all_cells)}")
    if args.dry_run:
        for c in all_cells:
            print(f"  {c.label}")
        return

    if not args.skip_deletion:
        deleted = _delete_contaminated_n50(args.results_dir, all_cells)
        print(f"Deleted {deleted} contaminated (N=50) result.json files")

    cells = _filter_pending(args.results_dir, all_cells)
    print(f"Pending (no result.json yet): {len(cells)}/{len(all_cells)}\n")

    if not cells:
        print("Nothing to do; all cells already have result.json.")
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
