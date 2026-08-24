# Canonical benchmark results

These JSON files are the authoritative outputs of the benchmark runs behind
every table in the MotionBench-XAI paper.  They let you regenerate and verify
the paper's tables **without rerunning any experiment**; the pipelines in this
repository regenerate the files themselves (see `REPRODUCING_PAPER.md` at the
repo root for the table-by-table map, and `REPRODUCIBILITY.md` for full
end-to-end runs).

| File | Contents | Produced by |
|---|---|---|
| `synthetic_temporal.json` | Synthetic suite, temporal players ($K$ windows): game-matched and all-conditional EC1/EC3 per method × dataset × classifier, exact deterministic grading | temporal attribution runs regraded against exact f-of-mean oracle targets |
| `synthetic_players.json` | Synthetic suite, spatial joints and joint×window cells: EC1/EC3 per method × dataset × classifier (`tables.{joint,cell}.{per_cell,clf_avg}`), hi-budget regraded targets (B=8192; B=32768 at M=68); each per-cell record also carries a `cross` block (same attributions regraded against the conditional target — the all-conditional convention of prior benchmarks) | `pipelines/player_eval.py` protocol (shared coalition designs, seed 7919) |
| `real_tracks.json` | Real data, temporal windows + PTB-XL per-lead + ESC-50 freq-band tracks: faithfulness and PlayerAOPC, pooled over 3 folds with per-fold values | real-data attribution scripts (`scripts/reproduce_real.sh` CARE-PD temporal, `scripts/reproduce_ptbxl.sh` PTB-XL per-lead, `scripts/reproduce_esc50.sh` ESC-50 temporal + freq-band) |
| `real_players.json` | CARE-PD spatial joints (M=17) and joint×window cells (M=68), ESC-50 band×window cells (M=16): faithfulness and PlayerAOPC, pooled + per-fold | real player-set entry points (gate-validated against `real_tracks.json` to float32) |
| `real_ptbxl_cells.json` | PTB-XL lead×window cells (M=48), all 5 KernelSHAP methods, 3 folds | PTB-XL cells entry point (gate: reproduces the stored per-lead track to ≤7e-9 before running) |
| `classifier_gate.json` | Classifier gate behind every aggregated table (G1: held-out accuracy ≥ chance+0.05; G2: train–test gap ≤ 0.30): synthetic train/val/test accuracies per dataset × architecture with pass/fail verdicts; real per-fold accuracies (CARE-PD incl. pooled accs and POTR's G1 failure, PTB-XL, ESC-50 per-fold AST fine-tunes) | checkpoint accuracy audit of the shipped classifiers + `real_tracks.json` `carepd_gate` + ESC-50 retrain metadata |
| `imputer_gate.json` | Imputer capability gate (hide-one-recover): per track (CARE-PD, PTB-XL, ESC-50) × imputer (VAEAC, flow) Pearson recovery per mask family (spatial / temporal / cell), plus donor and unconditional controls; protocol in `meta` | `scripts/run_imputer_gate.py` (`--mode gate|controls`; ptbxl/flow cell reproduced bit-exactly against the archived run before merging) |

Conventions shared by all files: value function `v(S) = f(mean completion)`;
each method graded against the exact ground truth of its own fill family
(game-matched) on the synthetic suite; faithfulness = Pearson correlation over
all sampled coalitions including boundary rows; PlayerAOPC = mean drop along
the explicit deletion path; N=200 sequences per dataset (per fold on real
data).

Aggregation note: paper tables average classifiers that pass a uniform gate
(held-out accuracy >= chance+0.05; train-test gap <= 0.30).  The per-classifier
cells shipped here are complete and ungated; the gate is applied at table
generation and stated in the paper's protocol.

