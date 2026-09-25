# Reproducing the paper, table by table

Two tiers of reproduction:

- **Tier 1 — verify against shipped results (minutes, no GPU).**
  The canonical result files in `results/canonical/` are the exact outputs
  behind every table; the mapping below says which file feeds which table.
- **Tier 2 — rerun the experiments (GPU).**  Every canonical file regenerates
  from the entry points below.  Checkpoints: `scripts/download_checkpoints.sh`
  (or retrain, see `REPRODUCIBILITY.md` §Training).  Real datasets: CARE-PD,
  PTB-XL, and ESC-50 are public; the cache builders in `scripts/` prepare
  them.

Fixed protocol shared by every run: `v(S) = f(mean completion)`; shared
coalition designs per (dataset, player set) with seed 7919 (exact enumeration
for M ≤ 12, kernel-mass-ordered sampling above); N = 200 sequences per
dataset (per fold on real data); synthetic grading against exact deterministic
oracle targets, game-matched.

| Paper table | Canonical file (Tier 1) | Rerun entry point (Tier 2) |
|---|---|---|
| Table 3 — synthetic, temporal players | `results/canonical/synthetic_temporal.json` (`tables.game_matched`, + `cells` for KS-Gauss) | `scripts/reproduce_synthetic.sh` (attribution) + deterministic regrade |
| Table 4a — synthetic, spatial joints | `results/canonical/synthetic_players.json` (`tables.joint`) | `experiments=player_set_eval` pipeline (`motionbench/pipelines/player_eval.py`), players=spatial; hi-budget grading targets via `scripts/regrade_hi_budget.py` (M=17: B=8192, seed 900001) |
| Table 4b — synthetic, joint×window cells | `results/canonical/synthetic_players.json` (`tables.cell`) | same pipeline, players=cells — note the budget override below; hi-budget targets via `scripts/regrade_hi_budget.py` (B=8192 seed 900001; B=32768 seed 910001 at M=68) |
| Table 5a — real, faithfulness | `results/canonical/real_tracks.json` + `real_players.json` (CARE-PD joints) | `scripts/reproduce_real.sh`, `scripts/reproduce_ptbxl.sh`, `scripts/reproduce_esc50.sh`, + real player-set entry points |
| Table 5b — real, PlayerAOPC | same as Table 5a | same as Table 5a |
| Appendix — real joint×window cells | `real_players.json` (CARE-PD M=68, ESC-50 M=16) + `real_ptbxl_cells.json` (PTB-XL M=48) | real player-set entry points |
| Appendix — EC1 tables | same files as Tables 3–4 (`ec1` fields of every cell) | same as Tables 3–4 |
| Appendix — cross-graded (all-conditional) tables | `synthetic_temporal.json` (`cond` block per method) + `synthetic_players.json` (`cross` block per cell) | stored attributions regraded against the conditional target (both targets ship in the run outputs) |
| Appendix — grading-noise floors | `synthetic_players.json` (`meta.grading_floors`, incl. the M=68 cross-budget agreement records) | independent replicate target at the same budget, disjoint coalition seed (`scripts/regrade_hi_budget.py` with a replicate `--seed`) |
| Appendix — classifier gate | `results/canonical/classifier_gate.json` | checkpoint accuracy audit of the shipped classifiers (values recompute from the training scripts + `real_tracks.json` `carepd_gate`) |
| Appendix — imputer capability gate | `results/canonical/imputer_gate.json` | `scripts/run_imputer_gate.py` (`--mode gate|controls` × track × imputer) |

**Method coalition budget (player-set reruns):** the canonical player-set
runs used `coalition_budget=2048` for the M=68 datasets (`gait_periodic`,
`skeleton_structured`, `skeleton_gait_combined` cells) and the default
`coalition_budget=1024` everywhere else.  When rerunning Tables 4a–4b, pass the
override for the M=68 cells, e.g.
`python -m motionbench.cli.run experiments=player_set_eval player_sets='[cells]' coalition_budget=2048`
(grading targets are unaffected — they are rebuilt at the hi budgets by
`scripts/regrade_hi_budget.py`).

Verification is built in at every seam:

- the sampled-coalition estimator, deterministic oracle fills, and player
  partitions in this package are pinned by the test suite (`tests/`,
  including the oracle gate G1–G6);
- the real player-set entry points reproduce the stored temporal/per-lead
  tracks bit-exactly (records in the canonical files' `gate` blocks and
  `docs/CONVENTIONS.md` §12);
- `results/canonical/README.md` documents each file's provenance.

If a rerun of yours disagrees with a canonical file beyond float tolerance,
please open an issue with the config and environment — that is exactly the
kind of report this benchmark exists to catch.
