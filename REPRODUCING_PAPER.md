# Reproducing the paper, table by table

Two tiers of reproduction:

- **Tier 1 — regenerate the tables from shipped results (minutes, no GPU).**
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
| Table 2 — synthetic, temporal players | `results/canonical/synthetic_temporal.json` (`tables.game_matched`, + `cells` for KS-Gauss) | `scripts/reproduce_synthetic.sh` (attribution) + deterministic regrade |
| Table 3 — synthetic, spatial joints | `results/canonical/synthetic_players.json` (`tables.joint`) | `experiments=player_set_eval` pipeline (`motionbench/pipelines/player_eval.py`), players=joints |
| Table 4 — synthetic, joint×window cells | `results/canonical/synthetic_players.json` (`tables.cell`) | same pipeline, players=cells; hi-budget targets B=8192 / B=32768 (M=68) |
| Table 5 — real, faithfulness | `results/canonical/real_tracks.json` + `real_players.json` (CARE-PD joints) | `scripts/reproduce_real.sh`, `scripts/reproduce_ptbxl.sh`, + real player-set entry points |
| Table 6 — real, PlayerAOPC | same as Table 5 | same as Table 5 |
| Appendix — real joint×window cells | `real_players.json` (CARE-PD M=68, ESC-50 M=16) + `real_ptbxl_cells.json` (PTB-XL M=48) | real player-set entry points |
| Appendix — EC1 tables | same files as Tables 2–4 (`ec1` fields of every cell) | same as Tables 2–4 |
| Appendix — cross-graded (all-conditional) tables | `synthetic_temporal.json` (`cond` block per method) + `synthetic_players.json` (`cross` block per cell) | stored attributions regraded against the conditional target (both targets ship in the run outputs) |
| Appendix — grading-noise floors | `synthetic_players.json` (`meta.grading_floors`, once measured floors land) | independent replicate target at the same budget, disjoint coalition seed |
| Fig. 4 — ground-truth walkthrough curves | regenerates deterministically from the package (burr_m5 config, seed 44) | `examples/` / paper repo `make_burr_example.py` |

Verification is built in at every seam:

- the sampled-coalition estimator, deterministic oracle fills, and player
  partitions in this package are parity-gated **bit-exact** against the
  archived experiment runs (`tests/`, `RESOLUTIONS.md`);
- the real player-set entry points reproduce the stored temporal/per-lead
  tracks to float32 before producing new numbers (gate results in
  `RESOLUTIONS.md`);
- `results/canonical/README.md` documents each file's provenance.

If a rerun of yours disagrees with a canonical file beyond float tolerance,
please open an issue with the config and environment — that is exactly the
kind of report this benchmark exists to catch.
