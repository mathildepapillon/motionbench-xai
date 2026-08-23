# MotionBench-XAI

**A benchmark for oracle-grounded Shapley attribution on spatiotemporal
data.**

MotionBench-XAI evaluates feature-attribution methods where their answers can
actually be checked: on generative models whose conditional distributions —
and therefore whose Shapley values — are known in closed form.  The benchmark
is the product of four axes:

```
        datasets  ×  player sets  ×  SHAP methods  ×  metrics
   (synthetic with   (what counts    (which imputer    (EC1/EC3 vs the
    exact oracles,    as one          defines v(S))     exact target,
    + real data)      "feature")                        faithfulness, …)
```

- **Datasets.** A parametric family of synthetic motion datasets with exact
  oracles — Gaussian Kronecker fields `x ~ N(0, Σ_J ⊗ I_F ⊗ Σ_T)` spanning
  spatial coupling (equicorrelated / skeleton-graph) × temporal structure
  (AR(1) / periodic), plus heavy-tailed Burr XII copula variants — and three
  real tracks (CARE-PD skeletal gait, PTB-XL 12-lead ECG, ESC-50 audio).
- **Player sets.** The unit of explanation is configurable and every method
  runs under the same coalition structure: temporal windows, joints,
  joint×window cells, anatomical groups, gait phases.
- **SHAP methods.** KernelSHAP with pluggable completion models — from
  off-manifold reference fills (zero/mean/marginal) through classical
  conditional estimators (empirical/kNN/copula) to learned on-manifold
  imputers (VAEAC, flow matching) — plus WindowSHAP, TimeSHAP, and gradient
  baselines.
- **Ground truth, two ways.** Every method is graded against the estimand it
  targets: the **marginal (interventional) game** `v(S) = f(x_S ⊔ E[x])` or
  the **conditional (on-manifold) game** `v(S) = f(x_S ⊔ E[x_hid | x_obs])`.
  Both targets are computed **deterministically** (exact conditional-mean
  operators; cusp-split quadrature for the copula family), so a perfect
  attribution scores exactly zero — no Monte-Carlo grading noise.

---

## Install

```bash
git clone https://github.com/mathildepapillon/motionbench-xai
cd motionbench-xai
python3 -m pip install -e .          # Python >= 3.10
# optional dev tools (pytest, ruff, mypy):
python3 -m pip install -e ".[dev]"
```

One optional dependency is not on PyPI: the WindowSHAP baselines need
`pip install git+https://github.com/vsubbian/WindowSHAP`.  Everything else —
including the quickstart below — works without it.

## Quickstart — one complete cell, CPU, minutes

The fastest tour is the self-contained example (generate data → train a
small MLP → attribute with two methods → grade against the exact oracle):

```bash
python3 examples/minimal_evaluation.py     # ~1 minute on a laptop CPU
```

The same cell through the benchmark pipeline proper:

```bash
# 1. Generate gauss_k4 and train its MLP classifier (~2 min on CPU).
python3 scripts/train_synthetic_clf.py \
    --datasets gaussian_k4 --classifiers synthetic_mlp --force-cpu

# 2. Attribute with KS-Zero and KS-Oracle under spatial players and grade
#    both against the exact deterministic oracles (EC1/EC3).
python3 -m motionbench.cli.run experiments=quickstart

# 3. Inspect results.
cat results/player_eval/spatial/gaussian_k4/synthetic_mlp/*/result.json
```

Expected: `kernelshap_zero` scores EC1 ≈ 0 on its marginal game (it *is* the
marginal target by construction — a built-in sanity check), while stochastic
conditional methods score their genuine estimation error against the exact
conditional target.

## Evaluating your own attribution method

Implement a `BaseImputer` (or a full `BaseAttributor`), drop a config in
`configs/methods/`, and run the same pipeline — see
[CONTRIBUTING.md](CONTRIBUTING.md) for the extension table and
`examples/minimal_evaluation.py` for the direct API. The key interfaces:

```python
players = SpatialJoints(J=5, F=3, T=16)  # any PlayerSet: z -> (J,F,T) mask
completions = imputer.impute(x, mask, n_samples)  # any BaseImputer: q(x_hid | x_obs)
Z, w = sampled_coalition_set(players.n_players, budget=1024)  # shared design
det = DeterministicConditionalOracle.from_oracle(dataset.oracle, players, Z)
phi_star = phi_from_values(Z, w, prob_fn(det.fill_all(x)))  # exact target
```

Shape conventions everywhere: samples are `(J, F, T)` (joints × features ×
time), boolean masks are `(J, F, T)` with **`True` = observed**, per-player
attributions are `(M,)`.

---

## Pipelines

| pipeline | entry | what it does |
|---|---|---|
| `synthetic` | `experiments=full_synthetic_sweep` | the paper's synthetic tables: all methods × datasets × classifiers, MC oracle grading + fidelity/stability metrics |
| `player_eval` | `experiments=player_set_eval` | any player set × any imputer on a **shared fixed coalition design**, graded against the **deterministic** oracles (exact enumeration for M ≤ 12, importance-corrected sampling above) |
| `real` | `experiments=care_pd_sweep` | CARE-PD real-data evaluation (AOPC/faithfulness; no synthetic oracle) |

All are Hydra-driven: `python3 -m motionbench.cli.run experiments=<name>
key=value …` (run from the repo root; configs resolve relative to the CWD).

## Repository structure

```
motionbench/
├── data/           # Datasets (synthetic generators + CARE-PD + PTB-XL loaders)
├── oracles/        # Ground truth: exact conditional samplers + deterministic
│                   #   conditional-mean oracles (oracles/deterministic.py)
├── players/        # Player sets (temporal, spatial, cells, anatomical, phases)
├── imputers/       # Completion models (zero/mean/marginal/empirical/VAEAC/flow)
├── attribution/    # Methods (KernelSHAP + sampled coalition designs, WindowSHAP,
│                   #   TimeSHAP, IG/DeepLIFT/LRP/GradCAM, …)
├── classifiers/    # Synthetic MLP/CNN/Transformer + ported CARE-PD + PTB-XL
├── metrics/        # EC1–EC3, faithfulness, AOPC, stability, sanity
├── pipelines/      # Hydra pipelines (synthetic_eval, player_eval, real_eval)
└── cli/            # `motionbench` / `python -m motionbench.cli.run`
configs/            # Hydra config tree (data, methods, players, classifiers,
                    #   experiments)
scripts/            # Training, reproduction, ablations
tests/              # Pytest suite (incl. oracle gate G1–G6)
docs/               # Architecture notes
```

Key documents:

- [`RESOLUTIONS.md`](RESOLUTIONS.md) — the executed conventions of this
  release (value-function semantics, labels, kernel bandwidths, seeds, …),
  including every point where the paper's prose and the released code
  differ.  Adapted from the independent validation study's log.
- [`checkpoints/README.md`](checkpoints/README.md) — which checkpoint files
  the pipelines expect, with SHA-256 digests of the reference training runs.
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — per-stage commands, budgets,
  data acquisition for the real tracks.

## Reproducing the paper

Two tiers — see [`REPRODUCING_PAPER.md`](REPRODUCING_PAPER.md) for the
table-by-table map:

- **From shipped results (no GPU):** every paper table regenerates from the
  canonical result files in [`results/canonical/`](results/canonical/).
- **From scratch:**

```bash
./scripts/reproduce_synthetic.sh   # synthetic half: self-contained
./scripts/reproduce_real.sh        # CARE-PD (needs the CARE-PD checkout + data)
./scripts/reproduce_ptbxl.sh       # PTB-XL (needs the PhysioNet records)
./scripts/reproduce_esc50.sh       # ESC-50 (needs the ESC-50 download; CC BY-NC)
```

The synthetic half needs nothing outside this repo (data is generated on the
fly; classifiers/imputers retrain from fixed seeds).  The real tracks need
third-party data we cannot redistribute — see
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for acquisition steps.  Checkpoint
download locations and digests are in
[`checkpoints/README.md`](checkpoints/README.md).

## Development

```bash
pytest tests/ -m "not slow and not gpu and not manual"   # fast suite
ruff check . && ruff format --check .                    # lint / format
mypy motionbench/                                        # types
```

GitHub Actions runs the fast tests, ruff, and mypy on every push
(`.github/workflows/ci.yml`).

## Links

- **Paper:** *MotionBench-XAI: A Benchmark for Manifold-Aware Shapley
  Attribution on Spatiotemporal Data* (NeurIPS 2026 Evaluations & Datasets
  Track submission). <!-- MAINTAINER TODO: add arXiv/OpenReview link -->
- **Independent validation study:** a from-scratch replication of this
  benchmark whose findings are folded into `RESOLUTIONS.md`, the
  deterministic oracles, and the oracle gate tests.
  <!-- MAINTAINER TODO: add link when the study is public -->
- **Checkpoints:** reference classifier / imputer checkpoints are published
  at the [`checkpoints-v2` release](https://github.com/mathildepapillon/motionbench-xai/releases/tag/checkpoints-v2)
  (`bash scripts/download_checkpoints.sh`; manifest and digests in
  [`checkpoints/README.md`](checkpoints/README.md)).
- **Upstream data:** [CARE-PD](https://github.com/TaatiTeam/CARE-PD),
  [PTB-XL](https://physionet.org/content/ptb-xl/).

## Citation

See [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" button).

```bibtex
@inproceedings{motionbench2026,
  title     = {MotionBench-XAI: A Benchmark for Manifold-Aware Shapley
               Attribution on Spatiotemporal Data},
  booktitle = {Advances in Neural Information Processing Systems,
               Evaluations and Datasets Track},
  year      = {2026},
}
```

This benchmark consumes data and classifier checkpoints from CARE-PD and
PTB-XL; please cite both upstream sources if you use the corresponding
tracks.  License: [MIT](LICENSE).
