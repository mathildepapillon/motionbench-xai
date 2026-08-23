# RESOLUTIONS.md — executed conventions of this release

This log records the implementation conventions this release executes,
including every point where the paper's prose leaves a detail open or states
a different convention than the released code runs.  The content is adapted
from the resolutions log of the **independent validation study** that
replicated the benchmark from scratch against the released code and the
manuscript (its findings are cited below as [study]); paths cited as [repo]
are files in this repository, [paper] is the manuscript.

Where the paper and the release disagree, the tables in the paper were
produced by the *release* conventions listed here.  The paper-arm
alternatives are available behind configuration flags where noted.

## 1. Value function (paper Eq. 5 vs executed code)

**Executed:** `v(S) = f(mean of R completions)` ("f-of-mean").  The
KernelSHAP masker draws `n_completion_samples` completions, averages them,
and evaluates the classifier once at the mean
(`motionbench/attribution/kernel_shap.py::_MotionBenchMasker`).
**Paper Eq. 5 states:** `v(S) = (1/R) Σ_r f(x_S ⊔ x̂⁽ʳ⁾)` ("mean-of-f").
The two coincide for deterministic imputers (Zero, Mean) and for R = 1
(VAEAC); they differ by a Jensen gap for Marginal / Empirical / Oracle
(R = 5/5/10).  [study] finding C3/A3.

**Resolution:** the executed semantics remain the default; both are now
configurable per method via `value_fn: f_of_mean | mean_of_f` in
`configs/methods/*.yaml` (see `kernel_shap.py`, "Estimator").  The
deterministic grading targets in `motionbench/oracles/deterministic.py` are
the exact `R → ∞` limits of the f-of-mean semantics.

## 2. Labels for the pillar synthetic datasets

**Executed:** tertiles of the **joint-0 grand mean**,
`score = x[:, 0, :, :].mean(axis=(1, 2))`, with cutoffs calibrated on each
generated batch.  No `label_fn` is passed for any pillar dataset in
`scripts/train_synthetic_clf.py::DATASET_CONFIGS`, so all fall through to
this dataset default.  [study] finding C1/A1.
**Paper states:** tertiles of the Olsen interaction score on K window means.
`OlsenInteraction` exists in `motionbench/data/synthetic/label_functions.py`
and can be passed as `label_fn`; the released tables use the proxy label.

Note two percentile conventions coexist in the release code and yield
slightly different cutoffs: `GaussianMotionDataset` uses
`np.percentile(score, [33, 67])` while `SkeletonStructuredDataset` (and the
label functions) use `linspace(0, 100, n_classes + 1)[1:-1]` (33.33/66.67).
Eval-time labels do not affect attribution (each sequence's explained target
is the classifier's own argmax), but retraining classifiers from scratch
reproduces the released checkpoints only under the per-class convention.

## 3. Olsen interaction: coefficients, seed, and window-group convention

The paper's tiled Olsen label (its Eq. for the interaction score) leaves the
coefficient distributions and the treatment of K not divisible by 4 open.
The convention (from the release's `OlsenInteraction` defaults, documented by
[study] A6):

- window means `w_k` over uniform `T // K` windows; `u_k = Φ(w_k / σ_k)`
  with `σ_k` = the batch standard deviation of `w_k` (clipped at 1e-8);
- one interaction term per **complete group of 4 windows** — `r` ranges over
  complete groups only, so K = 5 uses windows 0–3 and K < 4 is undefined;
- coefficients `c1, c2 ~ U(0.5, 2.0)`, `c3 ~ U(0.5, 1.0)` drawn from
  `numpy.random.default_rng(0)` (the fit seed is 0);
- class labels are tertiles of the score, cutoffs calibrated on the batch.

## 4. KS-Empirical kernel convention

`EmpiricalConditionalImputer` (`motionbench/imputers/empirical.py`)
implements Aas et al. (2021) Algorithm 2 with (per [study] A4):

- per-coordinate z-scoring of the donor pool;
- Ledoit–Wolf covariance on the observed sub-block per coalition;
- Gaussian kernel bandwidth **σ = 0.1 × median Mahalanobis distance**
  (the shapr `fixed_sigma = 0.1` convention applied to the median distance,
  not a fixed σ = 0.1);
- **η = 0.9** truncation of the donor weight mass
  (`configs/methods/kernelshap_empirical.yaml`, overriding the class default);
- a leave-one-out guard masking donors with `d² ≤ 1e-8 × median(d²)` —
  load-bearing because the donor pools are fit on the evaluation set itself,
  which contains the query sequence (see §7);
- weighted resampling of whole donor rows; observed coordinates overwritten
  bit-for-bit.

## 5. Executed VAEAC architecture

The paper appendix describes a "convolutional encoder–decoder"; the executed
VAEAC (`motionbench/imputers/vaeac.py` and the CARE-PD checkpoints) is a
**per-frame-token transformer** ([study] A5): three subnets (full encoder,
prior encoder, decoder), each a 2-layer TransformerEncoder trunk with
d_model = 256, nhead = 8, ff = 512, dropout 0.1, sinusoidal frame positional
encoding, per-frame latent dimension 64, and a Gaussian output head with one
learnable global log-σ.  Tokens: full = [x, mask], prior = [x·mask, mask],
decoder = [z, x·mask, mask] (one token per frame).  Loss: NLL on hidden
entries + KL(q‖p); Adam lr 1e-3, gradient clip 1.0, batch 16.  Inference
samples z from the prior at temperature 1 and restores observed entries
exactly.

## 6. Ground-truth grading budgets (heterogeneous in the release)

The released tables were graded with **two Monte-Carlo budgets** ([study]
B10): the Hydra sweep (`configs/experiments/full_synthetic_sweep.yaml`)
grades with `metric_oracle_n_mc = 10` and produced the off-manifold,
WindowSHAP, TimeSHAP and oracle rows; the KS-VAEAC / KS-Flow cells were
produced by `scripts/restore_contaminated_n50.py` → `scripts/_run_one_cell.py`,
which leave the default **n_mc = 50**.  Coalitions are enumerated exactly
wherever `2^K` fits the coalition budget.

The `player_eval` pipeline added in this release removes this axis entirely:
its grading targets are **deterministic** (exact conditional-mean fills, no
Monte Carlo), so a perfect attribution scores EC1 = EC3 = 0 with zero
grading noise (`motionbench/oracles/deterministic.py`).

## 7. Donor pools contain the query

Mean / MarginalDonor / Empirical imputers are fit on the N = 200 evaluation
dataset itself (`synthetic_eval.py::_build_and_fit_imputer(dataset=eval ds)`),
so donor pools contain the explained sequence.  [study] B7.  The
KS-Empirical LOO guard of §4 exists for exactly this reason.

## 8. Seed conventions

| What | Seed | Where |
|---|---|---|
| Evaluation sets (N = 200) | dataset config `seed` (gauss_k4 42, gauss_k8 43, burr_m5 44, burr_m10 45, skeleton 46, gait 47, skel+gait 51, xor 152) | `configs/data/*.yaml` |
| Classifier training data (N = 2000) | eval seed + 1000 | `scripts/train_synthetic_clf.py` |
| Classifier validation data (N = 500) | eval seed + 1001 | same |
| Classifier model init | `torch.manual_seed(42)`; retry with lr/5 and seed+1 if best val acc < 0.50 | same |
| Imputer training data (N = 1000 per family) | absolute seed 99 (disjoint from all eval/train seeds) | `scripts/build_synthetic_imputer_caches.py` |
| Attribution methods | `np.random.seed(42)` before every per-sequence SHAP call (`seed` in method configs) | `configs/methods/*.yaml` |
| Olsen coefficient fit | `default_rng(0)` | §3 |
| Ground-truth oracle draws | released pipeline passes fresh entropy per sample; for reproducibility prefer explicit `oracle_seed` (metrics) / derived per-sequence seeds (pipelines) | `metrics/ground_truth.py` |
| Fixed coalition designs (player_eval) | `coalition_seed` 7919, keyed as `[seed, M, budget]` | `motionbench/attribution/sampled_coalitions.py` |

Imputer training: one VAEAC + one Flow per distribution family, shared
across K/M variants (gaussian J5T16, burr J5T20, skeleton J17T16, gait
J17T16, skel+gait J17T16); N = 1000 draws at seed 99, 85/15 train/val,
80 epochs ([study] B6).

## 9. Synthetic classifier architectures (paper text vs executed configs)

The executed Transformer (`configs/classifiers/synthetic_transformer.yaml`
and the training script) is **d_model = 32, nhead = 4, num_layers = 2**; the
paper text and an earlier class docstring said 64/4/4.  [study] finding
C2/A2.  The executed configuration is authoritative for the released tables
and checkpoints; the class docstring has been corrected.  MLP: window-mean
features (`player_mode: temporal`), hidden 64.  CNN: 32→32→64, k = 5,
Conv→ReLU→BN, GAP, Linear.

## 10. Ground-truth Shapley solve

Constrained WLS over the coalition design: intercept column included;
empty/full coalitions added as hard constraint rows at weight 1e6; ridge
1e-8; Shapley kernel weight `(M−1)/(C(M,s)·s·(M−s))`
(`motionbench/utils/coalitions.py::solve_shapley_wls`).  Burr-family
conditionals are computed in latent z-space (`z = Φ⁻¹(F(x))`, Gaussian
conditional, `x = F⁻¹(Φ(z))`, observed entries restored bit-for-bit) with
probability clips at 1e-9; conditional covariances symmetrised and
ridge-regularised at 1e-8 ([study] B1/B3).

## 11. KS-Gauss (fitted Gaussian conditional; the study's `ks_shapr` row)

**Added in this release:** the shapr-style parametric Gaussian conditional
imputer of Aas et al. (2021) §3.2 as
`motionbench/imputers/shapr_gaussian.py::ShaprGaussianImputer`, wired into
the `player_eval` pipeline as method `kernelshap_gauss`
(`configs/methods/kernelshap_gauss.yaml`) — the paper's **KS-Gauss** rows,
produced by the validation study under the internal name `ks_shapr`.

**Executed protocol (pinned):** flattened `D = J*F*T` Gaussian with
**Ledoit-Wolf** covariance (the paper's plain ML covariance is singular at
pool size N = 1000 < D; `shrinkage: ml` remains available), fitted on the
family's **imputer-training pool** (N = 1000 fresh draws at seed 99, §8) —
*not* on the N = 200 evaluation set that the other classical imputers use
(§7).  The pipeline realises this via the method config's `fit_data`
override (`{N: 1000, seed: 99}`), merged onto the dataset config for the
fit-time dataset only.  5 completions per coalition; **conditional** grading
game.  Per-mask conditional parameters `(W, chol(Σ_c))` are cached; jitters
as in the study (`1e-10`-scaled on Σ_oo, `1e-8`-scaled escalating ×10 on
Σ_c).

**Parity gate** (`scripts/validate_ks_gauss_parity.py`): the study's cells
`{joint,cell}/gauss_k4/mlp/ks_shapr` (200 sequences each) were reproduced
end-to-end with release components only, threading the study's exact
per-sequence RNG stream (`default_rng([seed, idx])` through the coalition
rows in design order; the imputer's keyword-only `generator` argument exists
for this).  Fills are **bit-identical** to the study's (train pool, fitted
`mu`/`Σ`, and conditional draws all bit-equal); against the stored
`per_sequence.npz` the residual is classifier CPU-vs-GPU noise:

| cell | max&#124;Δφ&#124; | max&#124;Δφ*&#124; | max&#124;ΔEC1&#124; | target flips |
|---|---|---|---|---|
| joint/gauss_k4/mlp (M=5, exact) | 7.0e-08 | 9.3e-08 | 3.0e-08 | 0 |
| cell/gauss_k4/mlp (M=20, B=1024) | 3.3e-08 | 4.1e-08 | 4.0e-09 | 0 |

The study's hi-budget target regrade (B = 8192, seed 900001; the numbers in
its `analysis_final.json`) is reproduced by the same script: hi-budget
`max|Δφ*| = 3.1e-08`; the regraded EC1/EC3 (0.011237 / 0.099142) agree with
the study's `regrade_hi.json` to 2.6e-10 / 2.7e-09.

---

*Provenance: sections 1–9 adapt, with permission of scope, the RESOLUTIONS
log of the independent validation study that replicated this benchmark from
scratch (its experiment-1 `RESOLUTIONS.md`); the study's findings C1–C3 and
resolutions A1–A7 / B1–B10 are the source of record for what the released
pipeline executed.  The oracle unit tests in `tests/test_oracle_gate.py`
port that study's validation gate (properties G1–G6).*
