# Executed conventions

This document records the implementation conventions the released pipeline
executes, including every detail the paper's prose leaves open.  The
canonical result files in `results/canonical/` were produced under exactly
these conventions.  Where an alternative is available behind a configuration
flag, the flag is noted.

## 1. Value function

`v(S) = f(mean of R completions)` ("f-of-mean").  The KernelSHAP masker
draws `n_completion_samples` completions, averages them, and evaluates the
classifier once at the mean
(`motionbench/attribution/kernel_shap.py::_MotionBenchMasker`).  The
alternative "mean-of-f" semantics `v(S) = (1/R) Σ_r f(x_S ⊔ x̂⁽ʳ⁾)` is
configurable per method via `value_fn: f_of_mean | mean_of_f` in
`configs/methods/*.yaml`; the two coincide for deterministic imputers and
for R = 1, and differ by a Jensen gap otherwise.

Every stochastic fill uses the same draw count, `n_completion_samples: 5`
(`configs/methods/*.yaml`), so no method gains or loses averaging relative
to another.  The deterministic grading targets in
`motionbench/oracles/deterministic.py` are the exact `R → ∞` limits of the
f-of-mean semantics.

## 2. Labels for the pillar synthetic datasets

Class labels are tertiles of the **joint-0 grand mean**,
`score = x[:, 0, :, :].mean(axis=(1, 2))`, with cutoffs calibrated on each
generated batch.  No `label_fn` is passed for any pillar dataset in
`scripts/train_synthetic_clf.py::DATASET_CONFIGS`, so all fall through to
this dataset default.  `OlsenInteraction`
(`motionbench/data/synthetic/label_functions.py`) is available as an
explicit `label_fn` for interaction-driven labels (§3).

Two percentile conventions coexist in the code and yield slightly different
cutoffs: `GaussianMotionDataset` uses `np.percentile(score, [33, 67])` while
`SkeletonStructuredDataset` (and the label functions) use
`linspace(0, 100, n_classes + 1)[1:-1]` (33.33/66.67).  Eval-time labels do
not affect attribution (each sequence's explained target is the classifier's
own argmax), but retraining classifiers from scratch reproduces the released
checkpoints only under the per-class convention.

## 3. Olsen interaction: coefficients, seed, and window-group convention

The `OlsenInteraction` label convention:

- window means `w_k` over uniform `T // K` windows; `u_k = Φ(w_k / σ_k)`
  with `σ_k` = the batch standard deviation of `w_k` (clipped at 1e-8);
- one interaction term per **complete group of 4 windows** — `r` ranges over
  complete groups only, so K = 5 uses windows 0–3 and K < 4 is undefined;
- coefficients `c1, c2 ~ U(0.5, 2.0)`, `c3 ~ U(0.5, 1.0)` drawn from
  `numpy.random.default_rng(0)` (the fit seed is 0);
- class labels are tertiles of the score, cutoffs calibrated on the batch.

## 4. KS-Empirical kernel convention

`EmpiricalConditionalImputer` (`motionbench/imputers/empirical.py`)
implements Aas et al. (2021) Algorithm 2 with:

- per-coordinate z-scoring of the donor pool;
- Ledoit–Wolf covariance on the observed sub-block per coalition;
- Gaussian kernel bandwidth **σ = 0.1 × median Mahalanobis distance**
  (the shapr `fixed_sigma = 0.1` convention applied to the median distance,
  not a fixed σ = 0.1);
- **η = 0.9** truncation of the donor weight mass
  (`configs/methods/kernelshap_empirical.yaml`, overriding the class default);
- a leave-one-out guard masking donors with `d² ≤ 1e-8 × median(d²)` —
  load-bearing because the donor pools are fit on the evaluation set itself,
  which contains the query sequence (§7);
- weighted resampling of whole donor rows; observed coordinates overwritten
  bit-for-bit.

## 5. VAEAC architecture

The VAEAC (`motionbench/imputers/vaeac.py` and the CARE-PD checkpoints) is a
**per-frame-token transformer**: three subnets (full encoder, prior encoder,
decoder), each a 2-layer TransformerEncoder trunk with d_model = 256,
nhead = 8, ff = 512, dropout 0.1, sinusoidal frame positional encoding,
per-frame latent dimension 64, and a Gaussian output head with one learnable
global log-σ.  Tokens: full = [x, mask], prior = [x·mask, mask],
decoder = [z, x·mask, mask] (one token per frame).  Loss: NLL on hidden
entries + KL(q‖p); Adam lr 1e-3, gradient clip 1.0, batch 16.  Inference
samples z from the prior at temperature 1 and restores observed entries
exactly.

## 6. Grading targets

All grading targets are **deterministic**: the exact conditional-mean (or
marginal-mean) fill is plugged into the value function for every coalition,
so a perfect attribution scores EC1 = EC3 = 0 with zero grading noise
(`motionbench/oracles/deterministic.py`).  Coalitions are enumerated exactly
wherever `2^K` (temporal) or `2^M` (player sets, M ≤ 12) fits the coalition
budget.  Above the enumeration bound, targets are solved on hi-budget
sampled designs built by `scripts/regrade_hi_budget.py` (B = 8192,
seed 900001; additionally B = 32768, seed 910001 at M = 68), with
grading-noise floors measured from independent replicate targets at
disjoint coalition seeds (`meta.grading_floors` in
`results/canonical/synthetic_players.json`).

## 7. Donor pools contain the query

Mean / MarginalDonor / Empirical imputers are fit on the N = 200 evaluation
dataset itself (`synthetic_eval.py::_build_and_fit_imputer(dataset=eval ds)`),
so donor pools contain the explained sequence.  The KS-Empirical LOO guard
of §4 exists for exactly this reason.  KS-Gauss is the exception: it fits on
the disjoint imputer-training pool (§11).

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
| Hi-budget grading targets | B=8192 seed 900001; B=32768 seed 910001; replicate floors 910001/920001 | `scripts/regrade_hi_budget.py` |

Imputer training: one VAEAC + one Flow per distribution family, shared
across K/M variants (gaussian J5T16, burr J5T20, skeleton J17T16, gait
J17T16, skel+gait J17T16); N = 1000 draws at seed 99, 85/15 train/val,
80 epochs.

## 9. Synthetic classifier architectures

Transformer (`configs/classifiers/synthetic_transformer.yaml` and the
training script): **d_model = 32, nhead = 4, num_layers = 2**.  MLP:
window-mean features (`player_mode: temporal`), hidden 64.  CNN: 32→32→64,
k = 5, Conv→ReLU→BN, GAP, Linear.  The executed configurations are
authoritative for the released tables and checkpoints.

## 10. Ground-truth Shapley solve

Constrained WLS over the coalition design: intercept column included;
empty/full coalitions added as hard constraint rows at weight 1e6; ridge
1e-8; Shapley kernel weight `(M−1)/(C(M,s)·s·(M−s))`
(`motionbench/utils/coalitions.py::solve_shapley_wls`).  Burr-family
conditionals are computed in latent z-space (`z = Φ⁻¹(F(x))`, Gaussian
conditional, `x = F⁻¹(Φ(z))`, observed entries restored bit-for-bit) with
probability clips at 1e-9; conditional covariances symmetrised and
ridge-regularised at 1e-8.

## 11. KS-Gauss (fitted Gaussian conditional)

The shapr-style parametric Gaussian conditional imputer of Aas et al. (2021)
§3.2, implemented as
`motionbench/imputers/shapr_gaussian.py::ShaprGaussianImputer` and wired
into the `player_eval` pipeline as method `kernelshap_gauss`
(`configs/methods/kernelshap_gauss.yaml`).  In the canonical result files
the method key is `ks_shapr`; the paper reports it as **KS-Gauss**.

Executed protocol (pinned): flattened `D = J*F*T` Gaussian with
**Ledoit-Wolf** covariance (a plain ML covariance is singular at pool size
N = 1000 < D; `shrinkage: ml` remains available), fitted on the family's
**imputer-training pool** (N = 1000 fresh draws at seed 99, §8) — *not* on
the N = 200 evaluation set that the other classical imputers use (§7).  The
pipeline realises this via the method config's `fit_data` override
(`{N: 1000, seed: 99}`), merged onto the dataset config for the fit-time
dataset only.  5 completions per coalition; **conditional** grading game.
Per-mask conditional parameters `(W, chol(Σ_c))` are cached; jitters are
`1e-10`-scaled on Σ_oo and `1e-8`-scaled, escalating ×10, on Σ_c.

## 12. Real-data player-set tracks (spatial joints and cross-cutting cells)

The entry points `scripts/run_carepd_players_shap.py`,
`scripts/run_esc50_cells_shap.py` and `scripts/run_ptbxl_cells_shap.py`
(shared machinery in `scripts/_player_shap_common.py`) expose the
finer-granularity player sets on the real tracks:

| track | player set | M | coalitions |
|---|---|---|---|
| CARE-PD joint | `SpatialJoints` (one player per H36M joint) | 17 | sampled, B=2048 |
| CARE-PD cell | `JointWindowCells` (joint × K=4 windows of 20 frames) | 68 | sampled, B=2048 |
| ESC-50 cell | `BandWindowCells` (4 mel-bin quartiles × K=4 windows of 256 frames) | 16 | sampled, B=2048 |
| PTB-XL cell | `JointWindowCells` (lead × K=4 windows of 250 samples) | 48 | sampled, B=2048 |

Executed conventions (the value functions, eval subsets, mean/donor pools
and classifier semantics are identical to the temporal real-data sweeps —
`run_care_pd_multiclf.py`, `run_esc50_shap.py`, `run_ptbxl_leads_shap.py`):

- **Coalition design**: every M exceeds the exact-enumeration bound
  (M ≤ 12), so all four tracks use the fixed shared design
  `sampled_coalition_set(M, budget=2048, seed=7919)` (§8 seed table) —
  boundary rows pinned at rows 0/1 with weight 1e6, complete size pairs
  enumerated in kernel-mass order, importance-corrected sampled remainder.
  One design per M, reused across methods and folds.
- **phi**: constrained WLS with intercept, boundary constraints at 1e6,
  ridge 1e-8 (`phi_from_values`; §10 semantics), interior rows carrying the
  design's importance weights.
- **Faithfulness on sampled designs**: Pearson over ALL B+2 rows including
  the two boundary rows — mirrors the temporal protocol, which uses all
  2^K enumerated rows.
- **PlayerAOPC on sampled designs**: the M cumulative-deletion coalitions
  (players removed in decreasing |phi|, ties by player index) are evaluated
  **explicitly** — they are generally not in the sampled design.
  Deterministic fills are exact; VAEAC/Flow path fills are fresh draws from
  the same per-sequence stream.  The temporal protocol reads the path from
  the enumerated table, which has no sampled analogue.
- **Stochastic imputers**: one completion per coalition (n=1, the real-data
  convention), per-sequence stream
  `np.random.default_rng([1104, fold, seq_idx])`, consumed as one
  `torch.manual_seed` per `impute_multi` call (coalition fills first, then
  the deletion-path fills continue the stream).  The imputers are the
  reference checkpoints `checkpoints/imputers/{track}_{vaeac,flow}.pt`
  loaded by `motionbench.imputers.FrameVAEACImputer` /
  `FrameFlowImputer` (PTB-XL's smaller architecture is read from the
  checkpoint's `arch` dict).
- **POTR is excluded** from the CARE-PD player-set tracks: pooled accuracy
  0.374 fails the accuracy gate.
- **MotionBERT LayerNorm eps**: CARE-PD builds the DSTformer with
  `LayerNorm(eps=1e-6)` (`CARE-PD/model/backbone_loader.py`); the ported
  `_DSTformerBackbone` defaults to eps=1e-6 so CARE-PD fine-tuned
  checkpoints reproduce exactly.

The canonical files' `gate` blocks record the validation runs behind these
entry points (fold-level faithfulness and PlayerAOPC reproduced against the
stored temporal/per-lead tracks before any new numbers were produced).
