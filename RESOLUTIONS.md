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

---

*Provenance: sections 1–9 adapt, with permission of scope, the RESOLUTIONS
log of the independent validation study that replicated this benchmark from
scratch (its experiment-1 `RESOLUTIONS.md`); the study's findings C1–C3 and
resolutions A1–A7 / B1–B10 are the source of record for what the released
pipeline executed.  The oracle unit tests in `tests/test_oracle_gate.py`
port that study's validation gate (properties G1–G6).*

<!-- BEGIN §11 (appended with the real-data player-set entry points) -->

## 11. Real-data player-set tracks (spatial joints and cross-cutting cells)

Added with the entry points `scripts/run_carepd_players_shap.py`,
`scripts/run_esc50_cells_shap.py` and `scripts/run_ptbxl_cells_shap.py`
(shared machinery in `scripts/_player_shap_common.py`).  These expose the
finer-granularity player sets of the validation study's real-data
player-set runs ([study] experiment 11):

| track | player set | M | coalitions |
|---|---|---|---|
| CARE-PD joint | `SpatialJoints` (one player per H36M joint) | 17 | sampled, B=2048 |
| CARE-PD cell | `JointWindowCells` (joint × K=4 windows of 20 frames) | 68 | sampled, B=2048 |
| ESC-50 cell | `BandWindowCells` (4 mel-bin quartiles × K=4 windows of 256 frames) | 16 | sampled, B=2048 |
| PTB-XL cell | `JointWindowCells` (lead × K=4 windows of 250 samples) | 48 | sampled, B=2048 |

**Executed conventions** (all [study]-pinned; the value functions, eval
subsets, mean/donor pools and classifier semantics are identical to the
temporal real-data sweeps of §RESOLUTIONS-scope — `run_care_pd_multiclf.py`,
`run_esc50_shap.py`, `run_ptbxl_leads_shap.py`):

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
  2^K enumerated rows.  [study] decision; the paper says "across sampled
  coalitions" without pinning boundary-row inclusion.
- **PlayerAOPC on sampled designs**: the M cumulative-deletion coalitions
  (players removed in decreasing |phi|, ties by player index) are evaluated
  **explicitly** — they are generally not in the sampled design.
  Deterministic fills are exact; VAEAC/Flow path fills are fresh draws from
  the same per-sequence stream.  [study] decision; the temporal protocol
  reads the path from the enumerated table, which has no sampled analogue.
- **Stochastic imputers**: one completion per coalition (n=1, the real-data
  release convention), per-sequence stream
  `np.random.default_rng([1104, fold, seq_idx])`, consumed as one
  `torch.manual_seed` per `impute_multi` call (coalition fills first, then
  the deletion-path fills continue the stream).  The imputers are the
  study-format checkpoints `checkpoints/imputers/{track}_{vaeac,flow}.pt`
  loaded by `motionbench.imputers.FrameVAEACImputer` / `FrameFlowImputer`
  (inference ports of the study's frame-token models, kept numerically
  identical; PTB-XL's smaller architecture is read from the checkpoint's
  `arch` dict).
- **POTR is excluded** from the CARE-PD player-set tracks: pooled accuracy
  0.374 fails the accuracy gate.
- **MotionBERT LayerNorm eps**: CARE-PD builds the DSTformer with
  `LayerNorm(eps=1e-6)` (`CARE-PD/model/backbone_loader.py`); the ported
  `_DSTformerBackbone` previously used the `nn.LayerNorm` default 1e-5,
  which shifted softmax probabilities by ~1e-4.  Fixed to default to
  eps=1e-6 so CARE-PD fine-tuned checkpoints reproduce exactly.

**Validation gate** (deterministic methods kernelshap_{zero,mean,marginal},
fold 1, N=200, H100): the release entry points were run against the study's
data caches and checkpoints and compared to the study's canonical fold-level
numbers (its `analysis.json` / `analysis_ptbxl.json`).  Component
fingerprints on CPU (coalition designs, WLS solves, all four player-set
mask expansions, all three value functions, both frame imputers, and an
end-to-end 2-sequence PTB-XL cell run) were bit-exact before the GPU gate.
Gate results (fold-level |Δfaithfulness| and |ΔPlayerAOPC| vs canonical):

| track | classifier | zero |Δfaith|/|Δaopc| | mean | marginal |
|---|---|---|---|---|
| CARE-PD joint (M=17) | motionbert | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| CARE-PD joint (M=17) | motionagformer | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| CARE-PD cell (M=68) | motionbert | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| CARE-PD cell (M=68) | motionagformer | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| ESC-50 cell (M=16) | ast | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| PTB-XL cell (M=48) | ecg_resnet1d | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |

**PASS — bit-exact**: every fold-level faithfulness and PlayerAOPC value
reproduces the canonical number with zero deviation (18/18 gate cells;
same GPU class and op order as the study's runs; the study's own gate
standard was 1e-3 with observed ≤3.1e-9).

VAEAC/Flow are covered by the shared machinery + the protocol pins above
(bit-exact imputer fingerprints; the study's own gate standard — its
stochastic cells were likewise covered by pins, not reruns, because exp-3's
seeds are unknowable).
<!-- END §11 -->
