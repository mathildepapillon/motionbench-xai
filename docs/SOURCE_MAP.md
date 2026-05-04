# SOURCE_MAP — Old code → new code mapping

> **Source repo:** `/home/papillon/code/CARE-PD/` (contains the overloaded WIP that
> includes both the CARE-PD-specific classifier code and the MotionSHAP draft work).
>
> **Note on "MotionSHAP draft":** It lives *inside* CARE-PD — specifically in
> `CARE-PD/synthetic/`, `CARE-PD/model/vaeac/`, `CARE-PD/model/flow_matching/`,
> `CARE-PD/model/flow_shap/`, and `CARE-PD/shap_facade/`. The `manifoldshap/` repo
> is an older, unrelated project (Lie-VAE manifold SHAP for action classification).
>
> All paths below are relative to `/home/papillon/code/CARE-PD/`.

---

## 1. Synthetic data generators


| Old path                           | New module                                      | Action                                                                                                                                                                                             |
| ---------------------------------- | ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `synthetic/gaussian_motion.py`     | `motionbench/data/synthetic/gaussian_motion.py` | Port `GaussianMotionBenchmark`; strip MLP classifier into `classifiers/synthetic_mlp.py`; add `skeleton_adjacency`, `block_diagonal`, `data_driven` Σ_joint variants (Task 1A)                     |
| `synthetic/burr_motion.py`         | `motionbench/data/synthetic/burr_motion.py`     | Port `BurrMotionBenchmark`; generalize to `Marginal` ABC with Burr XII, StudentT, MoG, SkewNormal strategies (Task 1B)                                                                             |
| `synthetic/diagnostic_motion.py`   | `motionbench/data/synthetic/gait_periodic.py`   | Port Fourier-series gait generator; its exact analytic oracle maps to `GaussianOracle` since joints are independent (rho=0); add gait-periodic Toeplitz Σ_time via sum-of-cosines kernel (Task 1C) |
| `synthetic/burr_tabular.py`        | —                                               | Reference only (tabular Burr, not motion); do not port                                                                                                                                             |
| `synthetic/real_gait_benchmark.py` | `motionbench/data/real/care_pd.py`              | Reference for the `BenchmarkContext` concept; actual data loading is in `data/bmclab_datareader.py`                                                                                                |


---

## 2. Label functions

These are **currently embedded inside `GaussianMotionBenchmark`**. Task 1D extracts them.


| Old path → symbol                                                                                                                                | New module                                      | Action                                                                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `synthetic/gaussian_motion.py` → `nonlinear_olsen_score()`, `spatial_olsen_score()`, `_olsen_term()`, `GaussianMotionBenchmark.setup_label_fn()` | `motionbench/data/synthetic/label_functions.py` | Extract into standalone `LabelFunction` ABC; add `Linear`, `LocalizedTemporal`, `LocalizedSpatial`, `LocalizedSpatiotemporal`; expose `important_players(player_set)` (Task 1D) |


---

## 3. Player definitions


| Old path → symbol                                           | New module                                  | Action                                                                          |
| ----------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------- |
| `shap_facade/players.py` → `TemporalWindows`                | `motionbench/players/temporal_windows.py`   | Port; conformance to `PlayerSet` ABC                                            |
| `shap_facade/players.py` → `SpatialJoints`                  | `motionbench/players/spatial_joints.py`     | Port                                                                            |
| `model/actor/shap_masking.py` → `H36M_GROUPS`               | `motionbench/players/anatomical_groups.py`  | Extract anatomical group definitions for H36M-17; add `AnatomicalGroups` player |
| `shap_facade/players.py` (coalition → mask expansion logic) | `motionbench/players/joint_window_cells.py` | Port the spatiotemporal `(J, T)` mask logic                                     |
| `shap_facade/players.py` → `_coalition_to_element_mask()`   | `motionbench/utils/masking.py`              | Move as a standalone utility                                                    |


---

## 4. Oracles


| Old path → symbol                                                                                                                                                                                                                                     | New module                                      | Action                                                                                                                                                                                       |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `synthetic/gaussian_motion.py` → `GaussianMotionBenchmark.conditional_sample()`, `conditional_sample_spatial()`, `conditional_sample_spatiotemporal()`, `compute_v_true_all_coalitions()`, `compute_true_shapley()`, `compute_true_shapley_spatial()` | `motionbench/oracles/gaussian_oracle.py`        | Extract into `GaussianOracle(Oracle)`; add `true_shapley()` via enumeration (K≤12) or raise `NotImplementedError`; add `conditional_sample(x, mask, n)` conforming to `Oracle` ABC (Task 1A) |
| `synthetic/gaussian_motion.py` → `_ar1_cov()`, `_equicorr()`                                                                                                                                                                                          | `motionbench/data/synthetic/gaussian_motion.py` | Keep as module-level helpers                                                                                                                                                                 |
| `synthetic/gaussian_motion.py` → `_solve_shapley_wls()`, `_enumerate_temporal_coalitions()`, `_sample_kernelshap_coalitions()`, `_shapley_kernel_weight()`                                                                                            | `motionbench/utils/coalitions.py`               | Move as shared utilities                                                                                                                                                                     |
| `synthetic/burr_motion.py` → `BurrMotionBenchmark.conditional_sample*()` (copula conditional via Φ⁻¹ → Gaussian conditional → Φ)                                                                                                                      | `motionbench/oracles/copula_oracle.py`          | Extract into `CopulaOracle(Oracle)` (Task 1B)                                                                                                                                                |
| `shap_facade/imputers.py` → `OracleGaussianImputer`                                                                                                                                                                                                   | `motionbench/oracles/gaussian_oracle.py`        | Merge — `GaussianOracle` should satisfy both `Oracle` and `BaseImputer` contracts                                                                                                            |


---

## 5. Imputers — off-manifold baselines


| Old path → symbol                                                              | New module                                                      | Action                                                                      |
| ------------------------------------------------------------------------------ | --------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `shap_facade/imputers.py` → `ZeroImputer`                                      | `motionbench/imputers/off_manifold.py`                          | Port; adapt to `BaseImputer.impute(x, mask, n_samples)` contract            |
| `shap_facade/imputers.py` → `MeanImputer`                                      | `motionbench/imputers/off_manifold.py`                          | Port                                                                        |
| `shap_facade/imputers.py` → `MarginalImputer`                                  | `motionbench/imputers/off_manifold.py`                          | Port; rename `MarginalDonorImputer` in plan → keep `MarginalImputer` naming |
| —                                                                              | `motionbench/imputers/off_manifold.py` → `GaussianNoiseImputer` | New — trivial Gaussian noise fill, not in CARE-PD                           |
| `shap_facade/imputers.py` → `_coalition_to_element_mask()`, `_assert_layout()` | `motionbench/utils/masking.py`                                  | Move as shared utilities (same as §3)                                       |


---

## 6. Imputers — empirical / classical-conditional


| Old path → symbol                                     | New module                                                                                   | Action                                                                                                     |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `model/empirical/imputer.py` → `EmpiricalImputer`     | `motionbench/imputers/empirical.py` → `KNNConditionalImputer`, `EmpiricalConditionalImputer` | Port; refactor Ledoit-Wolf kNN logic to conform to `BaseImputer`; expose Aas 2021 §3.3 shrinkage (Task 2B) |
| `model/empirical/group_baselines.py`                  | `motionbench/imputers/empirical.py`                                                          | Reference for group-level variants                                                                         |
| `shap_facade/imputers.py` → `EmpiricalImputerAdapter` | `motionbench/imputers/empirical.py`                                                          | Port adapter pattern to `BaseImputer.impute()` interface                                                   |


---

## 7. Imputers — VAEAC


| Old path                                          | New module                                                    | Action                                                                                                        |
| ------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `model/vaeac/vaeac.py`                            | `motionbench/imputers/vaeac.py`                               | Port model architecture                                                                                       |
| `model/vaeac/heads.py`                            | `motionbench/imputers/vaeac.py`                               | Port head modules (prior encoder, posterior encoder, decoder)                                                 |
| `model/vaeac/imputer.py` → `VAEACImputer`         | `motionbench/imputers/vaeac.py` → `VAEACImputer(BaseImputer)` | Port; add `fit()` wrapper calling training loop; add `save()`/`load()`                                        |
| `model/gaitvae/vaeac.py`                          | —                                                             | Secondary VAEAC variant; use as reference if architecture differs from `model/vaeac/`; do not port separately |
| `train_vaeac.py`                                  | `scripts/train_vaeac.py`                                      | Port training loop; adapt to Hydra config (Task 2C)                                                           |
| `shap_facade/imputers.py` → `VAEACImputerAdapter` | `motionbench/imputers/vaeac.py`                               | Merge into `VAEACImputer.impute()`                                                                            |


---

## 8. Imputers — flow matching


| Old path                                                           | New module                                                                   | Action                                                                                                        |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `model/flow_matching/velocity_net.py`                              | `motionbench/imputers/flow_matching.py`                                      | Port velocity-field network architecture                                                                      |
| `model/flow_shap/imputer.py` → `FlowImputer`                       | `motionbench/imputers/flow_matching.py` → `FlowMatchingImputer(BaseImputer)` | Port; keep RePaint harmonisation; add `fit()`, `save()`, `load()`; investigate M=10 Burr regression (Task 2D) |
| `model/flow_shap/attribution.py`, `model/flow_shap/diagnostics.py` | —                                                                            | Reference only for debugging; do not port                                                                     |
| `train_flow_matching.py`                                           | `scripts/train_flow.py`                                                      | Port training loop; adapt to Hydra config (Task 2D)                                                           |
| `shap_facade/imputers.py` → `FlowImputerAdapter`                   | `motionbench/imputers/flow_matching.py`                                      | Merge                                                                                                         |


---

## 9. Attribution — KernelSHAP


| Old path                                                                            | New module                                                                        | Action                                                                                                                                  |
| ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `shap_facade/explainers.py` → `kernel_shap_explain()`                               | `motionbench/attribution/kernel_shap.py` → `KernelShapAttributor(BaseAttributor)` | Port; replace project-internal `shap_facade` masker with a proper `shap.maskers.Masker` subclass backed by `BaseImputer` (Task 2E)      |
| `synthetic/gaussian_motion.py` → `_solve_shapley_wls()`, coalition-sampling helpers | `motionbench/utils/coalitions.py`                                                 | Already covered in §4                                                                                                                   |
| `model/actor/shap_compute.py`                                                       | `motionbench/attribution/kernel_shap.py`                                          | Reference for player-level aggregation pattern; do not port directly — the plan requires a `shap.KernelExplainer`-backed implementation |


---

## 10. Attribution — non-SHAP methods (new, no port)

These modules have **no old code** to port from CARE-PD. Write from scratch using the indicated libraries.


| New module                                      | Library to wrap                                                                       |
| ----------------------------------------------- | ------------------------------------------------------------------------------------- |
| `motionbench/attribution/captum_methods.py`     | `captum` — IG, DeepLift, GradientShap, Saliency, SmoothGrad, InputXGradient (Task 3A) |
| `motionbench/attribution/lrp.py`                | `zennit` — ε, γ, α-β LRP variants (Task 3B)                                           |
| `motionbench/attribution/timeshap.py`           | `timeshap` library (Task 3C)                                                          |
| `motionbench/attribution/windowshap.py`         | `windowshap` library (Task 3C)                                                        |
| `motionbench/attribution/shats.py`              | `shats` library (Task 3C)                                                             |
| `motionbench/attribution/group_segment_shap.py` | `model/empirical/group_baselines.py` + paper pseudocode (Task 3C)                     |
| `motionbench/attribution/grad_cam.py`           | `captum.attr.LayerGradCam` (Task 3D)                                                  |


---

## 11. Classifiers — synthetic (new)

No old code to port for these. Write from scratch (Task 4A).


| New module                                         | Notes                                                                                       |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `motionbench/classifiers/synthetic_mlp.py`         | Extract `SyntheticMLPClassifier` from `CARE-PD/synthetic/gaussian_motion.py` and generalize |
| `motionbench/classifiers/synthetic_cnn.py`         | New — small 1D CNN                                                                          |
| `motionbench/classifiers/synthetic_transformer.py` | New — 4-layer transformer                                                                   |


---

## 12. Classifiers — CARE-PD encoders (port)

These are the **seven CARE-PD encoders**. Task 4B prioritises the first three; the remaining four are stretch goals.


| Old path                                                                              | New module                                                 | Action                                                                              | Priority |
| ------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ----------------------------------------------------------------------------------- | -------- |
| `model/poseformerv2/model_poseformer.py` + `configs/generate_config_poseformerv2.py`  | `motionbench/classifiers/ported_care_pd/poseformerv2.py`   | Port; strip training loop; add thin classification head; verify F1 vs CARE-PD paper | **P0**   |
| `model/potr/` (all files)                                                             | `motionbench/classifiers/ported_care_pd/potr.py`           | Port `PoseTransformerV2`/`PoseEncoderDecoder`; strip training loop                  | **P0**   |
| `model/motionbert/DSTformer.py` + `drop.py` + `configs/generate_config_motionbert.py` | `motionbench/classifiers/ported_care_pd/motionbert.py`     | Port `DSTformer`; strip training loop                                               | **P0**   |
| `model/motionagformer/MotionAGFormer.py` + `modules/`                                 | `motionbench/classifiers/ported_care_pd/motionagformer.py` | Port; strip training loop                                                           | P1       |
| `model/bilstm/bilstm_encoder.py`                                                      | `motionbench/classifiers/ported_care_pd/bilstm.py`         | Port; simple baseline                                                               | P2       |
| `model/mixste/model_cross.py`                                                         | `motionbench/classifiers/ported_care_pd/mixste.py`         | Port                                                                                | P2       |
| `model/motionclip/transformer.py` + `Encoder_TRANSFORMER`                             | `motionbench/classifiers/ported_care_pd/motionclip.py`     | Port encoder only                                                                   | P2       |
| `model/backbone_loader.py`                                                            | `motionbench/classifiers/ported_care_pd/__init__.py`       | Reference for checkpoint loading and thin head pattern; adapt `count_parameters`    |          |


---

## 13. Data loaders — real CARE-PD data


| Old path                                     | New module                              | Action                                                                                                     |
| -------------------------------------------- | --------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `data/bmclab_datareader.py` → `BMCLabReader` | `motionbench/data/real/care_pd.py`      | **Primary loader.** Port inference-only path; strip training augmentations; expose `BaseDataset` interface |
| `data/pdgam_datareader.py` → `PDGaMReader`   | `motionbench/data/real/care_pd.py`      | **Secondary loader.** Port if UPDRS-gait labels are needed (they are — UPDRS labels are the target)        |
| `data/dataloaders.py`                        | reference                               | Reference for `StratifiedKFold` split structure; do not port wholesale                                     |
| `data/skeleton_covariance/bmclab_h36m17/`    | `motionbench/data/synthetic/` (fixture) | Copy skeleton GMRF covariance matrix fixture; used for `skeleton_adjacency` Σ_joint in Tasks 1A and 1C     |
| `data/preprocessing/`                        | reference                               | Reference for SMPL → joint-positions pipeline; port the minimal inference piece needed for `care_pd.py`    |


---

## 14. Metrics


| Old path → symbol                                                                                      | New module                                 | Action                                                                        |
| ------------------------------------------------------------------------------------------------------ | ------------------------------------------ | ----------------------------------------------------------------------------- |
| `scripts/compute_attribution_quality_metrics.py` → EC1, EC2, EC3, EC1_norm, sign_agree, top_k, kendall | `motionbench/metrics/ground_truth.py`      | Port; wrap in `BaseMetric` ABC (Task 5A)                                      |
| `model/actor/shap_metrics.py`                                                                          | reference                                  | Reference for per-joint spatial metric patterns                               |
| `shap_facade/benchmarks.py`                                                                            | `motionbench/pipelines/synthetic_eval.py`  | Migrate `BenchmarkContext` pattern into the Hydra pipeline (Task 6A)          |
| —                                                                                                      | `motionbench/metrics/fidelity.py`          | New — wrap Quantus `FaithfulnessCorrelation`, `PixelFlipping`, etc. (Task 5B) |
| —                                                                                                      | `motionbench/metrics/stability.py`         | New — wrap Quantus `MaxSensitivity`, `Continuity` (Task 5C)                   |
| —                                                                                                      | `motionbench/metrics/sanity_checks.py`     | New — wrap Quantus `ModelParameterRandomisation` (Task 5C)                    |
| —                                                                                                      | `motionbench/metrics/ranking_agreement.py` | New — Spearman cross-protocol agreement matrix (Task 5D)                      |


---

## 15. Training scripts


| Old path                        | New path                                  | Action                                                                                   |
| ------------------------------- | ----------------------------------------- | ---------------------------------------------------------------------------------------- |
| `train_vaeac.py`                | `scripts/train_vaeac.py`                  | Port; replace argparse with Hydra; same config drives Task 2C and 2D for fair comparison |
| `train_flow_matching.py`        | `scripts/train_flow.py`                   | Port; replace argparse with Hydra                                                        |
| `train.py`                      | —                                         | CARE-PD classifier training; do not port (only inference needed)                         |
| `scripts/run_synthetic.py`      | `motionbench/pipelines/synthetic_eval.py` | Reference for the full synthetic sweep loop                                              |
| `scripts/compute_flow_shap*.py` | reference                                 | Reference for how the flow imputer is wired into KernelSHAP                              |


---

## 16. Utilities


| Old path                                                                                                                                                                                | New module                        | Action                                                              |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ------------------------------------------------------------------- |
| `utility/transforms/skeleton.py`                                                                                                                                                        | `motionbench/utils/skeleton.py`   | Port minimal skeleton transform utilities needed for data loading   |
| `utility/transforms/quaternion.py`                                                                                                                                                      | reference                         | Port only if SMPL loader needs it                                   |
| `synthetic/gaussian_motion.py` → `_ar1_cov()`, `_equicorr()`, `_shapley_kernel_weight()`, `_enumerate_temporal_coalitions()`, `_sample_kernelshap_coalitions()`, `_solve_shapley_wls()` | `motionbench/utils/coalitions.py` | Extract and test independently                                      |
| `shap_facade/imputers.py` → `_coalition_to_element_mask()`, `_assert_layout()`                                                                                                          | `motionbench/utils/masking.py`    | Extract                                                             |
| `utility/utils.py`                                                                                                                                                                      | reference                         | Misc utilities; port only as needed                                 |
| `const/const.py`, `const/path.py`                                                                                                                                                       | —                                 | CARE-PD-specific constants; do not port; replace with Hydra configs |


---

## 17. Files that are CARE-PD-specific — do not port

The following CARE-PD files contain evaluation-harness glue, plotting, or checkpoint-management code
that should not appear in motionbench-xai. Use as reference when needed.

- `run.py`, `eval_only.py`, `train.py`, `train_utils.py`
- `evaluate_shap_baselines.py`
- `scripts/aggregate_real_results.py`, `scripts/compute_baseline_shap.py`, `scripts/summarize_shap.py`
- `scripts/visualize_flow_shap.py`, `scripts/plot_skeleton_ec_resolution_sweep.py`
- `model/actor/shap_eval_shared.py`, `model/actor/cvae_data.py`
- `pretext/` (momask and motionagformer pretraining; pre-trained weights are loaded, not reproduced)
- `model/momask/`, `model/gaitvae/visualize.py`
- All `wandb/` outputs

---

## 18. Key design differences: old API → new API


| Old (CARE-PD)                                                         | New (motionbench)                                                  | Why it changed                                  |
| --------------------------------------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------- |
| `imp.sample_completions(x, coalition_mask, n_samples) → list[Tensor]` | `BaseImputer.impute(x_obs, mask, n_samples) → Tensor (n, J, F, T)` | Cleaner output type; list→batch tensor          |
| `bench.conditional_sample(x_np, s_obs, s_hid, n)` (window indices)    | `Oracle.conditional_sample(x, mask, n) → Tensor` (binary mask)     | Uniform mask convention across all player types |
| `_solve_shapley_wls()` duplicated in multiple modules                 | Single `coalitions.py` export                                      | DRY                                             |
| `Players.coalition_to_mask()` returns varying shapes based on layout  | `PlayerSet.coalition_mask(z) → Tensor (J, F, T)` bool              | Always element-level                            |
| `shap_facade/explainers.py` hardcodes WLS solve internally            | `KernelShapAttributor` delegates to `shap.KernelExplainer`         | Use library, not custom                         |


