# Checkpoint manifest

This release does **not** bundle large checkpoint files in the git tree.
This manifest documents which checkpoint files the pipelines expect, where
they go, and the SHA-256 digests of the reference files, so a downloaded (or
retrained) file can be verified.

Everything synthetic retrains from scratch on a single GPU with the scripts
in `scripts/` (see `REPRODUCIBILITY.md`); downloads are a convenience, not a
requirement.  Real-data classifiers and imputers retrain from the public
datasets with the same scripts.

## Download

Two archives (split by license — see **Licensing** below):

| archive | contents | size |
|---|---|---|
| `motionbench-xai-checkpoints.tar.gz` | synthetic classifiers + imputers, PTB-XL classifiers + imputers | ~140 MB |
| `motionbench-xai-checkpoints-esc50.tar.gz` | per-fold ESC-50 AST classifiers + ESC-50 imputers (**CC BY-NC**) | ~1.0 GB |

> **MAINTAINER TODO:** set the hosting base URL below before release and
> delete this note (decision pending: Hugging Face Hub vs GitHub Releases).

```bash
MOTIONBENCH_CKPT_URL=<base-url> bash scripts/download_checkpoints.sh          # core
MOTIONBENCH_CKPT_URL=<base-url> bash scripts/download_checkpoints.sh esc50   # + ESC-50 (CC BY-NC)
```

Each archive unpacks into the repo root at the paths below and carries its
own `SHA256SUMS` + `LICENSE_NOTES.md`; the download script verifies every
file against this manifest.

## Layout

```
motionbench/classifiers/checkpoints/
  synthetic/{dataset_config}/{classifier_config}.pt
                                        # e.g. gaussian_k4/synthetic_mlp.pt — exactly the
                                        # <configs/data name>/<configs/classifiers name>.pt
                                        # path the pipelines load; 21 files incl. seed replicas
  real/ptbxl_fold{n}.pt                   # PTB-XL per-fold ResNets
  real/esc50_ast_fold{n}.pt               # ESC-50 per-fold AST fine-tunes (esc50 archive)
results/synthetic_imputers/               # VAEAC / Flow synthetic imputers
checkpoints/imputers/                     # real-data VAEAC / Flow imputers
```

**Formats.**  Synthetic classifiers are native release format —
`{model_state_dict, config, val_acc, epoch, seed, ...}` with `config` holding
the constructor kwargs — and strict-load into
`motionbench.classifiers.Synthetic{MLP,CNN,Transformer}Classifier` with
bit-identical forward passes verified against the archived experiment runs.
Real-data files are in the validation-study format and load natively through
the player-set entry points (`scripts/run_*_shap.py`: key remap + strict
load) and `motionbench.imputers.FrameVAEACImputer.load` /
`FrameFlowImputer.load` (architecture read from the checkpoint's `arch`
dict; VAEAC `dec_trunk`/`dec_out` keys remapped on load).

## Reference digests

These are the exact files behind the paper's tables (per-classifier
accuracies: paper appendix, classifier-gate table).  The `*_seed43/44` MLP
files are seed replicas of the reference training run, shipped for variance
checks; the pipelines load the unsuffixed file.

### Core archive — synthetic classifiers (`motionbench/classifiers/checkpoints/synthetic/`)

| file | sha256 |
|---|---|
| `burr_m5/synthetic_cnn.pt` | `752b2fb5dca8ea63c1b3bb2177750edac1059172b93c16d3bc1cb6f919db6a53` |
| `burr_m5/synthetic_mlp.pt` | `f93ff19c36e01ae8da1a435544efa0d2c83fca2149aa16623eafaa8273f3e1f0` |
| `burr_m5/synthetic_mlp_seed43.pt` | `a270efa7014dc221ccbd56027d336b4cd3dbb7952c046d347ba57e61cf2dfea8` |
| `burr_m5/synthetic_mlp_seed44.pt` | `10e0cdcaa1c9779bdea899537225d3693e27ee8198681cbb0c6a45bfb62533cb` |
| `burr_m5/synthetic_transformer.pt` | `f4f1e89a3c02de06bf9c738670bfcac86b8c7ec2bbd553116037e502a359bcb8` |
| `gait_periodic/synthetic_cnn.pt` | `6ae28a8b39afbb5e3aef69d096a32140208e6c37d515fc7274d631fe1e5318ad` |
| `gait_periodic/synthetic_mlp.pt` | `4cd6d1bbff67f577efb055f5766eb4cb8524f08bbce2eb46e730020621d203e7` |
| `gait_periodic/synthetic_transformer.pt` | `43c7be80fc18410ed40bb32eb6c89facaec7e2fb8986d386de896e234fe824f3` |
| `gaussian_k4/synthetic_cnn.pt` | `211a751421a7b8b0d919c48934707fcef953ac6a973d5fe55bf67155ad017583` |
| `gaussian_k4/synthetic_mlp.pt` | `ebd9f6f2333752c87ae438f986cd619528f5f8d062c0520279d6ba3965d3eeac` |
| `gaussian_k4/synthetic_mlp_seed43.pt` | `8dc6378fda6e22a78abda776ba08f9b16ae6437010915cb38025a52dcebf7f6a` |
| `gaussian_k4/synthetic_mlp_seed44.pt` | `f03a5e05daa7082fb13f0df57ccb2693d8b70ed42698dec66b4a5b13956c7277` |
| `gaussian_k4/synthetic_transformer.pt` | `a4c085874497be07d0c84808ca6ca2896a1c5638fc6e503a6af884ae366297f0` |
| `skeleton_gait_combined/synthetic_cnn.pt` | `9fbe3af0849d645aca04338de1e44fe71e7551470658d10009a090771908e7fd` |
| `skeleton_gait_combined/synthetic_mlp.pt` | `29021b87cf29ecc6e55ecf702fe8d3f6c0359b9ded21f161528892d3237c43e4` |
| `skeleton_gait_combined/synthetic_transformer.pt` | `0f79de43d56fd7265b8c4865224e636715aa04c163246f9a02946fe8a98e522e` |
| `skeleton_structured/synthetic_cnn.pt` | `2041c40cb5f006a401d573b559fae2f081d98b17f6651597c7f5fb6b3855c8fb` |
| `skeleton_structured/synthetic_mlp.pt` | `0dea1107d8b544f35713daed0614eb03b5e0cd102b754ca2458d57b8f5a4ee07` |
| `skeleton_structured/synthetic_mlp_seed43.pt` | `2eb55d71b7b1c5928dee8a3b3cbf1ff8724d663669224131e2111b1185fff409` |
| `skeleton_structured/synthetic_mlp_seed44.pt` | `1c875b1ddea4b9464952f669f277502b3e6a64f74dd22b1b800dd7dc93d805a0` |
| `skeleton_structured/synthetic_transformer.pt` | `795b28aa7a1a1f234c59eacf90332ac7ea02963f66758a886b5bf4b09da71521` |

### Core archive — synthetic imputers (`results/synthetic_imputers/`)

| file | sha256 |
|---|---|
| `flow_burr_j5t20.pt` | `2ed0d8f186a7b7332482ceb208b14b59cfdd0a327769d5062764d0261e77056d` |
| `flow_gait_j17t16.pt` | `8dc65fdf93cfe168ff4abfb61c271f7fa53700730f9933989bca0c832c15ad7f` |
| `flow_gaussian_j5t16.pt` | `ebbb132604938d1db1d09a732c81d72ed7b8d7fb25619d1ab6792080b698c6bd` |
| `flow_skeleton_j17t16.pt` | `5c23e4b09ac89790b24c9c656840fdac57248f976756301ac806e24df33dc81b` |
| `flow_skelgait_j17t16.pt` | `29f69f20865b2fa470d0f4800f8a842e49ad1f1c09fbb3ec12be6750fe1e9300` |
| `vaeac_burr_j5t20.pt` | `71f6a275bfb33cc349041067ebacb156794b4f53d1568ea0d360c6aaa0910fad` |
| `vaeac_gait_j17t16.pt` | `49b54cc91d87752f1990c7c9d13d30b77512be2d14607975606527578e7bc149` |
| `vaeac_gaussian_j5t16.pt` | `e2a5788005b5eab7cfdb728b1aa1f863c958b0a54be853beab701d9c8099a18c` |
| `vaeac_skeleton_j17t16.pt` | `b63ae09a554f0856cc1b525c80aab182f1184a9da4a971fd4692b4222e178d6b` |
| `vaeac_skelgait_j17t16.pt` | `d9fd0bf621bc5be9a676967956d55d261ecde210fbfec8f4cc8176968ef9113e` |

### Core archive — PTB-XL (`motionbench/classifiers/checkpoints/real/`, `checkpoints/imputers/`)

| file | sha256 |
|---|---|
| `ptbxl_fold1.pt` | `55da14a595c1f4da9b21e7a9c88f44bdbc9c4b0d7c90486efbd7947cbb4d7de1` |
| `ptbxl_fold2.pt` | `2197bbc465d2e4d835a138b7c0805c97e793fbbabe6f873e373699d1f67e0e4a` |
| `ptbxl_fold3.pt` | `ca686c1c853e0d1a39cefa6371aef9fc26ce9b4f5996afb33bb22186e6e6d75f` |
| `ptbxl_flow.pt` | `b629bc09598d9541125e8b0432ba209b64f79f6513b8ca977d079bd184d3afef` |
| `ptbxl_vaeac.pt` | `b2db4f01bdb021e5f4fcbf02bd66286eb17616b52d896dfbae73e297516fec96` |

### ESC-50 archive (CC BY-NC)

| file | sha256 |
|---|---|
| `esc50_ast_fold1.pt` | `82763657ac83ff2eefda303baa2546f863e451f80e99f6c292d0bfaba0427ec8` |
| `esc50_ast_fold2.pt` | `b8d6463dcd4785ffaca64207ae5576d81041f113dce1420e4d01ca063a428fc8` |
| `esc50_ast_fold3.pt` | `dfae068bbf93b8da27719494c0154248cabd07ab5c4ecf131ae17020eb9a2087` |
| `esc50_flow.pt` | `496c522f60ef8d85b6b4690081301ea913b2b9a90b09b9c08315ca8cb59661db` |
| `esc50_vaeac.pt` | `ddf4d4678afdbb513095a67e52e290bc59e97796b5b09c03e9679d32fe70a50b` |

## Licensing

- Synthetic files: **MIT** (trained on generated data).
- PTB-XL-derived files: **CC BY 4.0** with the PTB-XL citation (the dataset's
  license).
- ESC-50-derived files (separate archive): **CC BY-NC 4.0** — non-commercial
  only, ESC-50's license carries through; see the archive's
  `LICENSE_NOTES.md`.
- **CARE-PD-derived weights (classifiers and imputers) are not distributed**:
  CARE-PD is CC BY-NC-ND 4.0 (NoDerivatives).  Retrain them locally from the
  dataset (downloaded under its own terms) with the trainers in `scripts/`;
  the player-set entry points gate-verify a retrained stack against the
  canonical results in `results/canonical/` (their per-file digests are
  listed above only where redistribution is permitted).

## Notes

- The three small checkpoints tracked in
  `motionbench/classifiers/checkpoints/*.pt` are smoke-test artifacts kept so
  the examples run offline; they are not the table-generating models.
- If a retrained file's digest differs, that is expected (training is not
  bit-reproducible across hardware); verify it functionally with the gate
  scripts instead (`REPRODUCING_PAPER.md`).
