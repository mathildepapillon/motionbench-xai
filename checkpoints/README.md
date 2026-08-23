# Checkpoint manifest

This release does **not** bundle large checkpoint files in the git tree.
This manifest documents exactly which checkpoint files the pipelines expect,
where they go, and the SHA-256 digests of the reference training runs, so a
downloaded (or retrained) file can be verified.

Everything synthetic can be **retrained from scratch on CPU/single GPU** with
the scripts in `scripts/` (see `REPRODUCIBILITY.md`); downloads are a
convenience, not a requirement.

## Download

> **MAINTAINER TODO:** fill in the hosting URL(s) below before release and
> delete this note.  The archive should unpack into the repo root so files
> land at the paths in the tables.
>
> ```bash
> # placeholder — replace <URL> with the final hosting location
> curl -L <URL>/motionbench-xai-checkpoints.tar.gz | tar xz -C .
> sha256sum -c checkpoints/SHA256SUMS   # optional integrity check
> ```

## Expected layout

```
motionbench/classifiers/checkpoints/
  synthetic/{dataset}/{classifier}.pt      # synthetic classifiers
  real/carepd_bmclab_fold{n}_{backbone}.pt # CARE-PD classifier heads
  real/ptbxl_fold{n}.pt                    # PTB-XL per-fold ResNets
results/synthetic_imputers/{family}/       # VAEAC / Flow synthetic imputers
results/ptbxl_imputers/vaeac/vaeac_best.pt # retrained PTB-XL VAEAC
checkpoints/                               # study-format real-data files
  carepd_clf/{backbone}_fold{n}.pt         #   (see "Real-data checkpoints")
  ptbxl_clf/fold{n}.pt
  imputers/{carepd,esc50,ptbxl}_{vaeac,flow}.pt
```

The `checkpoints/{carepd_clf,ptbxl_clf,imputers}/` tree holds the real-data
files in the validation study's own format (paths in the "Real-data
checkpoints" table below are relative to `checkpoints/`).  The real-data
player-set entry points (`scripts/run_carepd_players_shap.py`,
`run_esc50_cells_shap.py`, `run_ptbxl_cells_shap.py`) load classifiers from
either layout and read the VAEAC/Flow imputers from
`checkpoints/imputers/`.

Synthetic classifier checkpoints are dicts with keys `model_state_dict`,
`config` (constructor kwargs), `val_acc`, `epoch` (written by
`scripts/train_synthetic_clf.py`).  The `player_eval` pipeline accepts any
dict containing `model_state_dict`, or a bare state dict.

## Reference digests (validation-study training runs)

The digests below are the SHA-256 hashes of the checkpoint files from the
**independent validation study's** training runs (its experiment-9 release
bundle), recorded from its payload manifest.  These are the exact files its
replication figures were computed with.  **Format note:** these files are in
the study's own format — classifiers carry `model_state_dict` without a
`config` key (constructor kwargs must come from the dataset table), and
CNN/Transformer state-dict key names differ from this package's module names
(`body.*` vs `conv_layers.*`; `proj/pe/enc.*` vs
`input_proj/pos_enc/transformer.*`).  The **MLP classifiers load into this
package's `SyntheticMLPClassifier` unchanged** (verified strict-load +
bit-identical forward); CNN/Transformer/imputer files need a key-renaming
conversion or a retrain with this package's own trainers.

### Synthetic classifiers (release-semantics variant, seed 42; `mlp_seed4x` = seed replicas)

| file | sha256 |
|---|---|
| gauss_k4/mlp.pt | `95035028bde0f3f7de39af665efe56979466949b5f9c6e3b57f613df548a8d1c` |
| gauss_k4/mlp_seed43.pt | `0b108332692affa8da64c5dcbaf4b3ae2b0a1be38d6cd4b47fa810d6d77d10ca` |
| gauss_k4/mlp_seed44.pt | `a32242226d11280a5751b4537a1c0634ec4e7e33af0b3d4fcd0468ab35307f0c` |
| gauss_k4/cnn.pt | `4c2c999d28abad15b7831d813c542ebb3a1e1610c02a57eedb104f0e6f9aa1c9` |
| gauss_k4/transformer.pt | `6365e5c0f0ede54332ad70b729680ec734d6f83fdcfc22accb2d1a389ad42deb` |
| burr_m5/mlp.pt | `757ecff17cb122259142d53051a5372a28ab63a8e43641c97d3431009e44cb7c` |
| burr_m5/mlp_seed43.pt | `bdee546b954cda3d5d772c69008c94a024832dd9f7143471de82a6beda022d1f` |
| burr_m5/mlp_seed44.pt | `ac041b4d4ed2154e07df4922c712e433dd61f9b65bc6a6e4e7a56d06dcb613df` |
| burr_m5/cnn.pt | `98123c946a074955238cc31f31d43e75aa08127f7b55594f9aac81e039d2c78c` |
| burr_m5/transformer.pt | `62b8a5cd4389ecdacb8cf6cecce0e0f2b8f21d69f19cf931d485b9425f45b491` |
| skeleton/mlp.pt | `34da7d924e7a8a41f09a64272afb3bd632e02d04f910dcf07762a0f3359223ca` |
| skeleton/mlp_seed43.pt | `ad50217bc1cd5f41074872a64d50a9f0a4693f5b290fa98a38c3961d8908db03` |
| skeleton/mlp_seed44.pt | `fd17d8b6d1a0b44936af3443f9dc886a7b6da2b9cbe5f7a82096f450907d1e68` |
| skeleton/cnn.pt | `dcb1dfd75e293d7f86e095fc29b8f23ce1d3a5d046ac1038be49e824c1f6c877` |
| skeleton/transformer.pt | `3ec4dff4aab5cfd2dbb6662ddec711b218a49ee63f7b352a0e9d3df1d389b3e3` |
| gait/mlp.pt | `399b150547c84ff6cf8a6f8fbb30f1a1ce24cb248518b96e7a617fb3300920a0` |
| gait/cnn.pt | `4077962a95deeb8bc11af45a25b18046304099b75148eb637bd84a9441a5e461` |
| gait/transformer.pt | `f4657b218a057ccb67cd4c7acf6a5f32e6a690ed21e753d8158b3ff0af247737` |
| skel+gait/mlp.pt | `5a06be8661230c80b9dae40c253786b1dcb25ccebfcb8b465046c058c2fdf3b9` |
| skel+gait/cnn.pt | `7e97e86e6c1fb9d4304a773333f899b1d22ebe140377e7ef0fca8772228aaac5` |
| skel+gait/transformer.pt | `36674131501cfd9660fd34896e5c6160fade3233236f5903200fba469e021a96` |

(A parallel `classifiers_paper/` set trained under the paper-arm conventions
— Olsen labels, 64/4/4 Transformer — exists in the study bundle with its own
digests; ask the maintainer if needed.)

### Synthetic imputers (one VAEAC + one Flow per distribution family)

| file | sha256 |
|---|---|
| vaeac_gaussian_j5t16.pt | `e2a5788005b5eab7cfdb728b1aa1f863c958b0a54be853beab701d9c8099a18c` |
| vaeac_burr_j5t20.pt | `71f6a275bfb33cc349041067ebacb156794b4f53d1568ea0d360c6aaa0910fad` |
| vaeac_skeleton_j17t16.pt | `b63ae09a554f0856cc1b525c80aab182f1184a9da4a971fd4692b4222e178d6b` |
| vaeac_gait_j17t16.pt | `49b54cc91d87752f1990c7c9d13d30b77512be2d14607975606527578e7bc149` |
| vaeac_skelgait_j17t16.pt | `d9fd0bf621bc5be9a676967956d55d261ecde210fbfec8f4cc8176968ef9113e` |
| flow_gaussian_j5t16.pt | `ebbb132604938d1db1d09a732c81d72ed7b8d7fb25619d1ab6792080b698c6bd` |
| flow_burr_j5t20.pt | `2ed0d8f186a7b7332482ceb208b14b59cfdd0a327769d5062764d0261e77056d` |
| flow_skeleton_j17t16.pt | `5c23e4b09ac89790b24c9c656840fdac57248f976756301ac806e24df33dc81b` |
| flow_gait_j17t16.pt | `8dc65fdf93cfe168ff4abfb61c271f7fa53700730f9933989bca0c832c15ad7f` |
| flow_skelgait_j17t16.pt | `29f69f20865b2fa470d0f4800f8a842e49ad1f1c09fbb3ec12be6750fe1e9300` |

Study VAEAC files are `{state_dict, shape}` dicts; Flow files add
`num_steps`.  This package's `VAEACImputer.load` / `FlowMatchingImputer.load`
expect `{state_dict, config, fitted}` / `{constructor_params, state_dict,
train_losses}` respectively — retrain with `scripts/train_vaeac.py` /
`scripts/train_flow.py` (80 epochs, seed 99 training data; see
`RESOLUTIONS.md` §8) or convert.

### Real-data checkpoints

| file | sha256 | note |
|---|---|---|
| ptbxl_clf/fold1.pt | `55da14a595c1f4da9b21e7a9c88f44bdbc9c4b0d7c90486efbd7947cbb4d7de1` | bare state dict, 1-D ResNet, 2-class head |
| ptbxl_clf/fold2.pt | `2197bbc465d2e4d835a138b7c0805c97e793fbbabe6f873e373699d1f67e0e4a` | |
| ptbxl_clf/fold3.pt | `ca686c1c853e0d1a39cefa6371aef9fc26ce9b4f5996afb33bb22186e6e6d75f` | |
| carepd_clf/motionagformer_fold1.pt | `b504c6e67d9170509238c1bfcdd1642606532f5f42bf1f9a9346f10eabcefee1` | `{state_dict, backbone, fold}` |
| carepd_clf/motionagformer_fold2.pt | `6672b2167c28ed0d04dcdc6ed2dc7fc268501927fb547a47fafe880da23d71eb` | |
| carepd_clf/motionagformer_fold3.pt | `09158649eccb56edcf56af655ecbf2b1a27320ae4811cf2876c6c0b7fafa8750` | |
| carepd_clf/potr_fold1.pt | `2cd36a1ea88857cf489d76d8c82a3d5cef147586e29fb5530d09bad40dfdb45f` | |
| carepd_clf/potr_fold2.pt | `781951e4fe47c70d17cdd14ae5145d29db88aa2de5d0cae9089f7c03bafc5be8` | |
| carepd_clf/potr_fold3.pt | `1e176205a970e68938f6d5c1b0ce7e905a392bb30a82f69532370a7e6e15ba7a` | |
| carepd_clf/motionbert_fold1.pt | `1913e4fc49bf689f24e958c5ec8634d1c1495cb9bbbe87a7cda5876fabb756b2` | 170 MB; excluded from the study bundle (size cap) — host separately |
| carepd_clf/motionbert_fold2.pt | `6338fa278ba56c0a7025f22f6a93d418b2d4bd6fa991f3911a4bed1ca65aa620` | " |
| carepd_clf/motionbert_fold3.pt | `9b3651e1c6f9e847d15bb2eefa5fcd496208c1639cd6b3e0f2bbd7680587c887` | " |
| imputers/carepd_vaeac.pt | `056cfef14b20f04b3524ff8f68657ab571d478b68a35b18c9cb79c3cde369079` | `{state_dict, shape, arch}` |
| imputers/carepd_flow.pt | `c84092634d5b161a8343ce59ec602acc92dd0e44ee06efd1ee8a9b8cbd001755` | |
| imputers/ptbxl_vaeac.pt | `b2db4f01bdb021e5f4fcbf02bd66286eb17616b52d896dfbae73e297516fec96` | |
| imputers/ptbxl_flow.pt | `b629bc09598d9541125e8b0432ba209b64f79f6513b8ca977d079bd184d3afef` | |
| imputers/esc50_vaeac.pt | `ddf4d4678afdbb513095a67e52e290bc59e97796b5b09c03e9679d32fe70a50b` | |
| imputers/esc50_flow.pt | `496c522f60ef8d85b6b4690081301ea913b2b9a90b09b9c08315ca8cb59661db` | |

These load natively: the classifiers through the loaders in the player-set
entry points (key remap + strict load into the ported architectures), the
imputers through `motionbench.imputers.FrameVAEACImputer.load` /
`FrameFlowImputer.load` (architecture read from the checkpoint's `arch`
dict; VAEAC `dec_trunk`/`dec_out` keys remapped on load).

## Provenance notes (release-notes item 5)

- **PTB-XL VAEAC** was retrained after the original checkpoint was found
  contaminated; the retrained model and its realism-gate record belong at
  `results/ptbxl_imputers/vaeac/vaeac_best.pt` + `results/imputer_validation.md`.
- **CARE-PD VAEAC** failed the realism gate with value **0.375** (< 0.5
  threshold); the checkpoint is shipped for reproducibility with that gate
  value on record, and results using it are scoped accordingly in the paper.
- The three checkpoints in `motionbench/classifiers/checkpoints/*.pt`
  (synthetic_mlp_k4 / synthetic_cnn / synthetic_transformer) are small
  release-format smoke-test artifacts from an earlier training run, kept so
  examples work offline; they are not the table-generating models.
