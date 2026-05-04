# TASK 4B — Port CARE-PD encoders + reproducibility check

**Phase:** 4 | **Tag:** [needs thinking, verify against literature] | **PR title:** `[4B] CARE-PD encoders + reproducibility check`

## Worktree setup

```bash
git worktree add ../mbxai-task-4B-care-pd-clf -b task/4B-care-pd-classifiers
```

## Files to create

```
motionbench/classifiers/ported_care_pd/poseformerv2.py
motionbench/classifiers/ported_care_pd/potr.py
motionbench/classifiers/ported_care_pd/motionbert.py
motionbench/classifiers/ported_care_pd/motionagformer.py  # P1
motionbench/classifiers/ported_care_pd/bilstm.py          # P2
motionbench/data/real/care_pd.py
tests/test_care_pd.py
```

## Source mapping

- **PoseFormerV2:** `CARE-PD/model/poseformerv2/model_poseformer.py`
- **POTR:** `CARE-PD/model/potr/` (all files)
- **MotionBERT:** `CARE-PD/model/motionbert/DSTformer.py` + `drop.py`
- **MotionAGFormer:** `CARE-PD/model/motionagformer/MotionAGFormer.py` + `modules/`
- **BiLSTM (P2):** `CARE-PD/model/bilstm/bilstm_encoder.py`
- **Backbone loading reference:** `CARE-PD/model/backbone_loader.py`
- **Data loaders:** `CARE-PD/data/bmclab_datareader.py` (primary), `CARE-PD/data/pdgam_datareader.py`

## Spec

### Data loader: `motionbench.data.real.care_pd`

Port the **inference-only** path from `BMCLabReader` and `PDGaMReader`:
- Strip all training augmentations.
- Expose `BaseDataset` interface: `__getitem__` returns `(x, y)` where `x: (J, F, T)` and `y` is UPDRS-gait class.
- Add `metadata["skeleton"] = "h36m_17"`, `metadata["frame_rate"] = 27.0` (or actual value).
- `oracle = None` (real data has no oracle).

### Classifier pattern

Each encoder must follow:
```python
class PoseFormerV2Classifier(Classifier):
    def __init__(self, checkpoint_path: str | None = None, n_classes: int = 4): ...
    def forward(self, x: Tensor) -> Tensor: ...  # (B, J, F, T) → (B, n_classes)
```

A thin linear `nn.Linear(encoder_dim, n_classes)` classification head is added
on top of the encoder's pooled embedding. The encoder weights come from the
published CARE-PD checkpoint; the head is fine-tuned (or randomly initialised
if no fine-tuned weights are available — document clearly).

**Checkpoint URLs go in `README.md`. Checkpoints are NOT committed.**

### Reproducibility gate (CRITICAL)

For each ported encoder, verify that loading the CARE-PD checkpoint reproduces
the CARE-PD paper's reported F1 **within 0.02** on at least one test cohort.

If reproducibility cannot be verified:
1. Set `status: blocked` in `TASKS.md`.
2. Document the discrepancy and suspected cause in `TASKS.md` notes.
3. Do NOT merge and do NOT silently use unvalidated encoders.

### Tests

1. `test_forward_shapes` — `(B, J, F, T) → (B, n_classes)` for B=2.
2. `test_predict_proba` — output sums to ~1 per sample.
3. `test_care_pd_loader` — `BMCLabDataset.__getitem__` returns correct shapes.
4. `test_reproducibility_*` — mark `@pytest.mark.manual`; run manually against downloaded checkpoints.

## Definition of done

- [ ] At least P0 classifiers (PoseFormerV2, POTR, MotionBERT) ported
- [ ] Reproducibility verified or blocked status set in TASKS.md
- [ ] ruff + mypy pass (note: ported_care_pd/ is excluded from strict mypy)
- [ ] TASKS.md row 4B: done or blocked, notes
- [ ] PR: `[4B] CARE-PD encoders + reproducibility check`
