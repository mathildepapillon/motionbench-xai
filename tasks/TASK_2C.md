# TASK 2C — Port MotionSHAP-VAEAC

**Phase:** 2 | **Tag:** [mechanical] | **PR title:** `[2C] MotionSHAP-VAEAC port`

## Worktree setup

```bash
git worktree add ../mbxai-task-2C-vaeac -b task/2C-vaeac
```

## Files to create

```
motionbench/imputers/vaeac.py
scripts/train_vaeac.py
tests/test_vaeac.py
```

## Source mapping

- **Port from:** `CARE-PD/model/vaeac/vaeac.py` (model), `heads.py` (prior/posterior encoder, decoder), `imputer.py` (imputer wrapper)
- **Training script from:** `CARE-PD/train_vaeac.py`
- **Reference:** `CARE-PD/model/gaitvae/` (alternate VAEAC variant; use as reference if it differs)

## Spec

### `motionbench.imputers.vaeac`

```python
class VAEACImputer(BaseImputer):
    """VAEAC (VAE with Arbitrary Conditioning) imputer.

    Architecture: prior encoder + posterior encoder + decoder.
    Amortised inference — no per-query optimisation.
    Reference: Ivanov et al. (2019) "VAEAC: Missing Data Imputation with VAEAC."
    """

    def fit(self, train_data: BaseDataset) -> "VAEACImputer":
        """Store dataset ref; actual training is done via scripts/train_vaeac.py."""

    def impute(self, x_obs, mask, n_samples, seed=None) -> Tensor:
        """Prior encoder → sample z → decoder. Observed entries overwritten post-decode."""

    def save(self, path: str | Path) -> None: ...

    @classmethod
    def load(cls, path: str | Path) -> "VAEACImputer": ...
```

`is_on_manifold = True`

### `scripts/train_vaeac.py`

Port training loop from `CARE-PD/train_vaeac.py`. Replace argparse with Hydra config:
```
configs/methods/train_vaeac.yaml
```
The same config must be usable for Task 2D's flow training (use a shared base config).

### Tests

1. `test_vaeac_smoke` — 2 epochs on synthetic Gaussian data (J=5, F=3, T=16); verify training loss decreases. Mark `@pytest.mark.slow`.
2. `test_vaeac_serialization` — save and load; verify `impute` output is identical before/after.
3. `test_vaeac_shape` — `impute` returns `(n_samples, J, F, T)`.
4. `test_vaeac_observed_preserved` — standard contract.

## Definition of done

- [ ] Tests pass
- [ ] ruff + mypy pass
- [ ] TASKS.md row 2C: `status: done`, notes
- [ ] PR: `[2C] MotionSHAP-VAEAC port`
