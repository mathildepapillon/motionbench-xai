# TASK 2D — Port MotionSHAP-Flow + M=10 Burr regression investigation

**Phase:** 2 | **Tag:** [needs thinking, verify against literature] | **PR title:** `[2D] MotionSHAP-Flow port + M=10 ablation`

## Worktree setup

```bash
git worktree add ../mbxai-task-2D-flow -b task/2D-flow
```

## Files to create

```
motionbench/imputers/flow_matching.py
scripts/train_flow.py
tests/test_flow.py
```

## Source mapping

- **Port from:** `CARE-PD/model/flow_matching/velocity_net.py` (velocity field network)
- **Port from:** `CARE-PD/model/flow_shap/imputer.py` (`FlowImputer`) — RePaint harmonisation
- **Training script from:** `CARE-PD/train_flow_matching.py`
- **Reference:** `CARE-PD/model/flow_shap/diagnostics.py` for ablation tooling

## Spec

### `motionbench.imputers.flow_matching`

```python
class FlowMatchingImputer(BaseImputer):
    """OT-Flow / flow-matching conditional imputer with RePaint harmonisation.

    Samples approximately from p_θ(x_hid | x_obs) by:
    1. Initialise x_t=1 from Gaussian noise.
    2. Integrate the learned velocity field v_θ(x, t) backwards (t: 1→0).
    3. At each step, project observed coordinates back to x_obs (RePaint).

    Default solver: midpoint, num_steps=100.
    Reference: Lipman et al. (2023) "Flow Matching for Generative Modeling."
    Lugmayr et al. (2022) "RePaint: Inpainting using Denoising Diffusion
    Probabilistic Models." (RePaint harmonisation technique)
    """

    def fit(self, train_data: BaseDataset) -> "FlowMatchingImputer": ...
    def impute(self, x_obs, mask, n_samples, seed=None) -> Tensor: ...
    def save(self, path): ...

    @classmethod
    def load(cls, path) -> "FlowMatchingImputer": ...
```

`is_on_manifold = True`

### Critical sub-task: M=10 Burr regression

In the CARE-PD experiments, EC2 for Flow jumps to 0.269 at M=10 Burr (vs 0.072 for VAEAC).
You **must** investigate and document. Hypotheses:

1. ODE integration step count too low for heavy-tailed targets.
2. OT path with Gaussian noise init is poorly conditioned for Burr-XII marginals.
3. RePaint harmonisation rule needs adjustment when many coordinates are observed.

Write a small ablation in `tests/test_flow.py` (marked `@pytest.mark.manual`) that:
- Varies `num_steps` ∈ {10, 50, 100, 500}.
- Varies `noise_init_scale` ∈ {0.5, 1.0, 2.0}.
- Reports EC2 on a 50-sample Burr M=10 test set.

Document findings (even null results) in the module docstring and link to relevant papers.

### `scripts/train_flow.py`

Same structure as `scripts/train_vaeac.py`. Hydra config: `configs/methods/train_flow.yaml`.

### Tests

1. `test_flow_smoke` — 2 epochs on synthetic data, loss decreases. `@pytest.mark.slow`.
2. `test_flow_serialization` — save/load round trip.
3. `test_flow_shape` — output shape.
4. `test_flow_observed_preserved` — standard contract.
5. `test_flow_m10_burr_ablation` — M=10 ablation. `@pytest.mark.manual`.

## References

- Lipman et al. (2023) "Flow Matching for Generative Modeling." arXiv:2210.02747.
- Tong et al. (2024) "Improving and Generalizing Flow-Matching." arXiv:2302.00482.
- Lugmayr et al. (2022) "RePaint." CVPR.

## Definition of done

- [ ] Tests pass (non-manual)
- [ ] M=10 ablation documented (findings in docstring, even if inconclusive)
- [ ] ruff + mypy pass
- [ ] TASKS.md row 2D: `status: done`, notes
- [ ] PR: `[2D] MotionSHAP-Flow port + M=10 ablation`
