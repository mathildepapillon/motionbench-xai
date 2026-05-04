# TASK 3A — Captum-based attribution methods

**Phase:** 3 | **Tag:** [mechanical] | **PR title:** `[3A] Captum attribution methods`

## Worktree setup

```bash
git worktree add ../mbxai-task-3A-captum -b task/3A-captum
```

## Files to create

```
motionbench/attribution/captum_methods.py
tests/test_captum_methods.py
```

## Spec

Wrap each Captum method as a `BaseAttributor`. All wrappers:
- Aggregate `(J, F, T)` attribution map to `(M,)` via `players.aggregate(phi_coords)`.
- Use `tscaptum` for sequences longer than 200 frames.
- Accept `baseline` kwarg: `"zero"` | `"mean"` | `"gaussian"` | `"training_sample"`.

```python
class IntegratedGradientsAttributor(BaseAttributor):
    requires_gradient = True

class DeepLiftAttributor(BaseAttributor):
    requires_gradient = True

class GradientShapAttributor(BaseAttributor):
    requires_gradient = True

class SaliencyAttributor(BaseAttributor):
    requires_gradient = True

class SmoothGradAttributor(BaseAttributor):
    """Captum NoiseTunnel + Saliency."""
    requires_gradient = True

class InputXGradientAttributor(BaseAttributor):
    requires_gradient = True
```

### Tests

For each method: instantiate with a tiny differentiable model, call `attribute`, verify `(M,)` output shape. Tests must run on CPU in < 5 seconds each.

## References

- Captum documentation: https://captum.ai/docs/
- tsCaptum: https://github.com/josephenguehard/time_interpret

## Definition of done

- [ ] All 6 methods pass shape tests
- [ ] ruff + mypy pass
- [ ] TASKS.md row 3A: `status: done`, notes
- [ ] PR: `[3A] Captum attribution methods`
