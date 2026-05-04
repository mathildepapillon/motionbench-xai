# TASK 3D — Grad-CAM and attention-based methods

**Phase:** 3 | **Tag:** [mechanical] | **Depends on:** 4B | **PR title:** `[3D] Grad-CAM and attention attribution`

## Worktree setup

```bash
git worktree add ../mbxai-task-3D-cam -b task/3D-cam
```

## Files to create

```
motionbench/attribution/grad_cam.py
motionbench/attribution/attention_rollout.py
tests/test_cam.py
```

## Spec

```python
class GradCAMAttributor(BaseAttributor):
    """Captum LayerGradCam wrapper. Requires the target layer as a constructor arg."""
    requires_gradient = True

class AttentionRolloutAttributor(BaseAttributor):
    """Attention rollout for transformer encoders (Abnar & Zuidema 2020)."""
```

Test on a tiny CNN with synthetic motion data (not on the CARE-PD classifiers, to keep tests fast).

## References

- Captum LayerGradCam: https://captum.ai/api/layer.html#captum.attr.LayerGradCam
- Abnar & Zuidema (2020) "Quantifying Attention Flow in Transformers." ACL.

## Definition of done

- [ ] Shape tests pass on tiny CNN
- [ ] ruff + mypy pass
- [ ] TASKS.md row 3D: done, notes
- [ ] PR: `[3D] Grad-CAM and attention attribution`
