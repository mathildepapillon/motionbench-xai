"""Tests for the frame-token imputer ports (frame_vaeac / frame_flow).

Small random-weight models on CPU: protocol shape/dtype contracts,
observed-entry preservation, seeded determinism, per-mask vs batched-mask
consistency, and reference-format checkpoint loading (including the
``dec_trunk`` / ``dec_out`` key remap of the real-data VAEAC checkpoints).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from motionbench.imputers import FrameFlowImputer, FrameVAEACImputer

J, F, T = 4, 2, 10


@pytest.fixture(scope="module")
def x_and_masks():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(J, F, T)).astype(np.float32)
    masks = np.stack(
        [
            np.zeros((J, F, T), bool),
            np.ones((J, F, T), bool),
            (np.arange(J) % 2 == 0)[:, None, None] & np.ones((J, F, T), bool),
            (np.arange(T) < 5)[None, None, :] & np.ones((J, F, T), bool),
        ]
    )
    return x, masks


@pytest.fixture(scope="module")
def vaeac():
    torch.manual_seed(7)
    imp = FrameVAEACImputer(J, F, T, d_model=16, d_latent=4, nhead=2, n_layers=1, ff=32)
    for p in imp.model.parameters():
        p.data.normal_(0.0, 0.05)
    imp.model.eval()
    return imp


@pytest.fixture(scope="module")
def flow():
    torch.manual_seed(7)
    imp = FrameFlowImputer(J, F, T, num_steps=4, d_model=16, nhead=2, n_layers=1, ff=32, time_dim=8)
    for p in imp.net.parameters():
        p.data.normal_(0.0, 0.05)
    imp.net.eval()
    return imp


@pytest.mark.parametrize("kind", ["vaeac", "flow"])
class TestProtocol:
    def _imp(self, kind, vaeac, flow):
        return vaeac if kind == "vaeac" else flow

    def test_impute_shape_dtype(self, kind, vaeac, flow, x_and_masks):
        x, masks = x_and_masks
        out = self._imp(kind, vaeac, flow).impute(x, masks[2], 3, np.random.default_rng(1))
        assert out.shape == (3, J, F, T)
        assert out.dtype == np.float32

    def test_impute_multi_shape(self, kind, vaeac, flow, x_and_masks):
        x, masks = x_and_masks
        out = self._imp(kind, vaeac, flow).impute_multi(x, masks, 2, np.random.default_rng(1))
        assert out.shape == (len(masks), 2, J, F, T)

    def test_observed_entries_preserved(self, kind, vaeac, flow, x_and_masks):
        x, masks = x_and_masks
        out = self._imp(kind, vaeac, flow).impute_multi(x, masks, 2, np.random.default_rng(1))
        for i, mask in enumerate(masks):
            assert np.array_equal(out[i][:, mask], np.broadcast_to(x[mask], (2, mask.sum())))

    def test_hidden_entries_are_not_copies(self, kind, vaeac, flow, x_and_masks):
        x, masks = x_and_masks
        out = self._imp(kind, vaeac, flow).impute_multi(x, masks, 1, np.random.default_rng(1))
        hidden = ~masks[2]
        assert not np.allclose(out[2][0][hidden], x[hidden])

    def test_seeded_determinism(self, kind, vaeac, flow, x_and_masks):
        x, masks = x_and_masks
        imp = self._imp(kind, vaeac, flow)
        o1 = imp.impute_multi(x, masks, 2, np.random.default_rng([1104, 1, 0]))
        o2 = imp.impute_multi(x, masks, 2, np.random.default_rng([1104, 1, 0]))
        assert np.array_equal(o1, o2)
        o3 = imp.impute_multi(x, masks, 2, np.random.default_rng([1104, 1, 1]))
        assert not np.array_equal(o1, o3)

    def test_impute_multi_matches_impute_for_single_mask(self, kind, vaeac, flow, x_and_masks):
        """One mask, n=1: the batched path is the per-mask path exactly."""
        x, masks = x_and_masks
        imp = self._imp(kind, vaeac, flow)
        o_multi = imp.impute_multi(x, masks[2:3], 1, np.random.default_rng(5))
        o_single = imp.impute(x, masks[2], 1, np.random.default_rng(5))
        assert np.array_equal(o_multi[0], o_single)


class TestCheckpointLoading:
    def test_vaeac_reference_format_with_dec_remap(self, vaeac, x_and_masks, tmp_path):
        x, masks = x_and_masks
        sd = {}
        for k, v in vaeac.model.state_dict().items():
            if k.startswith("decoder.trunk."):
                sd["dec_trunk." + k[len("decoder.trunk.") :]] = v
            elif k.startswith("decoder.out."):
                sd["dec_out." + k[len("decoder.out.") :]] = v
            else:
                sd[k] = v
        path = tmp_path / "vaeac.pt"
        torch.save(
            {
                "state_dict": sd,
                "shape": (J, F, T),
                "arch": {
                    "J": J,
                    "F": F,
                    "d_model": 16,
                    "nhead": 2,
                    "num_layers": 1,
                    "ff": 32,
                    "d_latent": 4,
                },
            },
            path,
        )
        loaded = FrameVAEACImputer.load(path)
        o1 = vaeac.impute_multi(x, masks, 1, np.random.default_rng(3))
        o2 = loaded.impute_multi(x, masks, 1, np.random.default_rng(3))
        assert np.array_equal(o1, o2)

    def test_flow_reference_format(self, flow, x_and_masks, tmp_path):
        x, masks = x_and_masks
        path = tmp_path / "flow.pt"
        torch.save(
            {
                "state_dict": flow.net.state_dict(),
                "shape": (J, F, T),
                "num_steps": 4,
                "arch": {
                    "J": J,
                    "F": F,
                    "d_model": 16,
                    "nhead": 2,
                    "num_layers": 1,
                    "ff": 32,
                    "time_dim": 8,
                },
            },
            path,
        )
        loaded = FrameFlowImputer.load(path)
        assert loaded.num_steps == 4
        o1 = flow.impute_multi(x, masks, 1, np.random.default_rng(3))
        o2 = loaded.impute_multi(x, masks, 1, np.random.default_rng(3))
        assert np.array_equal(o1, o2)

    def test_vaeac_arch_mismatch_fails_loudly(self, vaeac, tmp_path):
        path = tmp_path / "vaeac_bad.pt"
        torch.save(
            {"state_dict": vaeac.model.state_dict(), "shape": (J, F, T)},  # default arch
            path,
        )
        with pytest.raises(RuntimeError):
            FrameVAEACImputer.load(path)
