"""Tests for CARE-PD ported classifiers and BMCLabDataset loader.

Run with:
    pytest tests/test_care_pd.py -q -m "not manual"

Manual / reproducibility tests (require downloaded checkpoints):
    pytest tests/test_care_pd.py -q -m manual
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Shape constants matching motionbench convention
# ---------------------------------------------------------------------------

B = 2        # batch size
J = 17       # joints (H36M-17)
FS = 3       # features (xyz)
T = 81       # frames (3 s @ 27 fps)
N_CLASSES = 4

_DEFAULT_CARE_PD = Path(__file__).resolve().parents[2] / "CARE-PD"
CHECKPOINT_ROOT = Path(
    os.environ.get(
        "CARE_PD_CHECKPOINTS",
        str(Path(os.environ.get("CARE_PD_ROOT", _DEFAULT_CARE_PD))
            / "assets" / "Pretrained_checkpoints"),
    )
)

# ---------------------------------------------------------------------------
# Lazy imports to avoid failing the whole test file if timm/torch_dct absent
# ---------------------------------------------------------------------------


def _import_poseformerv2():
    from motionbench.classifiers.ported_care_pd.poseformerv2 import PoseFormerV2Classifier
    return PoseFormerV2Classifier


def _import_motionbert():
    from motionbench.classifiers.ported_care_pd.motionbert import MotionBERTClassifier
    return MotionBERTClassifier


def _import_potr():
    from motionbench.classifiers.ported_care_pd.potr import POTRClassifier
    return POTRClassifier


def _import_motionagformer():
    from motionbench.classifiers.ported_care_pd.motionagformer import MotionAGFormerClassifier
    return MotionAGFormerClassifier


def _import_bilstm():
    from motionbench.classifiers.ported_care_pd.bilstm import BiLSTMClassifier
    return BiLSTMClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(b=B, j=J, f=FS, t=T) -> torch.Tensor:
    """Return a random ``(B, J, F, T)`` float32 tensor."""
    return torch.randn(b, j, f, t, dtype=torch.float32)


# ---------------------------------------------------------------------------
# test_forward_shapes — one test per classifier
# ---------------------------------------------------------------------------


class TestForwardShapesPoseFormerV2:
    """PoseFormerV2Classifier: (B, J, 2, T) → (B, n_classes).

    PoseFormerV2 is the only classifier in this suite that consumes 2-D
    image-projected pose coordinates (see motionbench/classifiers/base.py
    "Pose Format" reference); the fixture below therefore overrides the
    suite-wide ``FS=3`` to ``2``.
    """

    @pytest.fixture(scope="class")
    def model(self):
        PFC = _import_poseformerv2()
        m = PFC(checkpoint_path=None, n_classes=N_CLASSES)
        m.eval()
        return m

    def test_output_shape(self, model):
        x = _make_batch(f=2)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, N_CLASSES), f"Expected {(B, N_CLASSES)}, got {out.shape}"

    def test_output_dtype(self, model):
        x = _make_batch(f=2)
        with torch.no_grad():
            out = model(x)
        assert out.dtype == torch.float32

    def test_output_finite(self, model):
        x = _make_batch(f=2)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()


class TestForwardShapesMotionBERT:
    """MotionBERTClassifier: (B, J, F, T) → (B, n_classes)."""

    @pytest.fixture(scope="class")
    def model(self):
        MBC = _import_motionbert()
        m = MBC(checkpoint_path=None, n_classes=N_CLASSES)
        m.eval()
        return m

    def test_output_shape(self, model):
        x = _make_batch()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, N_CLASSES), f"Expected {(B, N_CLASSES)}, got {out.shape}"

    def test_output_dtype(self, model):
        x = _make_batch()
        with torch.no_grad():
            out = model(x)
        assert out.dtype == torch.float32

    def test_output_finite(self, model):
        x = _make_batch()
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()


class TestForwardShapesPOTR:
    """POTRClassifier: (B, J, F, T) → (B, n_classes).

    POTR's default ``source_seq_length=80`` (one frame shorter than the
    suite-wide ``T=81``); the fixtures below override the batch length
    accordingly.
    """

    @pytest.fixture(scope="class")
    def model(self):
        PC = _import_potr()
        m = PC(checkpoint_path=None, n_classes=N_CLASSES)
        m.eval()
        return m

    def test_output_shape(self, model):
        x = _make_batch(t=80)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, N_CLASSES), f"Expected {(B, N_CLASSES)}, got {out.shape}"

    def test_output_dtype(self, model):
        x = _make_batch(t=80)
        with torch.no_grad():
            out = model(x)
        assert out.dtype == torch.float32

    def test_output_finite(self, model):
        x = _make_batch(t=80)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()


class TestForwardShapesMotionAGFormer:
    """MotionAGFormerClassifier: (B, J, F, T) → (B, n_classes)."""

    @pytest.fixture(scope="class")
    def model(self):
        MAC = _import_motionagformer()
        # Use small model for fast tests
        m = MAC(
            checkpoint_path=None,
            n_classes=N_CLASSES,
            n_layers=2,
            dim_feat=16,
            dim_rep=32,
            num_heads=4,
        )
        m.eval()
        return m

    def test_output_shape(self, model):
        x = _make_batch()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, N_CLASSES), f"Expected {(B, N_CLASSES)}, got {out.shape}"

    def test_output_dtype(self, model):
        x = _make_batch()
        with torch.no_grad():
            out = model(x)
        assert out.dtype == torch.float32

    def test_output_finite(self, model):
        x = _make_batch()
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()


class TestForwardShapesBiLSTM:
    """BiLSTMClassifier: (B, J, F, T) → (B, n_classes)."""

    @pytest.fixture(scope="class")
    def model(self):
        BC = _import_bilstm()
        m = BC(checkpoint_path=None, n_classes=N_CLASSES)
        m.eval()
        return m

    def test_output_shape(self, model):
        x = _make_batch()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, N_CLASSES), f"Expected {(B, N_CLASSES)}, got {out.shape}"

    def test_output_dtype(self, model):
        x = _make_batch()
        with torch.no_grad():
            out = model(x)
        assert out.dtype == torch.float32

    def test_output_finite(self, model):
        x = _make_batch()
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# test_predict_proba — softmax outputs sum to ~1 per sample
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("clf_factory", [
    _import_poseformerv2,
    _import_motionbert,
    _import_potr,
    _import_bilstm,
])
def test_predict_proba(clf_factory):
    """After softmax, logits sum to ~1 per sample."""
    Clf = clf_factory()
    # Use smallest non-trivial config for speed
    if "PoseFormerV2" in Clf.__name__:
        model = Clf(n_classes=N_CLASSES)
    elif "MotionBERT" in Clf.__name__:
        model = Clf(n_classes=N_CLASSES)
    elif "POTR" in Clf.__name__:
        model = Clf(n_classes=N_CLASSES)
    else:
        model = Clf(n_classes=N_CLASSES)
    model.eval()

    if "PoseFormerV2" in Clf.__name__:
        x = _make_batch(f=2)
    elif "POTR" in Clf.__name__:
        x = _make_batch(t=80)
    else:
        x = _make_batch()
    with torch.no_grad():
        logits = model(x)
        proba = F.softmax(logits, dim=-1)

    sums = proba.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5), (
        f"Softmax sums not close to 1: {sums}"
    )


def test_predict_proba_motionagformer():
    """MotionAGFormer: softmax sums to ~1 per sample (small model)."""
    Clf = _import_motionagformer()
    model = Clf(
        n_classes=N_CLASSES,
        n_layers=2,
        dim_feat=16,
        dim_rep=32,
        num_heads=4,
    )
    model.eval()
    x = _make_batch()
    with torch.no_grad():
        logits = model(x)
        proba = F.softmax(logits, dim=-1)
    sums = proba.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5)


# ---------------------------------------------------------------------------
# test_care_pd_loader_shapes — BMCLabDataset returns correct shapes
# ---------------------------------------------------------------------------


def _make_dummy_npz(
    tmp_dir: Path,
    n_seqs: int = 5,
    t: int = 90,
    j: int = 17,
) -> Path:
    """Write a temporary .npz archive with dummy pose data."""
    data = {}
    for i in range(n_seqs):
        seq_name = f"S{i:02d}__walk01"
        # Shape: (T, J, 3)
        data[seq_name] = np.random.randn(t, j, 3).astype(np.float32)
    path = tmp_dir / "poses.npz"
    np.savez(str(path), **data)
    return path


def _make_dummy_labels(
    tmp_dir: Path,
    n_seqs: int = 5,
) -> Path:
    """Write a temporary labels joblib file."""
    import joblib
    labels = {}
    for i in range(n_seqs):
        sid = f"S{i:02d}"
        labels[sid] = {"walk01": {"UPDRS_GAIT": i % 4, "medication": "on", "other": 0}}
    path = tmp_dir / "labels.pkl"
    joblib.dump(labels, str(path))
    return path


class TestBMCLabDataset:
    """BMCLabDataset satisfies the BaseDataset protocol."""

    @pytest.fixture(scope="class")
    def dataset(self, tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp("care_pd")
        npz_path = _make_dummy_npz(tmp_dir)
        lbl_path = _make_dummy_labels(tmp_dir)
        from motionbench.data.real.care_pd import BMCLabDataset
        return BMCLabDataset(
            joints_paths=[str(npz_path)],
            labels_path=str(lbl_path),
            clip_len=81,
        )

    def test_len(self, dataset):
        assert len(dataset) == 5

    def test_getitem_x_shape(self, dataset):
        x, y = dataset[0]
        assert x.shape == (17, 3, 81), f"Expected (17, 3, 81), got {x.shape}"

    def test_getitem_x_dtype(self, dataset):
        x, y = dataset[0]
        assert x.dtype == torch.float32

    def test_getitem_y_dtype(self, dataset):
        x, y = dataset[0]
        assert y.dtype == torch.int64

    def test_getitem_y_range(self, dataset):
        for i in range(len(dataset)):
            _, y = dataset[i]
            assert 0 <= y.item() <= 3

    def test_shape_property(self, dataset):
        assert dataset.shape == (17, 3, 81)

    def test_metadata_keys(self, dataset):
        meta = dataset.metadata
        assert "skeleton" in meta
        assert "frame_rate" in meta
        assert meta["skeleton"] == "h36m_17"
        assert meta["frame_rate"] == 27.0

    def test_oracle_is_none(self, dataset):
        assert dataset.oracle is None

    def test_base_dataset_protocol(self, dataset):
        from motionbench.data.base import BaseDataset
        assert isinstance(dataset, BaseDataset)

    def test_clip_shorter_than_clip_len(self, tmp_path_factory):
        """Sequences shorter than clip_len are zero-padded."""
        tmp_dir = tmp_path_factory.mktemp("care_pd_short")
        # Shorter sequence (40 frames < 81)
        data = {"S00__walk01": np.random.randn(40, 17, 3).astype(np.float32)}
        npz_path = tmp_dir / "poses.npz"
        np.savez(str(npz_path), **data)

        import joblib
        labels = {"S00": {"walk01": {"UPDRS_GAIT": 1, "medication": "on", "other": 0}}}
        lbl_path = tmp_dir / "labels.pkl"
        joblib.dump(labels, str(lbl_path))

        from motionbench.data.real.care_pd import BMCLabDataset
        ds = BMCLabDataset([str(npz_path)], str(lbl_path), clip_len=81)
        x, y = ds[0]
        assert x.shape == (17, 3, 81)

    def test_clip_longer_than_clip_len(self, tmp_path_factory):
        """Sequences longer than clip_len are centre-cropped."""
        tmp_dir = tmp_path_factory.mktemp("care_pd_long")
        data = {"S00__walk01": np.random.randn(120, 17, 3).astype(np.float32)}
        npz_path = tmp_dir / "poses.npz"
        np.savez(str(npz_path), **data)

        import joblib
        labels = {"S00": {"walk01": {"UPDRS_GAIT": 2, "medication": "off", "other": 1}}}
        lbl_path = tmp_dir / "labels.pkl"
        joblib.dump(labels, str(lbl_path))

        from motionbench.data.real.care_pd import BMCLabDataset
        ds = BMCLabDataset([str(npz_path)], str(lbl_path), clip_len=81)
        x, y = ds[0]
        assert x.shape == (17, 3, 81)


# ---------------------------------------------------------------------------
# test_classifier_base — Classifier satisfies the nn.Module interface
# ---------------------------------------------------------------------------


def test_classifier_is_nn_module():
    """All classifiers are nn.Module subclasses."""
    models = [
        _import_bilstm()(n_classes=4),
        _import_motionbert()(n_classes=4),
        _import_potr()(n_classes=4),
    ]
    for m in models:
        assert isinstance(m, torch.nn.Module)


def test_classifier_n_classes_attribute():
    """n_classes attribute matches init argument."""
    BC = _import_bilstm()
    m = BC(n_classes=3)
    assert m.n_classes == 3


# ---------------------------------------------------------------------------
# test_reproducibility_* — manual tests (require checkpoints)
# ---------------------------------------------------------------------------


@pytest.mark.manual
class TestReproducibilityPoseFormerV2:
    """Load PoseFormerV2 checkpoint and verify F1 matches CARE-PD paper."""

    def test_checkpoint_loads(self):
        ckpt_path = CHECKPOINT_ROOT / "poseformerv2" / "9_81_46.0.bin"
        if not ckpt_path.exists():
            pytest.skip(f"Checkpoint not found: {ckpt_path}")
        PFC = _import_poseformerv2()
        model = PFC(checkpoint_path=str(ckpt_path), n_classes=4)
        model.eval()
        x = _make_batch()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, 4)

    def test_f1_within_threshold(self):
        """
        Placeholder for full CARE-PD test-split F1 check.

        CARE-PD paper reports (BMCLab, within-site LOSOCV, labels 0-2):
            PoseFormerV2: macro-F1 ≈ 0.37–0.55 depending on protocol.

        This test requires:
        1. CARE-PD BMCLab dataset files
        2. Fine-tuned classifier checkpoint (backbone+head, trained on CARE-PD)

        STATUS: BLOCKED — fine-tuned CARE-PD classifier checkpoints are not
        available in the repository.  Only backbone pre-training weights (H36M)
        are present.  See REPRODUCIBILITY.md §2.2 for the CARE-PD checkpoint acquisition steps.
        """
        pytest.skip(
            "Reproducibility gate BLOCKED: fine-tuned CARE-PD classifier "
            "checkpoints are not available. Only backbone pre-training weights "
            "(poseformerv2/9_81_46.0.bin, H36M) are present. "
            "Expected F1 (paper): ~0.54 (MIDA/BMCLab, labels 0-2). "
            "See REPRODUCIBILITY.md §2.2 for checkpoint acquisition."
        )


@pytest.mark.manual
class TestReproducibilityMotionBERT:
    """Load MotionBERT checkpoint and verify F1 matches CARE-PD paper."""

    def test_checkpoint_loads(self):
        ckpt_path = CHECKPOINT_ROOT / "motionbert" / "motionbert.bin"
        if not ckpt_path.exists():
            pytest.skip(f"Checkpoint not found: {ckpt_path}")
        MBC = _import_motionbert()
        model = MBC(checkpoint_path=str(ckpt_path), n_classes=4)
        model.eval()
        x = _make_batch()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, 4)

    def test_f1_within_threshold(self):
        """
        CARE-PD paper reports (BMCLab, MIDA, labels 0-2):
            MotionBERT: macro-F1 ≈ 0.45 (posterior view).

        STATUS: BLOCKED — see TestReproducibilityPoseFormerV2 for details.
        """
        pytest.skip(
            "Reproducibility gate BLOCKED: fine-tuned CARE-PD classifier "
            "checkpoints not available. Expected F1 (paper): ~0.45 (MIDA/BMCLab). "
            "See REPRODUCIBILITY.md §2.2 for checkpoint acquisition."
        )


@pytest.mark.manual
class TestReproducibilityPOTR:
    """Load POTR checkpoint and verify F1 matches CARE-PD paper."""

    def test_checkpoint_loads(self):
        ckpt_path = CHECKPOINT_ROOT / "potr" / "pre-trained_NTU_ckpt_epoch_199_enc_80_dec_20.pt"
        if not ckpt_path.exists():
            pytest.skip(f"Checkpoint not found: {ckpt_path}")
        PC = _import_potr()
        model = PC(checkpoint_path=str(ckpt_path), n_classes=4)
        model.eval()
        x = _make_batch()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, 4)

    def test_f1_within_threshold(self):
        """
        CARE-PD paper does not report POTR-specific F1 for BMCLab in Table D.1
        (only MixSTE, MotionAGFormer, MotionBERT, PoseFormerV2 are in the table).
        POTR is mentioned in the encoder list (Section 4) but its numeric results
        for UPDRS-gait F1 are shown in Fig. 2/3 only.

        STATUS: BLOCKED — fine-tuned CARE-PD classifier checkpoints not available.
        See REPRODUCIBILITY.md §2.2 for checkpoint acquisition.
        """
        pytest.skip(
            "Reproducibility gate BLOCKED: fine-tuned CARE-PD classifier "
            "checkpoints not available. See REPRODUCIBILITY.md §2.2 for checkpoint acquisition."
        )
