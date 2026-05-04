"""motionbench.classifiers.ported_care_pd — CARE-PD encoder classifiers.

This package contains inference-only ports of the seven CARE-PD motion
encoders.  Each encoder wraps a backbone with a thin
:class:`~torch.nn.Linear` classification head.

Priority 0 (required):

* :class:`~motionbench.classifiers.ported_care_pd.poseformerv2.PoseFormerV2Classifier`
* :class:`~motionbench.classifiers.ported_care_pd.motionbert.MotionBERTClassifier`
* :class:`~motionbench.classifiers.ported_care_pd.potr.POTRClassifier`

Priority 1:

* :class:`~motionbench.classifiers.ported_care_pd.motionagformer.MotionAGFormerClassifier`
* :class:`~motionbench.classifiers.ported_care_pd.bilstm.BiLSTMClassifier`

Checkpoint URLs
---------------
Pre-trained backbone weights (H36M / NTU) — **not fine-tuned CARE-PD
classifiers** (see TASKS.md row 4B for reproducibility status):

.. code-block:: text

    poseformerv2:   CARE-PD/assets/Pretrained_checkpoints/poseformerv2/9_81_46.0.bin
    motionbert:     CARE-PD/assets/Pretrained_checkpoints/motionbert/motionbert.bin
    potr:           CARE-PD/assets/Pretrained_checkpoints/potr/
                        pre-trained_NTU_ckpt_epoch_199_enc_80_dec_20.pt
    motionagformer: CARE-PD/assets/Pretrained_checkpoints/motionagformer/
                        motionagformer-s-h36m.pth.tr
"""

from __future__ import annotations

from motionbench.classifiers.ported_care_pd.bilstm import BiLSTMClassifier
from motionbench.classifiers.ported_care_pd.motionagformer import MotionAGFormerClassifier
from motionbench.classifiers.ported_care_pd.motionbert import MotionBERTClassifier
from motionbench.classifiers.ported_care_pd.poseformerv2 import PoseFormerV2Classifier
from motionbench.classifiers.ported_care_pd.potr import POTRClassifier

__all__ = [
    "PoseFormerV2Classifier",
    "MotionBERTClassifier",
    "POTRClassifier",
    "MotionAGFormerClassifier",
    "BiLSTMClassifier",
]
