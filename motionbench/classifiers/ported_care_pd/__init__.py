"""motionbench.classifiers.ported_care_pd — CARE-PD encoder classifiers.

This package contains inference-only ports of the CARE-PD motion encoders.
Each encoder wraps a backbone with a thin :class:`~torch.nn.Linear`
classification head.  All classifiers apply their model-specific
preprocessing *inside* ``forward`` so that XAI attributions remain in raw
coordinate space.

Active classifiers
------------------
* :class:`~motionbench.classifiers.ported_care_pd.motionbert.MotionBERTClassifier`
  — 3-D world-to-camera input, crop_scale + confidence preprocessing.
* :class:`~motionbench.classifiers.ported_care_pd.motionagformer.MotionAGFormerClassifier`
  — 3-D world-to-camera input, crop_scale + confidence preprocessing.
* :class:`~motionbench.classifiers.ported_care_pd.poseformerv2.PoseFormerV2Classifier`
  — 2-D image-projected pixel input, screen-normalisation preprocessing.
* :class:`~motionbench.classifiers.ported_care_pd.potr.POTRClassifier`
  — 3-D world-to-camera input, centre + z-score preprocessing (PENDING).

Checkpoint locations (fine-tuned, fold-1)
-----------------------------------------
.. code-block:: text

    motionbench/classifiers/checkpoints/care_pd/
        motionbert_bmclab_fold1.pth.tr
        motionagformer_bmclab_fold1.pth.tr
        poseformerv2_bmclab_fold1.pth.tr
"""

from __future__ import annotations

from motionbench.classifiers.ported_care_pd.motionagformer import MotionAGFormerClassifier
from motionbench.classifiers.ported_care_pd.motionbert import MotionBERTClassifier
from motionbench.classifiers.ported_care_pd.poseformerv2 import PoseFormerV2Classifier
from motionbench.classifiers.ported_care_pd.potr import POTRClassifier

__all__ = [
    "PoseFormerV2Classifier",
    "MotionBERTClassifier",
    "POTRClassifier",
    "MotionAGFormerClassifier",
]
