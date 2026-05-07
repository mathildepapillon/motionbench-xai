"""motionbench.classifiers.ported_ptbxl — ECG ResNet classifier for PTB-XL.

Exports
-------
ECGResNet1dClassifier
    1D ResNet-34 (Wang et al. 2017) adapted for 12-lead ECG classification.
    Architecture ported from the ``resnet1d_wang`` family benchmarked in
    Strodthoff et al. (2021) on PTB-XL.
"""

from motionbench.classifiers.ported_ptbxl.resnet1d import ECGResNet1dClassifier

__all__ = ["ECGResNet1dClassifier"]
