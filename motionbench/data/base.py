"""motionbench.data.base — Dataset protocols and base classes.

Two protocols are defined:

* :class:`BaseDataset` — any dataset (synthetic or real); ``oracle`` is
  ``Optional`` and is ``None`` for real-world data.
* :class:`GroundTruthDataset` — a synthetic dataset with a known oracle;
  ``oracle`` is required and non-``None``.

Using ``Protocol`` instead of ``ABC`` lets real-world loaders (e.g.
PyTorch ``Dataset`` subclasses) satisfy the interface via structural
subtyping without inheriting from motionbench base classes.

Shape conventions
-----------------
All datasets return sequences in the layout ``(J, F, T)`` where:

* J — number of skeletal joints (e.g. 17 for Human3.6M).
* F — number of coordinates per joint (e.g. 3 for xyz).
* T — number of time-steps per clip (e.g. 81 frames at 27 fps = 3 s).

The ``metadata`` dict always contains at minimum:

``skeleton``
    String identifier for the skeleton topology (e.g. ``"h36m_17"``).
``frame_rate``
    Clip frame rate in Hz (float).

Example
-------
>>> class TinyDataset:
...     def __getitem__(self, idx):
...         return torch.zeros(17, 3, 81), torch.tensor(0)
...     def __len__(self):
...         return 10
...     @property
...     def shape(self):
...         return (17, 3, 81)
...     @property
...     def metadata(self):
...         return {"skeleton": "h36m_17", "frame_rate": 27.0}
...     @property
...     def oracle(self):
...         return None
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, runtime_checkable

import torch
from torch import Tensor

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore[assignment]

if TYPE_CHECKING:
    from motionbench.oracles.base import Oracle


__all__ = ["BaseDataset", "GroundTruthDataset"]


@runtime_checkable
class BaseDataset(Protocol):
    """Structural protocol for all motionbench datasets.

    Any object that implements the four members below is a valid
    ``BaseDataset`` — there is no need to inherit from this class.

    Item format
    -----------
    ``__getitem__`` returns ``(x, y)`` where:

    * ``x`` — ``(J, F, T)`` float32 Tensor.
    * ``y`` — scalar int64 Tensor (class label) or float32 Tensor (regression target).
    """

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return (x, y) for sample at index *idx*.

        Args:
            idx: Sample index in ``[0, len(self))``.

        Returns:
            ``x``: ``(J, F, T)`` float32 motion tensor.
            ``y``: scalar label tensor.
        """
        ...

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        ...

    @property
    def shape(self) -> tuple[int, int, int]:
        """Spatial shape of every sample as ``(J, F, T)``."""
        ...

    @property
    def metadata(self) -> dict[str, object]:
        """Dataset-level metadata dict.

        Required keys:

        * ``"skeleton"`` — skeleton topology identifier string.
        * ``"frame_rate"`` — clip frame rate in Hz (float).

        Additional keys (e.g. ``"cohort"``, ``"n_classes"``) are allowed.
        """
        ...

    @property
    def oracle(self) -> Optional["Oracle"]:
        """Ground-truth oracle for this dataset, or ``None`` for real data."""
        ...


@runtime_checkable
class GroundTruthDataset(Protocol):
    """Structural protocol for synthetic datasets with a known oracle.

    Identical to :class:`BaseDataset` except that :py:attr:`oracle` is
    required and guaranteed to be non-``None``.  Agents writing metric
    evaluation code should type-annotate with ``GroundTruthDataset`` when
    an oracle is required; mypy will enforce the non-``Optional`` constraint.
    """

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]: ...
    def __len__(self) -> int: ...

    @property
    def shape(self) -> tuple[int, int, int]: ...

    @property
    def metadata(self) -> dict[str, object]: ...

    @property
    def oracle(self) -> "Oracle":
        """Ground-truth oracle (guaranteed non-None for synthetic datasets)."""
        ...
