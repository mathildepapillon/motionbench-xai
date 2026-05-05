"""motionbench.players.anatomical_groups — Anatomically-defined joint groups.

Joints are partitioned into clinically meaningful groups (e.g. left leg, right
leg, torso, arms).  Each group is one player.  Useful for reporting gait-XAI
results at the body-segment level rather than per-joint.

Predefined group schemas
-------------------------
``H36M_GROUPS`` — seven groups derived from the Human3.6M joint set (17 joints):
    root, spine, left_leg, right_leg, left_arm, right_arm, head.

``CARE_PD_GROUPS`` — clinical grouping matching CARE-PD paper annotation:
    lower_body (hips + knees + ankles + feet), upper_body (shoulders + elbows
    + wrists + hands), spine (root + neck + spine mid), head.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import Tensor

from motionbench.players.base import PlayerSet

__all__ = ["AnatomicalGroups", "H36M_GROUPS", "CARE_PD_GROUPS"]


# ---------------------------------------------------------------------------
# Predefined group schemas — joint indices follow H36M 17-joint convention
# ---------------------------------------------------------------------------

H36M_GROUPS: dict[str, list[int]] = {
    "root": [0],
    "spine": [7, 8],          # spine-mid, neck/nose
    "head": [9, 10],          # head, head-top
    "left_leg": [1, 2, 3],    # left hip, knee, ankle
    "right_leg": [4, 5, 6],   # right hip, knee, ankle
    "left_arm": [11, 12, 13], # left shoulder, elbow, wrist
    "right_arm": [14, 15, 16],# right shoulder, elbow, wrist
}

CARE_PD_GROUPS: dict[str, list[int]] = {
    "lower_body": [1, 2, 3, 4, 5, 6],       # hips, knees, ankles
    "upper_body": [11, 12, 13, 14, 15, 16],  # shoulders, elbows, wrists
    "spine": [0, 7, 8],                       # root, spine-mid, neck
    "head": [9, 10],                          # head, head-top
}


class AnatomicalGroups(PlayerSet):
    """Player set defined by a partition of joints into anatomical groups.

    Args:
        groups: Ordered dict mapping group name → list of joint indices.
            Every joint index in ``[0, J)`` must appear in exactly one group.
        J: Total number of joints.
        F: Number of features per joint.
        T: Number of time steps.

    Raises:
        ValueError: if the groups do not form a valid partition of ``[0, J)``.

    Example::

        players = AnatomicalGroups(H36M_GROUPS, J=17, F=3, T=81)
        # 7 players (one per anatomical group)
        phi = players.aggregate(phi_coords)  # (7,)
    """

    _group_names: list[str]
    _group_indices: list[list[int]]

    def __init__(
        self,
        groups: dict[str, list[int]],
        J: int,
        F: int,
        T: int,
    ) -> None:
        self._group_names = list(groups.keys())
        self._group_indices = [list(v) for v in groups.values()]
        self._J = J
        self._F = F
        self._T = T

        # Validate partition
        all_joints = sorted(j for idxs in self._group_indices for j in idxs)
        if all_joints != list(range(J)):
            missing = set(range(J)) - set(all_joints)
            overlap = [j for j in all_joints if all_joints.count(j) > 1]
            raise ValueError(
                f"groups must be a partition of [0, {J}). "
                f"Missing joints: {missing}. Overlapping joints: {set(overlap)}."
            )

    @property
    def n_players(self) -> int:
        return len(self._group_names)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._J, self._F, self._T

    @property
    def group_names(self) -> list[str]:
        """Ordered list of group names (matches player indices)."""
        return list(self._group_names)

    def coalition_mask(self, z: Tensor) -> Tensor:
        """Expand a coalition indicator to an element-level boolean mask.

        Args:
            z: ``(G,)`` binary tensor where G = number of groups.  1 = group observed.

        Returns:
            ``(J, F, T)`` bool tensor.

        Raises:
            ValueError: if ``z.shape != (G,)``.
        """
        G = len(self._group_names)
        if z.shape != (G,):
            raise ValueError(f"Expected z.shape==({G},); got {tuple(z.shape)}.")
        mask = torch.zeros(self._J, self._F, self._T, dtype=torch.bool)
        for g, joint_idxs in enumerate(self._group_indices):
            if z[g]:
                for j in joint_idxs:
                    mask[j, :, :] = True
        return mask

    def aggregate(self, phi_coords: Tensor) -> Tensor:
        """Aggregate per-coordinate attributions to per-group level.

        Args:
            phi_coords: ``(J, F, T)`` float tensor.

        Returns:
            ``(G,)`` float tensor — sum over all joints in each group.

        Raises:
            ValueError: if ``phi_coords.shape != (J, F, T)``.
        """
        if phi_coords.shape != (self._J, self._F, self._T):
            raise ValueError(
                f"Expected phi_coords.shape=={(self._J, self._F, self._T)}; "
                f"got {tuple(phi_coords.shape)}."
            )
        G = len(self._group_names)
        phi = torch.zeros(G, dtype=phi_coords.dtype)
        for g, joint_idxs in enumerate(self._group_indices):
            phi[g] = phi_coords[joint_idxs, :, :].sum()
        return phi
