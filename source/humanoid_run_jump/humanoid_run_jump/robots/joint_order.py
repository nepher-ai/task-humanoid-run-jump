# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Map Isaac Lab USD joint order <-> canonical ``G1_JOINT_NAMES`` order.

The BeyondMimic tracker, CSV motions, and AMP packs all use ``G1_JOINT_NAMES``
(MJCF / bones-speed order). Isaac Lab's articulation uses the USD parse order,
which is a permutation of the same 29 joints. All sim <-> tracker / AMP I/O must
go through this remap.
"""

from __future__ import annotations

import torch

from humanoid_run_jump.robots.g1_constants import G1_JOINT_NAMES, NUM_JOINTS


class JointOrderMap:
    """Bidirectional index map between sim joint vectors and G1 order."""

    def __init__(self, sim_joint_names: list[str], device: torch.device | str = "cpu"):
        sim_names = list(sim_joint_names)
        expected = list(G1_JOINT_NAMES)
        if len(sim_names) != NUM_JOINTS:
            raise RuntimeError(
                f"Expected {NUM_JOINTS} joints, got {len(sim_names)}: {sim_names}"
            )
        if set(sim_names) != set(expected):
            raise RuntimeError(
                "Robot joint name set does not match G1_JOINT_NAMES.\n"
                f"  only in sim: {set(sim_names) - set(expected)}\n"
                f"  only in G1_JOINT_NAMES: {set(expected) - set(sim_names)}"
            )

        # g1_to_sim[i] = sim index of G1_JOINT_NAMES[i]
        g1_to_sim = [sim_names.index(n) for n in expected]
        # sim_to_g1[j] = G1 index of sim_names[j]
        sim_to_g1 = [expected.index(n) for n in sim_names]

        self.sim_joint_names = sim_names
        self.g1_joint_names = expected
        self.identity = g1_to_sim == list(range(NUM_JOINTS))
        device = torch.device(device)
        self.g1_to_sim = torch.tensor(g1_to_sim, device=device, dtype=torch.long)
        self.sim_to_g1 = torch.tensor(sim_to_g1, device=device, dtype=torch.long)

        if not self.identity:
            print(
                "[JointOrderMap] USD joint order differs from G1_JOINT_NAMES; "
                "remapping dof channels for tracker/AMP I/O."
            )

    def to_g1(self, dof_sim: torch.Tensor) -> torch.Tensor:
        """Permute last dim from sim order -> G1 order."""
        if self.identity:
            return dof_sim
        return dof_sim[..., self.g1_to_sim]

    def to_sim(self, dof_g1: torch.Tensor) -> torch.Tensor:
        """Permute last dim from G1 order -> sim order."""
        if self.identity:
            return dof_g1
        return dof_g1[..., self.sim_to_g1]
