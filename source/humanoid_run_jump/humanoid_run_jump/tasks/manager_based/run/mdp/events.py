# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event / reset helpers for run environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _yaw_from_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Extract Z-yaw (rad) from a batch of wxyz quaternions."""
    w, x, y, z = quat_wxyz.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _apply_yaw_wxyz(quat_wxyz: torch.Tensor, dyaw: torch.Tensor) -> torch.Tensor:
    """Left-multiply ``quat_wxyz`` by a pure Z yaw of ``dyaw`` (radians).

    Returns a **new** ``(N, 4)`` wxyz tensor. Must not write through ``unbind``
    views into ``quat_wxyz`` — in-place assignment corrupts the last component
    because ``w0`` aliases ``quat[:, 0]`` / ``pose[:, 3]``.
    """
    half = dyaw * 0.5
    qw = torch.cos(half)
    qz = torch.sin(half)
    w0, x0, y0, z0 = quat_wxyz.unbind(-1)
    return torch.stack(
        [
            qw * w0 - qz * z0,
            qw * x0 + qz * y0,
            qw * y0 - qz * x0,
            qw * z0 + qz * w0,
        ],
        dim=-1,
    )


def reset_root_and_joints(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    joint_position_range: tuple[float, float] = (0.9, 1.1),
):
    """Reset root pose / velocity and joint state with light randomization."""
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)

    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]

    if pose_range is None:
        pose_range = {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.2, 0.2)}
    ranges = pose_range
    root_state[:, 0] += sample_uniform(*ranges.get("x", (0.0, 0.0)), n, env.device)
    root_state[:, 1] += sample_uniform(*ranges.get("y", (0.0, 0.0)), n, env.device)
    yaw = sample_uniform(*ranges.get("yaw", (0.0, 0.0)), n, env.device)
    # Multiply default * yaw about Z (wxyz); assign a fresh quat (no in-place unbind).
    root_state[:, 3:7] = _apply_yaw_wxyz(root_state[:, 3:7], yaw)

    if velocity_range is None:
        velocity_range = {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.1, 0.1)}
    root_state[:, 7] = sample_uniform(*velocity_range.get("x", (0.0, 0.0)), n, env.device)
    root_state[:, 8] = sample_uniform(*velocity_range.get("y", (0.0, 0.0)), n, env.device)
    root_state[:, 9] = sample_uniform(*velocity_range.get("z", (0.0, 0.0)), n, env.device)
    root_state[:, 10:13] = 0.0

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    scale = sample_uniform(*joint_position_range, (n, joint_pos.shape[1]), env.device)
    joint_pos *= scale

    asset.write_root_pose_to_sim(root_state[:, :7], env_ids)
    asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


def sample_uniform(
    low: float, high: float, size, device
) -> torch.Tensor:
    if isinstance(size, int):
        size = (size,)
    return torch.empty(size, device=device).uniform_(low, high)
