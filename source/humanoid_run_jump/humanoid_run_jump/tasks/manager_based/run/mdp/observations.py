# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for G1 run task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from humanoid_run_jump.robots.g1_constants import ANCHOR_BODY_NAME, ROOT_BODY_NAME
from humanoid_run_jump.robots.joint_order import JointOrderMap
from humanoid_run_jump.tracker.reduced_coords import (
    compute_reduced_coords_obs,
    quat_apply_inverse,
    quat_to_tan_norm,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _joint_map(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> JointOrderMap:
    """Reuse env-level map when present; otherwise build from the articulation."""
    cached = getattr(env, "joint_order_map", None)
    if cached is not None:
        return cached
    asset = env.scene[asset_cfg.name]
    return JointOrderMap(list(asset.data.joint_names), device=env.device)


def reduced_coords_proprio(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    anchor_body_name: str = ANCHOR_BODY_NAME,
    root_body_name: str = ROOT_BODY_NAME,
) -> torch.Tensor:
    """64-D reduced-coords proprioception matching the frozen tracker."""
    asset = env.scene[asset_cfg.name]
    jmap = _joint_map(env, asset_cfg)
    anchor_idx = asset.data.body_names.index(anchor_body_name)
    root_idx = asset.data.body_names.index(root_body_name)
    return compute_reduced_coords_obs(
        dof_pos=jmap.to_g1(asset.data.joint_pos),
        dof_vel=jmap.to_g1(asset.data.joint_vel),
        anchor_quat_wxyz=asset.data.body_quat_w[:, anchor_idx],
        root_ang_vel_w=asset.data.root_ang_vel_w,
        root_quat_wxyz=asset.data.body_quat_w[:, root_idx],
    )


def base_lin_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Root linear velocity in the body / root frame (m/s)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def velocity_commands(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Return the active velocity command ``[vx, vy, wz]``."""
    return env.command_manager.get_command(command_name)


def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Previous RL action (64-D target frame)."""
    return env.action_manager.action


def amp_obs_single(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    root_body_name: str = ROOT_BODY_NAME,
) -> torch.Tensor:
    """Single-frame AMP observation used by the discriminator.

    Layout::

        dof_pos | dof_vel | root_height | root_rot6d | root_lin_vel_b | root_ang_vel_b
        | key_body_pos_local (4 × 3, yaw-aligned relative to root)
    """
    from isaaclab.utils.math import quat_apply, yaw_quat
    from humanoid_run_jump.robots.g1_constants import AMP_KEY_BODY_NAMES

    asset = env.scene[asset_cfg.name]
    jmap = _joint_map(env, asset_cfg)
    root_idx = asset.data.body_names.index(root_body_name)
    body_names = list(asset.data.body_names)

    root_pos = asset.data.body_pos_w[:, root_idx]
    root_quat = asset.data.body_quat_w[:, root_idx]
    root_lin_vel_w = asset.data.root_lin_vel_w
    root_ang_vel_w = asset.data.root_ang_vel_w
    root_lin_vel_b = quat_apply_inverse(root_quat, root_lin_vel_w)
    root_ang_vel_b = quat_apply_inverse(root_quat, root_ang_vel_w)

    # Key bodies in root-yaw frame (matches packaged reference).
    q_yaw = yaw_quat(root_quat)
    q_inv = q_yaw.clone()
    q_inv[:, 3] = -q_inv[:, 3]
    key_locals = []
    for name in AMP_KEY_BODY_NAMES:
        try:
            idx = body_names.index(name)
            rel = asset.data.body_pos_w[:, idx] - root_pos
            key_locals.append(quat_apply(q_inv, rel))
        except ValueError:
            key_locals.append(torch.zeros(env.num_envs, 3, device=env.device))
    key_flat = torch.cat(key_locals, dim=-1)

    return torch.cat(
        [
            jmap.to_g1(asset.data.joint_pos),
            jmap.to_g1(asset.data.joint_vel),
            root_pos[:, 2:3],
            quat_to_tan_norm(root_quat),
            root_lin_vel_b,
            root_ang_vel_b,
            key_flat,
        ],
        dim=-1,
    )
