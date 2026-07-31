# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for G1 run/jump tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
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


def jump_commands(env: ManagerBasedRLEnv, command_name: str = "jump") -> torch.Tensor:
    """Jump command plus hand-off / apex / phase features.

    Layout::

        [h_obstacle, flight_distance, heading_error, progress,
         v_x*, v_z*,
         apex_rise*, push_fold*, swing_ext*, pitch_apex*,
         peak_pelvis_rise_so_far, tuck_l, tuck_r,
         phase_one_hot(7), has_liftoff, has_landed]
    """
    term = env.command_manager.get_term(command_name)
    cmd = env.command_manager.get_command(command_name)
    peak_rise = getattr(env, "_ep_peak_rise", None)
    if peak_rise is None:
        peak_rise = torch.zeros(env.num_envs, device=env.device)
    tuck_l = getattr(env, "_ep_tuck_l", None)
    if tuck_l is None:
        tuck_l = torch.zeros(env.num_envs, device=env.device)
    tuck_r = getattr(env, "_ep_tuck_r", None)
    if tuck_r is None:
        tuck_r = torch.zeros(env.num_envs, device=env.device)

    phase = getattr(env, "_ep_phase", None)
    if phase is None:
        phase_oh = torch.zeros(env.num_envs, 7, device=env.device)
        phase_oh[:, 0] = 1.0
    else:
        phase_oh = torch.nn.functional.one_hot(phase.clamp(0, 6), num_classes=7).float()

    has_liftoff = getattr(env, "_ep_has_liftoff", None)
    if has_liftoff is None:
        has_liftoff = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    has_landed = getattr(env, "_ep_has_landed", None)
    if has_landed is None:
        has_landed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    return torch.cat(
        [
            cmd,
            term.heading_error.unsqueeze(-1),
            term.progress.unsqueeze(-1),
            term.v_x_star.unsqueeze(-1),
            term.v_z_star.unsqueeze(-1),
            term.apex_rise_star.unsqueeze(-1),
            term.push_fold_star.unsqueeze(-1),
            term.swing_ext_star.unsqueeze(-1),
            term.pitch_apex_star.unsqueeze(-1),
            peak_rise.unsqueeze(-1),
            tuck_l.unsqueeze(-1),
            tuck_r.unsqueeze(-1),
            phase_oh,
            has_liftoff.float().unsqueeze(-1),
            has_landed.float().unsqueeze(-1),
        ],
        dim=-1,
    )


def foot_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    """Per-foot contact flags (1 = in contact) for takeoff / landing timing."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return (torch.norm(forces, dim=-1) > contact_force_threshold).float()


def foot_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    max_air_time: float = 1.0,
) -> torch.Tensor:
    """Per-foot air time normalized by ``max_air_time`` (requires track_air_time)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    return (air / max(max_air_time, 1e-3)).clamp(0.0, 1.0)


def base_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Root height above ground (m)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2:3]


def base_lin_vel_z(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """World-frame vertical root velocity (m/s)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w[:, 2:3]


def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
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


def next_obstacle_features(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Height, thickness, orientation of the next obstacle (course env)."""
    if not hasattr(env, "obstacle_features"):
        return torch.zeros(env.num_envs, 3, device=env.device)
    return env.obstacle_features
