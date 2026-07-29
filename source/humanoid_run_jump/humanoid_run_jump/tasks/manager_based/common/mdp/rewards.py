# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for run and jump environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from humanoid_run_jump.tracker.reduced_coords import quat_apply_inverse
from humanoid_run_jump.tasks.manager_based.common.mdp.gait import (
    foot_contact_mask,
    lead_foot_forward_mask,
    resolve_ankle_body_ids,
)
from humanoid_run_jump.tasks.manager_based.common.mdp.jump_envelope import (
    CLEARANCE_EXCESS_MARGIN,
    FOOT_SEP_MAX_APEX,
    FOOT_SEP_MAX_FLIGHT,
    TUCK_EXTENDED,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Run rewards
# ---------------------------------------------------------------------------


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    vel_b = quat_apply_inverse(asset.data.root_quat_w, asset.data.root_lin_vel_w)[:, :2]
    err = torch.sum(torch.square(cmd - vel_b), dim=1)
    return torch.exp(-err / (std**2))


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, 2]
    ang_b = quat_apply_inverse(asset.data.root_quat_w, asset.data.root_ang_vel_w)[:, 2]
    err = torch.square(cmd - ang_b)
    return torch.exp(-err / (std**2))


def alive_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def flat_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    proj = asset.data.projected_gravity_b
    return torch.sum(torch.square(proj[:, :2]), dim=1)


def lin_vel_z_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_w[:, 2])


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


# ---------------------------------------------------------------------------
# Jump helpers
# ---------------------------------------------------------------------------


def _jump_term(env: ManagerBasedRLEnv, command_name: str):
    return env.command_manager.get_term(command_name)


def _both_feet_air(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return (torch.norm(forces, dim=-1) <= contact_force_threshold).all(dim=-1)


def _pitch_rate_b(asset) -> torch.Tensor:
    ang_b = quat_apply_inverse(asset.data.root_quat_w, asset.data.root_ang_vel_w)
    return ang_b[:, 1]


def _tilt_from_upright(asset) -> torch.Tensor:
    cos_tilt = (-asset.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)
    return torch.acos(cos_tilt)


def _forward_pitch_b(asset) -> torch.Tensor:
    """Forward lean angle (rad): positive = chest toward body +x.

    Uses ``atan2(-g_b_x, -g_b_z)`` so CSV jump apex leans are positive (~0.15 rad).
    """
    g = asset.data.projected_gravity_b
    return torch.atan2(-g[:, 0], -g[:, 2])


def _clean_takeoff_scale(asset, omega0: float = 1.0) -> torch.Tensor:
    wp = _pitch_rate_b(asset)
    return torch.exp(-torch.square(wp / max(omega0, 1e-3)))


def _ep_fresh_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    ep_len = getattr(env, "episode_length_buf", None)
    if ep_len is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return ep_len <= 1


def _ep_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    phase = getattr(env, "_ep_phase", None)
    if phase is None:
        return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return phase


def _per_leg_tuck(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-leg pelvis→ankle 3D distances ``(d_l, d_r)``."""
    tuck_l = getattr(env, "_ep_tuck_l", None)
    tuck_r = getattr(env, "_ep_tuck_r", None)
    if tuck_l is not None and tuck_r is not None:
        return tuck_l, tuck_r
    asset = env.scene[asset_cfg.name]
    left_id, right_id = resolve_ankle_body_ids(env, asset_cfg.name)
    root = asset.data.root_pos_w
    d_l = torch.linalg.norm(root - asset.data.body_pos_w[:, left_id], dim=-1)
    d_r = torch.linalg.norm(root - asset.data.body_pos_w[:, right_id], dim=-1)
    return d_l, d_r


def _foot_sep(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    left_id, right_id = resolve_ankle_body_ids(env, asset_cfg.name)
    return torch.linalg.norm(
        asset.data.body_pos_w[:, left_id] - asset.data.body_pos_w[:, right_id], dim=-1
    )


# ---------------------------------------------------------------------------
# Jump dense shaping
# ---------------------------------------------------------------------------


def takeoff_launch(
    env: ManagerBasedRLEnv,
    command_name: str = "jump",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    contact_force_threshold: float = 5.0,
    vel_std: float = 1.0,
    vz_overshoot: float = 0.15,
    pitch_std: float = 0.12,
    clean_omega0: float = 2.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Launch shaping: envelope velocity + forward pitch (no upright bias).

    * PLANT/PUSH while left-supported: track ``(v_x*, v_z*)`` with asymmetric
      ``v_z`` (overshoot above ``v_z* + vz_overshoot`` is penalized) and forward
      pitch toward ``pitch_takeoff*``.
    * RISE while ascending: same with pitch-rate cleanliness.
    """
    phase = _ep_phase(env)
    in_pp = (phase == 1) | (phase == 2)
    in_rise = phase == 3
    asset = env.scene[asset_cfg.name]
    term = _jump_term(env, command_name)
    both_air = _both_feet_air(env, sensor_cfg, contact_force_threshold)
    contacts = foot_contact_mask(
        env, sensor_cfg=sensor_cfg, contact_force_threshold=contact_force_threshold
    )
    left_c = contacts[:, 0]

    target_v = term.target_vel_b
    vel_b = quat_apply_inverse(asset.data.root_quat_w, asset.data.root_lin_vel_w)
    # Asymmetric vz: treat overshoot as larger error so ballistic fly is not free.
    vz_err = vel_b[:, 2] - target_v[:, 2]
    vz_err = torch.where(vz_err > vz_overshoot, vz_err + (vz_err - vz_overshoot), vz_err)
    err_xy = torch.sum(torch.square(vel_b[:, :2] - target_v[:, :2]), dim=1)
    err = err_xy + torch.square(vz_err)
    vel_track = torch.exp(-err / max(vel_std, 1e-3))
    ascending = (asset.data.root_lin_vel_w[:, 2] > 0.0).float()
    clean = _clean_takeoff_scale(asset, clean_omega0)
    pitch = _forward_pitch_b(asset)
    pitch_err = pitch - term.pitch_takeoff_star
    pitch_track = torch.exp(-torch.square(pitch_err / max(pitch_std, 1e-3)))

    grounded = (
        in_pp.float() * left_c.float() * (~both_air).float() * vel_track * pitch_track
    )
    air = (
        in_rise.float()
        * both_air.float()
        * ascending
        * clean
        * vel_track
        * pitch_track
    )
    return grounded + air


def takeoff_foot(
    env: ManagerBasedRLEnv,
    once_per_episode: bool = True,
) -> torch.Tensor:
    """Sparse +1 for left-foot takeoff, once per episode at first liftoff.

    Right-foot takeoff is an ``illegal_takeoff`` termination, so only left
    credit is paid here.
    """
    valid = getattr(env, "_ep_takeoff_foot_valid", None)
    foot = getattr(env, "_ep_takeoff_foot", None)
    if valid is None or foot is None:
        return torch.zeros(env.num_envs, device=env.device)

    if not hasattr(env, "_jump_takeoff_foot_given"):
        env._jump_takeoff_foot_given = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    fresh = _ep_fresh_mask(env)
    env._jump_takeoff_foot_given = env._jump_takeoff_foot_given & (~fresh)

    hit = valid & (foot == 0) & (~env._jump_takeoff_foot_given)
    if once_per_episode:
        env._jump_takeoff_foot_given = env._jump_takeoff_foot_given | hit
    return hit.float()


def prep_step(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    contact_force_threshold: float = 5.0,
    min_lead_m: float = 0.04,
    air_penalty: float = 1.0,
) -> torch.Tensor:
    """PREP phase: left running step while staying grounded (not a hop)."""
    phase = _ep_phase(env)
    in_prep = phase == 0
    contacts = foot_contact_mask(
        env, sensor_cfg=sensor_cfg, contact_force_threshold=contact_force_threshold
    )
    left_c = contacts[:, 0]
    right_c = contacts[:, 1]
    both_air = (~left_c) & (~right_c)
    left_fwd = lead_foot_forward_mask(env, lead="left", min_lead_m=min_lead_m)
    grounded = left_c | right_c
    swing = in_prep & left_fwd & grounded
    # Non-sticky: only penalize the current airborne frame (prep_timeout handles dead-ends).
    air_pen = in_prep & both_air
    return in_prep.float() * (0.80 * swing.float() - air_penalty * air_pen.float())


def plant_push(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    contact_force_threshold: float = 5.0,
    min_lead_m: float = 0.04,
    push_vz_min: float = 0.25,
    com_std: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """PLANT/PUSH: left plant, CoM over left foot, right tuck, left push."""
    phase = _ep_phase(env)
    in_pp = (phase == 1) | (phase == 2)
    contacts = foot_contact_mask(
        env, sensor_cfg=sensor_cfg, contact_force_threshold=contact_force_threshold
    )
    left_c = contacts[:, 0]
    right_c = contacts[:, 1]
    both_air = (~left_c) & (~right_c)
    left_fwd = lead_foot_forward_mask(env, lead="left", min_lead_m=min_lead_m)
    asset = env.scene[asset_cfg.name]
    vz = asset.data.root_lin_vel_w[:, 2]
    left_id, right_id = resolve_ankle_body_ids(env, asset_cfg.name)
    left_xy = asset.data.body_pos_w[:, left_id, :2]
    root_xy = asset.data.root_pos_w[:, :2]
    com_err = torch.linalg.norm(root_xy - left_xy, dim=-1)
    com_track = torch.exp(-torch.square(com_err / max(com_std, 1e-3)))
    right_tuck = torch.linalg.norm(
        asset.data.root_pos_w - asset.data.body_pos_w[:, right_id], dim=-1
    )
    right_tuck_score = torch.exp(-torch.square((right_tuck - 0.45) / 0.12))

    plant = in_pp & left_fwd & left_c & (~both_air)
    push = in_pp & left_c & (~right_c) & left_fwd & (vz > push_vz_min) & (~both_air)

    return (
        0.35 * plant.float()
        + 0.25 * (plant.float() * com_track)
        + 0.20 * (push.float() * right_tuck_score)
        + 0.70 * push.float()
    )


def apex_tuck(
    env: ManagerBasedRLEnv,
    command_name: str = "jump",
    tuck_std: float = 0.15,
    pitch_std: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Drive both legs toward / below tuck_apex* while airborne ascending.

    One-sided tuck score gated by forward-pitch scale so a vertical tuck no
    longer pays full credit (angular-momentum-correct apex lean required).
    """
    has_liftoff = getattr(env, "_ep_has_liftoff", None)
    if has_liftoff is None:
        return torch.zeros(env.num_envs, device=env.device)
    both_air = _both_feet_air(
        env, SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
    )
    has_landed = getattr(env, "_ep_has_landed", None)
    apex_latched = getattr(env, "_ep_apex_latched", None)
    active = has_liftoff & both_air
    if has_landed is not None:
        active = active & (~has_landed)
    if apex_latched is not None:
        descending = env.scene["robot"].data.root_lin_vel_w[:, 2] < -0.20
        active = active & (~(apex_latched & descending))

    term = _jump_term(env, command_name)
    d_l, d_r = _per_leg_tuck(env, asset_cfg=asset_cfg)
    target = term.tuck_apex_star
    std = max(tuck_std, 1e-3)
    excess_l = (d_l - target).clamp(min=0.0)
    excess_r = (d_r - target).clamp(min=0.0)
    score_l = torch.exp(-torch.square(excess_l / std))
    score_r = torch.exp(-torch.square(excess_r / std))
    tuck = torch.minimum(score_l, score_r)

    pitch = _forward_pitch_b(env.scene[asset_cfg.name])
    pitch_err = pitch - term.pitch_apex_star
    pitch_scale = torch.exp(-torch.square(pitch_err / max(pitch_std, 1e-3)))
    # Kill credit for vertical / backward lean (pitch ≤ 0).
    pitch_scale = pitch_scale * (pitch > 0.02).float()
    return active.float() * tuck * pitch_scale


def apex_pitch(
    env: ManagerBasedRLEnv,
    command_name: str = "jump",
    pitch_std: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense forward-pitch tracking at apex (same air window as apex_tuck)."""
    has_liftoff = getattr(env, "_ep_has_liftoff", None)
    if has_liftoff is None:
        return torch.zeros(env.num_envs, device=env.device)
    both_air = _both_feet_air(
        env, SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
    )
    has_landed = getattr(env, "_ep_has_landed", None)
    apex_latched = getattr(env, "_ep_apex_latched", None)
    active = has_liftoff & both_air
    if has_landed is not None:
        active = active & (~has_landed)
    if apex_latched is not None:
        descending = env.scene["robot"].data.root_lin_vel_w[:, 2] < -0.20
        active = active & (~(apex_latched & descending))

    term = _jump_term(env, command_name)
    pitch = _forward_pitch_b(env.scene[asset_cfg.name])
    err = pitch - term.pitch_apex_star
    score = torch.exp(-torch.square(err / max(pitch_std, 1e-3)))
    score = score * (pitch > 0.02).float()
    return active.float() * score


def foot_clearance(
    env: ManagerBasedRLEnv,
    command_name: str = "jump",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense sole-height progress toward commanded ``h_obstacle`` during flight.

    Saturates at 1.0×h (no 1.2× farming). Excess height is a separate penalty.
    """
    has_liftoff = getattr(env, "_ep_has_liftoff", None)
    has_landed = getattr(env, "_ep_has_landed", None)
    if has_liftoff is None:
        return torch.zeros(env.num_envs, device=env.device)
    both_air = _both_feet_air(
        env, SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
    )
    active = has_liftoff & both_air
    if has_landed is not None:
        active = active & (~has_landed)

    asset = env.scene[asset_cfg.name]
    left_id, right_id = resolve_ankle_body_ids(env, asset_cfg.name)
    sole = 0.05
    env_sole = getattr(env, "_ankle_to_sole", None)
    if env_sole is not None:
        sole = float(env_sole)
    left_z = asset.data.body_pos_w[:, left_id, 2] - sole
    right_z = asset.data.body_pos_w[:, right_id, 2] - sole
    cur = torch.maximum(left_z, right_z)
    h = _jump_term(env, command_name).h_obstacle.clamp(min=0.05)
    return active.float() * (cur / h).clamp(0.0, 1.0)


def excess_height_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "jump",
    margin: float = CLEARANCE_EXCESS_MARGIN,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize sole height above ``h + margin`` during flight (anti-balloon)."""
    has_liftoff = getattr(env, "_ep_has_liftoff", None)
    has_landed = getattr(env, "_ep_has_landed", None)
    if has_liftoff is None:
        return torch.zeros(env.num_envs, device=env.device)
    both_air = _both_feet_air(
        env, SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
    )
    active = has_liftoff & both_air
    if has_landed is not None:
        active = active & (~has_landed)

    asset = env.scene[asset_cfg.name]
    left_id, right_id = resolve_ankle_body_ids(env, asset_cfg.name)
    sole = float(getattr(env, "_ankle_to_sole", 0.05))
    left_z = asset.data.body_pos_w[:, left_id, 2] - sole
    right_z = asset.data.body_pos_w[:, right_id, 2] - sole
    cur = torch.maximum(left_z, right_z)
    h = _jump_term(env, command_name).h_obstacle
    excess = (cur - h - float(margin)).clamp(min=0.0)
    return active.float() * (excess + 2.0 * torch.square(excess))


def leg_extend(
    env: ManagerBasedRLEnv,
    tuck_target: float = TUCK_EXTENDED,
    tuck_std: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Open both legs toward landing extension after apex, before touchdown.

    Score is ``min(score_l, score_r)`` so both ankles must open for credit.
    Active on descent after apex (phase EXTEND or apex_latched & descending).
    """
    apex_latched = getattr(env, "_ep_apex_latched", None)
    has_landed = getattr(env, "_ep_has_landed", None)
    if apex_latched is None:
        return torch.zeros(env.num_envs, device=env.device)
    phase = _ep_phase(env)
    vz = env.scene["robot"].data.root_lin_vel_w[:, 2]
    descending = vz < 0.05
    in_ext = (phase == 4) | (apex_latched & descending)
    if has_landed is not None:
        in_ext = in_ext & (~has_landed)
    d_l, d_r = _per_leg_tuck(env, asset_cfg=asset_cfg)
    std = max(tuck_std, 1e-3)
    score_l = torch.exp(-torch.square((d_l - tuck_target) / std))
    score_r = torch.exp(-torch.square((d_r - tuck_target) / std))
    return in_ext.float() * torch.minimum(score_l, score_r)


def extend_pitch(
    env: ManagerBasedRLEnv,
    pitch_std: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """On EXTEND/descent, reward pitch moving toward upright (pairs with apex lean)."""
    apex_latched = getattr(env, "_ep_apex_latched", None)
    has_landed = getattr(env, "_ep_has_landed", None)
    if apex_latched is None:
        return torch.zeros(env.num_envs, device=env.device)
    phase = _ep_phase(env)
    vz = env.scene[asset_cfg.name].data.root_lin_vel_w[:, 2]
    descending = vz < 0.05
    in_ext = (phase == 4) | (apex_latched & descending)
    if has_landed is not None:
        in_ext = in_ext & (~has_landed)
    # Target near-upright but still slightly forward (~0.05) for landing prep.
    pitch = _forward_pitch_b(env.scene[asset_cfg.name])
    target = torch.full_like(pitch, 0.05)
    score = torch.exp(-torch.square((pitch - target) / max(pitch_std, 1e-3)))
    # Prefer pitch that is not strongly backward.
    score = score * (pitch > -0.10).float()
    return in_ext.float() * score


def foot_split_penalty(
    env: ManagerBasedRLEnv,
    max_sep: float = FOOT_SEP_MAX_APEX,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for splits while airborne after liftoff (excess over CSV foot-sep cap).

    Returns excess meters (clipped) so a large negative weight hurts more than
    distance farming pays.
    """
    has_liftoff = getattr(env, "_ep_has_liftoff", None)
    has_landed = getattr(env, "_ep_has_landed", None)
    both_air = _both_feet_air(
        env, SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
    )
    active = both_air
    if has_liftoff is not None:
        active = active & has_liftoff
    if has_landed is not None:
        active = active & (~has_landed)
    sep = _foot_sep(env, asset_cfg=asset_cfg)
    # Linear + quadratic excess so mild splits hurt and extreme splits dominate.
    excess = (sep - max_sep).clamp(min=0.0)
    return active.float() * (excess + 2.0 * torch.square(excess))


def rejump_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    """Penalize leaving the ground again after the first landing (anti double-jump)."""
    phase = _ep_phase(env)
    post = phase >= 5
    has_landed = getattr(env, "_ep_has_landed", None)
    if has_landed is None:
        return torch.zeros(env.num_envs, device=env.device)
    both_air = _both_feet_air(env, sensor_cfg, contact_force_threshold)
    return (post & has_landed & both_air).float()


def flight_distance_progress(
    env: ManagerBasedRLEnv,
    command_name: str = "jump",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    contact_force_threshold: float = 5.0,
    max_sep: float = FOOT_SEP_MAX_FLIGHT,
) -> torch.Tensor:
    """Dense heading progress toward commanded flight distance *during flight*.

    Only while both feet are airborne after liftoff (not the entire post-flight
    settle). Zeroed when the episode has already exceeded the anti-splits
    foot-sep cap, so splits cannot farm distance reward.
    """
    term = _jump_term(env, command_name)
    target = term.flight_distance.clamp(min=0.1)
    flown = getattr(env, "_ep_flight_distance", None)
    if flown is not None:
        dist = torch.maximum(flown, term.progress.clamp(min=0.0))
    else:
        dist = term.progress.clamp(min=0.0)
    both_air = _both_feet_air(env, sensor_cfg, contact_force_threshold)
    has_liftoff = getattr(env, "_ep_has_liftoff", None)
    active = both_air
    if has_liftoff is not None:
        active = active & has_liftoff
    max_sep_ep = getattr(env, "_ep_max_foot_sep_flight", None)
    if max_sep_ep is not None:
        active = active & (max_sep_ep <= max_sep)
    else:
        active = active & (_foot_sep(env) <= max_sep)
    return active.float() * (dist / target).clamp(0.0, 1.2)


def heading_keep(
    env: ManagerBasedRLEnv,
    command_name: str = "jump",
    std: float = 0.35,
) -> torch.Tensor:
    """Dense reward for keeping yaw aligned with episode-start heading.

    Uses ``JumpCommand.heading_error`` (frozen ``_forward_w`` at resample).
    Active for the whole episode so prep and flight spin are both shaped.
    """
    term = _jump_term(env, command_name)
    err = term.heading_error.abs()
    return torch.exp(-torch.square(err / max(std, 1e-3)))


def land_and_idle(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "jump",
    stand_height: float = 0.75,
    height_std: float = 0.10,
    contact_force_threshold: float = 5.0,
    impact_force_scale: float = 400.0,
    max_tilt_rad: float = 0.40,
    min_height: float = 0.68,
    max_height: float = 0.95,
    max_horiz_speed: float = 0.60,
    max_joint_vel2: float = 40.0,
    max_heading_err: float = 0.35,
) -> torch.Tensor:
    """Merged touchdown + idle recovery after a real flight.

    Combines the former ``landing_quality`` and ``idle_recovery`` terms so the
    idle stand is paid once. Active after ``_ep_has_landed`` inside the
    post-landing window. Kneel-height / spun landings get little credit.
    """
    has_landed = getattr(env, "_ep_has_landed", None)
    post_steps = getattr(env, "_ep_post_land_steps", None)
    window = int(getattr(env, "_post_land_window_steps", 150))
    if has_landed is None:
        return torch.zeros(env.num_envs, device=env.device)

    in_window = has_landed.clone()
    if post_steps is not None:
        in_window = has_landed & (post_steps <= window)

    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_mag = torch.norm(forces, dim=-1)
    both_contact = (force_mag > contact_force_threshold).all(dim=-1)

    height = asset.data.root_pos_w[:, 2]
    tilt = _tilt_from_upright(asset)
    upright = tilt < max_tilt_rad
    height_ok = (height > min_height) & (height < max_height)
    horiz = torch.linalg.norm(asset.data.root_lin_vel_w[:, :2], dim=-1)
    vz = asset.data.root_lin_vel_w[:, 2].abs()
    joint_vel2 = torch.sum(torch.square(asset.data.joint_vel), dim=1)
    try:
        heading_err = _jump_term(env, command_name).heading_error.abs()
    except (KeyError, AttributeError):
        heading_err = torch.zeros(env.num_envs, device=env.device)
    heading_ok = heading_err < max_heading_err

    # Scale landing credit by obstacle clearance so flat hops cannot farm idle.
    peak_sole = getattr(env, "_ep_peak_foot_z", None)
    try:
        h_cmd = _jump_term(env, command_name).h_obstacle.clamp(min=0.05)
    except (KeyError, AttributeError):
        h_cmd = torch.ones(env.num_envs, device=env.device)
    if peak_sole is not None:
        clear_scale = (peak_sole.max(dim=-1).values / h_cmd).clamp(0.0, 1.0)
    else:
        clear_scale = torch.ones(env.num_envs, device=env.device)
    land_scale = 0.20 + 0.80 * clear_scale

    # Sparse-ish feet-first touchdown credit (once per episode).
    if not hasattr(env, "_jump_land_touch_given"):
        env._jump_land_touch_given = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    env._jump_land_touch_given = env._jump_land_touch_given & (~_ep_fresh_mask(env))
    had_flight = getattr(env, "_ep_had_flight", None)
    had = has_landed if had_flight is None else (has_landed | had_flight)
    touch = (
        both_contact
        & had
        & upright
        & height_ok
        & heading_ok
        & (~env._jump_land_touch_given)
    )
    env._jump_land_touch_given = env._jump_land_touch_given | touch
    touch_bonus = touch.float() * 0.5 * clear_scale

    height_track = torch.exp(-torch.square(height - stand_height) / (height_std**2))
    upright_exp = torch.exp(-torch.square(tilt / max(max_tilt_rad, 1e-3)))
    heading_exp = torch.exp(-torch.square(heading_err / max(max_heading_err, 1e-3)))
    slow = torch.exp(-torch.square(horiz / max(max_horiz_speed, 1e-3)))
    quiet = torch.exp(-(joint_vel2 / max(max_joint_vel2, 1e-3)))
    vz_ok = torch.exp(-torch.square(vz / 0.45))
    fz = forces[:, :, 2].abs().max(dim=-1).values
    impact_pen = (fz / max(impact_force_scale, 1.0)).clamp(0.0, 1.0)

    score = (
        0.30 * height_track
        + 0.20 * upright_exp
        + 0.20 * heading_exp
        + 0.12 * slow
        + 0.10 * quiet
        + 0.08 * vz_ok
        - 0.05 * impact_pen
    )
    urgency = torch.ones(env.num_envs, device=env.device)
    if post_steps is not None:
        urgency = 1.0 + 0.5 * (1.0 - (post_steps.float() / max(window, 1)).clamp(0.0, 1.0))

    return touch_bonus + in_window.float() * both_contact.float() * urgency * score * land_scale


def no_flight_timeout_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sparse −1 on the last step if the episode timed out with no true flight."""
    max_len = int(getattr(env, "max_episode_length", 0) or 0)
    ep_len = getattr(env, "episode_length_buf", None)
    if max_len <= 0 or ep_len is None:
        return torch.zeros(env.num_envs, device=env.device)

    at_timeout = ep_len >= (max_len - 1)
    had_flight = getattr(env, "_ep_had_flight", None)
    if had_flight is None:
        return torch.zeros(env.num_envs, device=env.device)
    return (at_timeout & (~had_flight)).float()


# ---------------------------------------------------------------------------
# Jump sparse success criteria
# ---------------------------------------------------------------------------


def _once_success(env: ManagerBasedRLEnv, attr: str, latch_attr: str) -> torch.Tensor:
    """+1 the first step ``env.<latch_attr>`` is true each episode."""
    given_attr = f"_jump_{attr}_given"
    if not hasattr(env, given_attr):
        setattr(
            env,
            given_attr,
            torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        )
    given = getattr(env, given_attr)
    given = given & (~_ep_fresh_mask(env))
    setattr(env, given_attr, given)
    ok = getattr(env, latch_attr, None)
    if ok is None:
        return torch.zeros(env.num_envs, device=env.device)
    hit = ok & (~given)
    setattr(env, given_attr, given | hit)
    return hit.float()


def success_rise(
    env: ManagerBasedRLEnv,
    command_name: str = "jump",
    once_per_episode: bool = True,
) -> torch.Tensor:
    """+1 once when peak sole clearance meets commanded ``h_obstacle``."""
    del command_name, once_per_episode
    return _once_success(env, "success_rise", "_ep_rise_ok")


def success_tuck(
    env: ManagerBasedRLEnv,
    command_name: str = "jump",
    once_per_episode: bool = True,
) -> torch.Tensor:
    """+1 once when both legs' min airborne tuck meet ``tuck_apex*``."""
    del command_name, once_per_episode
    return _once_success(env, "success_tuck", "_ep_tuck_ok")


def success_apex(
    env: ManagerBasedRLEnv,
    command_name: str = "jump",
    once_per_episode: bool = True,
) -> torch.Tensor:
    """+1 once when full tucked-apex success latches (rise + tuck + anti-splits)."""
    del command_name, once_per_episode
    cleared = getattr(env, "_ep_cleared_apex", None)
    if cleared is None:
        return torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_jump_success_apex_given"):
        env._jump_success_apex_given = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    fresh = _ep_fresh_mask(env)
    env._jump_success_apex_given = env._jump_success_apex_given & (~fresh)
    hit = cleared & (~env._jump_success_apex_given)
    env._jump_success_apex_given = env._jump_success_apex_given | hit
    return hit.float()


def success_flight_distance(
    env: ManagerBasedRLEnv,
    command_name: str = "jump",
    once_per_episode: bool = True,
) -> torch.Tensor:
    """+1 once when heading-aligned liftoff→landing distance ≥ commanded flight_distance."""
    del command_name
    if not hasattr(env, "_jump_success_dist_given"):
        env._jump_success_dist_given = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._jump_success_dist_given = env._jump_success_dist_given & (~_ep_fresh_mask(env))

    ok = getattr(env, "_ep_cleared_distance", None)
    if ok is None:
        return torch.zeros(env.num_envs, device=env.device)
    hit = ok & (~env._jump_success_dist_given)
    if once_per_episode:
        env._jump_success_dist_given = env._jump_success_dist_given | hit
    return hit.float()


def success_stable_landing(
    env: ManagerBasedRLEnv,
    command_name: str = "jump",
    once_per_episode: bool = True,
) -> torch.Tensor:
    """+1 once when idle standing is reached within the post-landing window.

    Dense ``land_and_idle`` still shapes this; full-success does not require it yet.
    """
    del command_name
    if not hasattr(env, "_jump_success_stable_given"):
        env._jump_success_stable_given = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._jump_success_stable_given = env._jump_success_stable_given & (~_ep_fresh_mask(env))

    ok = getattr(env, "_ep_stable_ok", None)
    if ok is None:
        return torch.zeros(env.num_envs, device=env.device)
    hit = ok & (~env._jump_success_stable_given)
    if once_per_episode:
        env._jump_success_stable_given = env._jump_success_stable_given | hit
    return hit.float()
