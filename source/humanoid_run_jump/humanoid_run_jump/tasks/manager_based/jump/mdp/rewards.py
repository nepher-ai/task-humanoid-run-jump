# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for run and jump environments.

Jump task uses ≤9 terms that specify WHAT to do (clear h, travel d, land,
stand). Pose style (HOW) is left to AMP + hard terminations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from humanoid_run_jump.tracker.reduced_coords import quat_apply_inverse
from humanoid_run_jump.tasks.manager_based.jump.mdp.gait import (
    foot_contact_mask,
    resolve_ankle_body_ids,
)
from humanoid_run_jump.tasks.manager_based.jump.mdp.jump_envelope import (
    FOOT_SEP_MAX_FLIGHT,
    TUCK_EXTENDED,
    foot_sep_flight_max,
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


def _forward_pitch_b(asset) -> torch.Tensor:
    """Forward lean angle (rad): positive = chest toward body +x."""
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


def _jump_cache(env: ManagerBasedRLEnv) -> dict | None:
    """Per-step quantities published by ``_update_jump_metrics_and_style``."""
    return getattr(env, "_jump_step", None)


def _both_feet_air_cached(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg | None = None,
    contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    cache = _jump_cache(env)
    if cache is not None and "both_air" in cache:
        return cache["both_air"]
    if sensor_cfg is None:
        sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
    return _both_feet_air(env, sensor_cfg, contact_force_threshold)


def _foot_sep(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    left_id, right_id = resolve_ankle_body_ids(env, asset_cfg.name)
    return torch.linalg.norm(
        asset.data.body_pos_w[:, left_id] - asset.data.body_pos_w[:, right_id], dim=-1
    )


def _foot_sep_cached(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    cache = _jump_cache(env)
    if cache is not None and "foot_sep" in cache:
        return cache["foot_sep"]
    return _foot_sep(env, asset_cfg=asset_cfg)


def _forward_pitch_cached(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    cache = _jump_cache(env)
    if cache is not None and "pitch" in cache:
        return cache["pitch"]
    return _forward_pitch_b(env.scene[asset_cfg.name])


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


# ---------------------------------------------------------------------------
# Jump dense shaping (9-term set)
# ---------------------------------------------------------------------------


def takeoff_launch(
    env: ManagerBasedRLEnv,
    command_name: str = "jump",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    contact_force_threshold: float = 5.0,
    vel_std: float = 1.0,
    vz_overshoot: float = 0.35,
    pitch_std: float = 0.12,
    clean_omega0: float = 2.2,
    knee_coil_target: float = 1.05,
    knee_coil_std: float = 0.40,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Launch shaping: coil (knee) + approach ``(v_x*, v_z*)`` and lift off cleanly.

    Plant/push pay is scaled by knee coil so stiff upright rabbit hops earn less
    than a crouched load before takeoff.
    """
    phase = _ep_phase(env)
    in_pp = (phase == 1) | (phase == 2)
    in_rise = phase == 3
    asset = env.scene[asset_cfg.name]
    term = _jump_term(env, command_name)
    both_air = _both_feet_air_cached(env, sensor_cfg, contact_force_threshold)
    contacts = foot_contact_mask(
        env, sensor_cfg=sensor_cfg, contact_force_threshold=contact_force_threshold
    )
    left_c = contacts[:, 0]

    target_v = term.target_vel_b
    vel_b = quat_apply_inverse(asset.data.root_quat_w, asset.data.root_lin_vel_w)
    vz_err = vel_b[:, 2] - target_v[:, 2]
    vz_err = torch.where(vz_err > vz_overshoot, vz_err + (vz_err - vz_overshoot), vz_err)
    err_xy = torch.sum(torch.square(vel_b[:, :2] - target_v[:, :2]), dim=1)
    err = err_xy + torch.square(vz_err)
    vel_track = torch.exp(-err / max(vel_std, 1e-3))
    ascending = (asset.data.root_lin_vel_w[:, 2] > 0.0).float()
    clean = _clean_takeoff_scale(asset, clean_omega0)
    pitch = _forward_pitch_cached(env, asset_cfg)
    pitch_err = pitch - term.pitch_takeoff_star
    pitch_track = torch.exp(-torch.square(pitch_err / max(pitch_std, 1e-3)))

    jmap = getattr(env, "joint_order_map", None)
    if jmap is not None:
        dof = jmap.to_g1(asset.data.joint_pos)
        knee_flex = 0.5 * (dof[:, 3] + dof[:, 9])
    else:
        knee_flex = torch.full(
            (env.num_envs,), float(knee_coil_target), device=env.device
        )
    knee_coil = torch.exp(
        -torch.square((knee_flex - float(knee_coil_target)) / max(knee_coil_std, 1e-3))
    )
    # Floor at 0.35 so stiff plant still gets some signal, but crouch pays more.
    coil_scale = 0.35 + 0.65 * knee_coil

    grounded = (
        in_pp.float()
        * left_c.float()
        * (~both_air).float()
        * vel_track
        * pitch_track
        * coil_scale
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


def obstacle_clearance(
    env: ManagerBasedRLEnv,
    command_name: str = "jump",
    overshoot: float = 1.25,
    overshoot_weight: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Two-sided sole clearance: reward up to ``h``, penalize ballooning past ``1.25×h``.

    ``clamp(peak_sole / h, 0, 1) - overshoot_weight * relu(peak_sole / h - overshoot)``.
    """
    del asset_cfg
    has_liftoff = getattr(env, "_ep_has_liftoff", None)
    has_landed = getattr(env, "_ep_has_landed", None)
    if has_liftoff is None:
        return torch.zeros(env.num_envs, device=env.device)
    active = has_liftoff.clone()
    if has_landed is not None:
        active = active & (~has_landed)

    peak_sole = getattr(env, "_ep_peak_foot_z", None)
    if peak_sole is None:
        return torch.zeros(env.num_envs, device=env.device)
    peak = peak_sole.max(dim=-1).values
    try:
        h = _jump_term(env, command_name).h_obstacle.clamp(min=0.05)
    except (KeyError, AttributeError):
        return torch.zeros(env.num_envs, device=env.device)

    ratio = peak / h
    reward = ratio.clamp(0.0, 1.0) - float(overshoot_weight) * (ratio - float(overshoot)).clamp(min=0.0)
    return active.float() * reward


# Backward-compatible alias.
foot_clearance = obstacle_clearance


def flight_distance_progress(
    env: ManagerBasedRLEnv,
    command_name: str = "jump",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    contact_force_threshold: float = 5.0,
    max_sep: float = FOOT_SEP_MAX_FLIGHT,
) -> torch.Tensor:
    """Dense heading progress toward commanded flight distance during flight."""
    del max_sep
    term = _jump_term(env, command_name)
    target = term.flight_distance.clamp(min=0.1)
    flown = getattr(env, "_ep_flight_distance", None)
    if flown is not None:
        dist = torch.maximum(flown, term.progress.clamp(min=0.0))
    else:
        dist = term.progress.clamp(min=0.0)
    both_air = _both_feet_air_cached(env, sensor_cfg, contact_force_threshold)
    has_liftoff = getattr(env, "_ep_has_liftoff", None)
    has_landed = getattr(env, "_ep_has_landed", None)
    active = both_air
    if has_liftoff is not None:
        active = active & has_liftoff
    # Cut after first land so a second hop cannot re-farm flight_distance.
    if has_landed is not None:
        active = active & (~has_landed)
    max_sep_ep = getattr(env, "_ep_max_foot_sep_flight", None)
    sep_cap = foot_sep_flight_max(term.flight_distance)
    if max_sep_ep is not None:
        active = active & (max_sep_ep <= sep_cap)
    else:
        active = active & (_foot_sep_cached(env) <= sep_cap)
    return active.float() * (dist / target).clamp(0.0, 1.2)


def apex_fold(
    env: ManagerBasedRLEnv,
    command_name: str = "jump",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Potential-based push-leg tuck toward ``push_fold*`` over full flight.

    Pays only when episode-min pelvis→ankle (``_ep_min_push_d``) improves,
    normalized by ``(TUCK_EXTENDED − push_fold*)`` and saturated at the target
    (no deeper-than-star bonus). Sums to ≤1 per episode — cannot be farmed by
    holding a tuck. Active for the whole airborne window so fold at/after apex
    counts; cut after land so EXTEND→land opening is undisturbed.
    """
    del asset_cfg
    has_liftoff = getattr(env, "_ep_has_liftoff", None)
    has_landed = getattr(env, "_ep_has_landed", None)
    min_push = getattr(env, "_ep_min_push_d", None)
    if has_liftoff is None or min_push is None:
        return torch.zeros(env.num_envs, device=env.device)
    both_air = _both_feet_air_cached(env)
    active = has_liftoff & both_air
    if has_landed is not None:
        active = active & (~has_landed)

    try:
        fold_star = _jump_term(env, command_name).push_fold_star.clamp(min=0.05)
    except (KeyError, AttributeError):
        return torch.zeros(env.num_envs, device=env.device)

    # Sentinel 10.0 = not yet measured in air.
    measured = min_push < 9.0
    denom = (float(TUCK_EXTENDED) - fold_star).clamp(min=0.05)
    progress = ((float(TUCK_EXTENDED) - min_push) / denom).clamp(0.0, 1.0)
    progress = torch.where(measured & active, progress, torch.zeros_like(progress))

    paid = getattr(env, "_ep_fold_progress_paid", None)
    if paid is None:
        paid = torch.zeros(env.num_envs, device=env.device)
        env._ep_fold_progress_paid = paid

    delta = (progress - paid).clamp(min=0.0)
    # In-place so amp_env episode resets keep the same buffer identity.
    paid.copy_(torch.maximum(paid, progress))
    return delta


def heading_keep(
    env: ManagerBasedRLEnv,
    command_name: str = "jump",
    std: float = 0.35,
) -> torch.Tensor:
    """Dense reward for keeping yaw aligned with episode-start heading.

    Cut after first landing so bounce-walk +x no longer farms heading.
    """
    term = _jump_term(env, command_name)
    err = term.heading_error.abs()
    r = torch.exp(-torch.square(err / max(std, 1e-3)))
    has_landed = getattr(env, "_ep_has_landed", None)
    if has_landed is not None:
        r = r * (~has_landed).float()
    return r


def land_absorb(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "jump",
    stand_height: float = 0.75,
    absorb_height: float = 0.62,
    height_std: float = 0.08,
    contact_force_threshold: float = 5.0,
    impact_force_scale: float = 400.0,
    min_height: float = 0.56,
    max_height: float = 0.95,
    max_horiz_speed: float = 0.40,
    max_joint_vel2: float = 35.0,
    max_heading_err: float = 0.35,
    absorb_steps: int = 12,
    pay_steps: int = 120,
    knee_absorb_target: float = 1.15,
    knee_absorb_std: float = 0.35,
    absorb_pitch_target: float = 0.35,
    absorb_pitch_std: float = 0.15,
) -> torch.Tensor:
    """Pure-positive crouch absorb → stand still after a real flight.

    Early frames: lower CoM + knee flex + forward lean.
    Later frames: stand height + slow + quiet (path to stop in 1–2 steps).
    Dense pay lasts through ``pay_steps`` (~2.4 s). Bounce-walk is handled by
    the ``bounce_walk`` termination — this term never subtracts.
    """
    del command_name
    has_landed = getattr(env, "_ep_has_landed", None)
    post_steps = getattr(env, "_ep_post_land_steps", None)
    if has_landed is None:
        return torch.zeros(env.num_envs, device=env.device)

    had_flight = getattr(env, "_ep_had_flight", None)
    if had_flight is None:
        had_flight = has_landed

    pay_horizon = max(int(pay_steps), 1)
    absorb = max(int(absorb_steps), 1)
    in_window = has_landed & had_flight
    if post_steps is not None:
        in_window = in_window & (post_steps <= pay_horizon)

    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_mag = torch.norm(forces, dim=-1)
    both_contact = (force_mag > contact_force_threshold).all(dim=-1)

    height = asset.data.root_pos_w[:, 2]
    height_ok = (height > min_height) & (height < max_height)

    if post_steps is not None:
        absorb_w = (1.0 - (post_steps.float() / float(absorb)).clamp(0.0, 1.0))
        settle_w = 1.0 - absorb_w
        height_target = absorb_w * float(absorb_height) + settle_w * float(stand_height)
        pitch_target = absorb_w * float(absorb_pitch_target) + settle_w * 0.08
    else:
        absorb_w = torch.ones(env.num_envs, device=env.device)
        settle_w = torch.zeros(env.num_envs, device=env.device)
        height_target = torch.full_like(height, float(absorb_height))
        pitch_target = torch.full_like(height, float(absorb_pitch_target))

    height_track = torch.exp(-torch.square(height - height_target) / (height_std**2))

    pitch = _forward_pitch_cached(env, asset_cfg)
    pitch_track = torch.exp(
        -torch.square((pitch - pitch_target) / max(absorb_pitch_std, 1e-3))
    )
    pitch_track = pitch_track * (pitch >= -0.05).float()

    jmap = getattr(env, "joint_order_map", None)
    if jmap is not None:
        dof = jmap.to_g1(asset.data.joint_pos)
        knee_flex = 0.5 * (dof[:, 3] + dof[:, 9])
    else:
        names = list(asset.data.joint_names)
        try:
            li = names.index("left_knee_joint")
            ri = names.index("right_knee_joint")
            knee_flex = 0.5 * (asset.data.joint_pos[:, li] + asset.data.joint_pos[:, ri])
        except ValueError:
            knee_flex = torch.full(
                (env.num_envs,), float(knee_absorb_target), device=env.device
            )
    knee_absorb = torch.exp(
        -torch.square((knee_flex - float(knee_absorb_target)) / max(knee_absorb_std, 1e-3))
    )

    horiz = torch.linalg.norm(asset.data.root_lin_vel_w[:, :2], dim=-1)
    vz = asset.data.root_lin_vel_w[:, 2].abs()
    joint_vel2 = torch.sum(torch.square(asset.data.joint_vel), dim=1)
    slow = torch.exp(-torch.square(horiz / max(max_horiz_speed, 1e-3)))
    quiet = torch.exp(-(joint_vel2 / max(max_joint_vel2, 1e-3)))
    vz_ok = torch.exp(-torch.square(vz / 0.30))
    fz = forces[:, :, 2].abs().max(dim=-1).values
    impact_pen = (fz / max(impact_force_scale, 1.0)).clamp(0.0, 1.0)

    try:
        heading_err = _jump_term(env, "jump").heading_error.abs()
    except (KeyError, AttributeError):
        heading_err = torch.zeros(env.num_envs, device=env.device)
    heading_exp = torch.exp(-torch.square(heading_err / max(max_heading_err, 1e-3)))

    if not hasattr(env, "_jump_land_touch_given"):
        env._jump_land_touch_given = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    env._jump_land_touch_given = env._jump_land_touch_given & (~_ep_fresh_mask(env))
    crouch_touch = (knee_flex >= 0.80) | (height <= float(absorb_height) + 0.08)
    touch = (
        both_contact
        & had_flight
        & has_landed
        & height_ok
        & crouch_touch
        & (~env._jump_land_touch_given)
    )
    env._jump_land_touch_given = env._jump_land_touch_given | touch
    touch_bonus = touch.float() * 0.5

    # Early crouch; later stand-still. Settle weights dominate so stopping pays.
    # Extra plant-hold so staying both-feet-down outranks a second hop.
    score = (
        0.14 * height_track
        + 0.12 * (absorb_w * knee_absorb)
        + 0.08 * (absorb_w * pitch_track)
        + 0.06 * heading_exp
        + 0.28 * (settle_w * slow)
        + 0.16 * (settle_w * quiet)
        + 0.12 * (settle_w * vz_ok)
        + 0.04 * (absorb_w * slow)
        - 0.04 * impact_pen
    ).clamp(min=0.0)
    plant_hold = 0.20 * both_contact.float()

    return (
        touch_bonus
        + in_window.float() * height_ok.float() * (both_contact.float() * score + plant_hold)
    )


# Backward-compatible alias.
land_and_idle = land_absorb


# ---------------------------------------------------------------------------
# Jump sparse success
# ---------------------------------------------------------------------------


def success_clear(
    env: ManagerBasedRLEnv,
    command_name: str = "jump",
    once_per_episode: bool = True,
) -> torch.Tensor:
    """+1 once when sole clearance ≥ h AND heading-aligned distance ≥ command."""
    del command_name
    if not hasattr(env, "_jump_success_clear_given"):
        env._jump_success_clear_given = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    env._jump_success_clear_given = env._jump_success_clear_given & (~_ep_fresh_mask(env))

    rise = getattr(env, "_ep_rise_ok", None)
    dist = getattr(env, "_ep_cleared_distance", None)
    if rise is None or dist is None:
        return torch.zeros(env.num_envs, device=env.device)
    ok = rise & dist
    hit = ok & (~env._jump_success_clear_given)
    if once_per_episode:
        env._jump_success_clear_given = env._jump_success_clear_given | hit
    return hit.float()


def success_land_stable(
    env: ManagerBasedRLEnv,
    command_name: str = "jump",
    once_per_episode: bool = True,
) -> torch.Tensor:
    """+1 once when idle standing latches after a soft landing."""
    del command_name
    if not hasattr(env, "_jump_success_land_stable_given"):
        env._jump_success_land_stable_given = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    env._jump_success_land_stable_given = env._jump_success_land_stable_given & (
        ~_ep_fresh_mask(env)
    )

    ok = getattr(env, "_ep_stable_ok", None)
    soft = getattr(env, "_ep_had_good_landing", None)
    if ok is None:
        return torch.zeros(env.num_envs, device=env.device)
    if soft is not None:
        ok = ok & soft
    hit = ok & (~env._jump_success_land_stable_given)
    if once_per_episode:
        env._jump_success_land_stable_given = env._jump_success_land_stable_given | hit
    return hit.float()
