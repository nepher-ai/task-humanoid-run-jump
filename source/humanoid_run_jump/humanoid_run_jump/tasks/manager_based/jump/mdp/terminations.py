# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from humanoid_run_jump.tasks.manager_based.jump.mdp.jump_envelope import (
    RISE_HARD_EXTRA,
    foot_sep_hard,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def update_jump_state(
    env: ManagerBasedRLEnv,
    flight_style_scale: float = 0.85,
    land_style_scale: float = 0.85,
    land_absorb_min_height: float = 0.56,
    land_absorb_steps: int = 12,
    land_max_forward_pitch: float = 0.60,
    land_min_pitch: float = -0.05,
    land_max_roll: float = 0.25,
    land_knee_min: float = 0.75,
    land_crouch_max_height: float = 0.72,
    land_max_horiz_speed: float = 1.15,
) -> torch.Tensor:
    """Refresh phase machine / latches before rewards. Never terminates.

    Declared first in ``TerminationsCfg`` so :meth:`JumpAmpEnv._update_jump_metrics_and_style`
    runs after ``episode_length_buf`` is incremented but before reward / other
    termination terms read the latches.
    """
    fn = getattr(env, "_update_jump_metrics_and_style", None)
    if fn is not None:
        fn(
            flight_style_scale=flight_style_scale,
            land_style_scale=land_style_scale,
            land_absorb_min_height=land_absorb_min_height,
            land_absorb_steps=land_absorb_steps,
            land_max_forward_pitch=land_max_forward_pitch,
            land_min_pitch=land_min_pitch,
            land_max_roll=land_max_roll,
            land_knee_min=land_knee_min,
            land_crouch_max_height=land_crouch_max_height,
            land_max_horiz_speed=land_max_horiz_speed,
        )
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def root_height_below(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


def bad_orientation(
    env: ManagerBasedRLEnv,
    limit_angle: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base tilts more than ``limit_angle`` from upright (rad)."""
    asset = env.scene[asset_cfg.name]
    cos_tilt = (-asset.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)
    return torch.acos(cos_tilt) > limit_angle


def illegal_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces",
        body_names=[
            "torso_link",
            "pelvis",
            "head",
            ".*_knee_link",
            ".*_shoulder_.*",
            ".*_elbow_.*",
            ".*_wrist_.*",
            ".*rubber_hand",
        ],
    ),
    threshold: float = 25.0,
) -> torch.Tensor:
    """Terminate when non-foot bodies slam the ground (head / torso / knees / arms)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return torch.norm(forces, dim=-1).max(dim=-1).values > threshold


def illegal_takeoff(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate when takeoff is armed but last support foot is not the left."""
    bad = getattr(env, "_ep_bad_takeoff", None)
    if bad is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return bad


def prep_timeout(env: ManagerBasedRLEnv, max_prep_steps: int = 50) -> torch.Tensor:
    """Terminate if the episode stays in PREP longer than ``max_prep_steps`` (~1 s)."""
    phase = getattr(env, "_ep_phase", None)
    prep_steps = getattr(env, "_ep_prep_steps", None)
    if phase is None or prep_steps is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return (phase == 0) & (prep_steps >= int(max_prep_steps))


def no_liftoff_timeout(
    env: ManagerBasedRLEnv, max_steps: int = 100
) -> torch.Tensor:
    """Fail if the policy never leaves the ground within ``max_steps`` (~2 s).

    Closes the PLANT stand-forever farm that ``prep_timeout`` cannot catch once
    the episode has exited PREP.
    """
    has_liftoff = getattr(env, "_ep_has_liftoff", None)
    ep_len = getattr(env, "episode_length_buf", None)
    if has_liftoff is None or ep_len is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return (~has_liftoff) & (ep_len >= int(max_steps))


def jump_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End the episode once rise+pitch + distance + soft land + stable idle latch.

    Fold/hurdle geometry is AMP style and is not required. Requires
    ``_ep_stable_ok`` so truncation does not cut off land recovery early.
    """
    ok = getattr(env, "_ep_full_success", None)
    if ok is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return ok


def post_land_timeout(
    env: ManagerBasedRLEnv,
    max_post_land_steps: int = 150,
) -> torch.Tensor:
    """Truncate ~3 s after first landing so settle frames do not dominate the rollout.

    ``time_out=True`` in the env cfg so this is not treated as a failure.
    """
    has_landed = getattr(env, "_ep_has_landed", None)
    post_steps = getattr(env, "_ep_post_land_steps", None)
    if has_landed is None or post_steps is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return has_landed & (post_steps >= int(max_post_land_steps))


def bounce_walk(
    env: ManagerBasedRLEnv,
    reair_grace_steps: int = 16,
    horiz_grace_steps: int = 70,
    max_horiz_speed: float = 0.85,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    contact_force_threshold: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Fail post-land double-hop early; fail residual fast walk later.

    * Re-air (``both_air``) after ``reair_grace_steps`` (~0.32 s) — catches the
      slight second jump seen in play.
    * Horiz speed above ``max_horiz_speed`` only after ``horiz_grace_steps``
      (~1.4 s) — allows crouch absorb to shed forward speed first.

    Exempt once ``_ep_stable_ok`` latches.
    """
    has_landed = getattr(env, "_ep_has_landed", None)
    had_flight = getattr(env, "_ep_had_flight", None)
    post_steps = getattr(env, "_ep_post_land_steps", None)
    if has_landed is None or post_steps is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if had_flight is None:
        had_flight = has_landed

    base = has_landed & had_flight
    stable = getattr(env, "_ep_stable_ok", None)
    if stable is not None:
        base = base & (~stable)

    asset = env.scene[asset_cfg.name]
    horiz = torch.linalg.norm(asset.data.root_lin_vel_w[:, :2], dim=-1)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    both_air = (torch.norm(forces, dim=-1) <= contact_force_threshold).all(dim=-1)

    reair_fail = base & (post_steps >= int(reair_grace_steps)) & both_air
    horiz_fail = (
        base
        & (post_steps >= int(horiz_grace_steps))
        & (horiz > float(max_horiz_speed))
    )
    return reair_fail | horiz_fail


def heading_blowout(
    env: ManagerBasedRLEnv,
    command_name: str = "jump",
    max_heading_err: float = 1.2,
) -> torch.Tensor:
    """Terminate after liftoff if yaw drifts more than ``max_heading_err`` from episode start."""
    has_liftoff = getattr(env, "_ep_has_liftoff", None)
    if has_liftoff is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    try:
        err = env.command_manager.get_term(command_name).heading_error.abs()
    except (KeyError, AttributeError, RuntimeError):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return has_liftoff & (err > float(max_heading_err))


def extreme_foot_split(
    env: ManagerBasedRLEnv,
    max_sep: float | None = None,
    command_name: str = "jump",
) -> torch.Tensor:
    """Terminate when in-flight foot separation far exceeds the CSV envelope.

    Cap is distance-coupled via :func:`foot_sep_hard` unless ``max_sep`` is set.
    """
    max_sep_ep = getattr(env, "_ep_max_foot_sep_flight", None)
    has_liftoff = getattr(env, "_ep_has_liftoff", None)
    if max_sep_ep is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if max_sep is not None:
        hard = torch.full_like(max_sep_ep, float(max_sep))
    else:
        try:
            flight = env.command_manager.get_term(command_name).flight_distance
            hard = foot_sep_hard(flight)
        except (KeyError, AttributeError, RuntimeError):
            hard = torch.full_like(max_sep_ep, float(getattr(env, "_foot_sep_hard", 1.20)))
    over = max_sep_ep > hard
    if has_liftoff is not None:
        over = over & has_liftoff
    return over


def balloon_height(
    env: ManagerBasedRLEnv,
    command_name: str = "jump",
    extra: float | None = None,
) -> torch.Tensor:
    """Terminate when peak pelvis rise exceeds ``apex_rise* + extra`` (anti-fly).

    Uses pelvis rise (not sole height) so clearance-by-tuck remains legal.
    Defaults to :data:`RISE_HARD_EXTRA` (0.20 m).
    """
    peak_rise = getattr(env, "_ep_peak_rise", None)
    has_liftoff = getattr(env, "_ep_has_liftoff", None)
    if peak_rise is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    hard_extra = float(extra) if extra is not None else float(
        getattr(env, "_rise_hard_extra", RISE_HARD_EXTRA)
    )
    try:
        rise_star = env.command_manager.get_term(command_name).apex_rise_star
    except (KeyError, AttributeError, RuntimeError):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    over = peak_rise > (rise_star + hard_extra)
    if has_liftoff is not None:
        over = over & has_liftoff
    return over
