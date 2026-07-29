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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def update_jump_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Refresh phase machine / latches before rewards. Never terminates.

    Declared first in ``TerminationsCfg`` so :meth:`AmpManagerBasedRLEnv._update_jump_metrics_and_style`
    runs after ``episode_length_buf`` is incremented but before reward / other
    termination terms read the latches.
    """
    fn = getattr(env, "_update_jump_metrics_and_style", None)
    if fn is not None:
        fn()
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def root_height_below(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


def base_height_below(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return root_height_below(env, minimum_height, asset_cfg)


def bad_orientation(
    env: ManagerBasedRLEnv,
    limit_angle: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base tilts more than ``limit_angle`` from upright (rad).

    Uses ``acos(-projected_gravity_b_z)`` so fully inverted poses fail (the old
    ``‖gravity_xy‖`` check is ~0 both upright *and* upside-down).
    """
    asset = env.scene[asset_cfg.name]
    # Body z aligned with -gravity ⇒ upright (angle 0); +gravity ⇒ inverted (π).
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


def jump_success(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """End the episode once tucked apex + distance + soft landing are latched.

    Soft landing requires feet contact near stand height, upright tilt, and
    heading within ~0.35 rad of episode start. Idle standing remains dense +
    sparse bonus, not a hard gate yet.
    """
    ok = getattr(env, "_ep_full_success", None)
    if ok is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return ok


def post_land_timeout(
    env: ManagerBasedRLEnv,
    max_post_land_steps: int = 100,
) -> torch.Tensor:
    """Truncate ~2 s after first landing so settle frames do not dominate the rollout.

    ``time_out=True`` in the env cfg so this is not treated as a failure.
    """
    has_landed = getattr(env, "_ep_has_landed", None)
    post_steps = getattr(env, "_ep_post_land_steps", None)
    if has_landed is None or post_steps is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return has_landed & (post_steps >= int(max_post_land_steps))


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
) -> torch.Tensor:
    """Terminate when in-flight foot separation far exceeds the CSV envelope.

    Defaults to ``env._foot_sep_hard`` (1.05 m) so mild overshoot is only
    penalized, while ballistic splits are hard-failed.
    """
    max_sep_ep = getattr(env, "_ep_max_foot_sep_flight", None)
    has_liftoff = getattr(env, "_ep_has_liftoff", None)
    if max_sep_ep is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    hard = float(max_sep) if max_sep is not None else float(
        getattr(env, "_foot_sep_hard", 1.05)
    )
    over = max_sep_ep > hard
    if has_liftoff is not None:
        over = over & has_liftoff
    return over


def balloon_height(
    env: ManagerBasedRLEnv,
    command_name: str = "jump",
    extra: float | None = None,
) -> torch.Tensor:
    """Terminate when peak sole clearance exceeds ``h + extra`` (anti-fly).

    Defaults to ``env._clearance_hard_extra`` (0.25 m).
    """
    peak = getattr(env, "_ep_peak_foot_z", None)
    has_liftoff = getattr(env, "_ep_has_liftoff", None)
    if peak is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    hard_extra = float(extra) if extra is not None else float(
        getattr(env, "_clearance_hard_extra", 0.25)
    )
    try:
        h = env.command_manager.get_term(command_name).h_obstacle
    except (KeyError, AttributeError, RuntimeError):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    over = peak.max(dim=-1).values > (h + hard_extra)
    if has_liftoff is not None:
        over = over & has_liftoff
    return over
