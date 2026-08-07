# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terminations and course-state updates for RunJumpHL."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

from humanoid_run_jump.tasks.manager_based.jump.mdp.gait import resolve_ankle_body_ids
from humanoid_run_jump.tasks.manager_based.jump.mdp.terminations import (  # noqa: F401
    update_jump_state,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

MODE_JUMP = 1


def update_course_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Advance obstacle index, detect clear/hit, refresh features. Never terminates."""
    if hasattr(env, "step_course_state"):
        env.step_course_state()
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def out_of_path(
    env: ManagerBasedRLEnv,
    half_width: float = 2.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the root leaves the corridor laterally."""
    asset = env.scene[asset_cfg.name]
    y = asset.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return y.abs() > half_width


def course_finished(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Success truncate when the robot reaches the path end."""
    finished = getattr(env, "course_finished", None)
    if finished is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return finished


def behind_start(
    env: ManagerBasedRLEnv,
    margin: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    s = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return s < -margin


def no_progress(
    env: ManagerBasedRLEnv,
    window_s: float = 4.0,
    min_delta: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Fail if root has not advanced along +x by ``min_delta`` over ``window_s``."""
    asset = env.scene[asset_cfg.name]
    s = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    if not hasattr(env, "_prog_s_ref"):
        env._prog_s_ref = s.clone()
        env._prog_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    dt = getattr(env, "step_dt", 0.02)
    window_steps = max(1, int(window_s / max(dt, 1e-3)))
    env._prog_steps += 1
    timed = env._prog_steps >= window_steps
    stalled = timed & ((s - env._prog_s_ref) < min_delta)
    # Refresh reference for envs that passed the window.
    if timed.any():
        ids = timed.nonzero(as_tuple=False).squeeze(-1)
        env._prog_s_ref[ids] = s[ids]
        env._prog_steps[ids] = 0
    return stalled


def root_height_below(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


def bad_orientation_hl(
    env: ManagerBasedRLEnv,
    limit_angle: float = 1.25,
    land_grace_steps: int = 25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Orientation fail, exempt during JUMP and briefly after landing."""
    asset = env.scene[asset_cfg.name]
    cos_tilt = (-asset.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)
    tilted = torch.acos(cos_tilt) > limit_angle

    mode = getattr(env, "hl_mode", None)
    if mode is None:
        return tilted
    in_jump = mode == MODE_JUMP
    # Post-land grace: mode just returned to RUN after jump, mode_steps small.
    mode_steps = getattr(env, "hl_mode_steps", torch.zeros(env.num_envs, device=env.device))
    # Track whether we ever jumped this episode.
    ever_jump = getattr(env, "_ep_ever_jumped", None)
    if ever_jump is None:
        env._ep_ever_jumped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        ever_jump = env._ep_ever_jumped
    ever_jump |= in_jump
    post_land = (~in_jump) & ever_jump & (mode_steps < land_grace_steps)
    exempt = in_jump | post_land
    return tilted & (~exempt)


def obstacle_crash(
    env: ManagerBasedRLEnv,
    hold_steps: int = 12,
    margin_xy: float = 0.03,
    margin_z: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Geometric crash: ankles / knees / pelvis intersecting the active obstacle."""
    if not hasattr(env, "next_front_s"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    asset = env.scene[asset_cfg.name]
    origins = env.scene.env_origins
    has = env.has_next_obstacle
    if not has.any():
        env.crash_hold[:] = 0
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    s_f = env.next_front_s
    t = env.next_t
    h = env.next_h
    s_c = s_f + 0.5 * t
    half = 0.5 * t + margin_xy

    # Body points: ankles, knees (if present), pelvis/root.
    left_id, right_id = resolve_ankle_body_ids(env, asset_cfg.name)
    body_ids = [left_id, right_id]
    names = list(asset.data.body_names)
    for needle in ("left_knee", "right_knee", "pelvis"):
        for i, n in enumerate(names):
            if needle in n and i not in body_ids:
                body_ids.append(i)
                break
    # Always include root.
    pos = asset.data.body_pos_w[:, body_ids, :]  # [N, B, 3]
    s = pos[:, :, 0] - origins[:, 0:1]
    z = pos[:, :, 2]

    in_x = (s - s_c.unsqueeze(-1)).abs() < half.unsqueeze(-1)
    in_z = z < (h.unsqueeze(-1) + margin_z)
    penetrating = has & (in_x & in_z).any(dim=-1)

    # Exempt while airborne over the obstacle during a valid jump.
    mode = getattr(env, "hl_mode", None)
    liftoff = getattr(env, "_ep_has_liftoff", None)
    if mode is not None and liftoff is not None:
        airborne_ok = (mode == MODE_JUMP) & liftoff & (z.min(dim=-1).values > h)
        penetrating = penetrating & (~airborne_ok)

    env.crash_hold = torch.where(
        penetrating,
        env.crash_hold + 1,
        torch.zeros_like(env.crash_hold),
    )
    return env.crash_hold >= hold_steps
