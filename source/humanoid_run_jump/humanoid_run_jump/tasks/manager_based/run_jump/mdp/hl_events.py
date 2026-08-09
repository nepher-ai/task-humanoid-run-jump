# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Course randomization and robot reset for RunJumpHL."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Re-export jump reset helpers so the env cfg can import from this package.
from humanoid_run_jump.tasks.manager_based.jump.mdp.events import (  # noqa: F401
    reset_from_runin_bank,
    reset_root_and_joints,
)


def _clear_course_buffers(env: ManagerBasedEnv, ids: torch.Tensor, thickness: float) -> None:
    """Reset per-env course / apex / latch bookkeeping for ``ids``."""
    env.obstacle_x[ids] = 0.0
    env.obstacle_h[ids] = 0.0
    env.obstacle_t[ids] = thickness
    env.obstacle_active[ids] = False
    env.num_obstacles[ids] = 0
    env.obstacle_index[ids] = 0
    env.cleared_count[ids] = 0
    env.hit_count[ids] = 0
    env.course_finished[ids] = False
    env.just_cleared[ids] = False
    env.just_hit[ids] = False
    env.crash_hold[ids] = 0
    env.s_max[ids] = 0.0
    env.progress_delta[ids] = 0.0
    env.vx_ema[ids] = 1.5  # neutral run-in speed so the stall term does not fire at t=0
    env.apex_armed[ids] = False
    env.apex_have_peak[ids] = False
    env.apex_peak_z[ids] = 0.0
    env.apex_peak_s[ids] = 0.0
    env.apex_target_s[ids] = 0.0
    env.apex_event[ids] = False
    env.apex_err[ids] = 0.0
    env.stab_timer[ids] = 0
    env.stable_clear_event[ids] = False
    env.latched_this_obstacle[ids] = False
    env.latch_scored[ids] = False
    env.no_jump_event[ids] = False
    env.no_jump_paid[ids] = False


def _write_obstacle_poses(
    env: ManagerBasedEnv,
    ids: torch.Tensor,
    cuboid_half_height: float = 0.75,
    park_z: float = -5.0,
) -> None:
    """Write active / parked cuboid root poses from course buffers."""
    n = len(ids)
    device = env.device
    max_slots = int(env.obstacle_x.shape[1])
    origins = env.scene.env_origins[ids]  # [n, 3]

    for k in range(max_slots):
        name = f"obstacle_{k}"
        if name not in env.scene.keys():
            continue
        obj = env.scene[name]
        pos = origins.clone()
        pos[:, 2] = park_z
        quat = torch.zeros(n, 4, device=device)
        quat[:, 0] = 1.0  # wxyz identity

        active = env.obstacle_active[ids, k]
        if active.any():
            a_ids = active.nonzero(as_tuple=False).squeeze(-1)
            # Front face at obstacle_x, centre at s_f + t/2; bury so top = h.
            t = env.obstacle_t[ids[a_ids], k]
            s_f = env.obstacle_x[ids[a_ids], k]
            h = env.obstacle_h[ids[a_ids], k]
            pos[a_ids, 0] = origins[a_ids, 0] + s_f + 0.5 * t
            pos[a_ids, 1] = origins[a_ids, 1]
            pos[a_ids, 2] = h - cuboid_half_height

        pose = torch.cat([pos, quat], dim=-1)
        obj.write_root_pose_to_sim(pose, env_ids=ids)
        zeros = torch.zeros(n, 6, device=device)
        obj.write_root_velocity_to_sim(zeros, env_ids=ids)

    if hasattr(env, "refresh_course_features"):
        env.refresh_course_features(ids)


def randomize_hl_course(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    path_length_range: tuple[float, float] = (20.0, 30.0),
    first_obstacle_range: tuple[float, float] = (7.0, 10.0),
    gap_range: tuple[float, float] = (8.0, 10.0),
    height_range: tuple[float, float] = (0.20, 0.75),
    thickness: float = 0.20,
    max_obstacles: int = 3,
    min_tail: float = 1.0,
    cuboid_half_height: float = 0.75,
    park_z: float = -5.0,
):
    """Sample a straight +x obstacle course and bury fixed-size cuboids to height.

    Obstacles are spawned as fixed-size ``(thickness, 5.0, 1.5)`` kinematic
    cuboids. Height is set by burial: ``z_center = h - cuboid_half_height`` so
    the top face sits at ``h``. Unused slots park at ``z = park_z``.
    """
    if len(env_ids) == 0:
        return
    ids = env_ids.to(device=env.device, dtype=torch.long)
    n = len(ids)
    device = env.device

    path_lo, path_hi = path_length_range
    first_lo, first_hi = first_obstacle_range
    gap_lo, gap_hi = gap_range
    h_lo, h_hi = height_range

    path_length = torch.empty(n, device=device).uniform_(path_lo, path_hi)
    env.path_length[ids] = path_length
    _clear_course_buffers(env, ids, thickness)

    max_slots = int(env.obstacle_x.shape[1])
    max_obs = min(int(max_obstacles), max_slots)

    # Curriculum may override height band / max obstacles on the env.
    # When the curriculum term is disabled (play), honour the event params so
    # CLI / PLAY cfg can pin height and obstacle count.
    curr = getattr(getattr(env, "cfg", None), "curriculum", None)
    use_curr = curr is not None and getattr(curr, "hl_course", None) is not None
    if use_curr:
        h_lo_eff = float(getattr(env, "_curr_h_lo", h_lo))
        h_hi_eff = float(getattr(env, "_curr_h_hi", h_hi))
        max_obs_eff = int(getattr(env, "_curr_max_obstacles", max_obs))
    else:
        h_lo_eff = float(h_lo)
        h_hi_eff = float(h_hi)
        max_obs_eff = int(max_obs)
    max_obs_eff = max(1, min(max_obs_eff, max_slots))

    for i, eid in enumerate(ids.tolist()):
        pl = float(path_length[i].item())
        x = float(torch.empty(1, device=device).uniform_(first_lo, first_hi).item())
        count = 0
        xs, hs = [], []
        while count < max_obs_eff and (x + min_tail) <= pl:
            h = float(torch.empty(1, device=device).uniform_(h_lo_eff, h_hi_eff).item())
            xs.append(x)
            hs.append(h)
            count += 1
            gap = float(torch.empty(1, device=device).uniform_(gap_lo, gap_hi).item())
            x = x + gap
        env.num_obstacles[eid] = count
        for k in range(count):
            env.obstacle_x[eid, k] = xs[k]
            env.obstacle_h[eid, k] = hs[k]
            env.obstacle_t[eid, k] = thickness
            env.obstacle_active[eid, k] = True

    _write_obstacle_poses(env, ids, cuboid_half_height=cuboid_half_height, park_z=park_z)


def randomize_hl_course_from_envhub(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    preset_cfg,
    cuboid_half_height: float = 0.75,
    park_z: float = -5.0,
    thickness: float = 0.20,
):
    """Place a deterministic EnvHub course into HL buffers and cuboid slots.

    ``preset_cfg`` must expose ``gen_course(env_ids, device)`` returning
    ``(path_length, xs, hs, ts, counts)``.
    """
    if len(env_ids) == 0:
        return
    ids = env_ids.to(device=env.device, dtype=torch.long)
    device = env.device

    path_length, xs, hs, ts, counts = preset_cfg.gen_course(ids, device=device)
    default_t = float(getattr(preset_cfg, "default_thickness", thickness))
    env.path_length[ids] = path_length
    _clear_course_buffers(env, ids, default_t)

    max_slots = int(env.obstacle_x.shape[1])
    cap = min(int(xs.shape[1]), max_slots)
    for i, eid in enumerate(ids.tolist()):
        count = int(min(int(counts[i].item()), cap))
        env.num_obstacles[eid] = count
        for k in range(count):
            env.obstacle_x[eid, k] = xs[i, k]
            env.obstacle_h[eid, k] = hs[i, k]
            env.obstacle_t[eid, k] = ts[i, k]
            env.obstacle_active[eid, k] = True

    _write_obstacle_poses(env, ids, cuboid_half_height=cuboid_half_height, park_z=park_z)
