# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation compatibility wrapper for the G1 RunJumpHL task.

Exposes the internal state of the HL course environment through the interface
expected by the eval-nav framework:

  ``task_completed``      – bool tensor (num_envs,): course finished
  ``task_failed``         – bool tensor: crash / path / fall terminations
  ``get_locomotion_data`` – per-step scalars: speed_2d, clearance, impact, …
  ``_log_state``          – dict snapshot for StateLogger
  ``_log_metadata``       – episode-level metadata
  ``wrap_for_eval``       – factory function used by EnvironmentManager
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_FAILURE_TERMS = (
    "obstacle_crash",
    "out_of_path",
    "behind_start",
    "no_progress",
    "root_height",
    "bad_orientation",
)

MODE_RUN = 0
MODE_JUMP = 1

# Landing-impact probe: peak downward root vz over a short post-touchdown window.
IMPACT_WINDOW_STEPS = 8
CLEARANCE_MARGIN_M = 0.15


class EvalCompatEnv:
    """Thin wrapper that surfaces RunJumpHL env state for eval-nav."""

    def __init__(self, env: "ManagerBasedRLEnv"):
        self._env = env
        self._control_steps: int = 0
        self._sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
        try:
            self._sensor_cfg.resolve(self._env.unwrapped.scene)
        except Exception:
            pass
        n = int(self._env.unwrapped.num_envs)
        device = self._env.unwrapped.device
        self._prev_both_contact = torch.zeros(n, dtype=torch.bool, device=device)
        self._impact_timer = torch.zeros(n, dtype=torch.long, device=device)
        self._impact_peak_vz = torch.zeros(n, device=device)
        self._impact_emit = torch.full((n,), -1.0, device=device)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def unwrapped(self):
        return self._env.unwrapped

    # ------------------------------------------------------------------
    # Eval-nav interface: task_completed / task_failed
    # ------------------------------------------------------------------

    @property
    def task_completed(self) -> torch.Tensor:
        """True for each env that finished the obstacle course."""
        env = self._env.unwrapped
        try:
            return env.termination_manager.get_term("course_finished").clone()
        except (AttributeError, KeyError, RuntimeError):
            pass
        finished = getattr(env, "course_finished", None)
        if finished is not None:
            return finished.clone()
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    @property
    def task_failed(self) -> torch.Tensor:
        """True for each env that terminated due to a failure term."""
        env = self._env.unwrapped
        failure = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        try:
            tm = env.termination_manager
            for name in _FAILURE_TERMS:
                try:
                    failure = failure | tm.get_term(name)
                except (KeyError, AttributeError):
                    pass
        except AttributeError:
            pass
        return failure

    # ------------------------------------------------------------------
    # Robot / scene accessors
    # ------------------------------------------------------------------

    @property
    def robot(self):
        return self._env.unwrapped.scene["robot"]

    @property
    def device(self) -> torch.device:
        return self._env.unwrapped.device

    # ------------------------------------------------------------------
    # Gym interface pass-through
    # ------------------------------------------------------------------

    def reset(self, *args, **kwargs):
        out = self._env.reset(*args, **kwargs)
        self._control_steps = 0
        self._reset_impact_probe()
        return out

    def step(self, action):
        out = self._env.step(action)
        self._control_steps += 1
        self._update_impact_probe()
        return out

    def close(self):
        return self._env.close()

    def render(self, *args, **kwargs):
        if hasattr(self._env, "render"):
            return self._env.render(*args, **kwargs)

    # ------------------------------------------------------------------
    # Landing-impact probe (peak downward root vz after JUMP touchdown)
    # ------------------------------------------------------------------

    def _reset_impact_probe(self) -> None:
        env = self._env.unwrapped
        n = int(env.num_envs)
        device = env.device
        self._prev_both_contact = torch.zeros(n, dtype=torch.bool, device=device)
        self._impact_timer = torch.zeros(n, dtype=torch.long, device=device)
        self._impact_peak_vz = torch.zeros(n, device=device)
        self._impact_emit = torch.full((n,), -1.0, device=device)

    def _both_foot_contact(self) -> torch.Tensor:
        from humanoid_run_jump.tasks.manager_based.jump.mdp.gait import foot_contact_mask

        contacts = foot_contact_mask(self._env.unwrapped, sensor_cfg=self._sensor_cfg)
        return contacts[:, 0] & contacts[:, 1]

    def _update_impact_probe(self) -> None:
        """Arm on dual-foot rising edge while JUMP; emit peak -vz after window."""
        env = self._env.unwrapped
        n = int(env.num_envs)
        if self._impact_timer.numel() != n:
            self._reset_impact_probe()

        self._impact_emit.fill_(-1.0)

        mode = getattr(env, "hl_mode", None)
        if mode is None:
            mode = torch.zeros(n, dtype=torch.long, device=env.device)
        else:
            mode = mode.long()

        both = self._both_foot_contact()
        rising = both & (~self._prev_both_contact) & (mode == MODE_JUMP)
        if rising.any():
            self._impact_timer = torch.where(
                rising,
                torch.full_like(self._impact_timer, IMPACT_WINDOW_STEPS),
                self._impact_timer,
            )
            self._impact_peak_vz = torch.where(
                rising, torch.zeros_like(self._impact_peak_vz), self._impact_peak_vz
            )

        active = self._impact_timer > 0
        if active.any():
            vz_w = env.scene["robot"].data.root_lin_vel_w[:, 2]
            down = (-vz_w).clamp(min=0.0)
            self._impact_peak_vz = torch.where(
                active, torch.maximum(self._impact_peak_vz, down), self._impact_peak_vz
            )
            self._impact_timer = torch.where(
                active, self._impact_timer - 1, self._impact_timer
            )
            done = active & (self._impact_timer == 0)
            if done.any():
                self._impact_emit = torch.where(done, self._impact_peak_vz, self._impact_emit)

        self._prev_both_contact = both

    # ------------------------------------------------------------------
    # Locomotion data (consumed by EpisodeRunner for per-step logging)
    # ------------------------------------------------------------------

    def get_locomotion_data(self, env_idx: int | None = None) -> dict[str, float] | None:
        """Return per-step locomotion scalars for env *env_idx*."""
        idx = env_idx if env_idx is not None else 0
        env = self._env.unwrapped
        robot = env.scene["robot"]

        lin_vel = robot.data.root_lin_vel_b[idx]
        ang_vel = robot.data.root_ang_vel_b[idx]
        speed_2d = float(torch.norm(lin_vel[:2]).cpu().item())

        cleared = 0
        obstacle_index = 0
        if hasattr(env, "cleared_count"):
            cleared = int(env.cleared_count[idx].cpu().item())
        if hasattr(env, "obstacle_index"):
            obstacle_index = int(env.obstacle_index[idx].cpu().item())

        action_l2 = 0.0
        hl = getattr(env, "last_hl_action", None)
        if hl is not None:
            action_l2 = float(torch.norm(hl[idx]).cpu().item())

        out: dict[str, float] = {
            "speed_2d": speed_2d,
            "lateral_speed": float(torch.abs(lin_vel[1]).cpu().item()),
            "vertical_speed": float(torch.abs(lin_vel[2]).cpu().item()),
            "yaw_rate": float(torch.abs(ang_vel[2]).cpu().item()),
            "roll_pitch_rate": float(torch.norm(ang_vel[:2]).cpu().item()),
            "cleared_count": float(cleared),
            "obstacle_index": float(obstacle_index),
            "action_l2": action_l2,
        }

        apex_event = getattr(env, "apex_event", None)
        if apex_event is not None and bool(apex_event[idx].cpu().item()):
            peak_z = float(env.apex_peak_z[idx].cpu().item())
            hurdle_h = 0.0
            has_next = getattr(env, "has_next_obstacle", None)
            if has_next is not None and bool(has_next[idx].cpu().item()):
                hurdle_h = float(env.next_h[idx].cpu().item())
            elif hasattr(env, "obstacle_h") and obstacle_index > 0:
                hurdle_h = float(env.obstacle_h[idx, obstacle_index - 1].cpu().item())
            elif hasattr(env, "obstacle_h"):
                hurdle_h = float(env.obstacle_h[idx, max(obstacle_index, 0)].cpu().item())
            clearance_m = peak_z - hurdle_h
            out["apex_clearance_score"] = float(
                max(0.0, min(1.0, clearance_m / CLEARANCE_MARGIN_M))
            )
            out["apex_err"] = float(env.apex_err[idx].cpu().item())

        stable = getattr(env, "stable_clear_event", None)
        if stable is not None and bool(stable[idx].cpu().item()):
            out["stable_clear"] = 1.0

        emit = float(self._impact_emit[idx].cpu().item())
        if emit >= 0.0:
            out["landing_impact_vz"] = emit

        return out

    # ------------------------------------------------------------------
    # State logging (consumed by eval-nav StateLogger)
    # ------------------------------------------------------------------

    def _log_state(
        self,
        env_idx: int | None = None,
        info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state: dict[str, Any] = {}
        idx = env_idx if env_idx is not None else 0
        try:
            env = self._env.unwrapped
            robot = env.scene["robot"]

            pos_w = robot.data.root_pos_w
            state["position"] = (
                pos_w[idx, :3].cpu().numpy() if torch.is_tensor(pos_w) else pos_w[idx, :3]
            )

            quat_w = robot.data.root_quat_w[idx]
            state["quat_w"] = float(quat_w[0].cpu().item())
            state["lin_vel_b"] = robot.data.root_lin_vel_b[idx].cpu().numpy()
            state["ang_vel_b"] = robot.data.root_ang_vel_b[idx].cpu().numpy()

            if hasattr(env, "obstacle_index"):
                state["obstacle_index"] = int(env.obstacle_index[idx].cpu().item())
            if hasattr(env, "cleared_count"):
                state["cleared_count"] = int(env.cleared_count[idx].cpu().item())
            if hasattr(env, "num_obstacles"):
                state["num_obstacles"] = int(env.num_obstacles[idx].cpu().item())
            if hasattr(env, "s_max"):
                state["s_max"] = float(env.s_max[idx].cpu().item())

            for prop_name in ("task_completed", "task_failed"):
                val = getattr(self, prop_name)
                if torch.is_tensor(val):
                    state[prop_name] = (
                        bool(val[idx].cpu().item()) if val.numel() > 1 else bool(val.cpu().item())
                    )
                else:
                    state[prop_name] = bool(val)

            if info:
                for key in ("success", "timeout"):
                    if key in info:
                        v = info[key]
                        if torch.is_tensor(v):
                            state[key] = (
                                float(v[idx].cpu().item()) if v.numel() > 1 else float(v.cpu().item())
                            )
                        else:
                            state[key] = v
        except Exception:
            pass
        return state

    def _log_metadata(self, env_idx: int | None = None) -> dict[str, Any] | None:
        idx = env_idx if env_idx is not None else 0
        try:
            env = self._env.unwrapped
            path_length = float(env.path_length[idx].cpu().item()) if hasattr(env, "path_length") else 0.0
            num_obstacles = int(env.num_obstacles[idx].cpu().item()) if hasattr(env, "num_obstacles") else 0
            xs, hs = [], []
            if hasattr(env, "obstacle_x") and hasattr(env, "obstacle_h"):
                for k in range(num_obstacles):
                    xs.append(float(env.obstacle_x[idx, k].cpu().item()))
                    hs.append(float(env.obstacle_h[idx, k].cpu().item()))
            return {
                "path_length": path_length,
                "num_obstacles": num_obstacles,
                "obstacle_xs": xs,
                "obstacle_hs": hs,
            }
        except Exception:
            return None


def wrap_for_eval(env: "ManagerBasedRLEnv") -> EvalCompatEnv:
    """Wrap a RunJumpHL env (or any ManagerBasedRLEnv) for evaluation."""
    return EvalCompatEnv(env)
