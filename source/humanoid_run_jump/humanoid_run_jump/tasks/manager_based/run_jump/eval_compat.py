# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation compatibility wrapper for the G1 RunJumpHL task.

Raw-state sampler for eval-nav. Derivation, event detection, and scoring live
entirely in eval-nav — this wrapper only reads simulator state.

Exposes:
  ``task_completed`` / ``task_failed`` – termination truth
  ``get_raw_state``                    – batched per-step raw physics (num_envs, …)
  ``_log_state``                       – per-env projection of the same raw dict
  ``_log_metadata``                    – episode-level course geometry + origins
  ``wrap_for_eval``                    – factory used by EnvironmentManager
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

RAW_STATE_VERSION = 1

_FAILURE_TERMS = (
    "obstacle_crash",
    "out_of_path",
    "behind_start",
    "no_progress",
    "root_height",
    "bad_orientation",
)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


class EvalCompatEnv:
    """Thin wrapper that surfaces raw RunJumpHL env state for eval-nav."""

    def __init__(self, env: "ManagerBasedRLEnv"):
        self._env = env
        self._cached_raw: dict[str, np.ndarray] | None = None
        self._sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
        try:
            self._sensor_cfg.resolve(self._env.unwrapped.scene)
        except Exception:
            pass
        self._ankle_body_names: list[str] = ["left_ankle_roll_link", "right_ankle_roll_link"]
        self._contact_body_names: list[str] = list(self._ankle_body_names)

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
        self._cached_raw = None
        return self._env.reset(*args, **kwargs)

    def step(self, action):
        return self._env.step(action)

    def close(self):
        return self._env.close()

    def render(self, *args, **kwargs):
        if hasattr(self._env, "render"):
            return self._env.render(*args, **kwargs)

    # ------------------------------------------------------------------
    # Raw state (batched; leading dim = num_envs)
    # ------------------------------------------------------------------

    def get_raw_state(self) -> dict[str, np.ndarray]:
        """Per-step raw simulator state. No derived / normalized values."""
        env = self._env.unwrapped
        robot = env.scene["robot"]
        n = int(env.num_envs)

        from humanoid_run_jump.tasks.manager_based.jump.mdp.gait import (
            _contact_ankle_columns,
            resolve_ankle_body_ids,
        )

        left_id, right_id = resolve_ankle_body_ids(env)
        ankle_pos = robot.data.body_pos_w[:, [left_id, right_id], :]

        foot_forces = torch.zeros(n, 2, 3, device=env.device)
        try:
            contact_sensor = env.scene.sensors[self._sensor_cfg.name]
            forces = contact_sensor.data.net_forces_w[:, self._sensor_cfg.body_ids, :]
            left_col, right_col = _contact_ankle_columns(env, self._sensor_cfg)
            foot_forces = torch.stack([forces[:, left_col], forces[:, right_col]], dim=1)
        except Exception:
            pass

        hl = getattr(env, "last_hl_action", None)
        if hl is None:
            hl_action = np.zeros((n, 1), dtype=np.float32)
        else:
            hl_action = _to_numpy(hl).astype(np.float32, copy=False)

        hl_mode = getattr(env, "hl_mode", None)
        if hl_mode is None:
            mode_arr = np.zeros(n, dtype=np.int64)
        else:
            mode_arr = _to_numpy(hl_mode.long()).astype(np.int64, copy=False)

        hl_mode_steps = getattr(env, "hl_mode_steps", None)
        if hl_mode_steps is None:
            mode_steps_arr = np.zeros(n, dtype=np.float32)
        else:
            mode_steps_arr = _to_numpy(hl_mode_steps).astype(np.float32, copy=False)

        def _counter(name: str) -> np.ndarray:
            t = getattr(env, name, None)
            if t is None:
                return np.zeros(n, dtype=np.int64)
            return _to_numpy(t.long()).astype(np.int64, copy=False)

        grav = getattr(robot.data, "projected_gravity_b", None)
        if grav is None:
            grav_arr = np.zeros((n, 3), dtype=np.float32)
        else:
            grav_arr = _to_numpy(grav).astype(np.float32, copy=False)

        raw = {
            "root_pos_w": _to_numpy(robot.data.root_pos_w).astype(np.float32, copy=False),
            "root_quat_w": _to_numpy(robot.data.root_quat_w).astype(np.float32, copy=False),
            "root_lin_vel_w": _to_numpy(robot.data.root_lin_vel_w).astype(np.float32, copy=False),
            "root_lin_vel_b": _to_numpy(robot.data.root_lin_vel_b).astype(np.float32, copy=False),
            "root_ang_vel_b": _to_numpy(robot.data.root_ang_vel_b).astype(np.float32, copy=False),
            "projected_gravity_b": grav_arr,
            "ankle_pos_w": _to_numpy(ankle_pos).astype(np.float32, copy=False),
            "foot_contact_force_w": _to_numpy(foot_forces).astype(np.float32, copy=False),
            "hl_action": hl_action,
            "hl_mode": mode_arr,
            "hl_mode_steps": mode_steps_arr,
            "obstacle_index": _counter("obstacle_index"),
            "cleared_count": _counter("cleared_count"),
            "hit_count": _counter("hit_count"),
        }
        self._cached_raw = raw
        return raw

    def _log_state(
        self,
        env_idx: int | None = None,
        info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Per-env projection of ``get_raw_state`` plus task status flags."""
        idx = env_idx if env_idx is not None else 0
        state: dict[str, Any] = {}
        try:
            raw = self._cached_raw if getattr(self, "_cached_raw", None) is not None else self.get_raw_state()
            for key, arr in raw.items():
                state[key] = arr[idx]

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
            origin = env.scene.env_origins[idx].detach().cpu().numpy().astype(np.float32)
            path_length = float(env.path_length[idx].cpu().item()) if hasattr(env, "path_length") else 0.0
            num_obstacles = int(env.num_obstacles[idx].cpu().item()) if hasattr(env, "num_obstacles") else 0
            xs, hs, ts = [], [], []
            if hasattr(env, "obstacle_x") and hasattr(env, "obstacle_h"):
                for k in range(num_obstacles):
                    xs.append(float(env.obstacle_x[idx, k].cpu().item()))
                    hs.append(float(env.obstacle_h[idx, k].cpu().item()))
                    if hasattr(env, "obstacle_t"):
                        ts.append(float(env.obstacle_t[idx, k].cpu().item()))
                    else:
                        ts.append(0.20)
            step_dt = float(getattr(env, "step_dt", 0.02) or 0.02)
            half_w = float(getattr(env, "out_of_path_half_width", 2.5))
            return {
                "env_origin": origin,
                "path_length": path_length,
                "num_obstacles": num_obstacles,
                "obstacle_xs": xs,
                "obstacle_hs": hs,
                "obstacle_ts": ts,
                "out_of_path_half_width": half_w,
                "step_dt": step_dt,
                "raw_state_version": RAW_STATE_VERSION,
                "ankle_body_names": list(self._ankle_body_names),
                "contact_body_names": list(self._contact_body_names),
            }
        except Exception:
            return None


def wrap_for_eval(env: "ManagerBasedRLEnv") -> EvalCompatEnv:
    """Wrap a RunJumpHL env (or any ManagerBasedRLEnv) for evaluation."""
    return EvalCompatEnv(env)
