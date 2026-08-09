# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation compatibility wrapper for the G1 RunJumpHL task.

Exposes the internal state of the HL course environment through the interface
expected by the eval-nav framework:

  ``task_completed``      – bool tensor (num_envs,): course finished
  ``task_failed``         – bool tensor: crash / path / fall terminations
  ``get_locomotion_data`` – per-step scalars: speed_2d, yaw_rate, style_score, …
  ``_log_state``          – dict snapshot for StateLogger
  ``_log_metadata``       – episode-level metadata
  ``wrap_for_eval``       – factory function used by EnvironmentManager
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

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


class EvalCompatEnv:
    """Thin wrapper that surfaces RunJumpHL env state for eval-nav."""

    def __init__(self, env: "ManagerBasedRLEnv"):
        self._env = env
        self._run_disc: torch.jit.ScriptModule | None = None
        self._jump_disc: torch.jit.ScriptModule | None = None
        self._style_cache: torch.Tensor | None = None
        self._style_step: int = -1
        self._control_steps: int = 0
        self._load_style_discriminators()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def unwrapped(self):
        return self._env.unwrapped

    # ------------------------------------------------------------------
    # Style discriminators (EnvHub organizer assets)
    # ------------------------------------------------------------------

    def _resolve_preset(self) -> Any:
        cfg = getattr(self._env.unwrapped, "cfg", None)
        if cfg is None:
            return None
        return getattr(cfg, "_envhub_preset", None)

    def _load_style_discriminators(self) -> None:
        """Load run/jump TorchScript discriminators from the EnvHub preset."""
        preset = self._resolve_preset()
        if preset is None:
            raise FileNotFoundError(
                "EvalCompatEnv requires an EnvHub preset on the env cfg "
                "(`cfg._envhub_preset`) to load style discriminators."
            )
        run_path = Path(getattr(preset, "run_discriminator_path", "") or "")
        jump_path = Path(getattr(preset, "jump_discriminator_path", "") or "")
        if not run_path.is_file():
            raise FileNotFoundError(
                f"Missing EnvHub run discriminator: {run_path}\n"
                "Place organizer `run_discriminator.pt` in humanoid-runjump-course-v1."
            )
        if not jump_path.is_file():
            raise FileNotFoundError(
                f"Missing EnvHub jump discriminator: {jump_path}\n"
                "Place organizer `jump_discriminator.pt` in humanoid-runjump-course-v1."
            )
        device = self._env.unwrapped.device
        self._run_disc = torch.jit.load(str(run_path), map_location=device)
        self._jump_disc = torch.jit.load(str(jump_path), map_location=device)
        self._run_disc.eval()
        self._jump_disc.eval()

    def _amp_obs_batch(self) -> torch.Tensor:
        env = self._env.unwrapped
        buf = getattr(env, "amp_observation_buffer", None)
        if buf is None:
            raise RuntimeError(
                "HL env has no amp_observation_buffer; EnvHub Play must set "
                "num_amp_observations=2 and update the buffer each step."
            )
        return buf.view(env.num_envs, -1)

    def _refresh_style_cache(self) -> None:
        """Score all envs once per control step (cached for per-env queries)."""
        if self._style_cache is not None and self._style_step == self._control_steps:
            return
        if self._run_disc is None or self._jump_disc is None:
            raise RuntimeError("Style discriminators are not loaded")

        env = self._env.unwrapped
        amp_obs = self._amp_obs_batch()
        mode = getattr(env, "hl_mode", None)
        if mode is None:
            mode = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        else:
            mode = mode.long()

        with torch.inference_mode():
            run_logits = self._run_disc(amp_obs).view(env.num_envs, -1)[:, 0]
            jump_logits = self._jump_disc(amp_obs).view(env.num_envs, -1)[:, 0]
            logits = torch.where(mode == MODE_JUMP, jump_logits, run_logits)
            style = torch.sigmoid(logits)

        self._style_cache = style
        self._style_step = self._control_steps

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
        self._style_cache = None
        self._style_step = -1
        return out

    def step(self, action):
        out = self._env.step(action)
        self._control_steps += 1
        self._style_cache = None
        return out

    def close(self):
        return self._env.close()

    def render(self, *args, **kwargs):
        if hasattr(self._env, "render"):
            return self._env.render(*args, **kwargs)

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

        self._refresh_style_cache()
        assert self._style_cache is not None
        style_score = float(self._style_cache[idx].cpu().item())
        mode = getattr(env, "hl_mode", None)
        hl_mode = float(mode[idx].cpu().item()) if mode is not None else 0.0

        return {
            "speed_2d": speed_2d,
            "lateral_speed": float(torch.abs(lin_vel[1]).cpu().item()),
            "vertical_speed": float(torch.abs(lin_vel[2]).cpu().item()),
            "yaw_rate": float(torch.abs(ang_vel[2]).cpu().item()),
            "roll_pitch_rate": float(torch.norm(ang_vel[:2]).cpu().item()),
            "cleared_count": float(cleared),
            "obstacle_index": float(obstacle_index),
            "style_score": style_score,
            "hl_mode": hl_mode,
        }

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
