# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based RL env with AMP observation buffer + reference motion sampling (run)."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

from humanoid_run_jump.motions.motion_lib import MotionLib
from humanoid_run_jump.robots.joint_order import JointOrderMap
from humanoid_run_jump.tasks.manager_based.run.mdp.observations import amp_obs_single
from humanoid_run_jump.tracker.reduced_coords import quat_apply_inverse, quat_to_tan_norm


def compute_amp_obs_from_motion(
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    key_body_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a single-frame AMP observation from motion tensors.

    Must match :func:`amp_obs_single` (includes 12 key-body channels). Motion
    packs store dofs in ``G1_JOINT_NAMES`` order. ``key_body_pos`` is ``[N, K, 3]``
    already in root-local yaw frame; when None, zeros of shape ``[N, 12]`` are
    appended.
    """
    root_lin_vel_b = quat_apply_inverse(root_rot, root_lin_vel)
    root_ang_vel_b = quat_apply_inverse(root_rot, root_ang_vel)
    parts = [
        dof_pos,
        dof_vel,
        root_pos[:, 2:3],
        quat_to_tan_norm(root_rot),
        root_lin_vel_b,
        root_ang_vel_b,
    ]
    if key_body_pos is None:
        key_flat = torch.zeros(dof_pos.shape[0], 12, device=dof_pos.device, dtype=dof_pos.dtype)
    else:
        key_flat = key_body_pos.reshape(key_body_pos.shape[0], -1)
    parts.append(key_flat)
    return torch.cat(parts, dim=-1)


class RunAmpEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv extended with AMP extras for skrl (run task).

    Required by skrl AMP runner:
      - ``amp_observation_space`` property
      - ``collect_reference_motions(num_samples)``
      - ``extras["amp_obs"]`` each step
    """

    cfg: ManagerBasedRLEnvCfg

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # USD order != G1_JOINT_NAMES; remap all tracker/AMP dof I/O through this map.
        self.joint_order_map = JointOrderMap(
            list(self.scene["robot"].data.joint_names), device=self.device
        )

        self.num_amp_observations: int = getattr(cfg, "num_amp_observations", 2)
        # Infer single-frame dim from a live observation.
        sample = amp_obs_single(self)
        self.amp_obs_frame_dim: int = int(sample.shape[-1])
        self.amp_observation_size: int = self.num_amp_observations * self.amp_obs_frame_dim
        self.amp_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.amp_observation_size,)
        )
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.num_amp_observations, self.amp_obs_frame_dim),
            device=self.device,
        )
        self.amp_observation_buffer[:, 0] = sample

        motion_file = getattr(cfg, "motion_file", None)
        self._motion_lib: MotionLib | None = None
        if motion_file:
            path = Path(motion_file)
            if not path.is_absolute():
                # Resolve relative to project root (task-humanoid-run-jump/).
                # .../tasks/manager_based/run/run_amp_env.py → task-humanoid-run-jump/
                project_root = Path(__file__).resolve().parents[6]
                cand = project_root / path
                path = cand if cand.exists() else Path.cwd() / motion_file
            if not path.exists():
                raise FileNotFoundError(
                    f"[RunAmpEnv] motion_file not found: {path}\n"
                    "Build it with: python scripts/build_amp_dataset.py"
                )
            self._motion_lib = MotionLib(path, device=self.device)

    def step(self, action: torch.Tensor):
        obs, rewards, terminated, truncated, extras = super().step(action)
        # Shift AMP history and write current frame (num_amp_observations is tiny).
        if self.num_amp_observations > 1:
            self.amp_observation_buffer[:, 1:] = self.amp_observation_buffer[:, :-1].clone()
        self.amp_observation_buffer[:, 0] = amp_obs_single(self)
        extras = extras if extras is not None else {}
        extras["amp_obs"] = self.amp_observation_buffer.view(self.num_envs, -1)

        # Promote Episode_Termination floats to 0-dim tensors for skrl logging.
        log = extras.get("log")
        if isinstance(log, dict):
            for k, v in list(log.items()):
                if isinstance(v, (int, float)) and (
                    k.startswith("Episode_Termination/") or k.startswith("Episode_Reward/")
                ):
                    log[k] = torch.tensor(float(v), device=self.device)
        return obs, rewards, terminated, truncated, extras

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        """Sample AMP observation windows from the reference motion dataset.

        History stride matches the env control period (``step_dt``) so reference
        and policy AMP buffers span the same wall-clock interval.
        """
        if self._motion_lib is None:
            raise RuntimeError(
                "[RunAmpEnv] collect_reference_motions called with no "
                "motion library — set cfg.motion_file and rebuild the dataset."
            )

        window = self._motion_lib.sample_window(
            num_samples=num_samples,
            num_frames=self.num_amp_observations,
            current_times=current_times,
            frame_dt=float(self.step_dt),
        )
        amp = compute_amp_obs_from_motion(
            dof_pos=window["dof_pos"],
            dof_vel=window["dof_vel"],
            root_pos=window["root_pos"],
            root_rot=window["root_rot"],
            root_lin_vel=window["root_lin_vel"],
            root_ang_vel=window["root_ang_vel"],
            key_body_pos=window["key_body_pos"],
        )
        return amp.view(num_samples, self.amp_observation_size)
