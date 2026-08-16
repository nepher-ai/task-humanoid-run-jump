# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen BeyondMimic tracker: TorchScript tracker.pt -> joint PD targets.

The weights come from a tracker trained in
https://github.com/nepher-ai/humanoid-g1-tracking (export via ``export_jit.py``).

The exported actor maps a 157-D observation to a 29-D raw action. BeyondMimic
PD scaling then converts that to position targets::

    q_target = default_dof_pos + (effort_limit / stiffness) * raw_action

The policy weights are never updated.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from humanoid_run_jump.robots.g1_constants import (
    DEFAULT_JOINT_POS,
    DAMPING_4010,
    DAMPING_5020,
    DAMPING_7520_14,
    DAMPING_7520_22,
    G1_JOINT_NAMES,
    STIFFNESS_4010,
    STIFFNESS_5020,
    STIFFNESS_7520_14,
    STIFFNESS_7520_22,
)
from humanoid_run_jump.tracker.reduced_coords import (
    TRACKER_ACT_DIM,
    TRACKER_OBS_DIM,
    build_tracker_obs,
    compute_reduced_coords_obs,
    encode_target_frame,
)

# Default location relative to the external project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_POLICY_PATH = _PROJECT_ROOT / "frozen_policies" / "tracker.pt"
DEFAULT_META_PATH = _PROJECT_ROOT / "frozen_policies" / "policy_meta.json"

# Explicit per-joint gains / efforts matching G1RobotConfig regex resolution order.
_JOINT_STIFFNESS = {
    "left_hip_pitch_joint": STIFFNESS_7520_14,
    "left_hip_roll_joint": STIFFNESS_7520_22,
    "left_hip_yaw_joint": STIFFNESS_7520_14,
    "left_knee_joint": STIFFNESS_7520_22,
    "left_ankle_pitch_joint": 2 * STIFFNESS_5020,
    "left_ankle_roll_joint": 2 * STIFFNESS_5020,
    "right_hip_pitch_joint": STIFFNESS_7520_14,
    "right_hip_roll_joint": STIFFNESS_7520_22,
    "right_hip_yaw_joint": STIFFNESS_7520_14,
    "right_knee_joint": STIFFNESS_7520_22,
    "right_ankle_pitch_joint": 2 * STIFFNESS_5020,
    "right_ankle_roll_joint": 2 * STIFFNESS_5020,
    "waist_yaw_joint": STIFFNESS_7520_14,
    "waist_roll_joint": 2.0 * STIFFNESS_5020,
    "waist_pitch_joint": 2.0 * STIFFNESS_5020,
    "left_shoulder_pitch_joint": STIFFNESS_5020,
    "left_shoulder_roll_joint": STIFFNESS_5020,
    "left_shoulder_yaw_joint": STIFFNESS_5020,
    "left_elbow_joint": STIFFNESS_5020,
    "left_wrist_roll_joint": STIFFNESS_5020,
    "left_wrist_pitch_joint": STIFFNESS_4010,
    "left_wrist_yaw_joint": STIFFNESS_4010,
    "right_shoulder_pitch_joint": STIFFNESS_5020,
    "right_shoulder_roll_joint": STIFFNESS_5020,
    "right_shoulder_yaw_joint": STIFFNESS_5020,
    "right_elbow_joint": STIFFNESS_5020,
    "right_wrist_roll_joint": STIFFNESS_5020,
    "right_wrist_pitch_joint": STIFFNESS_4010,
    "right_wrist_yaw_joint": STIFFNESS_4010,
}

_JOINT_DAMPING = {
    "left_hip_pitch_joint": DAMPING_7520_14,
    "left_hip_roll_joint": DAMPING_7520_22,
    "left_hip_yaw_joint": DAMPING_7520_14,
    "left_knee_joint": DAMPING_7520_22,
    "left_ankle_pitch_joint": 2 * DAMPING_5020,
    "left_ankle_roll_joint": 2 * DAMPING_5020,
    "right_hip_pitch_joint": DAMPING_7520_14,
    "right_hip_roll_joint": DAMPING_7520_22,
    "right_hip_yaw_joint": DAMPING_7520_14,
    "right_knee_joint": DAMPING_7520_22,
    "right_ankle_pitch_joint": 2 * DAMPING_5020,
    "right_ankle_roll_joint": 2 * DAMPING_5020,
    "waist_yaw_joint": DAMPING_7520_14,
    "waist_roll_joint": 2.0 * DAMPING_5020,
    "waist_pitch_joint": 2.0 * DAMPING_5020,
    "left_shoulder_pitch_joint": DAMPING_5020,
    "left_shoulder_roll_joint": DAMPING_5020,
    "left_shoulder_yaw_joint": DAMPING_5020,
    "left_elbow_joint": DAMPING_5020,
    "left_wrist_roll_joint": DAMPING_5020,
    "left_wrist_pitch_joint": DAMPING_4010,
    "left_wrist_yaw_joint": DAMPING_4010,
    "right_shoulder_pitch_joint": DAMPING_5020,
    "right_shoulder_roll_joint": DAMPING_5020,
    "right_shoulder_yaw_joint": DAMPING_5020,
    "right_elbow_joint": DAMPING_5020,
    "right_wrist_roll_joint": DAMPING_5020,
    "right_wrist_pitch_joint": DAMPING_4010,
    "right_wrist_yaw_joint": DAMPING_4010,
}

_JOINT_EFFORT = {
    "left_hip_pitch_joint": 88.0,
    "left_hip_roll_joint": 139.0,
    "left_hip_yaw_joint": 88.0,
    "left_knee_joint": 139.0,
    "left_ankle_pitch_joint": 50.0,
    "left_ankle_roll_joint": 50.0,
    "right_hip_pitch_joint": 88.0,
    "right_hip_roll_joint": 139.0,
    "right_hip_yaw_joint": 88.0,
    "right_knee_joint": 139.0,
    "right_ankle_pitch_joint": 50.0,
    "right_ankle_roll_joint": 50.0,
    "waist_yaw_joint": 88.0,
    "waist_roll_joint": 50.0,
    "waist_pitch_joint": 50.0,
    "left_shoulder_pitch_joint": 25.0,
    "left_shoulder_roll_joint": 25.0,
    "left_shoulder_yaw_joint": 25.0,
    "left_elbow_joint": 25.0,
    "left_wrist_roll_joint": 25.0,
    "left_wrist_pitch_joint": 5.0,
    "left_wrist_yaw_joint": 5.0,
    "right_shoulder_pitch_joint": 25.0,
    "right_shoulder_roll_joint": 25.0,
    "right_shoulder_yaw_joint": 25.0,
    "right_elbow_joint": 25.0,
    "right_wrist_roll_joint": 25.0,
    "right_wrist_pitch_joint": 5.0,
    "right_wrist_yaw_joint": 5.0,
}


def _default_dof_pos_vector(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Resolve DEFAULT_JOINT_POS regex patterns into a length-29 vector."""
    values = []
    for name in G1_JOINT_NAMES:
        val = 0.0
        for pattern, pos in DEFAULT_JOINT_POS.items():
            if re.fullmatch(pattern, name):
                val = float(pos)
                break
        values.append(val)
    return torch.tensor(values, device=device, dtype=dtype)


def beyond_mimic_pd_params(
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BeyondMimic PD offset / scale in ``G1_JOINT_NAMES`` order (no network).

    Position targets are ``offset + scale * raw`` with
    ``scale = effort_limit / stiffness``.
    """
    device = torch.device(device)
    offset = _default_dof_pos_vector(device, dtype)
    stiffness = torch.tensor(
        [_JOINT_STIFFNESS[n] for n in G1_JOINT_NAMES], device=device, dtype=dtype
    )
    effort = torch.tensor([_JOINT_EFFORT[n] for n in G1_JOINT_NAMES], device=device, dtype=dtype)
    return offset, effort / stiffness


def resolve_policy_path(policy_path: str | Path | None = None) -> Path:
    """Resolve the frozen policy path, raising a clear error if missing."""
    path = Path(policy_path) if policy_path is not None else DEFAULT_POLICY_PATH
    if not path.is_absolute():
        # Try CWD first, then project root.
        candidates = [Path.cwd() / path, _PROJECT_ROOT / path, path]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        path = (_PROJECT_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen tracker policy not found at '{path}'.\n"
            "Copy the exported TorchScript file to frozen_policies/tracker.pt, or run:\n"
            "  python scripts/export_frozen_policy.py "
            "--checkpoint <path-to-last.ckpt>"
        )
    return path


class FrozenTracker(nn.Module):
    """Load and run the frozen BeyondMimic tracker (no gradients)."""

    def __init__(
        self,
        policy_path: str | Path | None = None,
        device: str | torch.device = "cpu",
        meta_path: str | Path | None = None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.policy_path = resolve_policy_path(policy_path)
        self.meta: dict[str, Any] = {}
        meta_file = Path(meta_path) if meta_path else self.policy_path.with_name("policy_meta.json")
        if meta_file.exists():
            with open(meta_file, encoding="utf-8") as f:
                self.meta = json.load(f)

        self.obs_dim = int(self.meta.get("obs_dim", TRACKER_OBS_DIM))
        self.act_dim = int(self.meta.get("act_dim", TRACKER_ACT_DIM))

        self.policy = torch.jit.load(str(self.policy_path), map_location=self.device)
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)

        stiffness = torch.tensor(
            [_JOINT_STIFFNESS[n] for n in G1_JOINT_NAMES], device=self.device, dtype=torch.float32
        )
        effort = torch.tensor(
            [_JOINT_EFFORT[n] for n in G1_JOINT_NAMES], device=self.device, dtype=torch.float32
        )
        self.register_buffer("pd_action_offset", _default_dof_pos_vector(self.device, torch.float32))
        self.register_buffer("action_scale", effort / stiffness)
        self.register_buffer("stiffness", stiffness)
        self.register_buffer(
            "damping",
            torch.tensor([_JOINT_DAMPING[n] for n in G1_JOINT_NAMES], device=self.device, dtype=torch.float32),
        )

    @torch.inference_mode()
    def forward(self, tracker_obs: torch.Tensor) -> torch.Tensor:
        """Raw tracker action ``[N, 29]`` (pre-PD)."""
        if tracker_obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Tracker obs dim mismatch: got {tracker_obs.shape[-1]}, expected {self.obs_dim}"
            )
        return self.policy(tracker_obs)

    @torch.inference_mode()
    def raw_to_joint_targets(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Apply BeyondMimic PD scaling: offset + scale * raw."""
        return self.pd_action_offset + self.action_scale * raw_action

    @torch.inference_mode()
    def infer_joint_targets(
        self,
        dof_pos: torch.Tensor,
        dof_vel: torch.Tensor,
        anchor_quat_wxyz: torch.Tensor,
        root_ang_vel_w: torch.Tensor,
        target_frame: torch.Tensor,
        prev_processed_action: torch.Tensor,
        root_quat_wxyz: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full inference: proprio + target frame -> (joint targets, raw action).

        Returns:
            joint_targets: ``[N, 29]`` PD position targets.
            raw_action: ``[N, 29]`` pre-PD tracker output (store as prev action).
        """
        proprio = compute_reduced_coords_obs(
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            anchor_quat_wxyz=anchor_quat_wxyz,
            root_ang_vel_w=root_ang_vel_w,
            root_quat_wxyz=root_quat_wxyz,
        )
        obs = build_tracker_obs(proprio, target_frame, prev_processed_action)
        raw = self.forward(obs)
        targets = self.raw_to_joint_targets(raw)
        return targets, raw


__all__ = [
    "FrozenTracker",
    "DEFAULT_POLICY_PATH",
    "beyond_mimic_pd_params",
    "resolve_policy_path",
    "encode_target_frame",
    "compute_reduced_coords_obs",
    "build_tracker_obs",
]
