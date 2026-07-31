# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms: frozen tracker (run) or direct joint PD (jump)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from humanoid_run_jump.robots.g1_constants import ANCHOR_BODY_NAME, ROOT_BODY_NAME
from humanoid_run_jump.robots.joint_order import JointOrderMap
from humanoid_run_jump.tracker.frozen_tracker import FrozenTracker, beyond_mimic_pd_params
from humanoid_run_jump.tracker.reduced_coords import TARGET_FRAME_DIM, TRACKER_ACT_DIM

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class TrackerAction(ActionTerm):
    """Apply a 64-D reduced-coords target frame through the frozen BeyondMimic tracker.

    The RL policy outputs a target frame
    ``[rel_anchor_rot6d(6), dof_vel(29), dof_pos(29)]``. This term:

    1. Builds the 157-D tracker observation from proprio + target + prev action.
    2. Runs the frozen TorchScript policy (no gradients).
    3. Scales the raw output with BeyondMimic PD offsets and writes joint position
       targets to the articulation (remapped to USD joint order).
    """

    cfg: TrackerActionCfg
    _asset_name: str = "robot"

    def __init__(self, cfg: TrackerActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(
            self.num_envs, TRACKER_ACT_DIM, device=self.device
        )
        self._prev_processed = torch.zeros(self.num_envs, TRACKER_ACT_DIM, device=self.device)
        self._joint_targets = torch.zeros(self.num_envs, TRACKER_ACT_DIM, device=self.device)

        self._tracker = FrozenTracker(policy_path=cfg.policy_path, device=self.device)
        self._joint_map = JointOrderMap(list(self._asset.data.joint_names), device=self.device)

        body_names = self._asset.data.body_names
        self._anchor_idx = body_names.index(cfg.anchor_body_name)
        self._root_idx = body_names.index(cfg.root_body_name)

    @property
    def action_dim(self) -> int:
        return TARGET_FRAME_DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        # Clip to keep early training stable.
        self._raw_actions.clamp_(-self.cfg.clip, self.cfg.clip)

        # Tracker + prev-action buffer live in G1_JOINT_NAMES order.
        dof_pos = self._joint_map.to_g1(self._asset.data.joint_pos)
        dof_vel = self._joint_map.to_g1(self._asset.data.joint_vel)
        anchor_quat = self._asset.data.body_quat_w[:, self._anchor_idx]
        root_quat = self._asset.data.body_quat_w[:, self._root_idx]
        root_ang_vel = self._asset.data.root_ang_vel_w

        joint_targets_g1, _raw = self._tracker.infer_joint_targets(
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            anchor_quat_wxyz=anchor_quat,
            root_ang_vel_w=root_ang_vel,
            target_frame=self._raw_actions,
            prev_processed_action=self._prev_processed,
            root_quat_wxyz=root_quat,
        )
        self._joint_targets[:] = self._joint_map.to_sim(joint_targets_g1)
        # ProtoMotions BM stores post-PD joint targets as "processed_actions" (G1 order).
        self._processed_actions[:] = joint_targets_g1
        self._prev_processed[:] = joint_targets_g1

    def apply_actions(self):
        self._asset.set_joint_position_target(self._joint_targets)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._prev_processed[env_ids] = 0.0
        self._joint_targets[env_ids] = 0.0


@configclass
class TrackerActionCfg(ActionTermCfg):
    """Configuration for :class:`TrackerAction`."""

    class_type: type = TrackerAction
    asset_name: str = "robot"
    policy_path: str | None = None
    anchor_body_name: str = ANCHOR_BODY_NAME
    root_body_name: str = ROOT_BODY_NAME
    clip: float = 5.0


class JointPdAction(ActionTerm):
    """Direct 29-D joint PD targets (BeyondMimic scale) — no frozen tracker.

    Policy outputs raw actions in ``G1_JOINT_NAMES`` order. Targets are::

        q = default_pose + (effort / stiffness) * raw

    then remapped to USD joint order for the articulation.
    """

    cfg: JointPdActionCfg

    def __init__(self, cfg: JointPdActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, TRACKER_ACT_DIM, device=self.device)
        self._joint_targets = torch.zeros(self.num_envs, TRACKER_ACT_DIM, device=self.device)
        self._joint_map = JointOrderMap(list(self._asset.data.joint_names), device=self.device)

        offset, scale = beyond_mimic_pd_params(self.device)
        self._pd_offset = offset
        self._pd_scale = scale

    @property
    def action_dim(self) -> int:
        return TRACKER_ACT_DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._raw_actions.clamp_(-self.cfg.clip, self.cfg.clip)
        targets_g1 = self._pd_offset + self._pd_scale * self._raw_actions
        self._processed_actions[:] = targets_g1
        self._joint_targets[:] = self._joint_map.to_sim(targets_g1)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._joint_targets)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._joint_targets[env_ids] = 0.0


@configclass
class JointPdActionCfg(ActionTermCfg):
    """Configuration for :class:`JointPdAction`."""

    class_type: type = JointPdAction
    asset_name: str = "robot"
    clip: float = 5.0
