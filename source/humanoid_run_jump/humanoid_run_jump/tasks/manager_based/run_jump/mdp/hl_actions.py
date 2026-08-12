# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hierarchical switcher: 6-D HL action → frozen run/jump actors → tracker."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from humanoid_run_jump.agents.frozen_actor import FrozenActor
from humanoid_run_jump.robots.g1_constants import ANCHOR_BODY_NAME, ROOT_BODY_NAME
from humanoid_run_jump.robots.joint_order import JointOrderMap
from humanoid_run_jump.tasks.manager_based.jump.mdp.gait import (
    foot_contact_mask,
    resolve_ankle_body_ids,
    runin_handoff_mask,
)
from humanoid_run_jump.tasks.manager_based.jump.mdp.jump_envelope import (
    FLIGHT_DIST_MAX,
    H_MAX,
    H_MIN,
    clamp_obstacle_height,
    flight_cap,
)
from humanoid_run_jump.tasks.manager_based.jump.mdp.observations import (
    base_height,
    base_lin_vel_z,
    foot_air_time,
    foot_contacts,
    jump_commands,
    reduced_coords_proprio,
)
from humanoid_run_jump.tasks.manager_based.run.mdp.observations import base_lin_vel
from humanoid_run_jump.tasks.manager_based.run_jump.mdp.hl_stride import (
    FLIGHT_HL_MIN,
    H_MARGIN_MAX,
    H_MARGIN_MIN,
)
from humanoid_run_jump.tracker.frozen_tracker import FrozenTracker
from humanoid_run_jump.tracker.reduced_coords import TARGET_FRAME_DIM, TRACKER_ACT_DIM

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

MODE_RUN = 0
MODE_JUMP = 1

RUN_OBS_DIM = 134
JUMP_OBS_DIM = 156
HL_ACT_DIM = 6


class HierarchicalSwitchAction(ActionTerm):
    """Map 6-D HL action to frozen run/jump actors + frozen tracker.

    Action layout (after tanh)::

        [gate, vx, vy, wz, a_h, a_flight]

    RUN (gate ≤ 0): write velocity command, run actor → 64-D frame → tracker.
    RUN→JUMP: rising gate + right-plant hand-off; set JumpCommand; re-arm FSM.
    JUMP: jump actor → 64-D frame → tracker; exit after stable landing contact.
    """

    cfg: HierarchicalSwitchActionCfg

    def __init__(self, cfg: HierarchicalSwitchActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._raw_actions = torch.zeros(self.num_envs, HL_ACT_DIM, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, TRACKER_ACT_DIM, device=self.device)
        self._joint_targets = torch.zeros(self.num_envs, TRACKER_ACT_DIM, device=self.device)

        self._tracker = FrozenTracker(policy_path=cfg.tracker_path, device=self.device)
        self._run_actor = FrozenActor(
            policy_path=cfg.run_policy_path,
            device=self.device,
            expected_obs_dim=RUN_OBS_DIM,
            expected_act_dim=TARGET_FRAME_DIM,
        )
        self._jump_actor = FrozenActor(
            policy_path=cfg.jump_policy_path,
            device=self.device,
            expected_obs_dim=JUMP_OBS_DIM,
            expected_act_dim=TARGET_FRAME_DIM,
        )
        self._joint_map = JointOrderMap(list(self._asset.data.joint_names), device=self.device)

        body_names = self._asset.data.body_names
        self._anchor_idx = body_names.index(cfg.anchor_body_name)
        self._root_idx = body_names.index(cfg.root_body_name)

        # LL last-action buffers (NOT the 6-D HL action).
        self._prev_frame = torch.zeros(self.num_envs, TARGET_FRAME_DIM, device=self.device)
        self._prev_processed = torch.zeros(self.num_envs, TRACKER_ACT_DIM, device=self.device)

        # Mode / latch state mirrored onto env for obs / rewards.
        self._mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._mode_steps = torch.zeros(self.num_envs, device=self.device)
        self._prev_gate_pos = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._land_contact_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._d_latch = torch.zeros(self.num_envs, device=self.device)
        self._just_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hl_scaled = torch.zeros(self.num_envs, HL_ACT_DIM, device=self.device)
        self._hl_tanh = torch.zeros(self.num_envs, HL_ACT_DIM, device=self.device)
        self._repeat_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._latched_hl = torch.zeros(self.num_envs, HL_ACT_DIM, device=self.device)
        self._latched_tanh = torch.zeros(self.num_envs, HL_ACT_DIM, device=self.device)
        self._vx_at_latch = torch.zeros(self.num_envs, device=self.device)

        # Publish onto env for other MDP terms.
        env.hl_mode = self._mode
        env.hl_mode_steps = self._mode_steps
        env.hl_d_latch = self._d_latch
        env.hl_just_latched = self._just_latched
        env.last_hl_action = self._hl_scaled
        env.hl_action_tanh = self._hl_tanh
        env.hl_vx_at_latch = self._vx_at_latch
        env.hl_switch_action = self  # back-reference

        self._sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
        try:
            self._sensor_cfg.resolve(env.scene)
        except Exception:
            pass

    @property
    def action_dim(self) -> int:
        return HL_ACT_DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _scale_hl(self, raw: torch.Tensor) -> torch.Tensor:
        """tanh → physical ranges for [gate, vx, vy, wz, a_h, a_flight]."""
        t = torch.tanh(raw)
        gate = t[:, 0]
        vx = 0.5 * (t[:, 1] + 1.0) * (self.cfg.vx_max - self.cfg.vx_min) + self.cfg.vx_min
        vy = t[:, 2] * self.cfg.vy_max
        wz = t[:, 3] * self.cfg.wz_max
        a_h = t[:, 4]
        a_f = t[:, 5]
        return torch.stack([gate, vx, vy, wz, a_h, a_f], dim=-1)

    def _build_run_obs(self) -> torch.Tensor:
        proprio = reduced_coords_proprio(self._env)
        lin_vel = base_lin_vel(self._env)
        vel_cmd = self._env.command_manager.get_command("base_velocity")
        return torch.cat([proprio, lin_vel, vel_cmd, self._prev_frame], dim=-1)

    def _build_jump_obs(self) -> torch.Tensor:
        proprio = reduced_coords_proprio(self._env)
        cmds = jump_commands(self._env, command_name="jump")
        contacts = foot_contacts(self._env, sensor_cfg=self._sensor_cfg)
        air = foot_air_time(self._env, sensor_cfg=self._sensor_cfg)
        height = base_height(self._env)
        vz = base_lin_vel_z(self._env)
        return torch.cat([proprio, cmds, contacts, air, height, vz, self._prev_frame], dim=-1)

    def _apply_tracker(self, frame: torch.Tensor) -> None:
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
            target_frame=frame,
            prev_processed_action=self._prev_processed,
            root_quat_wxyz=root_quat,
        )
        self._joint_targets[:] = self._joint_map.to_sim(joint_targets_g1)
        self._processed_actions[:] = joint_targets_g1
        self._prev_processed[:] = joint_targets_g1
        self._prev_frame[:] = frame

    def _right_standoff(self) -> torch.Tensor:
        origins = self._env.scene.env_origins
        _, right_id = resolve_ankle_body_ids(self._env)
        right_s = self._asset.data.body_pos_w[:, right_id, 0] - origins[:, 0]
        s_f = getattr(self._env, "next_front_s", torch.zeros(self.num_envs, device=self.device))
        return s_f - right_s

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._just_latched[:] = False

        # Action repeat: latch HL decision for action_repeat control steps.
        new_decision = self._repeat_counter <= 0
        if new_decision.any():
            ids = new_decision.nonzero(as_tuple=False).squeeze(-1)
            self._latched_hl[ids] = self._scale_hl(self._raw_actions[ids])
            self._latched_tanh[ids] = torch.tanh(self._raw_actions[ids])
            self._repeat_counter[ids] = int(self.cfg.action_repeat)
        self._repeat_counter -= 1
        hl = self._latched_hl
        self._hl_scaled[:] = hl
        self._hl_tanh[:] = self._latched_tanh

        gate = hl[:, 0]
        vx, vy, wz = hl[:, 1], hl[:, 2], hl[:, 3]
        a_h, a_f = hl[:, 4], hl[:, 5]
        gate_pos = gate > 0.0
        rising = gate_pos & (~self._prev_gate_pos)

        # Candidate (h, flight) every step so dense rewards can score a_h / a_f
        # before the one-shot latch payout.
        h_obs_all = getattr(
            self._env, "next_h", torch.zeros(self.num_envs, device=self.device)
        )
        h_cand = clamp_obstacle_height(
            h_obs_all
            + self.cfg.h_margin_min
            + 0.5 * (a_h + 1.0) * (self.cfg.h_margin_max - self.cfg.h_margin_min)
        ).clamp(min=H_MIN, max=H_MAX)
        f_hi = flight_cap(h_cand).clamp(max=FLIGHT_DIST_MAX)
        f_cand = FLIGHT_HL_MIN + 0.5 * (a_f + 1.0) * (f_hi - FLIGHT_HL_MIN)
        if not hasattr(self._env, "hl_h_cand"):
            self._env.hl_h_cand = torch.zeros(self.num_envs, device=self.device)
            self._env.hl_flight_cand = torch.zeros(self.num_envs, device=self.device)
        self._env.hl_h_cand[:] = h_cand
        self._env.hl_flight_cand[:] = f_cand

        # Write velocity command every step (run actor reads it).
        vel_term = self._env.command_manager.get_term("base_velocity")
        if hasattr(vel_term, "set_velocity"):
            vel_term.set_velocity(vx, vy, wz)

        is_run = self._mode == MODE_RUN
        is_jump = self._mode == MODE_JUMP

        # --- RUN → JUMP latch ---
        handoff = runin_handoff_mask(
            self._env,
            lead=self.cfg.handoff_lead,
            sensor_cfg=self._sensor_cfg,
        )
        has_obs = getattr(
            self._env,
            "has_next_obstacle",
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )
        # Gate on standoff: wider than the reward band so near-misses are still
        # scored (and penalised) rather than silently blocked.
        d_all = self._right_standoff()
        in_latch_band = (d_all > self.cfg.latch_d_min) & (d_all < self.cfg.latch_d_max)
        latch = is_run & rising & handoff & has_obs & in_latch_band
        if latch.any():
            ids = latch.nonzero(as_tuple=False).squeeze(-1)
            d = d_all[ids]
            self._d_latch[ids] = d
            self._just_latched[ids] = True

            h_cmd = h_cand[ids]
            f_cmd = f_cand[ids]

            jump_term = self._env.command_manager.get_term("jump")
            jump_term.set_command(ids, h_cmd, f_cmd)

            # Re-arm jump FSM for these envs.
            fresh = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            fresh[ids] = True
            if hasattr(self._env, "_reset_fresh_episode_buffers"):
                self._env._reset_fresh_episode_buffers(fresh)

            self._mode[ids] = MODE_JUMP
            self._mode_steps[ids] = 0
            self._land_contact_steps[ids] = 0
            if not hasattr(self._env, "hl_h_cmd"):
                self._env.hl_h_cmd = torch.zeros(self.num_envs, device=self.device)
                self._env.hl_flight_cmd = torch.zeros(self.num_envs, device=self.device)
            self._env.hl_h_cmd[ids] = h_cmd
            self._env.hl_flight_cmd[ids] = f_cmd
            self._vx_at_latch[ids] = vx[ids]

        # --- JUMP → RUN exit ---
        if is_jump.any():
            contacts = foot_contact_mask(self._env, sensor_cfg=self._sensor_cfg)
            both = contacts[:, 0] & contacts[:, 1]
            landed = getattr(self._env, "_ep_has_landed", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
            in_jump = is_jump & landed
            self._land_contact_steps = torch.where(
                in_jump & both,
                self._land_contact_steps + 1,
                torch.where(in_jump, self._land_contact_steps, torch.zeros_like(self._land_contact_steps)),
            )
            exit_mask = is_jump & (self._land_contact_steps >= int(self.cfg.jump_exit_contact_steps))
            if exit_mask.any():
                eids = exit_mask.nonzero(as_tuple=False).squeeze(-1)
                self._mode[eids] = MODE_RUN
                self._mode_steps[eids] = 0
                self._land_contact_steps[eids] = 0

        self._mode_steps += 1
        self._prev_gate_pos[:] = gate_pos

        # --- Build LL obs and run actors ---
        is_run = self._mode == MODE_RUN
        is_jump = self._mode == MODE_JUMP
        frame = torch.zeros(self.num_envs, TARGET_FRAME_DIM, device=self.device)

        if is_run.any():
            run_obs = self._build_run_obs()
            with torch.inference_mode():
                run_out = self._run_actor.act(run_obs)
            frame = torch.where(is_run.unsqueeze(-1), run_out, frame)

        if is_jump.any():
            jump_obs = self._build_jump_obs()
            with torch.inference_mode():
                jump_out = self._jump_actor.act(jump_obs)
            frame = torch.where(is_jump.unsqueeze(-1), jump_out, frame)

        self._apply_tracker(frame)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._joint_targets)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._joint_targets[env_ids] = 0.0
        self._prev_frame[env_ids] = 0.0
        self._prev_processed[env_ids] = 0.0
        self._mode[env_ids] = MODE_RUN
        self._mode_steps[env_ids] = 0
        self._prev_gate_pos[env_ids] = False
        self._land_contact_steps[env_ids] = 0
        self._d_latch[env_ids] = 0.0
        self._just_latched[env_ids] = False
        self._hl_scaled[env_ids] = 0.0
        self._hl_tanh[env_ids] = 0.0
        self._repeat_counter[env_ids] = 0
        self._latched_hl[env_ids] = 0.0
        self._latched_tanh[env_ids] = 0.0
        self._vx_at_latch[env_ids] = 0.0
        if hasattr(self._env, "hl_h_cmd"):
            self._env.hl_h_cmd[env_ids] = 0.0
            self._env.hl_flight_cmd[env_ids] = 0.0
        if hasattr(self._env, "hl_h_cand"):
            self._env.hl_h_cand[env_ids] = 0.0
            self._env.hl_flight_cand[env_ids] = 0.0


@configclass
class HierarchicalSwitchActionCfg(ActionTermCfg):
    """Configuration for :class:`HierarchicalSwitchAction`."""

    class_type: type = HierarchicalSwitchAction
    asset_name: str = "robot"
    tracker_path: str | None = None  # → frozen_policies/tracker.pt
    run_policy_path: str | None = None  # → best_policy/run/policy.pt
    jump_policy_path: str | None = "best_policy/jump/policy.pt"
    anchor_body_name: str = ANCHOR_BODY_NAME
    root_body_name: str = ROOT_BODY_NAME
    action_repeat: int = 5
    handoff_lead: str = "right"
    jump_exit_contact_steps: int = 3
    vx_min: float = 0.8
    vx_max: float = 3.0
    vy_max: float = 0.5
    wz_max: float = 0.8
    # Hard latch gate (wider than the reward band [0.90, 1.20] so near-misses
    # are still scored by plant_in_band / jump_cmd_match).
    latch_d_min: float = 0.75
    latch_d_max: float = 1.45
    # Direct (h, flight) authority: policy maps a_h / a_f over these bands.
    h_margin_min: float = H_MARGIN_MIN
    h_margin_max: float = H_MARGIN_MAX
