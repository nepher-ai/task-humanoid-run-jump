# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Jump command: (h_obstacle, flight_distance) + CSV takeoff envelope refs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, yaw_quat

from humanoid_run_jump.tasks.manager_based.common.mdp.jump_envelope import (
    FLIGHT_DIST_MAX,
    FLIGHT_DIST_MIN,
    H_MAX,
    apex_rise_target,
    clamp_flight_distance,
    clamp_obstacle_height,
    map_takeoff_targets,
    pitch_apex_target,
    pitch_takeoff_target,
    tuck_apex_target,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class JumpCommand(CommandTerm):
    """Policy-facing jump command (2-D)::

        [h_obstacle, flight_distance]

    Derived (not free RL knobs; used by shaping / metrics)::

        (v_x*, v_y*, v_z*, crouch*) — from :func:`map_takeoff_targets`
        (apex_rise*, tuck_apex*, pitch_takeoff*, pitch_apex*) — envelope refs
    """

    cfg: JumpCommandCfg

    def __init__(self, cfg: JumpCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        # Primary command: [h_obstacle, flight_distance]
        self._command = torch.zeros(self.num_envs, 2, device=self.device)
        self._forward_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._forward_w[:, 0] = 1.0
        self._right_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._right_w[:, 1] = 1.0
        # World XY of the root at command resample (hand-off / episode start).
        self._origin_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self.heading_error = torch.zeros(self.num_envs, device=self.device)
        self.progress = torch.zeros(self.num_envs, device=self.device)
        self.v_x_star = torch.zeros(self.num_envs, device=self.device)
        self.v_y_star = torch.zeros(self.num_envs, device=self.device)
        self.v_z_star = torch.zeros(self.num_envs, device=self.device)
        self.crouch_star = torch.zeros(self.num_envs, device=self.device)
        self.apex_rise_star = torch.zeros(self.num_envs, device=self.device)
        self.tuck_apex_star = torch.zeros(self.num_envs, device=self.device)
        self.pitch_takeoff_star = torch.zeros(self.num_envs, device=self.device)
        self.pitch_apex_star = torch.zeros(self.num_envs, device=self.device)

        self.metrics["h_obstacle"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["flight_distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["heading_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["progress"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def h_obstacle(self) -> torch.Tensor:
        return self._command[:, 0]

    @property
    def flight_distance(self) -> torch.Tensor:
        return self._command[:, 1]

    @property
    def target_vel_b(self) -> torch.Tensor:
        """Mapped takeoff velocity ``[v_x*, v_y*, v_z*]`` in body frame."""
        return torch.stack([self.v_x_star, self.v_y_star, self.v_z_star], dim=-1)

    def __str__(self) -> str:
        return f"JumpCommand(h, flight_dist) n={self.num_envs}"

    def _update_metrics(self):
        self.metrics["h_obstacle"][:] = self._command[:, 0]
        self.metrics["flight_distance"][:] = self._command[:, 1]
        self.metrics["heading_error"][:] = self.heading_error
        self.metrics["progress"][:] = self.progress

    def _heading_axes_from_quat(self, root_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = root_quat.shape[0]
        fwd_l = torch.zeros(n, 3, device=self.device)
        fwd_l[:, 0] = 1.0
        right_l = torch.zeros(n, 3, device=self.device)
        right_l[:, 1] = 1.0
        q_yaw = yaw_quat(root_quat)
        forward = quat_apply(q_yaw, fwd_l)
        right = quat_apply(q_yaw, right_l)
        forward[:, 2] = 0.0
        right[:, 2] = 0.0
        forward = forward / forward.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        right = right / right.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return forward, right

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        if n == 0:
            return
        r = self.cfg.ranges

        h = clamp_obstacle_height(torch.empty(n, device=self.device).uniform_(*r.h_obstacle))
        # Height-coupled clamp: high obstacles cannot request longer flights
        # than the matching CSV bin demonstrates (see jump_envelope.flight_cap).
        flight = clamp_flight_distance(
            torch.empty(n, device=self.device).uniform_(*r.flight_distance), h
        )
        vx_star, vy_star, vz_star, crouch_star = map_takeoff_targets(h)
        rise_star = apex_rise_target(h)
        tuck_star = tuck_apex_target(h)
        pitch_to_star = pitch_takeoff_target(h)
        pitch_ap_star = pitch_apex_target(h)

        root_pos = self.robot.data.root_pos_w[env_ids]
        root_quat = self.robot.data.root_quat_w[env_ids]
        forward, right = self._heading_axes_from_quat(root_quat)
        self._forward_w[env_ids] = forward
        self._right_w[env_ids] = right
        self._origin_pos_w[env_ids] = root_pos

        self._command[env_ids, 0] = h
        self._command[env_ids, 1] = flight

        self.v_x_star[env_ids] = vx_star
        self.v_y_star[env_ids] = vy_star
        self.v_z_star[env_ids] = vz_star
        self.crouch_star[env_ids] = crouch_star
        self.apex_rise_star[env_ids] = rise_star
        self.tuck_apex_star[env_ids] = tuck_star
        self.pitch_takeoff_star[env_ids] = pitch_to_star
        self.pitch_apex_star[env_ids] = pitch_ap_star
        self.progress[env_ids] = 0.0
        self.heading_error[env_ids] = 0.0

    def _update_command(self):
        root = self.robot.data.root_pos_w
        delta = (root - self._origin_pos_w).clone()
        delta[:, 2] = 0.0
        self.progress[:] = (delta * self._forward_w).sum(dim=-1)

        cur_fwd, _ = self._heading_axes_from_quat(self.robot.data.root_quat_w)
        cross_z = cur_fwd[:, 0] * self._forward_w[:, 1] - cur_fwd[:, 1] * self._forward_w[:, 0]
        dot = (cur_fwd[:, :2] * self._forward_w[:, :2]).sum(dim=-1).clamp(-1.0, 1.0)
        self.heading_error[:] = torch.atan2(cross_z, dot)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.35
        goal_vel_w = quat_apply(self.robot.data.root_quat_w, self.target_vel_b)
        goal_scale, goal_quat = self._velocity_w_to_arrow(goal_vel_w, self.goal_vel_visualizer)
        self.goal_vel_visualizer.visualize(base_pos_w, goal_quat, goal_scale)

        cur_scale, cur_quat = self._velocity_w_to_arrow(
            self.robot.data.root_lin_vel_w, self.current_vel_visualizer
        )
        self.current_vel_visualizer.visualize(base_pos_w, cur_quat, cur_scale)

    def _velocity_w_to_arrow(
        self, vel_w: torch.Tensor, visualizer: VisualizationMarkers
    ) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(vel_w.shape[0], 1)
        speed = torch.linalg.norm(vel_w, dim=1)
        arrow_scale[:, 0] *= speed * 0.4

        horiz = torch.sqrt(vel_w[:, 0] ** 2 + vel_w[:, 1] ** 2 + 1e-8)
        pitch = torch.atan2(vel_w[:, 2], horiz)
        yaw = torch.atan2(vel_w[:, 1], vel_w[:, 0])
        zeros = torch.zeros_like(yaw)
        arrow_quat = quat_from_euler_xyz(zeros, -pitch, yaw)
        return arrow_scale, arrow_quat


@configclass
class JumpCommandCfg(CommandTermCfg):
    """Configuration for :class:`JumpCommand`."""

    class_type: type = JumpCommand
    asset_name: str = "robot"
    resampling_time_range: tuple[float, float] = (6.0, 10.0)
    debug_vis: bool = False

    @configclass
    class Ranges:
        """Jump command sampling band (clamped to tracker envelope)."""

        h_obstacle: tuple[float, float] = (0.25, H_MAX)
        # Absolute band; resample also applies height-coupled flight_cap.
        flight_distance: tuple[float, float] = (FLIGHT_DIST_MIN, FLIGHT_DIST_MAX)

    ranges: Ranges = Ranges()

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/jump_goal_vel"
    )
    """Green arrow: CSV-mapped takeoff velocity."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/jump_current_vel"
    )
    """Blue arrow: current root linear velocity."""

    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
