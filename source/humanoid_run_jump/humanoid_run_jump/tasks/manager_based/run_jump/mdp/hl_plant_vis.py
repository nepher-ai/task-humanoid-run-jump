# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ground markers for the right-foot plant band and the predicted footfall chains."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from humanoid_run_jump.tasks.manager_based.run_jump.mdp.hl_stride import (
    PLANT_BAND_MAX,
    PLANT_BAND_MIN,
    d_star,
    lattice_from_remaining,
    stride_length,
)

# How many future right plants to draw per chain. The 8 m approach horizon holds
# at most ~9 strides at the 0.90 m minimum, so 10 always covers the whole plan.
_CHAIN_STEPS = 10

# Flat green pad covering the full legal plant band along +x.
_PLANT_ZONE_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/HL/plant_zone",
    markers={
        "zone": sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.95, 0.25),
                opacity=0.45,
            ),
        ),
    },
)

# Grid-based d* centre for the current (flight_cand, vx_cmd).
_PLANT_CENTER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/HL/plant_center",
    markers={
        "center": sim_utils.SphereCfg(
            radius=0.07,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.85, 0.1),
                opacity=0.95,
            ),
        ),
    },
)

# Where the right foot lands if the robot keeps its current speed.
_PLANT_NOW_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/HL/plant_now",
    markers={
        "step": sim_utils.SphereCfg(
            radius=0.06,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.20, 0.20),
                opacity=0.85,
            ),
        ),
    },
)

# Where the right foot lands if the robot tracks the lattice speed target.
_PLANT_REQ_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/HL/plant_req",
    markers={
        "step": sim_utils.SphereCfg(
            radius=0.06,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.20, 0.75, 1.0),
                opacity=0.85,
            ),
        ),
    },
)

# Course start line (s = 0) across the corridor.
_START_LINE_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/HL/start_line",
    markers={
        "line": sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 1.0),
                opacity=0.90,
            ),
        ),
    },
)

# Course finish line (s = path_length) across the corridor.
_END_LINE_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/HL/end_line",
    markers={
        "line": sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.35, 0.05),
                opacity=0.90,
            ),
        ),
    },
)


class PlantTargetVisualizer:
    """Draw the plant band, footfall chains, and course start/end lines.

    - Green pad: the legal band ``[PLANT_BAND_MIN, PLANT_BAND_MAX]``.
    - Yellow sphere: grid ``d*`` for the policy's current ``(flight, vx)``.
    - Red spheres: right plants at the robot's **current** stride, held constant.
      The last one before the obstacle is where the foot actually ends up if
      nothing changes, so a red dot outside the green pad is the failure to fix.
    - Cyan spheres: right plants at the **required** stride ``L_req = remaining /
      n_commit``, i.e. the plan the ``lattice_cmd_speed`` reward steers ``vx_cmd``
      toward. Because the plan is deadbeat, the last cyan dot sits exactly on the
      yellow ``d*`` sphere.
    - White stripe: course start line at ``s = 0``.
    - Orange stripe: course finish line at ``s = path_length``.

    Both chains start from the last completed right plant so they stay put while
    the foot is in the air. The gap between the two chains is the stride — and so
    the speed — the policy still has to make up.
    """

    def __init__(
        self,
        env,
        lateral_width: float = 0.55,
        pad_height: float = 0.02,
        chain_offset_y: float = 0.16,
        line_thickness: float = 0.08,
        line_height: float = 0.04,
        show_plant: bool = True,
        show_lines: bool = True,
    ):
        self.env = env
        self.device = env.device
        self.show_plant = bool(show_plant)
        self.show_lines = bool(show_lines)
        self.band_lo = float(PLANT_BAND_MIN)
        self.band_hi = float(PLANT_BAND_MAX)
        self.band_width = self.band_hi - self.band_lo
        self.band_mid = 0.5 * (self.band_lo + self.band_hi)
        self.lateral_width = float(lateral_width)
        self.pad_height = float(pad_height)
        self.chain_offset_y = float(chain_offset_y)
        self.line_thickness = float(line_thickness)
        self.line_height = float(line_height)
        # Full corridor span (same half-width the out-of-path termination uses).
        self.corridor_width = 2.0 * float(getattr(env, "out_of_path_half_width", 2.5))
        self.zone = self.center = self.now = self.req = None
        self.start_line = self.end_line = None
        if self.show_plant:
            self.zone = VisualizationMarkers(_PLANT_ZONE_CFG)
            self.center = VisualizationMarkers(_PLANT_CENTER_CFG)
            self.now = VisualizationMarkers(_PLANT_NOW_CFG)
            self.req = VisualizationMarkers(_PLANT_REQ_CFG)
            for m in (self.zone, self.center, self.now, self.req):
                m.set_visibility(True)
        if self.show_lines:
            self.start_line = VisualizationMarkers(_START_LINE_CFG)
            self.end_line = VisualizationMarkers(_END_LINE_CFG)
            for m in (self.start_line, self.end_line):
                m.set_visibility(True)
        self._quat = torch.zeros(env.num_envs, 4, device=self.device)
        self._quat[:, 0] = 1.0
        self._chain_quat = torch.zeros(env.num_envs * _CHAIN_STEPS, 4, device=self.device)
        self._chain_quat[:, 0] = 1.0

    # -- helpers ------------------------------------------------------------

    def _current_stride(self, v_act: torch.Tensor) -> torch.Tensor:
        """Measured same-foot stride, falling back to the canvas value at ``v_act``."""
        tracker = getattr(self.env, "stride_tracker", None)
        canvas = stride_length(v_act)
        if tracker is None:
            return canvas
        return torch.where(tracker.has_prev, tracker.L_ema, canvas)

    def _chain_start(self, right_s: torch.Tensor) -> torch.Tensor:
        """Last completed right plant in course coords, else the current ankle."""
        start = getattr(self.env, "lattice_plant_s", None)
        return right_s if start is None else start

    def _emit(
        self,
        markers: VisualizationMarkers,
        s: torch.Tensor,
        valid: torch.Tensor,
        y_offset: float,
        z: float,
    ) -> None:
        """Draw one chain of ``(n, K)`` course positions, hiding invalid nodes."""
        origins = self.env.scene.env_origins
        pos = torch.zeros(s.shape[0], s.shape[1], 3, device=self.device)
        pos[..., 0] = origins[:, 0:1] + s
        pos[..., 1] = origins[:, 1:2] + y_offset
        pos[..., 2] = z
        # Sink hidden nodes rather than resizing the instancer every frame.
        pos[..., 2] = torch.where(valid, pos[..., 2], torch.full_like(pos[..., 2], -10.0))
        # Enlarge the final node of the chain: that is the plant that matters.
        is_last = valid & ~torch.cat([valid[:, 1:], torch.zeros_like(valid[:, :1])], dim=1)
        scale = torch.where(is_last, torch.full_like(s, 1.7), torch.ones_like(s))
        scales = scale.reshape(-1, 1).expand(-1, 3).contiguous()
        markers.visualize(
            translations=pos.reshape(-1, 3),
            orientations=self._chain_quat,
            scales=scales,
        )

    # -- main ---------------------------------------------------------------

    def _draw_line(self, markers: VisualizationMarkers, s: torch.Tensor) -> None:
        """Thin stripe across the corridor at course coordinate ``s`` (per env)."""
        origins = self.env.scene.env_origins
        n = self.env.num_envs
        pos = origins.clone()
        pos[:, 0] = origins[:, 0] + s
        pos[:, 1] = origins[:, 1]
        pos[:, 2] = self.line_height * 0.5
        scale = torch.zeros(n, 3, device=self.device)
        scale[:, 0] = self.line_thickness
        scale[:, 1] = self.corridor_width
        scale[:, 2] = self.line_height
        markers.visualize(translations=pos, orientations=self._quat, scales=scale)

    def update(self) -> None:
        env = self.env
        n = env.num_envs

        if self.show_lines:
            path_length = getattr(env, "path_length", None)
            if path_length is None:
                path_length = torch.full((n,), 25.0, device=self.device)
            self._draw_line(self.start_line, torch.zeros(n, device=self.device))
            self._draw_line(self.end_line, path_length)

        if not self.show_plant:
            return

        origins = env.scene.env_origins
        has = env.has_next_obstacle
        front_s = env.next_front_s

        # Pad centred on the band midpoint (s_front - mid).
        pad_s = front_s - self.band_mid
        pos = origins.clone()
        pos[:, 0] = origins[:, 0] + pad_s
        pos[:, 1] = origins[:, 1]
        pos[:, 2] = self.pad_height * 0.5
        pos[~has, 2] = -10.0

        zone_scale = torch.zeros(n, 3, device=self.device)
        zone_scale[:, 0] = self.band_width
        zone_scale[:, 1] = self.lateral_width
        zone_scale[:, 2] = self.pad_height
        zone_scale[~has] = 1e-3
        self.zone.visualize(translations=pos, orientations=self._quat, scales=zone_scale)

        # Yellow sphere at grid d* for the current candidate flight / vx.
        hl = getattr(env, "last_hl_action", None)
        vx = hl[:, 1] if hl is not None else torch.full((n,), 1.5, device=self.device)
        f_cand = getattr(env, "hl_flight_cand", None)
        if f_cand is None:
            f_cand = torch.full((n,), 0.85, device=self.device)
        d_tgt = getattr(env, "lattice_d_target", None)
        if d_tgt is None:
            d_tgt = d_star(env.next_t, f_cand, vx)
        center_pos = origins.clone()
        center_pos[:, 0] = origins[:, 0] + (front_s - d_tgt)
        center_pos[:, 1] = origins[:, 1]
        center_pos[:, 2] = self.pad_height + 0.05
        center_pos[~has, 2] = -10.0
        self.center.visualize(translations=center_pos, orientations=self._quat)

        # --- Footfall chains ------------------------------------------------
        from humanoid_run_jump.tasks.manager_based.jump.mdp.gait import resolve_ankle_body_ids

        asset = env.scene["robot"]
        _, right_id = resolve_ankle_body_ids(env)
        right_s = asset.data.body_pos_w[:, right_id, 0] - origins[:, 0]
        v_act = asset.data.root_lin_vel_b[:, 0]

        L_now = self._current_stride(v_act)
        start_s = self._chain_start(right_s)
        k = torch.arange(1, _CHAIN_STEPS + 1, device=self.device, dtype=L_now.dtype)

        # Current speed: same stride repeated. Keep every plant that still lands
        # short of the front face, so a stride that would put the foot on the box
        # stays visible instead of being clipped away.
        now_s = start_s.unsqueeze(1) + L_now.unsqueeze(1) * k.unsqueeze(0)
        now_valid = has.unsqueeze(1) & (now_s < front_s.unsqueeze(1))

        # Required speed: the committed plan is a constant stride whose n-th plant
        # lands on d*, so the chain is start + j * L_req truncated at n_commit.
        tracker = getattr(env, "stride_tracker", None)
        n_commit = tracker.n_commit if tracker is not None else None
        remaining = getattr(env, "lattice_remaining", None)
        if remaining is None:
            remaining = (front_s - start_s).clamp(min=0.0) - d_tgt
        n_req, L_req, _ = lattice_from_remaining(remaining, L_now, n_commit=n_commit)
        req_s = start_s.unsqueeze(1) + L_req.unsqueeze(1) * k.unsqueeze(0)
        req_valid = has.unsqueeze(1) & (k.unsqueeze(0) <= n_req.unsqueeze(1))

        z = self.pad_height + 0.06
        self._emit(self.now, now_s, now_valid, -self.chain_offset_y, z)
        self._emit(self.req, req_s, req_valid, self.chain_offset_y, z)

    def set_visibility(self, visible: bool) -> None:
        for m in (self.zone, self.center, self.now, self.req, self.start_line, self.end_line):
            if m is not None:
                m.set_visibility(visible)
