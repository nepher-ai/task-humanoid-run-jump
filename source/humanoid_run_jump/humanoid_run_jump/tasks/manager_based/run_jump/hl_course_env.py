# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RunJumpHL course environment: JumpAmpEnv without AMP, plus obstacle course state."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv

from humanoid_run_jump.tasks.manager_based.jump.jump_amp_env import JumpAmpEnv
from humanoid_run_jump.tasks.manager_based.jump.mdp.gait import (
    foot_contact_mask,
    resolve_ankle_body_ids,
)
from humanoid_run_jump.tasks.manager_based.run_jump.mdp.hl_stride import (
    PLANT_EDGE_SIGMA,
    StrideTracker,
    d_star,
)
from isaaclab.managers import SceneEntityCfg

MODE_RUN = 0
MODE_JUMP = 1

# 1 s EMA at the 50 Hz control rate.
_VX_EMA_ALPHA = 0.02
# Steps after a clear at which uprightness is probed (1 s).
_STABILITY_PROBE_STEPS = 50
_STABLE_MIN_HEIGHT = 0.55
_STABLE_MIN_COS_TILT = 0.83  # ≈ 34 deg
_STABLE_MIN_VX = 1.0
# Obstacles above this cannot be stepped over, so failing to gate is a real miss.
_MUST_JUMP_HEIGHT = 0.35
_MUST_JUMP_DEADLINE = 0.90
# Standoff at which the stride plan is latched; matches the reward approach zone.
_APPROACH_HORIZON = 8.0


class HlCourseEnv(JumpAmpEnv):
    """Obstacle-course env for a PPO high-level policy over frozen LL actors.

    Subclasses :class:`JumpAmpEnv` to reuse the jump phase machine / ``_ep_*``
    buffers. AMP is disabled (``motion_file=None``) and :meth:`step` skips the
    AMP buffer write.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        # Ensure AMP is off before JumpAmpEnv.__init__ builds the motion lib.
        if getattr(cfg, "motion_file", None):
            cfg.motion_file = None
        cfg.num_amp_observations = getattr(cfg, "num_amp_observations", 1)

        # ManagerBasedEnv hard-codes InteractiveScene; swap in the anisotropic
        # HL scene for the duration of parent construction.
        import isaaclab.envs.manager_based_env as mbe

        from humanoid_run_jump.tasks.manager_based.run_jump import hl_scene
        from humanoid_run_jump.tasks.manager_based.run_jump.hl_scene import HlInteractiveScene

        hl_scene._PENDING_ENV_SPACING_Y = float(
            getattr(cfg, "env_spacing_y", cfg.scene.env_spacing)
        )
        _orig_scene = mbe.InteractiveScene
        mbe.InteractiveScene = HlInteractiveScene
        try:
            super().__init__(cfg, render_mode=render_mode, **kwargs)
        finally:
            mbe.InteractiveScene = _orig_scene
            hl_scene._PENDING_ENV_SPACING_Y = None

        n = self.num_envs
        max_obs = int(getattr(cfg, "course_max_obstacles", 3))

        self.path_length = torch.full((n,), 25.0, device=self.device)
        self.obstacle_x = torch.zeros(n, max_obs, device=self.device)  # front-face s
        self.obstacle_h = torch.zeros(n, max_obs, device=self.device)
        self.obstacle_t = torch.full((n, max_obs), 0.20, device=self.device)
        self.obstacle_active = torch.zeros(n, max_obs, dtype=torch.bool, device=self.device)
        self.num_obstacles = torch.zeros(n, dtype=torch.long, device=self.device)
        self.obstacle_index = torch.zeros(n, dtype=torch.long, device=self.device)
        self.cleared_count = torch.zeros(n, dtype=torch.long, device=self.device)
        self.hit_count = torch.zeros(n, dtype=torch.long, device=self.device)
        self.course_finished = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.just_cleared = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.just_hit = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.crash_hold = torch.zeros(n, dtype=torch.long, device=self.device)

        # Active-obstacle features.
        self.has_next_obstacle = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.next_front_s = torch.zeros(n, device=self.device)
        self.next_h = torch.zeros(n, device=self.device)
        self.next_t = torch.full((n,), 0.20, device=self.device)
        self.dist_to_next = torch.zeros(n, device=self.device)
        self.obstacle_features = torch.zeros(n, 3, device=self.device)  # h, t, yaw(=0)
        self.remaining_frac = torch.ones(n, device=self.device)

        # Monotone course progress (potential); Δ of this is the dense backbone.
        self.s_max = torch.zeros(n, device=self.device)
        self.progress_delta = torch.zeros(n, device=self.device)
        self.vx_ema = torch.zeros(n, device=self.device)

        # Flight-apex measurement: ground truth for apex-over-obstacle scoring.
        self.apex_armed = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.apex_have_peak = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.apex_peak_z = torch.zeros(n, device=self.device)
        self.apex_peak_s = torch.zeros(n, device=self.device)
        self.apex_target_s = torch.zeros(n, device=self.device)
        self.apex_event = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.apex_err = torch.zeros(n, device=self.device)

        # Post-clear stability probe.
        self.stab_timer = torch.zeros(n, dtype=torch.long, device=self.device)
        self.stable_clear_event = torch.zeros(n, dtype=torch.bool, device=self.device)

        # Per-obstacle latch bookkeeping (stops latch-reward farming).
        self.latched_this_obstacle = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.latch_scored = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.plant_band_scored = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.jump_cmd_scored = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.no_jump_event = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.no_jump_paid = torch.zeros(n, dtype=torch.bool, device=self.device)

        self.stride_tracker = StrideTracker(n, self.device)

        self._sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
        try:
            self._sensor_cfg.resolve(self.scene)
        except Exception:
            pass

        # Half-width for corridor (used by terminations / obs).
        self.out_of_path_half_width = float(getattr(cfg, "out_of_path_half_width", 2.5))

        # Curriculum defaults. ``plant_edge_sigma`` defaults to the tight nominal
        # value so play sessions (no curriculum manager) score the final band.
        self._curr_max_obstacles = max_obs
        self._curr_h_lo = 0.35
        self._curr_h_hi = 0.55
        self.plant_edge_sigma = PLANT_EDGE_SIGMA
        self._hl_curriculum_stage = 0
        self._hl_clear_hist: list[float] = []

        # Optional ground markers: plant band / footfalls and/or start/end lines.
        self._plant_vis = None
        show_plant = bool(getattr(cfg, "show_plant_target", False))
        show_lines = bool(getattr(cfg, "show_course_lines", False))
        if show_plant or show_lines:
            from humanoid_run_jump.tasks.manager_based.run_jump.mdp.hl_plant_vis import (
                PlantTargetVisualizer,
            )

            self._plant_vis = PlantTargetVisualizer(
                self, show_plant=show_plant, show_lines=show_lines
            )

    # ------------------------------------------------------------------
    # Course helpers
    # ------------------------------------------------------------------

    def refresh_course_features(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = env_ids.to(device=self.device, dtype=torch.long)

        idx = self.obstacle_index[ids].clamp(min=0)
        max_slots = self.obstacle_x.shape[1]
        idx = idx.clamp(max=max_slots - 1)
        gather_ids = ids

        # Valid if index < num_obstacles.
        valid = self.obstacle_index[ids] < self.num_obstacles[ids]
        self.has_next_obstacle[ids] = valid

        # Gather per-env active obstacle params.
        s_f = self.obstacle_x[gather_ids, idx]
        h = self.obstacle_h[gather_ids, idx]
        t = self.obstacle_t[gather_ids, idx]
        self.next_front_s[ids] = torch.where(valid, s_f, self.path_length[ids])
        self.next_h[ids] = torch.where(valid, h, torch.zeros_like(h))
        self.next_t[ids] = torch.where(valid, t, torch.full_like(t, 0.20))

        asset = self.scene["robot"]
        origins = self.scene.env_origins
        root_s = asset.data.root_pos_w[ids, 0] - origins[ids, 0]
        self.dist_to_next[ids] = (self.next_front_s[ids] - root_s).clamp(min=-1.0)
        self.obstacle_features[ids, 0] = self.next_h[ids]
        self.obstacle_features[ids, 1] = self.next_t[ids]
        self.obstacle_features[ids, 2] = 0.0
        self.remaining_frac[ids] = (
            1.0 - root_s / self.path_length[ids].clamp(min=1.0)
        ).clamp(0.0, 1.0)

    def step_course_state(self) -> None:
        """Per-step: stride tracker, clear/hit detection, apex probe, feature refresh."""
        self.just_cleared[:] = False
        self.just_hit[:] = False
        self.apex_event[:] = False
        self.stable_clear_event[:] = False
        self.no_jump_event[:] = False

        asset = self.scene["robot"]
        origins = self.scene.env_origins
        left_id, right_id = resolve_ankle_body_ids(self)
        right_x = asset.data.body_pos_w[:, right_id, 0]
        contacts = foot_contact_mask(self, sensor_cfg=self._sensor_cfg)
        self.stride_tracker.update(right_x, contacts[:, 1])

        root_s = asset.data.root_pos_w[:, 0] - origins[:, 0]
        has = self.has_next_obstacle
        s_f = self.next_front_s
        t = self.next_t
        mode = getattr(self, "hl_mode", torch.zeros_like(self.obstacle_index))
        in_jump = mode == MODE_JUMP

        # --- Stride plan: latch a stride count on approach, spend one per plant ---
        # Referenced to the last completed right plant, so the plan holds still
        # while the foot is in the air. StrideTracker stores world x.
        right_s = right_x - origins[:, 0]
        tracker = self.stride_tracker
        self.lattice_plant_s = torch.where(
            tracker.has_last_plant, tracker.last_plant_s - origins[:, 0], right_s
        )
        d_plant = (s_f - self.lattice_plant_s).clamp(min=0.0)
        f_cand = getattr(self, "hl_flight_cand", None)
        if f_cand is None:
            f_cand = torch.full_like(d_plant, 0.85)
        hl_act = getattr(self, "last_hl_action", None)
        vx_cmd = hl_act[:, 1] if hl_act is not None else torch.full_like(d_plant, 1.5)
        self.lattice_d_target = d_star(t, f_cand, vx_cmd)
        self.lattice_remaining = d_plant - self.lattice_d_target
        tracker.update_commitment(
            self.lattice_remaining,
            has & (~in_jump) & (d_plant < _APPROACH_HORIZON),
        )

        # --- Monotone progress potential and smoothed forward speed ---
        self.progress_delta[:] = (root_s - self.s_max).clamp(min=0.0)
        self.s_max = torch.maximum(self.s_max, root_s)
        vx = asset.data.root_lin_vel_w[:, 0]
        self.vx_ema.mul_(1.0 - _VX_EMA_ALPHA).add_(_VX_EMA_ALPHA * vx)

        # --- Flight-apex probe: arm at latch, score when the jump exits ---
        just_latched = getattr(self, "hl_just_latched", None)
        if just_latched is not None and just_latched.any():
            lids = just_latched.nonzero(as_tuple=False).squeeze(-1)
            self.apex_armed[lids] = True
            self.apex_have_peak[lids] = False
            self.apex_peak_z[lids] = 0.0
            self.apex_target_s[lids] = s_f[lids] + 0.5 * t[lids]
            self.latched_this_obstacle[lids] = True

        if self.apex_armed.any():
            ankle_z = asset.data.body_pos_w[:, [left_id, right_id], 2]
            hi_z, _ = ankle_z.max(dim=-1)
            # Canvas PK_X is plant → root at max sole height, so score root_s.
            better = self.apex_armed & in_jump & (hi_z > self.apex_peak_z)
            self.apex_peak_z = torch.where(better, hi_z, self.apex_peak_z)
            self.apex_peak_s = torch.where(better, root_s, self.apex_peak_s)
            self.apex_have_peak |= better

            # Jump finished (mode fell back to RUN): score the apex placement.
            score = self.apex_armed & (~in_jump) & self.apex_have_peak
            if score.any():
                self.apex_err = torch.where(
                    score, (self.apex_peak_s - self.apex_target_s).abs(), self.apex_err
                )
                self.apex_event |= score
            self.apex_armed &= in_jump

        # --- Clear: root past back face of active obstacle without a crash hold ---
        cleared = has & (root_s > s_f + t + 0.05) & (self.crash_hold == 0)
        if cleared.any():
            ids = cleared.nonzero(as_tuple=False).squeeze(-1)
            self.just_cleared[ids] = True
            self.cleared_count[ids] += 1
            self.obstacle_index[ids] += 1
            self.crash_hold[ids] = 0
            self.stab_timer[ids] = _STABILITY_PROBE_STEPS
            self.latched_this_obstacle[ids] = False
            self.latch_scored[ids] = False
            self.no_jump_paid[ids] = False
            self.plant_band_scored[ids] = False
            self.jump_cmd_scored[ids] = False
            # The stride plan and the stride shortfall are both relative to the
            # active obstacle, so they are stale once the target advances.
            self.stride_tracker.reset_commitment(ids)
            if hasattr(self, "_has_stride_err"):
                self._has_stride_err[ids] = False

        # --- Missed jump: front face reached in RUN on an obstacle too tall to step over ---
        should_jump = (
            has
            & (~in_jump)
            & (~self.latched_this_obstacle)
            & (~self.no_jump_paid)
            & (self.next_h > _MUST_JUMP_HEIGHT)
            & (root_s > s_f - _MUST_JUMP_DEADLINE)
        )
        if should_jump.any():
            self.no_jump_event |= should_jump
            self.no_jump_paid |= should_jump

        # --- Post-clear stability probe ---
        probing = self.stab_timer > 0
        if probing.any():
            self.stab_timer = (self.stab_timer - 1).clamp(min=0)
            due = probing & (self.stab_timer == 0)
            if due.any():
                upright = (
                    (asset.data.root_pos_w[:, 2] > _STABLE_MIN_HEIGHT)
                    & ((-asset.data.projected_gravity_b[:, 2]) > _STABLE_MIN_COS_TILT)
                    & (vx > _STABLE_MIN_VX)
                )
                self.stable_clear_event |= due & upright

        # Finished: past path_length and no remaining obstacles.
        past_end = root_s >= self.path_length
        no_more = self.obstacle_index >= self.num_obstacles
        newly = past_end & no_more & (~self.course_finished)
        self.course_finished |= newly

        self.refresh_course_features()

    def step(self, action: torch.Tensor):
        # Skip AMP buffer maintenance from JumpAmpEnv.step.
        obs, rewards, terminated, truncated, extras = ManagerBasedRLEnv.step(self, action)
        done = terminated | truncated
        # Finalize jump success metrics for finished envs (same as JumpAmpEnv).
        if hasattr(self, "_update_flight_success_metrics"):
            self._update_flight_success_metrics(done)
        if done.any():
            ids = done.nonzero(as_tuple=False).squeeze(-1)
            self.stride_tracker.reset(ids)
            self.latch_scored[ids] = False
            self.plant_band_scored[ids] = False
            self.jump_cmd_scored[ids] = False
            if hasattr(self, "_ep_ever_jumped"):
                self._ep_ever_jumped[ids] = False
            if hasattr(self, "_miss_paid"):
                self._miss_paid[ids] = False
            if hasattr(self, "_finish_paid"):
                self._finish_paid[ids] = False
            if hasattr(self, "_has_pred_plant"):
                self._has_pred_plant[ids] = False
            if hasattr(self, "_has_stride_err"):
                self._has_stride_err[ids] = False
            if hasattr(self, "_prog_s_ref"):
                self._prog_s_ref[ids] = (
                    self.scene["robot"].data.root_pos_w[ids, 0] - self.scene.env_origins[ids, 0]
                )
                self._prog_steps[ids] = 0
        # skrl only TensorBoard-logs 0-dim tensors from infos["log"]; Isaac Lab
        # managers emit Python floats for Episode_Termination/* and curriculum stats.
        extras = extras if extras is not None else {}
        log = extras.get("log")
        if isinstance(log, dict):
            for k, v in list(log.items()):
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    log[k] = torch.tensor(float(v), device=self.device)
        if self._plant_vis is not None:
            self._plant_vis.update()
        return obs, rewards, terminated, truncated, extras
