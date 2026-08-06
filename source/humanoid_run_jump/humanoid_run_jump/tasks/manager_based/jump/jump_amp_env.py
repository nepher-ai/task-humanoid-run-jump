# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based RL env with AMP observation buffer + reference motion sampling."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

from humanoid_run_jump.motions.motion_lib import MotionLib
from humanoid_run_jump.robots.joint_order import JointOrderMap
from humanoid_run_jump.tasks.manager_based.jump.mdp.gait import (
    lead_foot_forward_mask,
    resolve_ankle_body_ids,
)
from humanoid_run_jump.tasks.manager_based.jump.mdp.jump_envelope import (
    APEX_ASYM_MIN,
    CLEARANCE_HARD_EXTRA,
    FOOT_SEP_HARD,
    FOOT_SEP_MAX_FLIGHT,
    PITCH_APEX_MAX,
    PITCH_APEX_MIN,
    RISE_HARD_EXTRA,
    foot_sep_flight_max,
)
from humanoid_run_jump.tasks.manager_based.jump.mdp.observations import amp_obs_single
from humanoid_run_jump.tracker.reduced_coords import quat_apply_inverse, quat_to_tan_norm

# Jump phase state machine (latched in JumpAmpEnv._ep_phase).
PHASE_PREP = 0
PHASE_PLANT = 1
PHASE_PUSH = 2
PHASE_RISE = 3
PHASE_EXTEND = 4
PHASE_LAND = 5
PHASE_IDLE = 6


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


class JumpAmpEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv extended with AMP extras for skrl.

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
                # .../tasks/manager_based/jump/jump_amp_env.py → task-humanoid-run-jump/
                project_root = Path(__file__).resolve().parents[6]
                cand = project_root / path
                path = cand if cand.exists() else Path.cwd() / motion_file
            if not path.exists():
                raise FileNotFoundError(
                    f"[JumpAmpEnv] motion_file not found: {path}\n"
                    "Build it with: python scripts/build_amp_dataset.py"
                )
            self._motion_lib = MotionLib(path, device=self.device)

        self._init_jump_metrics()

    def _init_jump_metrics(self) -> None:
        """Buffers for jump success + phase machine + AMP style."""
        n = self.num_envs
        self._ep_had_flight = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._ep_had_clean_takeoff = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._ep_had_good_landing = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._ep_land_ok_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self._ep_air_steps = torch.zeros(n, dtype=torch.long, device=self.device)

        # Per-foot peak sole height (world z − ankle_to_sole); used for clearance gate.
        self._ep_peak_foot_z = torch.zeros(n, 2, device=self.device)
        # Liftoff / landing XY along frozen heading.
        self._ep_liftoff_pos_w = torch.zeros(n, 3, device=self.device)
        self._ep_has_liftoff = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._ep_bad_takeoff = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._ep_flight_distance = torch.zeros(n, device=self.device)
        self._ep_prev_both_air = torch.zeros(n, dtype=torch.bool, device=self.device)
        # Last single-support foot before liftoff: 0=left, 1=right.
        # Default 1 (right) matches the run-in bank hand-off plant.
        self._ep_last_support = torch.ones(n, dtype=torch.long, device=self.device)
        self._ep_takeoff_foot = torch.zeros(n, dtype=torch.long, device=self.device)
        self._ep_takeoff_foot_valid = torch.zeros(n, dtype=torch.bool, device=self.device)

        # Phase machine: PREP → PLANT → PUSH → RISE → EXTEND → LAND → IDLE.
        self._ep_phase = torch.zeros(n, dtype=torch.long, device=self.device)
        self._ep_prep_steps = torch.zeros(n, dtype=torch.long, device=self.device)

        # Apex via airborne window extrema (not a single coincident frame).
        self._ep_liftoff_pelvis_z = torch.zeros(n, device=self.device)
        # Descent-begun flag for RISE→EXTEND only (not used for rise/hurdle success).
        self._ep_apex_latched = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._ep_max_foot_sep_flight = torch.zeros(n, device=self.device)
        # Live per-leg pelvis→ankle distances (obs / shaping).
        self._ep_tuck_l = torch.zeros(n, device=self.device)
        self._ep_tuck_r = torch.zeros(n, device=self.device)
        self._ep_peak_rise = torch.zeros(n, device=self.device)
        # Apex-snapshot / window hurdle geometry (push fold / swing extend / asymmetry).
        # Extrema over the airborne window; `_ep_apex_*` mirrors them for logging.
        self._ep_min_push_d = torch.full((n,), 10.0, device=self.device)
        # Potential already paid by apex_fold (saturates at push_fold*).
        self._ep_fold_progress_paid = torch.zeros(n, device=self.device)
        self._ep_max_swing_d = torch.zeros(n, device=self.device)
        self._ep_apex_push_d = torch.zeros(n, device=self.device)
        self._ep_apex_swing_d = torch.zeros(n, device=self.device)
        self._ep_apex_asym = torch.zeros(n, device=self.device)
        self._ep_apex_arm_span = torch.zeros(n, device=self.device)
        self._ep_rise_ok = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._ep_push_fold_ok = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._ep_swing_ext_ok = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._ep_hurdle_ok = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._ep_pitch_ok = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._ep_pitch_at_apex = torch.zeros(n, device=self.device)

        # Post-landing idle recovery (dense reward; not part of full success yet).
        self._ep_has_landed = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._ep_post_land_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self._ep_idle_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self._ep_cleared_apex = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._ep_cleared_distance = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._ep_stable_ok = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._ep_full_success = torch.zeros(n, dtype=torch.bool, device=self.device)

        # Config (overridable via env cfg attributes).
        self._ankle_to_sole = float(getattr(self.cfg, "ankle_to_sole", 0.05))
        # Post-landing deadline to reach idle (3 s at control dt = 0.02 → 150).
        self._post_land_window_steps = int(
            getattr(
                self.cfg,
                "post_land_window_steps",
                getattr(self.cfg, "stable_steps_required", 150),
            )
        )
        # Brief idle confirmation so one-frame flicker is not success.
        self._idle_hold_steps = int(getattr(self.cfg, "idle_hold_steps", 15))
        self._stand_height = float(getattr(self.cfg, "stand_height", 0.75))
        self._contact_force_threshold = 5.0
        self._prep_timeout_steps = int(getattr(self.cfg, "prep_timeout_steps", 50))
        self._foot_sep_max_flight = float(
            getattr(self.cfg, "foot_sep_max_flight", FOOT_SEP_MAX_FLIGHT)
        )
        self._foot_sep_hard = float(getattr(self.cfg, "foot_sep_hard", FOOT_SEP_HARD))
        self._pitch_apex_min = float(getattr(self.cfg, "pitch_apex_min", PITCH_APEX_MIN))
        self._pitch_apex_max = float(getattr(self.cfg, "pitch_apex_max", PITCH_APEX_MAX))
        self._clearance_hard_extra = float(
            getattr(self.cfg, "clearance_hard_extra", CLEARANCE_HARD_EXTRA)
        )
        self._rise_hard_extra = float(getattr(self.cfg, "rise_hard_extra", RISE_HARD_EXTRA))
        # Soft-land hold (control steps); slightly shorter so first good contacts latch.
        self._land_ok_hold_steps = int(getattr(self.cfg, "land_ok_hold_steps", 5))

        # EMA success rates (logged via extras → LoggingAMP). Keep as Python floats;
        # only update when episodes finish (already gated on done).
        self.flight_success_rate = 0.0
        self.flight_air_rate = 0.0
        self.clean_takeoff_rate = 0.0
        self.left_takeoff_rate = 0.0
        self.landing_success_rate = 0.0
        self.apex_success_rate = 0.0
        self.rise_ok_rate = 0.0
        self.push_fold_ok_rate = 0.0
        self.swing_ext_ok_rate = 0.0
        self.hurdle_ok_rate = 0.0
        self.splits_rate = 0.0
        self.distance_success_rate = 0.0
        self.stable_success_rate = 0.0
        self.apex_pitch_ok_rate = 0.0
        self.mean_peak_sole_over_h = 0.0
        self.mean_apex_pitch = 0.0
        self.mean_apex_asym = 0.0
        self.mean_sep_max = 0.0
        self.mean_arm_span_apex = 0.0
        self.mean_pelvis_rise = 0.0
        # Soft curriculum unlock (components), not the full conjunction.
        self.jump_curriculum_progress = 0.0
        self.amp_style_scale = torch.ones(n, device=self.device)

        # Cache jump-command presence (avoids try/except every step).
        self._jump_cmd_cached: bool | None = None

        try:
            self._left_ankle_id, self._right_ankle_id = resolve_ankle_body_ids(self, "robot")
        except Exception:
            self._left_ankle_id = self._right_ankle_id = -1

        try:
            names = list(self.scene["robot"].data.body_names)
            self._left_hand_id = names.index("left_rubber_hand")
            self._right_hand_id = names.index("right_rubber_hand")
        except Exception:
            self._left_hand_id = self._right_hand_id = -1

        # Cache contact-sensor ankle columns once (sensor body index, not articulation).
        self._contact_ankle_ids: list[int] = []
        self._contact_left_ankle_col: int = -1
        self._contact_right_ankle_col: int = -1
        sensors = getattr(self.scene, "sensors", None)
        if sensors is not None and "contact_forces" in sensors:
            contact = sensors["contact_forces"]
            body_names = list(getattr(contact, "body_names", []) or [])
            self._contact_ankle_ids = [i for i, n in enumerate(body_names) if "ankle_roll" in n]
            for i, n in enumerate(body_names):
                if "left_ankle_roll" in n:
                    self._contact_left_ankle_col = i
                elif "right_ankle_roll" in n:
                    self._contact_right_ankle_col = i

    def step(self, action: torch.Tensor):
        # Jump state is refreshed by the leading ``update_jump_state`` termination
        # term (before rewards / other terminations). Here we only finalize EMA
        # metrics for finished episodes and expose extras for LoggingAMP.
        obs, rewards, terminated, truncated, extras = super().step(action)
        # Shift AMP history and write current frame (num_amp_observations is tiny).
        if self.num_amp_observations > 1:
            self.amp_observation_buffer[:, 1:] = self.amp_observation_buffer[:, :-1].clone()
        self.amp_observation_buffer[:, 0] = amp_obs_single(self)
        extras = extras if extras is not None else {}
        extras["amp_obs"] = self.amp_observation_buffer.view(self.num_envs, -1)

        # Finalize episode success BEFORE clearing latches. ``super().step``
        # already reset done envs; ``_ep_*`` still holds the finished episode latch
        # until ``_update_flight_success_metrics`` clears done_ids.
        done = terminated | truncated
        self._update_flight_success_metrics(done)
        # Pass cached Python floats — no per-step GPU sync.
        extras["flight_success_rate"] = self.flight_success_rate
        extras["flight_air_rate"] = self.flight_air_rate
        extras["clean_takeoff_rate"] = self.clean_takeoff_rate
        extras["left_takeoff_rate"] = self.left_takeoff_rate
        extras["landing_success_rate"] = self.landing_success_rate
        extras["apex_success_rate"] = self.apex_success_rate
        extras["rise_ok_rate"] = self.rise_ok_rate
        extras["push_fold_ok_rate"] = self.push_fold_ok_rate
        extras["swing_ext_ok_rate"] = self.swing_ext_ok_rate
        extras["hurdle_ok_rate"] = self.hurdle_ok_rate
        extras["splits_rate"] = self.splits_rate
        extras["distance_success_rate"] = self.distance_success_rate
        extras["stable_success_rate"] = self.stable_success_rate
        extras["apex_pitch_ok_rate"] = self.apex_pitch_ok_rate
        extras["mean_peak_sole_over_h"] = self.mean_peak_sole_over_h
        extras["mean_apex_pitch"] = self.mean_apex_pitch
        extras["mean_apex_asym"] = self.mean_apex_asym
        extras["mean_sep_max"] = self.mean_sep_max
        extras["mean_arm_span_apex"] = self.mean_arm_span_apex
        extras["mean_pelvis_rise"] = self.mean_pelvis_rise
        extras["jump_curriculum_progress"] = self.jump_curriculum_progress
        extras["amp_style_scale"] = self.amp_style_scale
        # Promote Episode_Termination floats to 0-dim tensors for skrl logging.
        log = extras.get("log")
        if isinstance(log, dict):
            for k, v in list(log.items()):
                if isinstance(v, (int, float)) and (
                    k.startswith("Episode_Termination/") or k.startswith("Episode_Reward/")
                ):
                    log[k] = torch.tensor(float(v), device=self.device)
        return obs, rewards, terminated, truncated, extras

    def _ankle_forces(self, contact_force_threshold: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(both_air, both_contact)`` using cached ankle column indices."""
        thr = self._contact_force_threshold if contact_force_threshold is None else contact_force_threshold
        both_air = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        both_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        sensors = getattr(self.scene, "sensors", None)
        if sensors is None or "contact_forces" not in sensors:
            return both_air, both_contact
        forces = sensors["contact_forces"].data.net_forces_w
        if forces is None or forces.numel() == 0:
            return both_air, both_contact
        if self._contact_ankle_ids:
            foot_f = forces[:, self._contact_ankle_ids, :]
        else:
            foot_f = forces
        mag = torch.norm(foot_f, dim=-1)
        both_air = (mag <= thr).all(dim=-1)
        both_contact = (mag > thr).all(dim=-1)
        return both_air, both_contact

    def _per_foot_contact(
        self, contact_force_threshold: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(left_contact, right_contact)`` masks from cached ankle columns."""
        thr = self._contact_force_threshold if contact_force_threshold is None else contact_force_threshold
        left_c = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        right_c = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        sensors = getattr(self.scene, "sensors", None)
        if sensors is None or "contact_forces" not in sensors:
            return left_c, right_c
        forces = sensors["contact_forces"].data.net_forces_w
        if forces is None or forces.numel() == 0:
            return left_c, right_c
        if self._contact_left_ankle_col >= 0:
            left_c = torch.norm(forces[:, self._contact_left_ankle_col, :], dim=-1) > thr
        if self._contact_right_ankle_col >= 0:
            right_c = torch.norm(forces[:, self._contact_right_ankle_col, :], dim=-1) > thr
        return left_c, right_c

    def _reset_fresh_episode_buffers(self, fresh: torch.Tensor) -> None:
        """Vectorized clear of per-episode latches for freshly reset envs (no host sync)."""
        self._ep_air_steps = torch.where(fresh, torch.zeros_like(self._ep_air_steps), self._ep_air_steps)
        self._ep_land_ok_steps = torch.where(fresh, torch.zeros_like(self._ep_land_ok_steps), self._ep_land_ok_steps)
        self._ep_peak_foot_z = torch.where(
            fresh.unsqueeze(-1), torch.zeros_like(self._ep_peak_foot_z), self._ep_peak_foot_z
        )
        self._ep_has_liftoff = torch.where(fresh, torch.zeros_like(self._ep_has_liftoff), self._ep_has_liftoff)
        self._ep_bad_takeoff = torch.where(fresh, torch.zeros_like(self._ep_bad_takeoff), self._ep_bad_takeoff)
        self._ep_flight_distance = torch.where(
            fresh, torch.zeros_like(self._ep_flight_distance), self._ep_flight_distance
        )
        self._ep_prev_both_air = torch.where(
            fresh, torch.zeros_like(self._ep_prev_both_air), self._ep_prev_both_air
        )
        self._ep_last_support = torch.where(
            fresh, torch.ones_like(self._ep_last_support), self._ep_last_support
        )
        self._ep_takeoff_foot = torch.where(
            fresh, torch.zeros_like(self._ep_takeoff_foot), self._ep_takeoff_foot
        )
        self._ep_takeoff_foot_valid = torch.where(
            fresh, torch.zeros_like(self._ep_takeoff_foot_valid), self._ep_takeoff_foot_valid
        )
        self._ep_phase = torch.where(fresh, torch.zeros_like(self._ep_phase), self._ep_phase)
        self._ep_prep_steps = torch.where(fresh, torch.zeros_like(self._ep_prep_steps), self._ep_prep_steps)
        self._ep_liftoff_pelvis_z = torch.where(
            fresh, torch.zeros_like(self._ep_liftoff_pelvis_z), self._ep_liftoff_pelvis_z
        )
        self._ep_apex_latched = torch.where(
            fresh, torch.zeros_like(self._ep_apex_latched), self._ep_apex_latched
        )
        self._ep_max_foot_sep_flight = torch.where(
            fresh, torch.zeros_like(self._ep_max_foot_sep_flight), self._ep_max_foot_sep_flight
        )
        self._ep_tuck_l = torch.where(fresh, torch.zeros_like(self._ep_tuck_l), self._ep_tuck_l)
        self._ep_tuck_r = torch.where(fresh, torch.zeros_like(self._ep_tuck_r), self._ep_tuck_r)
        self._ep_peak_rise = torch.where(fresh, torch.zeros_like(self._ep_peak_rise), self._ep_peak_rise)
        self._ep_min_push_d = torch.where(
            fresh, torch.full_like(self._ep_min_push_d, 10.0), self._ep_min_push_d
        )
        self._ep_fold_progress_paid = torch.where(
            fresh, torch.zeros_like(self._ep_fold_progress_paid), self._ep_fold_progress_paid
        )
        self._ep_max_swing_d = torch.where(
            fresh, torch.zeros_like(self._ep_max_swing_d), self._ep_max_swing_d
        )
        self._ep_apex_push_d = torch.where(
            fresh, torch.zeros_like(self._ep_apex_push_d), self._ep_apex_push_d
        )
        self._ep_apex_swing_d = torch.where(
            fresh, torch.zeros_like(self._ep_apex_swing_d), self._ep_apex_swing_d
        )
        self._ep_apex_asym = torch.where(fresh, torch.zeros_like(self._ep_apex_asym), self._ep_apex_asym)
        self._ep_apex_arm_span = torch.where(
            fresh, torch.zeros_like(self._ep_apex_arm_span), self._ep_apex_arm_span
        )
        self._ep_rise_ok = torch.where(fresh, torch.zeros_like(self._ep_rise_ok), self._ep_rise_ok)
        self._ep_push_fold_ok = torch.where(
            fresh, torch.zeros_like(self._ep_push_fold_ok), self._ep_push_fold_ok
        )
        self._ep_swing_ext_ok = torch.where(
            fresh, torch.zeros_like(self._ep_swing_ext_ok), self._ep_swing_ext_ok
        )
        self._ep_hurdle_ok = torch.where(fresh, torch.zeros_like(self._ep_hurdle_ok), self._ep_hurdle_ok)
        self._ep_pitch_ok = torch.where(fresh, torch.zeros_like(self._ep_pitch_ok), self._ep_pitch_ok)
        self._ep_pitch_at_apex = torch.where(
            fresh, torch.zeros_like(self._ep_pitch_at_apex), self._ep_pitch_at_apex
        )
        self._ep_has_landed = torch.where(fresh, torch.zeros_like(self._ep_has_landed), self._ep_has_landed)
        self._ep_post_land_steps = torch.where(
            fresh, torch.zeros_like(self._ep_post_land_steps), self._ep_post_land_steps
        )
        self._ep_idle_steps = torch.where(fresh, torch.zeros_like(self._ep_idle_steps), self._ep_idle_steps)
        self._ep_cleared_apex = torch.where(
            fresh, torch.zeros_like(self._ep_cleared_apex), self._ep_cleared_apex
        )
        self._ep_cleared_distance = torch.where(
            fresh, torch.zeros_like(self._ep_cleared_distance), self._ep_cleared_distance
        )
        self._ep_stable_ok = torch.where(fresh, torch.zeros_like(self._ep_stable_ok), self._ep_stable_ok)
        self._ep_full_success = torch.where(
            fresh, torch.zeros_like(self._ep_full_success), self._ep_full_success
        )
        self._ep_had_flight = torch.where(fresh, torch.zeros_like(self._ep_had_flight), self._ep_had_flight)
        self._ep_had_clean_takeoff = torch.where(
            fresh, torch.zeros_like(self._ep_had_clean_takeoff), self._ep_had_clean_takeoff
        )
        self._ep_had_good_landing = torch.where(
            fresh, torch.zeros_like(self._ep_had_good_landing), self._ep_had_good_landing
        )

    def _update_jump_metrics_and_style(
        self,
        min_air_steps: int = 6,
        grace_steps: int = 8,
        clean_min_height: float = 0.88,
        clean_min_vz: float = 0.35,
        max_pitch_rate: float = 1.5,
        max_tilt_rad: float = 0.40,
        land_max_tilt: float = 0.40,
        land_max_forward_pitch: float = 0.60,
        land_min_pitch: float = -0.05,
        land_max_roll: float = 0.25,
        land_min_height: float = 0.68,
        land_absorb_min_height: float = 0.56,
        land_absorb_steps: int = 12,
        land_max_height: float = 0.95,
        land_max_heading_err: float = 0.35,
        land_ok_hold_steps: int = 5,
        land_knee_min: float = 0.75,
        land_crouch_max_height: float = 0.72,
        land_max_horiz_speed: float = 1.15,
        idle_max_tilt: float = 0.50,
        idle_max_forward_pitch: float = 0.40,
        idle_min_pitch: float = -0.08,
        idle_max_roll: float = 0.30,
        idle_height_tol: float = 0.20,
        max_horiz_speed: float = 0.55,
        max_vert_speed: float = 0.50,
        max_yaw_rate: float = 1.4,
        max_joint_vel2: float = 55.0,
        # Full AMP style in every phase — AMP is the sole pose teacher (incl. fold).
        flight_style_scale: float = 1.0,
        # Soften slightly after land; still high enough for recovery style prior.
        land_style_scale: float | None = 1.0,
        push_vz_min: float = 0.25,
    ) -> None:
        """Single-pass phase machine + apex latches + AMP style (no GPU sync).

        Called from the leading ``update_jump_state`` termination term so rewards
        and other terminations see fresh latches.
        """
        if not self._has_jump_command():
            self.amp_style_scale.fill_(1.0)
            return

        if land_style_scale is None:
            land_style_scale = flight_style_scale

        both_air, both_contact = self._ankle_forces()
        left_c, right_c = self._per_foot_contact()
        asset = self.scene["robot"]
        height = asset.data.root_pos_w[:, 2]
        vz = asset.data.root_lin_vel_w[:, 2]
        ep_len = getattr(self, "episode_length_buf", None)

        eligible = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if ep_len is not None:
            # After episode_length_buf += 1, fresh envs have ep_len == 1 once.
            fresh = ep_len <= 1
            self._reset_fresh_episode_buffers(fresh)
            eligible = ep_len >= grace_steps

        # Track last single-support foot before liftoff (0=left, 1=right).
        left_only = left_c & (~right_c)
        right_only = right_c & (~left_c)
        self._ep_last_support = torch.where(
            left_only,
            torch.zeros_like(self._ep_last_support),
            torch.where(right_only, torch.ones_like(self._ep_last_support), self._ep_last_support),
        )

        # Live per-leg tuck / foot separation (pelvis→ankle 3D distance).
        if self._left_ankle_id >= 0:
            left_pos = asset.data.body_pos_w[:, self._left_ankle_id]
            right_pos = asset.data.body_pos_w[:, self._right_ankle_id]
            root = asset.data.root_pos_w
            d_l = torch.linalg.norm(root - left_pos, dim=-1)
            d_r = torch.linalg.norm(root - right_pos, dim=-1)
            foot_sep = torch.linalg.norm(left_pos - right_pos, dim=-1)
        else:
            d_l = torch.zeros(self.num_envs, device=self.device)
            d_r = torch.zeros(self.num_envs, device=self.device)
            foot_sep = torch.zeros(self.num_envs, device=self.device)
        self._ep_tuck_l = d_l
        self._ep_tuck_r = d_r

        # Defaults for `_jump_step` cache (overwritten inside ankle block).
        pitch = torch.zeros(self.num_envs, device=self.device)
        left_z = torch.zeros(self.num_envs, device=self.device)
        right_z = torch.zeros(self.num_envs, device=self.device)
        d_push = d_l
        d_swing = d_r

        # --- phase state machine (plant first so liftoff can arm on same step) ---
        left_fwd = lead_foot_forward_mask(self, lead="left", min_lead_m=0.04)
        phase = self._ep_phase
        self._ep_prep_steps = torch.where(
            phase == PHASE_PREP,
            self._ep_prep_steps + 1,
            torch.zeros_like(self._ep_prep_steps),
        )

        to_plant = (phase == PHASE_PREP) & left_fwd & left_c & (~both_air)
        phase = torch.where(to_plant, torch.full_like(phase, PHASE_PLANT), phase)
        to_push = (
            (phase == PHASE_PLANT)
            & left_c
            & (~right_c)
            & (vz > push_vz_min)
            & (~both_air)
        )
        phase = torch.where(to_push, torch.full_like(phase, PHASE_PUSH), phase)

        # --- clean takeoff (metrics only) ---
        ang_b = quat_apply_inverse(asset.data.root_quat_w, asset.data.root_ang_vel_w)
        pitch_rate = ang_b[:, 1].abs()
        cos_tilt = (-asset.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)
        tilt = torch.acos(cos_tilt)
        clean = (
            eligible
            & both_air
            & self._ep_has_liftoff
            & (height >= clean_min_height)
            & (vz >= clean_min_vz)
            & (pitch_rate < max_pitch_rate)
            & (tilt < max_tilt_rad)
        )
        self._ep_had_clean_takeoff |= clean

        term = self.command_manager.get_term("jump")

        # --- foot peaks + armed left-foot liftoff / window apex / distance ---
        if self._left_ankle_id >= 0:
            left_z = asset.data.body_pos_w[:, self._left_ankle_id, 2] - self._ankle_to_sole
            right_z = asset.data.body_pos_w[:, self._right_ankle_id, 2] - self._ankle_to_sole
            cur = torch.stack([left_z, right_z], dim=-1)
            self._ep_peak_foot_z = torch.where(
                both_air.unsqueeze(-1),
                torch.maximum(self._ep_peak_foot_z, cur),
                self._ep_peak_foot_z,
            )

            rising = both_air & (~self._ep_prev_both_air)
            armed = phase >= PHASE_PLANT
            left_support = self._ep_last_support == 0
            first_lift = rising & (~self._ep_has_liftoff) & armed & left_support
            self._ep_bad_takeoff |= rising & (~self._ep_has_liftoff) & armed & (~left_support)

            self._ep_liftoff_pos_w = torch.where(
                first_lift.unsqueeze(-1), asset.data.root_pos_w, self._ep_liftoff_pos_w
            )
            self._ep_liftoff_pelvis_z = torch.where(first_lift, height, self._ep_liftoff_pelvis_z)
            self._ep_takeoff_foot = torch.where(
                first_lift, self._ep_last_support, self._ep_takeoff_foot
            )
            self._ep_takeoff_foot_valid = self._ep_takeoff_foot_valid | first_lift
            self._ep_has_liftoff |= first_lift

            push_is_left = torch.where(
                self._ep_takeoff_foot_valid,
                self._ep_takeoff_foot == 0,
                self._ep_last_support == 0,
            )
            d_push = torch.where(push_is_left, d_l, d_r)
            d_swing = torch.where(push_is_left, d_r, d_l)

            # Airborne window extrema for pelvis rise.
            in_air_window = self._ep_has_liftoff & both_air & (~self._ep_has_landed)
            rise = (height - self._ep_liftoff_pelvis_z).clamp(min=0.0)
            self._ep_peak_rise = torch.where(
                in_air_window,
                torch.maximum(self._ep_peak_rise, rise),
                self._ep_peak_rise,
            )

            # Forward pitch (positive = lean forward). Snapshot at apex latch.
            g = asset.data.projected_gravity_b
            pitch = torch.atan2(-g[:, 0], -g[:, 2])
            just_apex = self._ep_has_liftoff & (vz < 0.0) & (~self._ep_apex_latched)
            self._ep_pitch_at_apex = torch.where(just_apex, pitch, self._ep_pitch_at_apex)

            # Window extrema: best push fold / swing extension anywhere in flight.
            self._ep_min_push_d = torch.where(
                in_air_window,
                torch.minimum(self._ep_min_push_d, d_push),
                self._ep_min_push_d,
            )
            self._ep_max_swing_d = torch.where(
                in_air_window,
                torch.maximum(self._ep_max_swing_d, d_swing),
                self._ep_max_swing_d,
            )
            # Mirror extrema into apex_* for logging / land_scale / obs consumers.
            measured = in_air_window | self._ep_apex_latched
            self._ep_apex_push_d = torch.where(
                measured, self._ep_min_push_d, self._ep_apex_push_d
            )
            self._ep_apex_swing_d = torch.where(
                measured, self._ep_max_swing_d, self._ep_apex_swing_d
            )
            self._ep_apex_asym = torch.where(
                measured,
                (self._ep_max_swing_d - self._ep_min_push_d).abs(),
                self._ep_apex_asym,
            )
            if self._left_hand_id >= 0:
                span = torch.linalg.norm(
                    asset.data.body_pos_w[:, self._left_hand_id]
                    - asset.data.body_pos_w[:, self._right_hand_id],
                    dim=-1,
                )
                self._ep_apex_arm_span = torch.where(just_apex, span, self._ep_apex_arm_span)
            # Descent flag for RISE→EXTEND only (not used for rise/hurdle measurement).
            self._ep_apex_latched |= self._ep_has_liftoff & (vz < 0.0)

            in_flight = both_air & self._ep_has_liftoff
            self._ep_air_steps = self._ep_air_steps + in_flight.long()
            self._ep_had_flight |= self._ep_has_liftoff & (self._ep_air_steps >= min_air_steps)

            self._ep_max_foot_sep_flight = torch.where(
                in_flight,
                torch.maximum(self._ep_max_foot_sep_flight, foot_sep),
                self._ep_max_foot_sep_flight,
            )
            sep_cap = foot_sep_flight_max(term.flight_distance)
            no_splits = self._ep_max_foot_sep_flight <= sep_cap

            # Window-based success:
            # Rise gate = peak sole clearance ≥ commanded obstacle height (actual
            # task). CSV pelvis-rise thresholds are dynamically inconsistent with
            # takeoff vz under physics (~vz²/2g).
            peak_sole = self._ep_peak_foot_z.max(dim=-1).values
            self._ep_rise_ok = self._ep_has_liftoff & (peak_sole >= term.h_obstacle)

            has_apex = self._ep_apex_latched
            fold_measured = self._ep_min_push_d < 9.0
            self._ep_push_fold_ok = (
                has_apex
                & fold_measured
                & (self._ep_min_push_d <= term.push_fold_star)
                & (self._ep_min_push_d > 0)
            )
            self._ep_swing_ext_ok = has_apex & (self._ep_max_swing_d >= term.swing_ext_star)
            # Metrics-only: fold style is AMP's job (not required for episode success).
            self._ep_hurdle_ok = (
                self._ep_push_fold_ok
                & self._ep_swing_ext_ok
                & (self._ep_apex_asym >= APEX_ASYM_MIN)
            )

            # Pitch band latch during flight (required for cleared apex).
            in_air_pitch = self._ep_has_liftoff & both_air & (~self._ep_has_landed)
            pitch_in_band = (pitch >= self._pitch_apex_min) & (pitch <= self._pitch_apex_max)
            self._ep_pitch_ok = self._ep_pitch_ok | (in_air_pitch & pitch_in_band)
            # Cleared apex = rise + pitch + no splits. Fold/hurdle is AMP style,
            # not a hard success gate (matches the 9-term reward design).
            self._ep_cleared_apex = (
                self._ep_rise_ok & self._ep_pitch_ok & no_splits
            )

            # Flight distance (no credit once splits exceed distance-coupled cap).
            delta = (asset.data.root_pos_w - self._ep_liftoff_pos_w).clone()
            delta[:, 2] = 0.0
            dist = (delta * term._forward_w).sum(dim=-1).clamp(min=0.0)
            falling = (~both_air) & self._ep_prev_both_air & self._ep_has_liftoff
            update_dist = falling | in_flight
            self._ep_flight_distance = torch.where(
                update_dist,
                torch.maximum(self._ep_flight_distance, dist),
                self._ep_flight_distance,
            )
            self._ep_cleared_distance = (
                no_splits
                & self._ep_has_liftoff
                & (self._ep_flight_distance >= term.flight_distance)
            )

        # Continue phase transitions that depend on liftoff / apex.
        to_rise = ((phase == PHASE_PUSH) | (phase == PHASE_PLANT)) & both_air & self._ep_has_liftoff
        phase = torch.where(to_rise, torch.full_like(phase, PHASE_RISE), phase)
        to_extend = (phase == PHASE_RISE) & self._ep_apex_latched
        phase = torch.where(to_extend, torch.full_like(phase, PHASE_EXTEND), phase)

        # --- post-landing: recover to idle within a 3 s window ---
        # Soft land rejects kneel-height / spun landings (heading vs episode start).
        # First absorb frames allow a lower pelvis so crouch is not rejected.
        had_air = self._ep_had_flight | self._ep_has_liftoff
        absorbing = self._ep_has_landed & (
            self._ep_post_land_steps <= max(int(land_absorb_steps), 0)
        )
        min_h = torch.where(
            absorbing,
            torch.full_like(height, float(land_absorb_min_height)),
            torch.full_like(height, float(land_min_height)),
        )
        # Also allow crouch on the first contact frame before post_land_steps increments.
        first_touch = had_air & (~both_air) & both_contact & (~self._ep_has_landed)
        min_h = torch.where(
            first_touch,
            torch.full_like(height, float(land_absorb_min_height)),
            min_h,
        )
        height_ok = (height > min_h) & (height < land_max_height)
        # Pitch-aware soft land: allow athletic forward crouch lean; reject
        # back lean and large lateral roll. Isotropic tilt alone rejected the
        # reference 30–45° absorb plant (tilt > 0.40).
        g = asset.data.projected_gravity_b
        pitch = torch.atan2(-g[:, 0], -g[:, 2])
        roll_abs = g[:, 1].abs()
        in_absorb = absorbing | first_touch
        max_fwd = torch.where(
            in_absorb,
            torch.full_like(pitch, float(land_max_forward_pitch)),
            torch.full_like(pitch, min(float(land_max_forward_pitch), 0.45)),
        )
        pitch_ok = (pitch >= float(land_min_pitch)) & (pitch <= max_fwd)
        roll_ok = roll_abs < float(land_max_roll)
        # Keep a loose isotropic tilt cap so extreme sideways tumbles still fail.
        tilt_ok = tilt < max(float(land_max_tilt), float(land_max_forward_pitch) + 0.15)
        pose_ok = pitch_ok & roll_ok & tilt_ok
        horiz = torch.linalg.norm(asset.data.root_lin_vel_w[:, :2], dim=-1)
        yaw_rate = asset.data.root_ang_vel_w[:, 2].abs()
        heading_ok = term.heading_error.abs() < land_max_heading_err
        soft_land = (
            had_air
            & (~both_air)
            & both_contact
            & pose_ok
            & height_ok
            & heading_ok
            & (horiz < float(land_max_horiz_speed))
            & (yaw_rate < 1.2)
        )
        # Reject locked-straight "rabbit" plant during the absorb window:
        # require knee flex or a lowered pelvis. Outside absorb, crouch not required.
        jmap = getattr(self, "joint_order_map", None)
        if jmap is not None:
            dof = jmap.to_g1(asset.data.joint_pos)
            knee_flex = 0.5 * (dof[:, 3] + dof[:, 9])
        else:
            names = list(asset.data.joint_names)
            try:
                li = names.index("left_knee_joint")
                ri = names.index("right_knee_joint")
                knee_flex = 0.5 * (asset.data.joint_pos[:, li] + asset.data.joint_pos[:, ri])
            except ValueError:
                knee_flex = torch.full_like(height, float(land_knee_min))
        crouch_ok = (knee_flex >= float(land_knee_min)) | (
            height <= float(land_crouch_max_height)
        )
        soft_land = soft_land & ((~in_absorb) | crouch_ok)
        self._ep_land_ok_steps = torch.where(
            soft_land, self._ep_land_ok_steps + 1, torch.zeros_like(self._ep_land_ok_steps)
        )
        self._ep_had_good_landing |= self._ep_land_ok_steps >= int(
            getattr(self, "_land_ok_hold_steps", land_ok_hold_steps)
        )

        first_land = had_air & both_contact & (~both_air) & (~self._ep_has_landed)
        first_land = first_land | (
            self._ep_has_liftoff & (~both_air) & (~self._ep_has_landed) & (left_c | right_c)
        )
        self._ep_has_landed |= first_land
        to_land = (phase >= PHASE_RISE) & self._ep_has_landed
        phase = torch.where(to_land, torch.full_like(phase, PHASE_LAND), phase)
        to_idle = (phase == PHASE_LAND) & both_contact
        phase = torch.where(to_idle, torch.full_like(phase, PHASE_IDLE), phase)
        self._ep_phase = phase

        self._ep_post_land_steps = torch.where(
            self._ep_has_landed,
            self._ep_post_land_steps + 1,
            self._ep_post_land_steps,
        )
        in_window = self._ep_has_landed & (
            self._ep_post_land_steps <= self._post_land_window_steps
        )

        joint_vel2 = torch.sum(torch.square(asset.data.joint_vel), dim=1)
        idle_pitch_ok = (pitch >= float(idle_min_pitch)) & (
            pitch <= float(idle_max_forward_pitch)
        )
        idle_roll_ok = roll_abs < float(idle_max_roll)
        idle = (
            in_window
            & both_contact
            & (~both_air)
            & (tilt < idle_max_tilt)
            & idle_pitch_ok
            & idle_roll_ok
            & ((height - self._stand_height).abs() < idle_height_tol)
            & (horiz < max_horiz_speed)
            & (vz.abs() < max_vert_speed)
            & (yaw_rate < max_yaw_rate)
            & (joint_vel2 < max_joint_vel2)
        )
        self._ep_idle_steps = torch.where(
            idle, self._ep_idle_steps + 1, torch.zeros_like(self._ep_idle_steps)
        )
        # Idle latch only after a soft landing (blocks contact-only cheese).
        self._ep_stable_ok |= (
            self._ep_had_good_landing
            & (self._ep_idle_steps >= self._idle_hold_steps)
        )
        # Full success: rise+pitch apex + distance + soft land + stable idle.
        # Fold/hurdle is AMP style, not required here.
        self._ep_full_success |= (
            self._ep_cleared_apex
            & self._ep_cleared_distance
            & self._ep_has_landed
            & self._ep_had_good_landing
            & self._ep_stable_ok
        )

        self._ep_prev_both_air = both_air

        # Per-step cache for rewards / observations (no GPU sync).
        self._jump_step = {
            "both_air": both_air,
            "both_contact": both_contact,
            "foot_sep": foot_sep,
            "pitch": pitch,
            "peak_sole_cur": torch.maximum(left_z, right_z),
            "d_push": d_push,
            "d_swing": d_swing,
        }

        # Soft curriculum signal: landing + distance only (fold/hurdle left to AMP).
        self.jump_curriculum_progress = (
            0.60 * self.landing_success_rate + 0.40 * self.distance_success_rate
        )

        # --- AMP style scale ---
        # Full style in every phase — AMP is the sole source of pose style.
        suppress = self._ep_phase >= PHASE_PUSH
        self.amp_style_scale = torch.where(
            suppress,
            torch.full((self.num_envs,), flight_style_scale, device=self.device),
            torch.ones(self.num_envs, device=self.device),
        )
        self.amp_style_scale = torch.where(
            self._ep_has_landed,
            torch.full((self.num_envs,), float(land_style_scale), device=self.device),
            self.amp_style_scale,
        )

    def _has_jump_command(self) -> bool:
        if self._jump_cmd_cached is not None:
            return self._jump_cmd_cached
        cm = getattr(self, "command_manager", None)
        if cm is None:
            self._jump_cmd_cached = False
            return False
        try:
            cm.get_term("jump")
            self._jump_cmd_cached = True
        except (KeyError, AttributeError, RuntimeError):
            self._jump_cmd_cached = False
        return self._jump_cmd_cached

    def _update_flight_success_metrics(self, done: torch.Tensor, ema_alpha: float = 0.05) -> None:
        """Update EMA hop-success rates only when episodes finish (sync is rare)."""
        if not self._has_jump_command():
            return
        if not torch.any(done):
            return

        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        air = self._ep_had_flight[done_ids]
        clean = self._ep_had_clean_takeoff[done_ids]
        land = self._ep_had_good_landing[done_ids]
        apex = self._ep_cleared_apex[done_ids]
        dist = self._ep_cleared_distance[done_ids]
        stable = self._ep_stable_ok[done_ids]
        left_to = (
            self._ep_takeoff_foot_valid[done_ids] & (self._ep_takeoff_foot[done_ids] == 0)
        )
        rise_ok = self._ep_rise_ok[done_ids]
        push_fold_ok = self._ep_push_fold_ok[done_ids]
        swing_ext_ok = self._ep_swing_ext_ok[done_ids]
        hurdle_ok = self._ep_hurdle_ok[done_ids]
        pitch_ok = self._ep_pitch_ok[done_ids]
        splits = self._ep_max_foot_sep_flight[done_ids] > self._foot_sep_max_flight
        # Matches ``_ep_full_success``: apex + distance + soft land + stable.
        landed = self._ep_has_landed[done_ids]
        success = (apex & dist & landed & land & stable).float()

        # Peak sole / h and apex pitch for fly / lean diagnostics.
        try:
            h_cmd = self.command_manager.get_term("jump").h_obstacle[done_ids].clamp(min=0.05)
        except (KeyError, AttributeError, RuntimeError):
            h_cmd = torch.ones(done_ids.numel(), device=self.device)
        peak_sole = self._ep_peak_foot_z[done_ids].max(dim=-1).values
        sole_over_h = (peak_sole / h_cmd).clamp(0.0, 5.0)
        apex_pitch = self._ep_pitch_at_apex[done_ids]
        apex_asym = self._ep_apex_asym[done_ids]
        sep_max = self._ep_max_foot_sep_flight[done_ids]
        arm_span = self._ep_apex_arm_span[done_ids]
        pelvis_rise = self._ep_peak_rise[done_ids]

        def _ema(attr: str, rate: float) -> None:
            prev = float(getattr(self, attr, 0.0))
            setattr(self, attr, (1.0 - ema_alpha) * prev + ema_alpha * rate)

        _ema("flight_air_rate", float(air.float().mean().item()))
        _ema("clean_takeoff_rate", float(clean.float().mean().item()))
        _ema("left_takeoff_rate", float(left_to.float().mean().item()))
        _ema("landing_success_rate", float(land.float().mean().item()))
        _ema("apex_success_rate", float(apex.float().mean().item()))
        _ema("rise_ok_rate", float(rise_ok.float().mean().item()))
        _ema("push_fold_ok_rate", float(push_fold_ok.float().mean().item()))
        _ema("swing_ext_ok_rate", float(swing_ext_ok.float().mean().item()))
        _ema("hurdle_ok_rate", float(hurdle_ok.float().mean().item()))
        _ema("splits_rate", float(splits.float().mean().item()))
        _ema("distance_success_rate", float(dist.float().mean().item()))
        _ema("stable_success_rate", float(stable.float().mean().item()))
        _ema("flight_success_rate", float(success.mean().item()))
        _ema("apex_pitch_ok_rate", float(pitch_ok.float().mean().item()))
        _ema("mean_peak_sole_over_h", float(sole_over_h.mean().item()))
        _ema("mean_apex_pitch", float(apex_pitch.mean().item()))
        _ema("mean_apex_asym", float(apex_asym.mean().item()))
        _ema("mean_sep_max", float(sep_max.mean().item()))
        _ema("mean_arm_span_apex", float(arm_span.mean().item()))
        _ema("mean_pelvis_rise", float(pelvis_rise.mean().item()))
        self.jump_curriculum_progress = (
            0.40 * self.landing_success_rate
            + 0.25 * self.distance_success_rate
            + 0.20 * self.rise_ok_rate
            + 0.15 * self.hurdle_ok_rate
        )

        self._ep_had_flight[done_ids] = False
        self._ep_had_clean_takeoff[done_ids] = False
        self._ep_had_good_landing[done_ids] = False
        self._ep_air_steps[done_ids] = 0
        self._ep_land_ok_steps[done_ids] = 0
        self._ep_peak_foot_z[done_ids] = 0.0
        self._ep_has_liftoff[done_ids] = False
        self._ep_bad_takeoff[done_ids] = False
        self._ep_flight_distance[done_ids] = 0.0
        self._ep_prev_both_air[done_ids] = False
        self._ep_last_support[done_ids] = 1
        self._ep_takeoff_foot[done_ids] = 0
        self._ep_takeoff_foot_valid[done_ids] = False
        self._ep_phase[done_ids] = 0
        self._ep_prep_steps[done_ids] = 0
        self._ep_liftoff_pelvis_z[done_ids] = 0.0
        self._ep_apex_latched[done_ids] = False
        self._ep_max_foot_sep_flight[done_ids] = 0.0
        self._ep_tuck_l[done_ids] = 0.0
        self._ep_tuck_r[done_ids] = 0.0
        self._ep_peak_rise[done_ids] = 0.0
        self._ep_min_push_d[done_ids] = 10.0
        self._ep_fold_progress_paid[done_ids] = 0.0
        self._ep_max_swing_d[done_ids] = 0.0
        self._ep_apex_push_d[done_ids] = 0.0
        self._ep_apex_swing_d[done_ids] = 0.0
        self._ep_apex_asym[done_ids] = 0.0
        self._ep_apex_arm_span[done_ids] = 0.0
        self._ep_rise_ok[done_ids] = False
        self._ep_push_fold_ok[done_ids] = False
        self._ep_swing_ext_ok[done_ids] = False
        self._ep_hurdle_ok[done_ids] = False
        self._ep_pitch_ok[done_ids] = False
        self._ep_pitch_at_apex[done_ids] = 0.0
        self._ep_has_landed[done_ids] = False
        self._ep_post_land_steps[done_ids] = 0
        self._ep_idle_steps[done_ids] = 0
        self._ep_cleared_apex[done_ids] = False
        self._ep_cleared_distance[done_ids] = False
        self._ep_stable_ok[done_ids] = False
        self._ep_full_success[done_ids] = False

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        """Sample AMP observation windows from the reference motion dataset.

        History stride matches the env control period (``step_dt``) so reference
        and policy AMP buffers span the same wall-clock interval.
        """
        if self._motion_lib is None:
            raise RuntimeError(
                "[JumpAmpEnv] collect_reference_motions called with no "
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
