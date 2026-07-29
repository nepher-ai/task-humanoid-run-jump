# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Jump environment: run-in hand-off + (h, flight_distance) command + AMP."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import time_out

from humanoid_run_jump.robots.g1 import G1_CFG
from humanoid_run_jump.tasks.manager_based.common import mdp
from humanoid_run_jump.tasks.manager_based.common.mdp.actions import TrackerActionCfg
from humanoid_run_jump.tasks.manager_based.common.mdp.commands import JumpCommandCfg
from humanoid_run_jump.tasks.manager_based.common.mdp.jump_envelope import (
    FLIGHT_DIST_MAX,
    FLIGHT_DIST_MIN,
    H_MAX,
)

# ---------------------------------------------------------------------------
# Command bands (G1 motion frame). Tracker envelope: h ≤ 0.75, flight ≤ 1.20
# (CSV-measured; further height-coupled via flight_cap at resample).
# ---------------------------------------------------------------------------
JUMP_ENVELOPE = JumpCommandCfg.Ranges(
    h_obstacle=(0.25, H_MAX),
    flight_distance=(FLIGHT_DIST_MIN, FLIGHT_DIST_MAX),
)
JUMP_ENVELOPE_EASY = JumpCommandCfg.Ranges(
    h_obstacle=(0.25, 0.50),
    flight_distance=(0.50, 0.90),
)
# Play: 0.5 m height, mid-range flight distance (valid: cap(0.5)=1.20).
JUMP_PLAY_CMD = JumpCommandCfg.Ranges(
    h_obstacle=(0.50, 0.50),
    flight_distance=(1.00, 1.00),
)


@configclass
class JumpSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Only bodies needed for foot contact / illegal-contact terminations.
    # Avoid reporting contacts on every pelvis-subtree link (major PhysX cost).
    # knee_link included so kneel landings hard-fail via illegal_contact.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis/.*(ankle_roll_link|knee_link|torso_link|pelvis|head|shoulder|elbow|wrist|rubber_hand).*",
        history_length=1,
        track_air_time=True,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class CommandsCfg:
    jump = JumpCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        ranges=JumpCommandCfg.Ranges(
            h_obstacle=JUMP_ENVELOPE_EASY.h_obstacle,
            flight_distance=JUMP_ENVELOPE_EASY.flight_distance,
        ),
    )


@configclass
class ActionsCfg:
    target_frame = TrackerActionCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # proprio(64) + jump_commands(22) + foot_contacts(2) + foot_air_time(2)
        # + base_height(1) + base_lin_vel_z(1) + actions(64) = 156
        proprio = ObsTerm(func=mdp.reduced_coords_proprio, noise=Unoise(n_min=-0.01, n_max=0.01))
        jump_commands = ObsTerm(func=mdp.jump_commands, params={"command_name": "jump"})
        foot_contacts = ObsTerm(
            func=mdp.foot_contacts,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
        )
        foot_air_time = ObsTerm(
            func=mdp.foot_air_time,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "max_air_time": 1.0,
            },
        )
        base_height = ObsTerm(func=mdp.base_height)
        base_lin_vel_z = ObsTerm(func=mdp.base_lin_vel_z)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_and_joints,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.2, 0.2)},
            "velocity_range": {"x": (0.8, 2.2), "y": (-0.15, 0.15), "z": (-0.05, 0.05)},
        },
    )
    reset_runin = EventTerm(
        func=mdp.reset_from_runin_bank,
        mode="reset",
        params={
            "bank_path": "motions/packaged/runin_states.pt",
            "bank_prob": 0.95,
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.05, 0.05)},
        },
    )


@configclass
class RewardsCfg:
    """Phase-gated jump MDP: prep → plant/push → tucked apex → extend → land."""

    # --- Phase-structured dense shaping ---
    prep_step = RewTerm(
        func=mdp.prep_step,
        weight=4.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "min_lead_m": 0.04,
            "air_penalty": 1.0,
        },
    )
    plant_push = RewTerm(
        func=mdp.plant_push,
        weight=6.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "min_lead_m": 0.04,
            "push_vz_min": 0.25,
        },
    )
    # Sparse +1 left takeoff (right-foot takeoff is illegal_takeoff).
    takeoff_foot = RewTerm(
        func=mdp.takeoff_foot,
        weight=10.0,
    )
    takeoff_launch = RewTerm(
        func=mdp.takeoff_launch,
        weight=5.0,
        params={
            "command_name": "jump",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "vel_std": 1.0,
            "vz_overshoot": 0.15,
            "pitch_std": 0.12,
            "clean_omega0": 2.2,
        },
    )
    apex_tuck = RewTerm(
        func=mdp.apex_tuck,
        weight=28.0,
        params={"command_name": "jump", "tuck_std": 0.15, "pitch_std": 0.12},
    )
    apex_pitch = RewTerm(
        func=mdp.apex_pitch,
        weight=14.0,
        params={"command_name": "jump", "pitch_std": 0.12},
    )
    foot_clearance = RewTerm(
        func=mdp.foot_clearance,
        weight=10.0,
        params={"command_name": "jump"},
    )
    excess_height = RewTerm(
        func=mdp.excess_height_penalty,
        weight=-25.0,
        params={"command_name": "jump", "margin": 0.05},
    )
    leg_extend = RewTerm(
        func=mdp.leg_extend,
        weight=10.0,
        params={"tuck_std": 0.10},
    )
    extend_pitch = RewTerm(
        func=mdp.extend_pitch,
        weight=6.0,
        params={"pitch_std": 0.15},
    )
    foot_split_penalty = RewTerm(
        func=mdp.foot_split_penalty,
        weight=-50.0,
    )
    flight_distance = RewTerm(
        func=mdp.flight_distance_progress,
        weight=8.0,
        params={
            "command_name": "jump",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    # Keep yaw near episode-start heading (down-weighted vs jump-quality terms).
    heading_keep = RewTerm(
        func=mdp.heading_keep,
        weight=2.0,
        params={"command_name": "jump", "std": 0.35},
    )
    land_and_idle = RewTerm(
        func=mdp.land_and_idle,
        weight=12.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "stand_height": 0.75,
            "max_tilt_rad": 0.40,
            "min_height": 0.68,
            "max_horiz_speed": 0.60,
            "max_joint_vel2": 40.0,
            "max_heading_err": 0.35,
        },
    )
    rejump_penalty = RewTerm(
        func=mdp.rejump_penalty,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )

    # --- Sparse success criteria ---
    success_rise = RewTerm(
        func=mdp.success_rise,
        weight=10.0,
        params={"command_name": "jump"},
    )
    success_tuck = RewTerm(
        func=mdp.success_tuck,
        weight=12.0,
        params={"command_name": "jump"},
    )
    success_apex = RewTerm(
        func=mdp.success_apex,
        weight=20.0,
        params={"command_name": "jump"},
    )
    success_distance = RewTerm(
        func=mdp.success_flight_distance,
        weight=12.0,
        params={"command_name": "jump"},
    )
    # Idle bonus (dense land_and_idle shapes; not yet part of truncation gate).
    success_stable = RewTerm(
        func=mdp.success_stable_landing,
        weight=8.0,
        params={"command_name": "jump"},
    )

    # --- Regularizers ---
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    no_flight_timeout = RewTerm(func=mdp.no_flight_timeout_penalty, weight=-15.0)


@configclass
class TerminationsCfg:
    # Must be first: refreshes phase / latches before rewards and other terms.
    update_jump_state = DoneTerm(func=mdp.update_jump_state)
    time_out = DoneTerm(func=time_out, time_out=True)
    # Truncate on apex + distance + soft landing (feet, upright, heading) — not a failure.
    success = DoneTerm(func=mdp.jump_success, time_out=True)
    # Cap settle frames after landing (~2 s) so rollouts stay jump-dense.
    post_land_timeout = DoneTerm(
        func=mdp.post_land_timeout,
        time_out=True,
        params={"max_post_land_steps": 100},
    )
    illegal_takeoff = DoneTerm(func=mdp.illegal_takeoff)
    prep_timeout = DoneTerm(func=mdp.prep_timeout, params={"max_prep_steps": 50})
    root_height = DoneTerm(func=mdp.root_height_below, params={"minimum_height": 0.30})
    # Relaxed vs prior 1.05 so takeoff pitch/tuck is not instantly fatal.
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.25})
    # Hard-fail ballistic splits far beyond CSV (soft penalty handles mild overshoot).
    extreme_splits = DoneTerm(func=mdp.extreme_foot_split)
    # Hard-fail balloon flights (peak sole > h + 0.25 m).
    balloon_height = DoneTerm(func=mdp.balloon_height, params={"command_name": "jump"})
    # Hard-fail large yaw drift after liftoff (~70° from episode-start heading).
    heading_blowout = DoneTerm(
        func=mdp.heading_blowout,
        params={"command_name": "jump", "max_heading_err": 1.2},
    )
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "threshold": 50.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "torso_link",
                    "pelvis",
                    "head",
                    ".*_knee_link",
                    ".*_shoulder_.*",
                    ".*_elbow_.*",
                    ".*_wrist_.*",
                    ".*rubber_hand",
                ],
            ),
        },
    )


@configclass
class CurriculumCfg:
    """Success-gated expansion of (h_obstacle, flight_distance)."""

    jump_commands = CurrTerm(
        func=mdp.jump_cmd_levels,
        params={
            "command_name": "jump",
            "steps_to_final": 250000,
            "unlock_progress": 0.20,
            "relock_progress": 0.10,
            "initial_h_obstacle": JUMP_ENVELOPE_EASY.h_obstacle,
            "final_h_obstacle": JUMP_ENVELOPE.h_obstacle,
            "initial_flight_distance": JUMP_ENVELOPE_EASY.flight_distance,
            "final_flight_distance": JUMP_ENVELOPE.flight_distance,
            "envelope_h_obstacle": JUMP_ENVELOPE.h_obstacle,
            "envelope_flight_distance": JUMP_ENVELOPE.flight_distance,
        },
    )


@configclass
class G1JumpEnvCfg(ManagerBasedRLEnvCfg):
    """Jump training env seeded from frozen-run-actor hand-off states."""

    scene: JumpSceneCfg = JumpSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    motion_file: str = "motions/packaged/jump_motions.pt"
    num_amp_observations: int = 2
    # Jump success metric knobs consumed by AmpManagerBasedRLEnv.
    ankle_to_sole: float = 0.05
    stand_height: float = 0.75
    # 3 s post-landing window for idle dense/sparse (post_land_timeout truncates earlier).
    post_land_window_steps: int = 150
    # Alias kept for older configs / scripts.
    stable_steps_required: int = 150
    # Confirm idle for 0.3 s so flicker does not count as success.
    idle_hold_steps: int = 15
    # Soft-land hold (~0.1 s) so first good contacts can latch.
    land_ok_hold_steps: int = 5
    # Apex forward-pitch success band (rad).
    pitch_apex_min: float = 0.08
    pitch_apex_max: float = 0.40
    # Balloon hard-fail: peak sole > h + this (m).
    clearance_hard_extra: float = 0.25
    # PREP dead-end timeout (~1 s).
    prep_timeout_steps: int = 50
    # Soft truncate after landing (~2 s at control dt = 0.02).
    post_land_timeout_steps: int = 100

    def __post_init__(self):
        self.decimation = 4
        # Jump + 3 s settle after landing (run-in is inside reset / bank).
        # Success truncates early via TerminationsCfg.success; 7 s is the failure timeout.
        self.episode_length_s = 7.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.robot.spawn.activate_contact_sensors = True
        # Lighter solver for train throughput (run env keeps default 8/4).
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 4
        self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 1
        if self.scene.contact_forces is not None:
            # Match control rate, not physics substeps (4× fewer sensor updates).
            self.scene.contact_forces.update_period = self.sim.dt * self.decimation


@configclass
class G1JumpEnvCfg_PLAY(G1JumpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        # Match training episode length (success truncates early).
        self.episode_length_s = 7.0
        self.commands.jump.ranges.h_obstacle = JUMP_PLAY_CMD.h_obstacle
        self.commands.jump.ranges.flight_distance = JUMP_PLAY_CMD.flight_distance
        self.commands.jump.resampling_time_range = (20.0, 20.0)
        self.curriculum.jump_commands = None
        self.commands.jump.debug_vis = True
        # Still prefer run-in bank in play when available.
        self.events.reset_runin.params["bank_prob"] = 1.0
        # Zero spawn jitter so each robot sits at its env origin for inspection.
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_runin.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
