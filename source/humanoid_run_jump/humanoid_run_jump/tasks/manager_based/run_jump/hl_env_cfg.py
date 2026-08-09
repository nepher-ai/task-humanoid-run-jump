# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment config for Nepher-G1-RunJumpHL (PPO over three frozen models)."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import time_out

from humanoid_run_jump.robots.g1 import G1_CFG
from humanoid_run_jump.tasks.manager_based.jump.mdp.commands import JumpCommandCfg
from humanoid_run_jump.tasks.manager_based.jump.mdp.jump_envelope import (
    FLIGHT_DIST_MAX,
    FLIGHT_DIST_MIN,
    H_MAX,
    H_MIN,
)
from humanoid_run_jump.tasks.manager_based.run_jump import mdp
from humanoid_run_jump.tasks.manager_based.run_jump.hl_terrain import HlTerrainImporterCfg
from humanoid_run_jump.tasks.manager_based.run_jump.mdp.hl_actions import HierarchicalSwitchActionCfg
from humanoid_run_jump.tasks.manager_based.run_jump.mdp.hl_commands import HlVelocityCommandCfg

# Fixed cuboid size: (thickness, lateral, height). Height set by burial.
_OBS_SIZE = (0.20, 5.00, 1.50)
_CUBOID_HALF_H = 0.75


def _make_obstacle_cfg(idx: int) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{idx}",
        spawn=sim_utils.CuboidCfg(
            size=_OBS_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0)),
    )


@configclass
class HlSceneCfg(InteractiveSceneCfg):
    # Finite ground (not Isaac Lab's default 2e6 m plane) — avoids depth-buffer
    # shimmer / white-plane flicker in play.
    terrain = HlTerrainImporterCfg(
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
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis/.*(ankle_roll_link|knee_link|torso_link|pelvis|head|shoulder|elbow|wrist|rubber_hand).*",
        history_length=1,
        track_air_time=True,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    # Ten parked cuboid slots (EnvHub may activate up to 10). Train still caps
    # active hurdles via course_max_obstacles / event max_obstacles (= 3).
    obstacle_0 = _make_obstacle_cfg(0)
    obstacle_1 = _make_obstacle_cfg(1)
    obstacle_2 = _make_obstacle_cfg(2)
    obstacle_3 = _make_obstacle_cfg(3)
    obstacle_4 = _make_obstacle_cfg(4)
    obstacle_5 = _make_obstacle_cfg(5)
    obstacle_6 = _make_obstacle_cfg(6)
    obstacle_7 = _make_obstacle_cfg(7)
    obstacle_8 = _make_obstacle_cfg(8)
    obstacle_9 = _make_obstacle_cfg(9)


@configclass
class CommandsCfg:
    base_velocity = HlVelocityCommandCfg(default_vx=1.8)
    jump = JumpCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e6, 1.0e6),  # HL sets via set_command
        ranges=JumpCommandCfg.Ranges(
            h_obstacle=(H_MIN, H_MAX),
            flight_distance=(FLIGHT_DIST_MIN, FLIGHT_DIST_MAX),
        ),
    )


@configclass
class ActionsCfg:
    hl_switch = HierarchicalSwitchActionCfg(
        asset_name="robot",
        tracker_path=None,
        run_policy_path=None,
        jump_policy_path="best_policy/jump/policy.pt",
        action_repeat=5,
        handoff_lead="right",
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # proprio(64) + body(9) + feet(6) + course(19) + ll_status(12) + last_hl(6) ≈ 116
        proprio = ObsTerm(func=mdp.reduced_coords_proprio, noise=Unoise(n_min=-0.01, n_max=0.01))
        body = ObsTerm(func=mdp.body_state, noise=Unoise(n_min=-0.05, n_max=0.05))
        feet = ObsTerm(
            func=mdp.hl_feet,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
        )
        course = ObsTerm(func=mdp.course_features)
        ll_status = ObsTerm(func=mdp.ll_status)
        last_hl = ObsTerm(func=mdp.last_hl_action)

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
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.05, 0.05)},
            "velocity_range": {"x": (0.8, 2.0), "y": (-0.1, 0.1), "z": (-0.05, 0.05)},
        },
    )
    reset_runin = EventTerm(
        func=mdp.reset_from_runin_bank,
        mode="reset",
        params={
            "bank_path": "motions/packaged/runin_states.pt",
            "bank_prob": 0.95,
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.05, 0.05)},
            "min_fwd_vel": 1.0,
            "max_fwd_vel": 2.5,
        },
    )
    randomize_course = EventTerm(
        func=mdp.randomize_hl_course,
        mode="reset",
        params={
            "path_length_range": (20.0, 30.0),
            "first_obstacle_range": (7.0, 10.0),
            "gap_range": (8.0, 10.0),
            "height_range": (0.20, 0.75),
            "thickness": 0.20,
            "max_obstacles": 3,
            "min_tail": 1.0,
            "cuboid_half_height": _CUBOID_HALF_H,
        },
    )


@configclass
class RewardsCfg:
    """Weights are per-second; Isaac Lab multiplies each term by ``step_dt`` (0.02).

    The effective magnitudes below are chosen so that every jump decision moves
    the episode return by ≳0.2 of its spread. The previous scaling left all of
    them under 0.03, i.e. beneath the entropy bonus on the corresponding action
    channel, which is why only the dense running term was ever learned.
    """

    # Dense backbone: 2.0 per metre, ≈0.10/step at 2.5 m/s.
    course_progress = RewTerm(func=mdp.course_progress, weight=100.0)
    # Only bites while stalled; keeps the terminal penalty free of the
    # "wait to discount the penalty" exploit (needs stall > P*(1-gamma)).
    stall_cost = RewTerm(func=mdp.stall_cost, weight=-8.0, params={"vx_min": 0.6})
    lateral_offset = RewTerm(func=mdp.lateral_offset, weight=-3.0)
    heading_align = RewTerm(func=mdp.heading_align, weight=0.5)

    # Approach shaping. The command-space speed term carries most of the weight:
    # it grades the action taken this step, whereas the measured-speed version
    # only pays out 1.5-2 s later once the frozen run policy has accelerated,
    # which is past the effective credit horizon at gamma=0.99 / 50 Hz.
    lattice_cmd_speed = RewTerm(func=mdp.lattice_cmd_speed, weight=5.0)
    lattice_speed = RewTerm(func=mdp.lattice_speed, weight=1.0)
    # Potential difference in metres of stride misalignment, so the telescoped
    # total over an approach is bounded by half a stride and cannot be farmed.
    lattice_phase_gain = RewTerm(func=mdp.lattice_phase_gain, weight=120.0)
    plant_on_lattice = RewTerm(func=mdp.plant_on_lattice, weight=50.0)

    # Jump decision: absolute plant band, command match, and consistency.
    plant_in_band = RewTerm(func=mdp.plant_in_band, weight=200.0)
    jump_cmd_match = RewTerm(func=mdp.jump_cmd_match, weight=150.0)
    jump_cmd_shaping = RewTerm(func=mdp.jump_cmd_shaping, weight=4.0)
    latch_quality = RewTerm(func=mdp.latch_quality, weight=60.0)
    apex_over_obstacle = RewTerm(func=mdp.apex_over_obstacle, weight=250.0)
    no_jump_penalty = RewTerm(func=mdp.no_jump_penalty, weight=-200.0)

    # Command smoothness into the frozen low-level policies.
    hl_action_rate = RewTerm(func=mdp.hl_action_rate_l2, weight=-1.5)
    land_cmd_shock = RewTerm(func=mdp.land_cmd_shock, weight=-3.0)

    # Outcomes.
    obstacle_cleared = RewTerm(func=mdp.obstacle_cleared, weight=300.0)
    clear_and_stable = RewTerm(func=mdp.clear_and_stable, weight=200.0)
    obstacle_hit = RewTerm(func=mdp.obstacle_hit, weight=-300.0)
    finish = RewTerm(func=mdp.finish_bonus, weight=1000.0)
    fail_terminal = RewTerm(func=mdp.fail_terminal, weight=-750.0)


@configclass
class TerminationsCfg:
    update_jump_state = DoneTerm(
        func=mdp.update_jump_state,
        params={
            "flight_style_scale": 1.0,
            "land_style_scale": 1.0,
            "land_absorb_min_height": 0.56,
            "land_absorb_steps": 12,
            "land_max_forward_pitch": 0.60,
            "land_min_pitch": -0.05,
            "land_max_roll": 0.25,
            "land_knee_min": 0.75,
            "land_crouch_max_height": 0.72,
            "land_max_horiz_speed": 1.15,
        },
    )
    update_course_state = DoneTerm(func=mdp.update_course_state)
    time_out = DoneTerm(func=time_out, time_out=True)
    course_finished = DoneTerm(func=mdp.course_finished, time_out=True)
    out_of_path = DoneTerm(func=mdp.out_of_path, params={"half_width": 2.5})
    obstacle_crash = DoneTerm(func=mdp.obstacle_crash, params={"hold_steps": 12})
    behind_start = DoneTerm(func=mdp.behind_start, params={"margin": 1.0})
    no_progress = DoneTerm(func=mdp.no_progress, params={"window_s": 4.0, "min_delta": 0.30})
    root_height = DoneTerm(func=mdp.root_height_below, params={"minimum_height": 0.30})
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation_hl,
        params={"limit_angle": 1.25, "land_grace_steps": 25},
    )


@configclass
class CurriculumCfg:
    hl_course = CurrTerm(func=mdp.hl_course_levels)


@configclass
class G1RunJumpHLEnvCfg(ManagerBasedRLEnvCfg):
    """Train: PPO HL over frozen tracker + run + jump on a straight obstacle course."""

    scene: HlSceneCfg = HlSceneCfg(num_envs=2048, env_spacing=32.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # JumpAmpEnv knobs (AMP off).
    motion_file: str | None = None
    num_amp_observations: int = 1
    ankle_to_sole: float = 0.05
    stand_height: float = 0.75
    post_land_window_steps: int = 150
    idle_hold_steps: int = 8
    land_ok_hold_steps: int = 3
    pitch_apex_min: float = 0.05
    pitch_apex_max: float = 0.35
    rise_hard_extra: float = 0.20
    prep_timeout_steps: int = 50
    post_land_timeout_steps: int = 150

    # Course knobs.
    course_max_obstacles: int = 3
    out_of_path_half_width: float = 2.5
    # X spacing is scene.env_spacing (32 m). Y lives here because InteractiveScene
    # treats unknown scene-cfg fields as assets.
    env_spacing_y: float = 8.0
    # Draw right-foot plant band / footfalls. Off for train.
    show_plant_target: bool = False
    # Draw course start/end lines (kept on for --video even when plant markers are off).
    show_course_lines: bool = False

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 25.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 4
        self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 1
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt * self.decimation


@configclass
class G1RunJumpHLEnvCfg_PLAY(G1RunJumpHLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.episode_length_s = 40.0
        self.curriculum.hl_course = None
        self.show_plant_target = True
        self.show_course_lines = True
        # Full course difficulty in play.
        self.events.randomize_course.params["height_range"] = (0.20, 0.75)
        self.events.randomize_course.params["max_obstacles"] = 3
        self.events.reset_runin.params["bank_prob"] = 1.0
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_runin.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
