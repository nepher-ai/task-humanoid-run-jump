# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Obstacle-free velocity-commanded run environment for G1."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import (
    UniformVelocityCommandCfg,
    time_out,
)

from humanoid_run_jump.robots.g1 import G1_CFG
from humanoid_run_jump.tasks.manager_based.run import mdp
from humanoid_run_jump.tasks.manager_based.run.mdp.actions import TrackerActionCfg

##
# Scene
##


@configclass
class RunSceneCfg(InteractiveSceneCfg):
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
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class CommandsCfg:
    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.05,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            # Start easy; lin_vel_cmd_levels curriculum expands max to 3.0
            lin_vel_x=(0.0, 1.0),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    target_frame = TrackerActionCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        proprio = ObsTerm(func=mdp.reduced_coords_proprio, noise=Unoise(n_min=-0.01, n_max=0.01))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        velocity_commands = ObsTerm(func=mdp.velocity_commands, params={"command_name": "base_velocity"})
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
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-0.4, 0.4)},
            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.2, 0.2), "z": (-0.1, 0.1)},
        },
    )


@configclass
class RewardsCfg:
    # Wider std early so partial progress is rewarded (standing floor is lower at 1 m/s cmds)
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=2.0, params={"std": 0.8, "command_name": "base_velocity"}
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.75, params={"std": 0.5, "command_name": "base_velocity"}
    )
    # Keep alive modest so standing alone is not the dominant optimum
    alive = RewTerm(func=mdp.alive_bonus, weight=0.2)
    flat_orientation = RewTerm(func=mdp.flat_orientation, weight=-1.0)
    # Soften vz penalty — running CoM bounce is natural, not a failure mode
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.05)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)
    root_height = DoneTerm(func=mdp.root_height_below, params={"minimum_height": 0.35})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.2})


@configclass
class CurriculumCfg:
    lin_vel_cmd = CurrTerm(
        func=mdp.lin_vel_cmd_levels,
        params={
            "command_name": "base_velocity",
            "initial_max": 1.0,
            "final_max": 3.0,
            "steps_to_final": 80000,
        },
    )


@configclass
class G1RunEnvCfg(ManagerBasedRLEnvCfg):
    """Velocity-commanded run training env (no obstacles)."""

    scene: RunSceneCfg = RunSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # AMP extras consumed by RunAmpEnv
    motion_file: str = "motions/packaged/run_motions.pt"
    num_amp_observations: int = 2

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


@configclass
class G1RunEnvCfg_PLAY(G1RunEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.episode_length_s = 40.0
        # Full command range for evaluation (no curriculum ramp)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 3.0)
        self.curriculum.lin_vel_cmd = None
