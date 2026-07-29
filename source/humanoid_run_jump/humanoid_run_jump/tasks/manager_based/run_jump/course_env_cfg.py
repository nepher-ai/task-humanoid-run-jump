# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Obstacle-course environment scaffold (HL training TBD).

Environment specification
-------------------------
* Straight path, length ~ U(5, 30) m, width 3 m.
* Full-width obstacles every 3–5 m.
* Obstacle thickness ~ U(0.05, 0.20) m, height ~ U(0, 0.75) m (G1 motion
  frame; CSV tags 0.5/0.75 m are literal obstacle heights).
* HL observation includes next-obstacle (height, thickness, orientation angle).

This module registers a runnable env that randomizes course geometry and exposes
next-obstacle features. High-level policy learning is intentionally stubbed —
see ``hl_stub.py``.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
from humanoid_run_jump.tasks.manager_based.common import mdp
from humanoid_run_jump.tasks.manager_based.common.amp_env import AmpManagerBasedRLEnv
from humanoid_run_jump.tasks.manager_based.common.mdp.actions import TrackerActionCfg
from humanoid_run_jump.tasks.manager_based.common.mdp.observations import next_obstacle_features

# Maximum obstacles supported per env (path up to 30 m, spacing >= 3 m).
MAX_OBSTACLES = 10


@configclass
class CourseSceneCfg(InteractiveSceneCfg):
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
    # Placeholder obstacle prims (resized / relocated each reset).
    # A single wide box is used as a prototype; full multi-obstacle tiling is TODO.
    obstacle_0 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_0",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 3.0, 0.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 0.0, 0.15)),
    )


@configclass
class CommandsCfg:
    # Temporary: velocity command until HL switching is trained.
    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        heading_command=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(1.5, 2.5),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.2, 0.2),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    # TODO(hl): replace with HierarchicalSwitchAction that routes through
    # frozen run_policy / jump_policy generators. For now the low-level
    # generator is trained end-to-end on the course (debug only).
    target_frame = TrackerActionCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        proprio = ObsTerm(func=mdp.reduced_coords_proprio)
        velocity_commands = ObsTerm(func=mdp.velocity_commands, params={"command_name": "base_velocity"})
        next_obstacle = ObsTerm(func=next_obstacle_features)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


def randomize_course(env: AmpManagerBasedRLEnv, env_ids: torch.Tensor):
    """Sample path length and next-obstacle features for the selected envs.

    Full multi-obstacle USD tiling is deferred; this populates
    ``env.obstacle_features`` (height, thickness, yaw) used by HL observations.
    """
    n = len(env_ids)
    device = env.device
    path_length = torch.empty(n, device=device).uniform_(5.0, 30.0)
    height = torch.empty(n, device=device).uniform_(0.0, 0.75)
    thickness = torch.empty(n, device=device).uniform_(0.05, 0.20)
    yaw = torch.empty(n, device=device).uniform_(-0.15, 0.15)

    if not hasattr(env, "obstacle_features"):
        env.obstacle_features = torch.zeros(env.num_envs, 3, device=device)
    if not hasattr(env, "path_length"):
        env.path_length = torch.zeros(env.num_envs, device=device)

    env.obstacle_features[env_ids, 0] = height
    env.obstacle_features[env_ids, 1] = thickness
    env.obstacle_features[env_ids, 2] = yaw
    env.path_length[env_ids] = path_length

    if "obstacle_0" in env.scene.rigid_objects:
        gap = torch.empty(n, device=device).uniform_(3.0, 5.0)
        obs = env.scene.rigid_objects["obstacle_0"]
        pose = obs.data.default_root_state[env_ids].clone()
        pose[:, :3] += env.scene.env_origins[env_ids]
        pose[:, 0] += gap
        pose[:, 2] = height * 0.5
        obs.write_root_pose_to_sim(pose[:, :7], env_ids)


@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_and_joints,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.5, 0.5), "yaw": (-0.1, 0.1)},
            "velocity_range": {"x": (0.0, 0.5), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        },
    )
    randomize_course = EventTerm(func=randomize_course, mode="reset")


@configclass
class RewardsCfg:
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"std": 0.5, "command_name": "base_velocity"}
    )
    alive = RewTerm(func=mdp.alive_bonus, weight=0.5)
    flat_orientation = RewTerm(func=mdp.flat_orientation, weight=-1.0)
    # TODO(hl): add forward progress / obstacle clearance / finish-line rewards.


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)
    root_height = DoneTerm(func=mdp.root_height_below, params={"minimum_height": 0.30})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.2})


@configclass
class G1RunJumpCourseEnvCfg(ManagerBasedRLEnvCfg):
    """Scaffolded obstacle-course env for future HL training."""

    scene: CourseSceneCfg = CourseSceneCfg(num_envs=1024, env_spacing=8.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    motion_file: str = "motions/packaged/all_motions.pt"
    num_amp_observations: int = 2

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material


@configclass
class G1RunJumpCourseEnvCfg_PLAY(G1RunJumpCourseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.episode_length_s = 60.0
