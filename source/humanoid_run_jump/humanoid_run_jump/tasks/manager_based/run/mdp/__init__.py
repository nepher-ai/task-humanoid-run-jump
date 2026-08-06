# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run MDP terms for G1 run environment."""

from .actions import TrackerAction, TrackerActionCfg
from .curriculums import lin_vel_cmd_levels
from .events import reset_root_and_joints
from .gait import (
    foot_contact_mask,
    heading_forward,
    lead_foot_forward_mask,
    resolve_ankle_body_ids,
    runin_handoff_mask,
)
from .observations import (
    amp_obs_single,
    base_lin_vel,
    last_action,
    reduced_coords_proprio,
    velocity_commands,
)
from .rewards import (
    action_rate_l2,
    alive_bonus,
    flat_orientation,
    lin_vel_z_l2,
    track_ang_vel_z_exp,
    track_lin_vel_xy_exp,
)
from .terminations import bad_orientation, root_height_below

__all__ = [
    "TrackerAction",
    "TrackerActionCfg",
    "lin_vel_cmd_levels",
    "reset_root_and_joints",
    "foot_contact_mask",
    "heading_forward",
    "lead_foot_forward_mask",
    "resolve_ankle_body_ids",
    "runin_handoff_mask",
    "amp_obs_single",
    "base_lin_vel",
    "last_action",
    "reduced_coords_proprio",
    "velocity_commands",
    "action_rate_l2",
    "alive_bonus",
    "flat_orientation",
    "lin_vel_z_l2",
    "track_ang_vel_z_exp",
    "track_lin_vel_xy_exp",
    "bad_orientation",
    "root_height_below",
]
