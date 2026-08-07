# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the RunJumpHL high-level course environment."""

from humanoid_run_jump.tasks.manager_based.jump.mdp.events import (
    reset_from_runin_bank,
    reset_root_and_joints,
)
from humanoid_run_jump.tasks.manager_based.jump.mdp.observations import reduced_coords_proprio
from humanoid_run_jump.tasks.manager_based.jump.mdp.terminations import update_jump_state

from .hl_actions import HierarchicalSwitchAction, HierarchicalSwitchActionCfg
from .hl_commands import HlVelocityCommand, HlVelocityCommandCfg
from .hl_curriculums import hl_course_levels
from .hl_events import randomize_hl_course
from .hl_observations import (
    body_state,
    course_features,
    hl_feet,
    last_hl_action,
    ll_status,
)
from .hl_rewards import (
    apex_over_obstacle,
    clear_and_stable,
    course_progress,
    fail_terminal,
    finish_bonus,
    heading_align,
    hl_action_rate_l2,
    jump_cmd_match,
    jump_cmd_shaping,
    land_cmd_shock,
    latch_quality,
    lateral_offset,
    lattice_cmd_speed,
    lattice_phase_gain,
    lattice_speed,
    no_jump_penalty,
    obstacle_cleared,
    obstacle_hit,
    plant_in_band,
    plant_on_lattice,
    stall_cost,
)
from .hl_terminations import (
    bad_orientation_hl,
    behind_start,
    course_finished,
    no_progress,
    obstacle_crash,
    out_of_path,
    root_height_below,
    update_course_state,
)

__all__ = [
    "HierarchicalSwitchAction",
    "HierarchicalSwitchActionCfg",
    "HlVelocityCommand",
    "HlVelocityCommandCfg",
    "hl_course_levels",
    "randomize_hl_course",
    "reset_from_runin_bank",
    "reset_root_and_joints",
    "reduced_coords_proprio",
    "body_state",
    "course_features",
    "hl_feet",
    "last_hl_action",
    "ll_status",
    "apex_over_obstacle",
    "clear_and_stable",
    "course_progress",
    "fail_terminal",
    "finish_bonus",
    "heading_align",
    "hl_action_rate_l2",
    "jump_cmd_match",
    "jump_cmd_shaping",
    "land_cmd_shock",
    "latch_quality",
    "lateral_offset",
    "lattice_cmd_speed",
    "lattice_phase_gain",
    "lattice_speed",
    "no_jump_penalty",
    "obstacle_cleared",
    "obstacle_hit",
    "plant_in_band",
    "plant_on_lattice",
    "stall_cost",
    "bad_orientation_hl",
    "behind_start",
    "course_finished",
    "no_progress",
    "obstacle_crash",
    "out_of_path",
    "root_height_below",
    "update_course_state",
    "update_jump_state",
]
