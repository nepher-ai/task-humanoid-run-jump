# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for run environment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _progress(env: ManagerBasedRLEnv, steps_to_final: int) -> float:
    return min(1.0, float(env.common_step_counter) / float(max(steps_to_final, 1)))


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    initial_max: float = 1.0,
    final_max: float = 3.0,
    steps_to_final: int = 80000,
) -> float:
    """Linearly expand ``lin_vel_x`` upper bound over training steps."""
    del env_ids
    progress = _progress(env, steps_to_final)
    max_vx = _lerp(initial_max, final_max, progress)
    term = env.command_manager.get_term(command_name)
    lo = term.cfg.ranges.lin_vel_x[0]
    term.cfg.ranges.lin_vel_x = (lo, max_vx)
    return max_vx
