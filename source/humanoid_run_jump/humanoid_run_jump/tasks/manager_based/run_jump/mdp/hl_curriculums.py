# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum: expand obstacle count and height band with clear rate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# (max_obstacles, h_lo, h_hi, plant_edge_sigma) per stage.
#
# The first stage starts at 0.35 m on purpose. A 0.20-0.35 m box can be stepped
# over at running speed, so an easier opening stage is passable without ever
# gating a jump — and that run-only solution is a strong enough local optimum
# that the policy never leaves it. Low boxes only reappear at the last stage,
# where choosing between stepping and jumping is the actual skill.
#
# ``plant_edge_sigma`` widens the falloff outside the plant band early on, so a
# half-metre miss still earns partial credit and there is a gradient to follow,
# then tightens to the nominal value once the skill exists.
_STAGES: list[tuple[int, float, float, float]] = [
    (1, 0.35, 0.55, 0.50),
    (1, 0.40, 0.65, 0.35),
    (2, 0.30, 0.65, 0.25),
    (3, 0.20, 0.75, 0.20),
]


def hl_course_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    unlock_rate: float = 0.50,
    relock_rate: float = 0.25,
    window: int = 2000,
) -> dict:
    """Advance course difficulty on the *finish* rate, not on partial clears."""
    if not hasattr(env, "_hl_clear_hist"):
        env._hl_clear_hist = []
    if not hasattr(env, "_hl_curriculum_stage"):
        env._hl_curriculum_stage = 0

    # Track completions from the episodes that just finished.
    if len(env_ids) > 0 and hasattr(env, "course_finished"):
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        env._hl_clear_hist.extend(env.course_finished[ids].float().tolist())
        if len(env._hl_clear_hist) > window:
            env._hl_clear_hist = env._hl_clear_hist[-window:]

    rate = sum(env._hl_clear_hist) / len(env._hl_clear_hist) if env._hl_clear_hist else 0.0
    stage = env._hl_curriculum_stage
    if len(env._hl_clear_hist) >= window:
        if stage < len(_STAGES) - 1 and rate >= unlock_rate:
            stage += 1
            env._hl_clear_hist = []
        elif stage > 0 and rate < relock_rate:
            stage -= 1
            env._hl_clear_hist = []
    env._hl_curriculum_stage = stage

    max_obs, h_lo, h_hi, plant_sigma = _STAGES[stage]
    env._curr_max_obstacles = max_obs
    env._curr_h_lo = h_lo
    env._curr_h_hi = h_hi
    env.plant_edge_sigma = plant_sigma

    return {
        "stage": float(stage),
        "finish_rate": float(rate),
        "max_obstacles": float(max_obs),
        "h_hi": float(h_hi),
        "plant_sigma": float(plant_sigma),
    }
