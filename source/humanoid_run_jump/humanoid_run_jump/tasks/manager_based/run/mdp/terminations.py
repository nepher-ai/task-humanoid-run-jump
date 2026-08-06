# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for run environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_height_below(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


def bad_orientation(
    env: ManagerBasedRLEnv,
    limit_angle: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base tilts more than ``limit_angle`` from upright (rad)."""
    asset = env.scene[asset_cfg.name]
    cos_tilt = (-asset.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)
    return torch.acos(cos_tilt) > limit_angle
