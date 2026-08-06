# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gait-phase helpers for run→jump hand-off.

Default trigger (tracker-favorable phase from inspection):
  right ankle ahead of left along heading AND right foot in contact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def resolve_ankle_body_ids(
    env: ManagerBasedEnv,
    asset_name: str = "robot",
) -> tuple[int, int]:
    """Return ``(left_ankle_idx, right_ankle_idx)`` in the articulation body list.

    Cached on ``env`` after the first call (hot path for per-step rewards).
    """
    cache_key = f"_cached_ankle_body_ids_{asset_name}"
    cached = getattr(env, cache_key, None)
    if cached is not None:
        return cached
    asset = env.scene[asset_name]
    names = list(asset.data.body_names)
    left = next(i for i, n in enumerate(names) if "left_ankle_roll" in n)
    right = next(i for i, n in enumerate(names) if "right_ankle_roll" in n)
    setattr(env, cache_key, (left, right))
    return left, right


def heading_forward(root_quat_w: torch.Tensor) -> torch.Tensor:
    """Unit forward vector in the XY plane from root yaw."""
    n = root_quat_w.shape[0]
    device = root_quat_w.device
    fwd_l = torch.zeros(n, 3, device=device)
    fwd_l[:, 0] = 1.0
    forward = quat_apply(yaw_quat(root_quat_w), fwd_l)
    forward[:, 2] = 0.0
    return forward / forward.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def lead_foot_forward_mask(
    env: ManagerBasedEnv,
    lead: str = "right",
    asset_name: str = "robot",
    min_lead_m: float = 0.05,
) -> torch.Tensor:
    """True when ``lead`` ankle is ahead of the trailing ankle along heading."""
    asset = env.scene[asset_name]
    left_id, right_id = resolve_ankle_body_ids(env, asset_name)
    forward = heading_forward(asset.data.root_quat_w)
    left_xy = asset.data.body_pos_w[:, left_id, :3]
    right_xy = asset.data.body_pos_w[:, right_id, :3]
    delta = right_xy - left_xy
    along = (delta * forward).sum(dim=-1)
    if lead == "right":
        return along > min_lead_m
    if lead == "left":
        return along < -min_lead_m
    raise ValueError(f"lead must be 'left' or 'right', got {lead!r}")


def _contact_ankle_columns(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
) -> tuple[int, int]:
    """Map left/right ankles to columns in the contact sensor force tensor.

    Cached on ``env`` keyed by sensor name (avoids Python name scans every step).
    """
    cache_attr = f"_cached_contact_ankle_cols_{sensor_cfg.name}"
    cached = getattr(env, cache_attr, None)
    if cached is not None:
        return cached

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    sensor_names = list(getattr(contact_sensor, "body_names", []) or [])
    if sensor_cfg.body_ids is not None and len(sensor_names) > 0:
        ids = list(sensor_cfg.body_ids)
        names = [sensor_names[i] for i in ids]
    else:
        names = sensor_names
        ids = list(range(len(names)))

    left_col = next(i for i, n in enumerate(names) if "left_ankle_roll" in n)
    right_col = next(i for i, n in enumerate(names) if "right_ankle_roll" in n)
    setattr(env, cache_attr, (left_col, right_col))
    return left_col, right_col


def foot_contact_mask(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    """Per-foot contact bool ``[N, 2]`` ordered ``[left, right]``."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    contacts = torch.norm(forces, dim=-1) > contact_force_threshold
    left_col, right_col = _contact_ankle_columns(env, sensor_cfg)
    return torch.stack([contacts[:, left_col], contacts[:, right_col]], dim=-1)


def runin_handoff_mask(
    env: ManagerBasedEnv,
    lead: str = "right",
    asset_name: str = "robot",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    contact_force_threshold: float = 5.0,
    min_lead_m: float = 0.05,
    require_trail_air: bool = False,
) -> torch.Tensor:
    """Gait-phase gate for jumping: lead foot forward + in contact."""
    lead_fwd = lead_foot_forward_mask(env, lead=lead, asset_name=asset_name, min_lead_m=min_lead_m)
    contacts_lr = foot_contact_mask(
        env, sensor_cfg=sensor_cfg, contact_force_threshold=contact_force_threshold
    )
    lead_col = 1 if lead == "right" else 0
    trail_col = 0 if lead == "right" else 1
    ok = lead_fwd & contacts_lr[:, lead_col]
    if require_trail_air:
        ok = ok & (~contacts_lr[:, trail_col])
    return ok


__all__ = [
    "resolve_ankle_body_ids",
    "heading_forward",
    "lead_foot_forward_mask",
    "foot_contact_mask",
    "runin_handoff_mask",
]
