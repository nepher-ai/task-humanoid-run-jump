# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reduced-coordinates observation / target-frame math for the frozen BeyondMimic tracker.

Ported from ProtoMotions ``envs/obs/humanoid.py`` and ``envs/obs/target_poses.py``.
Quaternion convention: **wxyz** (Isaac Lab / Isaac Sim default).

Tracker actor observation layout (``obs_dim=157``)::

    reduced_coords_obs          (64) = dof_pos(29) + dof_vel(29)
                                       + root_local_ang_vel(3) + proj_gravity(3)
    reduced_coords_target_frame (64) = rel_anchor_rot6d(6) + dof_vel(29) + dof_pos(29)
    previous_processed_action   (29)

Target-frame order matches ``build_reduced_coords_target_poses`` with
``include_dof_vel=True``, ``include_xy_offset=False``.
"""

from __future__ import annotations

import torch
from torch import Tensor

NUM_DOFS = 29
PROPRIO_DIM = 64  # 29 + 29 + 3 + 3
TARGET_FRAME_DIM = 64  # 6 + 29 + 29
PREV_ACTION_DIM = 29
TRACKER_OBS_DIM = PROPRIO_DIM + TARGET_FRAME_DIM + PREV_ACTION_DIM  # 157
TRACKER_ACT_DIM = 29


def quat_conjugate(q: Tensor) -> Tensor:
    """Conjugate of wxyz quaternion (same as Isaac Lab)."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_mul(q1: Tensor, q2: Tensor) -> Tensor:
    """Hamilton product of wxyz quaternions (Isaac Lab ordering)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_apply(quat: Tensor, vec: Tensor) -> Tensor:
    """Rotate vectors by wxyz quaternion."""
    qvec = quat[..., 1:]
    uv = torch.cross(qvec, vec, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return vec + 2 * (quat[..., :1] * uv + uuv)


def quat_apply_inverse(quat: Tensor, vec: Tensor) -> Tensor:
    return quat_apply(quat_conjugate(quat), vec)


def quat_to_tan_norm(q_wxyz: Tensor) -> Tensor:
    """Convert quaternion (wxyz) to 6D tangent-normal representation."""
    ref_tan = torch.zeros_like(q_wxyz[..., :3])
    ref_tan[..., 0] = 1.0
    ref_norm = torch.zeros_like(q_wxyz[..., :3])
    ref_norm[..., 2] = 1.0
    tan = quat_apply(q_wxyz, ref_tan)
    norm = quat_apply(q_wxyz, ref_norm)
    return torch.cat([tan, norm], dim=-1)


def projected_gravity_b(quat_wxyz: Tensor) -> Tensor:
    """Gravity [0,0,-1] expressed in the body frame of ``quat_wxyz``."""
    gravity = torch.zeros(quat_wxyz.shape[0], 3, device=quat_wxyz.device, dtype=quat_wxyz.dtype)
    gravity[:, 2] = -1.0
    return quat_apply_inverse(quat_wxyz, gravity)


def compute_reduced_coords_obs(
    dof_pos: Tensor,
    dof_vel: Tensor,
    anchor_quat_wxyz: Tensor,
    root_ang_vel_w: Tensor,
    root_quat_wxyz: Tensor | None = None,
) -> Tensor:
    """Build 64-D reduced-coords proprioception for the frozen tracker."""
    frame_quat = root_quat_wxyz if root_quat_wxyz is not None else anchor_quat_wxyz
    local_ang_vel = quat_apply_inverse(frame_quat, root_ang_vel_w)
    proj_g = projected_gravity_b(anchor_quat_wxyz)
    return torch.cat([dof_pos, dof_vel, local_ang_vel, proj_g], dim=-1)


def encode_target_frame(
    current_anchor_quat_wxyz: Tensor,
    target_anchor_quat_wxyz: Tensor,
    target_dof_vel: Tensor,
    target_dof_pos: Tensor,
) -> Tensor:
    """Encode a single 64-D reduced-coords target frame for the tracker.

    Layout: ``rel_anchor_rot6d (6) | dof_vel (29) | dof_pos (29)``.
    """
    rel = quat_mul(quat_conjugate(current_anchor_quat_wxyz), target_anchor_quat_wxyz)
    rot6d = quat_to_tan_norm(rel)
    return torch.cat([rot6d, target_dof_vel, target_dof_pos], dim=-1)


def decode_target_frame(target_frame: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Split a 64-D target frame into ``(rot6d, dof_vel, dof_pos)``."""
    rot6d = target_frame[..., :6]
    dof_vel = target_frame[..., 6 : 6 + NUM_DOFS]
    dof_pos = target_frame[..., 6 + NUM_DOFS : 6 + 2 * NUM_DOFS]
    return rot6d, dof_vel, dof_pos


def build_tracker_obs(
    proprio: Tensor,
    target_frame: Tensor,
    prev_processed_action: Tensor,
) -> Tensor:
    """Concatenate tracker actor observation ``[N, 157]``."""
    return torch.cat([proprio, target_frame, prev_processed_action], dim=-1)
