# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""High-level policy observations for RunJumpHL."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

from humanoid_run_jump.tasks.manager_based.jump.mdp.gait import (
    foot_contact_mask,
    heading_forward,
    lead_foot_forward_mask,
    resolve_ankle_body_ids,
)
from humanoid_run_jump.tasks.manager_based.jump.mdp.observations import (
    reduced_coords_proprio,  # noqa: F401 — re-export for env cfg
)
from humanoid_run_jump.tasks.manager_based.run_jump.mdp.hl_stride import (
    PLANT_BAND_MAX,
    PLANT_BAND_MIN,
    d_star,
    in_plant_band,
    lattice_from_remaining,
    trim_feasible,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def body_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Root lin vel (3) + ang vel (3) + projected gravity (3) = 9."""
    asset = env.scene[asset_cfg.name]
    return torch.cat(
        [
            asset.data.root_lin_vel_b,
            asset.data.root_ang_vel_b,
            asset.data.projected_gravity_b,
        ],
        dim=-1,
    )


def hl_feet(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
) -> torch.Tensor:
    """Contacts (2) + signed lead (1) + right standoff (1) + ankle x offsets (2) = 6."""
    if isinstance(getattr(sensor_cfg, "body_ids", None), slice) or sensor_cfg.body_ids is None:
        try:
            sensor_cfg.resolve(env.scene)
        except Exception:
            pass
    asset = env.scene[asset_cfg.name]
    left_id, right_id = resolve_ankle_body_ids(env, asset_cfg.name)
    contacts = foot_contact_mask(env, sensor_cfg=sensor_cfg).float()
    right_lead = lead_foot_forward_mask(env, lead="right", asset_name=asset_cfg.name).float()
    left_lead = lead_foot_forward_mask(env, lead="left", asset_name=asset_cfg.name).float()
    signed_lead = right_lead - left_lead  # +1 right lead, -1 left lead

    origins = env.scene.env_origins
    right_x = asset.data.body_pos_w[:, right_id, 0] - origins[:, 0]
    d_right = getattr(env, "dist_to_next", None)
    if d_right is None:
        d_right = torch.zeros(env.num_envs, device=env.device)
    else:
        s_f = getattr(env, "next_front_s", None)
        if s_f is not None:
            d_right = (s_f - right_x).clamp(min=-1.0, max=30.0)
    d_right_n = (d_right / 3.0).clamp(-1.0, 10.0)

    left_x = asset.data.body_pos_w[:, left_id, 0] - origins[:, 0]
    root_x = asset.data.root_pos_w[:, 0] - origins[:, 0]
    return torch.stack(
        [
            contacts[:, 0],
            contacts[:, 1],
            signed_lead,
            d_right_n,
            (left_x - root_x).clamp(-1.5, 1.5),
            (right_x - root_x).clamp(-1.5, 1.5),
        ],
        dim=-1,
    )


def course_features(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Nearest-obstacle + remaining-path features (19-D).

    Layout::

        [valid, h_obs, thickness, d_root_norm, d_right_foot_norm,
         d_star, d_err_to_band, in_band, band_lo, band_hi, f_trim_feasible,
         L_req, v_star, v_cmd_err, stride_err, n_committed_norm,
         remaining_path_frac, y_off/2.5, heading_err]

    ``v_cmd_err`` and ``stride_err`` are the two quantities the speed-tracking and
    phase-gain rewards grade, so they are exposed directly rather than left for
    the network to reconstruct from ``v_star`` and the previous action.
    """
    n = env.num_envs
    device = env.device
    asset = env.scene["robot"]
    origins = env.scene.env_origins
    root = asset.data.root_pos_w
    root_s = root[:, 0] - origins[:, 0]
    root_y = root[:, 1] - origins[:, 1]

    _, right_id = resolve_ankle_body_ids(env)
    right_s = asset.data.body_pos_w[:, right_id, 0] - origins[:, 0]

    valid = getattr(env, "has_next_obstacle", None)
    if valid is None:
        valid = torch.zeros(n, dtype=torch.bool, device=device)
    h = getattr(env, "next_h", torch.zeros(n, device=device))
    t = getattr(env, "next_t", torch.full((n,), 0.20, device=device))
    s_f = getattr(env, "next_front_s", torch.zeros(n, device=device))

    d_root = (s_f - root_s).clamp(min=-1.0, max=30.0)
    d_right = (s_f - right_s).clamp(min=-1.0, max=30.0)

    # Index PK_X by commanded vx and the policy's current flight candidate.
    hl = getattr(env, "last_hl_action", None)
    vx = hl[:, 1] if hl is not None else torch.full((n,), 1.5, device=device)
    f_cand = getattr(env, "hl_flight_cand", None)
    if f_cand is None:
        f_cand = torch.full((n,), 0.85, device=device)
    d_tgt = d_star(t, f_cand, vx)
    # Signed distance to the nearest band edge (0 inside).
    below = (PLANT_BAND_MIN - d_right).clamp(min=0.0)
    above = (d_right - PLANT_BAND_MAX).clamp(min=0.0)
    d_err_band = above - below  # negative = too close, positive = too far
    band_flag = in_plant_band(d_right).float()

    tracker = getattr(env, "stride_tracker", None)
    if tracker is not None:
        L = tracker.L_ema
        n_commit = tracker.n_commit
    else:
        L = torch.full((n,), 1.70, device=device)
        n_commit = None
    # Plant-referenced gap: the plan counts strides from the last touchdown.
    remaining = getattr(env, "lattice_remaining", None)
    if remaining is None:
        remaining = d_right.clamp(min=0.0) - d_tgt
    n_rem, L_req, v_star = lattice_from_remaining(remaining, L, n_commit=n_commit)
    v_cmd_err = vx - v_star
    # Signed per-stride shortfall: how much longer (>0) or shorter (<0) each of the
    # remaining strides has to be for the n-th plant to land on d*.
    stride_err = L_req - L

    feasible = trim_feasible(d_right, t, h.clamp(min=0.25), vx).float()

    path_length = getattr(env, "path_length", torch.full((n,), 25.0, device=device))
    remaining_frac = (1.0 - root_s / path_length.clamp(min=1.0)).clamp(0.0, 1.0)

    fwd = heading_forward(asset.data.root_quat_w)
    heading_err = torch.atan2(fwd[:, 1], fwd[:, 0])

    out = torch.stack(
        [
            valid.float(),
            h,
            t,
            (d_root / 3.0).clamp(-1.0, 10.0),
            (d_right / 3.0).clamp(-1.0, 10.0),
            d_tgt,
            d_err_band.clamp(-5.0, 5.0),
            band_flag,
            torch.full((n,), PLANT_BAND_MIN, device=device),
            torch.full((n,), PLANT_BAND_MAX, device=device),
            feasible,
            L_req.clamp(0.0, 2.5),
            (v_star / 3.0).clamp(0.0, 1.5),
            (v_cmd_err / 3.0).clamp(-1.5, 1.5),
            stride_err.clamp(-1.2, 1.2),
            (n_rem / 8.0).clamp(0.0, 2.0),
            remaining_frac,
            (root_y / 2.5).clamp(-2.0, 2.0),
            heading_err.clamp(-1.5, 1.5),
        ],
        dim=-1,
    )
    # Zero course channels when no obstacle is valid (keep remaining_frac / y / heading).
    mask = valid.float().unsqueeze(-1)
    keep = out.clone()
    keep[:, 0:16] = out[:, 0:16] * mask
    keep[:, 16:] = out[:, 16:]
    return keep


def ll_status(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Mode one-hot (2) + jump phase (7) + liftoff/landed (2) + steps-in-mode norm (1) = 12."""
    n = env.num_envs
    device = env.device
    mode = getattr(env, "hl_mode", torch.zeros(n, dtype=torch.long, device=device))
    mode_oh = torch.nn.functional.one_hot(mode.clamp(0, 1), num_classes=2).float()

    phase = getattr(env, "_ep_phase", None)
    if phase is None:
        phase_oh = torch.zeros(n, 7, device=device)
        phase_oh[:, 0] = 1.0
    else:
        phase_oh = torch.nn.functional.one_hot(phase.clamp(0, 6), num_classes=7).float()

    liftoff = getattr(env, "_ep_has_liftoff", torch.zeros(n, dtype=torch.bool, device=device))
    landed = getattr(env, "_ep_has_landed", torch.zeros(n, dtype=torch.bool, device=device))
    steps = getattr(env, "hl_mode_steps", torch.zeros(n, device=device))
    steps_n = (steps / 100.0).clamp(0.0, 5.0)

    return torch.cat(
        [
            mode_oh,
            phase_oh,
            liftoff.float().unsqueeze(-1),
            landed.float().unsqueeze(-1),
            steps_n.unsqueeze(-1),
        ],
        dim=-1,
    )


def last_hl_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Previous 6-D HL action (raw, after tanh scaling storage)."""
    act = getattr(env, "last_hl_action", None)
    if act is None:
        return torch.zeros(env.num_envs, 6, device=env.device)
    return act
