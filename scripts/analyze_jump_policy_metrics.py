# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analyze jump-policy metrics from ``best_policy/jump/policy.pt``.

Rolls ``Nepher-G1-Jump-Play-v0`` over a grid of initial run-in speeds and jump
commands ``(h_obstacle, flight_distance)``. Emits CSVs under ``--output-dir``:

* ``jump_episodes.csv`` — one row per finished episode
* ``jump_metrics_summary.csv`` — means by (vx_bin, h, flight)

Metrics:

1. Takeoff stride length (heading-projected same-foot plant gap before liftoff)
2. Actual jump height (peak pelvis rise) and actual flight distance
3. Spatial displacement from episode-start / liftoff to trajectory peak

Usage (from ``task-humanoid-run-jump/``)::

    ..\\..\\isaaclab.bat -p scripts/analyze_jump_policy_metrics.py --headless --num_envs 48
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Analyze jump policy metrics vs vx_init and (h, flight).")
parser.add_argument("--num_envs", type=int, default=48)
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--num_episodes", type=int, default=288, help="Total finished episodes to collect.")
parser.add_argument("--max_steps", type=int, default=80000)
parser.add_argument(
    "--policy",
    type=str,
    default=None,
    help="Path to jump actor. Default: best_policy/jump/policy.pt",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="CSV output directory. Default: outputs/jump_analysis",
)
parser.add_argument("--min-stride-m", type=float, default=0.20)
parser.add_argument("--max-stride-m", type=float, default=2.80)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils import configclass

import humanoid_run_jump  # noqa: F401
from humanoid_run_jump.agents.frozen_actor import FrozenActor, resolve_actor_path
from humanoid_run_jump.robots.joint_order import JointOrderMap
from humanoid_run_jump.tasks.manager_based.jump.jump_env_cfg import G1JumpEnvCfg_PLAY
from humanoid_run_jump.tasks.manager_based.jump.mdp.gait import (
    foot_contact_mask,
    heading_forward,
    resolve_ankle_body_ids,
)
from humanoid_run_jump.tasks.manager_based.jump.mdp.jump_envelope import (
    FLIGHT_DIST_MAX,
    FLIGHT_DIST_MIN,
    H_MAX,
    H_MIN,
    clamp_flight_distance,
    clamp_obstacle_height,
)
from humanoid_run_jump.tracker.reduced_coords import TARGET_FRAME_DIM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = PROJECT_ROOT / "best_policy" / "jump" / "policy.pt"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "jump_analysis"
RUNIN_BANK = PROJECT_ROOT / "motions" / "packaged" / "runin_states.pt"
JUMP_OBS_DIM = 156

# Analysis grids
VX_BINS = [
    (0.75, 1.25, "1.0"),
    (1.25, 1.75, "1.5"),
    (1.75, 2.25, "2.0"),
    (2.25, 2.75, "2.5"),
    (2.75, 3.25, "3.0"),
]
H_CMDS = [0.30, 0.40, 0.50, 0.60, 0.75]
FLIGHT_CMDS = [0.60, 0.80, 1.00, 1.20]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[jump-analyze] wrote {path} ({len(rows)} rows)")


def _load_bank(device: torch.device) -> dict:
    payload = torch.load(RUNIN_BANK, map_location=device, weights_only=False)
    bank = {
        "root_pose": payload["root_pose"].to(device),
        "root_vel": payload["root_vel"].to(device),
        "dof_pos": payload["dof_pos"].to(device),
        "dof_vel": payload["dof_vel"].to(device),
        "vx_cmd": payload["vx_cmd"].to(device),
    }
    return bank


def _bank_indices_for_vx(bank: dict, vx_lo: float, vx_hi: float) -> torch.Tensor:
    vx = bank["vx_cmd"]
    # Prefer commanded speed; fall back to measured root vx if empty.
    mask = (vx >= vx_lo) & (vx < vx_hi)
    idx = mask.nonzero(as_tuple=False).flatten()
    if idx.numel() < 8:
        root_vx = bank["root_vel"][:, 0]
        mask = (root_vx >= vx_lo) & (root_vx < vx_hi)
        idx = mask.nonzero(as_tuple=False).flatten()
    if idx.numel() == 0:
        idx = torch.arange(bank["root_pose"].shape[0], device=vx.device)
    return idx


def _write_bank_state(env, env_ids: torch.Tensor, bank: dict, pick: torch.Tensor) -> None:
    """Place selected run-in bank samples into ``env_ids`` (canonical +x heading)."""
    if env_ids.numel() == 0:
        return
    asset = env.unwrapped.scene["robot"]
    jmap: JointOrderMap = env.unwrapped.joint_order_map
    n = int(env_ids.numel())
    device = env.unwrapped.device

    root_pose = bank["root_pose"][pick].clone()
    root_vel = bank["root_vel"][pick].clone()
    dof_pos = jmap.to_sim(bank["dof_pos"][pick].clone())
    dof_vel = jmap.to_sim(bank["dof_vel"][pick].clone())

    root_pose[:, :2] = env.unwrapped.scene.env_origins[env_ids, :2]

    # Force yaw → +x.
    w0, x0, y0, z0 = root_pose[:, 3], root_pose[:, 4], root_pose[:, 5], root_pose[:, 6]
    yaw = torch.atan2(2.0 * (w0 * z0 + x0 * y0), 1.0 - 2.0 * (y0 * y0 + z0 * z0))
    half = (-yaw) * 0.5
    qw = torch.cos(half)
    qz = torch.sin(half)
    root_pose[:, 3] = qw * w0 - qz * z0
    root_pose[:, 4] = qw * x0 + qz * y0
    root_pose[:, 5] = qw * y0 - qz * x0
    root_pose[:, 6] = qw * z0 + qz * w0

    c = torch.cos(-yaw)
    s = torch.sin(-yaw)
    vx, vy = root_vel[:, 0].clone(), root_vel[:, 1].clone()
    root_vel[:, 0] = c * vx - s * vy
    root_vel[:, 1] = s * vx + c * vy
    wx, wy = root_vel[:, 3].clone(), root_vel[:, 4].clone()
    root_vel[:, 3] = c * wx - s * wy
    root_vel[:, 4] = s * wx + c * wy

    asset.write_root_pose_to_sim(root_pose, env_ids)
    asset.write_root_velocity_to_sim(root_vel, env_ids)
    asset.write_joint_state_to_sim(dof_pos, dof_vel, None, env_ids)


def _policy_obs(env) -> torch.Tensor:
    obs = env.unwrapped.observation_manager.compute()
    if isinstance(obs, dict):
        return obs["policy"]
    return obs


def main() -> None:
    policy_path = resolve_actor_path(args_cli.policy) if args_cli.policy else DEFAULT_POLICY
    out_dir = Path(args_cli.output_dir) if args_cli.output_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build command grid cells, then cycle across envs.
    cells: list[tuple[int, float, float]] = []
    for bi in range(len(VX_BINS)):
        for h in H_CMDS:
            for fl in FLIGHT_CMDS:
                # Height-coupled flight clamp (same as JumpCommand).
                h_t = float(clamp_obstacle_height(torch.tensor(h)))
                fl_t = float(clamp_flight_distance(torch.tensor(fl), torch.tensor(h_t)))
                cells.append((bi, h_t, fl_t))
    # Deduplicate after clamp
    uniq = []
    seen = set()
    for c in cells:
        key = (c[0], round(c[1], 3), round(c[2], 3))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    cells = uniq
    print(f"[jump-analyze] grid cells={len(cells)} (vx_bins×h×flight after clamp)")

    @configclass
    class AnalyzeJumpCfg(G1JumpEnvCfg_PLAY):
        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = args_cli.num_envs
            self.seed = args_cli.seed
            self.commands.jump.debug_vis = False
            self.commands.jump.resampling_time_range = (1.0e6, 1.0e6)
            self.commands.jump.ranges.h_obstacle = (H_MIN, H_MAX)
            self.commands.jump.ranges.flight_distance = (FLIGHT_DIST_MIN, FLIGHT_DIST_MAX)
            self.curriculum.jump_commands = None
            # We inject bank states ourselves after reset.
            self.events.reset_runin.params["bank_prob"] = 0.0
            self.episode_length_s = 7.0

    env_cfg = AnalyzeJumpCfg()
    env = gym.make("Nepher-G1-Jump-Play-v0", cfg=env_cfg)
    unwrapped = env.unwrapped
    device = unwrapped.device
    n = unwrapped.num_envs
    step_dt = float(unwrapped.step_dt)

    actor = FrozenActor(
        policy_path=policy_path,
        device=device,
        expected_obs_dim=JUMP_OBS_DIM,
        expected_act_dim=TARGET_FRAME_DIM,
    )
    print(f"[jump-analyze] policy={actor.policy_path} obs={actor.obs_dim} act={actor.act_dim}")
    print(f"[jump-analyze] num_envs={n} step_dt={step_dt:.4f}s out={out_dir}")

    bank = _load_bank(device)
    bin_indices = [_bank_indices_for_vx(bank, lo, hi) for lo, hi, _ in VX_BINS]
    for i, (lo, hi, lab) in enumerate(VX_BINS):
        print(f"[jump-analyze] vx_bin {lab} [{lo},{hi}): {int(bin_indices[i].numel())} bank samples")

    # Per-env assignment cycling through cells
    env_cell = [cells[i % len(cells)] for i in range(n)]
    env_vx_bin = torch.tensor([c[0] for c in env_cell], device=device, dtype=torch.long)
    env_h = torch.tensor([c[1] for c in env_cell], device=device, dtype=torch.float32)
    env_fl = torch.tensor([c[2] for c in env_cell], device=device, dtype=torch.float32)

    jump_term = unwrapped.command_manager.get_term("jump")
    from isaaclab.managers import SceneEntityCfg

    sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
    sensor_cfg.resolve(unwrapped.scene)

    left_id, right_id = resolve_ankle_body_ids(unwrapped)

    # ---- per-env episode trackers ----
    def fresh_trackers():
        return {
            "start_pos": torch.zeros(n, 3, device=device),
            "start_vx": torch.zeros(n, device=device),
            "start_vx_cmd_bank": torch.zeros(n, device=device),
            "prev_contact": torch.zeros(n, 2, dtype=torch.bool, device=device),
            # Foot geometry for user-requested approach / peak metrics.
            # Indices: 0=left, 1=right.
            "right_first_xy": torch.zeros(n, 2, device=device),
            "has_right_first": torch.zeros(n, dtype=torch.bool, device=device),
            "right_prepush_xy": torch.zeros(n, 2, device=device),  # right plant immediately before left push
            "has_right_prepush": torch.zeros(n, dtype=torch.bool, device=device),
            "left_push_xy": torch.zeros(n, 2, device=device),
            "has_left_push": torch.zeros(n, dtype=torch.bool, device=device),
            "had_liftoff": torch.zeros(n, dtype=torch.bool, device=device),
            "liftoff_pos": torch.zeros(n, 3, device=device),
            "peak_rise": torch.zeros(n, device=device),
            "peak_foot_z": torch.zeros(n, device=device),
            "peak_pos": torch.zeros(n, 3, device=device),  # root at max foot height
            # Higher-foot ankle XY at the same instant as peak_pos (for apex offset c).
            "peak_sole_xy": torch.zeros(n, 2, device=device),
            "has_peak_sole": torch.zeros(n, dtype=torch.bool, device=device),
            "flight_dist": torch.zeros(n, device=device),
            "flight_dist_land": torch.full((n,), float("nan"), device=device),
            "air_steps": torch.zeros(n, dtype=torch.long, device=device),
            "max_phase": torch.zeros(n, dtype=torch.long, device=device),
            "had_land": torch.zeros(n, dtype=torch.bool, device=device),
            "land_pos": torch.zeros(n, 3, device=device),
            "armed": torch.zeros(n, dtype=torch.bool, device=device),
        }

    tr = fresh_trackers()
    episode_rows: list[dict] = []
    cell_cursor = n  # next cell index to assign on reset

    def assign_and_seed(env_ids: torch.Tensor) -> None:
        nonlocal cell_cursor
        if env_ids.numel() == 0:
            return
        ids = env_ids.to(device=device, dtype=torch.long)
        picks = []
        for i, eid in enumerate(ids.tolist()):
            bi, h, fl = cells[cell_cursor % len(cells)]
            cell_cursor += 1
            env_vx_bin[eid] = bi
            env_h[eid] = h
            env_fl[eid] = fl
            pool = bin_indices[bi]
            picks.append(int(pool[torch.randint(0, pool.numel(), (1,), device=device)].item()))
        pick_t = torch.tensor(picks, device=device, dtype=torch.long)
        _write_bank_state(env, ids, bank, pick_t)

        jump_term.set_command(ids, env_h[ids], env_fl[ids])
        if hasattr(jump_term, "time_left"):
            jump_term.time_left[ids] = 1.0e6

        asset = unwrapped.scene["robot"]
        tr["start_pos"][ids] = asset.data.root_pos_w[ids].clone()
        tr["start_vx"][ids] = asset.data.root_lin_vel_b[ids, 0].clone()
        tr["start_vx_cmd_bank"][ids] = bank["vx_cmd"][pick_t].clone()
        tr["prev_contact"][ids] = False
        tr["right_first_xy"][ids] = 0.0
        tr["has_right_first"][ids] = False
        tr["right_prepush_xy"][ids] = 0.0
        tr["has_right_prepush"][ids] = False
        tr["left_push_xy"][ids] = 0.0
        tr["has_left_push"][ids] = False
        tr["had_liftoff"][ids] = False
        tr["liftoff_pos"][ids] = 0.0
        tr["peak_rise"][ids] = 0.0
        tr["peak_foot_z"][ids] = 0.0
        tr["peak_pos"][ids] = asset.data.root_pos_w[ids].clone()
        tr["peak_sole_xy"][ids] = 0.0
        tr["has_peak_sole"][ids] = False
        tr["flight_dist"][ids] = 0.0
        tr["flight_dist_land"][ids] = float("nan")
        tr["air_steps"][ids] = 0
        tr["max_phase"][ids] = 0
        tr["had_land"][ids] = False
        tr["land_pos"][ids] = 0.0
        tr["armed"][ids] = True

        unwrapped.scene.write_data_to_sim()
        # If right foot already in contact at hand-off, count that as first contact.
        c0 = foot_contact_mask(unwrapped, sensor_cfg=sensor_cfg)
        rxy = asset.data.body_pos_w[:, right_id, :2]
        already_r = c0[:, 1].clone()
        already_r[:] = False
        already_r[ids] = c0[ids, 1]
        already_ids = already_r.nonzero(as_tuple=False).squeeze(-1)
        if already_ids.numel() > 0:
            tr["right_first_xy"][already_ids] = rxy[already_ids]
            tr["has_right_first"][already_ids] = True
            tr["right_prepush_xy"][already_ids] = rxy[already_ids]
            tr["has_right_prepush"][already_ids] = True
            tr["prev_contact"][already_ids] = c0[already_ids]

    def snapshot_episode(eid: int, terminated: bool, truncated: bool) -> dict:
        bi = int(env_vx_bin[eid].item())
        lo, hi, lab = VX_BINS[bi]
        peak = tr["peak_pos"][eid]
        liftoff = tr["liftoff_pos"][eid]
        land = tr["land_pos"][eid]
        had_lo = bool(tr["had_liftoff"][eid].item())
        had_land = bool(tr["had_land"][eid].item())

        # Approach stride: left push-off foot ↔ right foot that planted immediately prior.
        # absolute = XY-plane distance; x = Δx (world +x after canonicalize).
        approach_xy = float("nan")
        approach_x = float("nan")
        if bool(tr["has_left_push"][eid].item()) and bool(tr["has_right_prepush"][eid].item()):
            lxy = tr["left_push_xy"][eid]
            rxy = tr["right_prepush_xy"][eid]
            d = lxy - rxy
            approach_xy = float(torch.norm(d).item())
            approach_x = float(d[0].item())

        # Right-first-contact → trajectory peak (root at max foot height).
        rfirst_peak_xy = float("nan")
        rfirst_peak_x = float("nan")
        if bool(tr["has_right_first"][eid].item()):
            r0 = tr["right_first_xy"][eid]
            d2 = peak[:2] - r0
            rfirst_peak_xy = float(torch.norm(d2).item())
            rfirst_peak_x = float(d2[0].item())

        # Apex offset c = peak_sole_x - root_x at max sole height.
        apex_offset_c = float("nan")
        if bool(tr["has_peak_sole"][eid].item()):
            apex_offset_c = float((tr["peak_sole_xy"][eid, 0] - peak[0]).item())

        peak_foot = float(tr["peak_foot_z"][eid].item())
        peak_rise = float(tr["peak_rise"][eid].item())
        flight_air = float(tr["flight_dist"][eid].item())
        flight_land = float(tr["flight_dist_land"][eid].item())
        flight_best = flight_land if not np.isnan(flight_land) else flight_air

        max_phase = int(tr["max_phase"][eid].item())
        air_steps = int(tr["air_steps"][eid].item())
        valid_jump = int(
            had_lo
            and max_phase >= 3
            and (peak_foot >= 0.20 or flight_best >= 0.35 or air_steps >= 8)
        )

        return {
            "env_id": eid,
            "vx_bin": lab,
            "vx_bin_lo": lo,
            "vx_bin_hi": hi,
            "vx_init": float(tr["start_vx"][eid].item()),
            "vx_bank_cmd": float(tr["start_vx_cmd_bank"][eid].item()),
            "h_cmd": float(env_h[eid].item()),
            "flight_cmd": float(env_fl[eid].item()),
            # User-defined approach: left push-off ↔ prior right plant
            "approach_stride_xy_m": "" if np.isnan(approach_xy) else approach_xy,
            "approach_stride_x_m": "" if np.isnan(approach_x) else approach_x,
            "has_right_first_contact": int(bool(tr["has_right_first"][eid].item())),
            "has_left_push": int(bool(tr["has_left_push"][eid].item())),
            "had_liftoff": int(had_lo),
            "had_land": int(had_land),
            "valid_jump": valid_jump,
            "max_phase": max_phase,
            "air_steps": air_steps,
            "actual_jump_height_m": peak_foot,
            "actual_peak_foot_z_m": peak_foot,
            "actual_peak_rise_m": peak_rise,
            "actual_flight_distance_m": flight_best,
            "actual_flight_distance_air_m": flight_air,
            "actual_flight_distance_land_m": "" if np.isnan(flight_land) else flight_land,
            # Right-first-contact → peak
            "disp_right_first_to_peak_xy_m": "" if np.isnan(rfirst_peak_xy) else rfirst_peak_xy,
            "disp_right_first_to_peak_x_m": "" if np.isnan(rfirst_peak_x) else rfirst_peak_x,
            "apex_offset_c_m": "" if np.isnan(apex_offset_c) else apex_offset_c,
            "disp_liftoff_to_land_x_m": (
                float((land[0] - liftoff[0]).item()) if (had_lo and had_land) else ""
            ),
            "full_success": 0,
            "terminated": int(terminated),
            "truncated": int(truncated),
        }

    # Initial reset + seed
    env.reset()
    all_ids = torch.arange(n, device=device)
    assign_and_seed(all_ids)
    unwrapped.scene.write_data_to_sim()

    collected = 0
    step = 0
    print(f"[jump-analyze] collecting {args_cli.num_episodes} episodes...")

    while collected < args_cli.num_episodes and step < args_cli.max_steps and simulation_app.is_running():
        jump_term.set_command(all_ids, env_h, env_fl)
        if hasattr(jump_term, "time_left"):
            jump_term.time_left[:] = 1.0e6

        obs = _policy_obs(unwrapped)
        with torch.inference_mode():
            actions = actor.act(obs)
        _, _, terminated, truncated, _ = env.step(actions.clone())
        step += 1

        asset = unwrapped.scene["robot"]
        contacts = foot_contact_mask(unwrapped, sensor_cfg=sensor_cfg)
        foot_xy = torch.stack(
            [asset.data.body_pos_w[:, left_id, :2], asset.data.body_pos_w[:, right_id, :2]],
            dim=1,
        )
        root_pos = asset.data.root_pos_w

        env_liftoff = unwrapped._ep_has_liftoff
        env_landed = unwrapped._ep_has_landed
        env_phase = unwrapped._ep_phase
        env_peak_foot = unwrapped._ep_peak_foot_z.max(dim=-1).values
        env_peak_rise = unwrapped._ep_peak_rise
        env_flight = unwrapped._ep_flight_distance
        env_air_steps = unwrapped._ep_air_steps
        env_lo_pos = unwrapped._ep_liftoff_pos_w

        tr["max_phase"] = torch.maximum(tr["max_phase"], env_phase)
        tr["air_steps"] = torch.maximum(tr["air_steps"], env_air_steps)

        pre_lo = tr["armed"] & (~tr["had_liftoff"])
        plant = contacts & (~tr["prev_contact"])
        release = (~contacts) & tr["prev_contact"]

        # Right foot first contact + most recent right plant before left push.
        right_plant_ids = (plant[:, 1] & pre_lo).nonzero(as_tuple=False).squeeze(-1)
        if right_plant_ids.numel() > 0:
            first_mask = ~tr["has_right_first"][right_plant_ids]
            first_ids = right_plant_ids[first_mask]
            if first_ids.numel() > 0:
                tr["right_first_xy"][first_ids] = foot_xy[first_ids, 1]
                tr["has_right_first"][first_ids] = True
            # Always refresh "immediately prior" right plant until left pushes.
            upd = right_plant_ids[~tr["has_left_push"][right_plant_ids]]
            if upd.numel() > 0:
                tr["right_prepush_xy"][upd] = foot_xy[upd, 1]
                tr["has_right_prepush"][upd] = True

        # Also treat sustained right contact (no rising edge yet) as prepush plant.
        right_hold = contacts[:, 1] & pre_lo & (~tr["has_left_push"])
        hold_ids = right_hold.nonzero(as_tuple=False).squeeze(-1)
        if hold_ids.numel() > 0:
            need_first = hold_ids[~tr["has_right_first"][hold_ids]]
            if need_first.numel() > 0:
                tr["right_first_xy"][need_first] = foot_xy[need_first, 1]
                tr["has_right_first"][need_first] = True
            tr["right_prepush_xy"][hold_ids] = foot_xy[hold_ids, 1]
            tr["has_right_prepush"][hold_ids] = True

        # Left push-off: left foot leaves contact after a right plant has been seen.
        left_release = release[:, 0] & pre_lo & tr["has_right_prepush"] & (~tr["has_left_push"])
        left_ids = left_release.nonzero(as_tuple=False).squeeze(-1)
        if left_ids.numel() > 0:
            # Use last-contact left foot XY (previous frame contact → current release).
            # Foot XY on the release frame is still the push location.
            tr["left_push_xy"][left_ids] = foot_xy[left_ids, 0]
            tr["has_left_push"][left_ids] = True

        # True jump liftoff (env latch)
        just_lo = env_liftoff & (~tr["had_liftoff"]) & tr["armed"]
        if just_lo.any():
            ids = just_lo.nonzero(as_tuple=False).squeeze(-1)
            tr["had_liftoff"][ids] = True
            tr["liftoff_pos"][ids] = env_lo_pos[ids].clone()
            # If left never released cleanly, use left foot XY at liftoff as push pose.
            need_push = ids[~tr["has_left_push"][ids]]
            if need_push.numel() > 0:
                tr["left_push_xy"][need_push] = foot_xy[need_push, 0]
                tr["has_left_push"][need_push] = True
            # Ensure right prepush exists (fallback to first right / current right).
            need_r = ids[~tr["has_right_prepush"][ids]]
            if need_r.numel() > 0:
                src = torch.where(
                    tr["has_right_first"][need_r].unsqueeze(-1),
                    tr["right_first_xy"][need_r],
                    foot_xy[need_r, 1],
                )
                tr["right_prepush_xy"][need_r] = src
                tr["has_right_prepush"][need_r] = True
                need_rf = need_r[~tr["has_right_first"][need_r]]
                if need_rf.numel() > 0:
                    tr["right_first_xy"][need_rf] = foot_xy[need_rf, 1]
                    tr["has_right_first"][need_rf] = True

        # While jump-airborne, track peaks / flight.
        in_jump = tr["had_liftoff"] & (~tr["had_land"]) & tr["armed"]
        if in_jump.any():
            ids = in_jump.nonzero(as_tuple=False).squeeze(-1)
            foot_now = env_peak_foot[ids]
            better_f = foot_now > tr["peak_foot_z"][ids]
            if better_f.any():
                u = ids[better_f]
                tr["peak_foot_z"][u] = foot_now[better_f]
                tr["peak_pos"][u] = root_pos[u].clone()
                # Higher sole ankle XY at this peak (c = peak_sole_x - root_x).
                peak_per_foot = unwrapped._ep_peak_foot_z[u]
                use_right = peak_per_foot[:, 1] >= peak_per_foot[:, 0]
                sole_xy = torch.where(
                    use_right.unsqueeze(-1),
                    foot_xy[u, 1],
                    foot_xy[u, 0],
                )
                tr["peak_sole_xy"][u] = sole_xy
                tr["has_peak_sole"][u] = True
            rise_now = env_peak_rise[ids]
            better_r = rise_now > tr["peak_rise"][ids]
            if better_r.any():
                u = ids[better_r]
                tr["peak_rise"][u] = rise_now[better_r]
            tr["flight_dist"][ids] = torch.maximum(tr["flight_dist"][ids], env_flight[ids])

        just_land = env_landed & (~tr["had_land"]) & tr["armed"] & tr["had_liftoff"]
        if just_land.any():
            ids = just_land.nonzero(as_tuple=False).squeeze(-1)
            tr["had_land"][ids] = True
            tr["land_pos"][ids] = root_pos[ids].clone()
            delta_ll = root_pos[ids] - tr["liftoff_pos"][ids]
            fwd = jump_term._forward_w[ids]
            tr["flight_dist_land"][ids] = (delta_ll * fwd).sum(dim=-1).clamp(min=0.0)
            tr["flight_dist"][ids] = torch.maximum(tr["flight_dist"][ids], env_flight[ids])
            tr["peak_foot_z"][ids] = torch.maximum(tr["peak_foot_z"][ids], env_peak_foot[ids])
            tr["peak_rise"][ids] = torch.maximum(tr["peak_rise"][ids], env_peak_rise[ids])

        tr["prev_contact"] = contacts.clone()

        done = terminated | truncated
        if isinstance(done, torch.Tensor) and done.any():
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            for eid in done_ids.tolist():
                if not bool(tr["armed"][eid].item()):
                    continue
                if bool(unwrapped._ep_has_liftoff[eid].item()) and not bool(tr["had_liftoff"][eid].item()):
                    tr["had_liftoff"][eid] = True
                    tr["liftoff_pos"][eid] = unwrapped._ep_liftoff_pos_w[eid].clone()
                    if not bool(tr["has_left_push"][eid].item()):
                        tr["left_push_xy"][eid] = foot_xy[eid, 0]
                        tr["has_left_push"][eid] = True
                    if not bool(tr["has_right_prepush"][eid].item()):
                        tr["right_prepush_xy"][eid] = (
                            tr["right_first_xy"][eid]
                            if bool(tr["has_right_first"][eid].item())
                            else foot_xy[eid, 1]
                        )
                        tr["has_right_prepush"][eid] = True
                    if not bool(tr["has_right_first"][eid].item()):
                        tr["right_first_xy"][eid] = foot_xy[eid, 1]
                        tr["has_right_first"][eid] = True
                tr["peak_foot_z"][eid] = max(
                    float(tr["peak_foot_z"][eid].item()),
                    float(unwrapped._ep_peak_foot_z[eid].max().item()),
                )
                tr["peak_rise"][eid] = max(
                    float(tr["peak_rise"][eid].item()),
                    float(unwrapped._ep_peak_rise[eid].item()),
                )
                tr["flight_dist"][eid] = max(
                    float(tr["flight_dist"][eid].item()),
                    float(unwrapped._ep_flight_distance[eid].item()),
                )
                if bool(unwrapped._ep_has_landed[eid].item()) and not bool(tr["had_land"][eid].item()):
                    tr["had_land"][eid] = True
                    tr["land_pos"][eid] = asset.data.root_pos_w[eid].clone()
                    dll = tr["land_pos"][eid] - tr["liftoff_pos"][eid]
                    tr["flight_dist_land"][eid] = (dll * jump_term._forward_w[eid]).sum().clamp(min=0.0)

                row = snapshot_episode(
                    eid,
                    bool(terminated[eid].item()) if terminated.dim() > 0 else bool(terminated),
                    bool(truncated[eid].item()) if truncated.dim() > 0 else bool(truncated),
                )
                episode_rows.append(row)
                collected += 1
                if collected % 24 == 0 or collected >= args_cli.num_episodes:
                    n_valid = sum(1 for r in episode_rows if r["valid_jump"])
                    print(
                        f"[jump-analyze] collected {collected}/{args_cli.num_episodes} "
                        f"(valid_jumps={n_valid}, step={step})"
                    )
                if collected >= args_cli.num_episodes:
                    break
            if collected < args_cli.num_episodes:
                assign_and_seed(done_ids)
                unwrapped.scene.write_data_to_sim()

    env.close()

    if not episode_rows:
        raise RuntimeError("No jump episodes collected. Check jump policy / bank / contacts.")

    fields = list(episode_rows[0].keys())
    _write_csv(out_dir / "jump_episodes.csv", episode_rows, fields)

    # Prefer valid-jump rows for relationship summaries (failed early contacts bias means down).
    valid_rows = [r for r in episode_rows if r["valid_jump"] == 1]
    summarize_pool = valid_rows if len(valid_rows) >= max(20, len(episode_rows) // 10) else episode_rows
    print(
        f"[jump-analyze] summarizing on {len(summarize_pool)}/"
        f"{len(episode_rows)} episodes "
        f"({'valid_jump only' if summarize_pool is valid_rows else 'all — too few valid'})"
    )

    def _mean(vals):
        vals = [v for v in vals if v != "" and v is not None and not (isinstance(v, float) and np.isnan(v))]
        if not vals:
            return ""
        return float(np.mean(np.asarray(vals, dtype=np.float64)))

    def _std(vals):
        vals = [v for v in vals if v != "" and v is not None and not (isinstance(v, float) and np.isnan(v))]
        if not vals:
            return ""
        return float(np.std(np.asarray(vals, dtype=np.float64)))

    summary = []
    keys = sorted({(r["vx_bin"], r["h_cmd"], r["flight_cmd"]) for r in summarize_pool})
    for vx_bin, h_cmd, flight_cmd in keys:
        sub = [
            r
            for r in summarize_pool
            if r["vx_bin"] == vx_bin
            and abs(r["h_cmd"] - h_cmd) < 1e-6
            and abs(r["flight_cmd"] - flight_cmd) < 1e-6
        ]
        summary.append(
            {
                "vx_bin": vx_bin,
                "h_cmd": h_cmd,
                "flight_cmd": flight_cmd,
                "n_episodes": len(sub),
                "vx_init_mean": _mean([r["vx_init"] for r in sub]),
                "approach_stride_xy_mean_m": _mean([r["approach_stride_xy_m"] for r in sub]),
                "approach_stride_xy_std_m": _std([r["approach_stride_xy_m"] for r in sub]),
                "approach_stride_x_mean_m": _mean([r["approach_stride_x_m"] for r in sub]),
                "approach_stride_x_std_m": _std([r["approach_stride_x_m"] for r in sub]),
                "actual_jump_height_mean_m": _mean([r["actual_jump_height_m"] for r in sub]),
                "actual_jump_height_std_m": _std([r["actual_jump_height_m"] for r in sub]),
                "actual_peak_rise_mean_m": _mean([r["actual_peak_rise_m"] for r in sub]),
                "actual_flight_distance_mean_m": _mean([r["actual_flight_distance_m"] for r in sub]),
                "actual_flight_distance_std_m": _std([r["actual_flight_distance_m"] for r in sub]),
                "disp_right_first_to_peak_xy_mean_m": _mean(
                    [r["disp_right_first_to_peak_xy_m"] for r in sub]
                ),
                "disp_right_first_to_peak_x_mean_m": _mean(
                    [r["disp_right_first_to_peak_x_m"] for r in sub]
                ),
                "disp_liftoff_to_land_x_mean_m": _mean([r["disp_liftoff_to_land_x_m"] for r in sub]),
            }
        )
    _write_csv(
        out_dir / "jump_metrics_summary.csv",
        summary,
        list(summary[0].keys()) if summary else ["vx_bin"],
    )

    by_vx = []
    for _, _, lab in VX_BINS:
        sub = [r for r in summarize_pool if r["vx_bin"] == lab]
        all_sub = [r for r in episode_rows if r["vx_bin"] == lab]
        if not sub and not all_sub:
            continue
        by_vx.append(
            {
                "vx_bin": lab,
                "n_episodes": len(sub),
                "n_all": len(all_sub),
                "valid_jump_rate": (len(sub) / len(all_sub)) if all_sub else 0.0,
                "vx_init_mean": _mean([r["vx_init"] for r in sub]),
                "approach_stride_xy_mean_m": _mean([r["approach_stride_xy_m"] for r in sub]),
                "approach_stride_x_mean_m": _mean([r["approach_stride_x_m"] for r in sub]),
                "actual_jump_height_mean_m": _mean([r["actual_jump_height_m"] for r in sub]),
                "actual_flight_distance_mean_m": _mean([r["actual_flight_distance_m"] for r in sub]),
                "disp_right_first_to_peak_xy_mean_m": _mean(
                    [r["disp_right_first_to_peak_xy_m"] for r in sub]
                ),
                "disp_right_first_to_peak_x_mean_m": _mean(
                    [r["disp_right_first_to_peak_x_m"] for r in sub]
                ),
            }
        )
    _write_csv(out_dir / "jump_by_vx_bin.csv", by_vx, list(by_vx[0].keys()) if by_vx else ["vx_bin"])

    by_h = []
    for h in sorted({r["h_cmd"] for r in summarize_pool}):
        sub = [r for r in summarize_pool if abs(r["h_cmd"] - h) < 1e-6]
        by_h.append(
            {
                "h_cmd": h,
                "n_episodes": len(sub),
                "approach_stride_xy_mean_m": _mean([r["approach_stride_xy_m"] for r in sub]),
                "approach_stride_x_mean_m": _mean([r["approach_stride_x_m"] for r in sub]),
                "actual_jump_height_mean_m": _mean([r["actual_jump_height_m"] for r in sub]),
                "actual_flight_distance_mean_m": _mean([r["actual_flight_distance_m"] for r in sub]),
                "disp_right_first_to_peak_xy_mean_m": _mean(
                    [r["disp_right_first_to_peak_xy_m"] for r in sub]
                ),
                "disp_right_first_to_peak_x_mean_m": _mean(
                    [r["disp_right_first_to_peak_x_m"] for r in sub]
                ),
            }
        )
    _write_csv(out_dir / "jump_by_h_cmd.csv", by_h, list(by_h[0].keys()) if by_h else ["h_cmd"])

    by_fl = []
    for fl in sorted({r["flight_cmd"] for r in summarize_pool}):
        sub = [r for r in summarize_pool if abs(r["flight_cmd"] - fl) < 1e-6]
        by_fl.append(
            {
                "flight_cmd": fl,
                "n_episodes": len(sub),
                "approach_stride_xy_mean_m": _mean([r["approach_stride_xy_m"] for r in sub]),
                "approach_stride_x_mean_m": _mean([r["approach_stride_x_m"] for r in sub]),
                "actual_jump_height_mean_m": _mean([r["actual_jump_height_m"] for r in sub]),
                "actual_flight_distance_mean_m": _mean([r["actual_flight_distance_m"] for r in sub]),
                "disp_right_first_to_peak_xy_mean_m": _mean(
                    [r["disp_right_first_to_peak_xy_m"] for r in sub]
                ),
                "disp_right_first_to_peak_x_mean_m": _mean(
                    [r["disp_right_first_to_peak_x_m"] for r in sub]
                ),
            }
        )
    _write_csv(out_dir / "jump_by_flight_cmd.csv", by_fl, list(by_fl[0].keys()) if by_fl else ["flight_cmd"])

    manifest = [
        {
            "policy": str(policy_path),
            "num_envs": n,
            "num_episodes": len(episode_rows),
            "num_valid_jumps": len(valid_rows),
            "summarize_on": "valid_jump" if summarize_pool is valid_rows else "all",
            "step_dt": step_dt,
            "seed": args_cli.seed,
            "liftoff_rate": _mean([r["had_liftoff"] for r in episode_rows]),
            "valid_jump_rate": len(valid_rows) / max(len(episode_rows), 1),
            "mean_approach_stride_xy_m": _mean([r["approach_stride_xy_m"] for r in summarize_pool]),
            "mean_approach_stride_x_m": _mean([r["approach_stride_x_m"] for r in summarize_pool]),
            "mean_jump_height_foot_m": _mean([r["actual_jump_height_m"] for r in summarize_pool]),
            "mean_flight_distance_m": _mean([r["actual_flight_distance_m"] for r in summarize_pool]),
            "mean_disp_right_first_to_peak_xy_m": _mean(
                [r["disp_right_first_to_peak_xy_m"] for r in summarize_pool]
            ),
            "mean_disp_right_first_to_peak_x_m": _mean(
                [r["disp_right_first_to_peak_x_m"] for r in summarize_pool]
            ),
            "mean_apex_offset_c_m": _mean([r["apex_offset_c_m"] for r in summarize_pool]),
            "std_apex_offset_c_m": _std([r["apex_offset_c_m"] for r in summarize_pool]),
            "note": (
                "Approach = XY/X distance between left push-off foot and prior right plant. "
                "Peak disp = right-first-contact to root at max foot height (XY/X). "
                "apex_offset_c = peak_sole_x - root_x at max sole height."
            ),
        }
    ]
    _write_csv(out_dir / "analysis_manifest.csv", manifest, list(manifest[0].keys()))
    print("[jump-analyze] done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
