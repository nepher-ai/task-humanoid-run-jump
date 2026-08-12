# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analyze run-policy stride / velocity relationships from best_policy/run/policy.pt.

Produces three CSV suites under ``--output-dir`` (default ``outputs/stride_analysis``):

1. ``stride_vs_velocity.csv`` — left/right stride length vs steady velocity commands
2. ``leading_foot_from_idle.csv`` — first leading foot + first stride after idle→go
3. ``velocity_tracking_transients.csv`` — tracking + stride under commanded speed changes
   (plus ``velocity_tracking_summary.csv`` aggregates)

Usage (from ``task-humanoid-run-jump/``)::

    ..\\..\\isaaclab.bat -p scripts/analyze_run_stride_velocity.py --headless --num_envs 48
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Analyze run stride vs velocity with frozen run policy.")
parser.add_argument("--num_envs", type=int, default=48)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument(
    "--policy",
    type=str,
    default=None,
    help="Path to exported run actor. Default: best_policy/run/policy.pt",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="CSV output directory. Default: outputs/stride_analysis",
)
parser.add_argument("--steady-steps", type=int, default=400, help="Steps per steady-vx trial after warmup.")
parser.add_argument("--warmup-steps", type=int, default=80)
parser.add_argument("--idle-steps", type=int, default=60, help="Standing steps before go command.")
parser.add_argument("--go-steps", type=int, default=200, help="Steps after idle→go for lead-foot trial.")
parser.add_argument("--phase-a-steps", type=int, default=180, help="Steps at vx_a before speed change.")
parser.add_argument("--phase-b-steps", type=int, default=220, help="Steps at vx_b after speed change.")
parser.add_argument("--min-stride-m", type=float, default=0.15, help="Ignore tiny plant-to-plant gaps.")
parser.add_argument("--max-stride-m", type=float, default=2.5, help="Ignore outlier plant gaps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import humanoid_run_jump  # noqa: F401
from humanoid_run_jump.agents.frozen_actor import DEFAULT_RUN_POLICY_PATH, FrozenActor
from humanoid_run_jump.tasks.manager_based.run.mdp.gait import (
    foot_contact_mask,
    heading_forward,
    resolve_ankle_body_ids,
)
from humanoid_run_jump.tasks.manager_based.run.mdp.observations import (
    base_lin_vel,
    last_action,
    reduced_coords_proprio,
    velocity_commands,
)
from humanoid_run_jump.tasks.manager_based.run.run_env_cfg import G1RunEnvCfg_PLAY, RunSceneCfg
from humanoid_run_jump.tracker.reduced_coords import TARGET_FRAME_DIM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "stride_analysis"
RUN_OBS_DIM = 64 + 3 + 3 + 64

# Steady velocity grid (m/s) and idle→go / step-change schedules.
STEADY_VX = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
IDLE_GO_VX = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
# (vx_before, vx_after) pairs for transient tracking.
VX_CHANGES = [
    (1.0, 2.0),
    (2.0, 1.0),
    (1.0, 3.0),
    (0.5, 2.0),
    (2.5, 1.5),
    (1.5, 2.5),
]


def _build_run_obs(env) -> torch.Tensor:
    return torch.cat(
        [
            reduced_coords_proprio(env),
            base_lin_vel(env),
            velocity_commands(env, command_name="base_velocity"),
            last_action(env),
        ],
        dim=-1,
    )


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[analyze] wrote {path} ({len(rows)} rows)")


def _make_env(num_envs: int, seed: int):
    @configclass
    class AnalyzeSceneCfg(RunSceneCfg):
        contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/pelvis/.*",
            history_length=3,
            track_air_time=True,
        )

    @configclass
    class AnalyzeEnvCfg(G1RunEnvCfg_PLAY):
        scene: AnalyzeSceneCfg = AnalyzeSceneCfg(num_envs=64, env_spacing=3.0)

        def __post_init__(self):
            super().__post_init__()
            self.scene.robot.spawn.activate_contact_sensors = True
            if self.scene.contact_forces is not None:
                self.scene.contact_forces.update_period = self.sim.dt
            self.episode_length_s = 120.0
            self.commands.base_velocity.ranges.lin_vel_x = (0.0, 3.0)
            self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
            self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
            self.commands.base_velocity.rel_standing_envs = 0.0
            self.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
            self.commands.base_velocity.debug_vis = False
            self.curriculum.lin_vel_cmd = None

    env_cfg = AnalyzeEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    env = gym.make("Nepher-G1-Run-Play-v0", cfg=env_cfg)
    return env


def _vel_term(env):
    return env.unwrapped.command_manager.get_term("base_velocity")


def _set_vx(env, vx: torch.Tensor | float) -> None:
    """Force forward-only velocity commands (no standing / no auto-resample side effects)."""
    term = _vel_term(env)
    n = env.unwrapped.num_envs
    device = env.unwrapped.device
    if isinstance(vx, (int, float)):
        vx_t = torch.full((n,), float(vx), device=device)
    else:
        vx_t = vx.to(device=device).reshape(n)
    term.is_standing_env[:] = False
    term.vel_command_b[:, 0] = vx_t
    term.vel_command_b[:, 1] = 0.0
    term.vel_command_b[:, 2] = 0.0
    # Keep command timer from firing a resample.
    if hasattr(term, "time_left"):
        term.time_left[:] = 1.0e6


class StrideTracker:
    """Detect foot plants and measure same-foot stride lengths along heading."""

    def __init__(self, num_envs: int, device, min_stride_m: float, max_stride_m: float):
        self.n = num_envs
        self.device = device
        self.min_stride_m = min_stride_m
        self.max_stride_m = max_stride_m
        self.prev_contact = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        self.has_prev = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        self.last_plant_xy = torch.zeros(num_envs, 2, 2, device=device)  # [env, foot, xy]
        self.last_root_xy = torch.zeros(num_envs, 2, 2, device=device)
        self.events: list[dict] = []

    def reset_state(self) -> None:
        self.prev_contact.zero_()
        self.has_prev.zero_()
        self.last_plant_xy.zero_()
        self.last_root_xy.zero_()

    def update(
        self,
        env,
        contacts: torch.Tensor,
        step: int,
        phase: str,
        vx_cmd: torch.Tensor,
        vx_actual: torch.Tensor,
        env_meta: list[dict] | None = None,
        record: bool = True,
    ) -> list[dict]:
        asset = env.unwrapped.scene["robot"]
        left_id, right_id = resolve_ankle_body_ids(env.unwrapped)
        forward = heading_forward(asset.data.root_quat_w)
        foot_xy = torch.stack(
            [
                asset.data.body_pos_w[:, left_id, :2],
                asset.data.body_pos_w[:, right_id, :2],
            ],
            dim=1,
        )  # [N, 2, 2]
        root_xy = asset.data.root_pos_w[:, :2]
        plant = contacts & (~self.prev_contact)
        new_events: list[dict] = []
        for foot_i, foot_name in enumerate(("left", "right")):
            ids = plant[:, foot_i].nonzero(as_tuple=False).squeeze(-1)
            for idx in ids.tolist():
                fxy = foot_xy[idx, foot_i]
                rxy = root_xy[idx]
                fwd = forward[idx, :2]
                stride_foot = float("nan")
                stride_root = float("nan")
                if bool(self.has_prev[idx, foot_i]):
                    d_foot = fxy - self.last_plant_xy[idx, foot_i]
                    d_root = rxy - self.last_root_xy[idx, foot_i]
                    stride_foot = float((d_foot * fwd).sum().item())
                    stride_root = float((d_root * fwd).sum().item())
                self.last_plant_xy[idx, foot_i] = fxy
                self.last_root_xy[idx, foot_i] = rxy
                self.has_prev[idx, foot_i] = True
                ok = (
                    not np.isnan(stride_foot)
                    and self.min_stride_m <= stride_foot <= self.max_stride_m
                )
                if record and ok:
                    meta = env_meta[idx] if env_meta is not None else {}
                    row = {
                        "phase": phase,
                        "step": int(step),
                        "env_id": int(idx),
                        "foot": foot_name,
                        "vx_cmd": float(vx_cmd[idx].item()),
                        "vx_actual": float(vx_actual[idx].item()),
                        "stride_length_m": stride_foot,
                        "root_progress_m": stride_root,
                        **meta,
                    }
                    self.events.append(row)
                    new_events.append(row)
        self.prev_contact = contacts.clone()
        return new_events


def _assign_cycle(values: list, num_envs: int) -> list:
    return [values[i % len(values)] for i in range(num_envs)]


def main() -> None:
    policy_path = Path(args_cli.policy) if args_cli.policy else DEFAULT_RUN_POLICY_PATH
    out_dir = Path(args_cli.output_dir) if args_cli.output_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env(args_cli.num_envs, args_cli.seed)
    unwrapped = env.unwrapped
    device = unwrapped.device
    n = unwrapped.num_envs
    step_dt = float(unwrapped.step_dt)

    actor = FrozenActor(
        policy_path=policy_path,
        device=device,
        expected_obs_dim=RUN_OBS_DIM,
        expected_act_dim=TARGET_FRAME_DIM,
    )
    print(f"[analyze] policy={actor.policy_path}")
    print(f"[analyze] num_envs={n} step_dt={step_dt:.4f}s out={out_dir}")

    sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
    sensor_cfg.resolve(unwrapped.scene)
    tracker = StrideTracker(n, device, args_cli.min_stride_m, args_cli.max_stride_m)

    def step_policy():
        # Keep InferenceMode only around the JIT actor — env.reset/write_* need
        # mutable tensors and fail if prior steps ran under InferenceMode.
        obs = _build_run_obs(unwrapped)
        with torch.inference_mode():
            actions = actor.act(obs)
        env.step(actions.clone())

    def contacts_now() -> torch.Tensor:
        return foot_contact_mask(unwrapped, sensor_cfg=sensor_cfg)

    def vx_actual_now() -> torch.Tensor:
        return unwrapped.scene["robot"].data.root_lin_vel_b[:, 0]

    def vx_cmd_now() -> torch.Tensor:
        return unwrapped.command_manager.get_command("base_velocity")[:, 0]

    # ------------------------------------------------------------------
    # Experiment 1: steady stride length vs velocity command
    # ------------------------------------------------------------------
    print("[analyze] Experiment 1: stride length vs steady velocity commands")
    env.reset()
    tracker.reset_state()
    vx_list = _assign_cycle(STEADY_VX, n)
    vx_t = torch.tensor(vx_list, device=device, dtype=torch.float32)
    env_meta = [{"trial": "steady", "vx_target": float(v)} for v in vx_list]
    _set_vx(env, vx_t)

    for s in range(args_cli.warmup_steps):
        _set_vx(env, vx_t)
        step_policy()
        tracker.update(
            env, contacts_now(), s, "steady_warmup", vx_cmd_now(), vx_actual_now(), env_meta, record=False
        )

    for s in range(args_cli.steady_steps):
        _set_vx(env, vx_t)
        step_policy()
        tracker.update(
            env,
            contacts_now(),
            args_cli.warmup_steps + s,
            "steady",
            vx_cmd_now(),
            vx_actual_now(),
            env_meta,
            record=True,
        )

    steady_rows = [e for e in tracker.events if e["phase"] == "steady"]
    _write_csv(
        out_dir / "stride_vs_velocity.csv",
        steady_rows,
        [
            "phase",
            "step",
            "env_id",
            "foot",
            "vx_cmd",
            "vx_actual",
            "stride_length_m",
            "root_progress_m",
            "trial",
            "vx_target",
        ],
    )

    # Aggregate means for convenience
    agg_rows: list[dict] = []
    for vx in STEADY_VX:
        for foot in ("left", "right"):
            subset = [r for r in steady_rows if abs(r["vx_target"] - vx) < 1e-6 and r["foot"] == foot]
            if not subset:
                continue
            strides = np.array([r["stride_length_m"] for r in subset], dtype=np.float64)
            vxa = np.array([r["vx_actual"] for r in subset], dtype=np.float64)
            agg_rows.append(
                {
                    "vx_cmd": vx,
                    "foot": foot,
                    "n_strides": len(subset),
                    "stride_mean_m": float(strides.mean()),
                    "stride_std_m": float(strides.std(ddof=0)),
                    "stride_p25_m": float(np.percentile(strides, 25)),
                    "stride_p50_m": float(np.percentile(strides, 50)),
                    "stride_p75_m": float(np.percentile(strides, 75)),
                    "vx_actual_mean": float(vxa.mean()),
                    "vx_tracking_err_mean": float(np.mean(np.abs(vxa - vx))),
                }
            )
    _write_csv(
        out_dir / "stride_vs_velocity_summary.csv",
        agg_rows,
        [
            "vx_cmd",
            "foot",
            "n_strides",
            "stride_mean_m",
            "stride_std_m",
            "stride_p25_m",
            "stride_p50_m",
            "stride_p75_m",
            "vx_actual_mean",
            "vx_tracking_err_mean",
        ],
    )

    # ------------------------------------------------------------------
    # Experiment 2: leading foot from idle based on velocity command
    # ------------------------------------------------------------------
    print("[analyze] Experiment 2: leading foot selection from idle")
    env.reset()
    tracker.reset_state()
    go_list = _assign_cycle(IDLE_GO_VX, n)
    go_t = torch.tensor(go_list, device=device, dtype=torch.float32)
    lead_rows: list[dict] = []

    # Idle / standing
    _set_vx(env, 0.0)
    for s in range(args_cli.idle_steps):
        _set_vx(env, 0.0)
        step_policy()
        c = contacts_now()
        tracker.prev_contact = c.clone()  # arm edges after idle

    both_contact_at_go = contacts_now().all(dim=-1)
    first_lift = torch.full((n,), -1, dtype=torch.long, device=device)  # 0=left,1=right
    first_plant_after = torch.full((n,), -1, dtype=torch.long, device=device)
    lift_step = torch.full((n,), -1, dtype=torch.long, device=device)
    plant_step = torch.full((n,), -1, dtype=torch.long, device=device)
    first_stride = torch.full((n,), float("nan"), device=device)
    # Track first swing foot plant position for first-step length vs stance foot
    stance_xy = torch.zeros(n, 2, device=device)
    swing_plant_xy = torch.zeros(n, 2, device=device)
    asset0 = unwrapped.scene["robot"]
    left_id, right_id = resolve_ankle_body_ids(unwrapped)
    # Stance foot = the one that remains in contact when the other lifts
    prev_c = contacts_now().clone()

    for s in range(args_cli.go_steps):
        _set_vx(env, go_t)
        step_policy()
        c = contacts_now()
        fwd = heading_forward(asset0.data.root_quat_w)
        foot_xy = torch.stack(
            [asset0.data.body_pos_w[:, left_id, :2], asset0.data.body_pos_w[:, right_id, :2]],
            dim=1,
        )
        # First lift-off from contact
        lift = prev_c & (~c)
        for foot_i in (0, 1):
            ids = (lift[:, foot_i] & (first_lift < 0)).nonzero(as_tuple=False).squeeze(-1)
            if ids.numel() == 0:
                continue
            first_lift[ids] = foot_i
            lift_step[ids] = s
            other = 1 - foot_i
            stance_xy[ids] = foot_xy[ids, other]
        # First re-plant of the leading (swing) foot
        plant = c & (~prev_c)
        for foot_i in (0, 1):
            ids = (
                plant[:, foot_i]
                & (first_lift == foot_i)
                & (first_plant_after < 0)
            ).nonzero(as_tuple=False).squeeze(-1)
            if ids.numel() == 0:
                continue
            first_plant_after[ids] = foot_i
            plant_step[ids] = s
            swing_plant_xy[ids] = foot_xy[ids, foot_i]
            d = swing_plant_xy[ids] - stance_xy[ids]
            first_stride[ids] = (d * fwd[ids, :2]).sum(dim=-1)
        prev_c = c.clone()

        # Also collect subsequent strides tagged by go vx
        meta = [{"trial": "idle_go", "vx_target": float(v)} for v in go_list]
        tracker.update(
            env,
            c,
            s,
            "idle_go",
            vx_cmd_now(),
            vx_actual_now(),
            meta,
            record=(s > 10),
        )

    for i in range(n):
        lead = int(first_lift[i].item())
        lead_name = {0: "left", 1: "right", -1: "none"}[lead]
        plant = int(first_plant_after[i].item())
        plant_name = {0: "left", 1: "right", -1: "none"}[plant]
        lead_rows.append(
            {
                "env_id": i,
                "vx_cmd": float(go_list[i]),
                "both_contact_at_go": int(bool(both_contact_at_go[i].item())),
                "leading_foot": lead_name,
                "lift_step": int(lift_step[i].item()),
                "first_plant_foot": plant_name,
                "first_plant_step": int(plant_step[i].item()),
                "first_step_length_m": float(first_stride[i].item())
                if not torch.isnan(first_stride[i])
                else "",
                "vx_actual_at_end": float(vx_actual_now()[i].item()),
            }
        )

    _write_csv(
        out_dir / "leading_foot_from_idle.csv",
        lead_rows,
        [
            "env_id",
            "vx_cmd",
            "both_contact_at_go",
            "leading_foot",
            "lift_step",
            "first_plant_foot",
            "first_plant_step",
            "first_step_length_m",
            "vx_actual_at_end",
        ],
    )

    # Lead-foot summary by vx
    lead_sum: list[dict] = []
    for vx in IDLE_GO_VX:
        subset = [r for r in lead_rows if abs(r["vx_cmd"] - vx) < 1e-6]
        if not subset:
            continue
        n_tot = len(subset)
        n_l = sum(1 for r in subset if r["leading_foot"] == "left")
        n_r = sum(1 for r in subset if r["leading_foot"] == "right")
        n_none = sum(1 for r in subset if r["leading_foot"] == "none")
        steps = [
            float(r["first_step_length_m"])
            for r in subset
            if r["first_step_length_m"] != "" and r["first_step_length_m"] is not None
        ]
        lead_sum.append(
            {
                "vx_cmd": vx,
                "n_trials": n_tot,
                "lead_left": n_l,
                "lead_right": n_r,
                "lead_none": n_none,
                "lead_left_frac": n_l / max(n_tot, 1),
                "lead_right_frac": n_r / max(n_tot, 1),
                "first_step_mean_m": float(np.mean(steps)) if steps else "",
                "first_step_std_m": float(np.std(steps)) if steps else "",
            }
        )
    _write_csv(
        out_dir / "leading_foot_from_idle_summary.csv",
        lead_sum,
        [
            "vx_cmd",
            "n_trials",
            "lead_left",
            "lead_right",
            "lead_none",
            "lead_left_frac",
            "lead_right_frac",
            "first_step_mean_m",
            "first_step_std_m",
        ],
    )

    # ------------------------------------------------------------------
    # Experiment 3: changing velocity commands — tracking + stride
    # ------------------------------------------------------------------
    print("[analyze] Experiment 3: velocity tracking under changing commands")
    env.reset()
    tracker.reset_state()
    pairs = _assign_cycle(VX_CHANGES, n)
    vx_a = torch.tensor([p[0] for p in pairs], device=device, dtype=torch.float32)
    vx_b = torch.tensor([p[1] for p in pairs], device=device, dtype=torch.float32)
    meta = [
        {
            "trial": "vx_change",
            "vx_a": float(p[0]),
            "vx_b": float(p[1]),
            "delta_vx": float(p[1] - p[0]),
        }
        for p in pairs
    ]

    tracking_rows: list[dict] = []
    total_steps = args_cli.warmup_steps + args_cli.phase_a_steps + args_cli.phase_b_steps
    change_step = args_cli.warmup_steps + args_cli.phase_a_steps

    # subsample tracking log every k steps to keep CSV manageable
    log_every = 2

    for s in range(total_steps):
        if s < change_step:
            _set_vx(env, vx_a)
            phase = "warmup" if s < args_cli.warmup_steps else "pre_change"
        else:
            _set_vx(env, vx_b)
            phase = "post_change"
        step_policy()
        c = contacts_now()
        vxc = vx_cmd_now()
        vxa = vx_actual_now()
        record_stride = phase in ("pre_change", "post_change")
        tracker.update(env, c, s, phase, vxc, vxa, meta, record=record_stride)

        if s % log_every == 0 and phase in ("pre_change", "post_change"):
            t_rel = (s - change_step) * step_dt
            for i in range(n):
                tracking_rows.append(
                    {
                        "step": s,
                        "time_s": s * step_dt,
                        "time_from_change_s": t_rel,
                        "env_id": i,
                        "phase": phase,
                        "vx_a": pairs[i][0],
                        "vx_b": pairs[i][1],
                        "vx_cmd": float(vxc[i].item()),
                        "vx_actual": float(vxa[i].item()),
                        "vx_err": float((vxa[i] - vxc[i]).item()),
                        "abs_vx_err": float(abs((vxa[i] - vxc[i]).item())),
                    }
                )

    _write_csv(
        out_dir / "velocity_tracking_transients.csv",
        tracking_rows,
        [
            "step",
            "time_s",
            "time_from_change_s",
            "env_id",
            "phase",
            "vx_a",
            "vx_b",
            "vx_cmd",
            "vx_actual",
            "vx_err",
            "abs_vx_err",
        ],
    )

    stride_change_rows = [e for e in tracker.events if e["phase"] in ("pre_change", "post_change")]
    # Ensure meta keys present
    for e in stride_change_rows:
        e.setdefault("vx_a", "")
        e.setdefault("vx_b", "")
        e.setdefault("delta_vx", "")
        e.setdefault("trial", "vx_change")
    _write_csv(
        out_dir / "stride_under_velocity_change.csv",
        stride_change_rows,
        [
            "phase",
            "step",
            "env_id",
            "foot",
            "vx_cmd",
            "vx_actual",
            "stride_length_m",
            "root_progress_m",
            "trial",
            "vx_a",
            "vx_b",
            "delta_vx",
        ],
    )

    # Summary: mean abs tracking error pre/post + mean stride pre/post per pair
    track_sum: list[dict] = []
    unique_pairs = sorted(set(VX_CHANGES))
    for pa, pb in unique_pairs:
        pre = [
            r
            for r in tracking_rows
            if abs(r["vx_a"] - pa) < 1e-6 and abs(r["vx_b"] - pb) < 1e-6 and r["phase"] == "pre_change"
        ]
        post = [
            r
            for r in tracking_rows
            if abs(r["vx_a"] - pa) < 1e-6 and abs(r["vx_b"] - pb) < 1e-6 and r["phase"] == "post_change"
        ]
        # Settling: last 1.0s of post window
        post_settle = [r for r in post if r["time_from_change_s"] >= 1.0]
        pre_stride = [
            e
            for e in stride_change_rows
            if e["phase"] == "pre_change" and abs(float(e["vx_a"]) - pa) < 1e-6 and abs(float(e["vx_b"]) - pb) < 1e-6
        ]
        post_stride = [
            e
            for e in stride_change_rows
            if e["phase"] == "post_change" and abs(float(e["vx_a"]) - pa) < 1e-6 and abs(float(e["vx_b"]) - pb) < 1e-6
        ]

        def _mean_abs(rows):
            if not rows:
                return ""
            return float(np.mean([r["abs_vx_err"] for r in rows]))

        def _mean_stride(rows, foot):
            ss = [e["stride_length_m"] for e in rows if e["foot"] == foot]
            return float(np.mean(ss)) if ss else ""

        track_sum.append(
            {
                "vx_a": pa,
                "vx_b": pb,
                "delta_vx": pb - pa,
                "pre_abs_err_mean": _mean_abs(pre),
                "post_abs_err_mean": _mean_abs(post),
                "post_settle_abs_err_mean": _mean_abs(post_settle),
                "pre_stride_left_mean_m": _mean_stride(pre_stride, "left"),
                "pre_stride_right_mean_m": _mean_stride(pre_stride, "right"),
                "post_stride_left_mean_m": _mean_stride(post_stride, "left"),
                "post_stride_right_mean_m": _mean_stride(post_stride, "right"),
                "n_pre_track_samples": len(pre),
                "n_post_track_samples": len(post),
            }
        )

    _write_csv(
        out_dir / "velocity_tracking_summary.csv",
        track_sum,
        [
            "vx_a",
            "vx_b",
            "delta_vx",
            "pre_abs_err_mean",
            "post_abs_err_mean",
            "post_settle_abs_err_mean",
            "pre_stride_left_mean_m",
            "pre_stride_right_mean_m",
            "post_stride_left_mean_m",
            "post_stride_right_mean_m",
            "n_pre_track_samples",
            "n_post_track_samples",
        ],
    )

    # Manifest
    manifest = [
        {
            "policy": str(policy_path),
            "num_envs": n,
            "step_dt": step_dt,
            "seed": args_cli.seed,
            "steady_vx": str(STEADY_VX),
            "n_steady_strides": len(steady_rows),
            "n_lead_trials": len(lead_rows),
            "n_tracking_rows": len(tracking_rows),
            "n_change_strides": len(stride_change_rows),
        }
    ]
    _write_csv(
        out_dir / "analysis_manifest.csv",
        manifest,
        list(manifest[0].keys()),
    )

    env.close()
    print("[analyze] done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
