# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a run→jump hand-off state bank from the frozen run actor.

Rolls ``Nepher-G1-Run-Play-v0`` with ``best_policy/run/policy.pt``, snapshots
root + joint state when the configured gait-phase trigger fires, and writes
``motions/packaged/runin_states.pt``.

Each sample is collected under a randomly commanded forward speed drawn
uniformly from ``[--vx-min, --vx-max]`` (default 0.5–3.0 m/s).

Usage (from project root)::

    isaaclab.bat -p scripts/build_runin_bank.py --headless --num_envs 64 --num_samples 4096
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Build run-in hand-off state bank.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_samples", type=int, default=4096)
parser.add_argument("--max_steps", type=int, default=20000)
parser.add_argument("--warmup_steps", type=int, default=50)
parser.add_argument(
    "--vx-min",
    type=float,
    default=0.5,
    help="Minimum forward velocity command (m/s). Default: 0.5.",
)
parser.add_argument(
    "--vx-max",
    type=float,
    default=3.0,
    help="Maximum forward velocity command (m/s). Default: 3.0.",
)
parser.add_argument(
    "--vx",
    type=float,
    default=None,
    help="If set, fix forward speed to this value (overrides --vx-min/--vx-max).",
)
parser.add_argument(
    "--lead",
    type=str,
    default="right",
    choices=["left", "right"],
    help="Lead foot for hand-off trigger (default: right).",
)
parser.add_argument(
    "--policy",
    type=str,
    default=None,
    help="Path to exported run actor. Default: best_policy/run/policy.pt",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output .pt path. Default: motions/packaged/runin_states.pt",
)
parser.add_argument("--min_lead_m", type=float, default=0.05)
parser.add_argument("--require_trail_air", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg

import humanoid_run_jump  # noqa: F401
from humanoid_run_jump.agents.frozen_actor import DEFAULT_RUN_POLICY_PATH, FrozenActor
from humanoid_run_jump.tasks.manager_based.run.mdp.gait import runin_handoff_mask
from humanoid_run_jump.tasks.manager_based.run.mdp.observations import (
    base_lin_vel,
    last_action,
    reduced_coords_proprio,
    velocity_commands,
)
from humanoid_run_jump.tracker.reduced_coords import TARGET_FRAME_DIM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "motions" / "packaged" / "runin_states.pt"
# Run policy obs: proprio(64) + base_lin_vel(3) + velocity_commands(3) + last_action(64)
RUN_OBS_DIM = 64 + 3 + 3 + 64


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


def main():
    policy_path = Path(args_cli.policy) if args_cli.policy else DEFAULT_RUN_POLICY_PATH
    out_path = Path(args_cli.output) if args_cli.output else DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args_cli.vx is not None:
        vx_lo = vx_hi = float(args_cli.vx)
    else:
        vx_lo = float(args_cli.vx_min)
        vx_hi = float(args_cli.vx_max)
    if vx_lo > vx_hi:
        raise ValueError(f"vx-min ({vx_lo}) must be <= vx-max ({vx_hi})")

    # Import after AppLauncher so USD / sim are ready.
    from isaaclab.utils import configclass
    from humanoid_run_jump.tasks.manager_based.run.run_env_cfg import G1RunEnvCfg_PLAY, RunSceneCfg

    @configclass
    class RunInSceneCfg(RunSceneCfg):
        contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/pelvis/.*",
            history_length=3,
            track_air_time=True,
        )

    @configclass
    class RunInEnvCfg(G1RunEnvCfg_PLAY):
        scene: RunInSceneCfg = RunInSceneCfg(num_envs=64, env_spacing=3.0)

        def __post_init__(self):
            super().__post_init__()
            self.scene.robot.spawn.activate_contact_sensors = True
            if self.scene.contact_forces is not None:
                self.scene.contact_forces.update_period = self.sim.dt

    env_cfg = RunInEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    # Random forward speed per env / resample; lateral & yaw held at 0.
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (vx_lo, vx_hi)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    # Resample often so collected hand-offs span the speed band.
    env_cfg.commands.base_velocity.resampling_time_range = (2.0, 4.0)
    env_cfg.curriculum.lin_vel_cmd = None

    env = gym.make("Nepher-G1-Run-Play-v0", cfg=env_cfg)
    unwrapped = env.unwrapped
    device = unwrapped.device

    actor = FrozenActor(
        policy_path=policy_path,
        device=device,
        expected_obs_dim=RUN_OBS_DIM,
        expected_act_dim=TARGET_FRAME_DIM,
    )
    print(f"[runin] loaded actor from {actor.policy_path} (obs={actor.obs_dim}, act={actor.act_dim})")
    print(f"[runin] velocity command range: lin_vel_x=[{vx_lo:.2f}, {vx_hi:.2f}] m/s")

    sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
    sensor_cfg.resolve(unwrapped.scene)
    vel_cmd_term = unwrapped.command_manager.get_term("base_velocity")

    # Accumulators (G1 joint order for dof via joint_order_map).
    root_pose_buf: list[torch.Tensor] = []
    root_vel_buf: list[torch.Tensor] = []
    dof_pos_buf: list[torch.Tensor] = []
    dof_vel_buf: list[torch.Tensor] = []
    vx_cmd_buf: list[torch.Tensor] = []
    # Per-env cooldown so we do not re-sample every step of the same plant.
    cooldown = torch.zeros(unwrapped.num_envs, dtype=torch.long, device=device)
    cooldown_steps = 25

    obs, _ = env.reset()
    collected = 0
    step = 0
    print(
        f"[runin] collecting {args_cli.num_samples} samples "
        f"(lead={args_cli.lead}, vx=[{vx_lo:.2f},{vx_hi:.2f}])..."
    )

    while collected < args_cli.num_samples and step < args_cli.max_steps and simulation_app.is_running():
        with torch.inference_mode():
            run_obs = _build_run_obs(unwrapped)
            actions = actor.act(run_obs)
            obs, _, terminated, truncated, _ = env.step(actions)

        step += 1
        cooldown = torch.clamp(cooldown - 1, min=0)
        if step < args_cli.warmup_steps:
            continue

        trigger = runin_handoff_mask(
            unwrapped,
            lead=args_cli.lead,
            sensor_cfg=sensor_cfg,
            min_lead_m=args_cli.min_lead_m,
            require_trail_air=args_cli.require_trail_air,
        )
        eligible = trigger & (cooldown == 0)
        # Skip freshly-reset envs (unstable).
        ep_len = getattr(unwrapped, "episode_length_buf", None)
        if ep_len is not None:
            eligible = eligible & (ep_len > args_cli.warmup_steps)

        ids = eligible.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            continue

        asset = unwrapped.scene["robot"]
        jmap = unwrapped.joint_order_map
        # Canonicalize to env-local origin with +x heading. The run actor may
        # have drifted tens of meters / arbitrary yaw by hand-off time; storing
        # that drift makes play/train spawns scatter. Reset still re-anchors
        # idempotently, so older banks remain usable.
        root_pos = asset.data.root_pos_w[ids].clone()
        root_pos[:, :2] = 0.0
        root_quat = asset.data.root_quat_w[ids].clone()
        w0, x0, y0, z0 = root_quat.unbind(-1)
        bank_yaw = torch.atan2(2.0 * (w0 * z0 + x0 * y0), 1.0 - 2.0 * (y0 * y0 + z0 * z0))
        half = (-bank_yaw) * 0.5
        qw = torch.cos(half)
        qz = torch.sin(half)
        root_quat = torch.stack(
            [
                qw * w0 - qz * z0,
                qw * x0 + qz * y0,
                qw * y0 - qz * x0,
                qw * z0 + qz * w0,
            ],
            dim=-1,
        )
        root_pose = torch.cat([root_pos, root_quat], dim=-1)
        root_lin = asset.data.root_lin_vel_w[ids].clone()
        root_ang = asset.data.root_ang_vel_w[ids].clone()
        c = torch.cos(-bank_yaw)
        s = torch.sin(-bank_yaw)
        vx, vy = root_lin[:, 0].clone(), root_lin[:, 1].clone()
        root_lin[:, 0] = c * vx - s * vy
        root_lin[:, 1] = s * vx + c * vy
        wx, wy = root_ang[:, 0].clone(), root_ang[:, 1].clone()
        root_ang[:, 0] = c * wx - s * wy
        root_ang[:, 1] = s * wx + c * wy
        root_vel = torch.cat([root_lin, root_ang], dim=-1)
        dof_pos = jmap.to_g1(asset.data.joint_pos[ids].clone())
        dof_vel = jmap.to_g1(asset.data.joint_vel[ids].clone())
        # Commanded forward speed at the hand-off instant (per sample).
        vx_cmd = unwrapped.command_manager.get_command("base_velocity")[ids, 0].clone()

        # Cap this batch so we do not overshoot.
        remain = args_cli.num_samples - collected
        take = min(int(ids.numel()), remain)
        taken_ids = ids[:take]
        root_pose_buf.append(root_pose[:take].cpu())
        root_vel_buf.append(root_vel[:take].cpu())
        dof_pos_buf.append(dof_pos[:take].cpu())
        dof_vel_buf.append(dof_vel[:take].cpu())
        vx_cmd_buf.append(vx_cmd[:take].cpu())
        cooldown[taken_ids] = cooldown_steps
        collected += take
        if collected % 256 < take or collected >= args_cli.num_samples:
            print(f"[runin] collected {collected}/{args_cli.num_samples} (step={step})")

        # Resample a new random speed for sampled envs so the next hand-off
        # from that env draws a different vx in [vx_lo, vx_hi].
        vel_cmd_term._resample(taken_ids)

    env.close()

    if collected == 0:
        raise RuntimeError(
            "No hand-off samples collected. Check lead foot, contact sensors, and run policy."
        )

    vx_cmd_all = torch.cat(vx_cmd_buf, dim=0)[: args_cli.num_samples]
    payload = {
        "root_pose": torch.cat(root_pose_buf, dim=0)[: args_cli.num_samples],
        "root_vel": torch.cat(root_vel_buf, dim=0)[: args_cli.num_samples],
        "dof_pos": torch.cat(dof_pos_buf, dim=0)[: args_cli.num_samples],
        "dof_vel": torch.cat(dof_vel_buf, dim=0)[: args_cli.num_samples],
        "vx_cmd": vx_cmd_all,
        "lead": args_cli.lead,
        "vx_range": (vx_lo, vx_hi),
        "policy_path": str(policy_path),
        "num_samples": int(min(collected, args_cli.num_samples)),
    }
    torch.save(payload, out_path)
    print(
        f"[runin] wrote {out_path} with {payload['num_samples']} samples "
        f"(lead={args_cli.lead}, vx_cmd=[{float(vx_cmd_all.min()):.2f},"
        f"{float(vx_cmd_all.max()):.2f}] mean={float(vx_cmd_all.mean()):.2f})"
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
