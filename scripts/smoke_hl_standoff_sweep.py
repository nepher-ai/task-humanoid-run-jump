# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke-test / 1-D standoff sweep for Nepher-G1-RunJumpHL-Play-v0.

Forces RUN→JUMP latches at a grid of right-foot standoffs against a fixed
obstacle and reports clear rate. Use to cross-check the plant band / PK_X grid.

Usage (from ``task-humanoid-run-jump/``)::

    ..\\..\\isaaclab.bat -p scripts/smoke_hl_standoff_sweep.py --headless --num_envs 16
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="HL standoff sweep / smoke test.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--obstacle_h", type=float, default=0.50)
parser.add_argument("--obstacle_x", type=float, default=8.0)
parser.add_argument("--vx", type=float, default=2.4, help="Commanded forward speed (m/s).")
parser.add_argument(
    "--standoffs",
    type=str,
    default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    help="Comma-separated right-foot standoffs (m) to force-latch at.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import humanoid_run_jump  # noqa: F401
from humanoid_run_jump.tasks.manager_based.jump.mdp.gait import resolve_ankle_body_ids
from humanoid_run_jump.tasks.manager_based.run_jump.hl_env_cfg import G1RunJumpHLEnvCfg_PLAY
from humanoid_run_jump.tasks.manager_based.run_jump.mdp.hl_stride import (
    PLANT_BAND_MAX,
    PLANT_BAND_MIN,
    apex_from_plant,
    apex_trim,
)


def _pin_course(unwrapped, ids: torch.Tensor, obstacle_x: float, obstacle_h: float) -> None:
    device = unwrapped.device
    n = len(ids)
    unwrapped.path_length[ids] = 25.0
    unwrapped.obstacle_x[ids] = 0.0
    unwrapped.obstacle_h[ids] = 0.0
    unwrapped.obstacle_t[ids] = 0.20
    unwrapped.obstacle_active[ids] = False
    unwrapped.num_obstacles[ids] = 1
    unwrapped.obstacle_index[ids] = 0
    unwrapped.obstacle_x[ids, 0] = obstacle_x
    unwrapped.obstacle_h[ids, 0] = obstacle_h
    unwrapped.obstacle_t[ids, 0] = 0.20
    unwrapped.obstacle_active[ids, 0] = True
    unwrapped.course_finished[ids] = False
    unwrapped.cleared_count[ids] = 0
    unwrapped.hit_count[ids] = 0
    unwrapped.crash_hold[ids] = 0

    origins = unwrapped.scene.env_origins[ids]
    t = 0.20
    pos = origins.clone()
    pos[:, 0] = origins[:, 0] + obstacle_x + 0.5 * t
    pos[:, 1] = origins[:, 1]
    pos[:, 2] = obstacle_h - 0.75
    quat = torch.zeros(n, 4, device=device)
    quat[:, 0] = 1.0
    unwrapped.scene["obstacle_0"].write_root_pose_to_sim(
        torch.cat([pos, quat], dim=-1), env_ids=ids
    )
    unwrapped.scene["obstacle_0"].write_root_velocity_to_sim(
        torch.zeros(n, 6, device=device), env_ids=ids
    )
    for k in (1, 2):
        name = f"obstacle_{k}"
        if name in unwrapped.scene.keys():
            p = origins.clone()
            p[:, 2] = -5.0
            unwrapped.scene[name].write_root_pose_to_sim(
                torch.cat([p, quat], dim=-1), env_ids=ids
            )
    unwrapped.refresh_course_features(ids)


def main():
    print("[smoke] starting…", flush=True)
    standoffs = [float(x) for x in args_cli.standoffs.split(",") if x.strip()]
    cfg = G1RunJumpHLEnvCfg_PLAY()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = args_cli.seed
    # Read the gate every control step so a rising edge can coincide with a
    # right plant; at action_repeat=5 the approach window is shorter than one
    # decision and the sweep never latches.
    cfg.actions.hl_switch.action_repeat = 1
    print(f"[smoke] making env num_envs={cfg.scene.num_envs}", flush=True)
    try:
        env = gym.make("Nepher-G1-RunJumpHL-Play-v0", cfg=cfg)
    except Exception as exc:
        import traceback

        print("[smoke] gym.make FAILED:", exc, flush=True)
        traceback.print_exc()
        raise
    print("[smoke] env created", flush=True)
    unwrapped = env.unwrapped
    device = unwrapped.device
    n = unwrapped.num_envs
    origins = unwrapped.scene.env_origins

    print(
        f"[smoke] envs={n} plant_band=[{PLANT_BAND_MIN},{PLANT_BAND_MAX}] vx={args_cli.vx}",
        flush=True,
    )
    print(
        f"[smoke] standoffs={standoffs} obstacle_h={args_cli.obstacle_h} x={args_cli.obstacle_x}",
        flush=True,
    )

    obs, _ = env.reset()
    ids = torch.arange(n, device=device)
    _pin_course(unwrapped, ids, args_cli.obstacle_x, args_cli.obstacle_h)

    targets = torch.tensor([standoffs[i % len(standoffs)] for i in range(n)], device=device)
    latched = torch.zeros(n, dtype=torch.bool, device=device)
    results = {s: {"n": 0, "cleared": 0, "hit": 0} for s in standoffs}

    _, right_id = resolve_ankle_body_ids(unwrapped)
    switch = getattr(unwrapped, "hl_switch_action", None)

    # Invert the tanh → [vx_min, vx_max] mapping so --vx is a physical speed.
    acfg = unwrapped.cfg.actions.hl_switch
    t_vx = 2.0 * (args_cli.vx - acfg.vx_min) / (acfg.vx_max - acfg.vx_min) - 1.0
    raw_vx = float(torch.atanh(torch.tensor(t_vx).clamp(-0.999, 0.999)))
    print(f"[smoke] vx_cmd={args_cli.vx:.2f} m/s (raw action {raw_vx:+.3f})", flush=True)

    term_counts: dict[str, int] = {}
    apex_err_by_d = {s: [] for s in standoffs}
    total_latches = 0
    # Lowest sole height reached while either ankle is inside the obstacle slab.
    # This, not the apex x-placement, is what decides whether the box is cleared.
    min_clear = torch.full((n,), 99.0, device=device)
    clearances: list[float] = []
    peak_heights: list[float] = []
    max_s = torch.zeros(n, device=device)
    reach: list[float] = []
    past_box = 0
    left_id, _ = resolve_ankle_body_ids(unwrapped)
    ankle_to_sole = float(getattr(unwrapped.cfg, "ankle_to_sole", 0.05))

    for step in range(args_cli.steps):
        actions = torch.zeros(n, 6, device=device)
        actions[:, 1] = raw_vx
        right_s = unwrapped.scene["robot"].data.body_pos_w[:, right_id, 0] - origins[:, 0]
        d_right = unwrapped.next_front_s - right_s
        # Pulse the gate across a wider window: the latch also needs a right
        # plant, so a single rising edge at an exact standoff almost never fires.
        near = (~latched) & ((d_right - targets).abs() < 0.30)
        actions[near, 0] = 3.0 if step % 2 == 0 else -3.0

        obs, rew, term, trunc, info = env.step(actions)

        if switch is not None and switch._just_latched.any():
            latched[switch._just_latched] = True
            total_latches += int(switch._just_latched.sum())

        # Worst sole clearance over the box footprint, counted only while
        # airborne — otherwise the stance foot beside the box dominates.
        ankle_pos = unwrapped.scene["robot"].data.body_pos_w[:, [left_id, right_id], :]
        ankle_s = ankle_pos[:, :, 0] - origins[:, 0:1]
        sole_z = ankle_pos[:, :, 2] - ankle_to_sole
        airborne = (unwrapped.hl_mode == 1) & unwrapped._ep_has_liftoff
        over_box = (ankle_s - unwrapped.apex_target_s.unsqueeze(-1)).abs() < 0.10
        over_box &= airborne.unsqueeze(-1)
        gap = torch.where(over_box, sole_z - args_cli.obstacle_h, torch.full_like(sole_z, 99.0))
        min_clear = torch.minimum(min_clear, gap.min(dim=-1).values)

        # Measured apex placement, published by the env once each jump completes.
        if unwrapped.apex_event.any():
            for eid in unwrapped.apex_event.nonzero(as_tuple=False).squeeze(-1).tolist():
                key = min(standoffs, key=lambda x: abs(x - float(targets[eid].item())))
                apex_err_by_d[key].append(
                    float(unwrapped.apex_peak_s[eid] - unwrapped.apex_target_s[eid])
                )
                peak_heights.append(float(unwrapped.apex_peak_z[eid]) - ankle_to_sole)
                if float(min_clear[eid]) < 90.0:
                    clearances.append(float(min_clear[eid]))
                min_clear[eid] = 99.0

        root_s_now = unwrapped.scene["robot"].data.root_pos_w[:, 0] - origins[:, 0]
        max_s = torch.maximum(max_s, root_s_now)

        done = term | trunc
        if done.any():
            dids = done.nonzero(as_tuple=False).squeeze(-1)
            for name in unwrapped.termination_manager.active_terms:
                fired = unwrapped.termination_manager.get_term(name)[dids]
                if fired.any():
                    term_counts[name] = term_counts.get(name, 0) + int(fired.sum())
            crash = unwrapped.termination_manager.get_term("obstacle_crash")
            for eid in dids.tolist():
                s = float(targets[eid].item())
                key = min(standoffs, key=lambda x: abs(x - s))
                results[key]["n"] += 1
                reach_m = float(max_s[eid])
                reach.append(reach_m)
                # Isaac Lab resets done envs before returning from step(), so
                # cleared_count / hit_count are already zero. Use reach as the
                # clear proxy (same threshold as step_course_state: past back face).
                if reach_m > args_cli.obstacle_x + 0.25:
                    past_box += 1
                    results[key]["cleared"] += 1
                if bool(crash[eid]):
                    results[key]["hit"] += 1
                latched[eid] = False
                min_clear[eid] = 99.0
                max_s[eid] = 0.0
            _pin_course(unwrapped, dids, args_cli.obstacle_x, args_cli.obstacle_h)

        if step % 100 == 0:
            print(
                f"[smoke] step={step} latched={int(latched.sum())} "
                f"mode_jump={int((unwrapped.hl_mode == 1).sum())}"
            )

    print(f"\n[smoke] total latches={total_latches}")
    print("\n=== Standoff sweep clear rates ===")
    print(f"{'d':>6} {'n':>4} {'clear':>6} {'hit':>4} {'rate':>6} {'apex_err':>9}  notes")
    for s in standoffs:
        r = results[s]
        rate = r["cleared"] / max(r["n"], 1)
        vx_t = torch.tensor([args_cli.vx])
        f = float(
            apex_trim(
                torch.tensor([s]),
                torch.tensor([0.20]),
                torch.tensor([args_cli.obstacle_h]),
                vx_t,
            )[0]
        )
        p = float(apex_from_plant(torch.tensor([f]), vx_t)[0])
        target = s + 0.10  # plant + t/2 → expected root over obstacle centre
        errs = apex_err_by_d[s]
        err_txt = f"{sum(errs) / len(errs):+9.3f}" if errs else f"{'--':>9}"
        print(
            f"{s:6.2f} {r['n']:4d} {r['cleared']:6d} {r['hit']:4d} {rate:6.2f} {err_txt}  "
            f"P={p:.3f} target={target:.3f} f*={f:.3f}"
        )

    all_errs = [e for v in apex_err_by_d.values() for e in v]
    if all_errs:
        mean_err = sum(all_errs) / len(all_errs)
        print(
            f"\n[smoke] measured apex offset (peak_root_s - obstacle_centre): "
            f"mean={mean_err:+.3f} m over {len(all_errs)} jumps"
        )
        print(f"[smoke] plant band = [{PLANT_BAND_MIN}, {PLANT_BAND_MAX}] m")
    if peak_heights:
        print(
            f"[smoke] peak sole height: mean={sum(peak_heights) / len(peak_heights):.3f} m "
            f"(obstacle {args_cli.obstacle_h:.2f} m)"
        )
    if clearances:
        n_clip = sum(1 for c in clearances if c < 0.0)
        print(
            f"[smoke] sole clearance over the box: mean={sum(clearances) / len(clearances):+.3f} m, "
            f"min={min(clearances):+.3f} m, clipped {n_clip}/{len(clearances)}"
        )

    if reach:
        reach_sorted = sorted(reach)
        print(
            f"[smoke] episode reach (m): mean={sum(reach) / len(reach):.2f} "
            f"median={reach_sorted[len(reach_sorted) // 2]:.2f} max={reach_sorted[-1]:.2f}; "
            f"{past_box}/{len(reach)} got past the box at {args_cli.obstacle_x:.1f} m"
        )

    print("\n=== Termination counts ===")
    for name, count in sorted(term_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:22s} {count}")

    y = unwrapped.scene["robot"].data.root_pos_w[:, 1] - unwrapped.scene.env_origins[:, 1]
    print(f"\n[smoke] |y| max={float(y.abs().max()):.3f} (out_of_path at 2.5)")
    print("[smoke] done")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
