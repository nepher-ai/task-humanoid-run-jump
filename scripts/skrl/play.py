# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained skrl AMP checkpoint for G1 run / jump.

If ``--checkpoint`` is omitted, the latest run under ``logs/skrl/<experiment>/``
is selected and its last checkpoint is loaded. After load, the AMP policy mean
network is exported as TorchScript next to the checkpoint
(``checkpoints/exported/policy.pt``).

Usage::

    isaaclab.bat -p scripts/skrl/play.py --task Nepher-G1-Run-Play-v0
    isaaclab.bat -p scripts/skrl/play.py --task Nepher-G1-Jump-Play-v0 \\
        --h 0.50 --flight 1.00
    isaaclab.bat -p scripts/skrl/play.py --task Nepher-G1-Jump-Play-v0 \\
        --num_envs 16 --video
    isaaclab.bat -p scripts/skrl/play.py --task Nepher-G1-Run-Play-v0 \\
        --checkpoint logs/skrl/g1_run_amp/<run>/checkpoints/best_agent.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a skrl AMP checkpoint.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument(
    "--video_length",
    type=int,
    default=None,
    help="Video length in env steps. Default: sized for the camera choreography "
    "(3s overview + 3×12s robot tracks).",
)
parser.add_argument(
    "--video_overview_s",
    type=float,
    default=3.0,
    help="With --video: seconds of high wide shot covering all envs before tracking.",
)
parser.add_argument(
    "--video_track_s",
    type=float,
    default=12.0,
    help="With --video: seconds to track each individual robot.",
)
parser.add_argument(
    "--video_num_robots",
    type=int,
    default=3,
    help="With --video: number of robots to film after the overview.",
)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Nepher-G1-Run-Play-v0")
parser.add_argument("--agent", type=str, default=None)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to model checkpoint. If omitted, load the last checkpoint from the latest run.",
)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"]
)
parser.add_argument(
    "--algorithm", type=str, default="AMP", choices=["AMP", "PPO", "IPPO", "MAPPO"]
)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument(
    "--export_jit",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Export TorchScript AMP policy next to the checkpoint (default: True).",
)
parser.add_argument(
    "--h",
    type=float,
    default=None,
    dest="h_obstacle",
    help="Jump play: fixed obstacle height h_obstacle (m). Overrides JUMP_PLAY_CMD. "
    "RunJumpHL play: fixed course obstacle height (m).",
)
parser.add_argument(
    "--flight",
    type=float,
    default=None,
    dest="flight_distance",
    help="Jump play: fixed flight_distance (m). Overrides JUMP_PLAY_CMD.",
)
parser.add_argument(
    "--num_obstacles",
    type=int,
    default=None,
    help="RunJumpHL play: number of obstacles on the course (1-3).",
)
parser.add_argument(
    "--obstacle_h",
    type=float,
    default=None,
    help="RunJumpHL play: fixed obstacle height in metres (alias of --h for HL).",
)
parser.add_argument(
    "--show_plant_target",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="RunJumpHL play: draw plant band / d* / footfalls. "
    "Default: on for Play, off with --video. Start/end lines stay on either way. "
    "Use --no-show_plant_target to force plant markers off.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
import time

import gymnasium as gym
import skrl
import torch
from packaging import version

SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(f"Unsupported skrl version: {skrl.__version__}")
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import humanoid_run_jump  # noqa: F401

# Sibling modules share this script's directory on the import path.
from export_amp_policy import export_skrl_amp_policy_as_jit
from play_video import (
    RealTimeVideoRecorder,
    StableRgbRenderWrapper,
    VideoCameraDirector,
)

if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = (
        "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
    )
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Optional fixed jump command for play (Nepher-G1-Jump-Play-v0).
    jump_cmd = getattr(getattr(env_cfg, "commands", None), "jump", None)
    if jump_cmd is not None and (
        args_cli.h_obstacle is not None or args_cli.flight_distance is not None
    ):
        # Only pin JumpCommand ranges for the pure Jump task; RunJumpHL sets
        # jump cmds from the HL action / obstacle, not from this range.
        if "RunJumpHL" not in args_cli.task:
            if args_cli.h_obstacle is not None:
                h = float(args_cli.h_obstacle)
                jump_cmd.ranges.h_obstacle = (h, h)
            if args_cli.flight_distance is not None:
                d = float(args_cli.flight_distance)
                jump_cmd.ranges.flight_distance = (d, d)
            print(
                "[INFO] Jump play command: "
                f"h_obstacle={jump_cmd.ranges.h_obstacle[0]:.3f} m, "
                f"flight_distance={jump_cmd.ranges.flight_distance[0]:.3f} m"
            )

    # Quieter visuals for documentary video.
    if args_cli.video and jump_cmd is not None and hasattr(jump_cmd, "debug_vis"):
        jump_cmd.debug_vis = False

    # RunJumpHL play: pin course obstacle count / height via event params.
    if "RunJumpHL" in args_cli.task:
        # Plant markers clutter recorded video; start/end lines stay on.
        if args_cli.show_plant_target is not None:
            env_cfg.show_plant_target = bool(args_cli.show_plant_target)
        elif args_cli.video:
            env_cfg.show_plant_target = False
        if hasattr(env_cfg, "show_course_lines"):
            env_cfg.show_course_lines = True
        course_event = getattr(getattr(env_cfg, "events", None), "randomize_course", None)
        if course_event is not None:
            hl_h = args_cli.obstacle_h if args_cli.obstacle_h is not None else args_cli.h_obstacle
            if args_cli.num_obstacles is not None:
                n = int(args_cli.num_obstacles)
                if n < 1 or n > 3:
                    raise SystemExit(f"--num_obstacles must be in [1, 3], got {n}")
                course_event.params["max_obstacles"] = n
            if hl_h is not None:
                h = float(hl_h)
                course_event.params["height_range"] = (h, h)
            print(
                "[INFO] RunJumpHL course: "
                f"max_obstacles={course_event.params.get('max_obstacles')}, "
                f"height_range={course_event.params.get('height_range')}, "
                f"show_plant_target={getattr(env_cfg, 'show_plant_target', False)}, "
                f"show_course_lines={getattr(env_cfg, 'show_course_lines', False)}"
            )

    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg.get("seed", 42)
    env_cfg.seed = agent_cfg["seed"]

    # Resolve checkpoint: explicit path, else last file in latest matching run.
    log_root_path = os.path.abspath(
        os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    )
    print(f"[INFO] Looking for checkpoints under: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path,
            run_dir=f".*_{algorithm}_{args_cli.ml_framework}",
            other_dirs=["checkpoints"],
            sort_alpha=True,
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    # Documentary video: world-space overview camera before env creation.
    if args_cli.video:
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.eye = (12.0, 12.0, 10.0)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    cam_director = None
    video_recorder = None
    video_length = args_cli.video_length
    if args_cli.video:
        base_env = env.unwrapped
        # Dense renders keep the viewport RGB annotator warm across camera cuts.
        base_env.cfg.sim.render_interval = 1

        if base_env.num_envs < int(args_cli.video_num_robots):
            print(
                f"[WARN] --video with num_envs={base_env.num_envs}: "
                f"only {base_env.num_envs} robot(s) can be filmed "
                f"(requested {args_cli.video_num_robots}). Prefer --num_envs 16."
            )

        cam_director = VideoCameraDirector.for_task(
            base_env,
            args_cli.task,
            overview_s=float(args_cli.video_overview_s),
            track_s=float(args_cli.video_track_s),
            num_robots=int(args_cli.video_num_robots),
        )
        needed = cam_director.total_steps
        if video_length is None:
            video_length = needed
        else:
            video_length = max(int(video_length), needed)

        env = StableRgbRenderWrapper(env)
        video_recorder = RealTimeVideoRecorder(
            env,
            video_folder=os.path.join(log_dir, "videos", "play"),
            fps=VideoCameraDirector.TARGET_VIDEO_FPS,
            step_dt=float(dt),
        )
        print(
            f"[INFO] --video choreography ({args_cli.task}): "
            f"{args_cli.video_overview_s:.1f}s high overview of all envs → "
            f"{cam_director.num_robots} robots × {args_cli.video_track_s:.1f}s "
            f"(cut to nearest). {video_length} steps ≈ {cam_director.total_duration_s:.1f}s "
            f"@ {VideoCameraDirector.TARGET_VIDEO_FPS} fps real-time."
        )

    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    runner = Runner(env, agent_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    if args_cli.export_jit and args_cli.ml_framework.startswith("torch"):
        export_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_skrl_amp_policy_as_jit(runner.agent, export_dir, filename="policy.pt")

    obs, _ = env.reset()
    if cam_director is not None:
        cam_director.warm_start()

    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, _, _, _ = env.step(actions)

        if cam_director is not None:
            # Pose for this frame, then settle, then capture (not RecordVideo).
            cam_director.update(timestep)
            for _ in range(3):
                env.unwrapped.sim.render()
        if video_recorder is not None:
            video_recorder.capture()
            timestep += 1
            if timestep >= video_length:
                break

        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    if video_recorder is not None:
        video_recorder.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
