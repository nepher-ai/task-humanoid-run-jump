# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train G1 run / jump policies with skrl AMP.

Usage (from project root)::

    isaaclab.bat -p scripts/skrl/train.py --task Nepher-G1-Run-v0 --headless
    isaaclab.bat -p scripts/skrl/train.py --task Nepher-G1-Jump-v0 --headless
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an RL agent with skrl AMP.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Nepher-G1-Run-v0")
parser.add_argument("--agent", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--export_io_descriptors", action="store_true", default=False)
parser.add_argument(
    "--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"]
)
parser.add_argument(
    "--algorithm", type=str, default="AMP", choices=["AMP", "PPO", "IPPO", "MAPPO"]
)
parser.add_argument(
    "--show_plant_target",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="RunJumpHL: draw plant band / d* / current-vs-required footfalls. "
    "Default: on when not --headless, off under --headless. "
    "Use --no-show_plant_target to force off.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import os
import random
import time
from datetime import datetime

import gymnasium as gym
import skrl
from packaging import version

SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. Install skrl>={SKRL_VERSION}"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner

    # skrl Runner only resolves agent class "AMP"; swap in LoggingAMP so
    # task / style / total rewards appear in TensorBoard.
    from humanoid_run_jump.agents.logging_amp import LoggingAMP

    _runner_component = Runner._component

    def _component_with_logging_amp(self, name: str):
        if name.lower() == "amp":
            return LoggingAMP
        return _runner_component(self, name)

    Runner._component = _component_with_logging_amp
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
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import humanoid_run_jump  # noqa: F401

logger = logging.getLogger(__name__)

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

    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError("Distributed training is not supported on CPU.")

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    env_cfg.log_dir = log_dir

    # RunJumpHL: plant / footfall markers only make sense with a viewport.
    if "RunJumpHL" in args_cli.task and hasattr(env_cfg, "show_plant_target"):
        if args_cli.show_plant_target is not None:
            env_cfg.show_plant_target = bool(args_cli.show_plant_target)
        else:
            env_cfg.show_plant_target = not bool(getattr(args_cli, "headless", False))
        if hasattr(env_cfg, "show_course_lines"):
            env_cfg.show_course_lines = env_cfg.show_plant_target
        print(
            f"[INFO] RunJumpHL show_plant_target={env_cfg.show_plant_target}, "
            f"show_course_lines={getattr(env_cfg, 'show_course_lines', False)}"
        )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    runner = Runner(env, agent_cfg)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
    runner.run()
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
