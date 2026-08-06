# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 run task registrations."""

import gymnasium as gym

from . import agents

gym.register(
    id="Nepher-G1-Run-v0",
    entry_point="humanoid_run_jump.tasks.manager_based.run.run_amp_env:RunAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.run_env_cfg:G1RunEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
    },
)

gym.register(
    id="Nepher-G1-Run-Play-v0",
    entry_point="humanoid_run_jump.tasks.manager_based.run.run_amp_env:RunAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.run_env_cfg:G1RunEnvCfg_PLAY",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
    },
)
