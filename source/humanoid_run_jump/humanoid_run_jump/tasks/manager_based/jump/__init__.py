# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 jump task registrations."""

import gymnasium as gym

from . import agents

gym.register(
    id="Nepher-G1-Jump-v0",
    entry_point="humanoid_run_jump.tasks.manager_based.common.amp_env:AmpManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump_env_cfg:G1JumpEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
    },
)

gym.register(
    id="Nepher-G1-Jump-Play-v0",
    entry_point="humanoid_run_jump.tasks.manager_based.common.amp_env:AmpManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump_env_cfg:G1JumpEnvCfg_PLAY",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
    },
)
