# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 RunJumpHL task registrations (PPO high-level over frozen LL models)."""

import gymnasium as gym

from . import agents

gym.register(
    id="Nepher-G1-RunJumpHL-v0",
    entry_point="humanoid_run_jump.tasks.manager_based.run_jump.hl_course_env:HlCourseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hl_env_cfg:G1RunJumpHLEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg.yaml",
    },
)

gym.register(
    id="Nepher-G1-RunJumpHL-Play-v0",
    entry_point="humanoid_run_jump.tasks.manager_based.run_jump.hl_course_env:HlCourseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hl_env_cfg:G1RunJumpHLEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg.yaml",
    },
)
