# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 run-jump obstacle course registrations (HL stub)."""

import gymnasium as gym

gym.register(
    id="Nepher-G1-RunJump-v0",
    entry_point="humanoid_run_jump.tasks.manager_based.common.amp_env:AmpManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.course_env_cfg:G1RunJumpCourseEnvCfg",
        "skrl_amp_cfg_entry_point": (
            "humanoid_run_jump.tasks.manager_based.run.agents:skrl_amp_cfg.yaml"
        ),
    },
)

gym.register(
    id="Nepher-G1-RunJump-Play-v0",
    entry_point="humanoid_run_jump.tasks.manager_based.common.amp_env:AmpManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.course_env_cfg:G1RunJumpCourseEnvCfg_PLAY",
        "skrl_amp_cfg_entry_point": (
            "humanoid_run_jump.tasks.manager_based.run.agents:skrl_amp_cfg.yaml"
        ),
    },
)
