# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Humanoid G1 run-jump hierarchical RL package."""

from . import tasks  # noqa: F401

# Expose evaluation wrapper so eval-nav EnvironmentManager can find it via
# task_module="humanoid_run_jump".
from .tasks.manager_based.run_jump.eval_compat import wrap_for_eval  # noqa: F401
