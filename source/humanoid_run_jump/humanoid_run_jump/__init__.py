# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Humanoid G1 run-jump hierarchical RL package."""

from __future__ import annotations

from typing import Any

# Gym env registration (no Isaac Lab import at package load).
from . import tasks  # noqa: F401


def __getattr__(name: str) -> Any:
    """Lazy exports that pull Isaac Lab (used by eval-nav ``task_module``)."""
    if name == "wrap_for_eval":
        from .tasks.manager_based.run_jump.eval_compat import wrap_for_eval

        return wrap_for_eval
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["tasks", "wrap_for_eval"]
