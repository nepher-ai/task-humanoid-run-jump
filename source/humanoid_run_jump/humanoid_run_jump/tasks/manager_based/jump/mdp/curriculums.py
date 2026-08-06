# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for run / jump environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _progress(env: ManagerBasedRLEnv, steps_to_final: int) -> float:
    return min(1.0, float(env.common_step_counter) / float(max(steps_to_final, 1)))


def _success_gated_progress(
    env: ManagerBasedRLEnv,
    steps_to_final: int,
    unlock_progress: float = 0.35,
    relock_progress: float | None = None,
    min_landing_rate: float = 0.15,
    min_push_fold_rate: float | None = None,
    latch_attr: str = "jump_difficulty_unlocked",
    unlock_step_attr: str = "jump_difficulty_unlock_step",
    frozen_t_attr: str = "jump_difficulty_frozen_t",
) -> float:
    """Hold difficulty at 0 until hop skill unlocks, then ramp.

    Prefer ``jump_curriculum_progress`` (soft component mix). Fall back to
    ``flight_success_rate`` / ``jump_style_progress``.

    Unlock requires ``landing_success_rate >= min_landing_rate``. Fold rate is
    optional (pose style is AMP's job after reward simplification).

    **Two-way:** if success falls below ``relock_progress``, stop advancing and
    freeze at the last curriculum level (does not snap to easy; will not harden
    further while the hop skill is collapsing).
    """
    del min_push_fold_rate  # Kept for cfg backward-compat; no longer a gate.
    progress = float(
        getattr(
            env,
            "jump_curriculum_progress",
            getattr(env, "flight_success_rate", getattr(env, "jump_style_progress", 0.0)),
        )
    )
    landing_rate = float(getattr(env, "landing_success_rate", 0.0))
    relock = float(unlock_progress * 0.65 if relock_progress is None else relock_progress)
    unlocked = bool(getattr(env, latch_attr, False))
    frozen = float(getattr(env, frozen_t_attr, 0.0))

    gates_ok = landing_rate >= min_landing_rate
    if unlocked and (progress < relock or not gates_ok):
        setattr(env, latch_attr, False)
        return frozen

    if not unlocked:
        if progress < unlock_progress or not gates_ok:
            return frozen
        setattr(env, latch_attr, True)
        setattr(env, unlock_step_attr, int(env.common_step_counter))
        unlocked = True

    unlock_step = int(getattr(env, unlock_step_attr, env.common_step_counter))
    t = min(1.0, float(env.common_step_counter - unlock_step) / float(max(steps_to_final, 1)))
    # After a collapse/re-unlock, only allow curriculum to move forward.
    t = max(frozen, t)
    setattr(env, frozen_t_attr, t)
    return t


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def _lerp_range(
    initial: tuple[float, float],
    final: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    return (_lerp(initial[0], final[0], t), _lerp(initial[1], final[1], t))


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    initial_max: float = 1.0,
    final_max: float = 3.0,
    steps_to_final: int = 80000,
) -> float:
    """Linearly expand ``lin_vel_x`` upper bound over training steps."""
    del env_ids
    progress = _progress(env, steps_to_final)
    max_vx = _lerp(initial_max, final_max, progress)
    term = env.command_manager.get_term(command_name)
    lo = term.cfg.ranges.lin_vel_x[0]
    term.cfg.ranges.lin_vel_x = (lo, max_vx)
    return max_vx


def jump_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "jump",
    steps_to_final: int = 150000,
    unlock_progress: float = 0.35,
    relock_progress: float | None = None,
    min_landing_rate: float = 0.15,
    min_push_fold_rate: float | None = None,
    initial_h_obstacle: tuple[float, float] = (0.25, 0.50),
    final_h_obstacle: tuple[float, float] = (0.25, 0.75),
    initial_flight_distance: tuple[float, float] = (0.50, 0.90),
    final_flight_distance: tuple[float, float] = (0.50, 1.20),
    envelope_h_obstacle: tuple[float, float] | None = None,
    envelope_flight_distance: tuple[float, float] | None = None,
    # Legacy kwargs accepted but ignored (old obstacle-geometry curriculum).
    initial_d_obstacle: tuple[float, float] | None = None,
    final_d_obstacle: tuple[float, float] | None = None,
    initial_thickness: tuple[float, float] | None = None,
    final_thickness: tuple[float, float] | None = None,
    envelope_d_obstacle: tuple[float, float] | None = None,
    envelope_thickness: tuple[float, float] | None = None,
) -> float:
    """Expand ``(h_obstacle, flight_distance)`` after success EMA unlocks.

    Returns current ``h_obstacle`` upper bound for TensorBoard curriculum logging.
    """
    del env_ids
    del (
        initial_d_obstacle,
        final_d_obstacle,
        initial_thickness,
        final_thickness,
        envelope_d_obstacle,
        envelope_thickness,
    )
    t = _success_gated_progress(
        env,
        steps_to_final,
        unlock_progress=unlock_progress,
        relock_progress=relock_progress,
        min_landing_rate=min_landing_rate,
        min_push_fold_rate=min_push_fold_rate,
    )
    ranges = env.command_manager.get_term(command_name).cfg.ranges
    ranges.h_obstacle = _clamp_range(
        _lerp_range(initial_h_obstacle, final_h_obstacle, t), envelope_h_obstacle
    )
    ranges.flight_distance = _clamp_range(
        _lerp_range(initial_flight_distance, final_flight_distance, t),
        envelope_flight_distance,
    )
    return ranges.h_obstacle[1]


def _clamp_range(
    value: tuple[float, float],
    envelope: tuple[float, float] | None,
) -> tuple[float, float]:
    """Clamp a (lo, hi) range into ``envelope`` when provided."""
    if envelope is None:
        return value
    elo, ehi = envelope
    lo = max(min(value[0], ehi), elo)
    hi = max(min(value[1], ehi), elo)
    if lo > hi:
        lo, hi = elo, ehi
    return (lo, hi)
