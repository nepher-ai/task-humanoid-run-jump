# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CSV-fitted jump takeoff envelope for the frozen BeyondMimic tracker.

Jump CSVs are named ``jump_over_obstacle_0_5m_*`` / ``0_75m_*``. Motions are
already retargeted to G1. FK / filename tags show peak lower-foot height ≈ the
obstacle tag via a **hurdle step** (push-leg fold + swing-leg extension) —
root/pelvis only rises ~0.21–0.36 m — so treat ``0.50`` / ``0.75`` as literal
obstacle heights in the robot / motion frame.

Policy-facing jump command is ``(h_obstacle, flight_distance)``. Takeoff
``(v_x*, v_z*)`` bins remain internal shaping references from those CSV clips.

All per-leg / arm / rise anchors below are medians over the 15 non-mirrored
jump clips (MuJoCo FK). Interpolation spans the full ``[H_MIN, H_MAX]`` band
so shaping targets are not frozen for the early curriculum.
"""

from __future__ import annotations

import torch

# Tracker-safe obstacle height band (meters in the G1 motion frame).
H_MIN = 0.25
H_MAX = 0.75  # matches highest CSV tag (+ no headroom beyond demonstrated clips)

# Anchor bins: literal CSV filename heights (lerp endpoints).
_H_LO = 0.50
_H_HI = 0.75

# Body-frame takeoff means from measured CSV bins (liftoff vx / vz medians).
_VX_LO, _VX_HI = 1.85, 1.95
_VZ_LO, _VZ_HI = 1.72, 2.05
_CROUCH_LO, _CROUCH_HI = 0.26, 0.34

# Demonstrated heading-aligned flight distance band (liftoff → landing, m).
FLIGHT_DIST_MIN = 0.50
FLIGHT_DIST_MAX = 1.20

# Per-bin measured flight caps used by :func:`flight_cap`.
_FLIGHT_CAP_LO = 1.20  # at h <= 0.50 m
_FLIGHT_CAP_HI = 1.00  # at h = 0.75 m

# Legacy standoff helpers (kept for course / HL stubs that still place boxes).
_STANDOFF_LO = 0.45  # at 0.5 m bin
_STANDOFF_HI = 0.55  # at 0.75 m bin

THICKNESS_MIN = 0.05
THICKNESS_MAX = 0.25

# Legacy approach distance to obstacle front face (meters).
D_OBSTACLE_MIN = 1.2
D_OBSTACLE_MAX = 3.5

# --- Hurdle-apex success envelope (MuJoCo FK over non-_M jump CSVs) ---
# Pelvis rise liftoff→apex: measured med 0.244 (0.5 m) / 0.303 (0.75 m).
_APEX_RISE_LO, _APEX_RISE_HI = 0.22, 0.31
# Push-leg (takeoff foot) max pelvis→ankle at apex (m). Measured med 0.332 / 0.307.
_PUSH_FOLD_LO, _PUSH_FOLD_HI = 0.34, 0.30
# Swing-leg min pelvis→ankle at apex (m). Measured med 0.668 / 0.608.
_SWING_EXT_LO, _SWING_EXT_HI = 0.64, 0.60
# Min |d_push − d_swing| at apex to count as a hurdle (measured med 0.296).
APEX_ASYM_MIN = 0.22
# Mean pelvis→ankle when the push leg opens for landing.
TUCK_EXTENDED = 0.60  # measured 0.567–0.656 lift, 0.567–0.710 land
# Soft penalty onset / success gate / hard fail on foot-to-foot separation.
# Measured max over flight 0.891; widen so the reference hurdle split is legal.
FOOT_SEP_SOFT = 0.80
FOOT_SEP_MAX_APEX = 0.85  # measured max 0.732 at apex
FOOT_SEP_MAX_FLIGHT = 0.95  # success gate (tightened vs splits farm)
FOOT_SEP_HARD = 1.15  # hard fail
# Extra sep allowance per meter of commanded flight beyond FLIGHT_DIST_MIN.
FOOT_SEP_DIST_SCALE = 0.20
# Balloon hard-fail: pelvis rise above target + this (m).
RISE_HARD_EXTRA = 0.20

# --- Forward torso pitch (rad): positive = lean forward (chest toward +x) ---
# Measured apex med ≈ 0.145 / 0.160; clip min 0.055.
_PITCH_TAKEOFF_LO, _PITCH_TAKEOFF_HI = 0.10, 0.14
_PITCH_APEX_LO, _PITCH_APEX_HI = 0.15, 0.18
PITCH_APEX_MIN = 0.05
PITCH_APEX_MAX = 0.35
# Soft excess sole height (m) before dense penalty (kept for foot_clearance).
CLEARANCE_EXCESS_MARGIN = 0.05
# Deprecated sole-based balloon; rise-based hard fail uses RISE_HARD_EXTRA.
CLEARANCE_HARD_EXTRA = 0.40  # widened so sole balloon no longer rejects reference

# --- Arm anchors (MuJoCo FK over airborne windows) ---
ARM_SPAN_APEX = 0.86  # hand-to-hand span (m); measured med 0.86
HAND_RISE_APEX = 0.36  # mean hand z above pelvis (m); measured peak up to 0.61
ELBOW_FLIGHT = 0.45  # |elbow| target (rad); measured means 0.20–0.58


def clamp_obstacle_height(h: torch.Tensor) -> torch.Tensor:
    """Clamp obstacle height into the tracker-verified band."""
    return h.clamp(min=H_MIN, max=H_MAX)


def flight_cap(h: torch.Tensor) -> torch.Tensor:
    """Height-coupled max flight distance from measured CSV bins.

    Returns ``1.20`` m for ``h ≤ 0.50`` and lerps down to ``1.00`` m at
    ``h = 0.75`` so (height, distance) stay jointly in-distribution.
    """
    t = ((h - _H_LO) / (_H_HI - _H_LO)).clamp(0.0, 1.0)
    return _FLIGHT_CAP_LO + (_FLIGHT_CAP_HI - _FLIGHT_CAP_LO) * t


def clamp_flight_distance(
    d: torch.Tensor, h: torch.Tensor | None = None
) -> torch.Tensor:
    """Clamp commanded flight distance into the demonstrated band.

    When ``h`` is provided, also apply :func:`flight_cap` so high obstacles
    cannot request longer flights than the 0.75 m CSV clips demonstrate.
    """
    d = d.clamp(min=FLIGHT_DIST_MIN, max=FLIGHT_DIST_MAX)
    if h is not None:
        d = torch.minimum(d, flight_cap(h))
    return d


def clamp_thickness(thickness: torch.Tensor) -> torch.Tensor:
    return thickness.clamp(min=THICKNESS_MIN, max=THICKNESS_MAX)


def _lerp_factor(h: torch.Tensor) -> torch.Tensor:
    """Interpolation weight in [0, 1] spanning the full ``[H_MIN, H_MAX]`` band.

    Anchors below ``_H_LO`` extrapolate toward the low bin (not flat), so early
    curriculum heights still get distinct shaping targets.
    """
    return ((h - H_MIN) / (H_MAX - H_MIN)).clamp(0.0, 1.0)


def _bin_lerp(h: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Lerp ``lo → hi`` over ``[_H_LO, _H_HI]``, flat outside (legacy bin map)."""
    t = ((h - _H_LO) / (_H_HI - _H_LO)).clamp(0.0, 1.0)
    return lo + (hi - lo) * t


def map_takeoff_targets(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map obstacle height → ``(v_x*, v_y*, v_z*, crouch*)``.

    Lerps over the full ``[H_MIN, H_MAX]`` band. ``v_y*`` is always 0.
    """
    h = clamp_obstacle_height(h)
    t = _lerp_factor(h)
    vx = _VX_LO + (_VX_HI - _VX_LO) * t
    vz = _VZ_LO + (_VZ_HI - _VZ_LO) * t
    crouch = _CROUCH_LO + (_CROUCH_HI - _CROUCH_LO) * t
    vy = torch.zeros_like(vx)
    return vx, vy, vz, crouch


def apex_rise_target(h: torch.Tensor) -> torch.Tensor:
    """Soft pelvis-rise reference (m) for obs / balloon / excess_rise.

    Lerps ``0.22 → 0.31`` over ``[H_MIN, H_MAX]`` (measured CSV medians).
    """
    h = clamp_obstacle_height(h)
    t = _lerp_factor(h)
    return _APEX_RISE_LO + (_APEX_RISE_HI - _APEX_RISE_LO) * t


def push_fold_target(h: torch.Tensor) -> torch.Tensor:
    """Max push-leg pelvis→ankle distance (m) allowed at apex.

    Lerps ``0.34 → 0.30`` over ``[H_MIN, H_MAX]``.
    """
    h = clamp_obstacle_height(h)
    t = _lerp_factor(h)
    return _PUSH_FOLD_LO + (_PUSH_FOLD_HI - _PUSH_FOLD_LO) * t


def swing_ext_target(h: torch.Tensor) -> torch.Tensor:
    """Min swing-leg pelvis→ankle distance (m) required at apex.

    Lerps ``0.64 → 0.60`` over ``[H_MIN, H_MAX]``.
    """
    h = clamp_obstacle_height(h)
    t = _lerp_factor(h)
    return _SWING_EXT_LO + (_SWING_EXT_HI - _SWING_EXT_LO) * t


def tuck_apex_target(h: torch.Tensor) -> torch.Tensor:
    """Alias of :func:`push_fold_target` kept for obs / command compatibility."""
    return push_fold_target(h)


def foot_sep_soft(flight_distance: torch.Tensor) -> torch.Tensor:
    """Soft penalty onset for foot separation, widened by commanded flight."""
    extra = (flight_distance - FLIGHT_DIST_MIN).clamp(min=0.0) * FOOT_SEP_DIST_SCALE
    return torch.full_like(flight_distance, FOOT_SEP_SOFT) + extra


def foot_sep_flight_max(flight_distance: torch.Tensor) -> torch.Tensor:
    """Success-gate foot-sep cap, widened by commanded flight."""
    extra = (flight_distance - FLIGHT_DIST_MIN).clamp(min=0.0) * FOOT_SEP_DIST_SCALE
    return torch.full_like(flight_distance, FOOT_SEP_MAX_FLIGHT) + extra


def foot_sep_hard(flight_distance: torch.Tensor) -> torch.Tensor:
    """Hard-fail foot-sep cap, widened by commanded flight."""
    extra = (flight_distance - FLIGHT_DIST_MIN).clamp(min=0.0) * FOOT_SEP_DIST_SCALE
    return torch.full_like(flight_distance, FOOT_SEP_HARD) + extra


def pitch_takeoff_target(h: torch.Tensor) -> torch.Tensor:
    """Forward torso pitch (rad) during plant/push / early rise."""
    h = clamp_obstacle_height(h)
    t = _lerp_factor(h)
    return _PITCH_TAKEOFF_LO + (_PITCH_TAKEOFF_HI - _PITCH_TAKEOFF_LO) * t


def pitch_apex_target(h: torch.Tensor) -> torch.Tensor:
    """Forward torso pitch (rad) at hurdle apex."""
    h = clamp_obstacle_height(h)
    t = _lerp_factor(h)
    return _PITCH_APEX_LO + (_PITCH_APEX_HI - _PITCH_APEX_LO) * t


def takeoff_standoff(h: torch.Tensor, thickness: torch.Tensor | None = None) -> torch.Tensor:
    """Distance from obstacle front face back to the takeoff / plant line."""
    h = clamp_obstacle_height(h)
    t = ((h - _H_LO) / (_H_HI - _H_LO)).clamp(0.0, 1.0)
    standoff = _STANDOFF_LO + (_STANDOFF_HI - _STANDOFF_LO) * t
    if thickness is not None:
        extra = (clamp_thickness(thickness) - THICKNESS_MIN).clamp(min=0.0) * 0.25
        standoff = standoff + extra.clamp(max=0.05)
    return standoff


__all__ = [
    "H_MIN",
    "H_MAX",
    "FLIGHT_DIST_MIN",
    "FLIGHT_DIST_MAX",
    "THICKNESS_MIN",
    "THICKNESS_MAX",
    "D_OBSTACLE_MIN",
    "D_OBSTACLE_MAX",
    "TUCK_EXTENDED",
    "APEX_ASYM_MIN",
    "ARM_SPAN_APEX",
    "HAND_RISE_APEX",
    "ELBOW_FLIGHT",
    "FOOT_SEP_SOFT",
    "FOOT_SEP_MAX_APEX",
    "FOOT_SEP_MAX_FLIGHT",
    "FOOT_SEP_HARD",
    "FOOT_SEP_DIST_SCALE",
    "RISE_HARD_EXTRA",
    "PITCH_APEX_MIN",
    "PITCH_APEX_MAX",
    "CLEARANCE_EXCESS_MARGIN",
    "CLEARANCE_HARD_EXTRA",
    "clamp_obstacle_height",
    "flight_cap",
    "clamp_flight_distance",
    "clamp_thickness",
    "map_takeoff_targets",
    "apex_rise_target",
    "push_fold_target",
    "swing_ext_target",
    "tuck_apex_target",
    "foot_sep_soft",
    "foot_sep_flight_max",
    "foot_sep_hard",
    "pitch_takeoff_target",
    "pitch_apex_target",
    "takeoff_standoff",
]
