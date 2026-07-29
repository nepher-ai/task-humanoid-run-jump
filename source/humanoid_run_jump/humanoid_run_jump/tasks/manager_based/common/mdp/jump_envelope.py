# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CSV-fitted jump takeoff envelope for the frozen BeyondMimic tracker.

Jump CSVs are named ``jump_over_obstacle_0_5m_*`` / ``0_75m_*``. Motions are
already retargeted to G1. CSV root positions are in **cm** (converted to m on
load). FK / filename tags show peak lower-foot height ≈ the obstacle tag
(≈ 0.50 m / 0.75 m) via **leg tuck** — root/pelvis only rises ~0.25–0.37 m —
so treat ``0.50`` / ``0.75`` as **literal obstacle heights** in the robot /
motion frame.

Policy-facing jump command is ``(h_obstacle, flight_distance)``. Takeoff
``(v_x*, v_z*, crouch*)`` bins remain internal shaping references from those
CSV clips.

Measured heading-aligned elevated-phase travel across ``motions/jump*.csv``
(non-``_M`` clips)::

    0.5 m clips  → horiz air ≈ 0.60–1.17 m (max 1.17 m)
    0.75 m clips → horiz air ≈ 0.94–0.99 m

So ``FLIGHT_DIST_MAX = 1.20`` and :func:`flight_cap` couple distance to height
so the sampler never issues a combination outside the demonstrated region.
"""

from __future__ import annotations

import torch

# Tracker-safe obstacle height band (meters in the G1 motion frame).
H_MIN = 0.0
H_MAX = 0.75  # matches highest CSV tag (+ no headroom beyond demonstrated clips)

# Anchor bins: literal CSV filename heights.
_H_LO = 0.50
_H_HI = 0.75

# Body-frame takeoff means from those same CSV bins.
_VX_LO, _VX_HI = 2.00, 2.10
_VZ_LO, _VZ_HI = 1.59, 1.56
_CROUCH_LO, _CROUCH_HI = 0.26, 0.34

# Demonstrated heading-aligned flight distance band (liftoff → landing, m).
# Measured max elevated-phase travel ≈ 1.17 m; use 1.20 with a small margin.
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

# --- Tucked-apex success envelope (MuJoCo FK over non-_M jump CSVs) ---
# Soft pelvis-rise *obs / shaping* refs only. CSV FK reports 0.21–0.29 m, but under
# physics vz_w≈1.5–1.9 m/s only buys ~vz²/(2g) ≈ 0.12–0.18 m of CoM rise (and
# tucking pulls the pelvis below the CoM arc). Success gates on sole clearance
# vs ``h_obstacle`` instead — see ``AmpManagerBasedRLEnv._ep_rise_ok``.
_APEX_RISE_LO, _APEX_RISE_HI = 0.14, 0.16
# Max per-leg pelvis→ankle distance allowed at apex (m). Both legs must reach
# at or below this (one-sided). Softened +5 cm vs CSV mean maxima (0.544 /
# 0.478) so early training gets a usable gradient from a straight-leg start.
_TUCK_APEX_LO, _TUCK_APEX_HI = 0.60, 0.55
# Mean pelvis→ankle distance when legs are extended (liftoff / pre-landing).
TUCK_EXTENDED = 0.60  # measured 0.567–0.656 lift, 0.567–0.710 land
# Anti-splits caps on foot-to-foot separation.
FOOT_SEP_MAX_APEX = 0.80  # measured max 0.732 at apex
FOOT_SEP_MAX_FLIGHT = 0.90  # measured max 0.891 over whole flight
# Hard fail: far beyond CSV (splits farm / ballistic straddle).
FOOT_SEP_HARD = 1.05

# --- Forward torso pitch (rad): positive = lean forward (chest toward +x) ---
# Measured from jump CSVs via atan2(-g_b_x, -g_b_z):
#   liftoff med ≈ 0.07 / 0.14 ; apex med ≈ 0.15 / 0.16 ; land med ≈ 0.17 / 0.15
_PITCH_TAKEOFF_LO, _PITCH_TAKEOFF_HI = 0.10, 0.14
_PITCH_APEX_LO, _PITCH_APEX_HI = 0.18, 0.20
# Success band for apex pitch latch (wider than the dense Gaussian target).
PITCH_APEX_MIN = 0.08
PITCH_APEX_MAX = 0.40
# Soft excess sole height (m) before dense penalty; hard balloon cap above h.
CLEARANCE_EXCESS_MARGIN = 0.05
CLEARANCE_HARD_EXTRA = 0.25


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
    """Interpolation weight in [0, 1] from low→high bins; flat below ``_H_LO``."""
    return ((h - _H_LO) / (_H_HI - _H_LO)).clamp(0.0, 1.0)


def map_takeoff_targets(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map obstacle height → ``(v_x*, v_y*, v_z*, crouch*)``.

    For ``h < 0.50`` uses the 0.5 m CSV bin. For ``h`` up to 0.75 lerps toward
    the 0.75 m bin. ``v_y*`` is always 0.
    """
    h = clamp_obstacle_height(h)
    t = _lerp_factor(h)
    vx = _VX_LO + (_VX_HI - _VX_LO) * t
    vz = _VZ_LO + (_VZ_HI - _VZ_LO) * t
    crouch = _CROUCH_LO + (_CROUCH_HI - _CROUCH_LO) * t
    vy = torch.zeros_like(vx)
    return vx, vy, vz, crouch


def apex_rise_target(h: torch.Tensor) -> torch.Tensor:
    """Soft pelvis-rise reference (m) for obs / launch shaping.

    Lerps ``0.14 → 0.16`` over ``[0.50, 0.75]`` (physics-reachable under the
    CSV takeoff ``v_z*``). Apex *success* uses sole clearance ≥ ``h_obstacle``.
    """
    h = clamp_obstacle_height(h)
    t = _lerp_factor(h)
    return _APEX_RISE_LO + (_APEX_RISE_HI - _APEX_RISE_LO) * t


def tuck_apex_target(h: torch.Tensor) -> torch.Tensor:
    """Max per-leg pelvis→ankle distance (m) allowed at the jump apex.

    Both legs must independently reach at or below this value (one-sided).
    Lerps ``0.60 → 0.55`` over the ``[0.50, 0.75]`` bins (+5 cm vs CSV maxima
    for learnability; tighten later once tuck_ok_rate is healthy).
    """
    h = clamp_obstacle_height(h)
    t = _lerp_factor(h)
    return _TUCK_APEX_LO + (_TUCK_APEX_HI - _TUCK_APEX_LO) * t


def pitch_takeoff_target(h: torch.Tensor) -> torch.Tensor:
    """Forward torso pitch (rad) during plant/push / early rise.

    Lerps ``0.10 → 0.14`` over ``[0.50, 0.75]`` (CSV liftoff medians ~0.07/0.14;
    target slightly above the lower bin so lean is encouraged).
    """
    h = clamp_obstacle_height(h)
    t = _lerp_factor(h)
    return _PITCH_TAKEOFF_LO + (_PITCH_TAKEOFF_HI - _PITCH_TAKEOFF_LO) * t


def pitch_apex_target(h: torch.Tensor) -> torch.Tensor:
    """Forward torso pitch (rad) at tucked apex.

    Lerps ``0.18 → 0.20`` over ``[0.50, 0.75]`` (CSV apex medians ~0.15/0.16;
    slight headroom so vertical tuck is not rewarded).
    """
    h = clamp_obstacle_height(h)
    t = _lerp_factor(h)
    return _PITCH_APEX_LO + (_PITCH_APEX_HI - _PITCH_APEX_LO) * t


def takeoff_standoff(h: torch.Tensor, thickness: torch.Tensor | None = None) -> torch.Tensor:
    """Distance from obstacle front face back to the takeoff / plant line."""
    h = clamp_obstacle_height(h)
    t = _lerp_factor(h)
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
    "FOOT_SEP_MAX_APEX",
    "FOOT_SEP_MAX_FLIGHT",
    "FOOT_SEP_HARD",
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
    "tuck_apex_target",
    "pitch_takeoff_target",
    "pitch_apex_target",
    "takeoff_standoff",
]
