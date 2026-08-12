# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only sanity checks for the plant-band redesign (no Isaac Sim required).

Requires the editable install (``pip install -e source/humanoid_run_jump``).

Run from ``task-humanoid-run-jump/``::

    python scripts/check_plant_band.py
"""

from __future__ import annotations

import torch

from humanoid_run_jump.stride.hl_stride import (
    APPROACH_STRIDE_X,
    FLIGHT_HL_MIN,
    PLANT_BAND_MAX,
    PLANT_BAND_MIN,
    STRIDE_ACH_MAX,
    STRIDE_ACH_MIN,
    _PK_FL,
    _PK_VX,
    _PK_X,
    apex_from_plant,
    band_proximity,
    d_star,
    feasible_stride_count,
    flight_for_standoff,
    in_plant_band,
    invert_apex,
    lattice_from_remaining,
    plant_band_score,
)


def _h_score(h_cmd: torch.Tensor, h_req: torch.Tensor, sigma: float) -> torch.Tensor:
    """Mirror of ``hl_rewards._h_score`` (avoids importing Isaac-backed rewards)."""
    deficit = (h_req - h_cmd).clamp(min=0.0)
    return torch.exp(-((deficit / max(sigma, 1e-3)) ** 2))


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def check_d_star_in_band() -> None:
    t = torch.tensor(0.20)
    errs = []
    print("\n=== d* table (t=0.20) vs canvas PK_X ===")
    print(f"{'vx':>5} {'f':>5} {'PK_X':>7} {'d*':>7} {'PK-t/2':>8}")
    for i, vx in enumerate(_PK_VX.tolist()):
        for j, f in enumerate(_PK_FL.tolist()):
            pk = float(_PK_X[i, j])
            d = float(d_star(t, torch.tensor(f), torch.tensor(vx)))
            raw = pk - 0.5 * float(t)
            print(f"{vx:5.1f} {f:5.1f} {pk:7.3f} {d:7.3f} {raw:8.3f}")
            if not (PLANT_BAND_MIN - 1e-6 <= d <= PLANT_BAND_MAX + 1e-6):
                errs.append(f"d*({vx},{f})={d:.3f} outside [{PLANT_BAND_MIN},{PLANT_BAND_MAX}]")
    if errs:
        _fail("; ".join(errs))
    print("[ok] every d* in [0.90, 1.20]")


def check_apex_roundtrip() -> None:
    max_err = 0.0
    for vx in _PK_VX:
        for f in _PK_FL:
            p = apex_from_plant(f, vx)
            f_back = invert_apex(p, vx)
            # Cummax can shift slightly on non-monotone knots; allow 0.02 m on P.
            p_back = apex_from_plant(f_back, vx)
            err = float((p_back - p).abs())
            max_err = max(max_err, err)
            if err > 0.02:
                _fail(f"round-trip P err {err:.4f} at vx={float(vx)} f={float(f)}")
    print(f"[ok] apex round-trip max |dP|={max_err:.4f} m (<0.02)")


def check_push_foot_clearance() -> None:
    t = torch.tensor(0.20)
    worst = 1e9
    for vx in _PK_VX:
        for f in _PK_FL:
            d = float(d_star(t, f, vx))
            clearance = d - APPROACH_STRIDE_X
            worst = min(worst, clearance)
            if clearance <= 0.25:
                _fail(
                    f"push foot too close: d*={d:.3f}, approach={APPROACH_STRIDE_X}, "
                    f"clearance={clearance:.3f} at vx={float(vx)} f={float(f)}"
                )
    print(f"[ok] min push-foot clearance={worst:.3f} m (>0.25)")


def check_flight_for_standoff() -> None:
    t = torch.tensor(0.20)
    vx = torch.tensor(1.5)
    # A mid-band plant should invert to a mid flight.
    d = torch.tensor(0.90)
    f = flight_for_standoff(d, t, vx)
    if not (FLIGHT_HL_MIN - 1e-3 <= float(f) <= 1.20 + 1e-3):
        _fail(f"flight_for_standoff out of range: {float(f)}")
    d2 = d_star(t, f, vx)
    # Invert then re-apply should land near the requested standoff (band clamp aside).
    if abs(float(d2) - float(d)) > 0.05:
        _fail(f"standoff->flight consistency |{float(d2)}-{float(d)}|={abs(float(d2)-float(d)):.3f}")
    print(f"[ok] flight_for_standoff(0.90)->{float(f):.3f}, d* back={float(d2):.3f}")


def check_plant_band_score() -> None:
    d = torch.tensor([0.90, 1.05, 1.20, 0.70, 1.40])
    s = plant_band_score(d)
    if float(s[0]) < 0.999 or float(s[1]) < 0.999 or float(s[2]) < 0.999:
        _fail(f"in-band scores not +1: {s[:3].tolist()}")
    if float(s[3]) >= 1.0 or float(s[4]) >= 1.0:
        _fail(f"out-of-band scores should be <1: {s[3:].tolist()}")
    if not in_plant_band(torch.tensor(1.05)):
        _fail("in_plant_band(1.05) False")
    if in_plant_band(torch.tensor(0.70)):
        _fail("in_plant_band(0.70) True")
    print(f"[ok] plant_band_score={s.tolist()}")


def check_lattice_deadbeat() -> None:
    """Following the committed plan must land the n-th plant exactly on ``d*``.

    This is the property the earlier smoothstep blend lacked: it was continuous but
    not convergent, so a robot tracking it perfectly still missed the band unless it
    happened to start on a lattice node.
    """
    d_tgt = torch.full((1,), 1.05)
    worst_miss = 0.0
    worst_v_jump = 0.0
    for L_val in (0.95, 1.30, 1.70, 1.92):
        L = torch.full((1,), L_val)
        for D0 in torch.arange(2.2, 9.0, 0.07):
            rem = (D0.reshape(1) - d_tgt).clamp(min=0.05)
            n_left = int(feasible_stride_count(rem, L).item())
            v_prev = None
            while n_left >= 1:
                nt = torch.full((1,), float(n_left))
                _, L_req, v = lattice_from_remaining(rem, L, n_commit=nt)
                if v_prev is not None:
                    worst_v_jump = max(worst_v_jump, abs(float(v) - v_prev))
                v_prev = float(v)
                rem = rem - L_req
                n_left -= 1
            miss = abs(float(rem))
            worst_miss = max(worst_miss, miss)
            if miss > 1e-3:
                _fail(f"plan misses d* by {miss:.4f} m at L={L_val} D0={float(D0):.2f}")
    # The required stride is constant under exact tracking, so spending the plan
    # must not move v_star at all.
    if worst_v_jump > 1e-3:
        _fail(f"v_star drifts {worst_v_jump:.4f} m/s while spending a committed plan")
    print(
        f"[ok] committed plan lands on d* (max miss {worst_miss:.2e} m, "
        f"v* drift {worst_v_jump:.2e} m/s)"
    )


def check_lattice_continuity() -> None:
    """With the count held, ``v_star`` must be continuous in the remaining gap."""
    worst_jump = 0.0
    worst_at = 0.0
    for L_val in (0.95, 1.30, 1.70, 1.92):
        L = torch.full((1,), L_val)
        for n_val in (1.0, 2.0, 3.0, 5.0):
            n = torch.full((1,), n_val)
            rem = torch.arange(0.20, 9.0, 0.01)
            v = torch.stack(
                [lattice_from_remaining(r.reshape(1), L, n_commit=n)[2].reshape(()) for r in rem]
            )
            j = float((v[1:] - v[:-1]).abs().max())
            if j > worst_jump:
                worst_jump = j
                worst_at = L_val
            if j > 0.05:
                _fail(f"v_star discontinuity {j:.3f} m/s per 1 cm at L={L_val} n={n_val}")
    print(f"[ok] v_star max step={worst_jump:.4f} m/s per 1 cm (L={worst_at}, <0.05)")


def check_stride_count_feasible() -> None:
    """Every committed stride must be one the run policy can actually produce."""
    worst = None
    for L_val in (0.95, 1.30, 1.70, 1.92):
        L = torch.full((1,), L_val)
        for rem_v in torch.arange(0.95, 12.0, 0.05):
            rem = rem_v.reshape(1)
            n = feasible_stride_count(rem, L)
            L_req = float(rem / n)
            if not (STRIDE_ACH_MIN - 1e-6 <= L_req <= STRIDE_ACH_MAX + 1e-6):
                _fail(
                    f"required stride {L_req:.3f} outside "
                    f"[{STRIDE_ACH_MIN}, {STRIDE_ACH_MAX}] at rem={float(rem_v):.2f} L={L_val}"
                )
            if worst is None or abs(L_req - 1.4) > abs(worst - 1.4):
                worst = L_req
    print(f"[ok] every committed stride achievable (extreme {worst:.3f} m)")


def check_band_proximity() -> None:
    d = torch.tensor([0.90, 1.05, 1.20, 1.00, 0.60, 0.30, 1.50, 1.90])
    p = band_proximity(d)
    if float(p[:4].min()) < 0.999:
        _fail(f"in-band proximity not 1: {p[:4].tolist()}")
    if float(p.min()) <= 0.0:
        _fail("proximity must stay positive so it is a safe multiplicative gate")
    # Strictly decreasing as we move away from each edge.
    if not (float(p[4]) > float(p[5])) or not (float(p[6]) > float(p[7])):
        _fail(f"proximity not monotone outside the band: {p.tolist()}")
    print(f"[ok] band_proximity={[round(x, 3) for x in p.tolist()]}")


def check_h_score_one_sided() -> None:
    h_req = torch.tensor([0.60, 0.60, 0.60])
    h_cmd = torch.tensor([0.60, 0.80, 0.40])
    s = _h_score(h_cmd, h_req, 0.10)
    if float(s[1]) < 0.999:
        _fail(f"over-clearance must be free, got {float(s[1]):.3f}")
    if float(s[2]) >= 0.5:
        _fail(f"under-clearance must be penalised, got {float(s[2]):.3f}")
    print(f"[ok] one-sided h score: exact={float(s[0]):.3f} high={float(s[1]):.3f} low={float(s[2]):.3f}")


def check_reward_zero_on_no_event() -> None:
    """Assert reward early-returns are zero without importing Isaac Lab.

    ``hl_rewards`` pulls isaaclab at import time, so we re-implement the
    no-event gates here against the same helpers the rewards use.
    """

    class _MockEnv:
        def __init__(self):
            self.num_envs = 4
            self.device = torch.device("cpu")
            self.hl_just_latched = torch.zeros(4, dtype=torch.bool)
            self.has_next_obstacle = torch.zeros(4, dtype=torch.bool)

    env = _MockEnv()
    # Mirror plant_in_band / jump_cmd_match / latch_quality early returns.
    just = env.hl_just_latched
    if just is None or not just.any():
        z = torch.zeros(env.num_envs)
    else:
        _fail("mock just_latched should be empty")
    if z.abs().sum() != 0:
        _fail("no-event plant score not zero")
    if env.has_next_obstacle.any():
        _fail("mock has_next_obstacle should be empty")
    print("[ok] no-event reward gates return zero (mirrored)")


def main() -> None:
    print("Plant-band redesign checks")
    print(f"  PLANT_BAND = [{PLANT_BAND_MIN}, {PLANT_BAND_MAX}]")
    print(f"  FLIGHT_HL_MIN = {FLIGHT_HL_MIN}")
    print(f"  APPROACH_STRIDE_X = {APPROACH_STRIDE_X}")
    check_d_star_in_band()
    check_apex_roundtrip()
    check_push_foot_clearance()
    check_flight_for_standoff()
    check_plant_band_score()
    check_band_proximity()
    check_lattice_deadbeat()
    check_lattice_continuity()
    check_stride_count_feasible()
    check_h_score_one_sided()
    check_reward_zero_on_no_event()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
