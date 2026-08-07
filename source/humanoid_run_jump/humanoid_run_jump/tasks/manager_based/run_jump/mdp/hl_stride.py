# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canvas-derived stride lattice and apex helpers for RunJumpHL.

All numeric tables come from the Run Stride Velocity and Jump Policy Metrics
analysis canvases. The jump canvas section ``3. Right-first contact → trajectory
peak`` supplies ``PK_X_GRID`` (vx_bin × flight_cmd → plant→root peak X).
"""

from __future__ import annotations

import torch

from humanoid_run_jump.tasks.manager_based.jump.mdp.jump_envelope import (
    clamp_flight_distance,
    flight_cap,
)

# ---------------------------------------------------------------------------
# Plant band (right ankle standoff to obstacle front face).
# Canvas PK_X spans ~0.61–1.31 m plant→root; minus t/2 ≈ 0.10 yields ~0.51–1.21.
# Hard band for rewards / viz is the user-requested [0.90, 1.20].
# ---------------------------------------------------------------------------
PLANT_BAND_MIN = 0.90
PLANT_BAND_MAX = 1.20
# HL flight floor (table starts at 0.6; envelope FLIGHT_DIST_MIN is 0.50).
FLIGHT_HL_MIN = 0.60

# Reward / shaping sigmas (m).
PLANT_EDGE_SIGMA = 0.20
# Curriculum schedule for the band falloff: generous early so a 0.5 m miss still
# earns partial credit, tightening to PLANT_EDGE_SIGMA once the skill exists.
PLANT_EDGE_SIGMA_START = 0.50
# Proximity kernel for the dense per-step shaping. Deliberately much wider than
# the scoring sigma so there is gradient from several strides out.
SHAPING_BAND_SIGMA = 0.60
FLIGHT_MATCH_SIGMA = 0.25
H_MATCH_SIGMA = 0.10
# Commanded-speed tracking sigma (m/s) for the zero-lag lattice term.
V_CMD_MATCH_SIGMA = 0.35
# Nominal hurdle-height margin used as the h_cmd reward reference.
H_MARGIN_NOMINAL = 0.10
H_MARGIN_MIN = 0.05
H_MARGIN_MAX = 0.30

# Apex alignment reward sigma (m) — kept for any residual callers.
APEX_ALIGN_SIGMA = 0.20

# ---------------------------------------------------------------------------
# Jump canvas: R-first → peak X mean (m) by vx_bin × flight_cmd.
# Rows: vx_cmd ∈ {1.0, 1.5, 2.0, 2.5, 3.0}; cols: flight ∈ {0.6, 0.8, 1.0, 1.2}.
# ---------------------------------------------------------------------------
_PK_VX = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0], dtype=torch.float32)
_PK_FL = torch.tensor([0.6, 0.8, 1.0, 1.2], dtype=torch.float32)
_PK_X = torch.tensor(
    [
        [0.613, 1.018, 1.212, 1.206],
        [0.649, 0.918, 1.166, 1.208],
        [0.776, 1.012, 1.070, 1.199],
        [0.767, 0.934, 1.135, 1.308],
        [0.952, 0.944, 1.088, 1.248],
    ],
    dtype=torch.float32,
)
# Cummax along flight so inversion stays monotone (rows 0 and 4 have tiny dips).
_PK_X_MONO = torch.cummax(_PK_X, dim=1).values

# peak_z(h) = 1.193 * h + 0.040  (canvas fit)
PEAK_Z_SLOPE = 1.193
PEAK_Z_INTERCEPT = 0.040

# Landing distance from liftoff: F(f) = 0.41 * f + 0.655
FLIGHT_GAIN = 0.41
FLIGHT_OFFSET = 0.655

# Approach stride (right plant → left push) mean Δx from canvas.
APPROACH_STRIDE_X = 0.29

# ---------------------------------------------------------------------------
# Run canvas: mean same-foot stride vs commanded vx, and realized speed.
# Authority lives in 1.0–2.0 m/s; stride saturates above 2.0.
# ---------------------------------------------------------------------------
_VX_CMD = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0], dtype=torch.float32)
_STRIDE_MEAN = torch.tensor([0.90, 1.70, 1.92, 1.96, 2.09], dtype=torch.float32)
_VX_ACTUAL = torch.tensor([1.26, 1.85, 2.25, 2.55, 2.77], dtype=torch.float32)

# Invertible stride band for deadbeat (saturates above ~1.92 m).
STRIDE_CMD_MAX = 1.92
VX_CMD_MIN = 1.0
VX_CMD_MAX = 3.0

# Same-foot strides the run policy can actually produce, from the run canvas
# (0.90 m at 1.0 m/s up to 1.92 m at 2.0 m/s). Used to reject stride counts whose
# required stride is outside the achievable range.
STRIDE_ACH_MIN = 0.90
STRIDE_ACH_MAX = STRIDE_CMD_MAX


def peak_z(h: torch.Tensor) -> torch.Tensor:
    """Predicted peak sole height (m) from commanded obstacle height."""
    return PEAK_Z_SLOPE * h + PEAK_Z_INTERCEPT


def flight_actual(f: torch.Tensor) -> torch.Tensor:
    """Predicted landing distance (root→root) from flight_cmd."""
    return FLIGHT_GAIN * f + FLIGHT_OFFSET


def _lerp_axis(x: torch.Tensor, knots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Piecewise-linear segment indices and weight for ``x`` along ``knots``."""
    device = x.device
    dtype = x.dtype
    xs = knots.to(device=device, dtype=dtype)
    x_clamped = x.clamp(xs[0], xs[-1])
    idx = torch.searchsorted(xs, x_clamped, right=False).clamp(1, xs.numel() - 1)
    x0 = xs[idx - 1]
    x1 = xs[idx]
    t = (x_clamped - x0) / (x1 - x0).clamp(min=1e-6)
    return idx - 1, idx, t


def apex_from_plant(f: torch.Tensor, vx: torch.Tensor) -> torch.Tensor:
    """Bilinear P(f, vx): right-plant → root at max sole height (m along +x).

    Indexed by **commanded** vx (same frame as the jump-analysis bank), not
    measured root speed.
    """
    f = f.reshape(-1)
    vx = vx.reshape(-1)
    i0, i1, tu = _lerp_axis(vx, _PK_VX)
    j0, j1, tv = _lerp_axis(f, _PK_FL)
    grid = _PK_X_MONO.to(device=f.device, dtype=f.dtype)
    p00 = grid[i0, j0]
    p01 = grid[i0, j1]
    p10 = grid[i1, j0]
    p11 = grid[i1, j1]
    p0 = p00 + (p01 - p00) * tv
    p1 = p10 + (p11 - p10) * tv
    return p0 + (p1 - p0) * tu


def invert_apex(p: torch.Tensor, vx: torch.Tensor) -> torch.Tensor:
    """Invert P(f, vx) ≈ p for flight_cmd using the cummax row (monotone)."""
    p = p.reshape(-1)
    vx = vx.reshape(-1)
    device = p.device
    dtype = p.dtype
    i0, i1, tu = _lerp_axis(vx, _PK_VX)
    mono = _PK_X_MONO.to(device=device, dtype=dtype)
    fl = _PK_FL.to(device=device, dtype=dtype)
    # Interpolate the monotone row at this vx. Shape (N, 4).
    row = mono[i0] + (mono[i1] - mono[i0]) * tu.unsqueeze(-1)
    p_clamped = p.clamp(row[:, 0], row[:, -1])
    # Rightmost knot index with row[k] <= p, then interpolate to k+1.
    j_lo = torch.zeros(p.shape[0], dtype=torch.long, device=device)
    for k in range(fl.numel()):
        j_lo = torch.where(row[:, k] <= p_clamped, torch.full_like(j_lo, k), j_lo)
    j_lo = j_lo.clamp(0, fl.numel() - 2)
    j_hi = j_lo + 1
    y0 = torch.gather(row, 1, j_lo.unsqueeze(-1)).squeeze(-1)
    y1 = torch.gather(row, 1, j_hi.unsqueeze(-1)).squeeze(-1)
    f0 = fl[j_lo]
    f1 = fl[j_hi]
    t = (p_clamped - y0) / (y1 - y0).clamp(min=1e-6)
    return f0 + t * (f1 - f0)


def d_star(
    thickness: torch.Tensor,
    flight: torch.Tensor,
    vx: torch.Tensor,
) -> torch.Tensor:
    """Ankle-referenced target standoff to obstacle front face.

    Canvas ``PK_X`` is plant → root at peak. Placing the root over the obstacle
    centre requires::

        plant_x + P(f, vx) == s_front + t/2
        =>  d* = s_front - plant_x = P(f, vx) - t/2

    Clamped into ``[PLANT_BAND_MIN, PLANT_BAND_MAX]``.
    """
    raw = apex_from_plant(flight, vx) - 0.5 * thickness
    return raw.clamp(min=PLANT_BAND_MIN, max=PLANT_BAND_MAX)


def flight_for_standoff(
    d_latch: torch.Tensor,
    thickness: torch.Tensor,
    vx: torch.Tensor,
) -> torch.Tensor:
    """Flight_cmd the grid says matches a given right-foot plant standoff."""
    target_p = d_latch + 0.5 * thickness
    return invert_apex(target_p, vx)


def apex_trim(
    d_latch: torch.Tensor,
    thickness: torch.Tensor,
    h: torch.Tensor,
    vx: torch.Tensor,
) -> torch.Tensor:
    """Solve ``P(f, vx) = d + t/2`` for flight_cmd, clamped to the envelope."""
    f_raw = flight_for_standoff(d_latch, thickness, vx)
    f_raw = f_raw.clamp(min=FLIGHT_HL_MIN)
    return clamp_flight_distance(f_raw, h)


def trim_feasible(
    d_latch: torch.Tensor,
    thickness: torch.Tensor,
    h: torch.Tensor,
    vx: torch.Tensor,
) -> torch.Tensor:
    """True when ``apex_trim`` lands inside the envelope without clipping."""
    f_raw = flight_for_standoff(d_latch, thickness, vx).clamp(min=FLIGHT_HL_MIN)
    f_clamped = clamp_flight_distance(f_raw, h)
    return (f_raw - f_clamped).abs() < 1e-3


def in_plant_band(d: torch.Tensor) -> torch.Tensor:
    """Boolean mask: standoff inside ``[PLANT_BAND_MIN, PLANT_BAND_MAX]``."""
    return (d >= PLANT_BAND_MIN) & (d <= PLANT_BAND_MAX)


def plant_band_score(d: torch.Tensor, sigma: float = PLANT_EDGE_SIGMA) -> torch.Tensor:
    """Flat-top signed score in ``[-1, 1]``: +1 inside the band, gaussian falloff outside."""
    below = (PLANT_BAND_MIN - d).clamp(min=0.0)
    above = (d - PLANT_BAND_MAX).clamp(min=0.0)
    dist = below + above  # 0 inside the band
    # Inside → +1; outside → 2*exp(-(dist/σ)^2)-1 ∈ (-1, 1).
    return torch.where(
        dist <= 0.0,
        torch.ones_like(d),
        2.0 * torch.exp(-(dist / max(sigma, 1e-3)) ** 2) - 1.0,
    )


def band_proximity(d: torch.Tensor, sigma: float = SHAPING_BAND_SIGMA) -> torch.Tensor:
    """Unsigned proximity in ``(0, 1]``: 1 inside the band, gaussian falloff outside.

    Unlike :func:`plant_band_score` this never goes negative, so it is safe as a
    multiplicative gate on dense shaping terms.
    """
    below = (PLANT_BAND_MIN - d).clamp(min=0.0)
    above = (d - PLANT_BAND_MAX).clamp(min=0.0)
    dist = below + above
    return torch.exp(-((dist / max(sigma, 1e-3)) ** 2))


def stride_length(v_act: torch.Tensor) -> torch.Tensor:
    """Predicted same-foot stride length (m) from *actual* forward speed."""
    device = v_act.device
    dtype = v_act.dtype
    xs = _VX_ACTUAL.to(device=device, dtype=dtype)
    ys = _STRIDE_MEAN.to(device=device, dtype=dtype)
    v = v_act.clamp(xs[0], xs[-1])
    idx = torch.searchsorted(xs, v, right=False).clamp(1, xs.numel() - 1)
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    t = (v - x0) / (x1 - x0).clamp(min=1e-6)
    return y0 + t * (y1 - y0)


def cmd_from_stride(L_req: torch.Tensor) -> torch.Tensor:
    """Invert stride→vx_cmd, saturating at ``STRIDE_CMD_MAX`` (ill-conditioned above).

    Saturation is a plain clamp on the *input*, so the output is continuous. An
    earlier version jumped straight to ``VX_CMD_MAX`` once ``L_req`` exceeded the
    invertible band, which put a 1 m/s cliff in the middle of the speed target.
    Commanding 3 m/s buys only 1.92→2.09 m of stride anyway, while arriving
    sooner, which is the opposite of what a stride-lengthening request wants.
    """
    device = L_req.device
    dtype = L_req.dtype
    xs = _STRIDE_MEAN.to(device=device, dtype=dtype)
    ys = _VX_CMD.to(device=device, dtype=dtype)
    L = L_req.clamp(xs[0], STRIDE_CMD_MAX)
    idx = torch.searchsorted(xs, L, right=False).clamp(1, xs.numel() - 1)
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    t = (L - x0) / (x1 - x0).clamp(min=1e-6)
    v = y0 + t * (y1 - y0)
    return v.clamp(VX_CMD_MIN, VX_CMD_MAX)


def expected_actual_speed(v_cmd: torch.Tensor) -> torch.Tensor:
    """Map a commanded vx to the speed the frozen run policy actually realizes.

    The run canvas shows a systematic +0.25 m/s offset (1.0→1.26, 2.0→2.25,
    3.0→2.77), so comparing measured speed against a command directly biases any
    tracking error by nearly one std.
    """
    device = v_cmd.device
    dtype = v_cmd.dtype
    xs = _VX_CMD.to(device=device, dtype=dtype)
    ys = _VX_ACTUAL.to(device=device, dtype=dtype)
    v = v_cmd.clamp(xs[0], xs[-1])
    idx = torch.searchsorted(xs, v, right=False).clamp(1, xs.numel() - 1)
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    t = (v - x0) / (x1 - x0).clamp(min=1e-6)
    return y0 + t * (y1 - y0)


def feasible_stride_count(remaining: torch.Tensor, L_ema: torch.Tensor) -> torch.Tensor:
    """Stride count nearest the current stride that keeps ``remaining / n`` achievable.

    Plain ``round(remaining / L)`` can demand a stride the run policy cannot
    produce (e.g. 4.10 m in 2 strides is 2.05 m, above the 1.92 m ceiling), so the
    count is clamped into ``[ceil(remaining / L_max), floor(remaining / L_min)]``.
    """
    L = L_ema.clamp(min=STRIDE_ACH_MIN, max=STRIDE_ACH_MAX)
    n_min = torch.ceil(remaining / STRIDE_ACH_MAX).clamp(min=1.0)
    n_max = torch.maximum(torch.floor(remaining / STRIDE_ACH_MIN).clamp(min=1.0), n_min)
    n = (remaining / L).round().clamp(min=1.0, max=12.0)
    return torch.maximum(torch.minimum(n, n_max), n_min)


def lattice_from_remaining(
    remaining: torch.Tensor,
    L_ema: torch.Tensor,
    n_commit: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deadbeat lattice: return ``(n, L_req, v_star)`` for a plant-to-``d*`` gap.

    ``remaining`` must be measured from the **last completed right plant**, not the
    live ankle position: the plan says "cover this gap in ``n`` more plants", and
    the gap only means that from where the foot last touched down.

    With ``n_commit`` supplied (:class:`StrideTracker` holds it and decrements it at
    each right plant) the answer is the textbook deadbeat one, ``L_req = remaining /
    n``: a *constant* stride whose n-th plant lands exactly on ``d*``, staying
    constant under perfect tracking because ``remaining`` and ``n`` shrink together.
    It is also continuous, which recomputing ``round(remaining / L)`` every step is
    not — that jumped the required stride by a factor of two, roughly 1 m/s of
    ``v_star``, at every half-lattice boundary, and following it did not even reach
    ``d*`` unless the robot happened to start on a lattice node.

    Without ``n_commit`` the count is re-derived from :func:`feasible_stride_count`.
    That path is only for callers with no per-env state (the sanity script, the
    initial latch) and does sawtooth across lattice boundaries.
    """
    rem = remaining.clamp(min=0.05)
    n = feasible_stride_count(rem, L_ema) if n_commit is None else n_commit.clamp(min=1.0)
    L_req = rem / n
    return n, L_req, cmd_from_stride(L_req)


class StrideTracker:
    """Per-env EMA of right-foot same-plant stride length, plus the committed plan.

    ``n_commit`` is the number of right plants the robot has committed to taking
    before the obstacle. It is latched once on approach and decremented at each
    plant, which is what makes ``L_req = remaining / n_commit`` a constant,
    continuous and exactly deadbeat stride target. It is only re-latched when the
    plan becomes unachievable, so the target never jumps on a threshold crossing.
    """

    def __init__(self, num_envs: int, device: torch.device, alpha: float = 0.35):
        self.alpha = float(alpha)
        self.L_ema = torch.full((num_envs,), 1.70, device=device)
        self.prev_right_x = torch.zeros(num_envs, device=device)
        self.has_prev = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.prev_contact_r = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.last_plant_s = torch.zeros(num_envs, device=device)
        self.has_last_plant = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.plant_event = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.n_commit = torch.ones(num_envs, device=device)
        self.has_commit = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor | slice) -> None:
        self.L_ema[env_ids] = 1.70
        self.prev_right_x[env_ids] = 0.0
        self.has_prev[env_ids] = False
        self.prev_contact_r[env_ids] = False
        self.last_plant_s[env_ids] = 0.0
        self.has_last_plant[env_ids] = False
        self.plant_event[env_ids] = False
        self.reset_commitment(env_ids)

    def reset_commitment(self, env_ids: torch.Tensor | slice) -> None:
        """Drop the committed plan (new obstacle, or episode reset)."""
        self.n_commit[env_ids] = 1.0
        self.has_commit[env_ids] = False

    def update(self, right_x: torch.Tensor, right_contact: torch.Tensor) -> None:
        """Call each control step with right ankle world-x and contact mask."""
        plant = right_contact & (~self.prev_contact_r)
        self.plant_event[:] = plant
        if plant.any():
            ids = plant.nonzero(as_tuple=False).squeeze(-1)
            has = self.has_prev[ids]
            if has.any():
                hid = ids[has]
                gap = (right_x[hid] - self.prev_right_x[hid]).clamp(min=0.20, max=2.80)
                a = self.alpha
                self.L_ema[hid] = (1.0 - a) * self.L_ema[hid] + a * gap
            self.prev_right_x[ids] = right_x[ids]
            self.has_prev[ids] = True
            self.last_plant_s[ids] = right_x[ids]
            self.has_last_plant[ids] = True
        self.prev_contact_r[:] = right_contact

    def update_commitment(self, remaining: torch.Tensor, active: torch.Tensor) -> None:
        """Maintain ``n_commit`` for envs where ``active`` (obstacle ahead, running).

        Call once per control step *after* :meth:`update`, with
        ``remaining = d_right - d_target``.
        """
        rem = remaining.clamp(min=0.05)
        # One plant of the plan is spent whenever the right foot touches down.
        stepped = self.plant_event & self.has_commit & active
        n = torch.where(stepped, (self.n_commit - 1.0).clamp(min=1.0), self.n_commit)
        # Re-latch only when unset or when remaining/n is no longer achievable.
        n_min = torch.ceil(rem / STRIDE_ACH_MAX).clamp(min=1.0)
        n_max = torch.maximum(torch.floor(rem / STRIDE_ACH_MIN).clamp(min=1.0), n_min)
        stale = (n < n_min) | (n > n_max)
        relatch = active & ((~self.has_commit) | stale)
        n = torch.where(relatch, feasible_stride_count(rem, self.L_ema), n)
        self.n_commit = torch.where(active, n, self.n_commit)
        self.has_commit = self.has_commit | active


__all__ = [
    "PLANT_BAND_MIN",
    "PLANT_BAND_MAX",
    "FLIGHT_HL_MIN",
    "PLANT_EDGE_SIGMA",
    "PLANT_EDGE_SIGMA_START",
    "SHAPING_BAND_SIGMA",
    "V_CMD_MATCH_SIGMA",
    "FLIGHT_MATCH_SIGMA",
    "H_MATCH_SIGMA",
    "H_MARGIN_NOMINAL",
    "H_MARGIN_MIN",
    "H_MARGIN_MAX",
    "APEX_ALIGN_SIGMA",
    "APPROACH_STRIDE_X",
    "STRIDE_CMD_MAX",
    "STRIDE_ACH_MIN",
    "STRIDE_ACH_MAX",
    "VX_CMD_MIN",
    "VX_CMD_MAX",
    "peak_z",
    "flight_actual",
    "apex_from_plant",
    "invert_apex",
    "d_star",
    "flight_for_standoff",
    "apex_trim",
    "trim_feasible",
    "in_plant_band",
    "plant_band_score",
    "band_proximity",
    "stride_length",
    "cmd_from_stride",
    "expected_actual_speed",
    "feasible_stride_count",
    "lattice_from_remaining",
    "StrideTracker",
]
