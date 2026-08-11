# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the PPO high-level RunJumpHL policy.

Scaling rule used throughout: Isaac Lab multiplies every term by ``step_dt``
(0.02 s), so a weight of ``W`` on a once-per-event indicator is worth ``0.02*W``
of return. Event terms must stay comparable to the dense progress backbone
integrated over an episode, otherwise PPO cannot separate them from the entropy
bonus on the corresponding action channel.

Terms that gate a repeatable decision (latching, planting, lattice tracking) are
*signed* — ``2*exp(-e^2) - 1`` — so a badly timed event costs as much as a well
timed one earns. That removes the incentive to trigger the event repeatedly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

from humanoid_run_jump.tasks.manager_based.jump.mdp.gait import (
    heading_forward,
    resolve_ankle_body_ids,
)
from humanoid_run_jump.tasks.manager_based.run_jump.mdp.hl_stride import (
    FLIGHT_MATCH_SIGMA,
    H_MARGIN_NOMINAL,
    H_MATCH_SIGMA,
    PLANT_BAND_MAX,
    PLANT_BAND_MIN,
    PLANT_EDGE_SIGMA,
    SHAPING_BAND_SIGMA,
    V_CMD_MATCH_SIGMA,
    band_proximity,
    d_star,
    expected_actual_speed,
    flight_for_standoff,
    in_plant_band,
    lattice_from_remaining,
    plant_band_score,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

MODE_RUN = 0
MODE_JUMP = 1

# Tolerance on the measured root-vs-obstacle-centre offset at apex (m).
APEX_OVER_TOL_M = 0.30
# Corridor half width used to normalize the lateral penalty.
CORRIDOR_HALF_WIDTH_M = 2.5


def _signed(err: torch.Tensor) -> torch.Tensor:
    """Map a normalized error to ``[-1, 1]`` so bad events are punished."""
    return 2.0 * torch.exp(-err * err) - 1.0


def _mode(env: ManagerBasedRLEnv) -> torch.Tensor:
    return getattr(env, "hl_mode", torch.zeros(env.num_envs, dtype=torch.long, device=env.device))


def _vx_cmd(env: ManagerBasedRLEnv) -> torch.Tensor:
    """HL commanded forward speed (canvas PK_X is indexed by commanded vx)."""
    hl = getattr(env, "last_hl_action", None)
    if hl is None:
        return torch.full((env.num_envs,), 1.5, device=env.device)
    return hl[:, 1]


def _flight_cand(env: ManagerBasedRLEnv) -> torch.Tensor:
    f = getattr(env, "hl_flight_cand", None)
    if f is None:
        return torch.full((env.num_envs,), 0.85, device=env.device)
    return f


def _h_cand(env: ManagerBasedRLEnv) -> torch.Tensor:
    h = getattr(env, "hl_h_cand", None)
    if h is None:
        return getattr(env, "next_h", torch.zeros(env.num_envs, device=env.device)) + H_MARGIN_NOMINAL
    return h


def _plant_sigma(env: ManagerBasedRLEnv, fallback: float) -> float:
    """Curriculum-scheduled band falloff width, wide early and tight late."""
    return float(getattr(env, "plant_edge_sigma", fallback))


def _h_score(h_cmd: torch.Tensor, h_req: torch.Tensor, sigma: float) -> torch.Tensor:
    """One-sided clearance score: commanding *more* height than needed is free.

    A two-sided kernel fought the clear/hit rewards, which pay for extra margin.
    """
    deficit = (h_req - h_cmd).clamp(min=0.0)
    return torch.exp(-((deficit / max(sigma, 1e-3)) ** 2))


def _lattice_targets(
    env: ManagerBasedRLEnv, d_right: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(d_target, L_req, v_star)`` from the committed stride plan.

    Reads the plant-referenced gap the env maintains; falls back to the live ankle
    standoff only when the env has not published one yet.
    """
    d_tgt = getattr(env, "lattice_d_target", None)
    if d_tgt is None:
        d_tgt = d_star(env.next_t, _flight_cand(env), _vx_cmd(env))
    remaining = getattr(env, "lattice_remaining", None)
    if remaining is None:
        remaining = d_right - d_tgt
    tracker = getattr(env, "stride_tracker", None)
    if tracker is None:
        _, L_req, v_star = lattice_from_remaining(remaining, torch.full_like(d_right, 1.70))
        return d_tgt, L_req, v_star
    _, L_req, v_star = lattice_from_remaining(remaining, tracker.L_ema, n_commit=tracker.n_commit)
    return d_tgt, L_req, v_star


def _right_standoff(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    _, right_id = resolve_ankle_body_ids(env, asset_cfg.name)
    right_s = asset.data.body_pos_w[:, right_id, 0] - env.scene.env_origins[:, 0]
    return (env.next_front_s - right_s).clamp(min=0.0)


# ---------------------------------------------------------------------------
# Dense backbone
# ---------------------------------------------------------------------------


def course_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Increment of the monotone furthest-distance potential (m this step).

    Using ``Δ max(s)`` rather than forward velocity makes the term a potential:
    retreating and re-covering ground pays exactly zero, so there is no reward
    cycle to exploit, and the payout per metre is independent of speed.
    """
    delta = getattr(env, "progress_delta", None)
    if delta is None:
        return torch.zeros(env.num_envs, device=env.device)
    return delta


def stall_cost(
    env: ManagerBasedRLEnv,
    vx_min: float = 0.6,
    grace_s: float = 2.0,
) -> torch.Tensor:
    """Cost applied only while the robot is not advancing.

    Needed so that waiting in front of an obstacle can never beat attempting it:
    a terminal penalty ``P`` is only safe from the "stall to discount the
    penalty" exploit while ``stall_cost > P * (1 - gamma)``.

    ``grace_s`` skips the cost for the first seconds after reset so a standing
    start can accelerate under the frozen run policy without immediate penalty.
    """
    ema = getattr(env, "vx_ema", None)
    if ema is None:
        return torch.zeros(env.num_envs, device=env.device)
    stalled = (ema < vx_min) & (_mode(env) == MODE_RUN)
    ep_len = getattr(env, "episode_length_buf", None)
    if ep_len is not None and grace_s > 0.0:
        dt = float(getattr(env, "step_dt", 0.02) or 0.02)
        grace_steps = max(0, int(grace_s / max(dt, 1e-3)))
        stalled = stalled & (ep_len >= grace_steps)
    return stalled.float()


def lateral_offset(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Normalized squared lateral offset, so the cost at the corridor edge is 1."""
    asset = env.scene[asset_cfg.name]
    y = asset.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return (y / CORRIDOR_HALF_WIDTH_M) ** 2


def heading_align(
    env: ManagerBasedRLEnv,
    std: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    fwd = heading_forward(asset.data.root_quat_w)
    err = torch.atan2(fwd[:, 1], fwd[:, 0])
    return torch.exp(-(err / std) ** 2)


# ---------------------------------------------------------------------------
# Approach shaping
# ---------------------------------------------------------------------------


def lattice_cmd_speed(
    env: ManagerBasedRLEnv,
    std: float = V_CMD_MATCH_SIGMA,
    approach_horizon: float = 8.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track ``v_star`` in *command* space: score ``|vx_cmd - v_star|`` directly.

    The measured-speed version (:func:`lattice_speed`) is only observable 1.5–2 s
    after the command changes because the frozen run policy accelerates slowly,
    which is far outside PPO's effective credit horizon here (~0.6 s). Scoring the
    command itself removes the lag entirely: the action taken this step is the
    quantity being graded.
    """
    d_right = _right_standoff(env, asset_cfg)
    in_zone = (
        env.has_next_obstacle
        & (_mode(env) == MODE_RUN)
        & (d_right < approach_horizon)
        & (d_right > PLANT_BAND_MIN)
    )
    _, _, v_star = _lattice_targets(env, d_right)
    err = (_vx_cmd(env) - v_star) / max(std, 1e-3)
    return torch.where(in_zone, torch.exp(-err * err), torch.zeros_like(err))


def lattice_speed(
    env: ManagerBasedRLEnv,
    std: float = 0.30,
    approach_horizon: float = 8.0,
    min_closing_speed: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track the deadbeat approach speed that lands a right plant on ``d*``.

    ``d*`` tracks the policy's current flight candidate at the commanded vx, so
    the lattice retargets when the policy changes ``a_f``.
    """
    asset = env.scene[asset_cfg.name]
    v_act = asset.data.root_lin_vel_b[:, 0]
    d_right = _right_standoff(env, asset_cfg)
    closing = asset.data.root_lin_vel_w[:, 0] > min_closing_speed
    in_zone = (
        env.has_next_obstacle
        & (_mode(env) == MODE_RUN)
        & closing
        & (d_right < approach_horizon)
        & (d_right > PLANT_BAND_MIN)
    )

    _, _, v_star = _lattice_targets(env, d_right)
    err = (v_act - expected_actual_speed(v_star)) / std
    rew = torch.exp(-err * err)
    return torch.where(in_zone, rew, torch.zeros_like(rew))


def lattice_phase_gain(
    env: ManagerBasedRLEnv,
    max_distance: float = 8.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Per-plant reduction in stride shortfall ``|L_req - L_ema|`` (m).

    ``L_req`` is the constant stride the committed plan needs, so the shortfall is
    exactly how much longer or shorter the robot's stride has to become. Paying the
    *decrease* between consecutive right plants delivers credit one stride (~0.6 s)
    after the command that caused it, instead of at the latch several seconds later,
    and being a potential difference it cannot be farmed by oscillating.
    """
    if not hasattr(env, "stride_tracker"):
        return torch.zeros(env.num_envs, device=env.device)
    tracker = env.stride_tracker
    plant = tracker.plant_event
    d_right = _right_standoff(env, asset_cfg)
    active = plant & env.has_next_obstacle & (_mode(env) == MODE_RUN) & (d_right < max_distance)

    if not hasattr(env, "_stride_err_prev"):
        env._stride_err_prev = torch.zeros(env.num_envs, device=env.device)
        env._has_stride_err = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not active.any():
        return torch.zeros(env.num_envs, device=env.device)

    _, L_req, _ = _lattice_targets(env, d_right)
    stride_err = (L_req - tracker.L_ema).abs()

    rew = torch.zeros(env.num_envs, device=env.device)
    ids = active.nonzero(as_tuple=False).squeeze(-1)
    scored = ids[env._has_stride_err[ids]]
    if scored.numel() > 0:
        rew[scored] = env._stride_err_prev[scored] - stride_err[scored]
    env._stride_err_prev[ids] = stride_err[ids]
    env._has_stride_err[ids] = True
    return rew


def plant_on_lattice(
    env: ManagerBasedRLEnv,
    std: float = 0.15,
    max_distance: float = 8.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Signed bonus when a right plant lands where the committed plan predicted.

    Restricted to the approach zone, where a plan exists at all. Outside it the
    solver falls back to ``L_req ≈ L_ema``, which makes the prediction
    self-fulfilling and the reward free for simply holding speed.
    """
    if not hasattr(env, "stride_tracker"):
        return torch.zeros(env.num_envs, device=env.device)
    tracker = env.stride_tracker
    plant = tracker.plant_event
    if not plant.any():
        return torch.zeros(env.num_envs, device=env.device)

    asset = env.scene[asset_cfg.name]
    origins = env.scene.env_origins
    _, right_id = resolve_ankle_body_ids(env, asset_cfg.name)
    right_s = asset.data.body_pos_w[:, right_id, 0] - origins[:, 0]

    d_right = (env.next_front_s - right_s).clamp(min=0.0)
    active = plant & env.has_next_obstacle & (_mode(env) == MODE_RUN) & (d_right < max_distance)

    _, L_req, _ = _lattice_targets(env, d_right)

    if not hasattr(env, "_pred_plant_s"):
        env._pred_plant_s = torch.zeros(env.num_envs, device=env.device)
        env._has_pred_plant = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    rew = torch.zeros(env.num_envs, device=env.device)
    if active.any():
        ids = active.nonzero(as_tuple=False).squeeze(-1)
        scored = ids[env._has_pred_plant[ids]]
        if scored.numel() > 0:
            rew[scored] = _signed((right_s[scored] - env._pred_plant_s[scored]).abs() / std)
        env._pred_plant_s[ids] = right_s[ids] + L_req[ids]
        env._has_pred_plant[ids] = True
    return rew


# ---------------------------------------------------------------------------
# Jump decision
# ---------------------------------------------------------------------------


def plant_in_band(
    env: ManagerBasedRLEnv,
    sigma: float = PLANT_EDGE_SIGMA,
) -> torch.Tensor:
    """Signed flat-top score: +1 when latch standoff ∈ [0.90, 1.20], once per obstacle."""
    just = getattr(env, "hl_just_latched", None)
    if just is None or not just.any():
        return torch.zeros(env.num_envs, device=env.device)
    scored = getattr(env, "plant_band_scored", None)
    if scored is None:
        env.plant_band_scored = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        scored = env.plant_band_scored
    active = just & (~scored)
    if not active.any():
        return torch.zeros(env.num_envs, device=env.device)
    rew = plant_band_score(env.hl_d_latch, sigma=_plant_sigma(env, sigma))
    env.plant_band_scored |= active
    return torch.where(active, rew, torch.zeros_like(rew))


def jump_cmd_match(
    env: ManagerBasedRLEnv,
    flight_sigma: float = FLIGHT_MATCH_SIGMA,
    h_sigma: float = H_MATCH_SIGMA,
) -> torch.Tensor:
    """Signed score for (h, flight) vs the canvas-required pair, once per in-band latch."""
    just = getattr(env, "hl_just_latched", None)
    if just is None or not just.any():
        return torch.zeros(env.num_envs, device=env.device)
    scored = getattr(env, "jump_cmd_scored", None)
    if scored is None:
        env.jump_cmd_scored = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        scored = env.jump_cmd_scored
    d = env.hl_d_latch
    active = just & (~scored) & in_plant_band(d)
    if not active.any():
        # Still mark out-of-band latches as scored so we don't re-fire later.
        env.jump_cmd_scored |= just
        return torch.zeros(env.num_envs, device=env.device)

    vx = getattr(env, "hl_vx_at_latch", _vx_cmd(env))
    f_cmd = getattr(env, "hl_flight_cmd", _flight_cand(env))
    h_cmd = getattr(env, "hl_h_cmd", _h_cand(env))
    f_req = flight_for_standoff(d, env.next_t, vx)
    h_req = env.next_h + H_MARGIN_NOMINAL
    q = torch.exp(-((f_cmd - f_req) / max(flight_sigma, 1e-3)) ** 2) * _h_score(h_cmd, h_req, h_sigma)
    # Map [0, 1] → [-1, 1] so a bad command still costs.
    rew = 2.0 * q - 1.0
    env.jump_cmd_scored |= just
    return torch.where(active, rew, torch.zeros_like(rew))


def jump_cmd_shaping(
    env: ManagerBasedRLEnv,
    approach_horizon: float = 8.0,
    flight_sigma: float = FLIGHT_MATCH_SIGMA,
    h_sigma: float = H_MATCH_SIGMA,
    band_sigma: float = SHAPING_BAND_SIGMA,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense unsigned score of the current ``(a_h, a_f)`` weighted by band proximity.

    The gate is a wide gaussian on how far the predicted plant sits outside the
    band rather than a hard in-band test. A hard test paid nothing until the robot
    was *already* arriving correctly, so there was no gradient pointing at the
    band; the proximity kernel makes closing the last 0.3 m strictly profitable.
    """
    if not env.has_next_obstacle.any():
        return torch.zeros(env.num_envs, device=env.device)
    d_right = _right_standoff(env, asset_cfg)

    vx = _vx_cmd(env)
    f_cand = _flight_cand(env)
    # Predicted next-plant standoff after one stride of L_req.
    _, L_req, _ = _lattice_targets(env, d_right)
    # If already inside / near the band, score against the *current* standoff;
    # otherwise against the lattice's intended plant.
    d_pred = torch.where(d_right <= PLANT_BAND_MAX + 0.30, d_right, (d_right - L_req).clamp(min=0.0))
    active = env.has_next_obstacle & (_mode(env) == MODE_RUN) & (d_right < approach_horizon)
    if not active.any():
        return torch.zeros(env.num_envs, device=env.device)

    f_req = flight_for_standoff(d_pred.clamp(min=PLANT_BAND_MIN, max=PLANT_BAND_MAX), env.next_t, vx)
    h_req = env.next_h + H_MARGIN_NOMINAL
    q = torch.exp(-((f_cand - f_req) / max(flight_sigma, 1e-3)) ** 2) * _h_score(
        _h_cand(env), h_req, h_sigma
    )
    q = q * band_proximity(d_pred, sigma=band_sigma)
    return torch.where(active, q, torch.zeros_like(q))


def latch_quality(env: ManagerBasedRLEnv, sigma: float = 0.20) -> torch.Tensor:
    """Signed plant/command consistency at latch: ``|d_latch - d*(f_cmd, vx)|``.

    Once per obstacle. Complements ``plant_in_band`` (absolute band) with a
    peaked score against the grid target for the *committed* flight.
    """
    just = getattr(env, "hl_just_latched", None)
    if just is None or not just.any():
        return torch.zeros(env.num_envs, device=env.device)
    scored = getattr(env, "latch_scored", torch.zeros_like(just))
    active = just & (~scored)
    if not active.any():
        return torch.zeros(env.num_envs, device=env.device)

    vx = getattr(env, "hl_vx_at_latch", _vx_cmd(env))
    f_cmd = getattr(env, "hl_flight_cmd", _flight_cand(env))
    d_tgt = d_star(env.next_t, f_cmd, vx)
    rew = _signed((env.hl_d_latch - d_tgt) / max(sigma, 1e-3))
    env.latch_scored |= active
    return torch.where(active, rew, torch.zeros_like(rew))


def apex_over_obstacle(env: ManagerBasedRLEnv, tol: float = APEX_OVER_TOL_M) -> torch.Tensor:
    """Signed score for the *measured* apex placement, paid once per jump.

    Uses root s at peak sole height against the obstacle centre (canvas frame).
    """
    event = getattr(env, "apex_event", None)
    if event is None or not event.any():
        return torch.zeros(env.num_envs, device=env.device)
    rew = _signed(env.apex_err / max(tol, 1e-3))
    return torch.where(event, rew, torch.zeros_like(rew))


def no_jump_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fires once when an obstacle too tall to step over is reached without gating."""
    event = getattr(env, "no_jump_event", None)
    if event is None:
        return torch.zeros(env.num_envs, device=env.device)
    return event.float()


def land_cmd_shock(env: ManagerBasedRLEnv, grace_steps: int = 25) -> torch.Tensor:
    """Squared velocity-command jump across the flight and touchdown window.

    The run policy resumes at touchdown with whatever velocity command the HL
    last wrote, so a large mismatch against the pre-jump command is a direct
    cause of landing stumbles.
    """
    hl = getattr(env, "last_hl_action", None)
    v_latch = getattr(env, "hl_vx_at_latch", None)
    if hl is None or v_latch is None:
        return torch.zeros(env.num_envs, device=env.device)
    mode = _mode(env)
    mode_steps = getattr(env, "hl_mode_steps", torch.zeros(env.num_envs, device=env.device))
    window = (v_latch > 0.0) & ((mode == MODE_JUMP) | (mode_steps < grace_steps))
    diff = hl[:, 1] - v_latch
    return torch.where(window, diff * diff, torch.zeros_like(diff))


def hl_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Squared change of the HL action in tanh space (all channels comparable)."""
    hl = getattr(env, "hl_action_tanh", None)
    if hl is None:
        return torch.zeros(env.num_envs, device=env.device)
    prev = getattr(env, "_prev_hl_tanh", None)
    if prev is None:
        env._prev_hl_tanh = hl.clone()
        return torch.zeros(env.num_envs, device=env.device)
    diff = hl - prev
    env._prev_hl_tanh = hl.clone()
    return torch.sum(diff * diff, dim=-1)


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


def obstacle_cleared(env: ManagerBasedRLEnv) -> torch.Tensor:
    just = getattr(env, "just_cleared", None)
    if just is None:
        return torch.zeros(env.num_envs, device=env.device)
    return just.float()


def clear_and_stable(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Paid one second after a clear if the robot is still upright and running."""
    event = getattr(env, "stable_clear_event", None)
    if event is None:
        return torch.zeros(env.num_envs, device=env.device)
    return event.float()


def obstacle_hit(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Geometric crash indicator, read straight from the termination manager.

    Terminations are computed before rewards, so the crash term's own output is
    the authoritative source. Latching it inside ``step_course_state`` never
    worked because that runs earlier in the termination list.
    """
    manager = getattr(env, "termination_manager", None)
    if manager is None or "obstacle_crash" not in manager.active_terms:
        return torch.zeros(env.num_envs, device=env.device)
    hit = manager.get_term("obstacle_crash")
    if hasattr(env, "just_hit"):
        env.just_hit[:] = hit
        env.hit_count += hit.long()
    return hit.float()


def finish_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    finished = getattr(env, "course_finished", None)
    if finished is None:
        return torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_finish_paid"):
        env._finish_paid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    pay = finished & (~env._finish_paid)
    env._finish_paid |= finished
    return pay.float()


def fail_terminal(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty on every genuine failure, not just geometric crashes.

    ``terminated`` excludes ``time_out``-flagged terms, so reaching the goal
    (``course_finished``) and running out of episode time are both exempt.
    """
    manager = getattr(env, "termination_manager", None)
    if manager is None:
        return torch.zeros(env.num_envs, device=env.device)
    return manager.terminated.float()
