# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event / reset helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _yaw_from_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Extract Z-yaw (rad) from a batch of wxyz quaternions."""
    w, x, y, z = quat_wxyz.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _apply_yaw_wxyz(quat_wxyz: torch.Tensor, dyaw: torch.Tensor) -> torch.Tensor:
    """Left-multiply ``quat_wxyz`` by a pure Z yaw of ``dyaw`` (radians).

    Returns a **new** ``(N, 4)`` wxyz tensor. Must not write through ``unbind``
    views into ``quat_wxyz`` — in-place assignment corrupts the last component
    because ``w0`` aliases ``quat[:, 0]`` / ``pose[:, 3]``.
    """
    half = dyaw * 0.5
    qw = torch.cos(half)
    qz = torch.sin(half)
    w0, x0, y0, z0 = quat_wxyz.unbind(-1)
    return torch.stack(
        [
            qw * w0 - qz * z0,
            qw * x0 + qz * y0,
            qw * y0 - qz * x0,
            qw * z0 + qz * w0,
        ],
        dim=-1,
    )


def reset_root_and_joints(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    joint_position_range: tuple[float, float] = (0.9, 1.1),
):
    """Reset root pose / velocity and joint state with light randomization."""
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)

    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]

    if pose_range is None:
        pose_range = {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.2, 0.2)}
    ranges = pose_range
    root_state[:, 0] += sample_uniform(*ranges.get("x", (0.0, 0.0)), n, env.device)
    root_state[:, 1] += sample_uniform(*ranges.get("y", (0.0, 0.0)), n, env.device)
    yaw = sample_uniform(*ranges.get("yaw", (0.0, 0.0)), n, env.device)
    # Multiply default * yaw about Z (wxyz); assign a fresh quat (no in-place unbind).
    root_state[:, 3:7] = _apply_yaw_wxyz(root_state[:, 3:7], yaw)

    if velocity_range is None:
        velocity_range = {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.1, 0.1)}
    root_state[:, 7] = sample_uniform(*velocity_range.get("x", (0.0, 0.0)), n, env.device)
    root_state[:, 8] = sample_uniform(*velocity_range.get("y", (0.0, 0.0)), n, env.device)
    root_state[:, 9] = sample_uniform(*velocity_range.get("z", (0.0, 0.0)), n, env.device)
    root_state[:, 10:13] = 0.0

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    scale = sample_uniform(*joint_position_range, (n, joint_pos.shape[1]), env.device)
    joint_pos *= scale

    asset.write_root_pose_to_sim(root_state[:, :7], env_ids)
    asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


def reset_from_reference(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    rsi_prob: float = 0.5,
    phase_range: tuple[float, float] = (0.15, 0.55),
    max_root_height: float = 0.95,
    pose_range: dict[str, tuple[float, float]] | None = None,
    max_attempts: int = 8,
    clip_name_substr: str = "jump",
    max_horiz_speed: float = 2.5,
):
    """Overwrite a fraction of resets with Reference State Initialization (RSI).

    Samples frames from ``env._motion_lib`` biased toward the crouch / takeoff
    portion of each clip (``phase_range``) and rejects frames whose root height
    exceeds ``max_root_height`` so the subsequent jump-command resample still
    places a sensible takeoff line ahead of the robot.

    When the motion pack is mixed (run+jump), ``clip_name_substr=\"jump\"`` keeps
    RSI on jump clips only.

    ``max_horiz_speed`` caps reference root XY velocity so RSI run-in cannot
    instantly overshoot a short takeoff line after command resample.

    Intended to run *after* :func:`reset_root_and_joints` so standing resets
    remain for the complementary fraction.
    """
    motion_lib = getattr(env, "_motion_lib", None)
    if motion_lib is None or len(env_ids) == 0 or rsi_prob <= 0.0:
        return

    # Bernoulli mask — only RSI this fraction of the resetting envs.
    mask = torch.rand(len(env_ids), device=env.device) < rsi_prob
    if not torch.any(mask):
        return
    rsi_ids = env_ids[mask]
    n = len(rsi_ids)

    jmap = getattr(env, "joint_order_map", None)
    if jmap is None:
        from humanoid_run_jump.robots.joint_order import JointOrderMap

        asset = env.scene[asset_cfg.name]
        jmap = JointOrderMap(list(asset.data.joint_names), device=env.device)

    # Rejection-sample frames in the crouch/takeoff phase with bounded height.
    phase_lo, phase_hi = phase_range
    motion_ids = motion_lib.sample_motions(n, name_substr=clip_name_substr or None)
    lengths = motion_lib.motion_lengths[motion_ids.cpu()].numpy().astype(np.float64)
    root_z = torch.full((n,), float("inf"), device=env.device)
    frame: dict[str, torch.Tensor] | None = None

    for _ in range(max_attempts):
        phases = np.random.uniform(phase_lo, phase_hi, size=n)
        # Leave ≥1 frame headroom; clamp into clip.
        max_t = np.maximum((lengths - 2) * motion_lib.dt, 0.0)
        cand_times = np.clip(phases * lengths * motion_lib.dt, 0.0, max_t)
        cand = motion_lib.get_frame(motion_ids, cand_times)
        cand_z = cand["root_pos"][:, 2]
        replace = root_z > max_root_height
        if frame is None:
            frame = {k: v.clone() for k, v in cand.items()}
            root_z = cand_z.clone()
        else:
            for k in frame:
                frame[k][replace] = cand[k][replace]
            root_z = torch.where(replace, cand_z, root_z)
        if bool((root_z <= max_root_height).all()):
            break

    assert frame is not None
    asset = env.scene[asset_cfg.name]

    # Root pose: XY from env origin (+ light noise), Z/quat from reference.
    root_pos = frame["root_pos"].clone()
    root_pos[:, :2] = 0.0
    root_pos[:, :3] += env.scene.env_origins[rsi_ids]
    if pose_range is None:
        pose_range = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.1, 0.1)}
    root_pos[:, 0] += sample_uniform(*pose_range.get("x", (0.0, 0.0)), n, env.device)
    root_pos[:, 1] += sample_uniform(*pose_range.get("y", (0.0, 0.0)), n, env.device)

    yaw = sample_uniform(*pose_range.get("yaw", (0.0, 0.0)), n, env.device)
    root_quat = _apply_yaw_wxyz(frame["root_rot"], yaw)

    root_pose = torch.cat([root_pos, root_quat], dim=-1)
    root_vel = torch.cat([frame["root_lin_vel"], frame["root_ang_vel"]], dim=-1)
    if max_horiz_speed > 0.0:
        horiz = root_vel[:, :2]
        speed = torch.linalg.norm(horiz, dim=-1).clamp(min=1e-6)
        scale = (max_horiz_speed / speed).clamp(max=1.0)
        root_vel[:, :2] = horiz * scale.unsqueeze(-1)

    joint_pos = jmap.to_sim(frame["dof_pos"])
    joint_vel = jmap.to_sim(frame["dof_vel"])

    asset.write_root_pose_to_sim(root_pose, rsi_ids)
    asset.write_root_velocity_to_sim(root_vel, rsi_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, None, rsi_ids)


def reset_from_runin_bank(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    bank_path: str = "motions/packaged/runin_states.pt",
    bank_prob: float = 1.0,
    pose_range: dict[str, tuple[float, float]] | None = None,
    min_root_z: float = 0.55,
    max_tilt_rad: float = 0.45,
    max_lat_vel: float = 0.45,
    min_fwd_vel: float = 1.2,
    max_fwd_vel: float = 3.0,
    max_filter_tries: int = 24,
):
    """Overwrite resets with precomputed frozen-run-actor hand-off states.

    Samples from ``bank_path`` (built by ``scripts/build_runin_bank.py``). Bank
    stores root pose with XY relative to env origin, root velocity, and dofs in
    ``G1_JOINT_NAMES`` order. Intended to run *after* :func:`reset_root_and_joints`
    so the complementary ``1 - bank_prob`` fraction keeps standing resets.

    Filters bank samples for upright height/tilt, limited lateral velocity, and
    forward-speed compatibility with the HL cruise band so episodes are less
    likely to die before the first obstacle.

    Canonicalizes heading to world +x (plus ``pose_range`` yaw noise). After
    reset, :class:`~humanoid_run_jump.tasks.manager_based.jump.mdp.commands.JumpCommand`
    freezes flight progress along the robot's front at resample — jump direction
    is the reset heading, not a fixed world axis independent of facing.
    """
    if len(env_ids) == 0 or bank_prob <= 0.0:
        return

    bank = getattr(env, "_runin_bank", None)
    if bank is None:
        from pathlib import Path

        path = Path(bank_path)
        if not path.is_absolute():
            # events.py → .../jump/mdp → task-humanoid-run-jump/ is parents[7]
            project_root = Path(__file__).resolve().parents[7]
            cand = project_root / path
            path = cand if cand.exists() else Path.cwd() / bank_path
        if not path.exists():
            print(f"[reset_from_runin_bank] WARNING: bank not found: {path}")
            return
        payload = torch.load(path, map_location=env.device, weights_only=False)
        bank = {
            "root_pose": payload["root_pose"].to(env.device),
            "root_vel": payload["root_vel"].to(env.device),
            "dof_pos": payload["dof_pos"].to(env.device),
            "dof_vel": payload["dof_vel"].to(env.device),
        }
        # Precompute a stable subset once so resets stay cheap.
        root_pose_all = bank["root_pose"]
        root_vel_all = bank["root_vel"]
        z_ok = root_pose_all[:, 2] >= float(min_root_z)
        # Approx tilt from quaternion: projected gravity z ≈ 1 − 2(qx²+qy²)
        qx = root_pose_all[:, 4]
        qy = root_pose_all[:, 5]
        cos_tilt = (1.0 - 2.0 * (qx * qx + qy * qy)).clamp(-1.0, 1.0)
        tilt_ok = torch.acos(cos_tilt) <= float(max_tilt_rad)
        lat_ok = root_vel_all[:, 1].abs() <= float(max_lat_vel)
        fwd_ok = (root_vel_all[:, 0] >= float(min_fwd_vel)) & (
            root_vel_all[:, 0] <= float(max_fwd_vel)
        )
        stable = z_ok & tilt_ok & lat_ok & fwd_ok
        stable_idx = stable.nonzero(as_tuple=False).flatten()
        n_all = int(root_pose_all.shape[0])
        if len(stable_idx) < max(32, n_all // 20):
            # Fall back to height+tilt only if the full filter is too strict.
            stable_idx = (z_ok & tilt_ok).nonzero(as_tuple=False).flatten()
        if len(stable_idx) == 0:
            stable_idx = torch.arange(n_all, device=env.device)
        bank["stable_idx"] = stable_idx
        env._runin_bank = bank
        print(
            f"[reset_from_runin_bank] loaded {bank['root_pose'].shape[0]} states "
            f"({len(stable_idx)} stable) from {path}"
        )

    n_bank = int(bank["root_pose"].shape[0])
    if n_bank == 0:
        return

    mask = torch.rand(len(env_ids), device=env.device) < bank_prob
    if not torch.any(mask):
        return
    sel_ids = env_ids[mask]
    n = len(sel_ids)

    jmap = getattr(env, "joint_order_map", None)
    if jmap is None:
        from humanoid_run_jump.robots.joint_order import JointOrderMap

        asset = env.scene[asset_cfg.name]
        jmap = JointOrderMap(list(asset.data.joint_names), device=env.device)

    stable_idx = bank.get("stable_idx")
    if stable_idx is None or len(stable_idx) == 0:
        pick = torch.randint(0, n_bank, (n,), device=env.device)
    else:
        # Sample with replacement from the stable pool.
        choice = torch.randint(0, len(stable_idx), (n,), device=env.device)
        pick = stable_idx[choice]
        # Light online re-filter for the chosen batch (reject + redraw a few times).
        for _ in range(int(max_filter_tries)):
            rp = bank["root_pose"][pick]
            rv = bank["root_vel"][pick]
            qx = rp[:, 4]
            qy = rp[:, 5]
            cos_tilt = (1.0 - 2.0 * (qx * qx + qy * qy)).clamp(-1.0, 1.0)
            bad = (
                (rp[:, 2] < float(min_root_z))
                | (torch.acos(cos_tilt) > float(max_tilt_rad))
                | (rv[:, 1].abs() > float(max_lat_vel))
                | (rv[:, 0] < float(min_fwd_vel))
                | (rv[:, 0] > float(max_fwd_vel))
            )
            if not bool(bad.any().item()):
                break
            redraw = torch.randint(0, len(stable_idx), (int(bad.sum().item()),), device=env.device)
            pick = pick.clone()
            pick[bad] = stable_idx[redraw]
    idx = pick
    root_pose = bank["root_pose"][idx].clone()
    root_vel = bank["root_vel"][idx].clone()
    dof_pos = jmap.to_sim(bank["dof_pos"][idx].clone())
    dof_vel = jmap.to_sim(bank["dof_vel"][idx].clone())

    # Discard bank XY drift (run actor may have traveled tens of meters) and
    # place each robot at its own env origin (+ light pose_range noise).
    if pose_range is None:
        pose_range = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.05, 0.05)}
    root_pose[:, :2] = env.scene.env_origins[sel_ids, :2]
    root_pose[:, 0] += sample_uniform(*pose_range.get("x", (0.0, 0.0)), n, env.device)
    root_pose[:, 1] += sample_uniform(*pose_range.get("y", (0.0, 0.0)), n, env.device)

    # Canonicalize heading to +x (plus yaw noise). Bank yaw can be arbitrary
    # because the frozen run actor drifts; rotate by the delta, not by noise alone.
    bank_yaw = _yaw_from_wxyz(root_pose[:, 3:7])
    dyaw = sample_uniform(*pose_range.get("yaw", (0.0, 0.0)), n, env.device) - bank_yaw
    root_pose[:, 3:7] = _apply_yaw_wxyz(root_pose[:, 3:7], dyaw)

    # Bank velocities are world-frame; rotate XY lin/ang vel by the same dyaw
    # so heading and velocity stay consistent after re-anchoring.
    c = torch.cos(dyaw)
    s = torch.sin(dyaw)
    vx, vy = root_vel[:, 0].clone(), root_vel[:, 1].clone()
    root_vel[:, 0] = c * vx - s * vy
    root_vel[:, 1] = s * vx + c * vy
    wx, wy = root_vel[:, 3].clone(), root_vel[:, 4].clone()
    root_vel[:, 3] = c * wx - s * wy
    root_vel[:, 4] = s * wx + c * wy

    asset = env.scene[asset_cfg.name]
    asset.write_root_pose_to_sim(root_pose, sel_ids)
    asset.write_root_velocity_to_sim(root_vel, sel_ids)
    asset.write_joint_state_to_sim(dof_pos, dof_vel, None, sel_ids)


def reset_from_runin_bank_envhub(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    preset_cfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pose_range: dict[str, tuple[float, float]] | None = None,
):
    """Deterministic EnvHub run-in reset for fair tournament evaluation.

    Loads the organizer bank from ``preset_cfg.bank_path`` (EnvHub bundle) and
    selects rows with ``preset_cfg.gen_runin_index(env_ids)``. Unlike
    :func:`reset_from_runin_bank`, there is no random sampling and a missing
    bank raises ``FileNotFoundError`` instead of falling back to standing.
    """
    if len(env_ids) == 0:
        return

    from pathlib import Path

    bank_path = getattr(preset_cfg, "bank_path", None)
    if not bank_path:
        raise FileNotFoundError(
            "EnvHub preset is missing bank_path; expected organizer runin_states.pt "
            "inside the EnvHub course bundle."
        )
    path = Path(bank_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"EnvHub run-in bank not found: {path}. "
            "Tournament eval requires the organizer bank shipped with the EnvHub "
            "bundle (do not substitute a miner-local motions/packaged bank)."
        )

    cache_key = f"_runin_bank_envhub:{path.resolve()}"
    bank = getattr(env, cache_key, None)
    if bank is None:
        payload = torch.load(path, map_location=env.device, weights_only=False)
        for key in ("root_pose", "root_vel", "dof_pos", "dof_vel"):
            if key not in payload:
                raise KeyError(f"EnvHub run-in bank {path} missing key {key!r}")
        bank = {
            "root_pose": payload["root_pose"].to(env.device),
            "root_vel": payload["root_vel"].to(env.device),
            "dof_pos": payload["dof_pos"].to(env.device),
            "dof_vel": payload["dof_vel"].to(env.device),
            "path": str(path.resolve()),
        }
        setattr(env, cache_key, bank)
        # Also stash under the legacy attr so other helpers can inspect size.
        env._runin_bank_envhub = bank
        print(
            f"[reset_from_runin_bank_envhub] loaded {bank['root_pose'].shape[0]} "
            f"states from {path}"
        )

    n_bank = int(bank["root_pose"].shape[0])
    if n_bank == 0:
        raise RuntimeError(f"EnvHub run-in bank is empty: {path}")

    ids = env_ids.to(device=env.device, dtype=torch.long)
    n = len(ids)
    idx = preset_cfg.gen_runin_index(ids, device=env.device)
    if int(idx.min().item()) < 0 or int(idx.max().item()) >= n_bank:
        raise IndexError(
            f"EnvHub runin_index out of range for bank size {n_bank}: "
            f"min={int(idx.min().item())} max={int(idx.max().item())}"
        )

    jmap = getattr(env, "joint_order_map", None)
    if jmap is None:
        from humanoid_run_jump.robots.joint_order import JointOrderMap

        asset = env.scene[asset_cfg.name]
        jmap = JointOrderMap(list(asset.data.joint_names), device=env.device)

    root_pose = bank["root_pose"][idx].clone()
    root_vel = bank["root_vel"][idx].clone()
    dof_pos = jmap.to_sim(bank["dof_pos"][idx].clone())
    dof_vel = jmap.to_sim(bank["dof_vel"][idx].clone())

    if pose_range is None:
        pose_range = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    root_pose[:, :2] = env.scene.env_origins[ids, :2]
    root_pose[:, 0] += sample_uniform(*pose_range.get("x", (0.0, 0.0)), n, env.device)
    root_pose[:, 1] += sample_uniform(*pose_range.get("y", (0.0, 0.0)), n, env.device)

    bank_yaw = _yaw_from_wxyz(root_pose[:, 3:7])
    dyaw = sample_uniform(*pose_range.get("yaw", (0.0, 0.0)), n, env.device) - bank_yaw
    root_pose[:, 3:7] = _apply_yaw_wxyz(root_pose[:, 3:7], dyaw)

    c = torch.cos(dyaw)
    s = torch.sin(dyaw)
    vx, vy = root_vel[:, 0].clone(), root_vel[:, 1].clone()
    root_vel[:, 0] = c * vx - s * vy
    root_vel[:, 1] = s * vx + c * vy
    wx, wy = root_vel[:, 3].clone(), root_vel[:, 4].clone()
    root_vel[:, 3] = c * wx - s * wy
    root_vel[:, 4] = s * wx + c * wy

    asset = env.scene[asset_cfg.name]
    asset.write_root_pose_to_sim(root_pose, ids)
    asset.write_root_velocity_to_sim(root_vel, ids)
    asset.write_joint_state_to_sim(dof_pos, dof_vel, None, ids)


def sample_uniform(
    low: float, high: float, size, device
) -> torch.Tensor:
    if isinstance(size, int):
        size = (size,)
    return torch.empty(size, device=device).uniform_(low, high)
