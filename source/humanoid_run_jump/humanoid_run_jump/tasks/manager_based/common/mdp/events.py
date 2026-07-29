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
    # Apply yaw about Z to default quat (wxyz).
    half = yaw * 0.5
    qw = torch.cos(half)
    qz = torch.sin(half)
    q = root_state[:, 3:7]
    # Multiply default * yaw: (qw,0,0,qz) ⊗ q
    w0, x0, y0, z0 = q.unbind(-1)
    root_state[:, 3] = qw * w0 - qz * z0
    root_state[:, 4] = qw * x0 + qz * y0
    root_state[:, 5] = qw * y0 - qz * x0
    root_state[:, 6] = qw * z0 + qz * w0

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

    root_quat = frame["root_rot"].clone()
    yaw = sample_uniform(*pose_range.get("yaw", (0.0, 0.0)), n, env.device)
    half = yaw * 0.5
    qw = torch.cos(half)
    qz = torch.sin(half)
    w0, x0, y0, z0 = root_quat.unbind(-1)
    root_quat[:, 0] = qw * w0 - qz * z0
    root_quat[:, 1] = qw * x0 + qz * y0
    root_quat[:, 2] = qw * y0 - qz * x0
    root_quat[:, 3] = qw * z0 + qz * w0

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
):
    """Overwrite resets with precomputed frozen-run-actor hand-off states.

    Samples from ``bank_path`` (built by ``scripts/build_runin_bank.py``). Bank
    stores root pose with XY relative to env origin, root velocity, and dofs in
    ``G1_JOINT_NAMES`` order. Intended to run *after* :func:`reset_root_and_joints`
    so the complementary ``1 - bank_prob`` fraction keeps standing resets.
    """
    if len(env_ids) == 0 or bank_prob <= 0.0:
        return

    bank = getattr(env, "_runin_bank", None)
    if bank is None:
        from pathlib import Path

        path = Path(bank_path)
        if not path.is_absolute():
            # events.py → .../common/mdp → task-humanoid-run-jump/ is parents[7]
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
        env._runin_bank = bank
        print(f"[reset_from_runin_bank] loaded {bank['root_pose'].shape[0]} states from {path}")

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

    idx = torch.randint(0, n_bank, (n,), device=env.device)
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
    w0, x0, y0, z0 = root_pose[:, 3:7].unbind(-1)
    bank_yaw = torch.atan2(2.0 * (w0 * z0 + x0 * y0), 1.0 - 2.0 * (y0 * y0 + z0 * z0))
    dyaw = sample_uniform(*pose_range.get("yaw", (0.0, 0.0)), n, env.device) - bank_yaw
    half = dyaw * 0.5
    qw = torch.cos(half)
    qz = torch.sin(half)
    root_pose[:, 3] = qw * w0 - qz * z0
    root_pose[:, 4] = qw * x0 + qz * y0
    root_pose[:, 5] = qw * y0 - qz * x0
    root_pose[:, 6] = qw * z0 + qz * w0

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


def sample_uniform(
    low: float, high: float, size, device
) -> torch.Tensor:
    if isinstance(size, int):
        size = (size,)
    return torch.empty(size, device=device).uniform_(low, high)
