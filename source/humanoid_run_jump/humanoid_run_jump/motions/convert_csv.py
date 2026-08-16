# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert bones-studio/seed G1 CSV motion clips to AMP-ready tensors.

Clips come from https://huggingface.co/datasets/bones-studio/seed.

Expected CSV format (header row, 120 Hz)::

    Frame, root_translateX/Y/Z (cm), root_rotateX/Y/Z (extrinsic XYZ deg),
    <joint>_dof (deg) × 29

Output per clip (dict saved inside a packaged ``.pt`` / ``.npz``)::

    dof_positions, dof_velocities, root_positions, root_rotations (wxyz),
    root_linear_velocities, root_angular_velocities,
    key_body_positions (root-yaw local), airborne_mask, fps, dof_names, body_names
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from humanoid_run_jump.robots.g1_constants import AMP_KEY_BODY_NAMES, G1_JOINT_NAMES, ROOT_BODY_NAME


def _try_import_mujoco():
    """Import mujoco only when FK is requested (unsafe under Isaac Sim on Windows)."""
    try:
        import mujoco  # noqa: WPS433

        return mujoco
    except Exception:  # ImportError or DLL init failure
        return None


def euler_to_quat_wxyz(euler_deg: np.ndarray, order: str = "xyz") -> np.ndarray:
    """Convert extrinsic/intrinsic Euler degrees to wxyz quaternions."""
    rot = Rotation.from_euler(order, euler_deg, degrees=True)
    xyzw = rot.as_quat()
    return np.concatenate([xyzw[:, 3:4], xyzw[:, :3]], axis=-1)


def quat_wxyz_to_ang_vel(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """Finite-difference angular velocity from wxyz quaternions (world frame)."""
    xyzw = np.concatenate([quat_wxyz[:, 1:], quat_wxyz[:, :1]], axis=-1)
    r = Rotation.from_quat(xyzw)
    n = len(quat_wxyz)
    ang_vel = np.zeros((n, 3), dtype=np.float64)
    for i in range(n - 1):
        delta = r[i].inv() * r[i + 1]
        ang_vel[i] = delta.as_rotvec() / dt
    if n > 1:
        ang_vel[-1] = ang_vel[-2]
    world = r.apply(ang_vel)
    return world.astype(np.float32)


def load_g1_csv(
    csv_path: str | Path,
    input_fps: float = 120.0,
    output_fps: float | None = None,
    euler_order: str = "xyz",
) -> dict[str, Any]:
    """Load a bones-studio/seed G1 CSV into motion arrays (meters, radians, wxyz)."""
    csv_path = Path(csv_path)
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    root_pos = data[:, 1:4] / 100.0  # cm -> m
    root_rot = euler_to_quat_wxyz(data[:, 4:7], order=euler_order)
    dof_pos = np.deg2rad(data[:, 7:])

    if dof_pos.shape[1] != len(G1_JOINT_NAMES):
        raise ValueError(
            f"{csv_path.name}: expected {len(G1_JOINT_NAMES)} joints, got {dof_pos.shape[1]}"
        )

    if output_fps is not None and output_fps < input_fps:
        factor = int(round(input_fps / output_fps))
        root_pos = root_pos[::factor]
        root_rot = root_rot[::factor]
        dof_pos = dof_pos[::factor]
        fps = output_fps
    else:
        fps = input_fps

    dt = 1.0 / fps
    dof_vel = np.gradient(dof_pos, dt, axis=0).astype(np.float32)
    root_lin_vel = np.gradient(root_pos, dt, axis=0).astype(np.float32)
    root_ang_vel = quat_wxyz_to_ang_vel(root_rot, dt)

    return {
        "dof_positions": dof_pos.astype(np.float32),
        "dof_velocities": dof_vel,
        "root_positions": root_pos.astype(np.float32),
        "root_rotations": root_rot.astype(np.float32),
        "root_linear_velocities": root_lin_vel,
        "root_angular_velocities": root_ang_vel,
        "fps": float(fps),
        "dof_names": list(G1_JOINT_NAMES),
        "body_names": [ROOT_BODY_NAME] + list(AMP_KEY_BODY_NAMES),
        "num_frames": int(dof_pos.shape[0]),
        "source": str(csv_path),
    }


def _load_mjcf_model(mjcf_path: str | Path):
    """Load G1 MJCF with mesh assets and floor-contact pairs stripped.

    ``from_xml_path`` fails because the MJCF references a ``floor`` geom that is
    not in the file; ``measure_jump_clips.py`` already uses this pattern.
    """
    mujoco = _try_import_mujoco()
    if mujoco is None:
        raise RuntimeError("mujoco is not available")
    mjcf_path = Path(mjcf_path)
    mjcf_dir = mjcf_path.parent
    mesh_dir = mjcf_dir.parent / "mesh" / "G1"
    if not mesh_dir.is_dir():
        # Fallback: assets may live next to mjcf.
        mesh_dir = mjcf_dir / "mesh" / "G1"
    assets: dict[str, bytes] = {}
    if mesh_dir.is_dir():
        for f in os.listdir(mesh_dir):
            assets[f] = open(mesh_dir / f, "rb").read()
    xml = mjcf_path.read_text(encoding="utf-8")
    xml = re.sub(r"<contact>.*?</contact>", "", xml, flags=re.S)
    model = mujoco.MjModel.from_xml_string(xml, assets)
    return mujoco, model, mujoco.MjData(model)


def _yaw_inv_apply(quat_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world vector into root-yaw frame (inverse yaw)."""
    w, x, y, z = quat_wxyz
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = np.cos(-yaw), np.sin(-yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]], dtype=np.float32)


def _fk_key_bodies_mujoco(
    mjcf_path: str | Path,
    root_pos: np.ndarray,
    root_rot_wxyz: np.ndarray,
    dof_pos: np.ndarray,
    key_body_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-kinematics key-body **root-yaw-local** positions + airborne mask.

    Returns ``(key_pos [T,K,3], airborne [T] bool)`` where airborne means both
    ankle soles are above 5 cm (same criterion as measure_jump_clips).
    """
    mujoco, model, data = _load_mjcf_model(mjcf_path)
    n = dof_pos.shape[0]
    out = np.zeros((n, len(key_body_names), 3), dtype=np.float32)
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in key_body_names]
    for bid, name in zip(body_ids, key_body_names):
        if bid < 0:
            raise RuntimeError(f"MJCF body not found: {name}")

    left_ankle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_ankle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    sole = 0.05
    ankle_z = np.zeros((n, 2), dtype=np.float32)

    for i in range(n):
        data.qpos[0:3] = root_pos[i]
        data.qpos[3:7] = root_rot_wxyz[i]
        data.qpos[7 : 7 + dof_pos.shape[1]] = dof_pos[i]
        mujoco.mj_forward(model, data)
        for j, bid in enumerate(body_ids):
            world = data.xpos[bid].astype(np.float32)
            rel = world - root_pos[i].astype(np.float32)
            out[i, j] = _yaw_inv_apply(root_rot_wxyz[i], rel)
        ankle_z[i, 0] = float(data.xpos[left_ankle][2]) - sole
        ankle_z[i, 1] = float(data.xpos[right_ankle][2]) - sole

    # Normalize soles relative to the clip's ground (min sole over clip).
    gnd = float(min(ankle_z[:, 0].min(), ankle_z[:, 1].min()))
    ankle_z -= gnd
    airborne = (ankle_z[:, 0] > 0.05) & (ankle_z[:, 1] > 0.05)
    return out, airborne.astype(np.bool_)


def attach_key_body_positions(
    motion: dict[str, Any],
    mjcf_path: str | Path | None = None,
    use_mujoco: bool = True,
) -> dict[str, Any]:
    """Add ``key_body_positions`` [T, K, 3] root-yaw-local and ``airborne_mask``.

    ``use_mujoco`` defaults to True. Raises if FK is requested but fails — silent
    zeros previously starved the AMP discriminator of arm/foot placement signal.
    """
    key_names = list(AMP_KEY_BODY_NAMES)
    T = motion["num_frames"]
    if not use_mujoco:
        motion["key_body_positions"] = np.zeros((T, len(key_names), 3), dtype=np.float32)
        motion["key_body_names"] = key_names
        motion["airborne_mask"] = np.zeros(T, dtype=np.bool_)
        return motion
    if mjcf_path is None or not Path(mjcf_path).exists():
        raise FileNotFoundError(f"MJCF path required for key-body FK: {mjcf_path}")
    key_pos, airborne = _fk_key_bodies_mujoco(
        mjcf_path,
        motion["root_positions"],
        motion["root_rotations"],
        motion["dof_positions"],
        key_names,
    )
    motion["key_body_positions"] = key_pos
    motion["key_body_names"] = key_names
    motion["airborne_mask"] = airborne
    return motion


def convert_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    pattern: str = "*.csv",
    input_fps: float = 120.0,
    mjcf_path: str | Path | None = None,
    use_mujoco: bool = True,
) -> list[Path]:
    """Convert all matching CSVs under ``input_dir`` to individual ``.npz`` files."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for csv_path in sorted(input_dir.glob(pattern)):
        motion = load_g1_csv(csv_path, input_fps=input_fps)
        motion = attach_key_body_positions(
            motion, mjcf_path=mjcf_path, use_mujoco=use_mujoco
        )
        out_path = output_dir / f"{csv_path.stem}.npz"
        payload = {
            "dof_positions": motion["dof_positions"],
            "dof_velocities": motion["dof_velocities"],
            "root_positions": motion["root_positions"],
            "root_rotations": motion["root_rotations"],
            "root_linear_velocities": motion["root_linear_velocities"],
            "root_angular_velocities": motion["root_angular_velocities"],
            "key_body_positions": motion["key_body_positions"],
            "airborne_mask": motion["airborne_mask"].astype(np.uint8),
            "fps": motion["fps"],
            "num_frames": motion["num_frames"],
            "dof_names": np.array(motion["dof_names"]),
            "body_names": np.array(motion["body_names"]),
            "key_body_names": np.array(motion["key_body_names"]),
            "source_file": np.array(motion["source"]),
        }
        np.savez_compressed(out_path, **payload)
        written.append(out_path)
        air_frac = float(motion["airborne_mask"].mean()) if motion["num_frames"] else 0.0
        print(
            f"[convert_csv] {csv_path.name} -> {out_path.name} "
            f"({motion['num_frames']} frames, air={air_frac:.1%}, "
            f"key_absmax={float(np.abs(motion['key_body_positions']).max()):.3f})"
        )
    return written
