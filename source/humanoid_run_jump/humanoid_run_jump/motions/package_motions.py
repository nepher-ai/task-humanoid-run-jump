# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package converted motion ``.npz`` files into a single AMP reference dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def _as_str_list(values) -> list[str]:
    """Convert name arrays to plain Python ``str`` (no numpy scalar types)."""
    return [str(v) for v in list(values)]


def _to_float_tensor(arr: np.ndarray) -> torch.Tensor:
    """Copy into an owned float32 torch tensor (no numpy storage / pickle deps)."""
    return torch.tensor(np.ascontiguousarray(arr, dtype=np.float32), dtype=torch.float32)


def package_motions(
    motion_dir: str | Path,
    output_file: str | Path,
    pattern: str = "*.npz",
) -> Path:
    """Concatenate all ``.npz`` clips into one ``.pt`` dict for :class:`MotionLib`.

    The payload contains only ``torch.Tensor`` and built-in Python types so it can be
    loaded under Isaac Sim's older NumPy (no ``numpy._core`` pickle refs).

    ``key_body_positions`` are expected to already be root-yaw-local (see
    :func:`convert_csv.attach_key_body_positions`). ``airborne_mask`` is optional
    but strongly recommended for flight-weighted AMP sampling.
    """
    motion_dir = Path(motion_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(motion_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No motions matching {pattern} in {motion_dir}")

    chunks: dict[str, list[np.ndarray]] = {
        "dof_positions": [],
        "dof_velocities": [],
        "root_positions": [],
        "root_rotations": [],
        "root_linear_velocities": [],
        "root_angular_velocities": [],
        "key_body_positions": [],
        "airborne_mask": [],
    }
    lengths: list[int] = []
    names: list[str] = []
    weights: list[float] = []
    fps = None
    dof_names = None
    key_body_names = None
    has_airborne = True

    for f in files:
        data = np.load(f, allow_pickle=True)
        for key in ("dof_positions", "dof_velocities", "root_positions", "root_rotations",
                    "root_linear_velocities", "root_angular_velocities", "key_body_positions"):
            if key not in data:
                raise KeyError(f"{f.name} missing key '{key}'")
            chunks[key].append(np.asarray(data[key], dtype=np.float32))
        n = int(data["dof_positions"].shape[0])
        if "airborne_mask" in data:
            chunks["airborne_mask"].append(np.asarray(data["airborne_mask"], dtype=np.float32).reshape(n))
        else:
            has_airborne = False
            chunks["airborne_mask"].append(np.zeros(n, dtype=np.float32))
        lengths.append(n)
        names.append(str(f.name))
        weights.append(2.0 if "jump" in f.name.lower() else 1.0)
        fps = float(np.asarray(data["fps"]).item()) if fps is None else fps
        if dof_names is None:
            dof_names = _as_str_list(data["dof_names"])
        if key_body_names is None:
            key_body_names = _as_str_list(data["key_body_names"])

    packed: dict[str, Any] = {
        key: _to_float_tensor(np.concatenate(vals, axis=0))
        for key, vals in chunks.items()
        if key != "airborne_mask"
    }
    packed["airborne_mask"] = torch.tensor(
        np.concatenate(chunks["airborne_mask"], axis=0), dtype=torch.bool
    )
    packed["fps"] = float(fps)
    packed["dof_names"] = dof_names
    packed["key_body_names"] = key_body_names
    packed["body_names"] = ["pelvis"] + list(key_body_names)
    packed["motion_lengths"] = torch.tensor(lengths, dtype=torch.long)
    packed["motion_weights"] = torch.tensor(weights, dtype=torch.float32)
    packed["motion_files"] = names
    packed["num_motions"] = int(len(lengths))
    packed["num_frames"] = int(packed["dof_positions"].shape[0])

    torch.save(packed, output_file, _use_new_zipfile_serialization=True)
    n_jump = sum(1 for w, name in zip(weights, names) if "jump" in name.lower())
    air_frac = float(packed["airborne_mask"].float().mean()) if has_airborne else 0.0
    key_max = float(packed["key_body_positions"].abs().max())
    print(
        f"[package_motions] wrote {output_file} "
        f"({packed['num_motions']} clips, {packed['num_frames']} frames @ {fps} Hz; "
        f"{n_jump} jump clips weighted ×2; air={air_frac:.1%}; key_absmax={key_max:.3f})"
    )
    return output_file
