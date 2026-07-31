# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP motion library: sample reference frames for the discriminator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class MotionLib:
    """Sample time-aligned motion windows from a packaged ``.pt`` dataset."""

    def __init__(
        self,
        motion_file: str | Path,
        device: torch.device | str = "cpu",
        flight_sample_boost: float = 4.0,
    ):
        motion_file = Path(motion_file)
        if not motion_file.exists():
            raise FileNotFoundError(
                f"Motion dataset not found: {motion_file}\n"
                "Build it with: python scripts/build_amp_dataset.py"
            )
        data = torch.load(motion_file, map_location="cpu", weights_only=False)
        self.device = torch.device(device)
        self.fps = float(data["fps"])
        self.dt = 1.0 / self.fps
        self.dof_names: list[str] = [str(x) for x in data["dof_names"]]
        self.key_body_names: list[str] = [str(x) for x in data["key_body_names"]]
        self.motion_lengths = data["motion_lengths"].long().cpu()
        self.motion_weights = data["motion_weights"].float().cpu()
        self.motion_weights = self.motion_weights / self.motion_weights.sum()
        self.motion_files: list[str] = [str(x) for x in data.get("motion_files", [])]
        if len(self.motion_files) != len(self.motion_lengths):
            self.motion_files = [f"clip_{i}" for i in range(len(self.motion_lengths))]
        self.flight_sample_boost = float(flight_sample_boost)

        self._starts = torch.zeros(len(self.motion_lengths), dtype=torch.long)
        if len(self.motion_lengths) > 1:
            self._starts[1:] = torch.cumsum(self.motion_lengths[:-1], dim=0)

        def _to(t: torch.Tensor) -> torch.Tensor:
            return t.float().to(self.device)

        self.dof_positions = _to(data["dof_positions"])
        self.dof_velocities = _to(data["dof_velocities"])
        self.root_positions = _to(data["root_positions"])
        self.root_rotations = _to(data["root_rotations"])
        self.root_linear_velocities = _to(data["root_linear_velocities"])
        self.root_angular_velocities = _to(data["root_angular_velocities"])
        self.key_body_positions = _to(data["key_body_positions"])
        if "airborne_mask" in data:
            self.airborne_mask = data["airborne_mask"].bool().to(self.device)
        else:
            self.airborne_mask = torch.zeros(
                self.dof_positions.shape[0], dtype=torch.bool, device=self.device
            )
        self.num_frames = int(self.dof_positions.shape[0])
        air_frac = float(self.airborne_mask.float().mean()) if self.num_frames else 0.0
        print(
            f"[MotionLib] loaded {motion_file.name}: "
            f"{len(self.motion_lengths)} clips, {self.num_frames} frames @ {self.fps} Hz "
            f"(air={air_frac:.1%}, key_absmax={float(self.key_body_positions.abs().max()):.3f})"
        )

    def sample_motions(self, n: int, name_substr: str | None = None) -> torch.Tensor:
        """Sample clip indices with replacement according to weights.

        If ``name_substr`` is set (e.g. ``\"jump\"``), only clips whose filename
        contains that substring (case-insensitive) are eligible. Falls back to
        the full set when no name matches.
        """
        weights = self.motion_weights.to(self.device)
        if name_substr:
            mask = torch.tensor(
                [name_substr.lower() in name.lower() for name in self.motion_files],
                device=self.device,
                dtype=torch.bool,
            )
            if bool(mask.any()):
                weights = weights * mask.float()
                weights = weights / weights.sum().clamp(min=1e-8)
        return torch.multinomial(weights, n, replacement=True)

    def sample_times(
        self,
        n: int,
        motion_ids: torch.Tensor | None = None,
        *,
        history_span_s: float = 0.0,
    ) -> np.ndarray:
        """Sample times (seconds) within each clip, biased toward flight frames.

        Uniform sampling makes ~88% of jump-clip frames ground frames while the
        policy's episodes are flight-dominated, so the discriminator can separate
        on contact alone. Airborne frames are up-weighted by ``flight_sample_boost``.

        ``history_span_s`` reserves headroom so a ``num_frames`` window spanning
        that many seconds stays inside the clip (e.g. 0.02 s for a 2-frame
        control-rate history).
        """
        if motion_ids is None:
            motion_ids = self.sample_motions(n)
        motion_ids_cpu = motion_ids.cpu()
        lengths = self.motion_lengths[motion_ids_cpu].numpy().astype(np.int64)
        starts = self._starts[motion_ids_cpu].numpy().astype(np.int64)
        times = np.zeros(n, dtype=np.float64)
        air_cpu = self.airborne_mask.cpu().numpy()
        boost = max(self.flight_sample_boost, 1.0)
        # Frames of history headroom for the requested window span.
        headroom = max(int(np.ceil(max(history_span_s, 0.0) / self.dt)), 1)

        for i in range(n):
            length = int(lengths[i])
            if length <= headroom:
                times[i] = 0.0
                continue
            # Leave ≥headroom frames so t - history_span_s stays in-clip.
            usable = max(length - headroom, 1)
            start = int(starts[i])
            frame_w = np.ones(usable, dtype=np.float64)
            air_slice = air_cpu[start : start + usable]
            if air_slice.any():
                frame_w[air_slice] *= boost
            frame_w /= frame_w.sum()
            frame = int(np.random.choice(usable, p=frame_w))
            times[i] = frame * self.dt
        return times

    def _frame_index(self, motion_ids: torch.Tensor, times: np.ndarray) -> torch.Tensor:
        frames = np.clip(
            (times / self.dt).astype(np.int64),
            0,
            self.motion_lengths[motion_ids.cpu()].numpy() - 1,
        )
        starts = self._starts[motion_ids.cpu()].numpy()
        return torch.tensor(starts + frames, device=self.device, dtype=torch.long)

    def get_frame(self, motion_ids: torch.Tensor, times: np.ndarray) -> dict[str, torch.Tensor]:
        """Fetch a single frame per sample (nearest discrete motion frame)."""
        idx = self._frame_index(motion_ids, times)
        return {
            "dof_pos": self.dof_positions[idx],
            "dof_vel": self.dof_velocities[idx],
            "root_pos": self.root_positions[idx],
            "root_rot": self.root_rotations[idx],
            "root_lin_vel": self.root_linear_velocities[idx],
            "root_ang_vel": self.root_angular_velocities[idx],
            "key_body_pos": self.key_body_positions[idx],
        }

    def get_frame_interp(self, motion_ids: torch.Tensor, times: np.ndarray) -> dict[str, torch.Tensor]:
        """Fetch a time-interpolated frame per sample.

        Linear blend for positions/velocities; sign-aligned normalized lerp for
        root rotations (wxyz). Needed when ``frame_dt`` is not an integer
        multiple of the motion ``dt`` (e.g. control 20 ms vs motion 8.33 ms).
        """
        motion_ids_cpu = motion_ids.cpu()
        lengths = self.motion_lengths[motion_ids_cpu].numpy().astype(np.int64)
        starts = self._starts[motion_ids_cpu].numpy().astype(np.int64)
        # Continuous frame index inside each clip.
        frame_f = np.clip(times / self.dt, 0.0, np.maximum(lengths - 1, 0).astype(np.float64))
        i0 = np.floor(frame_f).astype(np.int64)
        i1 = np.minimum(i0 + 1, lengths - 1)
        alpha_np = (frame_f - i0).astype(np.float64)
        # Exact frame: skip blend.
        exact = alpha_np < 1e-8

        idx0 = torch.tensor(starts + i0, device=self.device, dtype=torch.long)
        idx1 = torch.tensor(starts + i1, device=self.device, dtype=torch.long)
        alpha = torch.tensor(alpha_np, device=self.device, dtype=torch.float32)
        exact_t = torch.tensor(exact, device=self.device, dtype=torch.bool)

        def _lerp(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
            # Broadcast alpha to match tensor rank.
            shape = (-1,) + (1,) * (x0.ndim - 1)
            aa = alpha.view(*shape)
            out = x0 + aa * (x1 - x0)
            return torch.where(exact_t.view(*shape), x0, out)

        dof_pos = _lerp(self.dof_positions[idx0], self.dof_positions[idx1])
        dof_vel = _lerp(self.dof_velocities[idx0], self.dof_velocities[idx1])
        root_pos = _lerp(self.root_positions[idx0], self.root_positions[idx1])
        root_lin_vel = _lerp(self.root_linear_velocities[idx0], self.root_linear_velocities[idx1])
        root_ang_vel = _lerp(self.root_angular_velocities[idx0], self.root_angular_velocities[idx1])
        key_body_pos = _lerp(self.key_body_positions[idx0], self.key_body_positions[idx1])

        # Quaternion wxyz: flip sign so dot >= 0, then lerp + renormalize.
        q0 = self.root_rotations[idx0]
        q1 = self.root_rotations[idx1]
        dot = (q0 * q1).sum(dim=-1, keepdim=True)
        q1 = torch.where(dot < 0.0, -q1, q1)
        aa = alpha.view(-1, 1)
        q = q0 + aa * (q1 - q0)
        q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        root_rot = torch.where(exact_t.view(-1, 1), q0, q)

        return {
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "root_lin_vel": root_lin_vel,
            "root_ang_vel": root_ang_vel,
            "key_body_pos": key_body_pos,
        }

    def sample_window(
        self,
        num_samples: int,
        num_frames: int = 2,
        current_times: np.ndarray | None = None,
        frame_dt: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample ``num_frames`` of history ending at ``current_times``.

        ``frame_dt`` is the history stride in seconds. Pass the env control
        period (e.g. ``step_dt`` = 0.02) so reference windows match the policy
        AMP buffer. When ``None``, uses the motion's native ``dt``.

        Returns tensors shaped ``[num_samples * num_frames, ...]`` ordered as
        ``[t, t-frame_dt, t-2*frame_dt, ...]`` flattened like Isaac Lab HumanoidAmpEnv.
        """
        stride = float(self.dt if frame_dt is None else frame_dt)
        history_span = stride * max(num_frames - 1, 0)
        motion_ids = self.sample_motions(num_samples)
        if current_times is None:
            current_times = self.sample_times(
                num_samples, motion_ids, history_span_s=history_span
            )
        offsets = stride * np.arange(0, num_frames)
        times = (np.expand_dims(current_times, axis=-1) - offsets).flatten()
        motion_ids_rep = motion_ids.repeat_interleave(num_frames)
        times = np.maximum(times, 0.0)
        # Interpolate when stride is not an integer multiple of motion dt.
        if abs(stride - self.dt) < 1e-9:
            return self.get_frame(motion_ids_rep, times)
        return self.get_frame_interp(motion_ids_rep, times)
