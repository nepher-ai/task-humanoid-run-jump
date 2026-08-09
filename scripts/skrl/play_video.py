# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Documentary ``--video`` mode for skrl play.

Choreography
------------
1. High vantage overview of the full env grid for ``overview_s`` (default 3 s).
2. Track one robot at a time for ``track_s`` (default 12 s).
3. After each track, cut to the **nearest** not-yet-filmed robot.
4. Film ``num_robots`` robots (default 3).

Implementation notes
--------------------
- Camera poses are written with ``sim.set_camera_view`` in world space every
  step (no ViewportCameraController asset tracking — that races the RGB
  annotator and strobes black frames).
- Frames are captured explicitly after the camera settles, then written as a
  real-time mp4 (subsample when control rate > fps; hold frames when slower).
  Gymnasium ``RecordVideo`` is intentionally not used.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ---------------------------------------------------------------------------
# Task framing presets (eye / lookat offsets relative to the robot root)
# ---------------------------------------------------------------------------
_TRACK_PRESETS: dict[str, dict[str, tuple[float, float, float]]] = {
    "run": {
        "track_eye": (2.8, -2.8, 1.6),
        "track_lookat": (0.0, 0.0, 0.6),
    },
    "jump": {
        # Side / slightly ahead so the obstacle and flight arc stay in frame.
        "track_eye": (1.2, -4.2, 2.0),
        "track_lookat": (0.8, 0.0, 0.9),
    },
    "runjump": {
        "track_eye": (3.5, -3.5, 2.2),
        "track_lookat": (1.0, 0.0, 0.7),
    },
}


def _preset_for_task(task: str) -> dict[str, tuple[float, float, float]]:
    name = task.lower()
    if "runjump" in name:
        return _TRACK_PRESETS["runjump"]
    if "jump" in name:
        return _TRACK_PRESETS["jump"]
    return _TRACK_PRESETS["run"]


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    t = float(np.clip(t, 0.0, 1.0))
    return (1.0 - t) * a + t * b


def _smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# Stable RGB reads
# ---------------------------------------------------------------------------


def _normalize_rgb(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.uint8)
    if frame.ndim == 3 and frame.shape[-1] > 3:
        frame = frame[:, :, :3]
    return np.ascontiguousarray(frame)


def _frame_has_content(frame: np.ndarray, min_mean: float = 5.0, min_std: float = 4.0) -> bool:
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return False
    if float(frame.mean()) < min_mean:
        return False
    return float(frame.std()) >= min_std


def _frame_mad(a: np.ndarray, b: np.ndarray) -> float:
    aa = a[::8, ::8].astype(np.int16, copy=False)
    bb = b[::8, ::8].astype(np.int16, copy=False)
    return float(np.mean(np.abs(aa - bb)))


class StableRgbRenderWrapper(gym.Wrapper):
    """Retry viewport RGB until the frame has content; skip solid-black buffers."""

    def __init__(self, env: gym.Env, max_retries: int = 8, settle_renders: int = 2):
        super().__init__(env)
        self._max_retries = int(max_retries)
        self._settle_renders = int(settle_renders)
        self._last_good: np.ndarray | None = None
        self._black_fallback_count = 0

    def _sim_render(self) -> None:
        base = self.unwrapped
        if hasattr(base, "sim"):
            base.sim.render()

    def _read_rgb(self, *args: Any, **kwargs: Any) -> np.ndarray | None:
        try:
            frame = self.env.render(recompute=True)
        except TypeError:
            frame = self.env.render(*args, **kwargs)
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        return _normalize_rgb(frame)

    def render(self, *args: Any, **kwargs: Any) -> Any:
        for _ in range(self._settle_renders):
            self._sim_render()
        for _ in range(self._max_retries):
            self._sim_render()
            frame = self._read_rgb(*args, **kwargs)
            if frame is None or not _frame_has_content(frame):
                continue
            self._last_good = frame.copy()
            return frame
        self._black_fallback_count += 1
        if self._black_fallback_count in (1, 10, 50) or self._black_fallback_count % 100 == 0:
            print(
                f"[WARN] StableRgbRenderWrapper: black/empty annotator fallback "
                f"x{self._black_fallback_count}"
            )
        if self._last_good is not None:
            return self._last_good.copy()
        return self._read_rgb(*args, **kwargs)


# ---------------------------------------------------------------------------
# Real-time mp4 writer
# ---------------------------------------------------------------------------


class RealTimeVideoRecorder:
    """Capture RGB samples and write a real-time mp4 at ``fps``.

    Real-time means ``video_seconds == sim_seconds``. With control period
    ``step_dt``:

    * If ``fps * step_dt >= 1`` (control slower than video), each captured step
      is held for ``round(fps * step_dt)`` output frames (Spot-style 5 Hz → 25 fps).
    * If ``fps * step_dt < 1`` (control faster than video, e.g. G1 50 Hz → 25 fps),
      steps are subsampled every ``round(1 / (fps * step_dt))`` so motion is not
      played back in slow motion. Naively clamping hold to 1 would stretch each
      sim second into ``1 / (fps * step_dt)`` video seconds (2× slow at 50 Hz).
    """

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        *,
        fps: int = 25,
        step_dt: float = 0.02,
        name: str = "rl-video-step-0",
    ) -> None:
        self._env = env
        self._fps = max(1, int(fps))
        rate = self._fps * float(step_dt)
        if rate >= 1.0 - 1e-9:
            self._hold = max(1, int(round(rate)))
            self._stride = 1
        else:
            self._hold = 1
            self._stride = max(1, int(round(1.0 / max(rate, 1e-9))))
        self._capture_i = 0
        self._frames: list[np.ndarray] = []
        self._last_written: np.ndarray | None = None
        self._folder = os.path.abspath(video_folder)
        self._name = name
        os.makedirs(self._folder, exist_ok=True)
        print(
            f"[INFO] RealTimeVideoRecorder: step_dt={step_dt:.3f}s → "
            f"stride {self._stride} step(s), hold {self._hold} frame(s) "
            f"@ {self._fps} fps (real-time playback)."
        )

    def capture(self) -> None:
        idx = self._capture_i
        self._capture_i += 1
        if idx % self._stride != 0:
            return
        frame = None
        for _ in range(4):
            candidate = self._env.render()
            if candidate is None or not isinstance(candidate, np.ndarray) or candidate.size == 0:
                continue
            candidate = _normalize_rgb(candidate)
            if not _frame_has_content(candidate):
                continue
            if self._last_written is None or _frame_mad(candidate, self._last_written) >= 0.25:
                frame = candidate
                break
            frame = candidate
        if frame is None:
            return
        self._last_written = frame.copy()
        for i in range(self._hold):
            out = frame.copy()
            # Tiny unique pixel so encoders do not drop duplicate holds.
            if i:
                y = i % out.shape[0]
                x = (i * 7) % out.shape[1]
                out[y, x, 0] = np.uint8((int(out[y, x, 0]) + 1) % 256)
            self._frames.append(out)

    def close(self) -> str | None:
        if not self._frames:
            print("[WARN] RealTimeVideoRecorder: no frames captured.")
            return None
        path = os.path.join(self._folder, f"{self._name}.mp4")
        try:
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

            clip = ImageSequenceClip(self._frames, fps=self._fps)
            clip.write_videofile(path, logger=None)
            del clip
        except Exception:
            import cv2

            h, w = self._frames[0].shape[:2]
            writer = cv2.VideoWriter(
                path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(self._fps),
                (w, h),
            )
            for fr in self._frames:
                writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            writer.release()
        dur = len(self._frames) / float(self._fps)
        print(
            f"[INFO] Wrote video: {path} "
            f"({len(self._frames)} frames, {dur:.1f} s @ {self._fps} fps)"
        )
        self._frames.clear()
        return path


# ---------------------------------------------------------------------------
# Camera director
# ---------------------------------------------------------------------------


class VideoCameraDirector:
    """Overview → per-robot tracking with nearest-robot cuts.

    Call ``update(step_index)`` once per control step *after* ``env.step``,
    then settle-render and capture.
    """

    TARGET_VIDEO_FPS: int = 25
    _TRANSITION_S: float = 0.75
    _FOLLOW_ALPHA: float = 0.18
    _OVERVIEW_ALPHA: float = 0.12
    _CUT_ALPHA: float = 0.08

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        *,
        overview_s: float = 3.0,
        track_s: float = 12.0,
        num_robots: int = 3,
        asset_name: str = "robot",
        track_eye: tuple[float, float, float] = (2.8, -2.8, 1.6),
        track_lookat: tuple[float, float, float] = (0.0, 0.0, 0.6),
    ):
        self.env = env
        self.overview_s = float(overview_s)
        self.track_s = float(track_s)
        self.num_robots = int(max(1, min(num_robots, env.num_envs)))
        self.asset_name = asset_name
        self.track_eye = np.asarray(track_eye, dtype=float)
        self.track_lookat = np.asarray(track_lookat, dtype=float)

        self._dt = float(env.step_dt)
        self._origins = env.scene.env_origins.detach().cpu().numpy()
        self._grid_center = self._origins.mean(axis=0).copy()
        self._grid_center[2] = 0.0

        xy = self._origins[:, :2]
        span = float(np.max(xy.max(axis=0) - xy.min(axis=0)))
        spacing = float(getattr(env.cfg.scene, "env_spacing", 8.0) or 8.0)
        self._grid_half = 0.5 * span + 0.5 * spacing

        self._filmed: list[int] = []
        self._visited: set[int] = set()
        self._smoothed_eye: np.ndarray | None = None
        self._smoothed_target: np.ndarray | None = None
        self._last_logged_phase: str | None = None

        # Park ViewportCameraController so it does not fight set_camera_view.
        cam = getattr(env, "viewport_camera_controller", None)
        if cam is not None:
            cam.cfg.origin_type = "world"
            cam.viewer_origin = torch.zeros(3, device=env.device)

    @classmethod
    def for_task(
        cls,
        env: ManagerBasedRLEnv,
        task: str,
        *,
        overview_s: float = 3.0,
        track_s: float = 12.0,
        num_robots: int = 3,
    ) -> VideoCameraDirector:
        preset = _preset_for_task(task)
        return cls(
            env,
            overview_s=overview_s,
            track_s=track_s,
            num_robots=num_robots,
            track_eye=preset["track_eye"],
            track_lookat=preset["track_lookat"],
        )

    @property
    def total_duration_s(self) -> float:
        return self.overview_s + self.num_robots * self.track_s

    @property
    def total_steps(self) -> int:
        return max(1, int(np.ceil(self.total_duration_s / max(self._dt, 1e-6))))

    def warm_start(self) -> None:
        """Apply the overview pose and settle the viewport before the first capture."""
        self._filmed.clear()
        self._visited.clear()
        self._smoothed_eye = None
        self._smoothed_target = None
        self._last_logged_phase = None
        self.update(0)
        for _ in range(8):
            self.env.sim.render()

    def update(self, step: int) -> None:
        """Set the viewport camera for control step ``step`` (0-based)."""
        elapsed_s = float(step) * self._dt
        eye, target, alpha, phase = self._pose_at_time(elapsed_s)

        if self._smoothed_eye is None:
            self._smoothed_eye = eye.copy()
            self._smoothed_target = target.copy()
        else:
            self._smoothed_eye = _lerp(self._smoothed_eye, eye, alpha)
            self._smoothed_target = _lerp(self._smoothed_target, target, alpha)

        self.env.sim.set_camera_view(
            eye=self._smoothed_eye.tolist(),
            target=self._smoothed_target.tolist(),
        )

        if phase != self._last_logged_phase:
            self._last_logged_phase = phase
            print(f"[INFO] Video camera: {phase}")

    # ------------------------------------------------------------------
    # Pose math
    # ------------------------------------------------------------------

    def _pose_at_time(self, elapsed_s: float) -> tuple[np.ndarray, np.ndarray, float, str]:
        if elapsed_s < self.overview_s:
            t = elapsed_s / max(self.overview_s, 1e-6)
            eye, target = self._overview_pose(t)
            alpha = self._OVERVIEW_ALPHA
            phase = f"overview ({self.overview_s:.1f}s, all envs)"
            # Ease into the first robot near the end of the overview.
            if elapsed_s > self.overview_s - self._TRANSITION_S:
                first = self._ensure_first_subject()
                blend = _smoothstep(
                    (elapsed_s - (self.overview_s - self._TRANSITION_S))
                    / max(self._TRANSITION_S, 1e-6)
                )
                n_eye, n_target = self._track_pose(first)
                eye = _lerp(eye, n_eye, blend)
                target = _lerp(target, n_target, blend)
                alpha = self._CUT_ALPHA
                phase = f"overview → track env {first}"
            else:
                self._ensure_first_subject()
            return eye, target, alpha, phase

        local = elapsed_s - self.overview_s
        seg = int(local // self.track_s)
        if seg >= self.num_robots:
            seg = self.num_robots - 1
        env_idx = self._subject_for_segment(seg)
        t_in = float(local - seg * self.track_s)
        eye, target = self._track_pose(env_idx)
        alpha = self._FOLLOW_ALPHA
        phase = f"tracking env {env_idx} ({seg + 1}/{self.num_robots}, {self.track_s:.1f}s)"

        if seg < self.num_robots - 1 and t_in > self.track_s - self._TRANSITION_S:
            nxt = self._subject_for_segment(seg + 1)
            blend = _smoothstep(
                (t_in - (self.track_s - self._TRANSITION_S)) / max(self._TRANSITION_S, 1e-6)
            )
            n_eye, n_target = self._track_pose(nxt)
            eye = _lerp(eye, n_eye, blend)
            target = _lerp(target, n_target, blend)
            alpha = self._CUT_ALPHA
            phase = f"cut env {env_idx} → {nxt}"

        return eye, target, alpha, phase

    def _overview_pose(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """High angled shot of the full grid; gentle push-in, no orbit."""
        t = float(np.clip(t, 0.0, 1.0))
        ease = _smoothstep(t)
        half = max(self._grid_half, 6.0)
        back0, back1 = 1.25 * half, 1.05 * half
        height0 = max(14.0, 0.9 * half + 8.0)
        height1 = max(11.0, 0.75 * half + 6.0)
        back = back0 + (back1 - back0) * ease
        height = height0 + (height1 - height0) * ease
        c = self._grid_center
        # Look from +X/+Y so a square grid reads clearly.
        eye = np.array([c[0] + back, c[1] + back, height], dtype=float)
        target = np.array([c[0], c[1], 0.4], dtype=float)
        return eye, target

    def _track_pose(self, env_idx: int) -> tuple[np.ndarray, np.ndarray]:
        root = self._robot_pos(env_idx)
        eye = root + self.track_eye
        target = root + self.track_lookat
        return eye, target

    def _robot_pos(self, env_idx: int) -> np.ndarray:
        try:
            robot = self.env.scene[self.asset_name]
            return robot.data.root_pos_w[env_idx].detach().cpu().numpy().astype(float)
        except Exception:
            pos = self._origins[env_idx].copy()
            pos[2] = 0.75
            return pos

    def _ensure_first_subject(self) -> int:
        if self._filmed:
            return self._filmed[0]
        # Start with the robot nearest the grid centre.
        first = self._nearest_unvisited(self._grid_center[:2])
        if first is None:
            first = 0
        self._filmed.append(first)
        self._visited.add(first)
        return first

    def _subject_for_segment(self, seg: int) -> int:
        """Ensure featured list has segment ``seg``; pick nearest unfilmed as needed."""
        self._ensure_first_subject()
        while len(self._filmed) <= seg:
            ref = self._robot_pos(self._filmed[-1])[:2]
            nxt = self._nearest_unvisited(ref)
            if nxt is None:
                break
            self._filmed.append(nxt)
            self._visited.add(nxt)
        return self._filmed[min(seg, len(self._filmed) - 1)]

    def _nearest_unvisited(self, from_xy: np.ndarray) -> int | None:
        best_i: int | None = None
        best_d = float("inf")
        for i in range(self.env.num_envs):
            if i in self._visited:
                continue
            pos = self._robot_pos(i)
            d = float(np.linalg.norm(pos[:2] - from_xy))
            if d < best_d:
                best_d = d
                best_i = i
        return best_i
