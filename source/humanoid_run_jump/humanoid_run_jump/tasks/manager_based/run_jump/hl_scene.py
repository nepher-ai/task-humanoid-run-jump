# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Anisotropic env grid for RunJumpHL (wide X for the course, tight Y)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab.scene import InteractiveScene
from isaacsim.core.cloner import GridCloner
from pxr import Gf, UsdGeom

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveSceneCfg

# Stashed by HlCourseEnv before scene construction (scene cfg cannot carry
# unknown fields — InteractiveScene treats them as assets).
_PENDING_ENV_SPACING_Y: float | None = None


class AnisotropicGridCloner(GridCloner):
    """GridCloner with independent X/Y spacing.

    Isaac Lab / Isaac Sim only expose a scalar ``spacing``. The HL course needs
    ~32 m along +x (path length) but only ~8 m along +y (5 m corridor).
    """

    def __init__(
        self,
        spacing_x: float,
        spacing_y: float,
        num_per_row: int = -1,
        stage=None,
    ):
        super().__init__(spacing=float(spacing_x), num_per_row=num_per_row, stage=stage)
        self._spacing_x = float(spacing_x)
        self._spacing_y = float(spacing_y)

    def get_clone_transforms(
        self,
        num_clones: int,
        position_offsets: np.ndarray = None,
        orientation_offsets: np.ndarray = None,
    ):
        if position_offsets is not None:
            if len(position_offsets) != num_clones:
                raise ValueError("Dimension mismatch between position_offsets and prim_paths!")
            if isinstance(position_offsets, torch.Tensor):
                position_offsets = position_offsets.detach().cpu().numpy()
            elif not isinstance(position_offsets, np.ndarray):
                position_offsets = np.asarray(position_offsets)
        if orientation_offsets is not None:
            if len(orientation_offsets) != num_clones:
                raise ValueError("Dimension mismatch between orientation_offsets and prim_paths!")
            if isinstance(orientation_offsets, torch.Tensor):
                orientation_offsets = orientation_offsets.detach().cpu().numpy()
            elif not isinstance(orientation_offsets, np.ndarray):
                orientation_offsets = np.asarray(orientation_offsets)

        if self._positions is not None and self._orientations is not None:
            return self._positions, self._orientations

        self._num_per_row = int(np.sqrt(num_clones)) if self._num_per_row == -1 else self._num_per_row
        num_rows = np.ceil(num_clones / self._num_per_row)
        num_cols = np.ceil(num_clones / num_rows)

        row_offset = 0.5 * self._spacing_x * (num_rows - 1)
        col_offset = 0.5 * self._spacing_y * (num_cols - 1)

        positions = []
        orientations = []
        up_axis = UsdGeom.GetStageUpAxis(self._stage)

        for i in range(num_clones):
            row = i // num_cols
            col = i % num_cols
            x = row_offset - row * self._spacing_x
            y = col * self._spacing_y - col_offset
            position = [x, y, 0] if up_axis == UsdGeom.Tokens.z else [x, 0, y]
            orientation = Gf.Quatd.GetIdentity()

            translation = position_offsets[i] + position if position_offsets is not None else position
            if orientation_offsets is not None:
                orientation = (
                    Gf.Quatd(orientation_offsets[i][0].item(), Gf.Vec3d(orientation_offsets[i][1:].tolist()))
                    * orientation
                )

            orientations.append(
                [
                    orientation.GetReal(),
                    orientation.GetImaginary()[0],
                    orientation.GetImaginary()[1],
                    orientation.GetImaginary()[2],
                ]
            )
            positions.append(translation)

        self._positions = positions
        self._orientations = orientations
        return positions, orientations


class HlInteractiveScene(InteractiveScene):
    """InteractiveScene that clones with ``env_spacing`` (X) × pending Y spacing."""

    def __init__(self, cfg: InteractiveSceneCfg):
        spacing_x = float(cfg.env_spacing)
        spacing_y = float(_PENDING_ENV_SPACING_Y) if _PENDING_ENV_SPACING_Y is not None else spacing_x
        # InteractiveScene always builds a scalar GridCloner; swap it for an
        # anisotropic one before any cloning runs (replicate_physics path).
        import isaaclab.scene.interactive_scene as scene_mod

        _orig = scene_mod.GridCloner

        def _factory(spacing, num_per_row=-1, stage=None):
            return AnisotropicGridCloner(spacing_x, spacing_y, num_per_row=num_per_row, stage=stage)

        scene_mod.GridCloner = _factory  # type: ignore[misc, assignment]
        try:
            super().__init__(cfg)
        finally:
            scene_mod.GridCloner = _orig

        # Plane terrain origins are built with isotropic env_spacing; compress Y
        # to match the cloner (y_iso * sy/sx == y_aniso).
        if abs(spacing_y - spacing_x) > 1e-9 and self._terrain is not None:
            origins = self._terrain.env_origins
            if origins is not None:
                origins[:, 1] *= spacing_y / spacing_x
