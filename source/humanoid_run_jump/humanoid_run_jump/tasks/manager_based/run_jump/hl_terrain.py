# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain helpers for RunJumpHL.

Isaac Lab's default plane is ``(2e6, 2e6)`` m. At play-camera distances that
oversized mesh fights the depth buffer and the ground shimmers / flashes white.
A few-kilometre plane is enough for the env grid and keeps depth stable.
"""

from __future__ import annotations

from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass

# Covers a 2048-env grid at 32 m X spacing with margin; far smaller than 2e6.
_HL_GROUND_SIZE_M = (4000.0, 4000.0)


class HlTerrainImporter(TerrainImporter):
    """TerrainImporter with a finite ground plane to avoid z-fighting flicker."""

    def import_ground_plane(self, name: str, size: tuple[float, float] = _HL_GROUND_SIZE_M):
        return super().import_ground_plane(name, size=size)


@configclass
class HlTerrainImporterCfg(TerrainImporterCfg):
    class_type: type = HlTerrainImporter
