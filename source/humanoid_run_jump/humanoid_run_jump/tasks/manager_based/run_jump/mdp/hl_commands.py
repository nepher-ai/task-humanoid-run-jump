# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HL-written velocity command (no resampling) for the frozen run actor."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class HlVelocityCommand(CommandTerm):
    """Externally written ``[vx, vy, wz]`` command.

    The high-level switcher writes this every control step. No time-based
    resampling — that would fight the HL policy.
    """

    cfg: HlVelocityCommandCfg

    def __init__(self, cfg: HlVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        # Default cruise so early frames before the first HL write are sane.
        self._command[:, 0] = float(cfg.default_vx)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def set_velocity(self, vx: torch.Tensor, vy: torch.Tensor, wz: torch.Tensor) -> None:
        self._command[:, 0] = vx
        self._command[:, 1] = vy
        self._command[:, 2] = wz

    def set_velocity_ids(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        vx: torch.Tensor,
        vy: torch.Tensor,
        wz: torch.Tensor,
    ) -> None:
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(env_ids) == 0:
            return
        self._command[env_ids, 0] = vx
        self._command[env_ids, 1] = vy
        self._command[env_ids, 2] = wz

    def _update_metrics(self):
        return

    def _resample_command(self, env_ids: Sequence[int]):
        # Keep last written values; optionally reset to default on episode reset.
        if len(env_ids) == 0:
            return
        self._command[env_ids, 0] = float(self.cfg.default_vx)
        self._command[env_ids, 1] = 0.0
        self._command[env_ids, 2] = 0.0

    def _update_command(self):
        return


@configclass
class HlVelocityCommandCfg(CommandTermCfg):
    """Configuration for :class:`HlVelocityCommand`."""

    class_type: type = HlVelocityCommand
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    default_vx: float = 1.8
