# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""High-level policy switching stub (not trained in this pass).

Architecture (target)
---------------------
The HL policy does **not** output joint actions. Each control step it:

1. Observes proprioception + next-obstacle features + low-level status.
2. Selects which frozen low-level generator is active: ``run`` or ``jump``.
3. Emits the corresponding command:
   - run  → velocity command ``[vx, vy, wz]``
   - jump → ``[h_obstacle, flight_distance]`` (same interface as ``JumpCommand``)
4. The selected generator produces a 64-D reduced-coords target frame.
5. The frozen BeyondMimic tracker converts that frame into joint PD targets.

Safe-jump rules (HL must respect)
---------------------------------
* Clamp ``h_obstacle`` to ``[0, 0.75]`` m (literal CSV / G1 motion-frame heights).
* Clamp ``flight_distance`` to ``[0.50, 1.20]`` m and apply height-coupled
  ``flight_cap(h)`` so (h, distance) stay inside ``motions/jump*.csv``.
* Prefer handing off to jump when the run gait is at the tracker-favorable
  lead-foot phase (default: right foot forward + contact).

Heuristic placeholder
---------------------
``HeuristicSwitcher`` transitions to jump when the next obstacle is within
``jump_trigger_distance`` meters, and back to run after a stable landing.
Replace this with a trained HL policy in a follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HeuristicSwitcherCfg:
    jump_trigger_distance: float = 2.5
    landing_height: float = 0.75
    landing_vz_thresh: float = 0.3
    max_obstacle_height: float = 0.75
    max_flight_distance: float = 1.20


class HeuristicSwitcher:
    """Distance-threshold policy selector (run=0, jump=1)."""

    RUN = 0
    JUMP = 1

    def __init__(self, num_envs: int, device: torch.device, cfg: HeuristicSwitcherCfg | None = None):
        self.cfg = cfg or HeuristicSwitcherCfg()
        self.mode = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.device = device

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self.mode[:] = self.RUN
        else:
            self.mode[env_ids] = self.RUN

    def step(
        self,
        dist_to_obstacle: torch.Tensor,
        root_height: torch.Tensor,
        root_vz: torch.Tensor,
        progress: torch.Tensor,
        flight_distance: torch.Tensor | None = None,
        h_obstacle: torch.Tensor | None = None,
        stable_ok: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update and return mode tensor ``[N]`` with values in {0,1}."""
        height_ok = torch.ones_like(dist_to_obstacle, dtype=torch.bool)
        if h_obstacle is not None:
            height_ok = h_obstacle <= self.cfg.max_obstacle_height
        dist_ok = torch.ones_like(height_ok)
        if flight_distance is not None:
            dist_ok = flight_distance <= self.cfg.max_flight_distance

        to_jump = (
            (self.mode == self.RUN)
            & (dist_to_obstacle < self.cfg.jump_trigger_distance)
            & height_ok
            & dist_ok
        )
        self.mode = torch.where(to_jump, torch.full_like(self.mode, self.JUMP), self.mode)

        if stable_ok is not None:
            to_run = (self.mode == self.JUMP) & stable_ok
        else:
            # Fallback: near standing height, small vz, past commanded flight.
            past = progress > (flight_distance * 0.9 if flight_distance is not None else 0.5)
            to_run = (
                (self.mode == self.JUMP)
                & (root_height < self.cfg.landing_height)
                & (root_vz.abs() < self.cfg.landing_vz_thresh)
                & past
            )
        self.mode = torch.where(to_run, torch.full_like(self.mode, self.RUN), self.mode)
        return self.mode


# TODO(hl): HierarchicalSwitchAction(ActionTerm)
#   - holds FrozenTracker + loaded run_policy.pt + jump_policy.pt
#   - HL action = discrete mode (+ optional continuous command overrides)
#   - routes generator outputs into tracker target frame
#   - jump command = [h_obstacle, flight_distance]
#   - hand-off gated on runin gait phase (right foot forward + contact)
