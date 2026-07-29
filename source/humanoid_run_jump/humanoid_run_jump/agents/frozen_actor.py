# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load an exported skrl AMP actor (TorchScript obs → mean action).

Matches :func:`scripts.skrl.export_amp_policy.export_skrl_amp_policy_as_jit`:
normalization (if present) is baked into the scripted module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RUN_POLICY_PATH = _PROJECT_ROOT / "best_policy" / "run" / "policy.pt"


def resolve_actor_path(policy_path: str | Path | None = None) -> Path:
    """Resolve an exported actor path (CWD, then project root)."""
    path = Path(policy_path) if policy_path is not None else DEFAULT_RUN_POLICY_PATH
    if not path.is_absolute():
        for candidate in (Path.cwd() / path, _PROJECT_ROOT / path, path):
            if candidate.exists():
                return candidate.resolve()
        path = (_PROJECT_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen actor policy not found at '{path}'.\n"
            "Expected an exported TorchScript file (obs → mean action)."
        )
    return path


class FrozenActor(nn.Module):
    """Frozen exported AMP policy: raw policy obs → deterministic mean action."""

    def __init__(
        self,
        policy_path: str | Path | None = None,
        device: str | torch.device = "cpu",
        meta_path: str | Path | None = None,
        expected_obs_dim: int | None = None,
        expected_act_dim: int | None = None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.policy_path = resolve_actor_path(policy_path)
        self.meta: dict[str, Any] = {}
        meta_file = Path(meta_path) if meta_path else self.policy_path.with_name("policy_meta.json")
        if meta_file.exists():
            with open(meta_file, encoding="utf-8") as f:
                self.meta = json.load(f)

        self.policy = torch.jit.load(str(self.policy_path), map_location=self.device)
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)

        # Prefer meta; else probe with a smoke forward after first call.
        self.obs_dim = int(self.meta["obs_dim"]) if "obs_dim" in self.meta else expected_obs_dim
        self.act_dim = int(self.meta["act_dim"]) if "act_dim" in self.meta else expected_act_dim
        if self.obs_dim is None or self.act_dim is None:
            # Infer from module buffers / a zero probe if dims known from expected_*.
            if expected_obs_dim is not None:
                self.obs_dim = int(expected_obs_dim)
                with torch.inference_mode():
                    out = self.policy(torch.zeros(1, self.obs_dim, device=self.device))
                self.act_dim = int(out.shape[-1])
            else:
                raise ValueError(
                    f"Cannot infer obs/act dims for '{self.policy_path}'. "
                    "Provide policy_meta.json or expected_obs_dim."
                )
        if expected_obs_dim is not None and self.obs_dim != expected_obs_dim:
            raise ValueError(
                f"Actor obs_dim mismatch: meta/file={self.obs_dim}, expected={expected_obs_dim}"
            )
        if expected_act_dim is not None and self.act_dim != expected_act_dim:
            raise ValueError(
                f"Actor act_dim mismatch: meta/file={self.act_dim}, expected={expected_act_dim}"
            )

    @torch.inference_mode()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Map raw policy observation ``[N, obs_dim]`` → mean action ``[N, act_dim]``."""
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"Actor obs dim mismatch: got {obs.shape[-1]}, expected {self.obs_dim}")
        return self.policy(obs)

    @torch.inference_mode()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`forward`."""
        return self.forward(obs)


__all__ = [
    "FrozenActor",
    "DEFAULT_RUN_POLICY_PATH",
    "resolve_actor_path",
]
