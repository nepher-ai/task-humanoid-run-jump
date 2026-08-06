# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a trained skrl AMP policy as TorchScript (obs → mean action)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn as nn


class _SkrlAmpPolicyJIT(nn.Module):
    """Deterministic AMP actor: optional RunningStandardScaler + MLP mean head."""

    def __init__(
        self,
        net: nn.Module,
        output_layer: nn.Module | None,
        obs_mean: torch.Tensor | None,
        obs_var: torch.Tensor | None,
        epsilon: float = 1e-8,
        clip_threshold: float = 5.0,
    ):
        super().__init__()
        self.net = net
        self.output_layer = output_layer
        self.normalize = obs_mean is not None and obs_var is not None
        self.epsilon = float(epsilon)
        self.clip_threshold = float(clip_threshold)
        if self.normalize:
            self.register_buffer("obs_mean", obs_mean.float().contiguous())
            self.register_buffer("obs_var", obs_var.float().contiguous())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        if self.normalize:
            x = torch.clamp(
                (x - self.obs_mean) / (torch.sqrt(self.obs_var) + self.epsilon),
                min=-self.clip_threshold,
                max=self.clip_threshold,
            )
        x = self.net(x)
        if self.output_layer is not None:
            x = self.output_layer(x)
        return x


def _resolve_action_head(policy: nn.Module) -> nn.Module | None:
    """Return the mean-action Linear head for separate or shared skrl models.

    - ``separate: True`` (AMP): ``output_layer``
    - ``separate: False`` (HL PPO shared): ``policy_layer``
    """
    for name in ("output_layer", "policy_layer"):
        layer = getattr(policy, name, None)
        if layer is not None:
            return copy.deepcopy(layer).cpu().eval()
    return None


def export_skrl_amp_policy_as_jit(agent, export_dir: str | Path, filename: str = "policy.pt") -> Path:
    """Trace the skrl policy mean network to ``export_dir/filename``.

    Input: raw policy observation ``[B, obs_dim]`` (same as env ``policy`` group).
    Output: deterministic mean action ``[B, act_dim]``.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    policy = agent.policy
    if not hasattr(policy, "net_container"):
        raise AttributeError(
            "skrl policy has no 'net_container'; expected GaussianMixin / SharedModel from Runner."
        )

    net = copy.deepcopy(policy.net_container).cpu().eval()
    output_layer = _resolve_action_head(policy)
    if output_layer is None:
        raise AttributeError(
            "skrl policy has neither 'output_layer' nor 'policy_layer'; cannot export mean actions."
        )

    obs_mean = obs_var = None
    epsilon, clip_threshold = 1e-8, 5.0
    preprocessor = getattr(agent, "_state_preprocessor", None)
    if preprocessor is not None and hasattr(preprocessor, "running_mean"):
        obs_mean = preprocessor.running_mean.detach().cpu().clone()
        obs_var = preprocessor.running_variance.detach().cpu().clone()
        epsilon = float(getattr(preprocessor, "epsilon", 1e-8))
        clip_threshold = float(getattr(preprocessor, "clip_threshold", 5.0))

    module = _SkrlAmpPolicyJIT(
        net=net,
        output_layer=output_layer,
        obs_mean=obs_mean,
        obs_var=obs_var,
        epsilon=epsilon,
        clip_threshold=clip_threshold,
    ).eval()

    # Infer dims from preprocessor / action space.
    if obs_mean is not None:
        obs_dim = int(obs_mean.numel())
    else:
        obs_dim = int(policy.num_observations)
    act_dim = int(policy.num_actions)

    example = torch.zeros(1, obs_dim)
    with torch.inference_mode():
        out = module(example)
    if tuple(out.shape) != (1, act_dim):
        raise RuntimeError(f"JIT smoke test shape mismatch: got {tuple(out.shape)}, expected (1, {act_dim})")

    scripted = torch.jit.trace(module, example)
    out_path = export_dir / filename
    scripted.save(str(out_path))

    meta = {
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "normalize_obs": obs_mean is not None,
        "epsilon": epsilon,
        "clip_threshold": clip_threshold,
        "input": "policy observation [B, obs_dim]",
        "output": "mean_actions [B, act_dim] (deterministic actor)",
        "note": (
            "Mean head only (no value / discriminator). For AMP run/jump, feed the 64-D "
            "target frame to TrackerAction; for HL PPO, feed the 6-D action to HierarchicalSwitchAction."
        ),
    }
    meta_path = export_dir / "policy_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[export] wrote {out_path} (obs_dim={obs_dim}, act_dim={act_dim})")
    print(f"[export] wrote {meta_path}")
    return out_path
