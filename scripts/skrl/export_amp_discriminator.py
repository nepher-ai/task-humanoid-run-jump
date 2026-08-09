# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a trained skrl AMP discriminator as TorchScript (amp_obs → logit).

Contract for EnvHub ``run_discriminator.pt`` / ``jump_discriminator.pt``::

    forward(amp_obs: FloatTensor[B, amp_dim]) -> FloatTensor[B, 1]  # logits

``amp_dim`` is the stacked AMP window used in training (typically 2 frames of
``amp_obs_single``). Any AMP state preprocessor (running mean/var) is baked in.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn as nn


class _SkrlAmpDiscriminatorJIT(nn.Module):
    """AMP discriminator: optional RunningStandardScaler + MLP → logit."""

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

    def forward(self, amp_obs: torch.Tensor) -> torch.Tensor:
        x = amp_obs
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


def _resolve_disc_head(discriminator: nn.Module) -> nn.Module | None:
    """Return the logit Linear head when it is separate from ``net_container``."""
    for name in ("output_layer", "value_layer"):
        layer = getattr(discriminator, name, None)
        if layer is not None:
            return copy.deepcopy(layer).cpu().eval()
    return None


def export_skrl_amp_discriminator_as_jit(
    agent,
    export_dir: str | Path,
    filename: str = "discriminator.pt",
) -> Path:
    """Trace the skrl AMP discriminator to ``export_dir/filename``.

    Input: raw AMP observation ``[B, amp_dim]`` (stacked frames, same as training).
    Output: discriminator logits ``[B, 1]`` (before sigmoid).
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    discriminator = getattr(agent, "discriminator", None)
    if discriminator is None:
        raise AttributeError("skrl agent has no 'discriminator' attribute")
    if not hasattr(discriminator, "net_container"):
        raise AttributeError(
            "skrl discriminator has no 'net_container'; expected DeterministicMixin model."
        )

    net = copy.deepcopy(discriminator.net_container).cpu().eval()
    output_layer = _resolve_disc_head(discriminator)

    obs_mean = obs_var = None
    epsilon, clip_threshold = 1e-8, 5.0
    preprocessor = getattr(agent, "_amp_state_preprocessor", None)
    if preprocessor is not None and hasattr(preprocessor, "running_mean"):
        obs_mean = preprocessor.running_mean.detach().cpu().clone()
        obs_var = preprocessor.running_variance.detach().cpu().clone()
        epsilon = float(getattr(preprocessor, "epsilon", 1e-8))
        clip_threshold = float(getattr(preprocessor, "clip_threshold", 5.0))

    module = _SkrlAmpDiscriminatorJIT(
        net=net,
        output_layer=output_layer,
        obs_mean=obs_mean,
        obs_var=obs_var,
        epsilon=epsilon,
        clip_threshold=clip_threshold,
    ).eval()

    if obs_mean is not None:
        amp_dim = int(obs_mean.numel())
    else:
        amp_dim = int(getattr(discriminator, "num_observations", 0))
        if amp_dim <= 0:
            raise RuntimeError(
                "Cannot infer amp_dim: no AMP preprocessor and discriminator.num_observations unset"
            )

    example = torch.zeros(1, amp_dim)
    with torch.inference_mode():
        out = module(example)
    if out.ndim != 2 or out.shape[0] != 1 or out.shape[-1] != 1:
        raise RuntimeError(f"JIT smoke test shape mismatch: got {tuple(out.shape)}, expected (1, 1)")

    scripted = torch.jit.trace(module, example)
    out_path = export_dir / filename
    scripted.save(str(out_path))

    meta = {
        "amp_dim": amp_dim,
        "normalize_obs": obs_mean is not None,
        "epsilon": epsilon,
        "clip_threshold": clip_threshold,
        "input": "AMP observation [B, amp_dim] (stacked amp_obs_single frames)",
        "output": "discriminator logits [B, 1] (before sigmoid)",
        "style_score": "sigmoid(logit) ∈ [0, 1] (P(human))",
    }
    meta_path = export_dir / f"{Path(filename).stem}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[export] wrote {out_path} (amp_dim={amp_dim})")
    print(f"[export] wrote {meta_path}")
    return out_path
