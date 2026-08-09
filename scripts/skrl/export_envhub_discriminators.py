# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline export of EnvHub run/jump AMP discriminators from skrl checkpoints.

Does not require Isaac Sim — rebuilds the DeterministicMixin discriminator MLP
from the checkpoint state dict and traces it with the AMP preprocessor baked in.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn

from export_amp_discriminator import _SkrlAmpDiscriminatorJIT

TASK_ROOT = Path(__file__).resolve().parents[2]
ENVHUB = TASK_ROOT.parent / "envhub" / "environments" / "humanoid-runjump-course-v1"

RUN_CKPT = (
    TASK_ROOT
    / "logs"
    / "skrl"
    / "g1_run_amp"
    / "2026-07-31_10-26-27_amp_torch"
    / "checkpoints"
    / "best_agent.pt"
)
JUMP_CKPT = (
    TASK_ROOT
    / "logs"
    / "skrl"
    / "g1_jump_amp"
    / "2026-07-31_04-48-47_amp_torch"
    / "checkpoints"
    / "best_agent.pt"
)


def _build_disc_net(state: dict[str, torch.Tensor]) -> nn.Sequential:
    """Rebuild skrl Discriminator net_container: Linear-ReLU-Linear-ReLU-Linear."""
    w0 = state["net_container.0.weight"]
    w2 = state["net_container.2.weight"]
    w4 = state["net_container.4.weight"]
    in_dim = int(w0.shape[1])
    h1 = int(w0.shape[0])
    h2 = int(w2.shape[0])
    out_dim = int(w4.shape[0])
    net = nn.Sequential(
        nn.Linear(in_dim, h1),
        nn.ReLU(),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Linear(h2, out_dim),
    )
    net.load_state_dict(
        {
            "0.weight": state["net_container.0.weight"],
            "0.bias": state["net_container.0.bias"],
            "2.weight": state["net_container.2.weight"],
            "2.bias": state["net_container.2.bias"],
            "4.weight": state["net_container.4.weight"],
            "4.bias": state["net_container.4.bias"],
        }
    )
    return net.eval()


def export_from_checkpoint(ckpt_path: Path, out_path: Path) -> dict:
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "discriminator" not in data:
        raise KeyError(f"No discriminator in {ckpt_path}")
    disc_state = data["discriminator"]
    amp_pp = data.get("amp_state_preprocessor")
    if amp_pp is None:
        raise KeyError(f"No amp_state_preprocessor in {ckpt_path}")

    net = _build_disc_net(disc_state)
    obs_mean = amp_pp["running_mean"].detach().float().contiguous()
    obs_var = amp_pp["running_variance"].detach().float().contiguous()
    module = _SkrlAmpDiscriminatorJIT(
        net=net,
        output_layer=None,
        obs_mean=obs_mean,
        obs_var=obs_var,
        epsilon=1e-8,
        clip_threshold=5.0,
    ).eval()

    amp_dim = int(obs_mean.numel())
    example = torch.zeros(1, amp_dim)
    with torch.inference_mode():
        out = module(example)
    if tuple(out.shape) != (1, 1):
        raise RuntimeError(f"Unexpected output shape {tuple(out.shape)} from {ckpt_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.trace(module, example)
    scripted.save(str(out_path))

    meta = {
        "source_checkpoint": str(ckpt_path),
        "amp_dim": amp_dim,
        "normalize_obs": True,
        "epsilon": 1e-8,
        "clip_threshold": 5.0,
        "input": "AMP observation [B, amp_dim] (2 × amp_obs_single)",
        "output": "discriminator logits [B, 1] (before sigmoid)",
    }
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[export] {out_path} (amp_dim={amp_dim}) from {ckpt_path.name}")
    return meta


def _sha256(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    data = path.read_bytes()
    h.update(data)
    return h.hexdigest(), len(data)


def update_checksums(bundle: Path, files: list[str]) -> None:
    checksums_path = bundle / "checksums.json"
    if checksums_path.is_file():
        checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    else:
        checksums = {}
    for name in files:
        path = bundle / name
        digest, nbytes = _sha256(path)
        checksums[name] = {"sha256": digest, "bytes": nbytes}
        print(f"[checksums] {name}: {digest} ({nbytes} bytes)")
    checksums_path.write_text(json.dumps(checksums, indent=2) + "\n", encoding="utf-8")
    print(f"[checksums] wrote {checksums_path}")


def main() -> None:
    if not RUN_CKPT.is_file():
        raise FileNotFoundError(RUN_CKPT)
    if not JUMP_CKPT.is_file():
        raise FileNotFoundError(JUMP_CKPT)

    run_out = ENVHUB / "run_discriminator.pt"
    jump_out = ENVHUB / "jump_discriminator.pt"
    export_from_checkpoint(RUN_CKPT, run_out)
    export_from_checkpoint(JUMP_CKPT, jump_out)
    update_checksums(ENVHUB, ["run_discriminator.pt", "jump_discriminator.pt"])


if __name__ == "__main__":
    main()
