# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline validation helpers for the jump run-in redesign.

Checks (no Isaac Sim required)::

1. Frozen run actor loads and has expected obs/act dims.
2. Packaged jump motions: which ankle is ahead at takeoff (lead-foot hint).
3. Run-in bank (if present): sanity of root height / speed distributions.

Usage (from project root)::

    python scripts/validate_jump_runin.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_PKG_ROOT = Path(__file__).resolve().parents[1] / "source" / "humanoid_run_jump"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from humanoid_run_jump.agents.frozen_actor import DEFAULT_RUN_POLICY_PATH, FrozenActor
from humanoid_run_jump.robots.g1_constants import G1_JOINT_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JUMP_MOTIONS = PROJECT_ROOT / "motions" / "packaged" / "jump_motions.pt"
RUNIN_BANK = PROJECT_ROOT / "motions" / "packaged" / "runin_states.pt"

# Left / right ankle indices in packaged key_body_* (when non-zero).
_LEFT_ANKLE = 0
_RIGHT_ANKLE = 1


def _check_run_actor(policy_path: Path) -> str:
    actor = FrozenActor(
        policy_path=policy_path,
        device="cpu",
        expected_obs_dim=134,
        expected_act_dim=64,
    )
    with torch.inference_mode():
        out = actor.act(torch.zeros(2, 134))
    assert out.shape == (2, 64), out.shape
    return f"OK  loaded {policy_path}  obs={actor.obs_dim} act={actor.act_dim}  smoke_out={tuple(out.shape)}"


def _lead_foot_from_jump_motions(path: Path, phase: float = 0.35, n_samples: int = 64) -> str:
    if not path.exists():
        return f"SKIP  jump motions not found: {path}"

    try:
        from humanoid_run_jump.motions.motion_lib import MotionLib
        import numpy as np

        lib = MotionLib(path, device="cpu")
        ids = lib.sample_motions(n_samples, name_substr="jump")
        lengths = lib.motion_lengths[ids].numpy().astype(np.float64)
        times = np.clip(phase * lengths * lib.dt, 0.0, np.maximum((lengths - 2) * lib.dt, 0.0))
        frame = lib.get_frame(ids, times)

        kb = frame["key_body_pos"]
        if float(kb.abs().mean()) > 1e-6:
            along = kb[:, _RIGHT_ANKLE, 0] - kb[:, _LEFT_ANKLE, 0]
            right_lead = float((along > 0.02).float().mean())
            left_lead = float((along < -0.02).float().mean())
            src = "key_body_pos.x"
        else:
            # Packaged key bodies are currently zeros; use hip pitch as a proxy.
            # On G1, more-negative hip pitch ≈ more flexed. At plant the stance
            # (lead) leg is typically less flexed / more extended.
            left_hip = frame["dof_pos"][:, G1_JOINT_NAMES.index("left_hip_pitch_joint")]
            right_hip = frame["dof_pos"][:, G1_JOINT_NAMES.index("right_hip_pitch_joint")]
            # right more extended than left ⇒ right lead/plant
            along = left_hip - right_hip  # >0 ⇒ right more extended
            right_lead = float((along > 0.05).float().mean())
            left_lead = float((along < -0.05).float().mean())
            src = "hip_pitch proxy (key_body_pos are zeros)"

        recommend = "right" if right_lead >= left_lead else "left"
        return (
            f"OK  jump takeoff lead @ phase={phase:.2f} via {src}: "
            f"right={right_lead:.2%} left={left_lead:.2%} "
            f"(recommend --lead {recommend}; default remains right per tracker note)"
        )
    except Exception as exc:
        return f"SKIP  could not inspect jump motions ({exc})"


def _check_runin_bank(path: Path) -> str:
    if not path.exists():
        return (
            f"WARN  run-in bank missing: {path}\n"
            "      Build it with:\n"
            "        isaaclab.bat -p scripts/build_runin_bank.py --headless"
        )
    data = torch.load(path, map_location="cpu", weights_only=False)
    required = ("root_pose", "root_vel", "dof_pos", "dof_vel")
    for k in required:
        if k not in data:
            return f"FAIL  bank missing key '{k}'"
    n = int(data["root_pose"].shape[0])
    root_z = data["root_pose"][:, 2]
    speed = torch.linalg.norm(data["root_vel"][:, :2], dim=-1)
    dof = data["dof_pos"]
    lead = data.get("lead", "?")
    return (
        f"OK  bank n={n} lead={lead}  "
        f"root_z=[{root_z.min():.2f},{root_z.max():.2f}] mean={root_z.mean():.2f}  "
        f"horiz_speed=[{speed.min():.2f},{speed.max():.2f}] mean={speed.mean():.2f}  "
        f"dof_pos shape={tuple(dof.shape)} (expect N, {len(G1_JOINT_NAMES)})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate jump run-in redesign artifacts.")
    parser.add_argument("--policy", type=str, default=str(DEFAULT_RUN_POLICY_PATH))
    parser.add_argument("--jump-motions", type=str, default=str(JUMP_MOTIONS))
    parser.add_argument("--bank", type=str, default=str(RUNIN_BANK))
    parser.add_argument("--phase", type=float, default=0.35)
    args = parser.parse_args()

    print("=== Jump run-in validation ===")
    print("[1] Frozen run actor:")
    try:
        print("   ", _check_run_actor(Path(args.policy)))
    except Exception as exc:
        print("    FAIL ", exc)
        sys.exit(1)

    print("[2] Jump-motion lead foot:")
    print("   ", _lead_foot_from_jump_motions(Path(args.jump_motions), phase=args.phase))

    print("[3] Run-in bank:")
    print("   ", _check_runin_bank(Path(args.bank)))
    print("=== done ===")


if __name__ == "__main__":
    main()
