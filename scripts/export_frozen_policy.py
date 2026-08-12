# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Copy or export a frozen BeyondMimic tracker into frozen_policies/.

Preferred path: copy an already-exported TorchScript policy::

    python scripts/export_frozen_policy.py \\
        --src ../humanoid-g1-tracking/results/g1_bm_l2c2_1frame/exported/policy.pt

If only a Lightning checkpoint is available, this script shells out to the
tracking project's ``export_jit.py`` (one-time build step; the resulting
``tracker.pt`` is then self-contained).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEST_DIR = PROJECT_ROOT / "frozen_policies"
DEST_NAME = "tracker.pt"


def copy_exported(src_pt: Path, src_meta: Path | None = None) -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    dest_pt = DEST_DIR / DEST_NAME
    shutil.copy2(src_pt, dest_pt)
    print(f"[export_frozen_policy] copied {src_pt} -> {dest_pt}")
    meta_src = src_meta or src_pt.with_name("policy_meta.json")
    if meta_src.exists():
        dest_meta = DEST_DIR / "policy_meta.json"
        shutil.copy2(meta_src, dest_meta)
        print(f"[export_frozen_policy] copied {meta_src} -> {dest_meta}")
    else:
        print("[export_frozen_policy] WARNING: policy_meta.json not found next to source.")


def export_from_ckpt(ckpt: Path) -> None:
    tracking_export = (
        PROJECT_ROOT.parent / "humanoid-g1-tracking" / "scripts" / "export_jit.py"
    )
    if not tracking_export.exists():
        raise FileNotFoundError(
            f"Cannot find tracking export script at {tracking_export}.\n"
            "Provide an already-exported policy.pt via --src instead."
        )
    out_dir = DEST_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(tracking_export),
        "--checkpoint",
        str(ckpt),
        "--output-dir",
        str(out_dir),
    ]
    print(f"[export_frozen_policy] running: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    # export_jit.py typically writes policy.pt; rename to the project default.
    produced = out_dir / "policy.pt"
    dest = out_dir / DEST_NAME
    if produced.exists() and produced.resolve() != dest.resolve():
        if dest.exists():
            dest.unlink()
        produced.rename(dest)
        print(f"[export_frozen_policy] renamed {produced.name} -> {dest.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Install frozen tracker tracker.pt")
    parser.add_argument(
        "--src",
        type=str,
        default=None,
        help="Path to an already-exported tracker TorchScript (preferred)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a tracking last.ckpt to export via export_jit.py",
    )
    args = parser.parse_args()

    default_src = (
        PROJECT_ROOT.parent
        / "humanoid-g1-tracking"
        / "results"
        / "g1_bm_l2c2_1frame"
        / "exported"
        / "policy.pt"
    )

    if args.src:
        copy_exported(Path(args.src))
    elif args.checkpoint:
        export_from_ckpt(Path(args.checkpoint))
    elif default_src.exists():
        copy_exported(default_src)
    else:
        raise SystemExit(
            "No --src / --checkpoint given and default exported policy.pt not found.\n"
            f"Expected at: {default_src}"
        )

    meta = DEST_DIR / "policy_meta.json"
    if meta.exists():
        with open(meta, encoding="utf-8") as f:
            data = json.load(f)
        print(
            f"[export_frozen_policy] ready: obs_dim={data.get('obs_dim')} "
            f"act_dim={data.get('act_dim')} future_steps={data.get('future_steps')}"
        )


if __name__ == "__main__":
    main()
