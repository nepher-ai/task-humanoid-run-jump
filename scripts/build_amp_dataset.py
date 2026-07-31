# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build AMP reference datasets from bones-speed CSVs (120 Hz).

Usage (from project root)::

    python scripts/build_amp_dataset.py
    python scripts/build_amp_dataset.py --subset run
    python scripts/build_amp_dataset.py --subset jump
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

# Allow running without an editable install.
_PKG_ROOT = Path(__file__).resolve().parents[1] / "source" / "humanoid_run_jump"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from humanoid_run_jump.motions.convert_csv import convert_directory
from humanoid_run_jump.motions.package_motions import package_motions
from humanoid_run_jump.robots.g1_constants import G1_MJCF_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOTIONS_DIR = PROJECT_ROOT / "motions"
PACKAGED_DIR = MOTIONS_DIR / "packaged"


def _filter_csvs(subset: str) -> list[Path]:
    all_csvs = sorted(MOTIONS_DIR.glob("*.csv"))
    if subset == "all":
        return all_csvs
    if subset == "run":
        return [p for p in all_csvs if p.name.startswith("run_")]
    if subset == "jump":
        return [p for p in all_csvs if "jump" in p.name]
    raise ValueError(f"Unknown subset: {subset}")


def build(subset: str, input_fps: float = 120.0) -> Path:
    csvs = _filter_csvs(subset)
    if not csvs:
        raise FileNotFoundError(f"No CSV motions found for subset={subset} in {MOTIONS_DIR}")

    with tempfile.TemporaryDirectory(prefix="amp_npz_") as tmp:
        tmp_dir = Path(tmp)
        for csv in csvs:
            shutil.copy2(csv, tmp_dir / csv.name)
        npz_dir = tmp_dir / "npz"
        convert_directory(
            input_dir=tmp_dir,
            output_dir=npz_dir,
            input_fps=input_fps,
            mjcf_path=G1_MJCF_PATH,
            use_mujoco=True,
        )
        out = PACKAGED_DIR / f"{subset}_motions.pt"
        package_motions(npz_dir, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AMP motion datasets from CSVs")
    parser.add_argument(
        "--subset",
        choices=["run", "jump", "all"],
        default="all",
        help="Which motion subset to package (default: all → also builds run and jump)",
    )
    parser.add_argument("--input-fps", type=float, default=120.0)
    args = parser.parse_args()

    PACKAGED_DIR.mkdir(parents=True, exist_ok=True)
    if args.subset == "all":
        for s in ("run", "jump", "all"):
            build(s, input_fps=args.input_fps)
    else:
        build(args.subset, input_fps=args.input_fps)


if __name__ == "__main__":
    main()
