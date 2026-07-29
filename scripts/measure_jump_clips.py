# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure tucked-apex thresholds from ``motions/jump_*.csv`` via MuJoCo FK.

Reproduces the constants in ``jump_envelope.py``:

* pelvis rise from liftoff to apex
* mean pelvis→ankle distance (tuck) at apex / liftoff / landing
* foot-to-foot separation during flight

Usage (from ``task-humanoid-run-jump/``)::

    python scripts/measure_jump_clips.py
"""

from __future__ import annotations

import glob
import os
import re
import statistics
import sys
from pathlib import Path

import mujoco
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "source" / "humanoid_run_jump"))

from humanoid_run_jump.motions.convert_csv import load_g1_csv  # noqa: E402
from humanoid_run_jump.robots.g1_constants import G1_MJCF_PATH  # noqa: E402

SOLE = 0.05
CONTACT_Z = 0.03


def _load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    mjcf_dir = Path(G1_MJCF_PATH).parent
    mesh_dir = mjcf_dir.parent / "mesh" / "G1"
    assets = {f: open(mesh_dir / f, "rb").read() for f in os.listdir(mesh_dir)}
    xml = Path(G1_MJCF_PATH).read_text(encoding="utf-8")
    # Drop floor contact pairs (geom "floor" is not in this MJCF).
    xml = re.sub(r"<contact>.*?</contact>", "", xml, flags=re.S)
    model = mujoco.MjModel.from_xml_string(xml, assets)
    return model, mujoco.MjData(model)


def _fk(model, data, csv_path: Path):
    motion = load_g1_csv(csv_path)
    rp = motion["root_positions"]
    rr = motion["root_rotations"]
    dp = motion["dof_positions"]
    fps = float(motion["fps"])
    n = len(rp)
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    lid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    P = np.zeros((n, 3))
    L = np.zeros((n, 3))
    R = np.zeros((n, 3))
    for i in range(n):
        data.qpos[0:3] = rp[i]
        data.qpos[3:7] = rr[i]
        data.qpos[7 : 7 + dp.shape[1]] = dp[i]
        mujoco.mj_forward(model, data)
        P[i] = data.xpos[pid]
        L[i] = data.xpos[lid]
        R[i] = data.xpos[rid]
    return P, L, R, fps


def _longest_air(sole_l, sole_r, thr: float = 0.05):
    air = (sole_l > thr) & (sole_r > thr)
    idx = np.where(air)[0]
    if idx.size == 0:
        return None
    splits = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    return max(splits, key=len)


def main() -> None:
    model, data = _load_model()
    rows = []
    for csv in sorted(glob.glob(str(_ROOT / "motions" / "jump_*.csv"))):
        if csv.endswith("_M.csv"):
            continue
        P, L, R, fps = _fk(model, data, Path(csv))
        sole_l = L[:, 2] - SOLE
        sole_r = R[:, 2] - SOLE
        gnd = min(sole_l.min(), sole_r.min())
        sole_l -= gnd
        sole_r -= gnd
        pz = P[:, 2] - gnd
        fl = _longest_air(sole_l, sole_r)
        if fl is None:
            print(f"no air: {csv}")
            continue
        t_lift, t_land = int(fl[0]), int(fl[-1])
        k = int(fl[np.argmax(pz[fl])])
        tavg = 0.5 * (
            np.linalg.norm(P - L, axis=1) + np.linalg.norm(P - R, axis=1)
        )
        sep = np.linalg.norm(L - R, axis=1)
        pre = max(t_lift - 1, 0)
        tof = "L" if sole_l[pre] < sole_r[pre] else "R"
        tag = "0.75" if "0_75m" in csv else "0.50"
        rows.append(
            dict(
                tag=tag,
                name=Path(csv).name[:44],
                tof=tof,
                pk=float(pz[k]),
                rise=float(pz[k] - pz[t_lift]),
                tuck_lift=float(tavg[t_lift]),
                tuck_pk=float(tavg[k]),
                tuck_land=float(tavg[t_land]),
                sep_pk=float(sep[k]),
                sep_max=float(sep[fl].max()),
                air_s=float((t_land - t_lift) / fps),
                rise_s=float((k - t_lift) / fps),
            )
        )

    hdr = (
        f"{'tag':5}{'tof':4}{'pk':>7}{'rise':>7}{'tk@L':>7}{'tk@pk':>7}"
        f"{'tk@land':>8}{'sep@pk':>7}{'sepmax':>7}{'air_s':>7}  name"
    )
    print(hdr)
    for r in rows:
        print(
            f"{r['tag']:5}{r['tof']:4}{r['pk']:7.3f}{r['rise']:7.3f}"
            f"{r['tuck_lift']:7.3f}{r['tuck_pk']:7.3f}{r['tuck_land']:8.3f}"
            f"{r['sep_pk']:7.3f}{r['sep_max']:7.3f}{r['air_s']:7.3f}  {r['name']}"
        )

    for tag in ("0.50", "0.75"):
        g = [r for r in rows if r["tag"] == tag]
        if not g:
            continue
        print(f"\n== {tag} m bin (n={len(g)}) takeoff={[r['tof'] for r in g]} ==")
        for key in (
            "pk",
            "rise",
            "tuck_lift",
            "tuck_pk",
            "tuck_land",
            "sep_pk",
            "sep_max",
            "air_s",
            "rise_s",
        ):
            v = [r[key] for r in g]
            print(
                f"  {key:10} min {min(v):.3f}  med {statistics.median(v):.3f}  max {max(v):.3f}"
            )

    print(
        "\nSuggested envelope anchors (see jump_envelope.py):\n"
        "  _APEX_RISE_LO/HI ~ 0.14 / 0.16  (obs only; success uses sole clearance)\n"
        "  _TUCK_APEX_LO/HI ~ 0.60 / 0.55\n"
        "  _PITCH_APEX_LO/HI ~ 0.18 / 0.20  (forward lean; positive = chest +x)\n"
        "  PITCH_APEX_MIN/MAX success band ~ 0.08 / 0.40\n"
        "  TUCK_EXTENDED ~ 0.60\n"
        "  FOOT_SEP_MAX_APEX / FLIGHT ~ 0.80 / 0.90\n"
        "  CLEARANCE_EXCESS_MARGIN / HARD_EXTRA ~ 0.05 / 0.25"
    )


if __name__ == "__main__":
    main()
