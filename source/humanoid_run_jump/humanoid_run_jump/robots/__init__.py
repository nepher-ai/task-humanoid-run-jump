# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot articulation configs."""

from .g1_constants import (
    AMP_KEY_BODY_NAMES,
    ANCHOR_BODY_NAME,
    G1_JOINT_NAMES,
    G1_MJCF_PATH,
    G1_USD_PATH,
    NUM_JOINTS,
    ROOT_BODY_NAME,
)
from .joint_order import JointOrderMap

# Lazy: ArticulationCfg requires Isaac Lab; import only when available.
try:
    from .g1 import G1_CFG, G1_MINIMAL_CFG
except ModuleNotFoundError:  # pragma: no cover - offline scripts
    G1_CFG = None  # type: ignore
    G1_MINIMAL_CFG = None  # type: ignore

__all__ = [
    "G1_CFG",
    "G1_MINIMAL_CFG",
    "G1_USD_PATH",
    "G1_MJCF_PATH",
    "G1_JOINT_NAMES",
    "NUM_JOINTS",
    "ANCHOR_BODY_NAME",
    "ROOT_BODY_NAME",
    "AMP_KEY_BODY_NAMES",
    "JointOrderMap",
]
