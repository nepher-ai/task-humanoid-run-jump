# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""EnvHub-backed G1 RunJumpHL environment configurations.

These configurations replace procedural course randomization and random run-in
sampling with deterministic scenario data loaded from an EnvHub bundle (e.g.
``humanoid-runjump-course-v1``). Course layout and run-in bank index are fixed
per parallel env for fair tournament evaluation.

Usage::

    cfg = G1RunJumpHLEnvCfg_Envhub(env_id="humanoid-runjump-course-v1", scene_id=0)
    env = gym.make("Nepher-G1-RunJumpHL-Envhub-Play-v0", cfg=cfg)
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

import humanoid_run_jump.tasks.manager_based.run_jump.mdp as mdp

from .hl_env_cfg import G1RunJumpHLEnvCfg, _CUBOID_HALF_H

_ZERO_POSE = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}


@configclass
class G1RunJumpHLEnvCfg_Envhub(G1RunJumpHLEnvCfg):
    """G1 RunJumpHL config backed by an EnvHub course + run-in probe bundle.

    Parameters
    ----------
    env_id:
        Identifier of the EnvHub environment bundle to load, e.g.
        ``"humanoid-runjump-course-v1"``.
    scene_id:
        Index of the scene / preset within the bundle (default: ``0``).
    """

    env_id: str = "humanoid-runjump-course-v1"
    scene_id: int | str = 0

    def __post_init__(self):
        super().__post_init__()

        from nepher import load_env
        from nepher.loader.preset_loader import PresetLoader

        env_bundle = load_env(self.env_id, category="navigation")
        preset_cfg = PresetLoader().load(env_bundle, int(self.scene_id), "navigation")
        self._envhub_preset = preset_cfg

        # Validation islands: 62 m X spacing; buffers sized for up to 10 hurdles.
        self.scene.env_spacing = 62.0
        preset_spacing = getattr(preset_cfg, "env_spacing", None)
        if preset_spacing is not None and float(preset_spacing) > 0.0:
            self.scene.env_spacing = float(preset_spacing)
        self.scene.env_spacing = max(self.scene.env_spacing, 62.0)
        self.course_max_obstacles = 10
        # Match AMP training window so EnvHub style discriminators see 2-frame obs.
        self.num_amp_observations = 2

        # Deterministic courses + run-in — disable curriculum and pose noise.
        self.curriculum.hl_course = None

        self.events.reset_base.params["pose_range"] = dict(_ZERO_POSE)
        self.events.reset_runin = EventTerm(
            func=mdp.reset_from_runin_bank_envhub,
            mode="reset",
            params={
                "preset_cfg": preset_cfg,
                "pose_range": dict(_ZERO_POSE),
            },
        )

        self.events.randomize_course = EventTerm(
            func=mdp.randomize_hl_course_from_envhub,
            mode="reset",
            params={
                "preset_cfg": preset_cfg,
                "cuboid_half_height": _CUBOID_HALF_H,
            },
        )


@configclass
class G1RunJumpHLEnvCfg_Envhub_PLAY(G1RunJumpHLEnvCfg_Envhub):
    """Play / evaluation variant of the EnvHub RunJumpHL config."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.episode_length_s = 40.0
        self.curriculum.hl_course = None
        self.show_plant_target = True
        self.show_course_lines = True
        # Pose noise already zeroed in Envhub base; keep bank_prob unused
        # (deterministic EnvHub reset always applies all env_ids).
        self.events.reset_base.params["pose_range"] = dict(_ZERO_POSE)
        self.events.reset_runin.params["pose_range"] = dict(_ZERO_POSE)
