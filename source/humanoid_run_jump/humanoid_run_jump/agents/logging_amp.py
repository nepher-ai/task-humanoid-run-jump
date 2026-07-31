# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP agent that TensorBoard-logs task / style / total rewards."""

from __future__ import annotations

import torch
from skrl.agents.torch.amp import AMP


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


class LoggingAMP(AMP):
    """skrl AMP with reward-component logging and optional time weight curriculum.

    When ``amp_reward_curriculum.enabled`` is set (time mode only), lerp task/style
    weights over ``steps_to_final``. Otherwise YAML ``task_reward_weight`` /
    ``style_reward_weight`` are used as-is.

    Also applies per-step ``amp_style_scale`` (phase-dependent style gain) and logs
    flight-success rates from the env extras.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep style scales on the training device to avoid per-step GPU→CPU sync.
        self._style_scale_buffer: list[torch.Tensor] = []

    def record_transition(
        self,
        states,
        actions,
        rewards,
        next_states,
        terminated,
        truncated,
        infos,
        timestep,
        timesteps,
    ):
        if isinstance(infos, dict):
            # Per-env AMP style scale for this rollout step (takeoff/flight softens style).
            if "amp_style_scale" in infos:
                scale = infos["amp_style_scale"]
                if isinstance(scale, torch.Tensor):
                    # Stay on GPU; detach only. Stacked later on the same device.
                    self._style_scale_buffer.append(scale.detach().float().reshape(-1))
                else:
                    self._style_scale_buffer.append(
                        torch.full(
                            (rewards.shape[-1] if rewards.ndim > 1 else 1,),
                            float(scale),
                            device=self.device,
                        )
                    )

            # Flight-success metrics (EMA already computed in the env; Python floats).
            for key in (
                "flight_success_rate",
                "flight_air_rate",
                "clean_takeoff_rate",
                "left_takeoff_rate",
                "landing_success_rate",
                "apex_success_rate",
                "rise_ok_rate",
                "push_fold_ok_rate",
                "swing_ext_ok_rate",
                "hurdle_ok_rate",
                "splits_rate",
                "distance_success_rate",
                "stable_success_rate",
                "jump_curriculum_progress",
                "apex_pitch_ok_rate",
                "mean_peak_sole_over_h",
                "mean_apex_pitch",
                "mean_apex_asym",
                "mean_sep_max",
                "mean_arm_span_apex",
                "mean_pelvis_rise",
            ):
                if key in infos:
                    self.track_data(f"Jump / {key}", float(infos[key]))

            # Termination / reward episode extras promoted to tensors by AmpManagerBasedRLEnv.
            log = infos.get("log")
            if isinstance(log, dict):
                for k, v in log.items():
                    if isinstance(v, torch.Tensor) and (
                        k.startswith("Episode_Termination/") or k.startswith("Episode_Reward/")
                    ):
                        self.track_data(k, float(v.detach().item()))

        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

    def _apply_amp_reward_curriculum(self, timestep: int) -> None:
        """Optional time-based weight lerp; no-op when curriculum disabled."""
        sched = self.cfg.get("amp_reward_curriculum") or {}
        if not sched.get("enabled", False):
            return

        steps = max(int(sched.get("steps_to_final", 100000)), 1)
        t = min(1.0, float(timestep) / float(steps))

        task_w = _lerp(
            float(sched.get("initial_task_weight", self.cfg.get("task_reward_weight", 0.9))),
            float(sched.get("final_task_weight", self.cfg.get("task_reward_weight", 0.7))),
            t,
        )
        style_w = _lerp(
            float(sched.get("initial_style_weight", self.cfg.get("style_reward_weight", 0.1))),
            float(sched.get("final_style_weight", self.cfg.get("style_reward_weight", 0.3))),
            t,
        )
        disc_scale = _lerp(
            float(
                sched.get(
                    "initial_discriminator_reward_scale",
                    self.cfg.get("discriminator_reward_scale", 1.0),
                )
            ),
            float(
                sched.get(
                    "final_discriminator_reward_scale",
                    self.cfg.get("discriminator_reward_scale", 2.0),
                )
            ),
            t,
        )

        self._task_reward_weight = task_w
        self._style_reward_weight = style_w
        self._discriminator_reward_scale = disc_scale

        self.track_data("Curriculum / AMP task weight", task_w)
        self.track_data("Curriculum / AMP style weight", style_w)
        self.track_data("Curriculum / AMP disc reward scale", disc_scale)
        self.track_data("Curriculum / AMP progress", t)

    def _update(self, timestep: int, timesteps: int) -> None:
        self._apply_amp_reward_curriculum(timestep)

        rewards = self.memory.get_tensor_by_name("rewards")
        amp_states = self.memory.get_tensor_by_name("amp_states")

        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            amp_logits, _, _ = self.discriminator.act(
                {"states": self._amp_state_preprocessor(amp_states)}, role="discriminator"
            )
            style_reward = -torch.log(
                torch.maximum(
                    1 - 1 / (1 + torch.exp(-amp_logits)),
                    torch.tensor(0.0001, device=self.device),
                )
            )
            style_reward *= self._discriminator_reward_scale
            style_reward = style_reward.view(rewards.shape)

            # Phase-weight style: takeoff/flight steps contribute less style reward.
            if self._style_scale_buffer:
                rollouts = int(self.cfg.get("rollouts", 32))
                if len(self._style_scale_buffer) > rollouts:
                    self._style_scale_buffer = self._style_scale_buffer[-rollouts:]
                stacked = torch.stack(self._style_scale_buffer, dim=0).to(
                    device=style_reward.device, dtype=style_reward.dtype
                )
                self._style_scale_buffer.clear()
                if stacked.numel() == style_reward.numel():
                    style_reward = style_reward * stacked.reshape_as(style_reward)
                else:
                    # Shape mismatch (e.g. partial rollout) — fall back to mean scale.
                    style_reward = style_reward * stacked.mean()
                self.track_data("Curriculum / AMP style scale (mean)", float(stacked.mean().item()))

        task_reward = rewards
        total_reward = self._task_reward_weight * task_reward + self._style_reward_weight * style_reward

        self.track_data("Reward / Task reward (mean)", task_reward.mean().item())
        self.track_data("Reward / Style reward (mean)", style_reward.mean().item())
        self.track_data("Reward / Total reward (mean)", total_reward.mean().item())

        # Bake phase-scaled combined reward into memory and disable the parent's
        # task/style mix so it does not recompute style without the phase scale.
        self.memory.set_tensor_by_name("rewards", total_reward)
        saved_task_w = self._task_reward_weight
        saved_style_w = self._style_reward_weight
        self._task_reward_weight = 1.0
        self._style_reward_weight = 0.0
        try:
            super()._update(timestep, timesteps)
        finally:
            self._task_reward_weight = saved_task_w
            self._style_reward_weight = saved_style_w
