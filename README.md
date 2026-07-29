# Humanoid G1 Run-Jump

Hierarchical run / jump for Unitree G1 with a frozen BeyondMimic tracker + AMP.

Self-contained Isaac Lab **external** project. Three policy tiers:

1. **`run_policy`** — velocity-commanded locomotion (trained here, obstacle-free).
2. **`jump_policy`** — jump from a **frozen-run-actor hand-off** at a fixed gait phase,
   commanded by `(h_obstacle, flight_distance)`.
3. **`hl_policy`** — switches between run/jump on an obstacle course (**stubbed**; future work).

Low-level policies output a **64-D reduced-coords target frame**. A **frozen** BeyondMimic
tracker (`frozen_policies/policy.pt`) converts that frame into 29-D joint PD targets.
The tracker is never updated.

```
HL (stub) ──select──► run_policy / jump_policy ──64-D frame──► frozen tracker ──PD──► G1
                              ▲
                         AMP discriminator
```

## Layout

```
task-humanoid-run-jump/
  best_policy/run/policy.pt      # Frozen exported run actor (TorchScript)
  frozen_policies/policy.pt      # BeyondMimic tracker (TorchScript)
  motions/*.csv                  # bones-speed clips @ 120 Hz
  motions/packaged/*.pt          # AMP datasets + runin_states.pt
  scripts/skrl/{train,play}.py
  scripts/build_amp_dataset.py
  scripts/build_runin_bank.py
  scripts/validate_jump_runin.py
  scripts/export_frozen_policy.py
  source/humanoid_run_jump/      # installable Isaac Lab extension
```

## Install

From this project root (`task-humanoid-run-jump/`):

```bash
%sim% -m pip install -e source/humanoid_run_jump
%sim% scripts/export_frozen_policy.py
%sim% scripts/build_amp_dataset.py
isaaclab.bat -p scripts/build_runin_bank.py --headless --num_envs 64 --num_samples 4096
python scripts/validate_jump_runin.py
```

Packaged datasets:

| File | Used by | Notes |
|---|---|---|
| `motions/packaged/run_motions.pt` | `Nepher-G1-Run-v0` | run clips only |
| `motions/packaged/jump_motions.pt` | `Nepher-G1-Jump-v0` | jump clips only |
| `motions/packaged/all_motions.pt` | `Nepher-G1-RunJump-v0` | run+jump; jump clips weighted ×2 |
| `motions/packaged/runin_states.pt` | `Nepher-G1-Jump-v0` | frozen-run hand-off bank |

## Registered environments

| ID | Description |
|---|---|
| `Nepher-G1-Run-v0` / `-Play-v0` | Obstacle-free velocity run + AMP |
| `Nepher-G1-Jump-v0` / `-Play-v0` | Run-in hand-off jump + AMP |
| `Nepher-G1-RunJump-v0` / `-Play-v0` | Obstacle course scaffold (HL stub) |

## Train

```bash
isaaclab.bat -p scripts/skrl/train.py --task Nepher-G1-Run-v0 --headless
isaaclab.bat -p scripts/skrl/train.py --task Nepher-G1-Jump-v0 --headless
```

Useful TensorBoard: `Jump / flight_success_rate`, `Jump / apex_success_rate`,
`Jump / rise_ok_rate`, `Jump / tuck_ok_rate`, `Jump / left_tuck_ok_rate`,
`Jump / right_tuck_ok_rate`, `Jump / splits_rate`, `Jump / distance_success_rate`,
`Jump / landing_success_rate`, `Jump / stable_success_rate`,
`Jump / left_takeoff_rate`, `Jump / flight_air_rate`,
`Jump / jump_curriculum_progress`, `Jump / apex_pitch_ok_rate`,
`Jump / mean_peak_sole_over_h`, `Jump / mean_apex_pitch`.

## Play

```bash
isaaclab.bat -p scripts/skrl/play.py --task Nepher-G1-Jump-Play-v0 --num_envs 1
# Fixed play command: h=0.50 m, flight_distance=1.0 m
```

## Interfaces

### Frozen tracker

* Input `157 = 64 (proprio) + 64 (target frame) + 29 (prev action)`
* Output: 29-D raw → BeyondMimic PD: `q = default_pose + (effort/stiffness) * raw`

### Frozen run actor

* Path: `best_policy/run/policy.pt`
* Input `134 = 64 (proprio) + 3 (base_lin_vel) + 3 (velocity_cmd) + 64 (last action)`
* Output: 64-D target frame (normalization baked into the TorchScript export)

### Jump command (`h_obstacle`, `flight_distance`)

Policy-facing command / obs extras (22-D; policy obs total **156**):

```
command: [h_obstacle, flight_distance]
obs extras: heading_error, progress, v_x*, v_z*, crouch*,
            apex_rise*, tuck_apex*, pitch_apex*,
            peak_pelvis_rise_so_far, tuck_l, tuck_r,
            phase_one_hot(7), has_liftoff, has_landed
```

* `h_obstacle` — height in **meters** (G1 motion frame), clamped to `[0, 0.75]`;
  used as the sole-clearance success gate (peak sole ≥ `h`); excess above
  `h + 0.05` is penalized; peak sole > `h + 0.25` hard-fails (`balloon_height`)
* `flight_distance` — required heading-aligned liftoff→landing distance (m),
  clamped to `[0.50, 1.20]` and further capped by height via `flight_cap(h)`
* `(v_x*, v_z*, crouch*)` — takeoff envelope refs from `JumpEnvelope.map(h)`
* `(apex_rise*, tuck_apex*, pitch_apex*)` — soft obs refs (physics-reachable
  pelvis rise; max per-leg tuck; forward torso pitch ~0.18–0.20 rad). Apex
  *success* requires sole clearance ≥ `h`, both-leg tuck, **and** forward
  pitch in `[0.08, 0.40]` rad (vertical tuck no longer counts)
* `(peak_pelvis_rise_so_far, tuck_l, tuck_r)` — live measured progress
  (per-leg tuck; both must meet `tuck_apex*`)

| CSV tag | `h` | `apex_rise*` | `tuck_apex*` | `pitch_apex*` | measured air travel |
|---|---|---|---|---|---|
| 0.50 m | 0.50 m | 0.14 | 0.60 | 0.18 | ≈ 0.60–1.17 m |
| 0.75 m | 0.75 m | 0.16 | 0.55 | 0.20 | ≈ 0.94–0.99 m |

Do not command obstacles above 0.75 m or flight distances beyond the
height-coupled cap without new motion data. Re-derive thresholds with
`python scripts/measure_jump_clips.py`.

#### Run-in hand-off

Episodes start from a precomputed **run-in state bank** built by rolling the
frozen run actor and snapshotting when the **right foot is forward and in
contact** (configurable `--lead`). The jump actor therefore never sees the
run-in steps (on-policy safe).

On reset, each sample is **re-anchored to its env origin** with a canonical
`+x` heading (bank XY drift / yaw from the run-away are discarded; world-frame
velocities are rotated to match). Env spacing is `3.0 m`. Play zeros pose
jitter so robots sit on a clean grid for inspection.

```bash
isaaclab.bat -p scripts/build_runin_bank.py --headless --lead right
```

#### Sampling / curriculum

| | Easy (start) | Envelope (final) | Play |
|---|---|---|---|
| `h_obstacle` | 0.25–0.50 m | 0.25–0.75 m | 0.50 m |
| `flight_distance` | 0.50–0.90 m | 0.50–1.20 m | 1.00 m |

#### Jump phase machine

```
PREP → PLANT → PUSH → RISE → EXTEND → LAND → IDLE
```

* **PREP** — right-foot hand-off; left running step (grounded; hop penalized)
* **PLANT / PUSH** — left plant, CoM over left foot, right tuck, left push
* **RISE** — both airborne ascending; **both** knees toward chest with
  **forward torso pitch** (angular-momentum-correct tuck)
* **EXTEND** — after apex (`vz` +→−); **both** legs open; pitch toward upright
* **LAND / IDLE** — touchdown then recover to idle (re-jump penalized)

Jump measurement arms only after **left plant**. Right-foot takeoff terminates
(`illegal_takeoff`). PREP longer than ~1 s terminates (`prep_timeout`).
Yaw must stay aligned with **episode-start** forward (`heading_keep` /
`heading_blowout`).

Policy obs includes a 7-D phase one-hot + `has_liftoff` / `has_landed` +
per-leg tuck + `pitch_apex*` (total **156-D**).

#### Rewards

Phase-gated dense terms + sparse takeoff-foot / success + regularizers:

**Dense**
* `prep_step` — left swing while grounded in PREP (−airborne, non-sticky)
* `plant_push` — left plant / CoM / right tuck / push
* `takeoff_foot` — sparse `+1` left last-support foot
* `takeoff_launch` — envelope velocity (asymmetric `v_z` overshoot penalty) +
  forward pitch track in PLANT/PUSH and ascending RISE
* `apex_tuck` — both legs ≤ `tuck_apex*` **× forward-pitch scale** (vertical
  tuck pays near-zero)
* `apex_pitch` — dense forward pitch toward `pitch_apex*`
* `foot_clearance` — sole / `h` capped at 1.0 (no 1.2× farming)
* `excess_height` — penalty for sole above `h + 0.05` m
* `leg_extend` — both legs → ~0.60 m on descent after apex
* `extend_pitch` — pitch toward upright on EXTEND/descent
* `foot_split_penalty` — excess foot sep over 0.80 m (weight −50)
* `flight_distance` — heading progress during airborne flight only
* `heading_keep` — yaw aligned with episode-start (down-weighted)
* `land_and_idle` — feet-first touchdown then idle; scaled by sole clearance
* `rejump_penalty` — no second hop after landing

**Sparse success** (once per episode each)
1. **rise**: peak sole clearance ≥ `h_obstacle`
2. **tuck**: **both** legs' min airborne tuck `≤ tuck_apex*`
3. **full apex**: rise **and** tuck **and** forward pitch in band **and**
   max flight foot-sep `≤ 0.90 m`
4. heading-aligned liftoff→landing distance `≥ flight_distance` (also
   requires foot-sep `≤ 0.90 m`)
5. recover to **idle standing** within **3 s after landing** (dense + sparse
   bonus; not required for episode success truncation yet)

Episode success truncation / curriculum unlock = apex **and** distance **and**
soft landing (both feet, pelvis height ≥ ~0.68 m, upright, heading within
~0.35 rad of episode start, held ~5 control steps). Curriculum unlock uses a
soft mix (`jump_curriculum_progress`) of distance / landing / rise / tuck
rates. Kneel contact on `*_knee_link` hard-fails via `illegal_contact`.
Episodes also soft-truncate ~2 s after first landing (`post_land_timeout`).

Hard terminate if in-flight foot-sep exceeds **1.05 m** (`extreme_splits`),
peak sole > `h + 0.25` (`balloon_height`), illegal right-foot takeoff, PREP
timeout, post-liftoff yaw `|heading_error| > 1.2` (`heading_blowout`), or
knee/torso/arm ground slam (`illegal_contact`).

**Regularizers:** `action_rate`, `no_flight_timeout`.

AMP mix is **fixed** (`task_reward_weight` 0.92 / `style_reward_weight` 0.08) with
jump-only motions (`jump_motions.pt`). Env `amp_style_scale` keeps full style
in PREP and suppresses from PUSH onward (and after landing).

### High-level (stub)

[`hl_stub.py`](source/humanoid_run_jump/humanoid_run_jump/tasks/manager_based/run_jump/hl_stub.py)
must pass the **same 2-D command** when selecting jump mode (`h_obstacle`,
`flight_distance`) and should prefer the tracker-favorable gait phase for the
hand-off.

## License

Copyright (c) 2026, Nepher Robotics. All rights reserved.  
Distributed under the [BSD 3-Clause License](LICENSE).
