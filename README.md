# Humanoid G1 Run-Jump

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.1-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![skrl](https://img.shields.io/badge/skrl-%3E%3D1.4.3-orange.svg)](https://skrl.readthedocs.io/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

Isaac Lab extension for hierarchical Unitree G1 run / jump with AMP and a
frozen BeyondMimic tracker.

```
HL (stub) ──select──► run_policy / jump_policy ──64-D frame──► frozen tracker ──PD──► G1
                              ▲
                         AMP discriminator
```

| Tier | Role |
|---|---|
| `run_policy` | Velocity-commanded locomotion (trained here, obstacle-free) |
| `jump_policy` | Jump from a frozen-run hand-off; command `(h_obstacle, flight_distance)` |
| `hl_policy` | Switches run/jump on an obstacle course (**stub**; future work) |
| Frozen tracker | Maps 64-D reduced-coords frame → 29-D joint PD targets (never updated) |

## Registered environments

| ID | Description |
|---|---|
| `Nepher-G1-Run-v0` / `-Play-v0` | Obstacle-free velocity run + AMP |
| `Nepher-G1-Jump-v0` / `-Play-v0` | Run-in hand-off jump + AMP |
| `Nepher-G1-RunJump-v0` / `-Play-v0` | Obstacle course scaffold (HL stub) |

## Installation

Requires [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
and `skrl>=1.4.3`.

From this project root (`task-humanoid-run-jump/`):

```bash
# Editable install
isaaclab.bat -p -m pip install -e source/humanoid_run_jump
pip install "skrl>=1.4.3"

# Frozen BeyondMimic tracker → frozen_policies/policy.pt
python scripts/export_frozen_policy.py --src <path/to/exported/policy.pt>

# AMP datasets from motions/*.csv
python scripts/build_amp_dataset.py

# Run-in hand-off bank (needs best_policy/run/policy.pt)
# Default: random lin_vel_x in [0.5, 3.0] m/s per sample
isaaclab.bat -p scripts/build_runin_bank.py --headless --num_envs 64 --num_samples 4096

# Sanity-check run actor + jump motions + run-in bank
python scripts/validate_jump_runin.py
```

### Required assets

| Path | Purpose |
|---|---|
| `frozen_policies/policy.pt` | BeyondMimic tracker (TorchScript) |
| `best_policy/run/policy.pt` | Frozen run actor for jump hand-off (TorchScript) |
| `motions/packaged/run_motions.pt` | AMP refs for `Nepher-G1-Run-v0` |
| `motions/packaged/jump_motions.pt` | AMP refs for `Nepher-G1-Jump-v0` |
| `motions/packaged/all_motions.pt` | AMP refs for `Nepher-G1-RunJump-v0` |
| `motions/packaged/runin_states.pt` | Jump episode reset bank |

> Tracker and run actor must be **TorchScript** exports, not raw skrl checkpoints.

## Train

```bash
isaaclab.bat -p scripts/skrl/train.py --task Nepher-G1-Run-v0 --headless
isaaclab.bat -p scripts/skrl/train.py --task Nepher-G1-Jump-v0 --headless
```

Checkpoints:

- Run: `logs/skrl/g1_run_amp/<timestamp>_amp_torch/`
- Jump: `logs/skrl/g1_jump_amp/<timestamp>_amp_torch/`

Resume with `--checkpoint <path/to/agent_*.pt>`. Start a **fresh** jump run when
changing reward / AMP logic — do not resume from contaminated checkpoints.

**Retrain checklist (kill post-land double-hop):** do **not** resume the ~22k
run (`bounce_walk`≈0.95, `stable_success`=0). Fresh `Nepher-G1-Jump-v0` only.
After ~15–30k watch:
* `Episode_Termination/bounce_walk` falling from ~0.95 (esp. re-air fail)
* `Episode_Reward/land_absorb` staying **≥ 0**
* `Jump / landing_success_rate` rising; `stable_success_rate` leaving 0
* Play: clear → crouch land → **stay planted** (no second hop)

Useful TensorBoard: `Reward / Style reward (mean)`, `Jump / landing_success_rate`,
`Jump / stable_success_rate`, `Jump / flight_success_rate`,
`Jump / distance_success_rate`, `Jump / flight_air_rate`,
`Episode_Termination/success`, `Episode_Termination/bounce_walk`,
`Episode_Termination/no_liftoff_timeout`, `Episode_Termination/time_out`,
`Episode_Termination/post_land_timeout`, `Episode_Termination/illegal_contact`.

## Play

```bash
isaaclab.bat -p scripts/skrl/play.py --task Nepher-G1-Run-Play-v0 --num_envs 16

isaaclab.bat -p scripts/skrl/play.py --task Nepher-G1-Jump-Play-v0 --num_envs 1
# Default play command: h=0.40 m, flight_distance=0.70 m
isaaclab.bat -p scripts/skrl/play.py --task Nepher-G1-Jump-Play-v0 --num_envs 1 --h 0.50 --flight 1.00
```

If `--checkpoint` is omitted, the latest run under `logs/skrl/<experiment>/` is
used. Play also exports TorchScript next to the checkpoint
(`checkpoints/exported/policy.pt`).

## Jump task

Episodes start from a precomputed **run-in state bank** (frozen run actor
snapshots at right-foot-forward contact). The jump actor never sees the run-in
steps. On reset, each sample is re-anchored to the env origin with canonical `+x`
heading.

### Command

Policy command: `(h_obstacle, flight_distance)`. Policy obs is **156-D**
(64 proprio + 22 jump extras + contacts/air-time/height/vz + 64 last action).

| | Easy (train start) | Envelope (final) | Play |
|---|---|---|---|
| `h_obstacle` | 0.25–0.50 m | 0.25–0.75 m | 0.40 m |
| `flight_distance` | 0.50–0.90 m | 0.50–1.20 m | 0.70 m |

Heights and flight caps are CSV-fitted to the G1 motion frame
(`jump_envelope.py`). Do not command above 0.75 m / beyond the height-coupled
flight cap without new motion data. Re-measure with:

```bash
python scripts/measure_jump_clips.py
```

### Rewards (8 terms)

Task reward specifies **what** to do; AMP + hard terminations specify **how**
(poses in `images/*.png`):

| Term | Weight | Role |
|---|---|---|
| `takeoff` | 10 | Coil knees + approach `(v_x*, v_z*)` and lift off |
| `obstacle_clearance` | 12 | Two-sided sole progress to `h` (penalize balloon past `1.25×h`) |
| `flight_distance` | 18 | Heading-aligned progress toward `d` (**cut after land**) |
| `land_absorb` | 14 | Pure-positive crouch → stand + plant-hold; no bounce subtraction |
| `heading_keep` | 2 | Stay yaw-aligned (**cut after land**) |
| `success_clear` | 40 | Sparse: sole ≥ `h` **and** distance ≥ command |
| `success_land_stable` | 60 | Sparse: idle stand after soft land |
| `action_rate` | −0.01 | Smooth actions |

### Phase machine

```
PREP → PLANT → PUSH → RISE → EXTEND → LAND → IDLE
```

Right-foot hand-off → left plant / push → flight → land → idle. Measurement
arms after left plant; right-foot takeoff terminates. Soft-land is pitch-aware
and **requires a crouch plant** (knee flex ≥ 0.75 or pelvis ≤ 0.72 m); plant
allows horiz ≤ 1.15 m/s so first contact can latch while shedding speed. Idle
is looser (horiz ≤ 0.55 m/s) and requires soft land first. `bounce_walk` splits
fails: **re-air** after ~0.32 s (kills the second hop); **fast horiz** only after
~1.4 s (lets absorb shed speed). `flight_distance` is cut after first land so a
hop cannot re-farm progress. `heading_keep` is zeroed after land.
`no_liftoff_timeout` (~2 s) kills PLANT stand-forever farm. Balloon flights
hard-fail above `apex_rise* + 0.20 m`. Torso/knee/arm contact terminates
(kills back-slam).

### Success / curriculum

Sparse success: `success_clear` (rise + distance) and `success_land_stable`
(idle). Full episode truncate (`flight_success`) requires rise+pitch apex +
distance + soft land + stable idle. **Fold/hurdle is AMP style only** — it is
logged (`hurdle_ok_rate`) but does not gate success.

Curriculum expands the command band when
`landing_success_rate ≥ 0.15` and `jump_curriculum_progress ≥ 0.35`
(`0.6 × landing + 0.4 × distance`).

Useful TensorBoard tags under `Jump /`: `landing_success_rate`,
`stable_success_rate`, `flight_success_rate`, `distance_success_rate`,
`flight_air_rate`, `jump_curriculum_progress`.

### AMP

Reference windows are sampled at the **control period** (`step_dt` = 20 ms),
time-interpolated from the 120 Hz motion pack, so history matches the policy
AMP buffer. Style scale is 1.0 in every phase. Mix:
`task_reward_weight` 0.65 / `style_reward_weight` 0.35 on jump-only motions.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/skrl/train.py` | Train AMP policies |
| `scripts/skrl/play.py` | Evaluate + export TorchScript |
| `scripts/export_frozen_policy.py` | Install BeyondMimic tracker |
| `scripts/build_amp_dataset.py` | Package CSV motions → `.pt` AMP datasets |
| `scripts/build_runin_bank.py` | Build jump hand-off state bank |
| `scripts/validate_jump_runin.py` | Offline sanity checks (no Sim) |
| `scripts/measure_jump_clips.py` | FK-measure jump CSV envelopes |

## Interfaces

### Frozen tracker

- Path: `frozen_policies/policy.pt`
- Input `157 = 64 (proprio) + 64 (target frame) + 29 (prev action)`
- Output: 29-D raw → BeyondMimic PD: `q = default_pose + (effort/stiffness) * raw`

### Frozen run actor

- Path: `best_policy/run/policy.pt`
- Input `134 = 64 (proprio) + 3 (base_lin_vel) + 3 (velocity_cmd) + 64 (last action)`
- Output: 64-D target frame (normalization baked into TorchScript)

### High-level stub

[`hl_stub.py`](source/humanoid_run_jump/humanoid_run_jump/tasks/manager_based/run_jump/hl_stub.py)
must pass the same 2-D jump command when selecting jump mode and should prefer
a tracker-favorable gait phase for the hand-off.

## License

Copyright (c) 2026, Nepher Robotics. All rights reserved.  
Distributed under the [BSD 3-Clause License](LICENSE).
