# Humanoid G1 Run-Jump

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.1-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![skrl](https://img.shields.io/badge/skrl-%3E%3D1.4.3-orange.svg)](https://skrl.readthedocs.io/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

Isaac Lab extension for Unitree G1 run / jump with AMP and a frozen BeyondMimic tracker.

```
run_policy / jump_policy ──64-D frame──► frozen tracker ──PD──► G1
                 ▲
            AMP discriminator
```

| Env | Role |
|---|---|
| `Nepher-G1-Run-v0` / `-Play-v0` | Velocity run + AMP |
| `Nepher-G1-Jump-v0` / `-Play-v0` | Run-in hand-off jump + AMP |
| `Nepher-G1-RunJumpHL-v0` / `-Play-v0` | Obstacle course hard-coded HL (frozen tracker + run + jump; no HL ANN) |

## Setup

Requires [Isaac Lab](https://isaac-sim.github.io/IsaacLab) and `skrl>=1.4.3`.

```bash
# From task-humanoid-run-jump/
isaaclab.bat -p -m pip install -e source/humanoid_run_jump
pip install "skrl>=1.4.3"

python scripts/export_frozen_policy.py --src <path/to/exported/policy.pt>
python scripts/build_amp_dataset.py
isaaclab.bat -p scripts/build_runin_bank.py --headless --num_envs 64 --num_samples 4096
python scripts/validate_jump_runin.py
```

| Asset | Purpose |
|---|---|
| `frozen_policies/policy.pt` | BeyondMimic tracker (TorchScript, frozen) |
| `best_policy/run/policy.pt` | Frozen run actor (TorchScript) |
| `best_policy/jump/policy.pt` | Frozen jump actor (TorchScript) |
| `motions/packaged/run_motions.pt` | Run AMP refs |
| `motions/packaged/jump_motions.pt` | Jump AMP refs |
| `motions/packaged/runin_states.pt` | Jump reset bank |

## Train

```bash
isaaclab.bat -p scripts/skrl/train.py --task Nepher-G1-Run-v0 --headless
isaaclab.bat -p scripts/skrl/train.py --task Nepher-G1-Jump-v0 --headless
```

Logs: `logs/skrl/g1_run_amp/` and `logs/skrl/g1_jump_amp/`.  
Resume with `--checkpoint <agent_*.pt>`. Start fresh after reward / AMP changes.

**RunJumpHL does not train** — the high-level layer is hard-coded (no rewards, no ANN).

## Play

```bash
isaaclab.bat -p scripts/skrl/play.py --task Nepher-G1-Run-Play-v0 --num_envs 16
isaaclab.bat -p scripts/skrl/play.py --task Nepher-G1-Jump-Play-v0 --num_envs 1
isaaclab.bat -p scripts/skrl/play.py --task Nepher-G1-Jump-Play-v0 --num_envs 1 --h 0.50 --flight 1.00
# HL: hard-coded lattice controller (no checkpoint)
isaaclab.bat -p scripts/play_hl_heuristic.py --task Nepher-G1-RunJumpHL-Play-v0 --num_envs 4
```

Defaults for jump play: `h=0.40`, `flight=0.70`. Latest checkpoint is used if `--checkpoint` is omitted; play also exports TorchScript next to it.

## Jump (overview)

Episodes start from a **run-in state bank** (frozen run snapshots). Command: `(h_obstacle, flight_distance)`.

| | Train start | Train end | Play default |
|---|---|---|---|
| `h_obstacle` | 0.25–0.50 m | 0.25–0.75 m | 0.40 m |
| `flight_distance` | 0.50–0.90 m | 0.50–1.20 m | 0.70 m |

Phases: `PREP → PLANT → PUSH → RISE → EXTEND → LAND → IDLE`.

Main rewards: takeoff, obstacle clearance, apex fold, flight distance, land absorb, heading, sparse clear/land-stable. Style (tuck / hurdle pose) is mostly AMP; `hurdle_ok_rate` is logged only.

AMP mix (jump): `task_reward_weight` 0.60 / `style_reward_weight` 0.40.

## RunJumpHL (overview)

Hard-coded high-level controller over three **frozen** TorchScript models (not trained here):

```
LatticeHardcodedHL ──[gate,vx,vy,wz]──► HierarchicalSwitchAction
        │                                    │
        │                              frozen run / jump actor
        │                                    │
        └──────────────────────────── frozen tracker ──► G1
```

No HL neural net and **no HL rewards**. The controller:

1. Tracks right-plant stride length ``L`` and targets lattice speed ``v*`` so ``n · L(v*) ≈ D`` (distance from right foot to takeoff centre).
2. Freezes ``vx`` in the last stride before the stamp.
3. Raises ``gate > 0`` when the right foot is inside the measured takeoff window; the switcher latches JUMP on run-in hand-off + in-band plant, then returns to RUN after landing contact.

Course geometry is **full user-spec** (up to 5 obstacles, h 0.20–0.75 m, path 12–36 m). Obstacles are solid; the jump gate is always enabled (no skill curriculum).

| Parameter | Range |
|---|---|
| Path | Straight 3 m-wide corridor along +x, length 12–36 m |
| First obstacle | 5–8 m from start |
| Inter-obstacle gap | 6–8 m |
| Last obstacle → goal | ≥ 1 m |
| Max obstacles | 5 (fits the spacing on a ≤36 m path) |
| Obstacle thickness | 0.10–0.20 m (10 fixed-size prims, permuted) |
| Obstacle length (lateral) | 3.0 m (matches corridor width) |
| Obstacle yaw | 0° (axis-aligned) |

HL command is `[gate, vx, vy, wz]` at ~10 Hz (`action_repeat=5`), with `vx ∈ [1.0, 3.0]` m/s. On RUN→JUMP latch, `h_obstacle` and `flight_distance` are **derived** from sensed obstacle height (+0.06 m clearance) and snapshotted right-foot plant distance (+ thickness + 0.12 m land margin), then clamped through `flight_cap(h)`. Resets use the frozen-run hand-off bank.

Episodes terminate on leaving the path (`out_of_path`), sustained obstacle contact (`obstacle_crash`, ~12 steps), retreat behind start, no +x progress within 4 s (`no_progress`). `bad_orientation` is exempt during JUMP and for ~25 control steps after landing; limit is 1.4 rad.

Policy obs remain 106-D for logging / debugging (unused by the hard-coded controller).

Takeoff window is **apex-centred** (measured plant→pelvis-apex p25/p50/p75 ≈ 0.745 / 0.891 / 1.034 m), offset by half-thickness, capped by the landing-span budget (p25 plant→land ≈ 1.43 m):

```
near = max(0.55, 0.745 − t/2)
far  = min(1.034 − t/2, 1.43 − t − 0.12)
```

At `t=0.15` this is roughly `[0.67, 0.96]` (width ≈ 0.29 m).

## Scripts

| Script | Purpose |
|---|---|
| `scripts/skrl/train.py` | Train |
| `scripts/skrl/play.py` | Play + export |
| `scripts/play_hl_heuristic.py` | HL hard-coded lattice play (no ANN) |
| `scripts/export_frozen_policy.py` | Install tracker |
| `scripts/build_amp_dataset.py` | Package AMP motions |
| `scripts/build_runin_bank.py` | Jump hand-off bank |
| `scripts/validate_jump_runin.py` | Offline checks |
| `scripts/measure_jump_takeoff_geometry.py` | Roll out frozen jump_policy; export plant→peak / plant→land geometry CSV |

## License

Copyright (c) 2026, Nepher Robotics. All rights reserved.  
Distributed under the [BSD 3-Clause License](LICENSE).
