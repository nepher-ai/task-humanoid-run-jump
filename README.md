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
| `Nepher-G1-RunJump-v0` / `-Play-v0` | Obstacle course (HL stub) |

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
| `frozen_policies/policy.pt` | BeyondMimic tracker (TorchScript) |
| `best_policy/run/policy.pt` | Run actor for jump hand-off (TorchScript) |
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

## Play

```bash
isaaclab.bat -p scripts/skrl/play.py --task Nepher-G1-Run-Play-v0 --num_envs 16
isaaclab.bat -p scripts/skrl/play.py --task Nepher-G1-Jump-Play-v0 --num_envs 1
isaaclab.bat -p scripts/skrl/play.py --task Nepher-G1-Jump-Play-v0 --num_envs 1 --h 0.50 --flight 1.00
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

## Scripts

| Script | Purpose |
|---|---|
| `scripts/skrl/train.py` | Train |
| `scripts/skrl/play.py` | Play + export |
| `scripts/export_frozen_policy.py` | Install tracker |
| `scripts/build_amp_dataset.py` | Package AMP motions |
| `scripts/build_runin_bank.py` | Jump hand-off bank |
| `scripts/validate_jump_runin.py` | Offline checks |
| `scripts/measure_jump_clips.py` | Measure jump envelopes |

## License

Copyright (c) 2026, Nepher Robotics. All rights reserved.  
Distributed under the [BSD 3-Clause License](LICENSE).
