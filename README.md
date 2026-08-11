# Humanoid G1 Run-Jump

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.1-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![skrl](https://img.shields.io/badge/skrl-%3E%3D1.4.3-orange.svg)](https://skrl.readthedocs.io/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

Unitree G1 **run** and **jump** specialists (AMP + BeyondMimic tracker), plus a
PPO **high-level** policy that switches them on straight obstacle courses.

## Environments

| Env | Role |
|---|---|
| `Nepher-G1-Run-v0` / `-Play-v0` | Velocity running + AMP |
| `Nepher-G1-Jump-v0` / `-Play-v0` | Run-in hand-off jump + AMP |
| `Nepher-G1-RunJumpHL-v0` / `-Play-v0` | HL obstacle course (frozen run + jump + tracker) |
| `Nepher-G1-RunJumpHL-Envhub-v0` / `-Envhub-Play-v0` | EnvHub fixed courses (standing-start eval) |

## Setup

Requires [Isaac Lab](https://isaac-sim.github.io/IsaacLab) and `skrl>=1.4.3`.

Install the BeyondMimic **tracker** from
[humanoid-g1-tracking](https://github.com/nepher-ai/humanoid-g1-tracking), then
package motions (and optionally build a run-in bank for the jump task):

```bash
# From task-humanoid-run-jump/
isaaclab.bat -p -m pip install -e source/humanoid_run_jump
pip install "skrl>=1.4.3"

python scripts/export_frozen_policy.py --src <path/to/exported/policy.pt>
python scripts/build_amp_dataset.py
isaaclab.bat -p scripts/build_runin_bank.py --headless --num_envs 64 --num_samples 4096
```

| Asset | Purpose |
|---|---|
| `frozen_policies/policy.pt` | BeyondMimic tracker |
| `best_policy/run/policy.pt` | Frozen run actor |
| `best_policy/jump/policy.pt` | Frozen jump actor |
| `best_policy/best.pt` (or `best_policy.pt`) | HL policy for play / eval |
| `motions/packaged/*.pt` | AMP motions + local `runin_states.pt` (jump train) |

## Train

Recommended order: **run → jump → HL**.

```bash
isaaclab.bat -p scripts/skrl/train.py --task Nepher-G1-Run-v0 --headless
isaaclab.bat -p scripts/skrl/train.py --task Nepher-G1-Jump-v0 --headless
isaaclab.bat -p scripts/skrl/train.py --task Nepher-G1-RunJumpHL-v0 --algorithm PPO --headless
```

Resume with `--checkpoint <agent_*.pt>`.

**HL resets:** standing only (`bank_prob=0`). First 2 s of `stall_cost` are
waived so the frozen run policy can accelerate. Procedural course: path 20–30 m,
up to 3 hurdles, height 0.20–0.75 m.

## Play

```bash
isaaclab.bat -p scripts/skrl/play.py --task Nepher-G1-Run-Play-v0 --num_envs 16
isaaclab.bat -p scripts/skrl/play.py --task Nepher-G1-Jump-Play-v0 --num_envs 1
isaaclab.bat -p scripts/skrl/play.py --task Nepher-G1-RunJumpHL-Play-v0 --algorithm PPO --num_envs 4
```

Add `--video` to record. Jump play defaults: `h=0.40`, `flight=0.70`
(`--h` / `--flight` to override). Local HL play uses a **standing** start.

## EnvHub evaluation

Bundle: `humanoid-runjump-course-v1` (64 fixed scenarios, 1–4 hurdles today;
scene capacity up to 10). Robots start **standing**.

Tournament scoring is via [eval-nav](../eval-nav) (`navigation.humanoid.runjump`
v1): success-rate × (time + clearance/landing + safety/energy). No AMP
discriminator and no EnvHub run-in bank.

```bash
pip install -e ../envhub

isaaclab.bat -p scripts/skrl/play.py \
  --task Nepher-G1-RunJumpHL-Envhub-Play-v0 --algorithm PPO --num_envs 16
```

## Scripts

| Script | Purpose |
|---|---|
| `scripts/skrl/train.py` | Train (AMP run/jump or HL PPO) |
| `scripts/skrl/play.py` | Play / export |
| `scripts/skrl/play_video.py` | Recorded play helper |
| `scripts/export_frozen_policy.py` | Install tracker |
| `scripts/build_amp_dataset.py` | Package AMP motions |
| `scripts/build_runin_bank.py` | Local run-in bank for jump train |
| `scripts/validate_jump_runin.py` | Offline bank checks |

## License

Copyright (c) 2026, Nepher Robotics. All rights reserved.  
Distributed under the [BSD 3-Clause License](LICENSE).
