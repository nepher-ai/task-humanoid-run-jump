# Unitree G1 Run and Jump in NVIDIA Isaac Lab

**AMP locomotion · hurdle jumping · BeyondMimic tracking · hierarchical PPO**

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.1-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![skrl](https://img.shields.io/badge/skrl-%3E%3D1.4.3-orange.svg)](https://skrl.readthedocs.io/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

**task-humanoid-run-jump** is a [Nepher Robotics](https://github.com/nepher-ai) extension for [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab) / [Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html). It trains **Unitree G1** humanoid policies to **run and jump obstacle-course hurdles**: [Adversarial Motion Priors (AMP)](https://xbpeng.github.io/projects/AMP/index.html) run and jump specialists, a frozen [BeyondMimic](https://beyondmimic.github.io/) whole-body tracker, and a PPO **high-level switcher** that picks run vs jump on a straight course.

Use this repository when you need Isaac Lab Gym environments for G1 **running**, **hurdle jumping**, or a **hierarchical run/jump stack** (`Nepher-G1-Run-*`, `Nepher-G1-Jump-*`, `Nepher-G1-RunJumpHL-*`, EnvHub `humanoid-runjump-course-v1`).

Agents: start at [`llms.txt`](llms.txt) (index) or [`llms-full.txt`](llms-full.txt) (single-file context). Query-shaped answers: [`docs/`](docs/README.md).

The frozen tracker is trained in [humanoid-g1-tracking](https://github.com/nepher-ai/humanoid-g1-tracking). Reference motion clips come from [bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed) on Hugging Face.

| | |
|---|---|
| **Robot** | Unitree G1 (29-DoF whole-body) |
| **Simulator** | NVIDIA Isaac Sim 5.1.0 + Isaac Lab 2.3.1 |
| **Algorithms** | skrl AMP (run, jump) · skrl PPO (high-level) |
| **HL stack** | PPO 6-D switcher → frozen AMP actor → frozen tracker 50 Hz → PhysX 200 Hz |
| **Tracker** | BeyondMimic, 157-D obs → 29 joint PD targets (`nepher-ai/humanoid-g1-tracking`) |
| **Motions** | Hugging Face [`bones-studio/seed`](https://huggingface.co/datasets/bones-studio/seed) |
| **Course** | Straight path 20–30 m, up to 3 hurdles, height 0.20–0.75 m |
| **Eval** | Procedural hurdles or deterministic EnvHub `humanoid-runjump-course-v1` |
| **License** | BSD 3-Clause |

## Who this is for

- Researchers training **humanoid RL** on the Unitree G1 in Isaac Lab who need **running plus jumping**, not velocity tracking alone.
- Teams composing **hierarchical frozen policies**: a high-level agent that only outputs a 6-D switch (`gate`, `vx`, `vy`, `ωz`, hurdle height, flight) while frozen AMP actors and a frozen tracker write the 29 joints.
- Benchmarks that must be **seed-reproducible** — EnvHub loads fixed standing-start hurdle courses.
- AI agents and search tools looking for an Isaac Lab G1 **obstacle course**, **hurdle jump**, **AMP locomotion**, or **BeyondMimic tracker** task with registered Gym IDs.

This repo is **not** a sim-to-real deploy kit, not a quadruped stack, not a vision-based navigator, and not the waypoint-racing stack in [task-humanoid-run-waypoints](https://github.com/nepher-ai/task-humanoid-run-waypoints). Observations are proprioception plus course / jump commands.

## Policy hierarchy

Most G1 Isaac Lab examples stop at tracking or velocity commands. This project **composes** run and jump specialists:

```
HL PPO switcher      6-D [gate, vx, vy, ωz, h, flight]     trainable
        │
        ▼
Frozen AMP actor     run (134-D) or jump (156-D) → 64-D frame    skrl AMP / TorchScript
        │
        ▼
Frozen tracker       157-D obs → 29-D joint PD targets     50 Hz   BeyondMimic
        │
        ▼
PhysX                Unitree G1 articulation              200 Hz
```

The high-level layer lives in one Isaac Lab `ActionTerm` (`HierarchicalSwitchAction`). `gate ≤ 0` keeps the run actor (velocity command). A rising gate plus a right-foot plant hands off to the jump actor with `(h_obstacle, flight_distance)`. After a stable landing the stack returns to run.

Train bottom-up: **tracker (external) → run AMP → jump AMP → HL PPO**.

## Gym environments

| ID | Backend | Role |
|---|---|---|
| `Nepher-G1-Run-v0` | skrl AMP | Velocity running (train) |
| `Nepher-G1-Run-Play-v0` | skrl AMP | Velocity running (eval) |
| `Nepher-G1-Jump-v0` | skrl AMP | Run-in hand-off jump (train) |
| `Nepher-G1-Jump-Play-v0` | skrl AMP | Run-in hand-off jump (eval) |
| `Nepher-G1-RunJumpHL-v0` | skrl PPO | HL obstacle course (train) |
| `Nepher-G1-RunJumpHL-Play-v0` | skrl PPO | HL obstacle course (eval) |
| `Nepher-G1-RunJumpHL-Envhub-v0` | skrl PPO | EnvHub fixed courses (train) |
| `Nepher-G1-RunJumpHL-Envhub-Play-v0` | skrl PPO | EnvHub fixed courses (eval) |

## Installation

Requires [Isaac Lab v2.3.1](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) and `skrl>=1.4.3`.

1. Clone and install this package (from the project root):

```bash
git clone https://github.com/nepher-ai/task-humanoid-run-jump.git
cd task-humanoid-run-jump
isaaclab.bat -p -m pip install -e source/humanoid_run_jump
pip install "skrl>=1.4.3"
```

2. Install a **tracker** trained in [humanoid-g1-tracking](https://github.com/nepher-ai/humanoid-g1-tracking). Export TorchScript there with `export_jit.py`, then copy it here:

```bash
python scripts/export_frozen_policy.py --src <path/to/exported/policy.pt>
```

3. Package AMP motions from [bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed) CSVs (place clips under `motions/`), then optionally build a local run-in bank for jump training:

```bash
python scripts/build_amp_dataset.py
isaaclab.bat -p scripts/build_runin_bank.py --headless --num_envs 64 --num_samples 4096
```

| Asset | Purpose |
|---|---|
| `frozen_policies/tracker.pt` | Frozen BeyondMimic tracker from [humanoid-g1-tracking](https://github.com/nepher-ai/humanoid-g1-tracking) |
| `best_policy/run/policy.pt` | Frozen AMP run actor |
| `best_policy/jump/policy.pt` | Frozen AMP jump actor |
| `best_policy/best.pt` (or `best_policy.pt`) | HL PPO for play / eval |
| `motions/packaged/*.pt` | AMP packs from [bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed) + local `runin_states.pt` |

## Train AMP run

Velocity-command running with an AMP discriminator on seed run clips:

```bash
isaaclab.bat -p scripts/skrl/train.py --task Nepher-G1-Run-v0 --headless
isaaclab.bat -p scripts/skrl/play.py  --task Nepher-G1-Run-Play-v0 --num_envs 16
```

The run actor writes a 64-D reduced-coords target frame; the frozen tracker turns that into 29 joint PD targets.

## Train AMP jump

Run-in hand-off jump over a commanded obstacle height and flight distance. Jump play defaults: `h=0.40`, `flight=0.70` (`--h` / `--flight` to override).

```bash
isaaclab.bat -p scripts/skrl/train.py --task Nepher-G1-Jump-v0 --headless
isaaclab.bat -p scripts/skrl/play.py  --task Nepher-G1-Jump-Play-v0 --num_envs 1
```

## High-level obstacle course

PPO over the three frozen models. Recommended order: **run → jump → HL**. Resume with `--checkpoint <agent_*.pt>`.

HL resets are **standing only** (`bank_prob=0`). The first 2 s of `stall_cost` are waived so the frozen run policy can accelerate. Procedural course: path 20–30 m, up to 3 hurdles, height 0.20–0.75 m.

```bash
isaaclab.bat -p scripts/skrl/train.py --task Nepher-G1-RunJumpHL-v0 --algorithm PPO --headless
isaaclab.bat -p scripts/skrl/play.py  --task Nepher-G1-RunJumpHL-Play-v0 --algorithm PPO --num_envs 4
```

Add `--video` to record. Local HL play uses a **standing** start.

## EnvHub evaluation

Bundle: `humanoid-runjump-course-v1` (64 fixed scenarios, 1–4 hurdles today; scene capacity up to 10). Robots start **standing**.

Tournament scoring is via [eval-nav](../eval-nav) (`navigation.humanoid.runjump` v1): success-rate × (time + clearance/landing + safety/energy). No AMP discriminator and no EnvHub run-in bank.

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
| `scripts/export_frozen_policy.py` | Install tracker from humanoid-g1-tracking |
| `scripts/build_amp_dataset.py` | Package AMP motions from bones-studio/seed CSVs |
| `scripts/build_runin_bank.py` | Local run-in bank for jump train |
| `scripts/validate_jump_runin.py` | Offline bank checks |

## Query pages

Each file answers one search. Prefer these over the whole README when the question is specific.

| Query | Page |
|---|---|
| Isaac Lab Unitree G1 hurdle jumping | [docs/isaac-lab-unitree-g1-hurdle-jumping.md](docs/isaac-lab-unitree-g1-hurdle-jumping.md) |
| Hierarchical AMP run G1 | [docs/isaac-lab-g1-amp-run.md](docs/isaac-lab-g1-amp-run.md) |
| G1 AMP jump / run-in hand-off | [docs/isaac-lab-g1-amp-jump.md](docs/isaac-lab-g1-amp-jump.md) |
| Hierarchical frozen policy stack | [docs/hierarchical-frozen-policy-stack.md](docs/hierarchical-frozen-policy-stack.md) |
| EnvHub `humanoid-runjump-course-v1` | [docs/envhub-humanoid-runjump-course.md](docs/envhub-humanoid-runjump-course.md) |
| Gym IDs `Nepher-G1-Run*` / `Jump*` / `RunJumpHL*` | [docs/nepher-g1-gym-ids.md](docs/nepher-g1-gym-ids.md) |
| BeyondMimic tracker / `tracker.pt` | [docs/isaac-lab-g1-beyondmimic-tracker.md](docs/isaac-lab-g1-beyondmimic-tracker.md) |
| bones-studio/seed AMP motions | [docs/bones-studio-seed-amp-motions.md](docs/bones-studio-seed-amp-motions.md) |
| Run-jump vs waypoint race | [docs/isaac-lab-g1-run-jump-vs-waypoint-race.md](docs/isaac-lab-g1-run-jump-vs-waypoint-race.md) |

Index: [docs/README.md](docs/README.md).

## FAQ

**What robot and simulator does this target?**
Unitree G1 (29-DoF) in NVIDIA Isaac Sim 5.1.0 with Isaac Lab 2.3.1 (Python 3.11, skrl ≥ 1.4.3).

**How is this different from a standard Isaac Lab G1 velocity task?**
Velocity run is included, but the jump and HL tasks stack **frozen AMP specialists** on a **frozen BeyondMimic tracker**. The high-level policy never writes joint targets; it only gates run vs jump and sets velocity / hurdle commands.

**Where does the tracker come from?**
Train and export it in [nepher-ai/humanoid-g1-tracking](https://github.com/nepher-ai/humanoid-g1-tracking) (`export_jit.py`), then `scripts/export_frozen_policy.py` installs `frozen_policies/tracker.pt`.

**Where does the motion data come from?**
Reference CSVs are from Hugging Face [bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed). `scripts/build_amp_dataset.py` packages them into `motions/packaged/*.pt`.

**Which Gym ID should I start with?**
Running: `Nepher-G1-Run-v0`. Jumping: `Nepher-G1-Jump-v0`. Full hurdle course: `Nepher-G1-RunJumpHL-v0`. Reproducible eval: `Nepher-G1-RunJumpHL-Envhub-Play-v0`.

**Is this sim-to-real?**
No. It is a simulation training and benchmark stack (procedural course + EnvHub). There is no hardware deploy script in this repo.

**Is this the waypoint racing repo?**
No. Waypoint racing lives in [task-humanoid-run-waypoints](https://github.com/nepher-ai/task-humanoid-run-waypoints). This repo is run + jump over hurdles.

## License

Copyright (c) 2026, [Nepher Robotics](https://github.com/nepher-ai). All rights reserved.
Distributed under the [BSD 3-Clause License](LICENSE).

**Keywords:** Unitree G1, humanoid robot, whole-body control, reinforcement learning, NVIDIA Isaac Lab, Isaac Sim, AMP, Adversarial Motion Priors, BeyondMimic, motion tracking, hurdle jump, obstacle course, hierarchical policy, frozen policy stack, PPO, skrl, EnvHub, bones-studio/seed, humanoid-g1-tracking.
