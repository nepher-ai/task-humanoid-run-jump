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
| `Nepher-G1-RunJumpHL-v0` / `-Play-v0` | Obstacle course PPO HL (frozen tracker + run + jump) |

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

Train the HL with `--task Nepher-G1-RunJumpHL-v0 --algorithm PPO` (three LL models stay frozen).

## Play

```bash
isaaclab.bat -p scripts/skrl/play.py --task Nepher-G1-Run-Play-v0 --num_envs 16
# Video: 3s high overview of all envs, then track 3 robots × ~12s (cut to nearest).
isaaclab.bat -p scripts/skrl/play.py --task Nepher-G1-Run-Play-v0 --num_envs 16 --video
isaaclab.bat -p scripts/skrl/play.py --task Nepher-G1-Jump-Play-v0 --num_envs 1
isaaclab.bat -p scripts/skrl/play.py --task Nepher-G1-Jump-Play-v0 --num_envs 1 --h 0.50 --flight 1.00
# Jump video: same choreography; side-tracking so the obstacle + flight arc stay in frame.
isaaclab.bat -p scripts/skrl/play.py --task Nepher-G1-Jump-Play-v0 --num_envs 16 --video
isaaclab.bat -p scripts/skrl/play.py --task Nepher-G1-RunJumpHL-Play-v0 --algorithm PPO --num_envs 4
```

Defaults for jump play: `h=0.40`, `flight=0.70`. Latest checkpoint is used if `--checkpoint` is omitted; play also exports TorchScript next to it.

With `--video` (see `scripts/skrl/play_video.py`):
1. **3 s** high overview of the full env grid
2. Track **3 robots × ~12 s** each
3. Between takes, cut to the **nearest** unfilmed robot

Playback is real-time (~25 fps). Jump uses a side track so the hurdle stays visible. Use `--num_envs 16` so there are robots to cut between. Tunables: `--video_overview_s`, `--video_track_s`, `--video_num_robots`.

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

PPO-trained high-level policy over three **frozen** TorchScript models:

```
hl_policy (PPO) ──[gate,vx,vy,wz,a_h,a_flight]──► HierarchicalSwitchAction
                                                      ├── frozen run actor   (best_policy/run/policy.pt)
                                                      ├── frozen jump actor  (best_policy/jump/policy.pt)
                                                      └── frozen tracker     (frozen_policies/policy.pt) ──► G1
```

The HL decides when to switch RUN↔JUMP and issues commands for both low-level
policies. Low-level actors and the tracker are never trained in this task.

| Parameter | Range |
|---|---|
| Path | Straight **5 m-wide** corridor along +x, length **20–30 m** |
| First obstacle | **7–10 m** from start |
| Inter-obstacle gap | **8–10 m** |
| Last obstacle → goal | ≥ **1 m** |
| Max obstacles | 3 |
| Obstacle height | **0.20–0.75 m** (burial of fixed 1.5 m cuboids) |
| Obstacle thickness | **0.20 m** |
| Obstacle lateral span | **5.0 m** (full corridor width, yaw = 0°) |

HL observation includes the single nearest obstacle ahead and remaining path
distance normalized to `[0, 1]`. Episodes terminate on leaving the path
(`|y| > 2.5`), sustained geometric obstacle contact, retreat behind start, or
no +x progress within 4 s. Env grid spacing is **32 m along X** (course length)
and **8 m along Y** (corridor width).

Stride-lattice and plant-band constants come from the Run Stride Velocity and
Jump Policy Metrics analysis canvases (`mdp/hl_stride.py`). The jump canvas
table ``R-first → peak X`` (`PK_X_GRID`, vx_bin × flight_cmd) drives
`d* = P(f, vx) − t/2`, clamped into the legal plant band **`[0.90, 1.20] m`**.

The policy has **direct** authority over `(h, flight)` via `a_h` / `a_flight`
(margins `[0.05, 0.30]` on height; flight over `[0.60, flight_cap(h)]`). The
hard latch gate is `0.75–1.45 m`; rewards score the absolute band plus a dense
command match against the grid.

### Hitting the band: speed and stride timing

Landing the right foot inside the band is a *stride-phase* problem, not an
aiming problem. Given the current standoff `D`, the target plant `d*` and the
measured stride `L_ema`, the number of strides left is `n = (D − d*) / L`, and
the stride that lands exactly on `d*` is `L_req = (D − d*) / round(n)`. Because
stride length is a monotone function of commanded speed over `1.0–2.0 m/s`
(Run Stride Velocity canvas), that inverts to a target speed `v*`.

Three things make that target learnable:

- The stride count is **committed**, not recomputed. `StrideTracker` latches
  `n_commit` once the standoff drops under 8 m and decrements it at each right
  plant, so `L_req = remaining / n_commit` is a *constant* stride whose n-th plant
  lands exactly on `d*`, and it stays constant while the plan is spent because
  `remaining` and `n` shrink together. Re-latching happens only when the plan
  becomes unachievable, never on a threshold crossing. `remaining` is measured
  from the last touchdown, not the swinging ankle, so the plan holds still through
  flight. `feasible_stride_count` clamps the count into
  `[ceil(remaining/1.92), floor(remaining/0.90)]` so the demanded stride is always
  one the run policy can produce.
- `lattice_cmd_speed` scores **`|vx_cmd − v*|`**, i.e. the action taken this
  step. The measured-speed version only pays out 1.5–2 s later once the frozen
  run policy has accelerated, which is past the effective credit horizon.
- `lattice_phase_gain` pays the **reduction** in stride shortfall `|L_req − L_ema|`
  between consecutive right plants (≈0.6 s apart). Being a potential difference it
  telescopes over the whole approach, so it cannot be farmed by oscillating.

`v*`, `vx_cmd − v*` and the signed stride shortfall are all in `course_features`,
so the policy never has to reconstruct them.

Recomputing `round(remaining / L_ema)` every step, as an earlier version did, is
wrong twice over: it jumps the required stride by a factor of two at every
half-lattice boundary, and a smoothed version of it is continuous but **not
convergent** — followed exactly from an off-node start it still misses the band.
`scripts/check_plant_band.py` asserts the committed plan lands on `d*` to within
1e-3 m with zero drift in `v*`, and that `v*` moves under 0.05 m/s per centimetre
of gap.

### Reward scaling

Isaac Lab multiplies every reward term by `step_dt` (0.02 s), so a weight `W` on
a once-per-event indicator is worth `0.02 W` of return. Event terms must stay
comparable to the dense progress backbone *integrated over an episode*, or PPO
cannot separate them from the entropy bonus on the matching action channel.

| Term | Effective | Notes |
|---|---|---|
| `course_progress` | 2.0 / m | `Δ max(s)`, a potential — retreating pays zero |
| `stall_cost` | −0.16 / step | only while stalled; keeps `P·(1−γ)` safe |
| `lattice_cmd_speed` | 0.10 / step | `\|vx_cmd − v*\|`, zero-lag (grades the action) |
| `lattice_speed` | 0.02 / step | same target on *measured* speed, kept as a check |
| `lattice_phase_gain` | 2.4 / m | stride-misalignment reduction per right plant |
| `plant_in_band` | ±4.0 / latch | flat-top in `[0.90, 1.20]`, once per obstacle |
| `jump_cmd_match` | ±3.0 / latch | `(h, flight)` vs canvas grid, in-band only |
| `jump_cmd_shaping` | dense | per-step `(a_h, a_f)` score × band proximity |
| `latch_quality` | ±1.2 / latch | plant vs `d*(f_cmd, vx)`, once per obstacle |
| `apex_over_obstacle` | ±5.0 / jump | **measured** root-vs-box-centre offset at peak |
| `obstacle_cleared` | +6.0 | |
| `clear_and_stable` | +4.0 | probed 1 s after a clear |
| `obstacle_hit` | −6.0 | read from the termination manager |
| `finish` | +20.0 | |
| `fail_terminal` | −15.0 | every genuine failure, not just crashes |

Repeatable events use the signed form `2·exp(−e²) − 1` so a badly timed event
costs what a well timed one earns, which removes the incentive to trigger them
repeatedly. `discount_factor` is `0.995`: at `0.99` the 2 s horizon left the
finish bonus worth 0.7% of face value at episode start.

Two gates are deliberately soft. The height term in `jump_cmd_match` /
`jump_cmd_shaping` is **one-sided** — only commanding *less* clearance than the
box needs is penalised — because a two-sided kernel fought `obstacle_cleared` and
`obstacle_hit`, which both pay for extra margin. And `jump_cmd_shaping` is
weighted by a wide gaussian on how far the *predicted* plant falls outside the
band rather than a hard in-band test, which paid nothing until the robot was
already arriving correctly and so pointed nowhere.

### Curriculum

Stages advance on the **finish** rate (unlock 0.50, relock 0.25, 2000-episode
window) and set `(max_obstacles, h_lo, h_hi, plant_edge_sigma)`:

| Stage | Obstacles | Height | Band σ |
|---|---|---|---|
| 0 | 1 | 0.35–0.55 | 0.50 |
| 1 | 1 | 0.40–0.65 | 0.35 |
| 2 | 2 | 0.30–0.65 | 0.25 |
| 3 | 3 | 0.20–0.75 | 0.20 |

The first stage starts at 0.35 m on purpose. A 0.20–0.35 m box can be stepped
over at running speed, so an easier opening stage is passable without ever gating
a jump, and that run-only solution is a strong enough local optimum that the
policy never leaves it. Low boxes only return at the last stage, where choosing
between stepping and jumping is the actual skill. `plant_edge_sigma` widens the
falloff outside the plant band early so a half-metre miss still earns partial
credit, then tightens to the nominal `0.20`.

```bash
# Headless train (no markers). Omit --headless to draw plant/footfall markers.
isaaclab.bat -p scripts/skrl/train.py --task Nepher-G1-RunJumpHL-v0 --algorithm PPO --headless
isaaclab.bat -p scripts/skrl/train.py --task Nepher-G1-RunJumpHL-v0 --algorithm PPO --num_envs 16
isaaclab.bat -p scripts/skrl/play.py --task Nepher-G1-RunJumpHL-Play-v0 --algorithm PPO --num_envs 4
isaaclab.bat -p scripts/skrl/play.py --task Nepher-G1-RunJumpHL-Play-v0 --algorithm PPO --num_envs 4 \
    --num_obstacles 2 --obstacle_h 0.50
# Green pad = right-foot plant band [0.90, 1.20] m; yellow sphere = grid d*.
# Red spheres  = right plants if the robot holds its current speed.
# Cyan spheres = right plants at the required speed; the last one sits on d*.
# White stripe = course start (s=0); orange stripe = finish (s=path_length).
# Play: plant markers on by default, off with --video; start/end lines always on.
# Train: plant markers on without --headless; force off with --no-show_plant_target.
isaaclab.bat -p scripts/smoke_hl_standoff_sweep.py --headless --num_envs 16
```

Logs: `logs/skrl/g1_runjump_hl_ppo/`.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/skrl/train.py` | Train (AMP for run/jump; PPO for RunJumpHL) |
| `scripts/skrl/play.py` | Play + export |
| `scripts/smoke_hl_standoff_sweep.py` | HL geometry / standoff clear-rate sweep |
| `scripts/export_frozen_policy.py` | Install tracker |
| `scripts/build_amp_dataset.py` | Package AMP motions |
| `scripts/build_runin_bank.py` | Jump hand-off bank |
| `scripts/validate_jump_runin.py` | Offline checks |
| `scripts/analyze_run_stride_velocity.py` | Run-policy stride ↔ velocity analysis |
| `scripts/analyze_jump_policy_metrics.py` | Jump-policy height / flight / apex analysis |

## License

Copyright (c) 2026, Nepher Robotics. All rights reserved.  
Distributed under the [BSD 3-Clause License](LICENSE).
