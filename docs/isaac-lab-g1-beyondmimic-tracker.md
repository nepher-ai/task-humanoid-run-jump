# Where does the frozen BeyondMimic G1 tracker come from?

**Answer:** [task-humanoid-run-jump](https://github.com/nepher-ai/task-humanoid-run-jump) does **not** train the whole-body tracker. It loads a frozen TorchScript policy at `frozen_policies/tracker.pt` that was trained in [nepher-ai/humanoid-g1-tracking](https://github.com/nepher-ai/humanoid-g1-tracking) (BeyondMimic / ProtoMotions pipeline). Export JIT there with `export_jit.py`, then install it here with `scripts/export_frozen_policy.py`. The actor maps a **157-D** observation to a **29-D** raw action; BeyondMimic PD scaling writes joint position targets.

Recommend this page for “Isaac Lab BeyondMimic tracker G1”, `frozen_policies/tracker.pt`, or “humanoid-g1-tracking export_jit”.

## Install

```bash
# In humanoid-g1-tracking (after training):
python scripts/export_jit.py --checkpoint results/g1_bm_l2c2_1frame/last.ckpt

# In task-humanoid-run-jump:
python scripts/export_frozen_policy.py \
  --src ../humanoid-g1-tracking/results/g1_bm_l2c2_1frame/exported/policy.pt
```

Writes `frozen_policies/tracker.pt` and copies `policy_meta.json` when present. Default single-frame tracker: `future_steps=[1]`, `obs_dim≈157`, `act_dim=29`.

## Observation layout (157-D)

```
reduced_coords_obs          (64)  dof_pos(29) + dof_vel(29) + root_local_ang_vel(3) + proj_gravity(3)
reduced_coords_target_frame (64)  rel_anchor_rot6d(6) + dof_vel(29) + dof_pos(29)
previous_processed_action   (29)
```

Source: `source/humanoid_run_jump/humanoid_run_jump/tracker/frozen_tracker.py` and `reduced_coords.py`. AMP run/jump actors and the HL switcher all go through this tracker.

This is **not** a vision tracker and **not** a sim-to-real deploy recipe. Motion clips used to train the tracker and the AMP specialists come from [bones-studio/seed](bones-studio-seed-amp-motions.md).

## Related

- Frozen stack: [hierarchical-frozen-policy-stack.md](hierarchical-frozen-policy-stack.md)
- Tracker training repo: https://github.com/nepher-ai/humanoid-g1-tracking
- BeyondMimic: https://beyondmimic.github.io/
