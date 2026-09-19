# Residual SAC V2 — SAC learns moving-human avoidance

## What changed

V1 trained the residual policy **under Predictive DWA**, so DWA could rescue bad actions and the learner did not have to become the primary human-avoidance controller.

V2 changes the responsibility split:

- **A\*** + deterministic baseline: route to pickup/table and static furniture braking.
- **Residual SAC V2**: normal dynamic-human avoidance (slow/turn/pass decisions).
- **EmergencyHumanShield**: only a last-resort imminent-collision STOP.
- **Predictive DWA**: not used in the V2 live runtime or V2 training environment.

The human features are also now robot-relative: forward/left position and relative forward/left velocity. This makes crossing patterns easier to learn independent of world orientation.

## Important

Do **not** use `dynamic_sac_residual_v1.zip` with the V2 runtime. Although the vector length is still 59, the feature semantics changed and V2 should be trained fresh.

## Files added

- `baseline_controller_v2.py`
- `residual_policy_v2.py`
- `residual_env_v2.py`
- `train_residual_sac_v2.py`
- `evaluate_residual_sac_v2.py`
- `run_residual_sac_v2.py`

No old runtime file needs to be deleted.

## 1. Train

From the same Python environment where Genesis and Stable-Baselines3 already work:

```bash
python train_residual_sac_v2.py --total-timesteps 500000
```

For an initial functionality check you can use:

```bash
python train_residual_sac_v2.py --total-timesteps 100000
```

but do not treat a short checkpoint as the final model.

The curriculum is:

1. 1 slow moving, non-cooperative person
2. 2 people
3. 3 people
4. 4 full-speed non-cooperative people

Checkpoints are written to `checkpoints_v2/` every 25,000 steps.

## 2. Evaluate before using the GUI

```bash
python evaluate_residual_sac_v2.py --model dynamic_sac_residual_v2.zip --episodes 200
```

Primary metrics:

- dynamic collision rate
- emergency shield intervention rate
- overall success rate
- static collision rate
- human clearance
- mean speed

Suggested graduation targets before calling the model final:

- overall success >= 80%
- dynamic collision <= 5%
- emergency shield rate <= 5%

These are project targets, not guaranteed outcomes.

## 3. Run live GUI

```bash
python run_residual_sac_v2.py --model dynamic_sac_residual_v2.zip
```

Default live human cooperation is `0.0`, so moving people do not artificially slow/yield for the robot. The SAC policy must handle them.

## Why V2 can stop better than V1

V1 residual linear scale was only 0.35. With a baseline cruise action around 0.78, even a maximum negative residual could leave the robot moving forward around 0.43 action units.

V2 uses a larger bounded residual scale and clips the final action safely, allowing the learned policy to genuinely slow or stop while still retaining the A* baseline structure.
