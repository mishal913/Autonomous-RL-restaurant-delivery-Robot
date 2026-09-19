# Residual SAC for human avoidance — retrain attempt

## Why the previous attempt was dropped

You noted it was pulled because dynamic (human) collisions were too high in
eval. Looking at the runtime code, there are three concrete, fixable reasons
that's a plausible outcome for a *full-action* SAC policy specifically:

1. **The policy's observation can't tell furniture from people.**
   `RestaurantDeliveryEnv._get_lidar()` rays are cast against
   `obstacle_data`, which is static clutter + tables + decor + **dynamic
   people all merged into one list** (`_refresh_obstacle_data`). The policy
   sees 16 radial distances and their 4x-scaled frame-to-frame deltas — no
   identity, no relative velocity, no heading. The DWA planner and
   `predict_dynamic_risk()` both use much richer signals internally
   (`person_states[i]["current_xy"/"velocity_xy"]`) that the learning
   policy simply never received.

2. **The reward doesn't disproportionately protect humans.**
   `_calculate_reward()` applies the same `-100` for `self.collision`
   whether it's a table or a person, and dynamic-risk shaping (`-1.8` per
   unit lost clearance, `-2.5` per unit lost TTC) is comparable in scale to
   the static-clearance term. Nothing in the objective says "a human near-miss
   is categorically worse than a chair near-miss."

3. **A full-action policy has to relearn navigation *and* avoidance
   jointly**, which means early-training exploration constantly produces
   real collisions (with nothing to stop it) — a noisy, unstable regime for
   off-policy learning, especially against 4 people at 35% cooperation.

## What's different this time: residual + shield

- **`baseline_controller.py`** — the existing A*-heading-pursuit +
  furniture-braking logic (same as `run_final_restaurant.py`), with the SAC
  turn-blend term removed. Deterministic, no learning, one source of truth
  now shared by runtime and training instead of two copies drifting apart.

- **`residual_env.py`** — the new policy only outputs a *bounded correction*
  on top of that baseline (`residual_linear_scale=0.35`,
  `residual_turn_scale=0.55`), and every action — including fully random
  actions during early exploration — is passed through the exact same
  `PredictiveDWAPlanner.filter_action()` used in production before touching
  the physics sim. Consequences:
  - genuine human collisions become very rare *during training itself*,
    so the replay buffer isn't dominated by "gave up and crashed" transitions
  - the reward can target something more informative than collision
    frequency alone: **how often the shield had to override the policy**
    (`intervention_penalty`). Driving that rate toward zero is a dense,
    well-shaped signal for "predict and avoid early" instead of "get rescued
    late."
  - dynamic collisions still get an extra `-150` on top of the base env's
    `-100` (`dynamic_collision_extra_penalty`), so if the shield's last-resort
    "no safe velocity" path is ever hit, the policy is pushed hard away from
    situations that create it.
  - observations now include, per person (zero-padded to 4): relative x/y,
    relative vx/vy, and a cooperative flag — the same state the hand-tuned
    DWA planner uses. This directly fixes root cause #1 above.

- **Curriculum** (`train_residual_sac.py`, `CURRICULUM` tuple): 1 fully
  cooperative person → 2 at 75% → 3 at 55% → 4 at 35% (production-matching),
  ramped over the first ~1.2M timesteps. This lets the policy master clean
  single-person passing before facing the adversarial mix it's ultimately
  judged on — the same reasoning behind `RestaurantDeliveryEnv.set_curriculum`,
  extended with a cooperation-probability axis it doesn't otherwise expose.

- **`evaluate_residual_sac.py`** — the thing that actually resolves your
  "not yet proven" status: N randomized-seed episodes at the production
  setting (4 people, 0.35 cooperation), reporting dynamic-collision rate,
  5th-percentile human clearance, and mean shield-intervention rate,
  plus a per-episode CSV in the same spirit as the runtime's trace files.

## What I could and couldn't do in this sandbox

I wrote and syntax-checked all four files, but I could **not** actually run
training here: this container has no `genesis-world` install, no GPU, and
no display — and real SAC training on a physics sim is a many-hour-to-days
job regardless. You'll need to run this on your own machine (or wherever
`dynamic_sac_v3.zip` was originally trained).

## How to run it

```bash
cd residual_human_sac
pip install "genesis-world[render]" stable-baselines3 gymnasium

# Train (expect ~1.5-2M timesteps to get through the full curriculum;
# start smaller e.g. --total-timesteps 200000 as a smoke test first)
python train_residual_sac.py --total-timesteps 2000000 \
    --out dynamic_sac_residual_v1.zip

# Evaluate against the production scenario (4 people, 0.35 cooperation)
python evaluate_residual_sac.py --model dynamic_sac_residual_v1.zip \
    --episodes 200 --people 4 --human-cooperation 0.35 \
    --out eval_residual_v1.csv
```

Watch `[intervention] ... shield_rate=...` in the training logs — that's the
number that should trend toward 0 over the curriculum, more so than raw
episode reward (which the shield partially decouples from policy quality).

## Wiring it back into `run_final_restaurant.py`

The runtime's `hybrid_action()`/`docking_action()` + SAC blend gets replaced
by: `baseline_controller.compute_baseline_action(env, path)` for the
baseline, plus `residual_model.predict(residual_obs)` for the correction,
combined the same way `residual_env.py` does it — then handed to the
*existing* `PredictiveDWAPlanner.filter_action()` unchanged. The DWA safety
layer, docking controller, A* planner, order system, and overlay all stay
exactly as they are; only the "what does the learned component contribute"
piece changes. I can wire this into a drop-in replacement for
`run_final_restaurant.py` next if you want it, once you've got a trained
checkpoint to point it at.
