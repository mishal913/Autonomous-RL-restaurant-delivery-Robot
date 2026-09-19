"""Residual-RL wrapper for human avoidance, trained under a DWA safety shield.

Design, and why it differs from the dropped "residual SAC" attempt:

1. RESIDUAL, not full control. The policy only outputs a bounded correction
   on top of a fixed, learning-free baseline (A* waypoint pursuit + furniture
   braking, see baseline_controller.py). It never has to relearn "drive
   toward the goal" or "don't hit tables" from scratch -- it only has to
   learn "how should I adjust course/speed to keep humans clear."

2. SHIELDED training. Every action, even during early random exploration,
   is passed through the exact same PredictiveDWAPlanner used in production
   before being applied to the physics sim. This means:
     - the robot essentially cannot rack up dynamic (human) collisions
       during training, so the replay buffer never fills up with the "give
       up and crash" transitions that make Q-learning unstable near people
     - the reward can instead measure something more informative than
       sparse collision events: *how often the shield had to override the
       policy*. Driving that intervention rate toward zero is the actual
       training signal for "learn to avoid needing a rescue."

3. EXPLICIT dynamic-agent observations. The deployed policy's observation
   (16 LiDAR rays + 16 deltas + 5 scalars) never separates people from
   furniture and carries no relative-velocity information -- a likely reason
   the earlier attempt could not reliably learn predictive avoidance. This
   env appends, per active person (up to 4, zero-padded): relative x, y,
   relative vx, vy, and a cooperative flag, all directly from
   env.person_states / env.predict_dynamic_risk(), which the DWA planner
   already relies on internally. Giving the learner the same signal the
   hand-tuned planner uses is a much fairer comparison than radial LiDAR.

4. REWARD reweighting. The base env's reward penalizes static and dynamic
   collisions identically (-100 each). For a policy whose entire job is
   human safety, dynamic collisions get an additional, much larger penalty
   here, and shield interventions and residual magnitude are penalized so
   the policy is pushed toward small, early, shield-approved corrections
   rather than sharp last-moment ones.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from baseline_controller import REPLAN_EVERY, compute_baseline_action
from predictive_dwa import PredictiveDWAPlanner

MAX_PEOPLE = 4
PERSON_FEATURES = 5  # rel_x, rel_y, rel_vx, rel_vy, cooperative_flag
REL_POS_NORM = 12.0   # metres, world is roughly 23m x 17m
REL_VEL_NORM = 2.5    # m/s, generous over MAX_LINEAR_SPEED=1.5 + person speed


@dataclass
class ResidualEnvConfig:
    residual_linear_scale: float = 0.35
    residual_turn_scale: float = 0.55
    residual_l2_penalty: float = 0.015
    intervention_penalty: float = 0.55
    dynamic_collision_extra_penalty: float = 150.0
    replan_every: int = REPLAN_EVERY


class ResidualHumanAvoidanceEnv(gym.Env):
    """Wraps RestaurantDeliveryEnv. Action = bounded residual correction."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, base_env, config: ResidualEnvConfig | None = None):
        super().__init__()
        self.env = base_env
        self.cfg = config or ResidualEnvConfig()
        self.dwa = PredictiveDWAPlanner()

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        base_low = self.env.observation_space.low
        base_high = self.env.observation_space.high
        extra_low = np.concatenate(
            [
                np.array([-1.0, -1.0], dtype=np.float32),  # baseline action
                np.tile(
                    np.array([-1.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32),
                    MAX_PEOPLE,
                ),
            ]
        )
        extra_high = np.concatenate(
            [
                np.array([1.0, 1.0], dtype=np.float32),
                np.tile(
                    np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                    MAX_PEOPLE,
                ),
            ]
        )
        self.observation_space = spaces.Box(
            low=np.concatenate([base_low, extra_low]).astype(np.float32),
            high=np.concatenate([base_high, extra_high]).astype(np.float32),
            dtype=np.float32,
        )

        self._path = None
        self._steps_since_replan = 0
        self._last_baseline_action = np.zeros(2, dtype=np.float32)
        self._episode_intervention_steps = 0
        self._episode_steps = 0
        self._previous_phase = None

    # ------------------------------------------------------------------
    def _person_features(self) -> np.ndarray:
        robot_xy = np.asarray(self.env._get_robot_position(), dtype=np.float64)
        feats = np.zeros(MAX_PEOPLE * PERSON_FEATURES, dtype=np.float32)
        states = self.env.person_states[: self.env.active_people_count]
        for index, state in enumerate(states[:MAX_PEOPLE]):
            if "current_xy" not in state:
                continue
            rel = np.asarray(state["current_xy"], dtype=np.float64) - robot_xy
            vel = np.asarray(state.get("velocity_xy", [0.0, 0.0]), dtype=np.float64)
            base = index * PERSON_FEATURES
            feats[base + 0] = float(np.clip(rel[0] / REL_POS_NORM, -1.0, 1.0))
            feats[base + 1] = float(np.clip(rel[1] / REL_POS_NORM, -1.0, 1.0))
            feats[base + 2] = float(np.clip(vel[0] / REL_VEL_NORM, -1.0, 1.0))
            feats[base + 3] = float(np.clip(vel[1] / REL_VEL_NORM, -1.0, 1.0))
            feats[base + 4] = 1.0 if bool(state.get("cooperative", True)) else 0.0
        return feats

    def _build_observation(self, base_obs: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                base_obs.astype(np.float32),
                self._last_baseline_action.astype(np.float32),
                self._person_features(),
            ]
        ).astype(np.float32)

    def _refresh_path_if_needed(self, force: bool = False) -> None:
        if force or self._path is None or self._steps_since_replan >= self.cfg.replan_every:
            candidate = self.env.plan_path()
            if candidate:
                self._path = candidate
            self._steps_since_replan = 0

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        base_obs, info = self.env.reset(seed=seed, options=options)
        self.dwa.reset()
        self._path = None
        self._steps_since_replan = self.cfg.replan_every  # force replan
        self._refresh_path_if_needed(force=True)
        self._last_baseline_action = np.zeros(2, dtype=np.float32)
        self._episode_intervention_steps = 0
        self._episode_steps = 0
        self._previous_phase = str(info.get("phase", ""))
        return self._build_observation(base_obs), info

    def step(self, residual_action: np.ndarray):
        residual_action = np.clip(np.asarray(residual_action, dtype=np.float32), -1.0, 1.0)
        current_phase = "to_table" if self.env.package_picked else "to_pickup"
        if current_phase != self._previous_phase:
            self._path = None  # target changed; stale path would misdirect
            self._previous_phase = current_phase
        self._refresh_path_if_needed()

        baseline_action, waypoint = compute_baseline_action(self.env, self._path)
        self._last_baseline_action = baseline_action

        scale = np.array(
            [self.cfg.residual_linear_scale, self.cfg.residual_turn_scale],
            dtype=np.float32,
        )
        proposed_action = np.clip(baseline_action + scale * residual_action, -1.0, 1.0)

        decision = self.dwa.filter_action(self.env, proposed_action, waypoint)
        final_action = np.asarray(decision.action, dtype=np.float32)

        base_obs, base_reward, terminated, truncated, info = self.env.step(final_action)
        self._steps_since_replan += 1
        self._episode_steps += 1

        reward = float(base_reward)
        reward -= self.cfg.residual_l2_penalty * float(np.dot(residual_action, residual_action))
        if decision.intervened:
            reward -= self.cfg.intervention_penalty
            self._episode_intervention_steps += 1
        if info.get("collision_type") == "dynamic":
            reward -= self.cfg.dynamic_collision_extra_penalty

        info = dict(info)
        info["dwa_intervened"] = bool(decision.intervened)
        info["dwa_reason"] = str(decision.reason)
        info["residual_action"] = residual_action.tolist()
        info["baseline_action"] = baseline_action.tolist()
        if terminated or truncated:
            info["episode_intervention_rate"] = (
                self._episode_intervention_steps / max(1, self._episode_steps)
            )

        return self._build_observation(base_obs), reward, terminated, truncated, info

    def close(self):
        self.env.close()
