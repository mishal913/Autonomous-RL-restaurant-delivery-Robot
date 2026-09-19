"""Residual SAC V2 training environment: SAC learns moving-human avoidance.

Unlike V1, normal actions are NOT sent through Predictive DWA. The residual
policy sees robot-frame human position/relative velocity and is trained to
improve predicted clearance early. Only an imminent-collision emergency stop
remains as a last-resort safety guard.
"""
from __future__ import annotations

from dataclasses import dataclass
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from baseline_controller_v2 import REPLAN_EVERY, compute_baseline_action
from residual_policy_v2 import (
    MAX_PEOPLE,
    PERSON_FEATURES,
    EmergencyHumanShield,
    build_residual_observation,
    compose_residual_action,
)


@dataclass
class ResidualEnvV2Config:
    residual_l2_penalty: float = 0.020
    shield_intervention_penalty: float = 4.0
    dynamic_collision_extra_penalty: float = 300.0
    unsafe_clearance_m: float = 0.85
    near_human_penalty: float = 1.20
    ttc_penalty: float = 1.50
    early_avoidance_bonus: float = 0.50
    worsened_clearance_penalty: float = 0.80
    unnecessary_stop_penalty: float = 0.10
    replan_every: int = REPLAN_EVERY


class ResidualHumanAvoidanceEnvV2(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, base_env, config: ResidualEnvV2Config | None = None):
        super().__init__()
        self.env = base_env
        self.cfg = config or ResidualEnvV2Config()
        self.shield = EmergencyHumanShield()

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        base_low = np.asarray(self.env.observation_space.low, dtype=np.float32)
        base_high = np.asarray(self.env.observation_space.high, dtype=np.float32)
        extra_low = np.concatenate([
            np.array([-1.0, -1.0], dtype=np.float32),
            np.tile(np.array([-1.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32), MAX_PEOPLE),
        ])
        extra_high = np.concatenate([
            np.array([1.0, 1.0], dtype=np.float32),
            np.tile(np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32), MAX_PEOPLE),
        ])
        self.observation_space = spaces.Box(
            low=np.concatenate([base_low, extra_low]).astype(np.float32),
            high=np.concatenate([base_high, extra_high]).astype(np.float32),
            dtype=np.float32,
        )

        self._path = None
        self._steps_since_replan = 0
        self._baseline_action = np.zeros(2, dtype=np.float32)
        self._previous_phase = None
        self._episode_steps = 0
        self._shield_steps = 0

    def _phase(self) -> str:
        return "to_table" if self.env.package_picked else "to_pickup"

    def _refresh_path_if_needed(self, force: bool = False) -> None:
        if force or self._path is None or self._steps_since_replan >= self.cfg.replan_every:
            candidate = self.env.plan_path()
            if candidate:
                self._path = candidate
            self._steps_since_replan = 0

    def _refresh_baseline(self) -> tuple[np.ndarray, np.ndarray]:
        self._refresh_path_if_needed()
        action, waypoint = compute_baseline_action(self.env, self._path)
        self._baseline_action = np.asarray(action, dtype=np.float32)
        return self._baseline_action, waypoint

    def reset(self, *, seed=None, options=None):
        base_obs, info = self.env.reset(seed=seed, options=options)
        self.shield.reset()
        self._path = None
        self._steps_since_replan = self.cfg.replan_every
        self._previous_phase = self._phase()
        self._episode_steps = 0
        self._shield_steps = 0
        baseline, _ = self._refresh_baseline()
        return build_residual_observation(self.env, base_obs, baseline), info

    def step(self, residual_action: np.ndarray):
        phase = self._phase()
        if phase != self._previous_phase:
            self._path = None
            self._previous_phase = phase

        baseline_action, _ = self._refresh_baseline()
        residual = np.clip(np.asarray(residual_action, dtype=np.float32), -1.0, 1.0)
        proposed_action = compose_residual_action(baseline_action, residual)

        # Compare what the fixed baseline would do with what SAC proposes.
        baseline_risk = self.env.predict_dynamic_risk(action=baseline_action, horizon=1.6)
        proposed_risk = self.env.predict_dynamic_risk(action=proposed_action, horizon=1.6)

        shield_decision = self.shield.filter_action(self.env, proposed_action)
        final_action = np.asarray(shield_decision.action, dtype=np.float32)
        base_obs, base_reward, terminated, truncated, info = self.env.step(final_action)
        self._steps_since_replan += 1
        self._episode_steps += 1

        reward = float(base_reward)
        reward -= self.cfg.residual_l2_penalty * float(np.dot(residual, residual))

        proposed_clearance = float(proposed_risk.get("clearance", 99.0))
        baseline_clearance = float(baseline_risk.get("clearance", 99.0))
        proposed_ttc = float(proposed_risk.get("ttc", 99.0))

        if proposed_clearance < self.cfg.unsafe_clearance_m:
            gap = (self.cfg.unsafe_clearance_m - proposed_clearance) / self.cfg.unsafe_clearance_m
            reward -= self.cfg.near_human_penalty * max(0.0, gap)

        if bool(proposed_risk.get("danger", False)):
            urgency = max(0.0, (1.6 - min(proposed_ttc, 1.6)) / 1.6)
            reward -= self.cfg.ttc_penalty * urgency

        # Directly reward SAC for improving a dangerous baseline trajectory.
        improvement = proposed_clearance - baseline_clearance
        baseline_needs_help = bool(baseline_risk.get("danger", False)) or baseline_clearance < 0.35
        if baseline_needs_help and improvement > 0.10:
            reward += self.cfg.early_avoidance_bonus * min(improvement / 0.60, 1.0)
        elif baseline_clearance < 1.2 and improvement < -0.10:
            reward -= self.cfg.worsened_clearance_penalty * min(abs(improvement) / 0.60, 1.0)

        # Prevent the trivial "always stop" solution when no human threat exists.
        if baseline_action[0] > 0.25 and proposed_action[0] < 0.08 and proposed_clearance > 1.20:
            reward -= self.cfg.unnecessary_stop_penalty

        if shield_decision.intervened:
            reward -= self.cfg.shield_intervention_penalty
            self._shield_steps += 1
        if info.get("collision_type") == "dynamic":
            reward -= self.cfg.dynamic_collision_extra_penalty

        info = dict(info)
        info.update({
            "emergency_shield_intervened": bool(shield_decision.intervened),
            "emergency_shield_reason": str(shield_decision.reason),
            "residual_action": residual.tolist(),
            "baseline_action": baseline_action.tolist(),
            "proposed_action": proposed_action.tolist(),
            "baseline_predicted_clearance": baseline_clearance,
            "sac_predicted_clearance": proposed_clearance,
            "sac_predicted_ttc": proposed_ttc,
        })

        if terminated or truncated:
            info["episode_emergency_shield_rate"] = self._shield_steps / max(1, self._episode_steps)

        # Build NEXT observation with the next state's current baseline action.
        # On terminal transitions we deliberately avoid replanning a scene that
        # has already collided/delivered; Gym only needs a shape-correct final obs.
        if terminated or truncated:
            obs = build_residual_observation(self.env, base_obs, baseline_action)
        else:
            next_phase = self._phase()
            if next_phase != self._previous_phase:
                self._path = None
                self._previous_phase = next_phase
            next_baseline, _ = self._refresh_baseline()
            obs = build_residual_observation(self.env, base_obs, next_baseline)
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()
