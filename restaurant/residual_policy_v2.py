"""Shared Residual SAC V2 observation/action logic and minimal safety shield.

Key design point: SAC, not DWA, performs normal moving-human avoidance.
The shield only acts when a collision is already imminent.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

MAX_PEOPLE = 4
PERSON_FEATURES = 5
REL_POS_NORM = 8.0
REL_VEL_NORM = 2.5

# V1 used 0.35 linear correction, so a 0.78 cruise could not be slowed below
# ~0.43 by SAC. V2 must be able to genuinely slow/stop for crossing humans.
RESIDUAL_LINEAR_SCALE = 0.90
RESIDUAL_TURN_SCALE = 0.85
MIN_FINAL_LINEAR = -0.20
MAX_FINAL_LINEAR = 0.90


def person_features_robot_frame(env) -> np.ndarray:
    """Per person: forward, left, relative-forward-velocity,
    relative-left-velocity, active flag. All in the robot's local frame.

    Robot-frame features make the same crossing geometry look similar no
    matter which world direction the robot is facing.
    """
    robot_xy = np.asarray(env._get_robot_position(), dtype=np.float64)
    yaw = float(env._get_robot_yaw())
    forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    left = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    robot_velocity = float(env.linear_velocity) * forward

    feats = np.zeros(MAX_PEOPLE * PERSON_FEATURES, dtype=np.float32)
    states = env.person_states[: env.active_people_count]
    for index, state in enumerate(states[:MAX_PEOPLE]):
        if "current_xy" not in state:
            continue
        rel_world = np.asarray(state["current_xy"], dtype=np.float64) - robot_xy
        human_velocity = np.asarray(state.get("velocity_xy", [0.0, 0.0]), dtype=np.float64)
        rel_velocity = human_velocity - robot_velocity

        base = index * PERSON_FEATURES
        feats[base + 0] = float(np.clip(np.dot(rel_world, forward) / REL_POS_NORM, -1.0, 1.0))
        feats[base + 1] = float(np.clip(np.dot(rel_world, left) / REL_POS_NORM, -1.0, 1.0))
        feats[base + 2] = float(np.clip(np.dot(rel_velocity, forward) / REL_VEL_NORM, -1.0, 1.0))
        feats[base + 3] = float(np.clip(np.dot(rel_velocity, left) / REL_VEL_NORM, -1.0, 1.0))
        feats[base + 4] = 1.0
    return feats


def build_residual_observation(env, base_obs: np.ndarray, baseline_action: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(base_obs, dtype=np.float32),
            np.asarray(baseline_action, dtype=np.float32),
            person_features_robot_frame(env),
        ]
    ).astype(np.float32)


def compose_residual_action(baseline_action: np.ndarray, residual_action: np.ndarray) -> np.ndarray:
    residual = np.clip(np.asarray(residual_action, dtype=np.float32), -1.0, 1.0)
    baseline = np.asarray(baseline_action, dtype=np.float32)
    linear = float(np.clip(
        baseline[0] + RESIDUAL_LINEAR_SCALE * residual[0],
        MIN_FINAL_LINEAR,
        MAX_FINAL_LINEAR,
    ))
    turn = float(np.clip(
        baseline[1] + RESIDUAL_TURN_SCALE * residual[1],
        -1.0,
        1.0,
    ))
    return np.array([linear, turn], dtype=np.float32)


@dataclass
class EmergencyShieldConfig:
    horizon_s: float = 0.75
    stop_ttc_s: float = 0.35
    critical_clearance_m: float = -0.05


@dataclass
class EmergencyShieldDecision:
    action: np.ndarray
    intervened: bool
    reason: str
    predicted_clearance: float
    predicted_ttc: float
    person: str | None


class EmergencyHumanShield:
    """Last-resort stop only. It does not plan arcs, passing sides, or waits."""

    def __init__(self, config: EmergencyShieldConfig | None = None):
        self.cfg = config or EmergencyShieldConfig()

    def reset(self) -> None:
        return None

    def filter_action(self, env, proposed_action: np.ndarray) -> EmergencyShieldDecision:
        proposed = np.asarray(proposed_action, dtype=np.float32)
        risk = env.predict_dynamic_risk(action=proposed, horizon=self.cfg.horizon_s)
        clearance = float(risk.get("clearance", 99.0))
        ttc = float(risk.get("ttc", 99.0))
        danger = bool(risk.get("danger", False))

        imminent = danger and ttc <= self.cfg.stop_ttc_s
        critical = clearance <= self.cfg.critical_clearance_m
        if imminent or critical:
            # Preserve no hard-coded passing choice: stop and let the next SAC
            # decision recover once the emergency geometry changes.
            action = np.array([0.0, 0.0], dtype=np.float32)
            reason = "EMERGENCY_STOP_TTC" if imminent else "EMERGENCY_STOP_CLEARANCE"
            return EmergencyShieldDecision(
                action=action,
                intervened=True,
                reason=reason,
                predicted_clearance=clearance,
                predicted_ttc=ttc,
                person=risk.get("person"),
            )

        return EmergencyShieldDecision(
            action=proposed,
            intervened=False,
            reason="RESIDUAL_SAC",
            predicted_clearance=clearance,
            predicted_ttc=ttc,
            person=risk.get("person"),
        )
