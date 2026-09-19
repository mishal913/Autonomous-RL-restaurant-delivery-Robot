"""Deterministic A* baseline for Residual SAC V2.

The baseline handles goal-directed navigation and static furniture safety.
Dynamic-human avoidance is intentionally NOT hard-coded here; Residual SAC V2
learns that correction.  A tiny emergency shield lives in residual_policy_v2.py.
"""
from __future__ import annotations

import math
import numpy as np

REPLAN_EVERY = 8
WAYPOINT_LOOKAHEAD = 3
CRUISE_ACTION = 0.78
SLOW_ACTION = 0.20
TURN_IN_PLACE_DEG = 48.0
# V1 started direct docking at 2.20 m, which caused early corner cutting.
DOCKING_DISTANCE = 1.30


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def choose_waypoint(path, robot_xy: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    if not path:
        return np.asarray(fallback, dtype=np.float32)
    distances = [float(np.linalg.norm(point - robot_xy)) for point in path]
    nearest = int(np.argmin(distances))
    target_index = min(nearest + WAYPOINT_LOOKAHEAD, len(path) - 1)
    return np.asarray(path[target_index], dtype=np.float32)


def static_lidar_metres(env) -> np.ndarray:
    """Furniture-only LiDAR; people are excluded on purpose."""
    origin = env._get_robot_position()
    yaw = float(env._get_robot_yaw())
    layout = env.static_obstacle_data + env.table_obstacle_data + env.decor_obstacle_data
    values = []
    for index in range(env.NUM_LIDAR_RAYS):
        angle = yaw + 2.0 * math.pi * index / env.NUM_LIDAR_RAYS
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        closest = env.LIDAR_MAX_DISTANCE
        for x, y, sx, sy, _ in layout:
            distance = env._ray_rectangle_distance(origin, direction, x, y, sx, sy)
            if distance is not None:
                closest = min(closest, float(distance))
        values.append(float(closest))
    return np.asarray(values, dtype=np.float32)


def _apply_static_braking(env, linear: float, turn: float) -> tuple[float, float]:
    lidar_m = static_lidar_metres(env)
    front_min = float(np.min(lidar_m[[14, 15, 0, 1, 2]]))

    if front_min < 0.96:
        linear = 0.0
        if abs(turn) < 0.18:
            left_clearance = float(np.median(lidar_m[[2, 3, 4, 5, 6]]))
            right_clearance = float(np.median(lidar_m[[10, 11, 12, 13, 14]]))
            turn = 0.38 if left_clearance >= right_clearance else -0.38
    elif front_min < 1.18:
        linear = min(linear, 0.10)
    elif front_min < 1.42:
        linear = min(linear, 0.22)
    elif front_min < 1.70:
        linear = min(linear, 0.34)
    return float(linear), float(turn)


def baseline_hybrid_action(env, waypoint: np.ndarray) -> np.ndarray:
    robot_xy = env._get_robot_position()
    yaw = env._get_robot_yaw()
    delta = waypoint - robot_xy
    desired_heading = math.atan2(float(delta[1]), float(delta[0]))
    heading_error = normalize_angle(desired_heading - yaw)
    error_deg = abs(math.degrees(heading_error))

    turn = float(np.clip(1.40 * heading_error, -1.0, 1.0))
    if error_deg > TURN_IN_PLACE_DEG:
        linear = 0.0
    elif error_deg > 26.0:
        linear = SLOW_ACTION
    else:
        linear = float(np.clip(CRUISE_ACTION, 0.0, 0.82))

    linear, turn = _apply_static_braking(env, linear, turn)
    return np.array([linear, turn], dtype=np.float32)


def docking_action(env) -> np.ndarray:
    """Short final approach with the same static-furniture braking as A*."""
    robot_xy = env._get_robot_position()
    target = env._get_active_target()
    yaw = env._get_robot_yaw()
    delta = target - robot_xy
    distance = float(np.linalg.norm(delta))
    desired = math.atan2(float(delta[1]), float(delta[0]))
    heading_error = normalize_angle(desired - yaw)
    error_deg = abs(math.degrees(heading_error))
    turn = float(np.clip(1.45 * heading_error, -0.75, 0.75))

    if error_deg > 34.0:
        linear = 0.0
    elif error_deg > 18.0:
        linear = 0.16
    else:
        linear = float(np.clip(0.18 + 0.14 * distance, 0.20, 0.40))

    linear, turn = _apply_static_braking(env, linear, turn)
    return np.array([linear, turn], dtype=np.float32)


def compute_baseline_action(env, path) -> tuple[np.ndarray, np.ndarray]:
    robot_xy = env._get_robot_position()
    target = env._get_active_target()
    distance = float(np.linalg.norm(target - robot_xy))
    if distance < DOCKING_DISTANCE:
        waypoint = target
        action = docking_action(env)
    elif path:
        waypoint = choose_waypoint(path, robot_xy, target)
        action = baseline_hybrid_action(env, waypoint)
    else:
        waypoint = target
        action = np.array([0.0, 0.30], dtype=np.float32)
    return action, waypoint
