"""Deterministic baseline navigation controller.

Extracted from run_final_restaurant.py so it has exactly one implementation
shared by:
  - the live runtime (run_final_restaurant.py should import from here instead
    of keeping its own copy)
  - the residual-SAC training environment (residual_env.py)

This controller has NO learned component. It is pure A*-waypoint pursuit +
LiDAR-based furniture braking. In the original runtime script it was blended
with a 0.12-weighted SAC turn contribution; here it is used as-is, and 100%
of the learned correction comes from the residual policy instead. Keeping
the baseline learning-free makes it a stable, reproducible reference point
for the residual network to correct against.
"""
from __future__ import annotations

import math

import numpy as np

REPLAN_EVERY = 8
WAYPOINT_LOOKAHEAD = 3
PLANNER_TURN_WEIGHT = 1.0  # was 0.88 in the SAC-blended runtime; residual
                            # policy supplies the remaining correction instead
                            # of a second learned signal fighting this one.
CRUISE_ACTION = 0.78
SLOW_ACTION = 0.20
TURN_IN_PLACE_DEG = 48.0
DOCKING_DISTANCE = 2.20


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
    """Furniture-only LiDAR. Moving people are intentionally excluded here;
    the DWA safety shield handles them using true kinematic state instead."""
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


def baseline_hybrid_action(env, waypoint: np.ndarray) -> np.ndarray:
    """Pure A*-waypoint pursuit + furniture braking. No learned input."""
    robot_xy = env._get_robot_position()
    yaw = env._get_robot_yaw()
    delta = waypoint - robot_xy
    desired_heading = math.atan2(float(delta[1]), float(delta[0]))
    heading_error = normalize_angle(desired_heading - yaw)
    error_deg = abs(math.degrees(heading_error))

    turn = float(np.clip(PLANNER_TURN_WEIGHT * 1.40 * heading_error, -1.0, 1.0))

    if error_deg > TURN_IN_PLACE_DEG:
        linear = 0.0
    elif error_deg > 26.0:
        linear = SLOW_ACTION
    else:
        linear = float(np.clip(CRUISE_ACTION, 0.0, 0.82))

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

    return np.array([linear, turn], dtype=np.float32)


def docking_action(env) -> np.ndarray:
    """Stable final approach to the pickup/delivery target. No learned input."""
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
        linear = 0.14
    else:
        linear = float(np.clip(0.16 + 0.10 * distance, 0.18, 0.34))
    return np.array([linear, turn], dtype=np.float32)


def compute_baseline_action(env, path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (action, waypoint_used) exactly like the runtime loop's logic."""
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
