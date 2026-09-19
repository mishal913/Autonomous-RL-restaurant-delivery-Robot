from __future__ import annotations

import argparse
import csv
from collections import deque
import math
from pathlib import Path
import time

import genesis as gs
import numpy as np
from stable_baselines3 import SAC

from predictive_dwa import PredictiveDWAPlanner
from restaurant_env_final import RestaurantDeliveryEnv
from restaurant_overlay import RestaurantOverlay


REPLAN_EVERY = 8
WAYPOINT_LOOKAHEAD = 3
PLANNER_TURN_WEIGHT = 0.88
SAC_TURN_WEIGHT = 0.12
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


def static_lidar_metres(env: RestaurantDeliveryEnv) -> np.ndarray:
    """Furniture-only LiDAR. Moving people are left to Predictive DWA."""
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


def hybrid_action(
    env: RestaurantDeliveryEnv,
    sac_action: np.ndarray,
    waypoint: np.ndarray,
) -> np.ndarray:
    robot_xy = env._get_robot_position()
    yaw = env._get_robot_yaw()
    delta = waypoint - robot_xy
    desired_heading = math.atan2(float(delta[1]), float(delta[0]))
    heading_error = normalize_angle(desired_heading - yaw)
    error_deg = abs(math.degrees(heading_error))

    planner_turn = float(np.clip(1.40 * heading_error, -1.0, 1.0))
    turn = float(
        np.clip(
            PLANNER_TURN_WEIGHT * planner_turn
            + SAC_TURN_WEIGHT * float(sac_action[1]),
            -1.0,
            1.0,
        )
    )

    if error_deg > TURN_IN_PLACE_DEG:
        linear = 0.0
    elif error_deg > 26.0:
        linear = SLOW_ACTION
    else:
        linear = float(np.clip(max(CRUISE_ACTION, float(sac_action[0])), 0.0, 0.82))

    # Static furniture braking. Dynamic humans are intentionally excluded here.
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


def docking_action(env: RestaurantDeliveryEnv) -> np.ndarray:
    """Stable final approach to pickup/delivery target."""
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


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Final single-window Genesis restaurant delivery robot"
    )
    parser.add_argument("--model", type=Path, default=project / "dynamic_sac_v3.zip")
    parser.add_argument("--seed", type=int, default=260902)
    parser.add_argument("--people", type=int, choices=range(0, 5), default=4)
    parser.add_argument("--no-clutter", action="store_true")
    parser.add_argument(
        "--human-cooperation",
        type=float,
        default=0.35,
        help="Probability that a person cooperatively yields (0..1). DWA must handle the rest.",
    )
    return parser.parse_args()


def viewer_alive(scene) -> bool:
    viewer = getattr(scene, "viewer", None)
    if viewer is None:
        viewer = getattr(getattr(scene, "visualizer", None), "viewer", None)
    if viewer is None:
        return False
    value = getattr(viewer, "is_alive", None)
    return bool(value() if callable(value) else value)


def stop_robot(env: RestaurantDeliveryEnv) -> None:
    env.linear_velocity = 0.0
    env.angular_velocity = 0.0
    env.robot.control_dofs_velocity(np.zeros(4, dtype=np.float32), env.wheel_dofs)


def update_overlay(
    overlay: RestaurantOverlay,
    env: RestaurantDeliveryEnv,
    info: dict,
    *,
    episode: int,
    queue_depth: int,
    planner_reason: str,
    action: np.ndarray,
    outcome: str = "RUNNING",
) -> None:
    order = info.get("order") or {}
    overlay.update(
        system="GENESIS LIVE",
        episode=episode,
        mission="PICKUP -> DELIVERY",
        order_id=order.get("order_id", "-"),
        table_id=order.get("table_id", "-"),
        item=order.get("item", "-"),
        quantity=order.get("quantity", "-"),
        order_status=order.get("status", "-"),
        phase=str(info.get("phase", "-")).replace("_", " ").upper(),
        speed_mps=abs(float(action[0])) * env.MAX_LINEAR_SPEED,
        distance=float(info.get("distance_to_target", 0.0)),
        human=(info.get("dynamic_person") or "NONE").replace("-", " ").upper(),
        human_clearance=float(info.get("dynamic_clearance", 99.0)),
        ttc=float(info.get("dynamic_ttc", 99.0)),
        planner=str(planner_reason).replace("-", " ").upper(),
        queue_depth=queue_depth,
        step=int(info.get("step", env.current_step)),
        sim_time=float(info.get("sim_time", env.sim_time)),
        outcome=outcome,
        people=env.active_people_count,
    )


def main() -> None:
    args = parse_args()
    project = Path(__file__).resolve().parent
    if not args.model.exists():
        raise FileNotFoundError(f"SAC model not found: {args.model}")

    gs.init(backend=gs.cpu)
    env = RestaurantDeliveryEnv(
        render=True,
        embedded_gui=True,
        robot_urdf=project / "restaurant_delivery_robot.urdf",
        table_urdf=project / "restaurant_table_set.urdf",
        waiter_urdf=project / "restaurant_waiter.urdf",
        customer_urdf=project / "restaurant_customer.urdf",
        dynamic_people=args.people > 0,
        active_people_count=args.people,
        randomized_static_clutter=not args.no_clutter,
        human_cooperation_probability=float(np.clip(args.human_cooperation, 0.0, 1.0)),
    )

    try:
        model = SAC.load(args.model, env=env, device="cpu")
    except ValueError as exc:
        raise ValueError(
            "The supplied model is not compatible with the final V3 37-value observation space."
        ) from exc

    overlay = RestaurantOverlay()
    overlay.attach(env.scene)
    local_planner = PredictiveDWAPlanner()
    pending_orders: deque[dict] = deque()

    episode = 0
    active = False
    cancelled = False
    obs = None
    info = None
    path = None
    previous_phase = ""
    steps = 0
    trace_rows: list[dict] = []
    trace_dir = Path(__file__).resolve().parent / "traces_final_live"
    trace_dir.mkdir(parents=True, exist_ok=True)

    overlay.update(
        system="GENESIS LIVE",
        mission="SELECT FOOD AND PRESS ADD ORDER / START DELIVERY",
        outcome="READY",
        people=args.people,
    )

    print("=" * 84)
    print("ROBO SERVE - FINAL SINGLE-WINDOW RESTAURANT ROBOT")
    print("=" * 84)
    print("Interface  : embedded Genesis ImGui order + telemetry panel")
    print("Layout     : 8 numbered tables randomized every episode")
    print(f"People     : {args.people} moving waiters/customers")
    print("Navigation : A* + V3 SAC + Predictive DWA")
    print("Human mode : mixed cooperative/non-cooperative traffic")
    print("=" * 84)

    while viewer_alive(env.scene):
        # Collect every viewer-generated command without blocking the simulator.
        while True:
            command = overlay.next_command()
            if command is None:
                break
            kind = command.get("command")
            if kind == "order":
                pending_orders.append(command)
                overlay.set_queue_depth(len(pending_orders))
            elif kind == "cancel":
                if active:
                    cancelled = True
                else:
                    pending_orders.clear()
                    overlay.set_queue_depth(0)

        if not active:
            if not pending_orders:
                time.sleep(0.02)
                continue

            order_request = pending_orders.popleft()
            episode += 1
            cancelled = False
            obs, info = env.reset(
                seed=args.seed + episode,
                options={
                    "table_id": order_request["table_id"],
                    "item": order_request["item"],
                    "quantity": order_request["quantity"],
                },
            )
            local_planner.reset()
            path = env.plan_path()
            previous_phase = str(info["phase"])
            steps = 0
            trace_rows = []
            active = True
            update_overlay(
                overlay,
                env,
                info,
                episode=episode,
                queue_depth=len(pending_orders),
                planner_reason="nominal",
                action=np.zeros(2, dtype=np.float32),
            )
            print(
                f"Episode {episode}: {info['order']['order_id']} | "
                f"Table {info['order']['table_id']} | {info['order']['item']} x{info['order']['quantity']}"
            )
            continue

        if cancelled:
            stop_robot(env)
            env.order_system.mark_failed()
            active = False
            outcome = "CANCELLED"
            overlay.add_history(f"Episode {episode}: CANCELLED")
            overlay.update(
                mission="WAITING FOR NEXT ORDER",
                order_status="FAILED",
                outcome=outcome,
                queue_depth=len(pending_orders),
            )
            continue

        current_phase = "to_table" if env.package_picked else "to_pickup"
        target_switched = current_phase != previous_phase
        current_distance = float(
            np.linalg.norm(env._get_active_target() - env._get_robot_position())
        )
        if target_switched:
            path = None
        if path is None or steps % REPLAN_EVERY == 0 or target_switched:
            candidate_path = env.plan_path()
            if candidate_path:
                path = candidate_path
            previous_phase = current_phase

        sac_action, _ = model.predict(obs, deterministic=True)
        sac_action = np.asarray(sac_action, dtype=np.float32)

        if current_distance < DOCKING_DISTANCE:
            waypoint = env._get_active_target()
            proposed_action = docking_action(env)
        elif path:
            waypoint = choose_waypoint(
                path,
                env._get_robot_position(),
                env._get_active_target(),
            )
            proposed_action = hybrid_action(env, sac_action, waypoint)
        else:
            waypoint = env._get_active_target()
            proposed_action = np.array([0.0, 0.30], dtype=np.float32)

        decision = local_planner.filter_action(env, proposed_action, waypoint)
        action = np.asarray(decision.action, dtype=np.float32)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1

        # Diagnostic-only logging. This does not alter robot control.
        robot_xy = np.asarray(env._get_robot_position(), dtype=np.float64)
        row = {
            "episode": episode,
            "step": steps,
            "sim_time": float(getattr(env, "sim_time", 0.0)),
            "phase": str(info.get("phase", "")),
            "robot_x": float(robot_xy[0]),
            "robot_y": float(robot_xy[1]),
            "robot_yaw": float(env._get_robot_yaw()),
            "sac_linear": float(sac_action[0]),
            "sac_turn": float(sac_action[1]),
            "proposed_linear": float(proposed_action[0]),
            "proposed_turn": float(proposed_action[1]),
            "final_linear": float(action[0]),
            "final_turn": float(action[1]),
            "dwa_intervened": int(decision.intervened),
            "dwa_reason": str(decision.reason),
            "pred_dynamic_clearance": float(decision.predicted_dynamic_clearance),
            "pred_static_clearance": float(decision.predicted_static_clearance),
            "pred_ttc": float(decision.predicted_ttc),
            "env_dynamic_clearance": float(info.get("dynamic_clearance", 99.0)),
            "env_dynamic_ttc": float(info.get("dynamic_ttc", 99.0)),
            "env_dynamic_person": str(info.get("dynamic_person") or "NONE"),
            "collision": int(bool(info.get("collision", False))),
            "collision_type": str(info.get("collision_type") or ""),
        }
        for person_index, state in enumerate(env.person_states[: env.active_people_count]):
            current = np.asarray(
                state.get("current_xy", local_planner._person_at(state, float(env.sim_time))),
                dtype=np.float64,
            )
            future = np.asarray(
                local_planner._person_at(state, float(env.sim_time) + 0.10),
                dtype=np.float64,
            )
            velocity = (future - current) / 0.10
            row[f"p{person_index}_type"] = str(state.get("type", ""))
            row[f"p{person_index}_x"] = float(current[0])
            row[f"p{person_index}_y"] = float(current[1])
            row[f"p{person_index}_vx"] = float(velocity[0])
            row[f"p{person_index}_vy"] = float(velocity[1])
            row[f"p{person_index}_distance"] = float(np.linalg.norm(current - robot_xy))
            row[f"p{person_index}_cooperative"] = int(bool(state.get("cooperative", True)))
            row[f"p{person_index}_reciprocal"] = int(bool(state.get("reciprocal_active", False)))
        trace_rows.append(row)

        if steps % 2 == 0 or decision.intervened or target_switched:
            update_overlay(
                overlay,
                env,
                info,
                episode=episode,
                queue_depth=len(pending_orders),
                planner_reason=decision.reason,
                action=action,
            )

        if terminated or truncated:
            stop_robot(env)
            if info.get("success"):
                outcome = "SUCCESS"
            elif info.get("collision"):
                outcome = f"COLLISION/{str(info.get('collision_type') or 'unknown').upper()}"
            elif info.get("out_of_bounds"):
                outcome = "OUT OF BOUNDS"
            else:
                outcome = "TIMEOUT"

            update_overlay(
                overlay,
                env,
                info,
                episode=episode,
                queue_depth=len(pending_orders),
                planner_reason=decision.reason,
                action=np.zeros(2, dtype=np.float32),
                outcome=outcome,
            )
            overlay.add_history(
                f"E{episode} | T{info['order']['table_id']} | {outcome} | {steps} steps"
            )
            print(f"Outcome: {outcome} | steps={steps}")
            if trace_rows:
                order = info.get("order") or {}
                trace_path = trace_dir / (
                    f"episode_{episode:03d}_{order.get('order_id', 'order')}_"
                    f"table_{order.get('table_id', 'x')}_{outcome.replace('/', '_').replace(' ', '_')}.csv"
                )
                with trace_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(trace_rows)
                print("Live trace:", trace_path)
            active = False

    stop_robot(env)
    env.close()


if __name__ == "__main__":
    main()
