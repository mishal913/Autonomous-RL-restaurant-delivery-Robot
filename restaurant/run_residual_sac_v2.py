"""Live Genesis restaurant runtime for Residual SAC V2.

Normal dynamic-human avoidance is produced by Residual SAC. There is NO DWA
trajectory planner in this runtime. A minimal emergency stop is the only
hard-coded dynamic safety override.
"""
from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path
import time

import genesis as gs
import numpy as np
from stable_baselines3 import SAC

from baseline_controller_v2 import REPLAN_EVERY, compute_baseline_action
from residual_policy_v2 import (
    MAX_PEOPLE,
    PERSON_FEATURES,
    EmergencyHumanShield,
    build_residual_observation,
    compose_residual_action,
)
from restaurant_env_final import RestaurantDeliveryEnv
from restaurant_overlay import RestaurantOverlay


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Residual SAC V2 live restaurant robot")
    parser.add_argument("--model", type=Path, default=project / "dynamic_sac_residual_v2.zip")
    parser.add_argument("--seed", type=int, default=260909)
    parser.add_argument("--people", type=int, choices=range(0, 5), default=4)
    parser.add_argument("--no-clutter", action="store_true")
    parser.add_argument(
        "--human-cooperation",
        type=float,
        default=0.0,
        help="Default 0.0: people do not scripted-yield; SAC must avoid them.",
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
    controller_reason: str,
    action: np.ndarray,
    outcome: str = "RUNNING",
) -> None:
    order = info.get("order") or {}
    overlay.update(
        system="GENESIS LIVE / RESIDUAL SAC V2",
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
        planner=str(controller_reason).replace("_", " ").upper(),
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
        raise FileNotFoundError(
            f"Residual SAC V2 model not found: {args.model}\n"
            "Train it first with: python train_residual_sac_v2.py --total-timesteps 500000"
        )

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

    model = SAC.load(str(args.model), device="cpu")
    expected_obs = int(env.observation_space.shape[0]) + 2 + MAX_PEOPLE * PERSON_FEATURES
    model_shape = tuple(getattr(model.observation_space, "shape", ()))
    if model_shape != (expected_obs,):
        raise ValueError(
            f"Model observation shape {model_shape} is not V2 shape {(expected_obs,)}. "
            "Do NOT use dynamic_sac_residual_v1.zip here; V2 feature semantics are different."
        )

    overlay = RestaurantOverlay()
    overlay.attach(env.scene)
    shield = EmergencyHumanShield()
    pending_orders: deque[dict] = deque()

    episode = 0
    active = False
    cancelled = False
    base_obs = None
    info = None
    path = None
    previous_phase = ""
    steps = 0
    trace_rows: list[dict] = []
    trace_dir = project / "traces_residual_v2_live"
    trace_dir.mkdir(parents=True, exist_ok=True)

    overlay.update(
        system="GENESIS LIVE / RESIDUAL SAC V2",
        mission="SELECT FOOD AND PRESS ADD ORDER / START DELIVERY",
        outcome="READY",
        people=args.people,
    )

    print("=" * 88)
    print("ROBO SERVE - RESIDUAL SAC V2")
    print("=" * 88)
    print("Navigation      : A* deterministic baseline")
    print("Human avoidance : Residual SAC V2 (learned)")
    print("Safety          : imminent-collision emergency STOP only")
    print("Predictive DWA  : REMOVED from normal control")
    print(f"People          : {args.people} | cooperation={args.human_cooperation:.2f}")
    print("=" * 88)

    while viewer_alive(env.scene):
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

            request = pending_orders.popleft()
            episode += 1
            cancelled = False
            base_obs, info = env.reset(
                seed=args.seed + episode,
                options={
                    "table_id": request["table_id"],
                    "item": request["item"],
                    "quantity": request["quantity"],
                },
            )
            shield.reset()
            path = env.plan_path()
            previous_phase = "to_table" if env.package_picked else "to_pickup"
            steps = 0
            trace_rows = []
            active = True
            update_overlay(
                overlay,
                env,
                info,
                episode=episode,
                queue_depth=len(pending_orders),
                controller_reason="RESIDUAL_SAC",
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
            overlay.add_history(f"Episode {episode}: CANCELLED")
            overlay.update(
                mission="WAITING FOR NEXT ORDER",
                order_status="FAILED",
                outcome="CANCELLED",
                queue_depth=len(pending_orders),
            )
            continue

        phase = "to_table" if env.package_picked else "to_pickup"
        target_switched = phase != previous_phase
        if target_switched:
            path = None
        if path is None or steps % REPLAN_EVERY == 0 or target_switched:
            candidate = env.plan_path()
            if candidate:
                path = candidate
            previous_phase = phase

        baseline_action, waypoint = compute_baseline_action(env, path)
        policy_obs = build_residual_observation(env, base_obs, baseline_action)
        residual_action, _ = model.predict(policy_obs, deterministic=True)
        residual_action = np.asarray(residual_action, dtype=np.float32)
        proposed_action = compose_residual_action(baseline_action, residual_action)
        baseline_risk = env.predict_dynamic_risk(action=baseline_action, horizon=1.6)
        sac_risk = env.predict_dynamic_risk(action=proposed_action, horizon=1.6)

        decision = shield.filter_action(env, proposed_action)
        action = np.asarray(decision.action, dtype=np.float32)
        base_obs, _, terminated, truncated, info = env.step(action)
        steps += 1

        robot_xy = np.asarray(env._get_robot_position(), dtype=np.float64)
        row = {
            "episode": episode,
            "step": steps,
            "sim_time": float(getattr(env, "sim_time", 0.0)),
            "phase": str(info.get("phase", "")),
            "robot_x": float(robot_xy[0]),
            "robot_y": float(robot_xy[1]),
            "robot_yaw": float(env._get_robot_yaw()),
            "baseline_linear": float(baseline_action[0]),
            "baseline_turn": float(baseline_action[1]),
            "residual_linear": float(residual_action[0]),
            "residual_turn": float(residual_action[1]),
            "proposed_linear": float(proposed_action[0]),
            "proposed_turn": float(proposed_action[1]),
            "final_linear": float(action[0]),
            "final_turn": float(action[1]),
            "shield_intervened": int(decision.intervened),
            "shield_reason": str(decision.reason),
            "baseline_pred_clearance": float(baseline_risk.get("clearance", 99.0)),
            "sac_pred_clearance": float(sac_risk.get("clearance", 99.0)),
            "sac_pred_ttc": float(sac_risk.get("ttc", 99.0)),
            "env_dynamic_clearance": float(info.get("dynamic_clearance", 99.0)),
            "env_dynamic_ttc": float(info.get("dynamic_ttc", 99.0)),
            "env_dynamic_person": str(info.get("dynamic_person") or "NONE"),
            "collision": int(bool(info.get("collision", False))),
            "collision_type": str(info.get("collision_type") or ""),
        }
        for person_index, state in enumerate(env.person_states[: env.active_people_count]):
            current = np.asarray(state.get("current_xy", [0.0, 0.0]), dtype=np.float64)
            velocity = np.asarray(state.get("velocity_xy", [0.0, 0.0]), dtype=np.float64)
            row[f"p{person_index}_type"] = str(state.get("type", ""))
            row[f"p{person_index}_x"] = float(current[0])
            row[f"p{person_index}_y"] = float(current[1])
            row[f"p{person_index}_vx"] = float(velocity[0])
            row[f"p{person_index}_vy"] = float(velocity[1])
            row[f"p{person_index}_distance"] = float(np.linalg.norm(current - robot_xy))
        trace_rows.append(row)

        reason = decision.reason if decision.intervened else "RESIDUAL_SAC"
        if steps % 2 == 0 or decision.intervened or target_switched:
            update_overlay(
                overlay,
                env,
                info,
                episode=episode,
                queue_depth=len(pending_orders),
                controller_reason=reason,
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
                controller_reason=reason,
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
                print("V2 live trace:", trace_path)
            active = False

    stop_robot(env)
    env.close()


if __name__ == "__main__":
    main()
