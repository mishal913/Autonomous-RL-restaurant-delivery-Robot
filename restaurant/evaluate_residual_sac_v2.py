"""Batch evaluation for Residual SAC V2 without DWA.

The key metrics are dynamic collision rate AND emergency-shield rate. A low
collision rate achieved only by frequent shield stops means SAC has not yet
learned reliable avoidance.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import genesis as gs
import numpy as np
from stable_baselines3 import SAC

from residual_env_v2 import ResidualHumanAvoidanceEnvV2, ResidualEnvV2Config
from restaurant_env_final import RestaurantDeliveryEnv


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate Residual SAC V2")
    parser.add_argument("--model", type=Path, default=project / "dynamic_sac_residual_v2.zip")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=200_000)
    parser.add_argument("--people", type=int, default=4)
    parser.add_argument("--assets-dir", type=Path, default=project)
    parser.add_argument("--out", type=Path, default=project / "eval_residual_v2.csv")
    return parser.parse_args()


def run_episode(env: ResidualHumanAvoidanceEnvV2, model: SAC, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    terminated = truncated = False
    steps = 0
    shield_steps = 0
    success = dynamic_collision = static_collision = out_of_bounds = False
    min_clearance = 99.0
    speed_samples = []
    residual_magnitudes = []
    clearance_improvements = []

    while not (terminated or truncated):
        residual, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(residual)
        steps += 1
        shield_steps += int(bool(info.get("emergency_shield_intervened", False)))
        speed_samples.append(abs(float(env.env.linear_velocity)))
        residual_magnitudes.append(float(np.linalg.norm(np.asarray(info.get("residual_action", [0, 0])))))
        min_clearance = min(min_clearance, float(info.get("dynamic_clearance", 99.0)))
        clearance_improvements.append(
            float(info.get("sac_predicted_clearance", 99.0))
            - float(info.get("baseline_predicted_clearance", 99.0))
        )
        success = success or bool(info.get("success"))
        dynamic_collision = dynamic_collision or info.get("collision_type") == "dynamic"
        static_collision = static_collision or info.get("collision_type") == "static"
        out_of_bounds = out_of_bounds or bool(info.get("out_of_bounds"))

    return {
        "seed": seed,
        "steps": steps,
        "sim_time_s": steps * env.env.ACTION_REPEAT * 0.01,
        "success": int(success),
        "dynamic_collision": int(dynamic_collision),
        "static_collision": int(static_collision),
        "out_of_bounds": int(out_of_bounds),
        "min_dynamic_clearance": min_clearance,
        "emergency_shield_rate": shield_steps / max(1, steps),
        "mean_speed_mps": float(np.mean(speed_samples)) if speed_samples else 0.0,
        "mean_residual_magnitude": float(np.mean(residual_magnitudes)) if residual_magnitudes else 0.0,
        "mean_clearance_improvement_vs_baseline": (
            float(np.mean(clearance_improvements)) if clearance_improvements else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    gs.init(backend=gs.cpu)

    base_env = RestaurantDeliveryEnv(
        render=False,
        embedded_gui=False,
        robot_urdf=args.assets_dir / "restaurant_delivery_robot.urdf",
        table_urdf=args.assets_dir / "restaurant_table_set.urdf",
        waiter_urdf=args.assets_dir / "restaurant_waiter.urdf",
        customer_urdf=args.assets_dir / "restaurant_customer.urdf",
        active_people_count=args.people,
        human_cooperation_probability=0.0,
    )
    base_env.people_speed_scale = 1.0
    wrapped = ResidualHumanAvoidanceEnvV2(base_env, ResidualEnvV2Config())
    model = SAC.load(str(args.model), device="cpu")

    rows = []
    for offset in range(args.episodes):
        row = run_episode(wrapped, model, args.seed_start + offset)
        rows.append(row)
        print(
            f"[{offset+1}/{args.episodes}] success={row['success']} "
            f"dyn={row['dynamic_collision']} static={row['static_collision']} "
            f"shield={row['emergency_shield_rate']:.3f} "
            f"clearance={row['min_dynamic_clearance']:.2f}m"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = max(1, len(rows))
    success_rate = sum(r["success"] for r in rows) / n
    dynamic_rate = sum(r["dynamic_collision"] for r in rows) / n
    static_rate = sum(r["static_collision"] for r in rows) / n
    shield_rate = sum(r["emergency_shield_rate"] for r in rows) / n
    mean_speed = sum(r["mean_speed_mps"] for r in rows) / n
    clearances = sorted(r["min_dynamic_clearance"] for r in rows)
    p5 = clearances[max(0, int(0.05 * len(clearances)) - 1)] if clearances else 99.0

    print("=" * 66)
    print(f"Episodes                     : {len(rows)}")
    print(f"Success rate                 : {success_rate:.1%}")
    print(f"Dynamic collision rate       : {dynamic_rate:.1%}  <-- primary SAC-human metric")
    print(f"Static collision rate        : {static_rate:.1%}")
    print(f"Mean emergency shield rate   : {shield_rate:.1%}  <-- should become very low")
    print(f"5th percentile clearance     : {p5:.2f} m")
    print(f"Mean speed                   : {mean_speed:.2f} m/s")
    print("Suggested graduation gates   : success >=80%, dynamic <=5%, shield <=5%")
    print("=" * 66)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
