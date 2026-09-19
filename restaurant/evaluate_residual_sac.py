"""Batch evaluation for the trained residual human-avoidance policy.

This is the script that turns "reliable avoidance of all 4 humans" from a
judgment call into a number. Run it against the OLD dynamic_sac_v3 blend
(via --baseline-mode sac-blend) and the NEW residual policy
(--baseline-mode residual) over the same seed range and compare.

Usage:
    python evaluate_residual_sac.py --model dynamic_sac_residual_v1.zip \
        --episodes 200 --people 4 --human-cooperation 0.35 \
        --out eval_residual_v1.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import genesis as gs
import numpy as np
from stable_baselines3 import SAC

from residual_env import ResidualEnvConfig, ResidualHumanAvoidanceEnv
from restaurant_env_final import RestaurantDeliveryEnv


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate residual human-avoidance SAC")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=100_000)
    parser.add_argument("--people", type=int, default=4)
    parser.add_argument("--human-cooperation", type=float, default=0.35)
    parser.add_argument("--assets-dir", type=Path, default=project)
    parser.add_argument("--out", type=Path, default=project / "eval_results.csv")
    return parser.parse_args()


def run_episode(wrapped_env: ResidualHumanAvoidanceEnv, model: SAC, seed: int) -> dict:
    obs, info = wrapped_env.reset(seed=seed)
    terminated = truncated = False
    steps = 0
    min_dynamic_clearance = float("inf")
    min_dynamic_ttc = float("inf")
    intervened_steps = 0
    dynamic_collision = False
    static_collision = False
    out_of_bounds = False
    success = False
    speed_samples = []

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = wrapped_env.step(action)
        steps += 1
        speed_samples.append(abs(float(wrapped_env.env.linear_velocity)))
        min_dynamic_clearance = min(min_dynamic_clearance, float(info.get("dynamic_clearance", 99.0)))
        if info.get("dynamic_danger"):
            min_dynamic_ttc = min(min_dynamic_ttc, float(info.get("dynamic_ttc", 99.0)))
        if info.get("dwa_intervened"):
            intervened_steps += 1
        if info.get("collision_type") == "dynamic":
            dynamic_collision = True
        if info.get("collision_type") == "static":
            static_collision = True
        if info.get("out_of_bounds"):
            out_of_bounds = True
        if info.get("success"):
            success = True

    return {
        "seed": seed,
        "steps": steps,
        "sim_time_s": steps * wrapped_env.env.ACTION_REPEAT * 0.01,
        "mean_speed_mps": float(np.mean(speed_samples)) if speed_samples else 0.0,
        "success": int(success),
        "dynamic_collision": int(dynamic_collision),
        "static_collision": int(static_collision),
        "out_of_bounds": int(out_of_bounds),
        "min_dynamic_clearance": (
            min_dynamic_clearance if np.isfinite(min_dynamic_clearance) else 99.0
        ),
        "min_dynamic_ttc": min_dynamic_ttc if np.isfinite(min_dynamic_ttc) else 99.0,
        "dwa_intervention_rate": intervened_steps / max(1, steps),
    }


def main() -> None:
    args = parse_args()
    gs.init(backend=gs.cpu)

    base_env = RestaurantDeliveryEnv(
        render=False,
        embedded_gui=False,
        robot_urdf=args.assets_dir / "restaurant_delivery_robot.urdf",
        table_urdf=args.assets_dir / "restaurant_table_set.urdf",
        waiter_urdf=args.assets_dir / "restaurant_waiter.urdf",
        customer_urdf=args.assets_dir / "restaurant_customer.urdf",
        active_people_count=args.people,
        human_cooperation_probability=args.human_cooperation,
    )
    wrapped_env = ResidualHumanAvoidanceEnv(base_env, ResidualEnvConfig())
    model = SAC.load(str(args.model))

    rows = []
    for offset in range(args.episodes):
        seed = args.seed_start + offset
        row = run_episode(wrapped_env, model, seed)
        rows.append(row)
        print(
            f"[{offset + 1}/{args.episodes}] seed={seed} "
            f"success={row['success']} dyn_collision={row['dynamic_collision']} "
            f"min_clearance={row['min_dynamic_clearance']:.2f}m "
            f"shield_rate={row['dwa_intervention_rate']:.3f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    success_rate = sum(r["success"] for r in rows) / n
    dynamic_collision_rate = sum(r["dynamic_collision"] for r in rows) / n
    static_collision_rate = sum(r["static_collision"] for r in rows) / n
    oob_rate = sum(r["out_of_bounds"] for r in rows) / n
    clearances = sorted(r["min_dynamic_clearance"] for r in rows)
    p5_clearance = clearances[max(0, int(0.05 * n) - 1)]
    mean_shield_rate = sum(r["dwa_intervention_rate"] for r in rows) / n
    mean_speed = sum(r["mean_speed_mps"] for r in rows) / n
    mean_sim_time = sum(r["sim_time_s"] for r in rows) / n

    print("=" * 60)
    print(f"Episodes                 : {n}")
    print(f"Success rate              : {success_rate:.1%}")
    print(f"Dynamic (human) collision : {dynamic_collision_rate:.1%}  <-- the key number")
    print(f"Static collision rate     : {static_collision_rate:.1%}")
    print(f"Out-of-bounds rate        : {oob_rate:.1%}")
    print(f"5th-percentile clearance  : {p5_clearance:.2f} m")
    print(f"Mean DWA shield rate      : {mean_shield_rate:.1%}  (lower = policy needs less rescue)")
    print(f"Mean linear speed         : {mean_speed:.2f} m/s  <-- catches over-cautious freezing/crawling")
    print(f"Mean mission time         : {mean_sim_time:.1f} s")
    print("=" * 60)
    print(f"Full per-episode data written to {args.out}")


if __name__ == "__main__":
    main()
