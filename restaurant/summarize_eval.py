"""Recompute the aggregate summary from an already-saved eval CSV.

Use this if the per-episode lines scrolled the summary off your terminal --
no need to rerun evaluate_residual_sac.py, the CSV already has everything.

Usage:
    python summarize_eval.py eval_v1.csv
"""
from __future__ import annotations

import csv
import sys


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python summarize_eval.py <path_to_eval_csv>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        print("CSV is empty.")
        sys.exit(1)

    n = len(rows)

    def rate(field: str) -> float:
        return sum(int(r[field]) for r in rows) / n

    def mean(field: str) -> float:
        return sum(float(r[field]) for r in rows) / n

    clearances = sorted(float(r["min_dynamic_clearance"]) for r in rows)
    p5_clearance = clearances[max(0, int(0.05 * n) - 1)]

    print("=" * 60)
    print(f"Episodes                 : {n}")
    print(f"Success rate              : {rate('success'):.1%}")
    print(f"Dynamic (human) collision : {rate('dynamic_collision'):.1%}  <-- the key number")
    print(f"Static collision rate     : {rate('static_collision'):.1%}")
    print(f"Out-of-bounds rate        : {rate('out_of_bounds'):.1%}")
    print(f"5th-percentile clearance  : {p5_clearance:.2f} m")
    print(f"Mean DWA shield rate      : {mean('dwa_intervention_rate'):.1%}")
    if "mean_speed_mps" in rows[0]:
        print(f"Mean linear speed         : {mean('mean_speed_mps'):.2f} m/s")
        print(f"Mean mission time         : {mean('sim_time_s'):.1f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
