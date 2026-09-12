"""Reporting and export utilities for CSV run logs and JSON summaries."""

import argparse
import csv
import json
import os
from pathlib import Path

from src.config import DEFAULT_SEED, DEFAULT_COVERAGE_PENALTY
from src.generator import generate_world
from src.dispatcher import NaiveDispatcher, CoverageAwareDispatcher
from src.simulator import Simulator
from src.metrics import compute_metrics


def export_run_log_csv(incidents, output_path: str | Path) -> None:
    """Export per-incident assignment records to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "incident_id",
            "arrival_minute",
            "priority",
            "assigned_vehicle_id",
            "reached_minute",
            "response_time",
        ])
        for inc in incidents:
            resp_str = f"{inc.response_time:.4f}" if inc.response_time is not None else ""
            reached_str = f"{inc.reached_minute:.4f}" if inc.reached_minute is not None else ""
            veh_id_str = inc.assigned_vehicle_id if inc.assigned_vehicle_id is not None else ""
            writer.writerow([
                inc.id,
                inc.arrival_minute,
                inc.priority,
                veh_id_str,
                reached_str,
                resp_str,
            ])


def export_summary_json(metrics, output_path: str | Path) -> None:
    """Export summary metrics dictionary to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics.to_summary_dict(), f, indent=2)


def run_parameter_sweep(
    seed: int = DEFAULT_SEED,
    penalties: list[float] | None = None,
) -> list[dict]:
    """Run parameter sweep over coverage penalties and return comparative results."""
    if penalties is None:
        penalties = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]

    results = []
    vehicles_init, incidents_init = generate_world(seed=seed)

    for p in penalties:
        if p == 0.0:
            dispatcher = NaiveDispatcher()
            name = "Naive (penalty=0)"
        else:
            dispatcher = CoverageAwareDispatcher(coverage_penalty=p)
            name = f"CoverageAware (penalty={p})"

        sim = Simulator(dispatcher=dispatcher)
        sim_res = sim.run(vehicles_init, incidents_init)
        metrics = compute_metrics(sim_res, seed=seed)
        summary = metrics.to_summary_dict()
        summary["policy"] = name
        summary["penalty"] = p
        results.append(summary)

    return results


def main() -> None:
    """CLI runner for dispatch simulation and reporting."""
    parser = argparse.ArgumentParser(description="Emergency Fleet Assignment Simulator")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument(
        "--policy",
        type=str,
        choices=["naive", "coverage"],
        default="coverage",
        help="Dispatch policy",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=DEFAULT_COVERAGE_PENALTY,
        help="Coverage penalty for coverage policy",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default="outputs/run_log.csv",
        help="Path to output CSV",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="outputs/summary.json",
        help="Path to output JSON",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run parameter sweep and display comparative table",
    )

    args = parser.parse_args()

    if args.sweep:
        print(f"\n--- Running Parameter Sweep (Seed: {args.seed}) ---")
        sweep_results = run_parameter_sweep(seed=args.seed)
        print(
            f"{'Policy':<28} | {'Weighted RT':<12} | {'P3 RT':<10} | "
            f"{'Outage Mins':<12} | {'Runtime (s)':<12}"
        )
        print("-" * 84)
        for r in sweep_results:
            print(
                f"{r['policy']:<28} | "
                f"{r['priority_weighted_response_time']:<12.4f} | "
                f"{r['priority3_response_time']:<10.4f} | "
                f"{r['coverage_outage_minutes']:<12} | "
                f"{r['runtime_seconds']:<12.6f}"
            )
        sweep_out_path = Path("outputs/sweep_results.json")
        sweep_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sweep_out_path, "w", encoding="utf-8") as f:
            json.dump(sweep_results, f, indent=2)
        print(f"\nSaved sweep results to {sweep_out_path}")
        return

    # Single simulation run
    vehicles, incidents = generate_world(seed=args.seed)
    if args.policy == "naive":
        dispatcher = NaiveDispatcher()
    else:
        dispatcher = CoverageAwareDispatcher(coverage_penalty=args.penalty)

    sim = Simulator(dispatcher=dispatcher)
    sim_result = sim.run(vehicles, incidents)
    metrics = compute_metrics(sim_result, seed=args.seed)

    export_run_log_csv(sim_result.incidents, args.csv_out)
    export_summary_json(metrics, args.json_out)

    print("\n================ Simulation Summary ================")
    print(f"Policy: {args.policy} (penalty={args.penalty if args.policy == 'coverage' else 0})")
    print(f"Random Seed: {args.seed}")
    print(f"Priority-Weighted Mean Response Time: {metrics.priority_weighted_response_time:.4f} min")
    print(f"Priority-3 Mean Response Time:        {metrics.priority3_response_time:.4f} min")
    print(f"Coverage Outage Minutes (out of 121): {metrics.coverage_outage_minutes} min")
    print(f"Runtime:                              {metrics.runtime_seconds:.6f} s")
    print(f"Run Log CSV:                          {args.csv_out}")
    print(f"Summary JSON:                         {args.json_out}")
    print("====================================================\n")


if __name__ == "__main__":
    main()
