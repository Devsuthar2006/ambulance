"""CLI command-center runner for 100-call interactive demo."""

import argparse
import sys
import time
from src.config import DEFAULT_SEED
from src.generator import generate_world
from src.dispatcher import get_dispatcher
from src.simulator import Simulator
from src.triage.mock_triage import MockTriage
from src.voice.demo_provider import DemoAudioProvider
from src.metrics import compute_metrics


def run_100_call_demo(calls: int = 100, policy_name: str = "coverage", speed: float = 1.0, seed: int = DEFAULT_SEED):
    """Simulates emergency calls streaming through AI Triage -> Dispatch -> Resolution."""
    print("=" * 80)
    print("  RESQAI — REAL-TIME EMERGENCY RESPONSE INTELLIGENCE PLATFORM")
    print(f"  MODE B: {calls}-CALL LIVE CONTROL ROOM DEMO | POLICY: {policy_name.upper()}")
    print("  SAFETY: DEMO / SIMULATION — NOT A REAL EMERGENCY SERVICE")
    print("=" * 80)
    print()

    triage = MockTriage()
    voice = DemoAudioProvider()
    dispatcher = get_dispatcher(policy_name)
    vehicles, incidents = generate_world(seed)
    incidents = incidents[:calls]

    # Run discrete event simulation
    sim = Simulator(dispatcher=dispatcher)
    res = sim.run(vehicles, incidents)

    assigned_map = {inc.id: inc for inc in res.incidents if inc.assigned_vehicle_id is not None}

    print(f"{'TIME':<8} | {'EVENT':<14} | {'DETAILS':<52}")
    print("-" * 80)

    # Stream the first 25 calls visually in real-time, then batch remaining
    display_limit = min(25, calls)
    for i in range(display_limit):
        inc = incidents[i]
        preset = voice.get_preset(i)
        triage_res = triage.triage(preset["text"])

        # Call received
        print(f"t={inc.arrival_minute:02d}:00   | CALL RECEIVED  | Call #{inc.id:02d}: \"{preset['text'][:46]}...\"")
        time.sleep(0.05 / speed)

        # AI Triage
        prio_tag = f"P{triage_res.priority}"
        print(f"t={inc.arrival_minute:02d}:01   | AI TRIAGE      | [{prio_tag}] {triage_res.emergency_type} ({triage_res.confidence*100:.0f}% conf) Evidence: {', '.join(triage_res.evidence[:2])}")
        time.sleep(0.05 / speed)

        # Dispatch
        assigned_inc = assigned_map.get(inc.id)
        if assigned_inc:
            v_id = assigned_inc.assigned_vehicle_id
            travel = assigned_inc.response_time - (assigned_inc.assigned_minute - assigned_inc.arrival_minute)
            print(f"t={inc.arrival_minute:02d}:02   | DISPATCHED     | 🚑 Ambulance #{v_id:02d} dispatched -> ETA {travel:.1f}m (Reach: t={assigned_inc.reached_minute:.1f}m)")
        else:
            print(f"t={inc.arrival_minute:02d}:02   | QUEUED         | Fleet saturated. Call #{inc.id} placed in priority queue.")

        print("-" * 80)
        time.sleep(0.1 / speed)

    if calls > display_limit:
        print(f"\n... Processing remaining {calls - display_limit} calls through AI Triage & Dispatch Engine ...")

    # Final scored summary
    metrics = compute_metrics(res, seed=seed)
    print("\n" + "=" * 80)
    print("  FINAL DEMO SUMMARY METRICS")
    print("=" * 80)
    print(f"  Total Calls Processed:           {metrics.total_incidents}")
    print(f"  Calls Successfully Dispatched:   {metrics.assigned_incidents}")
    print(f"  Priority-Weighted Mean RT:       {metrics.priority_weighted_response_time:.4f} min")
    print(f"  Priority-3 (Critical) Mean RT:   {metrics.priority3_response_time:.4f} min")
    print(f"  Coverage Outage Minutes:         {metrics.coverage_outage_minutes} / 121 min")
    print(f"  Simulation Engine Runtime:       {metrics.runtime_seconds:.4f} s")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="RESQAI 100-Call Emergency Demo")
    parser.add_argument("--calls", type=int, default=100, help="Number of calls to simulate (default: 100)")
    parser.add_argument("--policy", type=str, choices=["nearest", "coverage", "adaptive"], default="coverage", help="Dispatch policy")
    parser.add_argument("--speed", type=float, default=2.0, help="Playback speed multiplier")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed")
    args = parser.parse_args()

    run_100_call_demo(calls=args.calls, policy_name=args.policy, speed=args.speed, seed=args.seed)


if __name__ == "__main__":
    main()
