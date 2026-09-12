"""Exports high-fidelity continuous simulation playback data with event logs."""

import json
from pathlib import Path
import numpy as np

from src.config import DEFAULT_SEED, DEFAULT_COVERAGE_PENALTY
from src.generator import generate_world
from src.dispatcher import NaiveDispatcher, CoverageAwareDispatcher
from src.simulator import Simulator
from src.traffic import get_default_traffic_model


def build_simulation_trace(dispatcher, vehicles_init, incidents_init, traffic_model=None):
    """Run simulation and collect full assignment paths, events, and timeline states."""
    sim = Simulator(dispatcher=dispatcher, traffic_model=traffic_model)
    res = sim.run(vehicles_init, incidents_init)

    # Initial vehicle map
    veh_initial = {v.id: {"x": round(float(v.x), 2), "y": round(float(v.y), 2), "quadrant": v.quadrant} for v in vehicles_init}

    # Build per-vehicle assignment history
    # PRD §3 guarantee: each incident has exactly one assigned_vehicle_id.
    # We sort globally by (assigned_minute, incident_id) then group per vehicle,
    # ensuring each vehicle's list is in strict chronological order.
    veh_assignments: dict[int, list] = {v.id: [] for v in vehicles_init}

    assigned_incs = [inc for inc in res.incidents if inc.assigned_vehicle_id is not None]
    # Sort globally so grouping preserves causal order within each vehicle
    assigned_incs.sort(key=lambda inc: (inc.assigned_minute, inc.id))

    # Track current resting position per vehicle (starts at initial base)
    veh_current_pos = {v.id: (v.x, v.y) for v in vehicles_init}

    for inc in assigned_incs:
        v_id = inc.assigned_vehicle_id
        start_x, start_y = veh_current_pos[v_id]
        dest_x, dest_y = inc.x, inc.y

        veh_assignments[v_id].append({
            "incident_id": inc.id,
            "start_x": round(float(start_x), 2),
            "start_y": round(float(start_y), 2),
            "dest_x": round(float(dest_x), 2),
            "dest_y": round(float(dest_y), 2),
            "assigned_minute": round(float(inc.assigned_minute), 2),
            "reached_minute": round(float(inc.reached_minute), 2),
            "completed_minute": round(float(inc.completed_minute), 2),
            "priority": inc.priority,
        })
        # After completing service, vehicle rests at incident site (PRD §3: idles at incident location)
        veh_current_pos[v_id] = (dest_x, dest_y)

    # Guarantee each vehicle's own list is in strict chronological order
    # (handles edge cases where two vehicles got assigned at the exact same minute)
    for v_id in veh_assignments:
        veh_assignments[v_id].sort(key=lambda a: (a["assigned_minute"], a["incident_id"]))

    # Build per-minute rich human-readable events
    # Pre-index events by minute
    arrivals_by_min = {}
    assignments_by_min = {}
    arrivals_on_scene_by_min = {}
    completions_by_min = {}

    for inc in res.incidents:
        arrivals_by_min.setdefault(inc.arrival_minute, []).append(inc)
        if inc.assigned_minute is not None:
            assignments_by_min.setdefault(int(inc.assigned_minute), []).append(inc)
        if inc.reached_minute is not None:
            arrivals_on_scene_by_min.setdefault(int(np.floor(inc.reached_minute)), []).append(inc)
        if inc.completed_minute is not None:
            completions_by_min.setdefault(int(np.floor(inc.completed_minute)), []).append(inc)

    timeline = []
    cumulative_outages = 0

    prio_labels = {1: "P1 Routine", 2: "P2 Urgent", 3: "P3 CRITICAL"}

    for t in range(121):
        idle_counts = res.coverage_history[t][1]
        has_outage = any(c == 0 for c in idle_counts)
        if has_outage:
            cumulative_outages += 1

        events = []
        # Outage alerts
        if has_outage:
            zero_quads = [f"Q{q}" for q, c in enumerate(idle_counts) if c == 0]
            events.append({
                "type": "outage",
                "text": f"🚨 Coverage Outage: {', '.join(zero_quads)} has 0 idle ambulances!",
                "level": "danger",
            })

        # Arrivals
        if t in arrivals_by_min:
            for inc in arrivals_by_min[t]:
                events.append({
                    "type": "arrival",
                    "text": f"📍 Call #{inc.id} arrived [{prio_labels[inc.priority]}] at ({inc.x:.1f}, {inc.y:.1f})",
                    "level": "critical" if inc.priority == 3 else ("warning" if inc.priority == 2 else "info"),
                    "incident_id": inc.id,
                })

        # Dispatches
        if t in assignments_by_min:
            for inc in assignments_by_min[t]:
                events.append({
                    "type": "dispatch",
                    "text": f"🚑 Ambulance #{inc.assigned_vehicle_id} dispatched to Call #{inc.id} (ETA: {inc.reached_minute:.1f}m)",
                    "level": "dispatch",
                    "incident_id": inc.id,
                    "vehicle_id": inc.assigned_vehicle_id,
                })

        # Completions
        if t in completions_by_min:
            for inc in completions_by_min[t]:
                events.append({
                    "type": "complete",
                    "text": f"✅ Ambulance #{inc.assigned_vehicle_id} finished service for Call #{inc.id}, now idle",
                    "level": "success",
                    "vehicle_id": inc.assigned_vehicle_id,
                })

        timeline.append({
            "minute": t,
            "idle_counts": idle_counts,
            "has_outage": has_outage,
            "cumulative_outages": cumulative_outages,
            "events": events,
        })

    incidents_data = []
    for inc in res.incidents:
        incidents_data.append({
            "id": inc.id,
            "arrival_minute": inc.arrival_minute,
            "x": round(float(inc.x), 2),
            "y": round(float(inc.y), 2),
            "priority": inc.priority,
            "weight": inc.weight,
            "assigned_vehicle_id": inc.assigned_vehicle_id,
            "assigned_minute": round(float(inc.assigned_minute), 2) if inc.assigned_minute is not None else None,
            "reached_minute": round(float(inc.reached_minute), 2) if inc.reached_minute is not None else None,
            "completed_minute": round(float(inc.completed_minute), 2) if inc.completed_minute is not None else None,
            "response_time": round(float(inc.response_time), 2) if inc.response_time is not None else None,
        })

    # Summary metrics
    assigned = [i for i in res.incidents if i.response_time is not None]
    weighted_rt = float(np.sum([i.weight * i.response_time for i in assigned]) / np.sum([i.weight for i in assigned]))
    p3_assigned = [i for i in assigned if i.priority == 3]
    p3_rt = float(np.mean([i.response_time for i in p3_assigned])) if p3_assigned else 0.0

    return {
        "vehicles": veh_initial,
        "vehicle_assignments": veh_assignments,
        "incidents": incidents_data,
        "timeline": timeline,
        "metrics": {
            "weighted_response_time": round(weighted_rt, 2),
            "p3_response_time": round(p3_rt, 2),
            "coverage_outages": cumulative_outages,
            "assigned_count": len(assigned),
            "total_incidents": len(res.incidents),
        }
    }


def main():
    vehicles, incidents = generate_world(DEFAULT_SEED)
    tm = get_default_traffic_model()

    print("Building high-fidelity continuous playback traces with traffic...")
    naive_trace = build_simulation_trace(NaiveDispatcher(traffic_model=tm), vehicles, incidents, traffic_model=tm)
    cov_trace = build_simulation_trace(CoverageAwareDispatcher(DEFAULT_COVERAGE_PENALTY, traffic_model=tm), vehicles, incidents, traffic_model=tm)

    payload = {
        "seed": DEFAULT_SEED,
        "traffic": tm.to_dict(),
        "bounds": tm.bounds,
        "naive": naive_trace,
        "coverage_aware": cov_trace,
    }

    out_path = Path("outputs/playback_data.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    # Also copy to web/outputs
    web_out = Path("web/outputs/playback_data.json")
    web_out.parent.mkdir(parents=True, exist_ok=True)
    with open(web_out, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"Exported rich playback trace to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
