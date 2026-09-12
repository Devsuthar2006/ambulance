"""Causal discrete-event simulator for emergency fleet dispatch."""

import copy
import heapq
import time
from dataclasses import dataclass
import numpy as np

from src.config import (
    SIM_START_MINUTE,
    SIM_END_MINUTE,
    SERVICE_TIME,
    VEHICLE_SPEED,
    NUM_QUADRANTS,
)
from src.models import Vehicle, Incident
from src.dispatcher import BaseDispatcher


@dataclass
class SimulationResult:
    """Results collected from a full simulation run."""
    incidents: list[Incident]
    vehicles: list[Vehicle]
    coverage_outage_minutes: int
    coverage_history: list[tuple[int, list[int]]]
    runtime_seconds: float
    total_minutes: int


class Simulator:
    """Online, causal emergency dispatch simulation engine."""

    def __init__(
        self,
        dispatcher: BaseDispatcher,
        start_minute: int = SIM_START_MINUTE,
        end_minute: int = SIM_END_MINUTE,
    ) -> None:
        self.dispatcher = dispatcher
        self.start_minute = start_minute
        self.end_minute = end_minute

    def run(
        self,
        initial_vehicles: list[Vehicle],
        all_incidents: list[Incident],
    ) -> SimulationResult:
        """Run the simulation from start_minute to end_minute under strict causality."""
        start_time = time.perf_counter()

        # Deep-copy inputs to avoid state leakage across runs
        vehicles = copy.deepcopy(initial_vehicles)
        incidents = copy.deepcopy(all_incidents)

        # Index incidents by arrival minute for O(1) causal release
        incidents_by_arrival: dict[int, list[Incident]] = {}
        for inc in incidents:
            incidents_by_arrival.setdefault(inc.arrival_minute, []).append(inc)
        for arr_min in incidents_by_arrival:
            incidents_by_arrival[arr_min].sort(key=lambda item: item.id)

        # Fleet map and O(1) idle quadrant counts
        fleet: dict[int, Vehicle] = {v.id: v for v in vehicles}
        idle_counts = np.zeros(NUM_QUADRANTS, dtype=np.int64)
        for v in vehicles:
            if v.idle:
                idle_counts[v.quadrant] += 1

        # Waiting queue: min-heap keyed by (-priority, arrival_minute, incident_id)
        # Stored element: (-priority, arrival_minute, incident_id, incident)
        waiting_queue: list[tuple[int, int, int, Incident]] = []

        # Vehicle completion events: min-heap keyed by (completion_time, vehicle_id, end_x, end_y)
        completion_heap: list[tuple[float, int, float, float]] = []

        coverage_outage_minutes = 0
        coverage_history: list[tuple[int, list[int]]] = []

        # Discrete minute clock: t = 0 ... 120
        for t in range(self.start_minute, self.end_minute + 1):

            # --- Step 1: Free vehicles whose completion_time <= t ---
            while completion_heap and completion_heap[0][0] <= t:
                comp_time, v_id, end_x, end_y = heapq.heappop(completion_heap)
                v = fleet[v_id]
                v.idle = True
                v.update_position(end_x, end_y)
                idle_counts[v.quadrant] += 1

            # --- Step 2: Reveal incidents with arrival_minute == t in ascending incident ID ---
            if t in incidents_by_arrival:
                for inc in incidents_by_arrival[t]:
                    heapq.heappush(
                        waiting_queue,
                        (-inc.priority, inc.arrival_minute, inc.id, inc),
                    )

            # --- Step 3 & 4: Process waiting queue in required order ---
            # Order: priority descending -> arrival ascending -> incident ID ascending
            unassigned: list[tuple[int, int, int, Incident]] = []
            while waiting_queue:
                # If no vehicles are idle, no assignments can be made this minute
                idle_vehicles = [v for v in vehicles if v.idle]
                if not idle_vehicles:
                    # All remaining queued incidents remain waiting
                    unassigned.extend(waiting_queue)
                    waiting_queue.clear()
                    break

                entry = heapq.heappop(waiting_queue)
                _, _, _, inc = entry

                chosen_vehicle = self.dispatcher.assign(
                    idle_vehicles=idle_vehicles,
                    incident=inc,
                    idle_counts_by_quadrant=idle_counts,
                )

                if chosen_vehicle is not None:
                    # Assign vehicle
                    v = fleet[chosen_vehicle.id]
                    dist = float(np.hypot(v.x - inc.x, v.y - inc.y))
                    travel_time = dist / VEHICLE_SPEED
                    reached_minute = t + travel_time
                    completion_minute = reached_minute + SERVICE_TIME

                    # Update incident records
                    inc.assigned_vehicle_id = v.id
                    inc.assigned_minute = float(t)
                    inc.reached_minute = reached_minute
                    inc.completed_minute = completion_minute

                    # Update vehicle state
                    v.idle = False
                    v.busy_until = completion_minute
                    v.assigned_count += 1
                    idle_counts[v.quadrant] -= 1

                    # Register completion event
                    heapq.heappush(
                        completion_heap,
                        (completion_minute, v.id, inc.x, inc.y),
                    )
                else:
                    unassigned.append(entry)

            # Restore unassigned incidents back to priority queue
            for entry in unassigned:
                heapq.heappush(waiting_queue, entry)

            # --- Step 5: Evaluate coverage ---
            # If ANY quadrant has zero idle vehicles this minute, it counts as one outage minute
            has_outage = bool(np.any(idle_counts == 0))
            if has_outage:
                coverage_outage_minutes += 1

            coverage_history.append((t, idle_counts.tolist()))

        runtime_seconds = time.perf_counter() - start_time

        return SimulationResult(
            incidents=incidents,
            vehicles=vehicles,
            coverage_outage_minutes=coverage_outage_minutes,
            coverage_history=coverage_history,
            runtime_seconds=runtime_seconds,
            total_minutes=self.end_minute - self.start_minute + 1,
        )
