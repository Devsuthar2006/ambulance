"""Dispatch policies for assigning idle vehicles to emergency incidents with explainability and traffic awareness."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from src.models import Vehicle, Incident
from src.config import (
    DEFAULT_COVERAGE_PENALTY,
    DEFAULT_SCARCITY_PENALTY,
    TOTAL_VEHICLES,
    VEHICLE_SPEED,
)
from src.traffic.traffic_model import TrafficModel


@dataclass
class CandidateEvaluation:
    """Breakdown of costs evaluated for an available vehicle."""
    vehicle_id: int
    quadrant: int
    travel_time: float
    is_last_in_quadrant: bool
    coverage_penalty_cost: float
    scarcity_penalty_cost: float
    total_cost: float
    raw_distance: float = 0.0
    traffic_multiplier: float = 1.0


class BaseDispatcher(ABC):
    """Abstract base class for online emergency dispatchers."""

    traffic_model: TrafficModel | None = None

    @abstractmethod
    def assign(
        self,
        idle_vehicles: list[Vehicle],
        incident: Incident,
        idle_counts_by_quadrant: np.ndarray,
    ) -> Vehicle | None:
        """Select an idle vehicle for the given incident or return None."""
        pass

    def assign_with_explanation(
        self,
        idle_vehicles: list[Vehicle],
        incident: Incident,
        idle_counts_by_quadrant: np.ndarray,
    ) -> tuple[Vehicle | None, dict]:
        """Assign vehicle and produce a structured mathematical decision explanation."""
        if not idle_vehicles:
            return None, {
                "decision": "QUEUED",
                "reason": "All fleet vehicles are currently busy. Incident queued with priority.",
                "candidates": [],
                "selected_vehicle_id": None,
                "nearest_vehicle_id": None,
                "policy": self.__class__.__name__,
            }

        # Calculate costs and candidates
        chosen = self.assign(idle_vehicles, incident, idle_counts_by_quadrant)
        candidates, nearest_id = self._evaluate_candidates(idle_vehicles, incident, idle_counts_by_quadrant)

        # Build natural-language justification
        nearest_cand = next((c for c in candidates if c.vehicle_id == nearest_id), candidates[0])
        selected_cand = next((c for c in candidates if c.vehicle_id == chosen.id), candidates[0])

        if chosen.id != nearest_id:
            # Check if closer unit was rejected due to traffic gridlock
            if selected_cand.travel_time < nearest_cand.travel_time and nearest_cand.traffic_multiplier > 1.3:
                reason = (
                    f"Vehicle {chosen.id} (ETA {selected_cand.travel_time:.1f}m, clear route) selected over "
                    f"closer Vehicle {nearest_id} ({nearest_cand.raw_distance:.1f} dist, ETA {nearest_cand.travel_time:.1f}m) "
                    f"because Vehicle {nearest_id} is severely delayed by {nearest_cand.traffic_multiplier:.1f}x traffic gridlock. "
                    f"Priority {incident.priority} emergency reached {nearest_cand.travel_time - selected_cand.travel_time:.1f}m faster."
                )
                strategy = "TRAFFIC_GRIDLOCK_BYPASS"
            elif nearest_cand.is_last_in_quadrant:
                reason = (
                    f"Vehicle {chosen.id} selected instead of closer Vehicle {nearest_id} "
                    f"({selected_cand.travel_time:.1f}m vs {nearest_cand.travel_time:.1f}m) "
                    f"because Vehicle {nearest_id} is the last available unit in Q{nearest_cand.quadrant}. "
                    f"Dispatching Vehicle {chosen.id} preserves critical quadrant coverage."
                )
                strategy = "PRESERVE_COVERAGE"
            else:
                reason = (
                    f"Vehicle {chosen.id} selected over Vehicle {nearest_id} to minimize total objective cost "
                    f"({selected_cand.total_cost:.1f} vs {nearest_cand.total_cost:.1f})."
                )
                strategy = "OPTIMAL_RESILIENT"
        else:
            if selected_cand.traffic_multiplier > 1.5:
                reason = (
                    f"Vehicle {chosen.id} selected as nearest available unit despite {selected_cand.traffic_multiplier:.1f}x corridor traffic "
                    f"(ETA {selected_cand.travel_time:.1f}m)."
                )
            else:
                reason = (
                    f"Vehicle {chosen.id} selected as nearest optimal unit ({selected_cand.travel_time:.1f}m) "
                    f"without compromising sector coverage safety."
                )
            strategy = "OPTIMAL_NEAREST"

        explanation = {
            "decision": "DISPATCHED",
            "strategy": strategy,
            "reason": reason,
            "selected_vehicle_id": chosen.id,
            "nearest_vehicle_id": nearest_id,
            "policy": self.__class__.__name__,
            "incident_id": incident.id,
            "priority": incident.priority,
            "weight": incident.weight,
            "candidates": [
                {
                    "vehicle_id": c.vehicle_id,
                    "quadrant": c.quadrant,
                    "raw_distance": round(c.raw_distance, 2),
                    "traffic_multiplier": round(c.traffic_multiplier, 2),
                    "travel_time": round(c.travel_time, 2),
                    "is_last": c.is_last_in_quadrant,
                    "coverage_penalty": round(c.coverage_penalty_cost, 2),
                    "scarcity_penalty": round(c.scarcity_penalty_cost, 2),
                    "total_cost": round(c.total_cost, 2),
                }
                for c in candidates
            ],
        }

        return chosen, explanation

    @abstractmethod
    def _evaluate_candidates(
        self,
        idle_vehicles: list[Vehicle],
        incident: Incident,
        idle_counts_by_quadrant: np.ndarray,
    ) -> tuple[list[CandidateEvaluation], int]:
        """Internal helper to compute detailed costs for all candidates."""
        pass


class NaiveDispatcher(BaseDispatcher):
    """Baseline greedy dispatcher: assigns nearest idle vehicle (traffic-blind)."""

    def __init__(self, traffic_model: TrafficModel | None = None) -> None:
        self.traffic_model = traffic_model

    def assign(
        self,
        idle_vehicles: list[Vehicle],
        incident: Incident,
        idle_counts_by_quadrant: np.ndarray,
    ) -> Vehicle | None:
        if not idle_vehicles:
            return None

        coords = np.array([[v.x, v.y] for v in idle_vehicles], dtype=np.float64)
        dists = np.hypot(coords[:, 0] - incident.x, coords[:, 1] - incident.y)
        vehicle_ids = np.array([v.id for v in idle_vehicles], dtype=np.int64)

        # Pure greedy distance sort (blind to traffic)
        order = np.lexsort((vehicle_ids, dists))
        best_idx = int(order[0])
        return idle_vehicles[best_idx]

    def _evaluate_candidates(
        self,
        idle_vehicles: list[Vehicle],
        incident: Incident,
        idle_counts_by_quadrant: np.ndarray,
    ) -> tuple[list[CandidateEvaluation], int]:
        coords = np.array([[v.x, v.y] for v in idle_vehicles], dtype=np.float64)
        dists = np.hypot(coords[:, 0] - incident.x, coords[:, 1] - incident.y)
        vehicle_ids = np.array([v.id for v in idle_vehicles], dtype=np.int64)

        nearest_idx = int(np.lexsort((vehicle_ids, dists))[0])
        nearest_id = idle_vehicles[nearest_idx].id

        evals = []
        for i, v in enumerate(idle_vehicles):
            is_last = bool(idle_counts_by_quadrant[v.quadrant] == 1)
            raw_dist = float(dists[i])
            if self.traffic_model:
                mult = self.traffic_model.get_traffic_multiplier(v.x, v.y, incident.x, incident.y)
                travel_time = (raw_dist / VEHICLE_SPEED) * mult
            else:
                mult = 1.0
                travel_time = raw_dist / VEHICLE_SPEED

            evals.append(
                CandidateEvaluation(
                    vehicle_id=v.id,
                    quadrant=v.quadrant,
                    raw_distance=raw_dist,
                    traffic_multiplier=float(mult),
                    travel_time=float(travel_time),
                    is_last_in_quadrant=is_last,
                    coverage_penalty_cost=0.0,
                    scarcity_penalty_cost=0.0,
                    total_cost=raw_dist,  # Naive policy optimizes raw distance
                )
            )
        evals.sort(key=lambda c: c.total_cost)
        return evals, nearest_id


class CoverageAwareDispatcher(BaseDispatcher):
    """Coverage-aware dispatcher with quadrant preservation penalty and traffic-aware travel time."""

    def __init__(
        self,
        coverage_penalty: float = DEFAULT_COVERAGE_PENALTY,
        traffic_model: TrafficModel | None = None,
    ) -> None:
        self.coverage_penalty = float(coverage_penalty)
        self.traffic_model = traffic_model

    def assign(
        self,
        idle_vehicles: list[Vehicle],
        incident: Incident,
        idle_counts_by_quadrant: np.ndarray,
    ) -> Vehicle | None:
        if not idle_vehicles:
            return None

        idle_counts = np.asarray(idle_counts_by_quadrant, dtype=np.int64)
        coords = np.array([[v.x, v.y] for v in idle_vehicles], dtype=np.float64)
        raw_dists = np.hypot(coords[:, 0] - incident.x, coords[:, 1] - incident.y)
        quadrants = np.array([v.quadrant for v in idle_vehicles], dtype=np.int64)
        vehicle_ids = np.array([v.id for v in idle_vehicles], dtype=np.int64)

        # Compute travel times factoring in traffic
        travel_times = np.zeros(len(idle_vehicles), dtype=np.float64)
        for i, v in enumerate(idle_vehicles):
            if self.traffic_model:
                travel_times[i] = self.traffic_model.get_travel_time(v.x, v.y, incident.x, incident.y)
            else:
                travel_times[i] = raw_dists[i] / VEHICLE_SPEED

        is_last = (idle_counts[quadrants] == 1).astype(np.float64)
        costs = travel_times + (self.coverage_penalty * is_last) / float(incident.weight)

        order = np.lexsort((vehicle_ids, costs))
        best_idx = int(order[0])
        return idle_vehicles[best_idx]

    def _evaluate_candidates(
        self,
        idle_vehicles: list[Vehicle],
        incident: Incident,
        idle_counts_by_quadrant: np.ndarray,
    ) -> tuple[list[CandidateEvaluation], int]:
        idle_counts = np.asarray(idle_counts_by_quadrant, dtype=np.int64)
        coords = np.array([[v.x, v.y] for v in idle_vehicles], dtype=np.float64)
        raw_dists = np.hypot(coords[:, 0] - incident.x, coords[:, 1] - incident.y)
        quadrants = np.array([v.quadrant for v in idle_vehicles], dtype=np.int64)
        vehicle_ids = np.array([v.id for v in idle_vehicles], dtype=np.int64)

        nearest_idx = int(np.lexsort((vehicle_ids, raw_dists))[0])
        nearest_id = idle_vehicles[nearest_idx].id

        is_last = (idle_counts[quadrants] == 1).astype(np.float64)
        cov_costs = (self.coverage_penalty * is_last) / float(incident.weight)

        evals = []
        for i, v in enumerate(idle_vehicles):
            raw_d = float(raw_dists[i])
            if self.traffic_model:
                mult = self.traffic_model.get_traffic_multiplier(v.x, v.y, incident.x, incident.y)
                travel_time = (raw_d / VEHICLE_SPEED) * mult
            else:
                mult = 1.0
                travel_time = raw_d / VEHICLE_SPEED

            total_cost = travel_time + float(cov_costs[i])

            evals.append(
                CandidateEvaluation(
                    vehicle_id=v.id,
                    quadrant=v.quadrant,
                    raw_distance=raw_d,
                    traffic_multiplier=float(mult),
                    travel_time=float(travel_time),
                    is_last_in_quadrant=bool(is_last[i] == 1.0),
                    coverage_penalty_cost=float(cov_costs[i]),
                    scarcity_penalty_cost=0.0,
                    total_cost=float(total_cost),
                )
            )
        evals.sort(key=lambda c: c.total_cost)
        return evals, nearest_id


class AdaptiveDispatcher(BaseDispatcher):
    """Advanced policy factoring in traffic travel time, coverage penalty, and fleet scarcity pressure."""

    def __init__(
        self,
        coverage_penalty: float = DEFAULT_COVERAGE_PENALTY,
        scarcity_penalty: float = DEFAULT_SCARCITY_PENALTY,
        traffic_model: TrafficModel | None = None,
    ) -> None:
        self.coverage_penalty = float(coverage_penalty)
        self.scarcity_penalty = float(scarcity_penalty)
        self.traffic_model = traffic_model

    def assign(
        self,
        idle_vehicles: list[Vehicle],
        incident: Incident,
        idle_counts_by_quadrant: np.ndarray,
    ) -> Vehicle | None:
        if not idle_vehicles:
            return None

        idle_counts = np.asarray(idle_counts_by_quadrant, dtype=np.int64)
        coords = np.array([[v.x, v.y] for v in idle_vehicles], dtype=np.float64)
        raw_dists = np.hypot(coords[:, 0] - incident.x, coords[:, 1] - incident.y)
        quadrants = np.array([v.quadrant for v in idle_vehicles], dtype=np.int64)
        vehicle_ids = np.array([v.id for v in idle_vehicles], dtype=np.int64)

        # 1. Travel time factoring in traffic
        travel_times = np.zeros(len(idle_vehicles), dtype=np.float64)
        for i, v in enumerate(idle_vehicles):
            if self.traffic_model:
                travel_times[i] = self.traffic_model.get_travel_time(v.x, v.y, incident.x, incident.y)
            else:
                travel_times[i] = raw_dists[i] / VEHICLE_SPEED

        # 2. Coverage preservation component
        is_last = (idle_counts[quadrants] == 1).astype(np.float64)
        cov_cost = (self.coverage_penalty * is_last) / float(incident.weight)

        # 3. Scarcity pressure component
        total_idle = np.sum(idle_counts)
        scarcity_factor = max(0.0, float((TOTAL_VEHICLES / 2 - total_idle) / (TOTAL_VEHICLES / 2)))
        scarcity_cost = (self.scarcity_penalty * scarcity_factor) / float(incident.weight)

        incident_quad = (1 if incident.x >= 50.0 else 0) + 2 * (1 if incident.y >= 50.0 else 0)
        cross_quadrant = (quadrants != incident_quad).astype(np.float64)
        scarcity_friction = cross_quadrant * scarcity_cost

        total_costs = travel_times + cov_cost + scarcity_friction

        order = np.lexsort((vehicle_ids, total_costs))
        best_idx = int(order[0])
        return idle_vehicles[best_idx]

    def _evaluate_candidates(
        self,
        idle_vehicles: list[Vehicle],
        incident: Incident,
        idle_counts_by_quadrant: np.ndarray,
    ) -> tuple[list[CandidateEvaluation], int]:
        idle_counts = np.asarray(idle_counts_by_quadrant, dtype=np.int64)
        coords = np.array([[v.x, v.y] for v in idle_vehicles], dtype=np.float64)
        raw_dists = np.hypot(coords[:, 0] - incident.x, coords[:, 1] - incident.y)
        quadrants = np.array([v.quadrant for v in idle_vehicles], dtype=np.int64)
        vehicle_ids = np.array([v.id for v in idle_vehicles], dtype=np.int64)

        nearest_idx = int(np.lexsort((vehicle_ids, raw_dists))[0])
        nearest_id = idle_vehicles[nearest_idx].id

        is_last = (idle_counts[quadrants] == 1).astype(np.float64)
        cov_cost = (self.coverage_penalty * is_last) / float(incident.weight)

        total_idle = np.sum(idle_counts)
        scarcity_factor = max(0.0, float((TOTAL_VEHICLES / 2 - total_idle) / (TOTAL_VEHICLES / 2)))
        scarcity_cost = (self.scarcity_penalty * scarcity_factor) / float(incident.weight)

        incident_quad = (1 if incident.x >= 50.0 else 0) + 2 * (1 if incident.y >= 50.0 else 0)
        cross_quadrant = (quadrants != incident_quad).astype(np.float64)
        scarcity_friction = cross_quadrant * scarcity_cost

        evals = []
        for i, v in enumerate(idle_vehicles):
            raw_d = float(raw_dists[i])
            if self.traffic_model:
                mult = self.traffic_model.get_traffic_multiplier(v.x, v.y, incident.x, incident.y)
                travel_time = (raw_d / VEHICLE_SPEED) * mult
            else:
                mult = 1.0
                travel_time = raw_d / VEHICLE_SPEED

            total_cost = travel_time + float(cov_cost[i]) + float(scarcity_friction[i])

            evals.append(
                CandidateEvaluation(
                    vehicle_id=v.id,
                    quadrant=v.quadrant,
                    raw_distance=raw_d,
                    traffic_multiplier=float(mult),
                    travel_time=float(travel_time),
                    is_last_in_quadrant=bool(is_last[i] == 1.0),
                    coverage_penalty_cost=float(cov_cost[i]),
                    scarcity_penalty_cost=float(scarcity_friction[i]),
                    total_cost=float(total_cost),
                )
            )
        evals.sort(key=lambda c: c.total_cost)
        return evals, nearest_id


def get_dispatcher(
    policy_name: str,
    coverage_penalty: float = DEFAULT_COVERAGE_PENALTY,
    scarcity_penalty: float = DEFAULT_SCARCITY_PENALTY,
    traffic_model: TrafficModel | None = None,
) -> BaseDispatcher:
    """Factory for selecting dispatch policy with optional traffic awareness."""
    policy = policy_name.lower().strip()
    if policy in ("nearest", "naive"):
        return NaiveDispatcher(traffic_model=traffic_model)
    elif policy in ("coverage", "coverage_aware"):
        return CoverageAwareDispatcher(coverage_penalty=coverage_penalty, traffic_model=traffic_model)
    elif policy == "adaptive":
        return AdaptiveDispatcher(
            coverage_penalty=coverage_penalty,
            scarcity_penalty=scarcity_penalty,
            traffic_model=traffic_model,
        )
    else:
        raise ValueError(f"Unknown policy '{policy_name}'. Choose from: nearest, coverage, adaptive")
