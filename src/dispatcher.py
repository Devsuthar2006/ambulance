"""Dispatch policies for assigning idle vehicles to emergency incidents with explainability."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from src.models import Vehicle, Incident
from src.config import DEFAULT_COVERAGE_PENALTY, DEFAULT_SCARCITY_PENALTY, TOTAL_VEHICLES


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


class BaseDispatcher(ABC):
    """Abstract base class for online emergency dispatchers."""

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
            reason = (
                f"Vehicle {chosen.id} selected instead of closer Vehicle {nearest_id} "
                f"({selected_cand.travel_time:.1f}m vs {nearest_cand.travel_time:.1f}m) "
                f"because Vehicle {nearest_id} is the last available unit in Q{nearest_cand.quadrant}. "
                f"Dispatching Vehicle {chosen.id} preserves critical quadrant coverage."
            )
            strategy = "PRESERVE_COVERAGE"
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
    """Baseline greedy dispatcher: assigns the nearest idle vehicle."""

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
            evals.append(
                CandidateEvaluation(
                    vehicle_id=v.id,
                    quadrant=v.quadrant,
                    travel_time=float(dists[i]),
                    is_last_in_quadrant=is_last,
                    coverage_penalty_cost=0.0,
                    scarcity_penalty_cost=0.0,
                    total_cost=float(dists[i]),
                )
            )
        evals.sort(key=lambda c: c.total_cost)
        return evals, nearest_id


class CoverageAwareDispatcher(BaseDispatcher):
    """Original Coverage-aware dispatcher with quadrant preservation penalty."""

    def __init__(self, coverage_penalty: float = DEFAULT_COVERAGE_PENALTY) -> None:
        self.coverage_penalty = float(coverage_penalty)

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
        dists = np.hypot(coords[:, 0] - incident.x, coords[:, 1] - incident.y)
        quadrants = np.array([v.quadrant for v in idle_vehicles], dtype=np.int64)
        vehicle_ids = np.array([v.id for v in idle_vehicles], dtype=np.int64)

        is_last = (idle_counts[quadrants] == 1).astype(np.float64)
        costs = dists + (self.coverage_penalty * is_last) / float(incident.weight)

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
        dists = np.hypot(coords[:, 0] - incident.x, coords[:, 1] - incident.y)
        quadrants = np.array([v.quadrant for v in idle_vehicles], dtype=np.int64)
        vehicle_ids = np.array([v.id for v in idle_vehicles], dtype=np.int64)

        nearest_idx = int(np.lexsort((vehicle_ids, dists))[0])
        nearest_id = idle_vehicles[nearest_idx].id

        is_last = (idle_counts[quadrants] == 1).astype(np.float64)
        cov_costs = (self.coverage_penalty * is_last) / float(incident.weight)
        total_costs = dists + cov_costs

        evals = []
        for i, v in enumerate(idle_vehicles):
            evals.append(
                CandidateEvaluation(
                    vehicle_id=v.id,
                    quadrant=v.quadrant,
                    travel_time=float(dists[i]),
                    is_last_in_quadrant=bool(is_last[i] == 1.0),
                    coverage_penalty_cost=float(cov_costs[i]),
                    scarcity_penalty_cost=0.0,
                    total_cost=float(total_costs[i]),
                )
            )
        evals.sort(key=lambda c: c.total_cost)
        return evals, nearest_id


class AdaptiveDispatcher(BaseDispatcher):
    """Advanced policy factoring in travel time, coverage penalty, and fleet scarcity pressure."""

    def __init__(
        self,
        coverage_penalty: float = DEFAULT_COVERAGE_PENALTY,
        scarcity_penalty: float = DEFAULT_SCARCITY_PENALTY,
    ) -> None:
        self.coverage_penalty = float(coverage_penalty)
        self.scarcity_penalty = float(scarcity_penalty)

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
        dists = np.hypot(coords[:, 0] - incident.x, coords[:, 1] - incident.y)
        quadrants = np.array([v.quadrant for v in idle_vehicles], dtype=np.int64)
        vehicle_ids = np.array([v.id for v in idle_vehicles], dtype=np.int64)

        # 1. Coverage preservation component
        is_last = (idle_counts[quadrants] == 1).astype(np.float64)
        cov_cost = (self.coverage_penalty * is_last) / float(incident.weight)

        # 2. Scarcity pressure component
        total_idle = np.sum(idle_counts)
        scarcity_factor = max(0.0, float((TOTAL_VEHICLES / 2 - total_idle) / (TOTAL_VEHICLES / 2)))
        scarcity_cost = (self.scarcity_penalty * scarcity_factor) / float(incident.weight)

        # Cross-quadrant dispatch extra friction under scarcity
        incident_quad = (1 if incident.x >= 50.0 else 0) + 2 * (1 if incident.y >= 50.0 else 0)
        cross_quadrant = (quadrants != incident_quad).astype(np.float64)
        scarcity_friction = cross_quadrant * scarcity_cost

        total_costs = dists + cov_cost + scarcity_friction

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
        dists = np.hypot(coords[:, 0] - incident.x, coords[:, 1] - incident.y)
        quadrants = np.array([v.quadrant for v in idle_vehicles], dtype=np.int64)
        vehicle_ids = np.array([v.id for v in idle_vehicles], dtype=np.int64)

        nearest_idx = int(np.lexsort((vehicle_ids, dists))[0])
        nearest_id = idle_vehicles[nearest_idx].id

        is_last = (idle_counts[quadrants] == 1).astype(np.float64)
        cov_cost = (self.coverage_penalty * is_last) / float(incident.weight)

        total_idle = np.sum(idle_counts)
        scarcity_factor = max(0.0, float((TOTAL_VEHICLES / 2 - total_idle) / (TOTAL_VEHICLES / 2)))
        scarcity_cost = (self.scarcity_penalty * scarcity_factor) / float(incident.weight)

        incident_quad = (1 if incident.x >= 50.0 else 0) + 2 * (1 if incident.y >= 50.0 else 0)
        cross_quadrant = (quadrants != incident_quad).astype(np.float64)
        scarcity_friction = cross_quadrant * scarcity_cost

        total_costs = dists + cov_cost + scarcity_friction


        evals = []
        for i, v in enumerate(idle_vehicles):
            evals.append(
                CandidateEvaluation(
                    vehicle_id=v.id,
                    quadrant=v.quadrant,
                    travel_time=float(dists[i]),
                    is_last_in_quadrant=bool(is_last[i] == 1.0),
                    coverage_penalty_cost=float(cov_cost[i]),
                    scarcity_penalty_cost=float(scarcity_friction[i]),
                    total_cost=float(total_costs[i]),
                )
            )
        evals.sort(key=lambda c: c.total_cost)
        return evals, nearest_id


def get_dispatcher(policy_name: str, coverage_penalty: float = DEFAULT_COVERAGE_PENALTY, scarcity_penalty: float = DEFAULT_SCARCITY_PENALTY) -> BaseDispatcher:
    """Factory for selecting dispatch policy."""
    policy = policy_name.lower().strip()
    if policy == "nearest":
        return NaiveDispatcher()
    elif policy in ("coverage", "coverage_aware"):
        return CoverageAwareDispatcher(coverage_penalty=coverage_penalty)
    elif policy == "adaptive":
        return AdaptiveDispatcher(coverage_penalty=coverage_penalty, scarcity_penalty=scarcity_penalty)
    else:
        raise ValueError(f"Unknown policy '{policy_name}'. Choose from: nearest, coverage, adaptive")
