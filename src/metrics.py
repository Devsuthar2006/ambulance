"""Evaluation metrics for the Emergency Fleet Assignment benchmark."""

from dataclasses import dataclass
import numpy as np
from src.models import Incident
from src.simulator import SimulationResult
from src.config import DEFAULT_SEED


@dataclass
class EvaluationMetrics:
    """Benchmark performance metrics."""
    priority_weighted_response_time: float
    priority3_response_time: float
    coverage_outage_minutes: int
    unweighted_mean_response_time: float
    total_incidents: int
    assigned_incidents: int
    runtime_seconds: float
    seed: int

    def to_summary_dict(self) -> dict[str, float | int]:
        """Format as required by Section 9 of the PRD."""
        return {
            "priority_weighted_response_time": round(float(self.priority_weighted_response_time), 4),
            "priority3_response_time": round(float(self.priority3_response_time), 4),
            "coverage_outage_minutes": int(self.coverage_outage_minutes),
            "runtime_seconds": round(float(self.runtime_seconds), 6),
            "seed": int(self.seed),
        }


def compute_metrics(
    result: SimulationResult,
    seed: int = DEFAULT_SEED,
) -> EvaluationMetrics:
    """Compute score metrics from finished simulation results."""
    assigned_incidents = [inc for inc in result.incidents if inc.reached_minute is not None]
    
    if not assigned_incidents:
        return EvaluationMetrics(
            priority_weighted_response_time=float("inf"),
            priority3_response_time=float("inf"),
            coverage_outage_minutes=result.coverage_outage_minutes,
            unweighted_mean_response_time=float("inf"),
            total_incidents=len(result.incidents),
            assigned_incidents=0,
            runtime_seconds=result.runtime_seconds,
            seed=seed,
        )

    response_times = np.array([inc.response_time for inc in assigned_incidents], dtype=np.float64)
    weights = np.array([inc.weight for inc in assigned_incidents], dtype=np.float64)
    priorities = np.array([inc.priority for inc in assigned_incidents], dtype=np.int64)

    weighted_response = float(np.sum(response_times * weights) / np.sum(weights))
    unweighted_mean = float(np.mean(response_times))

    p3_mask = priorities == 3
    if np.any(p3_mask):
        p3_response = float(np.mean(response_times[p3_mask]))
    else:
        p3_response = 0.0

    return EvaluationMetrics(
        priority_weighted_response_time=weighted_response,
        priority3_response_time=p3_response,
        coverage_outage_minutes=result.coverage_outage_minutes,
        unweighted_mean_response_time=unweighted_mean,
        total_incidents=len(result.incidents),
        assigned_incidents=len(assigned_incidents),
        runtime_seconds=result.runtime_seconds,
        seed=seed,
    )
