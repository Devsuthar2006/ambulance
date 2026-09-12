"""Tests for AdaptiveDispatcher and explainable dispatch decisions."""

import numpy as np
from src.models import Vehicle, Incident
from src.dispatcher import AdaptiveDispatcher, CoverageAwareDispatcher


def test_assign_with_explanation_structure():
    """assign_with_explanation returns structured candidates and transparent justification."""
    dispatcher = CoverageAwareDispatcher(coverage_penalty=20.0)

    # v0 in Q0 (sole idle), v1 in Q1 (surplus)
    v0 = Vehicle(id=0, x=48.0, y=25.0)
    v1 = Vehicle(id=1, x=53.0, y=25.0)
    idle_counts = np.array([1, 3, 0, 0], dtype=int)

    inc = Incident(id=101, arrival_minute=0, x=47.0, y=25.0, priority=1)

    chosen, explanation = dispatcher.assign_with_explanation([v0, v1], inc, idle_counts)

    assert chosen is not None
    assert chosen.id == v1.id
    assert explanation["decision"] == "DISPATCHED"
    assert explanation["strategy"] == "PRESERVE_COVERAGE"
    assert explanation["selected_vehicle_id"] == 1
    assert explanation["nearest_vehicle_id"] == 0
    assert "preserves critical quadrant coverage" in explanation["reason"]
    assert len(explanation["candidates"]) == 2


def test_adaptive_scarcity_penalty():
    """Adaptive dispatcher applies scarcity friction when fleet reserves are severely depleted."""
    # Low fleet reserves: only 2 idle vehicles in the entire region
    dispatcher = AdaptiveDispatcher(coverage_penalty=20.0, scarcity_penalty=15.0)

    v0 = Vehicle(id=0, x=10.0, y=10.0)  # In Q0
    v1 = Vehicle(id=1, x=80.0, y=80.0)  # In Q3

    # Total idle = 2 (out of 20) -> high scarcity
    idle_counts = np.array([1, 0, 0, 1], dtype=int)

    # Incident in Q0
    inc_q0 = Incident(id=201, arrival_minute=5, x=12.0, y=10.0, priority=1)

    evals, nearest_id = dispatcher._evaluate_candidates([v0, v1], inc_q0, idle_counts)

    # v1 is cross-quadrant and pays extra scarcity friction
    v1_eval = next(c for c in evals if c.vehicle_id == 1)
    assert v1_eval.scarcity_penalty_cost > 0.0
