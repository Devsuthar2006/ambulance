"""Tests for metrics calculation against analytical examples."""

import pytest
from src.models import Incident
from src.simulator import SimulationResult
from src.metrics import compute_metrics


def test_metrics_analytical_calculation():
    """Verify priority-weighted response time, priority-3 time, and unweighted time."""
    # Create 3 resolved incidents
    inc1 = Incident(id=1, arrival_minute=0, x=0, y=0, priority=1)
    inc1.reached_minute = 10.0  # response = 10.0, weight = 1.0

    inc2 = Incident(id=2, arrival_minute=2, x=0, y=0, priority=2)
    inc2.reached_minute = 7.0   # response = 5.0, weight = 3.0

    inc3 = Incident(id=3, arrival_minute=5, x=0, y=0, priority=3)
    inc3.reached_minute = 7.0   # response = 2.0, weight = 7.0

    mock_result = SimulationResult(
        incidents=[inc1, inc2, inc3],
        vehicles=[],
        coverage_outage_minutes=14,
        coverage_history=[],
        runtime_seconds=0.0123,
        total_minutes=121,
    )

    metrics = compute_metrics(mock_result, seed=20260911)

    # Expected:
    # weighted = (1*10 + 3*5 + 7*2) / (1 + 3 + 7) = 39 / 11 = 3.54545...
    # p3 = 2.0
    # unweighted = (10 + 5 + 2) / 3 = 17 / 3 = 5.6666...
    assert pytest.approx(metrics.priority_weighted_response_time, rel=1e-4) == 39.0 / 11.0
    assert pytest.approx(metrics.priority3_response_time, rel=1e-4) == 2.0
    assert pytest.approx(metrics.unweighted_mean_response_time, rel=1e-4) == 17.0 / 3.0
    assert metrics.coverage_outage_minutes == 14

    summary = metrics.to_summary_dict()
    assert "priority_weighted_response_time" in summary
    assert "priority3_response_time" in summary
    assert "coverage_outage_minutes" in summary
    assert "runtime_seconds" in summary
    assert summary["seed"] == 20260911
