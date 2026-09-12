"""Tests for dispatch policy logic, micro-scenarios, and coverage preservation."""

import numpy as np
from src.models import Vehicle, Incident
from src.dispatcher import NaiveDispatcher, CoverageAwareDispatcher


def test_naive_dispatcher_selects_closest():
    """Naive dispatcher strictly selects the vehicle with minimum Euclidean distance."""
    dispatcher = NaiveDispatcher()
    
    # Vehicles at (10, 10) and (20, 20)
    v1 = Vehicle(id=1, x=10.0, y=10.0)
    v2 = Vehicle(id=2, x=20.0, y=20.0)
    idle_counts = np.array([2, 0, 0, 0], dtype=np.int64)

    # Incident at (11, 11) - closer to v1
    inc = Incident(id=100, arrival_minute=0, x=11.0, y=11.0, priority=1)
    chosen = dispatcher.assign([v1, v2], inc, idle_counts)
    assert chosen is not None
    assert chosen.id == v1.id

    # Incident at (19, 19) - closer to v2
    inc2 = Incident(id=101, arrival_minute=0, x=19.0, y=19.0, priority=1)
    chosen2 = dispatcher.assign([v1, v2], inc2, idle_counts)
    assert chosen2 is not None
    assert chosen2.id == v2.id


def test_coverage_protection_deflects_routine_call():
    """A routine Priority 1 call deflects away from the sole idle vehicle in a quadrant."""
    dispatcher = CoverageAwareDispatcher(coverage_penalty=20.0)

    # v0 is at (10, 10) in Q0, and is the SOLE idle vehicle in Q0
    v0 = Vehicle(id=0, x=10.0, y=10.0)
    # v1 is at (15, 10) in Q1, where Q1 has 3 idle vehicles
    v1 = Vehicle(id=1, x=60.0, y=10.0)  # In Q1 (x >= 50)

    idle_counts = np.array([1, 3, 0, 0], dtype=np.int64)

    # Incident at (12, 10) with Priority 1 (weight = 1.0)
    # Dist to v0 = 2.0. Cost v0 = 2.0 + 20.0 / 1.0 = 22.0
    # Dist to v1 = 48.0. Cost v1 = 48.0 + 0 = 48.0
    # Let's put v1 at (52.0, 10.0) so dist to v1 = 40.0, still > 22?
    # Let's put v1 at (25.0, 10.0) wait, x < 50 is Q0.
    # Suppose v0 is in Q0 at (48, 25). v1 is in Q1 at (52, 25).
    v0 = Vehicle(id=0, x=48.0, y=25.0)  # Q0
    v1 = Vehicle(id=1, x=53.0, y=25.0)  # Q1
    idle_counts = np.array([1, 3, 0, 0], dtype=np.int64)

    # Incident at (47.0, 25.0) Priority 1
    # Dist to v0 = 1.0. Cost v0 = 1.0 + 20 / 1.0 = 21.0
    # Dist to v1 = 6.0. Cost v1 = 6.0 + 0 = 6.0
    inc_p1 = Incident(id=1, arrival_minute=0, x=47.0, y=25.0, priority=1)
    
    chosen = dispatcher.assign([v0, v1], inc_p1, idle_counts)
    assert chosen is not None
    # Coverage policy chooses v1 to protect Q0 from emptying
    assert chosen.id == v1.id

    # Naive policy would choose v0 because 1.0 < 6.0
    naive = NaiveDispatcher()
    chosen_naive = naive.assign([v0, v1], inc_p1, idle_counts)
    assert chosen_naive is not None
    assert chosen_naive.id == v0.id


def test_priority3_overrides_coverage_penalty():
    """A critical Priority 3 call softens the penalty (divided by 7) so closest vehicle responds."""
    dispatcher = CoverageAwareDispatcher(coverage_penalty=20.0)

    v0 = Vehicle(id=0, x=48.0, y=25.0)  # Sole idle in Q0
    v1 = Vehicle(id=1, x=53.0, y=25.0)  # Q1 has 3 idle
    idle_counts = np.array([1, 3, 0, 0], dtype=np.int64)

    # Incident at (47.0, 25.0) Priority 3 (weight = 7.0)
    # Dist to v0 = 1.0. Cost v0 = 1.0 + 20 / 7.0 = 1.0 + 2.857 = 3.857
    # Dist to v1 = 6.0. Cost v1 = 6.0 + 0 = 6.0
    inc_p3 = Incident(id=2, arrival_minute=0, x=47.0, y=25.0, priority=3)

    chosen = dispatcher.assign([v0, v1], inc_p3, idle_counts)
    assert chosen is not None
    # For critical P3, 3.857 < 6.0, so v0 is assigned immediately!
    assert chosen.id == v0.id


def test_coverage_regression_scenario():
    """Engineered scenario where naive policy empties a quadrant while coverage-aware preserves it."""
    from src.simulator import Simulator

    # 4 quadrants, all have >= 1 idle vehicle initially
    # Q0 has exactly 1 idle vehicle at (48, 25)
    # Q1 has 3 idle vehicles at (52, 25), (55, 25), (60, 25)
    # Q2 and Q3 have 1 idle vehicle each
    fleet = [
        Vehicle(id=0, x=48.0, y=25.0),  # Q0 (sole idle)
        Vehicle(id=1, x=52.0, y=25.0),  # Q1
        Vehicle(id=2, x=55.0, y=25.0),  # Q1
        Vehicle(id=3, x=60.0, y=25.0),  # Q1
        Vehicle(id=4, x=10.0, y=75.0),  # Q2
        Vehicle(id=5, x=75.0, y=75.0),  # Q3
    ]

    # Routine Priority 1 incident arrives at (47, 25) in Q0 at t=0
    incidents = [
        Incident(id=101, arrival_minute=0, x=47.0, y=25.0, priority=1)
    ]

    # Run Naive dispatcher
    sim_naive = Simulator(dispatcher=NaiveDispatcher(), start_minute=0, end_minute=0)
    res_naive = sim_naive.run(fleet, incidents)

    # Naive sends vehicle 0 (dist=1.0), leaving Q0 with 0 idle vehicles -> Outage = 1
    assert res_naive.incidents[0].assigned_vehicle_id == 0
    assert res_naive.coverage_outage_minutes == 1

    # Run CoverageAware dispatcher
    sim_cov = Simulator(dispatcher=CoverageAwareDispatcher(coverage_penalty=20.0), start_minute=0, end_minute=0)
    res_cov = sim_cov.run(fleet, incidents)

    # CoverageAware sends vehicle 1 from Q1 (dist=5.0), keeping vehicle 0 in Q0 -> Outage = 0
    assert res_cov.incidents[0].assigned_vehicle_id == 1
    assert res_cov.coverage_outage_minutes == 0

