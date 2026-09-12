"""Tests for simulator engine causality, queue ordering, and event handling."""

from src.models import Vehicle, Incident
from src.simulator import Simulator
from src.dispatcher import NaiveDispatcher


def test_queue_ordering_priority_arrival_id():
    """Simulator processes incidents strictly by: priority desc -> arrival asc -> id asc."""
    # Place one vehicle far away so it can only serve 1 incident per minute
    v = Vehicle(id=0, x=0.0, y=0.0)
    
    # 4 incidents arriving at t=0
    # inc_p1: priority 1, id 10
    # inc_p2_early: priority 2, id 20, arrival 0
    # inc_p2_late: priority 2, id 21, arrival 1
    # inc_p3: priority 3, id 30, arrival 1
    inc_p1 = Incident(id=10, arrival_minute=0, x=5.0, y=5.0, priority=1)
    inc_p2_early = Incident(id=20, arrival_minute=0, x=5.0, y=5.0, priority=2)
    inc_p2_late = Incident(id=21, arrival_minute=1, x=5.0, y=5.0, priority=2)
    inc_p3 = Incident(id=30, arrival_minute=1, x=5.0, y=5.0, priority=3)

    # At minute 0: v serves inc_p2_early (since p2 > p1)
    # At minute 1: inc_p3 and inc_p2_late arrive.
    # Suppose we simulate with 1 vehicle at t=0:
    sim = Simulator(dispatcher=NaiveDispatcher(), start_minute=0, end_minute=0)
    res = sim.run([v], [inc_p1, inc_p2_early])
    
    assigned_ids = [inc.id for inc in res.incidents if inc.assigned_vehicle_id is not None]
    assert assigned_ids == [20]  # Priority 2 served before Priority 1


def test_strict_causality_no_future_assignments():
    """Incidents cannot be assigned before their arrival minute."""
    v = Vehicle(id=0, x=0.0, y=0.0)
    future_inc = Incident(id=1, arrival_minute=10, x=0.0, y=0.0, priority=3)

    # Simulate up to minute 9
    sim = Simulator(dispatcher=NaiveDispatcher(), start_minute=0, end_minute=9)
    res = sim.run([v], [future_inc])

    assert res.incidents[0].assigned_vehicle_id is None
    assert res.incidents[0].reached_minute is None


def test_vehicle_relocation_and_completion_timing():
    """Vehicle is relocated to incident coordinates and freed after travel + 8 min service."""
    # Vehicle at (0, 0), Incident at (3, 4) at minute 0
    # Distance = hypot(3, 4) = 5.0 min travel time
    # Reached minute = 0 + 5.0 = 5.0
    # Busy until = 5.0 + 8.0 = 13.0
    v = Vehicle(id=0, x=0.0, y=0.0)
    inc = Incident(id=1, arrival_minute=0, x=3.0, y=4.0, priority=2)

    # Simulate up to minute 12 (vehicle still busy)
    sim_busy = Simulator(dispatcher=NaiveDispatcher(), start_minute=0, end_minute=12)
    res_busy = sim_busy.run([v], [inc])
    assert res_busy.vehicles[0].idle is False
    assert round(res_busy.incidents[0].reached_minute, 4) == 5.0
    assert round(res_busy.incidents[0].completed_minute, 4) == 13.0

    # Simulate up to minute 13 (completion time <= 13, vehicle freed at (3, 4))
    sim_free = Simulator(dispatcher=NaiveDispatcher(), start_minute=0, end_minute=13)
    res_free = sim_free.run([v], [inc])
    freed_v = res_free.vehicles[0]
    assert freed_v.idle is True
    assert freed_v.x == 3.0
    assert freed_v.y == 4.0


def test_coverage_outage_counting():
    """Outage is detected and counted if any quadrant lacks idle vehicles."""
    # Only 1 vehicle in Q0, so Q1, Q2, Q3 have 0 idle vehicles
    v0 = Vehicle(id=0, x=10.0, y=10.0)  # in Q0
    
    sim = Simulator(dispatcher=NaiveDispatcher(), start_minute=0, end_minute=5)
    res = sim.run([v0], [])
    
    # 6 minutes total (0, 1, 2, 3, 4, 5) — each minute Q1-Q3 have 0 idle vehicles
    assert res.coverage_outage_minutes == 6
