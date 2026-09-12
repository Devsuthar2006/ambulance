"""Unit tests for traffic model, traffic-aware dispatching, and gridlock bypass."""

import pytest
import numpy as np
from src.models import Vehicle, Incident
from src.traffic import TrafficModel, TrafficZone, TrafficCorridor, get_default_traffic_model
from src.traffic.traffic_model import grid_to_latlon, latlon_to_grid
from src.dispatcher import NaiveDispatcher, CoverageAwareDispatcher, AdaptiveDispatcher
from src.simulator import Simulator


def test_grid_latlon_conversion():
    """Verify conversion between simulation grid coordinates and geographic lat/lon."""
    lat, lon = grid_to_latlon(50.0, 50.0)
    assert 37.70 < lat < 37.82
    assert -122.52 < lon < -122.37

    # Round-trip conversion
    x, y = latlon_to_grid(lat, lon)
    assert abs(x - 50.0) < 0.01
    assert abs(y - 50.0) < 0.01


def test_traffic_zone_decay():
    """Verify traffic multiplier decays from center to zone radius."""
    zone = TrafficZone(
        id="Z1",
        name="Test Zone",
        center_x=50.0,
        center_y=50.0,
        radius=10.0,
        multiplier=3.0,
    )
    # At center, should be max multiplier
    assert abs(zone.get_multiplier_at(50.0, 50.0) - 3.0) < 0.01
    # Outside radius, should be 1.0 (clear)
    assert abs(zone.get_multiplier_at(70.0, 50.0) - 1.0) < 0.01
    # Midpoint should be between 1.0 and 3.0
    mid_mult = zone.get_multiplier_at(55.0, 50.0)
    assert 1.5 < mid_mult < 2.5


def test_traffic_model_travel_time():
    """Verify travel time increases across congested zones."""
    tm = get_default_traffic_model()

    # Clear coastal avenue
    time_clear = tm.get_travel_time(10.0, 20.0, 10.0, 30.0)
    # Downtown gridlocked sector (same distance of 10 units)
    time_gridlock = tm.get_travel_time(70.0, 65.0, 70.0, 75.0)

    assert time_gridlock > time_clear * 2.0


def test_gridlock_bypass_priority3_emergency():
    """Verify that Traffic-Aware Dispatcher bypasses a closer ambulance stuck in traffic

    for a critical P3 emergency when a slightly further ambulance has a faster ETA.
    """
    tm = get_default_traffic_model()

    # Emergency at (75, 60)
    inc = Incident(id=101, arrival_minute=0, x=75.0, y=60.0, priority=3)

    # v1 is closer geographically (dist ~ 9.5), but in Downtown gridlock (mult ~ 3.0x -> ETA ~ 28.9m)
    v1 = Vehicle(id=1, x=72.0, y=69.0)

    # v2 is further geographically (dist = 12.0), but in clear South corridor (mult ~ 1.3x -> ETA ~ 15.6m)
    v2 = Vehicle(id=2, x=75.0, y=48.0)

    idle_vehicles = [v1, v2]
    idle_counts = np.array([2, 2, 2, 2])

    # 1. Naive picks v1 because dist(v1) < dist(v2)
    naive = NaiveDispatcher(traffic_model=tm)
    chosen_naive, expl_naive = naive.assign_with_explanation(idle_vehicles, inc, idle_counts)
    assert chosen_naive.id == 1

    # 2. CoverageAware with traffic picks v2 because ETA(v2) < ETA(v1)
    cov = CoverageAwareDispatcher(traffic_model=tm)
    chosen_cov, expl_cov = cov.assign_with_explanation(idle_vehicles, inc, idle_counts)
    assert chosen_cov.id == 2
    assert expl_cov["strategy"] == "TRAFFIC_GRIDLOCK_BYPASS"
    assert "faster" in expl_cov["reason"]


def test_simulator_with_traffic():
    """Verify simulation executes properly with traffic model."""
    tm = get_default_traffic_model()
    vehicles = [
        Vehicle(id=i, x=float((i % 4) * 25 + 12), y=float((i // 4) * 20 + 10))
        for i in range(20)
    ]
    incidents = [
        Incident(id=i, arrival_minute=i * 2, x=float((i * 13) % 100), y=float((i * 27) % 100), priority=(i % 3) + 1)
        for i in range(15)
    ]

    disp = CoverageAwareDispatcher(traffic_model=tm)
    sim = Simulator(dispatcher=disp, traffic_model=tm, start_minute=0, end_minute=40)
    res = sim.run(vehicles, incidents)

    assert len(res.incidents) == 15
    # At least some incidents should be assigned and reached
    assigned = [inc for inc in res.incidents if inc.assigned_vehicle_id is not None]
    assert len(assigned) > 0
    for inc in assigned:
        assert inc.reached_minute is not None
        assert inc.reached_minute >= inc.assigned_minute
