"""Tests for world generation reproducibility and constraints."""

import pytest
from src.generator import generate_world
from src.config import (
    DEFAULT_SEED,
    TOTAL_VEHICLES,
    TOTAL_INCIDENTS,
    NUM_QUADRANTS,
    NUM_VEHICLES_PER_QUADRANT,
    INCIDENT_ARRIVAL_MIN,
    INCIDENT_ARRIVAL_MAX,
)
from src.models import quadrant_of


def test_generator_reproducibility():
    """Generating twice with identical seed produces bit-identical fleets and incident lists."""
    v1, i1 = generate_world(seed=DEFAULT_SEED)
    v2, i2 = generate_world(seed=DEFAULT_SEED)

    assert len(v1) == len(v2) == TOTAL_VEHICLES
    assert len(i1) == len(i2) == TOTAL_INCIDENTS

    for veh1, veh2 in zip(v1, v2):
        assert veh1.id == veh2.id
        assert veh1.x == veh2.x
        assert veh1.y == veh2.y
        assert veh1.quadrant == veh2.quadrant

    for inc1, inc2 in zip(i1, i2):
        assert inc1.id == inc2.id
        assert inc1.arrival_minute == inc2.arrival_minute
        assert inc1.x == inc2.x
        assert inc1.y == inc2.y
        assert inc1.priority == inc2.priority
        assert inc1.weight == inc2.weight


def test_generator_different_seeds():
    """Different seeds generate different data."""
    v1, i1 = generate_world(seed=12345)
    v2, i2 = generate_world(seed=54321)

    assert any(veh1.x != veh2.x for veh1, veh2 in zip(v1, v2))
    assert any(inc1.x != inc2.x for inc1, inc2 in zip(i1, i2))


def test_vehicle_quadrant_distribution():
    """Exactly 5 vehicles are generated per quadrant inside boundaries."""
    vehicles, _ = generate_world(seed=DEFAULT_SEED)

    quadrant_counts = {q: 0 for q in range(NUM_QUADRANTS)}
    for v in vehicles:
        assert 0.0 <= v.x < 100.0
        assert 0.0 <= v.y < 100.0
        q = quadrant_of(v.x, v.y)
        assert v.quadrant == q
        quadrant_counts[q] += 1

    for q, count in quadrant_counts.items():
        assert count == NUM_VEHICLES_PER_QUADRANT


def test_incident_bounds_and_weights():
    """Incidents follow specified arrival time intervals, space bounds, and priority weights."""
    _, incidents = generate_world(seed=DEFAULT_SEED)

    for inc in incidents:
        assert INCIDENT_ARRIVAL_MIN <= inc.arrival_minute <= INCIDENT_ARRIVAL_MAX
        assert 0.0 <= inc.x <= 100.0
        assert 0.0 <= inc.y <= 100.0
        assert inc.priority in (1, 2, 3)
        expected_weight = {1: 1.0, 2: 3.0, 3: 7.0}[inc.priority]
        assert inc.weight == expected_weight
