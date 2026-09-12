"""Deterministic world generator for vehicles and incidents using NumPy PCG64."""

import numpy as np
from src.config import (
    DEFAULT_SEED,
    GRID_WIDTH,
    GRID_HEIGHT,
    NUM_QUADRANTS,
    NUM_VEHICLES_PER_QUADRANT,
    TOTAL_INCIDENTS,
    INCIDENT_ARRIVAL_MIN,
    INCIDENT_ARRIVAL_MAX,
    PRIORITY_LEVELS,
    PRIORITY_PROBABILITIES,
    QUADRANT_SPLIT_X,
    QUADRANT_SPLIT_Y,
)
from src.models import Vehicle, Incident


def generate_world(seed: int = DEFAULT_SEED) -> tuple[list[Vehicle], list[Incident]]:
    """Generate vehicles and incidents deterministically from the given seed.
    
    Returns:
        tuple[list[Vehicle], list[Incident]]: Generated fleet and incident stream.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    
    # Quadrant coordinate boundaries: (x_min, x_max, y_min, y_max)
    quadrant_bounds = {
        0: (0.0, QUADRANT_SPLIT_X, 0.0, QUADRANT_SPLIT_Y),
        1: (QUADRANT_SPLIT_X, GRID_WIDTH, 0.0, QUADRANT_SPLIT_Y),
        2: (0.0, QUADRANT_SPLIT_X, QUADRANT_SPLIT_Y, GRID_HEIGHT),
        3: (QUADRANT_SPLIT_X, GRID_WIDTH, QUADRANT_SPLIT_Y, GRID_HEIGHT),
    }

    # 1. Generate 20 vehicles (5 in each quadrant)
    vehicles: list[Vehicle] = []
    vehicle_id = 0
    for q in range(NUM_QUADRANTS):
        x_min, x_max, y_min, y_max = quadrant_bounds[q]
        for _ in range(NUM_VEHICLES_PER_QUADRANT):
            vx = float(rng.uniform(x_min, x_max))
            vy = float(rng.uniform(y_min, y_max))
            v = Vehicle(id=vehicle_id, x=vx, y=vy)
            assert v.quadrant == q, f"Vehicle {vehicle_id} landed in quadrant {v.quadrant}, expected {q}"
            vehicles.append(v)
            vehicle_id += 1

    # 2. Generate 100 incidents
    incidents: list[Incident] = []
    for inc_id in range(TOTAL_INCIDENTS):
        arrival_minute = int(rng.integers(INCIDENT_ARRIVAL_MIN, INCIDENT_ARRIVAL_MAX + 1))
        ix = float(rng.uniform(0.0, GRID_WIDTH))
        iy = float(rng.uniform(0.0, GRID_HEIGHT))
        priority = int(rng.choice(PRIORITY_LEVELS, p=PRIORITY_PROBABILITIES))
        incidents.append(
            Incident(
                id=inc_id,
                arrival_minute=arrival_minute,
                x=ix,
                y=iy,
                priority=priority,
            )
        )

    return vehicles, incidents
