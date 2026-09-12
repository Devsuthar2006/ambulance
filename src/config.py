"""Configuration constants for the Emergency Fleet Assignment Simulation & RESQAI Platform."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Region boundaries
GRID_WIDTH: float = 100.0
GRID_HEIGHT: float = 100.0
QUADRANT_SPLIT_X: float = 50.0
QUADRANT_SPLIT_Y: float = 50.0

# Fleet and Incidents
NUM_VEHICLES_PER_QUADRANT: int = 5
NUM_QUADRANTS: int = 4
TOTAL_VEHICLES: int = NUM_VEHICLES_PER_QUADRANT * NUM_QUADRANTS  # 20
TOTAL_INCIDENTS: int = 50

# Vehicle dynamics
VEHICLE_SPEED: float = 1.0  # coordinate unit / minute
SERVICE_TIME: float = 8.0   # minutes busy after arrival

# Incident timing and priority
INCIDENT_ARRIVAL_MIN: int = 0
INCIDENT_ARRIVAL_MAX: int = 59  # uniform in [0, 59] inclusive
PRIORITY_LEVELS: tuple[int, ...] = (1, 2, 3)
PRIORITY_PROBABILITIES: tuple[float, ...] = (0.60, 0.30, 0.10)
PRIORITY_WEIGHTS: dict[int, float] = {1: 1.0, 2: 3.0, 3: 7.0}

# Simulation clock
SIM_START_MINUTE: int = 0
SIM_END_MINUTE: int = 120  # t = 0 ... 120 inclusive (121 evaluation points)

# Default random seed
DEFAULT_SEED: int = int(os.environ.get("SEED", 20260911))

# Dispatcher defaults
DEFAULT_COVERAGE_PENALTY: float = float(os.environ.get("COVERAGE_PENALTY", 20.0))
DEFAULT_SCARCITY_PENALTY: float = float(os.environ.get("SCARCITY_PENALTY", 15.0))

# AI Triage & Emergency Taxonomy
EMERGENCY_CATEGORIES: tuple[str, ...] = (
    "MEDICAL",
    "TRAFFIC_ACCIDENT",
    "CARDIAC",
    "BREATHING",
    "FIRE",
    "INJURY",
    "UNCONSCIOUS_PERSON",
    "OTHER",
)

# Server & Provider Defaults
API_HOST: str = os.environ.get("API_HOST", "0.0.0.0")
API_PORT: int = int(os.environ.get("PORT", 8000))
TRIAGE_PROVIDER: str = os.environ.get("TRIAGE_PROVIDER", "mock")
VOICE_PROVIDER: str = os.environ.get("VOICE_PROVIDER", "demo")
