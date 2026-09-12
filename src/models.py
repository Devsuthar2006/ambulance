"""Data models for vehicles, incidents, and quadrant calculations."""

from dataclasses import dataclass, field
from src.config import (
    QUADRANT_SPLIT_X,
    QUADRANT_SPLIT_Y,
    PRIORITY_WEIGHTS,
)


def quadrant_of(x: float, y: float) -> int:
    """Return quadrant index 0..3 based on standard boundaries.
    
    0: Left & Lower (x < 50, y < 50)
    1: Right & Lower (x >= 50, y < 50)
    2: Left & Upper (x < 50, y >= 50)
    3: Right & Upper (x >= 50, y >= 50)
    """
    is_right = 1 if x >= QUADRANT_SPLIT_X else 0
    is_upper = 1 if y >= QUADRANT_SPLIT_Y else 0
    return is_right + 2 * is_upper


@dataclass
class Vehicle:
    """Emergency vehicle state."""
    id: int
    x: float
    y: float
    idle: bool = True
    busy_until: float = 0.0
    quadrant: int = field(init=False)
    assigned_count: int = 0

    def __post_init__(self) -> None:
        self.quadrant = quadrant_of(self.x, self.y)

    def update_position(self, new_x: float, new_y: float) -> None:
        """Update vehicle position and recompute its quadrant."""
        self.x = new_x
        self.y = new_y
        self.quadrant = quadrant_of(self.x, self.y)

    def to_dict(self) -> dict:
        """Serialize vehicle to JSON-friendly dict."""
        return {
            "id": self.id,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "idle": self.idle,
            "busy_until": round(self.busy_until, 2),
            "quadrant": self.quadrant,
            "assigned_count": self.assigned_count,
        }


@dataclass
class Incident:
    """Unified emergency incident model."""
    id: int
    arrival_minute: int
    x: float
    y: float
    priority: int
    weight: float = field(init=False)
    emergency_type: str = "MEDICAL"
    source: str = "BENCHMARK"  # "BENCHMARK" | "SIMULATION" | "PHONE" | "BROWSER"
    transcript: str | None = None
    evidence: list[str] = field(default_factory=list)
    confidence: float = 1.0
    explanation: dict | None = None
    assigned_vehicle_id: int | None = None
    assigned_minute: float | None = None
    reached_minute: float | None = None
    completed_minute: float | None = None

    def __post_init__(self) -> None:
        self.weight = PRIORITY_WEIGHTS[self.priority]

    @property
    def response_time(self) -> float | None:
        """Response time is reached_minute - arrival_minute."""
        if self.reached_minute is None:
            return None
        return self.reached_minute - self.arrival_minute

    def to_dict(self) -> dict:
        """Serialize incident to JSON-compatible dictionary."""
        return {
            "id": self.id,
            "arrival_minute": self.arrival_minute,
            "x": round(float(self.x), 2),
            "y": round(float(self.y), 2),
            "priority": self.priority,
            "weight": self.weight,
            "emergency_type": self.emergency_type,
            "source": self.source,
            "transcript": self.transcript,
            "evidence": self.evidence,
            "confidence": round(float(self.confidence), 2),
            "assigned_vehicle_id": self.assigned_vehicle_id,
            "assigned_minute": round(float(self.assigned_minute), 2) if self.assigned_minute is not None else None,
            "reached_minute": round(float(self.reached_minute), 2) if self.reached_minute is not None else None,
            "completed_minute": round(float(self.completed_minute), 2) if self.completed_minute is not None else None,
            "response_time": round(float(self.response_time), 2) if self.response_time is not None else None,
            "explanation": self.explanation,
        }
