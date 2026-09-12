"""Event streaming and Pub/Sub bus for real-time dashboard updates."""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Callable


class EventType(str, Enum):
    INCIDENT_CREATED = "INCIDENT_CREATED"
    TRIAGE_COMPLETED = "TRIAGE_COMPLETED"
    DISPATCH_DECISION = "DISPATCH_DECISION"
    VEHICLE_DISPATCHED = "VEHICLE_DISPATCHED"
    VEHICLE_MOVED = "VEHICLE_MOVED"
    VEHICLE_ARRIVED = "VEHICLE_ARRIVED"
    SERVICE_STARTED = "SERVICE_STARTED"
    SERVICE_COMPLETED = "SERVICE_COMPLETED"
    VEHICLE_AVAILABLE = "VEHICLE_AVAILABLE"
    COVERAGE_CHANGED = "COVERAGE_CHANGED"
    COVERAGE_WARNING = "COVERAGE_WARNING"
    COVERAGE_OUTAGE = "COVERAGE_OUTAGE"
    SIMULATION_START = "SIMULATION_START"
    SIMULATION_RESET = "SIMULATION_RESET"


@dataclass
class SimulationEvent:
    """Structured event emitted during real-time simulation and dispatch."""
    event_type: EventType
    minute: float
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    level: str = "info"  # "info" | "dispatch" | "warning" | "danger" | "critical" | "success"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.event_type.value,
            "minute": round(float(self.minute), 2),
            "message": self.message,
            "data": self.data,
            "level": self.level,
            "timestamp": self.timestamp,
        }


class EventBus:
    """In-memory event broadcaster for decoupled dashboard integration."""

    def __init__(self, max_history: int = 2000) -> None:
        self.subscribers: set[asyncio.Queue] = set()
        self.history: list[SimulationEvent] = []
        self.max_history = max_history

    def emit(self, event: SimulationEvent) -> None:
        """Publish an event to history and all active WebSocket queues."""
        self.history.append(event)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # Notify active subscriber queues safely
        for queue in list(self.subscribers):
            try:
                queue.put_nowait(event.to_dict())
            except Exception:
                pass

    def subscribe(self) -> asyncio.Queue:
        """Register a new subscriber queue."""
        q: asyncio.Queue = asyncio.Queue()
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Unregister a subscriber queue."""
        self.subscribers.discard(q)

    def clear(self) -> None:
        self.history.clear()
