"""Abstract interface for AI Emergency Triage engines."""

from abc import ABC, abstractmethod
from src.triage.schemas import TriageResult


class TriageEngine(ABC):
    """Abstract emergency classification and prioritization engine."""

    @abstractmethod
    def triage(self, transcript: str) -> TriageResult:
        """Classify unstructured emergency text into structured triage data."""
        pass
