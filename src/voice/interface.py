"""Abstract interface for Voice Intake providers."""

from abc import ABC, abstractmethod


class VoiceProvider(ABC):
    """Abstract emergency voice interaction provider."""

    @abstractmethod
    def process_call(self, payload: dict) -> str:
        """Convert incoming audio/speech payload to raw transcript text."""
        pass
