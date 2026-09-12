"""Browser microphone and Web Speech API intake provider."""

from src.voice.interface import VoiceProvider


class BrowserVoiceProvider(VoiceProvider):
    """Processes audio and transcriptions received from client-side Web Speech API."""

    def process_call(self, payload: dict) -> str:
        transcript = payload.get("transcript", "").strip()
        if not transcript:
            return "Emergency caller connected via browser microphone. Static on line."
        return transcript
