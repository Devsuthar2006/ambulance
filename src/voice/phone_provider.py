"""Telephony webhook intake provider (Twilio / Telnyx compatible)."""

import os
from src.voice.interface import VoiceProvider


class PhoneWebhookProvider(VoiceProvider):
    """Processes incoming phone call transcripts from external telephony webhooks."""

    def __init__(self) -> None:
        self.phone_number = os.environ.get("PHONE_NUMBER", "+1-800-555-RESQ")
        self.api_key = os.environ.get("PHONE_PROVIDER_API_KEY")

    def process_call(self, payload: dict) -> str:
        """Parse standard Twilio Voice or Telnyx webhook payload."""
        # Twilio sends 'SpeechResult' or 'TranscriptionText'
        speech_result = payload.get("SpeechResult") or payload.get("TranscriptionText") or payload.get("text")
        if speech_result:
            return speech_result.strip()

        caller = payload.get("From", "Unknown Caller")
        return f"Incoming phone call from {caller}. No verbal statement detected."

    def generate_twiml_response(self, action_url: str = "/api/call/voice") -> str:
        """Generate TwiML XML to prompt caller and gather speech."""
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Response>'
            '<Say voice="alice">Emergency response system. What is your emergency?</Say>'
            f'<Gather input="speech" action="{action_url}" timeout="3" speechTimeout="auto" />'
            '</Response>'
        )
