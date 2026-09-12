"""Voice and audio intake providers for RESQAI."""
from src.voice.interface import VoiceProvider
from src.voice.demo_provider import DemoAudioProvider, DEMO_EMERGENCY_CALLS
from src.voice.browser_provider import BrowserVoiceProvider
from src.voice.phone_provider import PhoneWebhookProvider

__all__ = [
    "VoiceProvider",
    "DemoAudioProvider",
    "BrowserVoiceProvider",
    "PhoneWebhookProvider",
    "DEMO_EMERGENCY_CALLS",
]
