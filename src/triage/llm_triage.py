"""LLM-based Emergency Triage Engine with graceful fallback to MockTriage."""

import json
import os
import urllib.request
import urllib.error
import src.config  # ensures .env is loaded
from src.triage.interface import TriageEngine
from src.triage.schemas import TriageResult
from src.triage.mock_triage import MockTriage


class GroqTriage(TriageEngine):
    """Ultra-fast Groq-powered AI Emergency Triage Engine (~150ms LPU latency)."""

    def __init__(self, api_key: str | None = None, model: str = "qwen/qwen3.8-27b") -> None:
        self.api_key = api_key or os.environ.get("GROQ_API_KEY") or os.environ.get("AI_API_KEY")
        self.model = os.environ.get("GROQ_MODEL", model)
        self.fallback = MockTriage()

    def triage(self, transcript: str) -> TriageResult:
        if not self.api_key:
            return self.fallback.triage(transcript)

        try:
            system_prompt = (
                "You are an Emergency Dispatch AI Triagist. Analyze the emergency call and classify.\n"
                "Allowed categories: [CARDIAC, BREATHING, TRAFFIC_ACCIDENT, FIRE, UNCONSCIOUS_PERSON, INJURY, MEDICAL, OTHER].\n"
                "Priority scale:\n"
                "- 3: Critical life-threatening emergencies (cardiac arrest, unconscious, choking, severe trauma, building fire).\n"
                "- 2: Urgent emergencies (fractures, traffic accident, severe asthma, uncontrolled pain).\n"
                "- 1: Routine non-urgent (sprained ankle, minor cuts, stable illness).\n"
                "Output JSON strictly with keys:\n"
                "- emergency_type: string (one of the allowed categories)\n"
                "- priority: integer (1, 2, or 3)\n"
                "- confidence: float (0.85 to 0.99)\n"
                "- evidence: list of short quoted facts from the caller transcript\n"
                "- summary: concise clinical assessment"
            )

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Emergency Call Transcript:\n\"{transcript}\""},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
            }

            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": "RESQAI-CommandCenter/1.0",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=4.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = json.loads(data["choices"][0]["message"]["content"])

                # Handle priority type parsing (number or string)
                raw_prio = content.get("priority", 2)
                if isinstance(raw_prio, str):
                    raw_prio = 3 if "critical" in raw_prio.lower() or "3" in raw_prio else (1 if "routine" in raw_prio.lower() or "1" in raw_prio else 2)
                else:
                    raw_prio = int(raw_prio)

                # Ensure emergency type matches valid uppercase taxonomy
                raw_type = str(content.get("emergency_type", "MEDICAL")).upper()

                return TriageResult(
                    emergency_type=raw_type,
                    priority=raw_prio,
                    confidence=float(content.get("confidence", 0.95)),
                    evidence=list(content.get("evidence", ["Groq LPU Real-Time AI Inference"])),
                    summary=str(content.get("summary", "Classified via Groq AI LPU")),
                )
        except Exception:
            # Fallback to deterministic mock triage on any network error or rate limit
            return self.fallback.triage(transcript)


class LLMTriage(TriageEngine):
    """AI Triage Engine using LLM API with zero-downtime fallback to MockTriage."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-3.5-turbo") -> None:
        self.api_key = api_key or os.environ.get("AI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.fallback = MockTriage()

    def triage(self, transcript: str) -> TriageResult:
        if not self.api_key:
            # Automatic fallback to deterministic mock triage
            return self.fallback.triage(transcript)

        # If key is Groq key, delegate to Groq
        if self.api_key.startswith("gsk_"):
            return GroqTriage(api_key=self.api_key).triage(transcript)

        try:
            # Structured prompt enforcing exact JSON output
            system_prompt = (
                "You are an Emergency Dispatch AI Triagist. "
                "Analyze the caller's statement and classify the emergency.\n"
                "Allowed categories: [MEDICAL, TRAFFIC_ACCIDENT, CARDIAC, BREATHING, FIRE, INJURY, UNCONSCIOUS_PERSON, OTHER]\n"
                "Allowed priorities: 1 (Routine), 2 (Urgent), 3 (Critical life-threatening)\n"
                "Output JSON strictly with keys: emergency_type, priority, confidence, evidence (list of short strings), summary"
            )

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Emergency Call Transcript:\n\"{transcript}\""},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
            }

            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=4.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = json.loads(data["choices"][0]["message"]["content"])
                return TriageResult(
                    emergency_type=content.get("emergency_type", "MEDICAL"),
                    priority=int(content.get("priority", 2)),
                    confidence=float(content.get("confidence", 0.90)),
                    evidence=content.get("evidence", ["Analyzed via AI LLM Triage"]),
                    summary=content.get("summary", "Emergency classified by AI"),
                )
        except Exception:
            # Safe fallback on network failure, rate limit, or invalid response
            return self.fallback.triage(transcript)


def get_triage_engine() -> TriageEngine:
    """Factory to instantiate the appropriate triage engine based on environment configuration."""
    provider = os.environ.get("TRIAGE_PROVIDER", "mock").lower()
    groq_key = os.environ.get("GROQ_API_KEY")
    ai_key = os.environ.get("AI_API_KEY")

    # Priority 1: Groq provider or Groq API key
    if provider == "groq" or groq_key or (ai_key and ai_key.startswith("gsk_")):
        return GroqTriage(api_key=groq_key or ai_key)

    # Priority 2: Generic LLM / OpenAI
    if provider in ("llm", "openai", "gemini") and (ai_key or os.environ.get("OPENAI_API_KEY")):
        return LLMTriage()

    return MockTriage()
