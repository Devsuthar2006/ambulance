"""Schemas for AI Emergency Triage results."""

from dataclasses import dataclass, field
from pydantic import BaseModel, Field


class TriageResult(BaseModel):
    """Structured result of emergency conversation classification."""
    emergency_type: str = Field(..., description="Category from EMERGENCY_CATEGORIES")
    priority: int = Field(..., ge=1, le=3, description="Priority level 1, 2, or 3")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    evidence: list[str] = Field(default_factory=list, description="Key symptom/scenario evidence phrases")
    summary: str = Field(..., description="Short plain-English assessment")
    location_hint: tuple[float, float] | None = Field(default=None, description="Optional parsed (x, y) coordinates")

    @property
    def priority_label(self) -> str:
        labels = {1: "P1 — Routine", 2: "P2 — Urgent", 3: "P3 — Critical"}
        return labels.get(self.priority, f"P{self.priority}")

    def to_dict(self) -> dict:
        return {
            "emergency_type": self.emergency_type,
            "priority": self.priority,
            "priority_label": self.priority_label,
            "confidence": round(self.confidence, 2),
            "evidence": self.evidence,
            "summary": self.summary,
            "location_hint": self.location_hint,
        }
