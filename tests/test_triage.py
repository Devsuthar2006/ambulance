"""Tests for AI Emergency Triage schema, mock classification, and fallback safety."""

from src.triage.mock_triage import MockTriage
from src.triage.llm_triage import LLMTriage
from src.triage.schemas import TriageResult
from src.models import Incident


def test_mock_triage_critical_detection():
    """MockTriage correctly classifies life-threatening statements as Priority 3."""
    triage = MockTriage()

    # Bike crash unconscious
    res1 = triage.triage("There has been a serious bike accident. The person is unconscious and bleeding badly.")
    assert res1.priority == 3
    assert res1.emergency_type == "TRAFFIC_ACCIDENT"
    assert res1.confidence >= 0.90
    assert any("unconscious" in ev.lower() for ev in res1.evidence)

    # Cardiac arrest
    res2 = triage.triage("My father collapsed on the floor clutching his chest, not responding!")
    assert res2.priority == 3
    assert res2.emergency_type in ("CARDIAC", "UNCONSCIOUS_PERSON")
    assert res2.confidence >= 0.90


def test_mock_triage_urgent_and_routine():
    """MockTriage distinguishes urgent P2 from routine P1 calls."""
    triage = MockTriage()

    # P2 Urgent
    res_p2 = triage.triage("Car crash at intersection, driver has broken leg and is trapped inside.")
    assert res_p2.priority == 2
    assert res_p2.emergency_type == "TRAFFIC_ACCIDENT"

    # P1 Routine
    res_p1 = triage.triage("Elderly patient slipped out of bed, completely alert and uninjured, needs lifting assistance.")
    assert res_p1.priority == 1
    assert res_p1.emergency_type == "MEDICAL"


def test_triage_to_incident_conversion():
    """TriageResult converts cleanly into structured Incident model."""
    triage = MockTriage()
    res = triage.triage("Pedestrian struck by car at (45, 60), severe head injury and bleeding.")

    inc = Incident(
        id=501,
        arrival_minute=10,
        x=res.location_hint[0] if res.location_hint else 45.0,
        y=res.location_hint[1] if res.location_hint else 60.0,
        priority=res.priority,
        emergency_type=res.emergency_type,
        source="PHONE",
        transcript="Pedestrian struck by car at (45, 60), severe head injury and bleeding.",
        evidence=res.evidence,
        confidence=res.confidence,
    )

    assert inc.priority == 3
    assert inc.weight == 7.0
    assert inc.x == 45.0
    assert inc.y == 60.0
    assert inc.source == "PHONE"


def test_llm_triage_graceful_fallback():
    """LLMTriage gracefully falls back to MockTriage when API key is missing or invalid."""
    llm = LLMTriage(api_key="invalid_fake_key_for_testing")
    # Must not raise an exception, must return valid TriageResult
    res = llm.triage("Severe building fire with trapped residents.")
    assert isinstance(res, TriageResult)
    assert res.priority in (2, 3)
    assert res.emergency_type == "FIRE"
