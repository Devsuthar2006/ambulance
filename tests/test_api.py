"""Tests for RESQAI FastAPI REST endpoints."""

from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)


def test_health_endpoint():
    """Health check endpoint returns online status and safety disclaimer."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["platform"] == "RESQAI"
    assert "NOT A REAL EMERGENCY SERVICE" in data["disclaimer"]


def test_triage_test_endpoint():
    """Triage test endpoint classifies custom caller statement."""
    payload = {"transcript": "Elderly person collapsed in hallway, clutching chest."}
    response = client.post("/api/triage/test", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["priority"] == 3
    assert data["emergency_type"] in ("CARDIAC", "UNCONSCIOUS_PERSON")
    assert data["confidence"] >= 0.85


def test_voice_call_intake_endpoint():
    """Voice call endpoint processes speech, runs triage, and creates explainable dispatch."""
    payload = {
        "transcript": "Serious bike accident at (72, 31), rider is unconscious and bleeding.",
        "x": 72.0,
        "y": 31.0,
        "policy": "coverage",
    }
    response = client.post("/api/call/voice", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "DISPATCHED"
    assert data["triage"]["priority"] == 3
    assert data["explanation"] is not None
    assert "candidates" in data["explanation"]


def test_scenario_run_endpoint():
    """Scenario execution endpoint returns comparative metrics across policies."""
    payload = {"scenario": "NORMAL"}
    response = client.post("/api/scenarios/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "policies" in data
    assert "coverage" in data["policies"]
    assert "nearest" in data["policies"]
    assert "adaptive" in data["policies"]


def test_benchmark_compare_endpoint():
    """Benchmark compare endpoint returns real comparative metrics."""
    response = client.get("/api/benchmark/compare")
    assert response.status_code == 200
    data = response.json()
    assert "policies" in data
