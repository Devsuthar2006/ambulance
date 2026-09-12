"""FastAPI server for RESQAI Real-Time Emergency Response Platform."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import DEFAULT_SEED, DEFAULT_COVERAGE_PENALTY, DEFAULT_SCARCITY_PENALTY, API_HOST, API_PORT
from src.generator import generate_world
from src.models import Vehicle, Incident, quadrant_of
from src.dispatcher import get_dispatcher
from src.triage.llm_triage import get_triage_engine
from src.voice.demo_provider import DemoAudioProvider, DEMO_EMERGENCY_CALLS
from src.voice.browser_provider import BrowserVoiceProvider
from src.voice.phone_provider import PhoneWebhookProvider
from src.events.event_bus import EventBus, SimulationEvent, EventType
from src.scenarios import ScenarioRunner
from src.risk.risk_engine import RiskEngine

app = FastAPI(
    title="RESQAI — Real-Time Emergency Response Intelligence Platform",
    description="Intelligent emergency triage, causal coverage-preserving fleet assignment, and live command-center stream.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Platform Singletons
event_bus = EventBus()
triage_engine = get_triage_engine()
demo_voice = DemoAudioProvider()
browser_voice = BrowserVoiceProvider()
phone_voice = PhoneWebhookProvider()
scenario_runner = ScenarioRunner()
risk_engine = RiskEngine()

# Dispatch lock — prevents concurrent requests from assigning the same vehicle
# to two different incidents (PRD §3: each incident gets exactly one vehicle)
_dispatch_lock = asyncio.Lock()

# Live demo fleet state (initialized from seed)
live_vehicles, _ = generate_world(DEFAULT_SEED)
live_incidents: list[Incident] = []
live_minute: float = 0.0
next_incident_id: int = 2000

# Track which incident IDs are already dispatched / queued (prevents re-dispatch)
_dispatched_incident_ids: set[int] = set()


# Request Schemas
class CallRequest(BaseModel):
    transcript: str | None = None
    call_id: str | None = None
    source: str = "BROWSER"  # "PHONE" | "BROWSER" | "SIMULATION"
    x: float | None = None
    y: float | None = None
    policy: str = "coverage"


class TriageTestRequest(BaseModel):
    transcript: str


class ScenarioRequest(BaseModel):
    scenario: str = "NORMAL"
    seed: int = DEFAULT_SEED


@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "platform": "RESQAI",
        "version": "2.0.0",
        "seed": DEFAULT_SEED,
        "disclaimer": "DEMO / SIMULATION — NOT A REAL EMERGENCY SERVICE",
    }


@app.get("/api/demo-calls")
async def get_demo_presets():
    """Return realistic demo emergency call presets for the dashboard UI."""
    return {"calls": DEMO_EMERGENCY_CALLS}


@app.post("/api/triage/test")
async def test_triage(req: TriageTestRequest):
    """Run AI Triage on arbitrary emergency transcript."""
    res = triage_engine.triage(req.transcript)
    return res.to_dict()


@app.post("/api/call/voice")
async def handle_voice_call(request: Request):
    """Universal Voice Intake endpoint: handles Twilio webhooks, browser mic, or JSON payloads.

    PRD §3 compliance: the dispatch lock ensures strict serial processing so that each
    incident is assigned to exactly one vehicle — never two ambulances to the same call.
    """
    global next_incident_id, live_minute, _dispatched_incident_ids

    # Parse JSON or form data (Twilio sends form data)
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
    else:
        form_data = await request.form()
        body = dict(form_data)

    source = body.get("source", "PHONE" if "From" in body else "BROWSER")
    if source == "PHONE":
        transcript = phone_voice.process_call(body)
    elif source == "BROWSER":
        transcript = browser_voice.process_call(body)
    else:
        transcript = demo_voice.process_call(body)

    # 1. AI Triage
    triage_res = triage_engine.triage(transcript)

    # 2. Location determination
    loc_x = float(body.get("x", 0.0)) if body.get("x") is not None else None
    loc_y = float(body.get("y", 0.0)) if body.get("y") is not None else None

    if loc_x is None or loc_y is None:
        if triage_res.location_hint:
            loc_x, loc_y = triage_res.location_hint
        else:
            # Deterministic location hash or quadrant distribution
            loc_x = round(float((next_incident_id * 17) % 100), 2)
            loc_y = round(float((next_incident_id * 31) % 100), 2)

    # 3. Create Incident
    inc_id = next_incident_id
    next_incident_id += 1

    inc = Incident(
        id=inc_id,
        arrival_minute=int(live_minute),
        x=loc_x,
        y=loc_y,
        priority=triage_res.priority,
        emergency_type=triage_res.emergency_type,
        source=source,
        transcript=transcript,
        evidence=triage_res.evidence,
        confidence=triage_res.confidence,
    )
    live_incidents.append(inc)

    # Emit Incident and Triage events
    event_bus.emit(
        SimulationEvent(
            event_type=EventType.INCIDENT_CREATED,
            minute=live_minute,
            message=f"Call #{inc.id} arrived via {source} [{triage_res.priority_label}] at ({loc_x:.1f}, {loc_y:.1f})",
            level="critical" if inc.priority == 3 else ("warning" if inc.priority == 2 else "info"),
            data=inc.to_dict(),
        )
    )

    event_bus.emit(
        SimulationEvent(
            event_type=EventType.TRIAGE_COMPLETED,
            minute=live_minute,
            message=f"AI Triage: {triage_res.emergency_type} ({triage_res.confidence*100:.0f}% conf) — {triage_res.summary}",
            level="info",
            data=triage_res.to_dict(),
        )
    )

    # 4. Dispatch Assignment with Explainability
    # PRD §3 HARD CONSTRAINT: acquire lock so no two concurrent requests can
    # simultaneously read the fleet as idle and assign the same vehicle twice.
    policy_name = body.get("policy", "coverage")
    dispatcher = get_dispatcher(policy_name)

    vehicle_dict = None
    new_idle_counts = [0, 0, 0, 0]

    async with _dispatch_lock:
        # Re-check vehicle availability inside lock (serialize all mutations)
        idle_counts = [0, 0, 0, 0]
        idle_vehicles = [v for v in live_vehicles if v.idle]
        for v in idle_vehicles:
            idle_counts[v.quadrant] += 1

        chosen_vehicle, explanation = dispatcher.assign_with_explanation(
            idle_vehicles=idle_vehicles,
            incident=inc,
            idle_counts_by_quadrant=idle_counts,
        )

        inc.explanation = explanation

        if chosen_vehicle is not None:
            # Verify vehicle is truly still idle (double-check inside lock)
            v = next((veh for veh in live_vehicles if veh.id == chosen_vehicle.id and veh.idle), None)
            if v is None:
                # Vehicle sniped by concurrent request — fall back to next best
                remaining_idle = [veh for veh in live_vehicles if veh.idle]
                if remaining_idle:
                    idle_counts_fb = [0, 0, 0, 0]
                    for veh in remaining_idle:
                        idle_counts_fb[veh.quadrant] += 1
                    fallback, explanation = dispatcher.assign_with_explanation(
                        idle_vehicles=remaining_idle,
                        incident=inc,
                        idle_counts_by_quadrant=idle_counts_fb,
                    )
                    inc.explanation = explanation
                    v = next((veh for veh in live_vehicles if veh.id == fallback.id), None) if fallback else None

            if v is not None:
                dist = float(((v.x - inc.x)**2 + (v.y - inc.y)**2)**0.5)
                travel_time = dist / 1.0
                reached_time = live_minute + travel_time
                completed_time = reached_time + 8.0

                inc.assigned_vehicle_id = v.id
                inc.assigned_minute = live_minute
                inc.reached_minute = reached_time
                inc.completed_minute = completed_time

                # Mark vehicle busy INSIDE the lock — prevents any concurrent
                # request from seeing this vehicle as idle
                v.idle = False
                v.busy_until = completed_time
                _dispatched_incident_ids.add(inc.id)

                vehicle_dict = {
                    "id": v.id,
                    "start_x": v.x,
                    "start_y": v.y,
                    "dest_x": inc.x,
                    "dest_y": inc.y,
                    "travel_time": round(travel_time, 2),
                    "reached_minute": round(reached_time, 2),
                    "completed_minute": round(completed_time, 2),
                    "quadrant": v.quadrant,
                }
                # Vehicle completes service at incident location (PRD §3: idles at incident site)
                v.x = inc.x
                v.y = inc.y
            else:
                chosen_vehicle = None  # All vehicles busy; incident queued

        # Recompute idle counts after assignment, still inside lock
        for veh in live_vehicles:
            if veh.idle:
                new_idle_counts[veh.quadrant] += 1
    # --- lock released ---

    if chosen_vehicle is not None and vehicle_dict is not None:
        event_bus.emit(
            SimulationEvent(
                event_type=EventType.DISPATCH_DECISION,
                minute=live_minute,
                message=explanation["reason"],
                level="dispatch",
                data=explanation,
            )
        )

        event_bus.emit(
            SimulationEvent(
                event_type=EventType.VEHICLE_DISPATCHED,
                minute=live_minute,
                message=f"Ambulance #{inc.assigned_vehicle_id} en route to Call #{inc.id} (ETA: {vehicle_dict['travel_time']:.1f}m)",
                level="dispatch",
                data={"vehicle_id": inc.assigned_vehicle_id, "incident_id": inc.id, "travel_time": vehicle_dict['travel_time']},
            )
        )

        if any(c == 0 for c in new_idle_counts):
            zero_quads = [f"Q{i}" for i, c in enumerate(new_idle_counts) if c == 0]
            event_bus.emit(
                SimulationEvent(
                    event_type=EventType.COVERAGE_OUTAGE,
                    minute=live_minute,
                    message=f"Coverage Outage: {', '.join(zero_quads)} has 0 idle ambulances!",
                    level="danger",
                    data={"idle_counts": new_idle_counts},
                )
            )

    return {
        "status": "DISPATCHED" if (chosen_vehicle and vehicle_dict) else "QUEUED",
        "incident": inc.to_dict(),
        "vehicle": vehicle_dict,
        "triage": triage_res.to_dict(),
        "explanation": explanation,
        "quadrant_idle_counts": new_idle_counts if (chosen_vehicle and vehicle_dict) else idle_counts,
    }


@app.post("/api/live/reset")
async def reset_live_fleet():
    """Reset live fleet and custom calls back to clean initial standby state."""
    global live_vehicles, live_incidents, live_minute, next_incident_id, _dispatched_incident_ids
    async with _dispatch_lock:
        live_vehicles, _ = generate_world(DEFAULT_SEED)
        live_incidents = []
        live_minute = 0.0
        next_incident_id = 2000
        _dispatched_incident_ids = set()
    event_bus.emit(
        SimulationEvent(
            event_type=EventType.SIMULATION_START,
            minute=0.0,
            message="Live fleet reset to standby readiness (all 20 units ready at bases).",
            level="info",
        )
    )
    return {"status": "RESET", "vehicle_count": len(live_vehicles)}


@app.get("/api/live/state")
async def get_live_state():
    """Return live fleet state, active incidents, and quadrant coverage."""
    idle_counts = [0, 0, 0, 0]
    for v in live_vehicles:
        if v.idle:
            idle_counts[v.quadrant] += 1
    return {
        "live_minute": live_minute,
        "vehicles": [v.to_dict() for v in live_vehicles],
        "incidents": [inc.to_dict() for inc in live_incidents],
        "idle_counts": idle_counts,
    }


@app.post("/api/scenarios/run")
async def run_scenario(req: ScenarioRequest):
    """Execute a What-If scenario across all 3 dispatchers and return comparative metrics."""
    return scenario_runner.run_comparison(scenario_name=req.scenario, seed=req.seed)


@app.get("/api/benchmark/compare")
async def compare_benchmark(seed: int = DEFAULT_SEED):
    """Run official A/B comparison across Nearest, Coverage-Aware, and Adaptive policies."""
    return scenario_runner.run_comparison(scenario_name="NORMAL", seed=seed)


@app.get("/api/playback")
async def get_playback():
    """Serve precomputed 100-call playback traces."""
    path = Path("outputs/playback_data.json")
    if not path.exists():
        from src.export_playback import main as export_main
        export_main()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """Live bidirectional WebSocket for streaming dispatch and coverage events."""
    await websocket.accept()
    queue = event_bus.subscribe()
    try:
        # Send initial connection greeting and last 10 historical events
        await websocket.send_json({
            "type": "SYSTEM_CONNECTED",
            "message": "Connected to RESQAI Real-Time Dispatch Stream.",
            "history": [e.to_dict() for e in event_bus.history[-15:]],
        })

        while True:
            # Check for queue events or client commands
            try:
                event_data = await asyncio.wait_for(queue.get(), timeout=0.5)
                await websocket.send_json(event_data)
            except asyncio.TimeoutError:
                pass
    except WebSocketDisconnect:
        event_bus.unsubscribe(queue)
    except Exception:
        event_bus.unsubscribe(queue)


# Serve web dashboard at root
@app.get("/")
async def serve_dashboard():
    return FileResponse("web/index.html")


@app.get("/web/{path:path}")
async def serve_web_assets(path: str):
    file_path = Path("web") / path
    if file_path.is_file():
        return FileResponse(file_path)
    return FileResponse("web/index.html")
