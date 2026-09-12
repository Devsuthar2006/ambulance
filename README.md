# RESQAI — Real-Time Emergency Response Intelligence Platform

> **"We don't just optimize the current emergency. We protect the fleet's ability to respond to the next one."**

An end-to-end, causal emergency response intelligence platform built by **Devilal Suthar**. Extends the Emergency Fleet Assignment with Coverage Preservation engine with **AI Emergency Triage**, **Multi-Channel Voice Intake**, **Transparent Decision Explanations**, and a **Live Command-Center Dashboard**.

---

## ⚠️ Safety & Prototype Disclaimer

> **DEMO / SIMULATION — NOT A REAL EMERGENCY SERVICE**  
> RESQAI is an algorithmic research and hackathon simulation prototype designed for system evaluation and logistics optimization. It does **not** connect to real-world 911/112 public safety answering points, does **not** provide medical diagnoses, and is **not** certified for production life-safety operations.

---

## 1. Executive Summary & Problem Statement

### The Problem
Traditional emergency dispatchers operate greedily: they send the ambulance with the shortest travel time to the current incident. While locally optimal for one caller, this policy is **globally harmful**:
- If Ambulance A is 2 minutes away but is the **sole remaining idle unit in its quadrant**, dispatching it leaves that entire sector completely undefended.
- When a high-severity cardiac arrest or vehicle collision strikes that empty sector minutes later, response times explode because units must cross the entire city.

### The Solution: RESQAI
RESQAI couples **AI natural language understanding** with **provably causal, coverage-aware fleet optimization**:
1. **AI Understands the Caller**: Converts messy, panicky natural language into structured emergency categories (`TRAFFIC_ACCIDENT`, `CARDIAC`, etc.), priority tiers ($P_1, P_2, P_3$), extracted symptom evidence, and confidence scores.
2. **Optimization Engine Allocates the Fleet**: Uses an explainable cost function that balances Euclidean travel time against quadrant coverage preservation penalties, automatically softened by incident urgency.
3. **Causal Integrity**: Never inspects future calls, future vehicle completions, or future random states.

---

## 2. System Architecture

```
                                  VOICE INTAKE LAYER
                   ┌───────────────────────┬──────────────────────┐
                   │                       │                      │
             Real Phone Call        Browser Mic / WebRTC     Synthetic Call Feed
            (Twilio / Webhook)      (Web Speech API)         (100 Demo Calls)
                   │                       │                      │
                   └───────────────────────┼──────────────────────┘
                                           ▼
                                 UNSTRUCTURED TRANSCRIPT
                                           │
                                           ▼
                                    AI TRIAGE LAYER
                           ┌──────────────────────────────┐
                           │ TriageEngine (Interface)      │
                           ├──────────────────────────────┤
                           │ - LLMTriage (Gemini/OpenAI)  │
                           │ - MockTriage (Deterministic) │
                           └───────────────┬──────────────┘
                                           │
                                           ▼
                           STRUCTURED INCIDENT MODEL
                           (P1/P2/P3, Type, (x, y), Conf)
                                           │
                                           ▼
                   ┌──────────────────────────────────────────────┐
                   │              DISPATCH ENGINE                 │
                   │  - Nearest (Greedy Baseline)                 │
                   │  - Coverage-Aware (AI-01 Core Algorithm)     │
                   │  - Adaptive (Fleet Scarcity + Demand Risk)   │
                   └───────────────┬──────────────────────────────┘
                                   │
                   ┌───────────────┴──────────────────────────────┐
                   ▼                                              ▼
       DISPATCH EXPLANATION                          LIVE SIMULATOR / EVENT BUS
       (Candidate costs, delta,                      (Strictly Causal 5-Step Loop)
        coverage impact explanation)                              │
                                                                  ▼
                                                      FASTAPI WEBSOCKET STREAM
                                                                  │
                                                                  ▼
                                                   COMMAND-CENTER DASHBOARD
                                           (60 FPS Map, Live Coverage, Event Feed,
                                            Inspector, What-If Scenario Runner)
```

---

## 3. Core Principles & AI Positioning

### AI Positioning
> **AI is NOT responsible for choosing the ambulance.**  
> AI understands the emergency call. The constrained mathematical optimization engine decides how to allocate the fleet.

This makes RESQAI **transparent**, **explainable**, and **technically defensible**.

---

## 4. Dispatch Algorithms & Mathematical Formulation

### Policy 1: Nearest (Baseline)
$$\text{cost}_{\text{nearest}}(v, \text{inc}) = \text{EuclideanDist}(v, \text{inc})$$

### Policy 2: Coverage-Aware (Original AI-01 Algorithm)
$$\text{cost}_{\text{coverage}}(v, \text{inc}) = \text{dist}(v, \text{inc}) + \text{COVERAGE\_PENALTY} \times \frac{\mathbb{I}[\text{is\_last\_idle\_in\_quadrant}(v)]}{\text{inc.weight}}$$
- Routine $P_1$ calls ($w=1$) pay the full penalty ($+20.0$), nudging the system to assign from an adjacent surplus sector.
- Critical $P_3$ calls ($w=7$) soften the penalty by $\frac{1}{7}$ ($+2.86$), ensuring life-or-death emergencies always get the closest ambulance.

### Policy 3: Adaptive (Scarcity-Aware Extension)
$$\text{scarcity\_factor} = \max\left(0.0, \frac{\text{TOTAL\_VEHICLES}/2 - \text{total\_idle}}{\text{TOTAL\_VEHICLES}/2}\right)$$
$$\text{cost}_{\text{adaptive}}(v, \text{inc}) = \text{cost}_{\text{coverage}}(v, \text{inc}) + \frac{\text{SCARCITY\_PENALTY} \cdot \text{scarcity\_factor} \cdot \mathbb{I}[\text{cross\_quadrant}]}{\text{inc.weight}}$$
Under city-wide fleet exhaustion, cross-quadrant trips for low-priority calls are penalized to keep regional reserves ready for critical trauma.

---

## 5. System Modes

| Mode | Purpose | Description |
|---|---|---|
| **Mode A: Live Voice** | Interactive Demo | Speak via browser mic or phone call $\to$ real-time AI triage $\to$ live fleet dispatch $\to$ animated map response. |
| **Mode B: 100-Call Sim** | Control Room Simulation | Streams 100 calls with realistic transcripts through triage, dispatch, vehicle transit, on-scene service, and resolution. |
| **Mode C: Official Benchmark** | Rigorous Verification | 100% deterministic, NumPy PCG64 seed `20260911`, zero external dependencies, 20 vehicles, 100 incidents. |
| **Mode D: What-If Stress Tests** | Resilience Analysis | 5 judge scenarios: Normal, 3 Units Down, +30% Surge, P3 Crisis (50%), and Q2 Sector Disaster. |

---

## 6. Installation & Configuration

### Prerequisites
- Python 3.11+ (tested on Python 3.14)
- macOS / Linux / Windows WSL

### Quick Start
```bash
# Clone and enter directory
cd ambulanceD

# Setup virtual environment and install dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Run complete test suite (27 tests)
.venv/bin/pytest -v tests/

# Execute one-command benchmark
./run.sh
```

### Configuration (`.env`)
Copy the provided template:
```bash
cp .env.example .env
```
Key parameters:
- `SEED`: RNG seed (default: `20260911`)
- `COVERAGE_PENALTY`: Quadrant preservation penalty (default: `20.0`)
- `SCARCITY_PENALTY`: Fleet scarcity penalty (default: `15.0`)
- `TRIAGE_PROVIDER`: `"mock"` (default offline) or `"llm"` (OpenAI/Gemini)
- `AI_API_KEY`: Optional API key for LLM triage
- `PORT`: Web server port (default: `8000`)

---

## 7. Running the Platform

### Option A: Launch the Command-Center Web Dashboard
```bash
.venv/bin/uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser:
- 🚑 **60 FPS Animated Map**: Smooth sub-minute vehicle transit with heading rotation and flashing sirens.
- 🎙️ **Voice Intake Station**: Click call presets or speak via microphone.
- 🧠 **Explainable Decision Panel**: View evaluated candidates and transparent plain-English dispatch reasoning.
- 📊 **Sector Coverage Gauges**: Live Q0-Q3 availability badges with outage alert pulses.
- ⚡ **What-If Scenario Runner**: Execute stress tests with one click.

### Option B: Terminal 100-Call Control Room Demo
```bash
.venv/bin/python run_demo.py --calls 100 --speed 2
```

### Option C: Run Headless Official Benchmark
```bash
.venv/bin/python -m src.report --seed 20260911 --policy coverage
```

---

## 8. Benchmark Results (Official Seed 20260911)

| Metric | Target | Result | Status |
|---|---|---|---|
| **Priority-Weighted Response Time** | Lower is better | **57.1410 min** | Verified |
| **Priority-3 (Critical) Response Time** | Lower is better | **50.4139 min** | Verified |
| **Coverage Outage Minutes** | Lower is better | **115 / 121 min** | Verified |
| **Valid Online Assignment Rate** | 100% | **100%** | Passed |
| **Simulation Runtime** | Real-time efficiency | **< 2 ms** (0.0019s) | High Speed |
| **Benchmark Reproducibility** | Zero variance | **Bit-identical** | Regression Passed |

---

## 9. Testing & Quality Assurance

The test suite includes **27 automated tests** across 6 modules:
- `tests/test_benchmark_reproducibility.py`: Asserts official benchmark produces exact bit-for-bit results.
- `tests/test_triage.py`: Tests AI triage schema, category detection, confidence scoring, and mock fallback.
- `tests/test_adaptive_dispatcher.py`: Tests explainability output and scarcity penalty scaling.
- `tests/test_scenarios.py`: Tests What-If scenario generation without mutating baseline generators.
- `tests/test_api.py`: Tests REST endpoints (`/api/health`, `/api/call/voice`, `/api/scenarios/run`).
- `tests/test_dispatcher.py`, `test_generator.py`, `test_simulator.py`, `test_metrics.py`: Complete coverage of core mechanics.

Run all tests:
```bash
.venv/bin/pytest -v tests/
```

---

## 10. Future-Ready Architecture Roadmap

Designed with clean interfaces to support:
- 🗺️ **Real GPS & OpenStreetMap Routing**: Replace Euclidean distance with Valhalla / OSRM travel matrices.
- 🚦 **Traffic-Aware Dynamic Speed**: Time-of-day congestion factors.
- 🏥 **Hospital Capacity Optimization**: Multi-objective routing to facilities with available trauma beds.
- 🔄 **Proactive Repositioning**: Automated relocation of idle units into predicted high-risk zones.
- 🚒 **Multi-Agency Incident Coordination**: Police, Fire, and EMS triage dispatch.

---

## 11. Judge Demo Walkthrough Script

1. **Open Dashboard**: Open `http://localhost:8000`. Show the 100×100 grid, 4 quadrants, and 20 ambulances.
2. **Judge Voice Call**: Click "🚴 Serious Bike Crash (P3)" or speak into the microphone: *"There has been a serious bike accident. The person is unconscious and bleeding badly."*
3. **AI Triage**: Show the AI Triage card immediately classify: `TRAFFIC_ACCIDENT`, `P3 CRITICAL`, `98% Confidence`, with extracted evidence.
4. **Explainable Decision**: Point out the candidate evaluation table:
   > *"Notice Vehicle 12 was selected instead of closer Vehicle 7. Vehicle 7 was 0.9m closer, but it is the sole idle ambulance in Quadrant 2. Dispatching Vehicle 12 preserved sector safety."*
5. **Watch the Map**: Watch Vehicle 12 visibly glide with flashing sirens to $(72, 31)$, switch to on-scene service (8 min countdown), and become available again.
6. **Stress Test**: Click "3 Units Down" or "+30% Surge" to show real-time A/B comparative metrics across Nearest, Coverage-Aware, and Adaptive policies.
