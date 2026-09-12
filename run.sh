#!/usr/bin/env bash
set -e

# ==============================================================================
# RESQAI — Real-Time Emergency Response Intelligence Platform
# One-command execution deliverable
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "  RESQAI — REAL-TIME EMERGENCY RESPONSE INTELLIGENCE PLATFORM"
echo "  DEMO / SIMULATION — NOT A REAL EMERGENCY SERVICE"
echo "================================================================================"

# 1. Ensure virtual environment exists
if [ ! -d ".venv" ]; then
    echo "[1/5] Creating Python virtual environment (.venv)..."
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip > /dev/null 2>&1
    .venv/bin/pip install -r requirements.txt > /dev/null 2>&1
else
    echo "[1/5] Python virtual environment ready."
fi

# 2. Run automated test suite (27 unit tests)
echo "[2/5] Running automated test suite (reproducibility, causality, triage, API)..."
.venv/bin/pytest -q tests/

# 3. Run primary benchmark simulation (Mode C)
echo "[3/5] Executing official benchmark simulation (seed: 20260911)..."
.venv/bin/python -m src.report \
    --seed 20260911 \
    --policy coverage \
    --penalty 20.0 \
    --csv-out outputs/run_log.csv \
    --json-out outputs/summary.json

# 4. Run parameter sweep comparison
echo "[4/5] Running A/B parameter sweep comparison..."
.venv/bin/python -m src.report --seed 20260911 --sweep

# 5. Generate visualization artifacts and rich playback traces
echo "[5/5] Generating visual demonstration plots and playback traces..."
.venv/bin/python -m src.visualize
.venv/bin/python -m src.export_playback

echo ""
echo "================================================================================"
echo "  Benchmark verification complete!"
echo "  Artifacts generated in outputs/:"
echo "    - outputs/run_log.csv"
echo "    - outputs/summary.json"
echo "    - outputs/sweep_results.json"
echo "    - outputs/initial_map.png"
echo "    - outputs/coverage_timeline.png"
echo "    - outputs/playback_data.json"
echo ""
echo "  To launch the live RESQAI Command Center web platform:"
echo "    .venv/bin/uvicorn src.api.server:app --host 0.0.0.0 --port 8000"
echo "  To run the 100-call interactive demo in terminal:"
echo "    .venv/bin/python run_demo.py --calls 100"
echo "================================================================================"

if [ "$1" == "--server" ]; then
    echo "Starting RESQAI Command Center on http://localhost:8000 ..."
    .venv/bin/uvicorn src.api.server:app --host 0.0.0.0 --port 8000
fi
