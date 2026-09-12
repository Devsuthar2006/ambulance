"""What-If stress test scenarios for judge evaluation and resilience analysis."""

import copy
import numpy as np
from src.config import DEFAULT_SEED
from src.generator import generate_world
from src.models import Vehicle, Incident
from src.dispatcher import get_dispatcher
from src.simulator import Simulator
from src.metrics import compute_metrics
from src.traffic import get_default_traffic_model


class ScenarioRunner:
    """Runs configurable What-If scenarios without altering official benchmark state."""

    SCENARIOS = {
        "NORMAL": "Standard benchmark fleet (20 vehicles, 100 incidents).",
        "TRAFFIC_GRIDLOCK": "Downtown arterial gridlock: 3.5x congestion in Q3 (Market & SOMA). Evaluates traffic bypass on P3 calls.",
        "VEHICLES_DOWN": "Fleet breakdown: 3 ambulances unavailable (17 active).",
        "DEMAND_SURGE": "30% demand surge: 130 incidents across the region.",
        "P3_SURGE": "Mass casualty event: 50% of calls are Priority 3 Critical.",
        "HIGH_RISK_QUADRANT": "Localized disaster in Q2 (Upper-Left sector crisis).",
    }

    def generate_scenario(
        self,
        scenario_name: str,
        seed: int = DEFAULT_SEED,
    ) -> tuple[list[Vehicle], list[Incident]]:
        """Generate fleet and incidents modified specifically for the scenario."""
        name = scenario_name.upper().strip()
        vehicles, incidents = generate_world(seed=seed)

        if name == "NORMAL":
            return vehicles, incidents

        elif name == "VEHICLES_DOWN":
            # Disable 3 vehicles across different quadrants
            disabled_ids = {0, 5, 10}
            for v in vehicles:
                if v.id in disabled_ids:
                    v.idle = False
                    v.busy_until = 9999.0
            return vehicles, incidents

        elif name == "DEMAND_SURGE":
            # Add 30 extra incidents (130 total)
            rng = np.random.Generator(np.random.PCG64(seed + 999))
            extra_incidents = []
            for i in range(100, 130):
                extra_incidents.append(
                    Incident(
                        id=i,
                        arrival_minute=int(rng.integers(0, 60)),
                        x=float(rng.uniform(0.0, 100.0)),
                        y=float(rng.uniform(0.0, 100.0)),
                        priority=int(rng.choice([1, 2, 3], p=[0.60, 0.30, 0.10])),
                        source="SIMULATION",
                    )
                )
            all_inc = incidents + extra_incidents
            all_inc.sort(key=lambda inc: (inc.arrival_minute, inc.id))
            return vehicles, all_inc

        elif name == "P3_SURGE":
            # 50% Priority 3 Critical calls
            rng = np.random.Generator(np.random.PCG64(seed + 888))
            for inc in incidents:
                inc.priority = int(rng.choice([1, 2, 3], p=[0.25, 0.25, 0.50]))
                inc.__post_init__()
            return vehicles, incidents

        elif name == "HIGH_RISK_QUADRANT":
            # Concentrate incidents between minute 10 and 35 into Quadrant 2 (x < 50, y >= 50)
            rng = np.random.Generator(np.random.PCG64(seed + 777))
            for inc in incidents:
                if 10 <= inc.arrival_minute <= 35:
                    inc.x = float(rng.uniform(0.0, 50.0))
                    inc.y = float(rng.uniform(50.0, 100.0))
            return vehicles, incidents

        elif name == "TRAFFIC_GRIDLOCK":
            # Concentrate emergencies around the downtown chokepoints
            rng = np.random.Generator(np.random.PCG64(seed + 666))
            for inc in incidents:
                if rng.random() < 0.45:
                    inc.x = float(rng.uniform(60.0, 85.0))
                    inc.y = float(rng.uniform(55.0, 75.0))
            return vehicles, incidents

        else:
            raise ValueError(f"Unknown scenario '{scenario_name}'. Allowed: {list(self.SCENARIOS.keys())}")

    def run_comparison(
        self,
        scenario_name: str = "NORMAL",
        seed: int = DEFAULT_SEED,
    ) -> dict:
        """Run all 3 dispatchers (nearest, coverage, adaptive) on the specified scenario."""
        v_base, i_base = self.generate_scenario(scenario_name, seed=seed)
        traffic_model = get_default_traffic_model() if scenario_name.upper() == "TRAFFIC_GRIDLOCK" else None

        policies = [
            ("nearest", "Nearest (Baseline)"),
            ("coverage", "Coverage-Aware (AI-01)"),
            ("adaptive", "Adaptive (Scarcity-Aware)"),
        ]

        results = {}
        for pol_id, pol_label in policies:
            disp = get_dispatcher(pol_id, traffic_model=traffic_model)
            sim = Simulator(dispatcher=disp, traffic_model=traffic_model)
            res = sim.run(copy.deepcopy(v_base), copy.deepcopy(i_base))
            metrics = compute_metrics(res, seed=seed)

            results[pol_id] = {
                "label": pol_label,
                "weighted_response_time": round(float(metrics.priority_weighted_response_time), 2),
                "p3_response_time": round(float(metrics.priority3_response_time), 2),
                "coverage_outages": int(metrics.coverage_outage_minutes),
                "assigned_count": int(metrics.assigned_incidents),
                "total_incidents": int(metrics.total_incidents),
                "runtime_seconds": round(float(metrics.runtime_seconds), 4),
            }

        return {
            "scenario": scenario_name,
            "description": self.SCENARIOS.get(scenario_name, ""),
            "policies": results,
        }
