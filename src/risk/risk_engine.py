"""Regional risk evaluation engine using only currently observed simulation state."""

from dataclasses import dataclass
import numpy as np
from src.models import Vehicle, Incident, quadrant_of
from src.config import NUM_QUADRANTS, NUM_VEHICLES_PER_QUADRANT


@dataclass
class QuadrantRiskReport:
    """Live risk diagnostic for a geographic quadrant."""
    quadrant: int
    name: str
    risk_score: int  # 0 to 100
    status: str      # "HEALTHY" | "WARNING" | "CRITICAL"
    idle_vehicles: int
    busy_vehicles: int
    demand_pressure: str   # "LOW" | "MEDIUM" | "HIGH"
    coverage_pressure: str # "LOW" | "MEDIUM" | "HIGH"
    active_p3_count: int


class RiskEngine:
    """Evaluates quadrant vulnerability and recommends proactive fleet balancing."""

    QUAD_NAMES = {
        0: "Q0 (Lower-Left)",
        1: "Q1 (Lower-Right)",
        2: "Q2 (Upper-Left)",
        3: "Q3 (Upper-Right)",
    }

    def evaluate_quadrants(
        self,
        vehicles: list[Vehicle],
        waiting_queue: list[Incident],
        recent_incidents: list[Incident],
    ) -> dict[int, QuadrantRiskReport]:
        """Compute 0-100 risk score per quadrant without looking into the future."""
        reports = {}

        # 1. Idle and busy counts
        idle_counts = np.zeros(NUM_QUADRANTS, dtype=int)
        busy_counts = np.zeros(NUM_QUADRANTS, dtype=int)
        for v in vehicles:
            if v.idle:
                idle_counts[v.quadrant] += 1
            else:
                busy_counts[v.quadrant] += 1

        # 2. Demand and queue pressure
        queued_per_quad = np.zeros(NUM_QUADRANTS, dtype=int)
        p3_per_quad = np.zeros(NUM_QUADRANTS, dtype=int)
        for inc in waiting_queue:
            q = quadrant_of(inc.x, inc.y)
            queued_per_quad[q] += 1
            if inc.priority == 3:
                p3_per_quad[q] += 1

        # 3. Calculate scores
        for q in range(NUM_QUADRANTS):
            idle = idle_counts[q]
            busy = busy_counts[q]
            queued = queued_per_quad[q]
            p3 = p3_per_quad[q]

            # Coverage pressure component (0 - 50 pts)
            if idle == 0:
                cov_pts = 50
                cov_press = "HIGH"
            elif idle == 1:
                cov_pts = 35
                cov_press = "HIGH"
            elif idle == 2:
                cov_pts = 20
                cov_press = "MEDIUM"
            else:
                cov_pts = 5
                cov_press = "LOW"

            # Demand pressure component (0 - 30 pts)
            demand_pts = min(30, queued * 10 + len(recent_incidents) * 2)
            demand_press = "HIGH" if demand_pts >= 20 else ("MEDIUM" if demand_pts >= 10 else "LOW")

            # Urgency pressure (0 - 20 pts)
            urgency_pts = min(20, p3 * 10)

            total_risk = min(100, cov_pts + demand_pts + urgency_pts)

            if total_risk >= 70 or idle == 0:
                status = "CRITICAL"
            elif total_risk >= 40 or idle <= 1:
                status = "WARNING"
            else:
                status = "HEALTHY"

            reports[q] = QuadrantRiskReport(
                quadrant=q,
                name=self.QUAD_NAMES[q],
                risk_score=int(total_risk),
                status=status,
                idle_vehicles=int(idle),
                busy_vehicles=int(busy),
                demand_pressure=demand_press,
                coverage_pressure=cov_press,
                active_p3_count=int(p3),
            )

        return reports

    def recommend_reposition(
        self,
        freed_vehicle: Vehicle,
        reports: dict[int, QuadrantRiskReport],
    ) -> dict | None:
        """Optional future-ready fleet rebalancing: recommend relocating an idle unit if another quadrant is in outage."""
        current_q = freed_vehicle.quadrant
        current_report = reports[current_q]

        # Only consider relocating if current quadrant has surplus (>= 2 idle)
        if current_report.idle_vehicles < 2:
            return None

        # Find most vulnerable quadrant in CRITICAL status
        worst_q = None
        worst_score = 0
        for q, rep in reports.items():
            if q != current_q and rep.status == "CRITICAL" and rep.risk_score > worst_score:
                worst_score = rep.risk_score
                worst_q = q

        if worst_q is not None and worst_score >= 70:
            return {
                "vehicle_id": freed_vehicle.id,
                "from_quadrant": current_q,
                "to_quadrant": worst_q,
                "reason": f"Quadrant Q{worst_q} in CRITICAL risk (score: {worst_score}/100) with 0 idle units.",
            }

        return None
