"""Visualization tools for simulation trajectories, coverage, and trade-offs."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.config import DEFAULT_SEED
from src.generator import generate_world
from src.dispatcher import NaiveDispatcher, CoverageAwareDispatcher
from src.simulator import Simulator


def plot_coverage_comparison(seed: int = DEFAULT_SEED, output_path: str = "outputs/coverage_timeline.png") -> None:
    """Plot minute-by-minute quadrant idle counts for Naive vs Coverage-Aware policies."""
    vehicles_init, incidents_init = generate_world(seed=seed)

    # 1. Run Naive
    sim_naive = Simulator(dispatcher=NaiveDispatcher())
    res_naive = sim_naive.run(vehicles_init, incidents_init)

    # 2. Run CoverageAware (penalty=20)
    sim_cov = Simulator(dispatcher=CoverageAwareDispatcher(coverage_penalty=20.0))
    res_cov = sim_cov.run(vehicles_init, incidents_init)

    minutes = [h[0] for h in res_naive.coverage_history]
    min_idle_naive = [min(h[1]) for h in res_naive.coverage_history]
    min_idle_cov = [min(h[1]) for h in res_cov.coverage_history]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(minutes, min_idle_naive, label=f"Naive Policy (Outages: {res_naive.coverage_outage_minutes} mins)", color="#d9534f", lw=2)
    ax.plot(minutes, min_idle_cov, label=f"Coverage-Aware (Outages: {res_cov.coverage_outage_minutes} mins)", color="#2e6da4", lw=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7, label="Coverage Outage Boundary (= 0 idle)")

    ax.set_title("Minimum Idle Vehicles Across All Quadrants (t = 0 to 120)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Simulation Minute (t)", fontsize=11)
    ax.set_ylabel("Min Idle Count in Any Quadrant", fontsize=11)
    ax.set_ylim(-0.5, 6.0)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved coverage timeline plot to {output_path}")


def plot_map_distribution(seed: int = DEFAULT_SEED, output_path: str = "outputs/initial_map.png") -> None:
    """Plot initial vehicle locations and incident locations across quadrants."""
    vehicles, incidents = generate_world(seed=seed)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    
    # Quadrant lines
    ax.axvline(50, color="black", linestyle="--", lw=1.5, alpha=0.7)
    ax.axhline(50, color="black", linestyle="--", lw=1.5, alpha=0.7)

    # Quadrant labels
    ax.text(25, 25, "Quadrant 0\n(Lower-Left)", ha="center", va="center", color="gray", fontsize=11, alpha=0.5)
    ax.text(75, 25, "Quadrant 1\n(Lower-Right)", ha="center", va="center", color="gray", fontsize=11, alpha=0.5)
    ax.text(25, 75, "Quadrant 2\n(Upper-Left)", ha="center", va="center", color="gray", fontsize=11, alpha=0.5)
    ax.text(75, 75, "Quadrant 3\n(Upper-Right)", ha="center", va="center", color="gray", fontsize=11, alpha=0.5)

    # Vehicles
    vx = [v.x for v in vehicles]
    vy = [v.y for v in vehicles]
    ax.scatter(vx, vy, c="#28a745", marker="^", s=90, edgecolors="black", label="Initial Vehicles (5/quadrant)", zorder=5)

    # Incidents by priority
    for prio, col, sz in [(1, "#6c757d", 25), (2, "#fd7e14", 45), (3, "#dc3545", 80)]:
        ix = [inc.x for inc in incidents if inc.priority == prio]
        iy = [inc.y for inc in incidents if inc.priority == prio]
        ax.scatter(ix, iy, c=col, marker="o", s=sz, alpha=0.75, label=f"Priority {prio} Incident ({len(ix)})", zorder=4)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("Region Map: 20 Initial Vehicles & 100 Incident Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), framealpha=0.9)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved initial map plot to {output_path}")


if __name__ == "__main__":
    plot_map_distribution()
    plot_coverage_comparison()
