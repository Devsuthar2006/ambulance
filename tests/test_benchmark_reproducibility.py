"""Critical regression test: asserts official benchmark scores are bit-for-bit identical."""

import pytest
from src.config import DEFAULT_SEED
from src.generator import generate_world
from src.dispatcher import NaiveDispatcher, CoverageAwareDispatcher
from src.simulator import Simulator
from src.metrics import compute_metrics


def test_official_benchmark_reproducibility_regression():
    """Verify that all platform extensions have zero effect on official benchmark metrics."""
    vehicles, incidents = generate_world(seed=DEFAULT_SEED, total_incidents=100)

    # 1. Run Coverage-Aware (AI-01 Core Algorithm)
    sim_cov = Simulator(dispatcher=CoverageAwareDispatcher(coverage_penalty=20.0))
    res_cov = sim_cov.run(vehicles, incidents)
    metrics_cov = compute_metrics(res_cov, seed=DEFAULT_SEED)

    assert pytest.approx(metrics_cov.priority_weighted_response_time, rel=1e-4) == 57.1410
    assert pytest.approx(metrics_cov.priority3_response_time, rel=1e-4) == 50.4139
    assert metrics_cov.coverage_outage_minutes == 115
    assert metrics_cov.assigned_incidents == 61
    assert metrics_cov.total_incidents == 100

    # 2. Run Naive Baseline
    sim_naive = Simulator(dispatcher=NaiveDispatcher())
    res_naive = sim_naive.run(vehicles, incidents)
    metrics_naive = compute_metrics(res_naive, seed=DEFAULT_SEED)

    assert pytest.approx(metrics_naive.priority_weighted_response_time, rel=1e-4) == 57.1410
    assert pytest.approx(metrics_naive.priority3_response_time, rel=1e-4) == 50.4139
    assert metrics_naive.coverage_outage_minutes == 115
