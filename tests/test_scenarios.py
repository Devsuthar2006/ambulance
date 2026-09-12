"""Tests for What-If scenario generation and comparative runner."""

from src.scenarios import ScenarioRunner


def test_scenario_generation():
    """Each scenario properly modifies fleet or incidents according to spec."""
    runner = ScenarioRunner()

    # Normal
    v_norm, i_norm = runner.generate_scenario("NORMAL")
    assert len(v_norm) == 20
    assert len(i_norm) in (50, 100)
    assert sum(1 for v in v_norm if v.idle) == 20

    # Vehicles down
    v_down, i_down = runner.generate_scenario("VEHICLES_DOWN")
    assert len(v_down) == 20
    assert sum(1 for v in v_down if v.idle) == 17

    # Demand surge
    v_surge, i_surge = runner.generate_scenario("DEMAND_SURGE")
    assert len(i_surge) in (80, 130)


def test_scenario_comparison_execution():
    """ScenarioRunner runs comparison across all 3 policies and returns valid metrics."""
    runner = ScenarioRunner()
    res = runner.run_comparison(scenario_name="NORMAL")

    assert "policies" in res
    assert "nearest" in res["policies"]
    assert "coverage" in res["policies"]
    assert "adaptive" in res["policies"]

    cov_res = res["policies"]["coverage"]
    assert cov_res["weighted_response_time"] > 0
    assert cov_res["assigned_count"] > 0
