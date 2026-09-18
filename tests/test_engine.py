"""Engine correctness, driven through the pydantic -> engine adapter.

Each public sample case is parsed by the real request model, converted by
the real adapter, optimized by the real engine, and checked three ways:

  * the schedule passes an independent replay of every GridWise rule;
  * the cost equals the published (organizer-optimal) cost exactly;
  * the reported totals match values recalculated from the returned plan.

The LLM is not involved: each case's published directive_interpretation is
fed in as if a model had produced it. That is what isolates the math.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.engine.optimizer import optimize
from app.engine.validate import make_validator, replay
from app.schemas.energy import DirectiveInterpretation, OptimizeEnergyRequest
from app.services.energy_optimizer import to_response, to_scenario

FIXTURE = Path(__file__).parent / "fixtures" / "public_sample_cases.json"
TOL = 0.01


def load_cases() -> list[dict]:
    return json.loads(FIXTURE.read_text())["cases"]


@pytest.fixture(scope="module")
def cases() -> list[dict]:
    return load_cases()


def ids() -> list[str]:
    return [c["id"] for c in load_cases()]


def build(case: dict):
    request = OptimizeEnergyRequest(**case["input"])
    entries = [
        DirectiveInterpretation(**e)
        for e in case["expected_output"]["directive_interpretation"]
    ]
    return request, entries, to_scenario(request, entries)


@pytest.mark.parametrize("case", load_cases(), ids=ids())
def test_engine_reaches_the_published_optimum(case: dict) -> None:
    _, _, scenario = build(case)
    result = optimize(scenario, validate_fn=make_validator())
    expected = float(case["expected_output"]["total_cost_bdt"])
    assert result.feasible, result.violations
    assert result.total_cost_bdt == pytest.approx(expected, abs=TOL)


@pytest.mark.parametrize("case", load_cases(), ids=ids())
def test_engine_output_passes_independent_replay(case: dict) -> None:
    _, _, scenario = build(case)
    result = optimize(scenario, validate_fn=make_validator())
    from app.engine.optimizer import build_overlay

    assert replay(scenario, build_overlay(scenario), result.plan) == []


@pytest.mark.parametrize("case", load_cases(), ids=ids())
def test_response_totals_match_the_plan(case: dict) -> None:
    request, entries, scenario = build(case)
    result = optimize(scenario, validate_fn=make_validator())
    resp = to_response(request, entries, result)

    tariff = {h.hour: h.tariff_bdt_per_kwh for h in request.hours}
    plan = resp.hourly_plan
    assert resp.total_grid_kwh == pytest.approx(
        sum(p.grid_kwh for p in plan), abs=TOL
    )
    assert resp.total_cost_bdt == pytest.approx(
        sum(p.grid_kwh * tariff[p.hour] for p in plan), abs=TOL
    )
    assert resp.peak_grid_kwh == pytest.approx(
        max(p.grid_kwh for p in plan), abs=TOL
    )
    # Optimizer internals and the assembled response must agree.
    assert resp.total_cost_bdt == pytest.approx(result.total_cost_bdt, abs=TOL)


@pytest.mark.parametrize("case", load_cases(), ids=ids())
def test_interpretation_round_trips_through_the_adapter(case: dict) -> None:
    """The structured_adjustment we report must be the one we optimized with."""
    _, entries, scenario = build(case)
    for entry, directive in zip(entries, scenario.directives):
        assert directive.directive_type == entry.directive_type.value
        assert directive.applies == entry.applies
        if entry.structured_adjustment is None:
            assert directive.adjustment() is None
        else:
            assert directive.adjustment() == entry.structured_adjustment.model_dump()


@pytest.mark.parametrize("case", load_cases(), ids=ids())
def test_structured_adjustment_has_no_null_riders(case: dict) -> None:
    """Regression guard: each adjustment must carry only its own type's keys.

    A reserve must serialize as {"hours": [...], "minimum_energy_kwh": n},
    not with "factor": null and "max_grid_kwh": null riding along.
    """
    expected_keys = {
        "solar_reduction": {"hours", "factor"},
        "minimum_battery_reserve": {"hours", "minimum_energy_kwh"},
        "no_charge_window": {"hours"},
        "no_discharge_window": {"hours"},
        "max_grid_window": {"hours", "max_grid_kwh"},
    }
    _, entries, _ = build(case)
    for e in entries:
        dumped = e.model_dump()["structured_adjustment"]
        if e.directive_type.value == "no_op":
            assert dumped is None
        else:
            assert set(dumped) == expected_keys[e.directive_type.value]
            assert None not in dumped.values()


@pytest.mark.parametrize("case", load_cases(), ids=ids())
def test_battery_neutrality_and_rate_limits(case: dict) -> None:
    _, _, scenario = build(case)
    result = optimize(scenario, validate_fn=make_validator())
    bat = scenario.battery
    assert result.plan[-1].battery_energy_after_kwh == pytest.approx(
        bat.initial_energy_kwh, abs=TOL
    )
    for p in result.plan:
        if p.battery_action == "charge":
            assert p.battery_kwh <= bat.max_charge_kwh_per_hour + TOL
        elif p.battery_action == "discharge":
            assert p.battery_kwh <= bat.max_discharge_kwh_per_hour + TOL
        else:
            assert p.battery_kwh == 0


def test_directives_actually_change_the_schedule(cases: list[dict]) -> None:
    """Guard against an engine that silently ignores what it is given."""
    case = next(c for c in cases if c["id"] == "SAMPLE-03")
    request = OptimizeEnergyRequest(**case["input"])

    from app.engine.models import Scenario

    bare = Scenario(
        scenario_id=request.scenario_id,
        hours=to_scenario(request, []).hours,
        battery=to_scenario(request, []).battery,
    )
    _, entries, with_directive = build(case)

    a = optimize(bare, validate_fn=make_validator())
    b = optimize(with_directive, validate_fn=make_validator())
    assert a.total_cost_bdt < b.total_cost_bdt, (
        "a reserve directive can only restrict the schedule, so honoring it "
        "must not make the plan cheaper"
    )
    # ...and the reserve must actually be respected in the affected hours.
    for h in entries[0].structured_adjustment.hours:
        assert b.plan[h].battery_energy_after_kwh >= (
            entries[0].structured_adjustment.minimum_energy_kwh - TOL
        )
