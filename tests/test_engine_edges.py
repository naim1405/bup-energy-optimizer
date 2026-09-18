"""Engine paths the 10 public samples never reach.

Covers the infeasibility fallback, the post-processing reconcile step, and
the degenerate simultaneous charge/discharge case.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.engine.models import Directive, Scenario
from app.engine.optimizer import build_overlay, optimize
from app.engine.validate import make_validator, replay
from app.schemas.energy import OptimizeEnergyRequest
from app.services.energy_optimizer import to_scenario

CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "public_sample_cases.json").read_text()
)["cases"]
BY_ID = {c["id"]: c for c in CASES}
TOL = 0.01


def scenario_for(case_id: str, directives: list[Directive]) -> Scenario:
    request = OptimizeEnergyRequest(**BY_ID[case_id]["input"])
    base = to_scenario(request, [])
    return Scenario(
        scenario_id=base.scenario_id, hours=base.hours,
        battery=base.battery, directives=tuple(directives),
    )


# ---------------------------------------------------------------------------
# Infeasibility -> safe fallback, never an exception
# ---------------------------------------------------------------------------


def test_impossible_grid_cap_falls_back_instead_of_raising() -> None:
    scen = scenario_for("SAMPLE-05", [
        Directive(note_index=0, directive_type="max_grid_window",
                  hours=(18, 19, 20), max_grid_kwh=1.0),
    ])
    result = optimize(scen, validate_fn=make_validator())
    assert result.feasible is False
    assert any("infeasible" in v for v in result.violations)
    assert len(result.plan) == 24
    assert all(p.grid_kwh >= 0 and p.solar_used_kwh >= 0 for p in result.plan)
    assert all(p.battery_energy_after_kwh >= 0 for p in result.plan)


def test_impossible_reserve_is_clamped_by_the_model_bounds() -> None:
    """A reserve above capacity cannot be expressed; the engine must cope."""
    bat = OptimizeEnergyRequest(**BY_ID["SAMPLE-03"]["input"]).battery
    scen = scenario_for("SAMPLE-03", [
        Directive(note_index=0, directive_type="minimum_battery_reserve",
                  hours=(18,), minimum_energy_kwh=bat.capacity_kwh + 500),
    ])
    result = optimize(scen, validate_fn=make_validator())
    assert len(result.plan) == 24
    assert result.feasible is False


# ---------------------------------------------------------------------------
# Reconcile: self-consistency repair
# ---------------------------------------------------------------------------


def test_reconcile_repairs_a_corrupted_energy_column() -> None:
    from app.engine.optimizer import _reconcile

    scen = scenario_for("SAMPLE-02", [])
    clean = optimize(scen, validate_fn=make_validator())
    assert clean.violations == []

    ov = build_overlay(scen)
    broken = [
        replace(p, battery_energy_after_kwh=p.battery_energy_after_kwh + 0.4)
        for p in clean.plan
    ]
    _reconcile(broken, scen, ov)
    assert [p.battery_energy_after_kwh for p in broken] == \
           [p.battery_energy_after_kwh for p in clean.plan]
    assert replay(scen, ov, broken) == []


def test_reconcile_repairs_a_real_flow_imbalance() -> None:
    from app.engine.optimizer import _reconcile

    scen = scenario_for("SAMPLE-02", [])
    clean = optimize(scen, validate_fn=make_validator())
    ov = build_overlay(scen)

    target = next(p for p in clean.plan
                  if p.battery_action == "charge" and p.battery_kwh > 1.0)
    broken = [
        replace(p, battery_kwh=round(p.battery_kwh + 0.3, 6))
        if p.hour == target.hour else p for p in clean.plan
    ]
    _reconcile(broken, scen, ov)

    charged = sum(p.battery_kwh for p in broken if p.battery_action == "charge")
    discharged = sum(p.battery_kwh for p in broken if p.battery_action == "discharge")
    assert charged - discharged == pytest.approx(0.0, abs=1e-6)
    assert broken[-1].battery_energy_after_kwh == pytest.approx(
        scen.battery.initial_energy_kwh, abs=TOL
    )
    assert replay(scen, ov, broken) == []


# ---------------------------------------------------------------------------
# Degeneracy
# ---------------------------------------------------------------------------


def test_no_hour_both_charges_and_discharges() -> None:
    """The LP may emit a cost-neutral loop; post-processing must collapse it."""
    for case in CASES:
        scen = scenario_for(case["id"], [])
        result = optimize(scen, validate_fn=make_validator())
        for p in result.plan:
            assert p.battery_action in ("charge", "discharge", "idle")
            if p.battery_action == "idle":
                assert p.battery_kwh == 0


# ---------------------------------------------------------------------------
# Directive overlays, applied one at a time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_id, directive, check",
    [
        ("SAMPLE-02",
         Directive(0, "solar_reduction", (11, 12, 13), factor=0.0),
         lambda s, r: all(r.plan[h].solar_used_kwh <= TOL for h in (11, 12, 13))),
        ("SAMPLE-02",
         Directive(0, "no_charge_window", (2, 3, 4)),
         lambda s, r: all(r.plan[h].battery_action != "charge"
                          for h in (2, 3, 4))),
        ("SAMPLE-02",
         Directive(0, "no_discharge_window", (18, 19)),
         lambda s, r: all(r.plan[h].battery_action != "discharge"
                          for h in (18, 19))),
        # SAMPLE-05's own published cap: it binds and is feasible. On
        # SAMPLE-02 no cap between 165 and 170 kWh both binds and stays
        # feasible, so this directive needs its own fixture.
        ("SAMPLE-05",
         Directive(0, "max_grid_window", (18, 19, 20), max_grid_kwh=155.0),
         lambda s, r: all(r.plan[h].grid_kwh <= 155.0 + TOL
                          for h in (18, 19, 20))),
        ("SAMPLE-02",
         Directive(0, "minimum_battery_reserve", (18, 19, 20),
                   minimum_energy_kwh=150.0),
         lambda s, r: all(r.plan[h].battery_energy_after_kwh >= 150.0 - TOL
                          for h in (18, 19, 20))),
    ],
    ids=["solar_reduction", "no_charge", "no_discharge", "grid_cap", "reserve"],
)
def test_each_directive_is_honoured_in_isolation(case_id, directive, check) -> None:
    scen = scenario_for(case_id, [directive])
    result = optimize(scen, validate_fn=make_validator())
    assert result.feasible, result.violations
    assert check(scen, result), f"{directive.directive_type} was not honoured"
    assert result.violations == []


def test_grid_cap_actually_binds() -> None:
    """Guard against a cap that is merely stored but never enforced.

    Uses SAMPLE-01 hour 2, where the uncapped optimum imports 130 kWh. A 117
    kWh cap binds and is still feasible. The evening hours are no use here:
    they have no solar and the battery is already fully committed, so any cap
    below the uncapped peak is simply infeasible.
    """
    uncapped = optimize(scenario_for("SAMPLE-01", []),
                        validate_fn=make_validator())
    capped = optimize(
        scenario_for("SAMPLE-01", [
            Directive(0, "max_grid_window", (2,), max_grid_kwh=117.0),
        ]),
        validate_fn=make_validator(),
    )
    assert uncapped.plan[2].grid_kwh > 117.0, "fixture no longer exercises the cap"
    assert capped.feasible, capped.violations
    assert capped.plan[2].grid_kwh <= 117.0 + TOL
    # Capping a cheap charging hour can only cost more, never less.
    assert capped.total_cost_bdt >= uncapped.total_cost_bdt - TOL


def test_stacked_directives_are_all_honoured() -> None:
    scen = scenario_for("SAMPLE-07", [
        Directive(0, "minimum_battery_reserve", (18, 19, 20, 21),
                  minimum_energy_kwh=90.0),
        Directive(1, "max_grid_window", (19, 20), max_grid_kwh=180.0),
    ])
    result = optimize(scen, validate_fn=make_validator())
    assert result.feasible, result.violations
    for h in (18, 19, 20, 21):
        assert result.plan[h].battery_energy_after_kwh >= 90.0 - TOL
    for h in (19, 20):
        assert result.plan[h].grid_kwh <= 180.0 + TOL
    # And it must land on the published optimum for this case.
    assert result.total_cost_bdt == pytest.approx(
        float(BY_ID["SAMPLE-07"]["expected_output"]["total_cost_bdt"]), abs=TOL
    )


def test_overlapping_solar_reductions_stack_multiplicatively() -> None:
    scen = scenario_for("SAMPLE-01", [
        Directive(0, "solar_reduction", (12,), factor=0.5),
        Directive(1, "solar_reduction", (12,), factor=0.4),
    ])
    ov = build_overlay(scen)
    base = BY_ID["SAMPLE-01"]["input"]["hours"][12]["solar_kwh"]
    assert ov.effective_solar[12] == pytest.approx(base * 0.5 * 0.4, abs=1e-9)


def test_reserve_directive_cannot_lower_the_base_floor() -> None:
    scen = scenario_for("SAMPLE-01", [
        Directive(0, "minimum_battery_reserve", (5,), minimum_energy_kwh=1.0),
    ])
    ov = build_overlay(scen)
    base = BY_ID["SAMPLE-01"]["input"]["battery"]["minimum_energy_kwh"]
    assert ov.min_floor[5] == pytest.approx(base, abs=1e-9)


# ---------------------------------------------------------------------------
# Generalization: unseen scenarios must stay valid and optimal
# ---------------------------------------------------------------------------


def _random_case(rng, seed_tag: str) -> tuple[OptimizeEnergyRequest, list[Directive]]:
    cap = float(rng.choice([150, 200, 300, 500]))
    hours = [
        {
            "hour": h,
            "demand_kwh": round(float(rng.uniform(120, 260) * (
                1 + 0.5 * np.exp(-((h - 19) ** 2) / 6))), 1),
            "solar_kwh": round(max(0.0, float(
                rng.uniform(0.7, 1.3) * 170 * np.exp(-((h - 12.5) ** 2) / 9))), 1),
            "tariff_bdt_per_kwh": round(float(
                rng.uniform(5, 12) + (12 if 17 <= h <= 21 else 0)), 2),
        }
        for h in range(24)
    ]
    request = OptimizeEnergyRequest(
        scenario_id=seed_tag,
        operator_notes=["n"],
        hours=hours,
        battery={
            "capacity_kwh": cap,
            "initial_energy_kwh": round(cap * rng.uniform(0.3, 0.7), 1),
            "minimum_energy_kwh": round(cap * rng.uniform(0.05, 0.2), 1),
            "max_charge_kwh_per_hour": round(cap * rng.uniform(0.15, 0.3), 1),
            "max_discharge_kwh_per_hour": round(cap * rng.uniform(0.15, 0.3), 1),
        },
    )
    kinds = ["solar_reduction", "minimum_battery_reserve", "no_charge_window",
             "no_discharge_window", "max_grid_window"]
    directives = []
    for i in range(rng.randint(0, 3)):
        kind = rng.choice(kinds)
        start = rng.randint(0, 22)
        hs = tuple(range(start, min(24, start + rng.randint(1, 5))))
        kw: dict = {}
        if kind == "solar_reduction":
            kw["factor"] = rng.choice([0.0, 0.2, 0.5, 0.8])
        elif kind == "minimum_battery_reserve":
            kw["minimum_energy_kwh"] = round(cap * rng.uniform(0.1, 0.6), 1)
        elif kind == "max_grid_window":
            kw["max_grid_kwh"] = round(rng.uniform(80, 260), 1)
        directives.append(Directive(i, kind, hs, **kw))
    return request, directives


def test_unseen_scenarios_stay_valid_and_optimal() -> None:
    """The engine must generalize, not just reproduce the public numbers."""
    from scipy.optimize import linprog

    from app.engine.optimizer import _build

    rng = random.Random(20260918)
    feasible = relaxed = 0
    for i in range(120):
        request, directives = _random_case(rng, f"FUZZ-{i}")
        scen = Scenario(
            scenario_id=request.scenario_id,
            hours=to_scenario(request, []).hours,
            battery=to_scenario(request, []).battery,
            directives=tuple(directives),
        )
        result = optimize(scen, validate_fn=make_validator())
        assert len(result.plan) == 24
        if not result.feasible:
            relaxed += 1
            continue
        feasible += 1
        assert result.violations == [], result.violations[:3]

        # Cross-check optimality with a different LP algorithm.
        ov = build_overlay(scen)
        lp = _build(scen, ov, slack=False)
        alt = linprog(lp["c"], A_ub=lp["A_ub"], b_ub=lp["b_ub"], A_eq=lp["A_eq"],
                      b_eq=lp["b_eq"], bounds=lp["bounds"], method="highs-ipm")
        if alt.success and alt.x is not None:
            tariff = np.array(scen.tariff)
            assert float(tariff @ alt.x[0:24]) <= result.total_cost_bdt + TOL, (
                f"{scen.scenario_id}: interior point found a cheaper plan"
            )

    assert feasible > 60, f"only {feasible} feasible of 120; generator is broken"
