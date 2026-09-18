"""End-to-end integration: HTTP request -> engine -> HTTP response.

The interpreter is still the no_op stub, so these tests check the plumbing
rather than language understanding. ``test_directive_flows_end_to_end``
monkeypatches the stub to prove a real directive travels the whole path
(LLM-shaped output -> to_api_model -> engine Directive -> LP -> response),
which is the part that has to keep working when the model is wired in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.llm import NoteInterpretation
from app.services import energy_optimizer

client = TestClient(app)
FIXTURE = Path(__file__).parent / "fixtures" / "public_sample_cases.json"
TOL = 0.01


@pytest.fixture(scope="module")
def cases() -> dict[str, dict]:
    return {c["id"]: c for c in json.loads(FIXTURE.read_text())["cases"]}


def idle_baseline_cost(case: dict) -> float:
    """Cost of doing nothing: solar up to demand, the rest from the grid."""
    return sum(
        max(0.0, h["demand_kwh"] - h["solar_kwh"]) * h["tariff_bdt_per_kwh"]
        for h in case["input"]["hours"]
    )


# ---------------------------------------------------------------------------
# The stubbed path: valid and optimized for the base scenario
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_id", ["SAMPLE-01", "SAMPLE-05", "SAMPLE-09"])
def test_endpoint_returns_a_valid_schedule(cases, case_id) -> None:
    case = cases[case_id]
    r = client.post("/optimize-energy", json=case["input"])
    assert r.status_code == 200
    body = r.json()
    plan = {p["hour"]: p for p in body["hourly_plan"]}

    initial = case["input"]["battery"]["initial_energy_kwh"]
    e = initial
    for h in case["input"]["hours"]:
        p = plan[h["hour"]]
        chg = p["battery_kwh"] if p["battery_action"] == "charge" else 0.0
        dis = p["battery_kwh"] if p["battery_action"] == "discharge" else 0.0
        # Energy balance (9.5)
        assert p["grid_kwh"] + p["solar_used_kwh"] + dis == pytest.approx(
            h["demand_kwh"] + chg, abs=TOL
        )
        # Solar within availability (9.4); no directives, so base solar.
        assert p["solar_used_kwh"] <= h["solar_kwh"] + TOL
        assert p["grid_kwh"] >= 0
        # State machine (9.1) and bounds (9.2)
        e = e + chg - dis
        assert p["battery_energy_after_kwh"] == pytest.approx(e, abs=TOL)
        assert case["input"]["battery"]["minimum_energy_kwh"] - TOL <= e
        assert e <= case["input"]["battery"]["capacity_kwh"] + TOL
    # End-of-day neutrality (9.6)
    assert e == pytest.approx(initial, abs=TOL)


@pytest.mark.parametrize("case_id", ["SAMPLE-01", "SAMPLE-05", "SAMPLE-09"])
def test_endpoint_beats_the_do_nothing_baseline(cases, case_id) -> None:
    case = cases[case_id]
    body = client.post("/optimize-energy", json=case["input"]).json()
    assert body["total_cost_bdt"] < idle_baseline_cost(case) - TOL


@pytest.mark.parametrize("case_id", ["SAMPLE-01", "SAMPLE-05", "SAMPLE-09"])
def test_stubbed_cost_equals_the_no_directive_optimum(cases, case_id) -> None:
    """The endpoint must return exactly what the engine computes, no drift."""
    case = cases[case_id]
    request = energy_optimizer.OptimizeEnergyRequest(**case["input"])
    scenario = energy_optimizer.to_scenario(request, [])
    result = energy_optimizer.optimize(scenario,
                                       validate_fn=energy_optimizer.make_validator())
    body = client.post("/optimize-energy", json=case["input"]).json()
    assert body["total_cost_bdt"] == pytest.approx(result.total_cost_bdt, abs=TOL)


def test_battery_is_actually_used(cases) -> None:
    """A plan that never touches the battery would be the old mock."""
    plan = client.post(
        "/optimize-energy", json=cases["SAMPLE-01"]["input"]
    ).json()["hourly_plan"]
    actions = {p["battery_action"] for p in plan}
    assert "charge" in actions and "discharge" in actions


def test_every_note_still_reported_as_no_op(cases) -> None:
    entries = client.post(
        "/optimize-energy", json=cases["SAMPLE-06"]["input"]
    ).json()["directive_interpretation"]
    assert [e["note_index"] for e in entries] == [0, 1, 2]
    assert all(e["applies"] is False for e in entries)
    assert all(e["directive_type"] == "no_op" for e in entries)
    assert all(e["structured_adjustment"] is None for e in entries)


# ---------------------------------------------------------------------------
# The seam the LLM will plug into
# ---------------------------------------------------------------------------


def test_directive_flows_end_to_end(monkeypatch, cases) -> None:
    """Replace only interpret_notes and confirm the plan obeys the directive."""
    case = cases["SAMPLE-03"]          # reserve 100 kWh during hours 18-20
    capacity = case["input"]["battery"]["capacity_kwh"]

    def fake_interpret(notes):
        return [
            NoteInterpretation(
                note_index=0,
                applies=True,
                directive_type="minimum_battery_reserve",
                hours=[18, 19, 20],
                reserve_percent_of_capacity=50,   # 50% of capacity -> 100 kWh
                explanation="emergency reserve",
            )
        ]

    monkeypatch.setattr(energy_optimizer, "interpret_notes", fake_interpret)
    body = client.post("/optimize-energy", json=case["input"]).json()

    entry = body["directive_interpretation"][0]
    assert entry["applies"] is True
    assert entry["directive_type"] == "minimum_battery_reserve"
    assert entry["structured_adjustment"] == {
        "hours": [18, 19, 20], "minimum_energy_kwh": capacity * 0.5
    }

    plan = {p["hour"]: p for p in body["hourly_plan"]}
    for h in (18, 19, 20):
        assert plan[h]["battery_energy_after_kwh"] >= capacity * 0.5 - TOL

    # And it must match the published optimum for this case.
    expected = float(case["expected_output"]["total_cost_bdt"])
    assert body["total_cost_bdt"] == pytest.approx(expected, abs=TOL)


def test_solar_reduction_flows_end_to_end(monkeypatch, cases) -> None:
    case = cases["SAMPLE-09"]          # 80% reduction -> factor 0.2, hours 11-13
    monkeypatch.setattr(
        energy_optimizer,
        "interpret_notes",
        lambda notes: [
            NoteInterpretation(
                note_index=0, applies=True, directive_type="solar_reduction",
                hours=[11, 12, 13], factor=0.2, explanation="inverter work",
            ),
            NoteInterpretation(
                note_index=1, applies=False, directive_type="no_op",
                explanation="unrelated",
            ),
        ],
    )
    body = client.post("/optimize-energy", json=case["input"]).json()
    solar = {h["hour"]: h["solar_kwh"] for h in case["input"]["hours"]}
    plan = {p["hour"]: p for p in body["hourly_plan"]}

    for h in (11, 12, 13):
        assert plan[h]["solar_used_kwh"] <= solar[h] * 0.2 + TOL
    assert body["total_cost_bdt"] == pytest.approx(
        float(case["expected_output"]["total_cost_bdt"]), abs=TOL
    )


def test_max_grid_cap_flows_end_to_end(monkeypatch, cases) -> None:
    case = cases["SAMPLE-05"]          # grid <= 155 kWh during hours 18-20
    monkeypatch.setattr(
        energy_optimizer,
        "interpret_notes",
        lambda notes: [
            NoteInterpretation(
                note_index=0, applies=True, directive_type="max_grid_window",
                hours=[18, 19, 20], max_grid_kwh=155, explanation="feeder limit",
            )
        ],
    )
    plan = client.post(
        "/optimize-energy", json=case["input"]
    ).json()["hourly_plan"]
    for p in plan:
        if p["hour"] in (18, 19, 20):
            assert p["grid_kwh"] <= 155 + TOL


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_impossible_directive_still_returns_200(monkeypatch, cases) -> None:
    """A misread number must degrade, not 500."""
    monkeypatch.setattr(
        energy_optimizer,
        "interpret_notes",
        lambda notes: [
            NoteInterpretation(
                note_index=0, applies=True, directive_type="max_grid_window",
                hours=[18, 19, 20], max_grid_kwh=1.0, explanation="absurd cap",
            )
        ],
    )
    r = client.post("/optimize-energy", json=cases["SAMPLE-05"]["input"])
    assert r.status_code == 200
    body = r.json()
    assert len(body["hourly_plan"]) == 24
    assert "WARNING" in body["plan_summary"]
    assert all(p["grid_kwh"] >= 0 for p in body["hourly_plan"])


def test_request_latency_is_well_inside_the_budget(cases) -> None:
    """The guide requires p95 <= 5s; the LP alone should take milliseconds."""
    import time

    t0 = time.perf_counter()
    for _ in range(5):
        assert client.post(
            "/optimize-energy", json=cases["SAMPLE-07"]["input"]
        ).status_code == 200
    per_call = (time.perf_counter() - t0) / 5
    assert per_call < 1.0, f"{per_call:.3f}s per call is far too slow"
