"""Tests for POST /optimize-energy."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

BATTERY = {
    "capacity_kwh": 200,
    "initial_energy_kwh": 100,
    "minimum_energy_kwh": 40,
    "max_charge_kwh_per_hour": 50,
    "max_discharge_kwh_per_hour": 50,
}


def make_request(n_notes: int = 2, **overrides):
    hours = [
        {
            "hour": h,
            "demand_kwh": 120 + (60 if 17 <= h <= 21 else 0),
            "solar_kwh": round(max(0.0, 160 - 12 * abs(h - 12.5)), 1),
            "tariff_bdt_per_kwh": 8 + (14 if 17 <= h <= 21 else 0),
        }
        for h in range(24)
    ]
    body = {
        "scenario_id": "TEST-001",
        "operator_notes": ["note " + str(i) for i in range(n_notes)],
        "hours": hours,
        "battery": dict(BATTERY),
    }
    body.update(overrides)
    return body


def post(body):
    return client.post("/optimize-energy", json=body)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_returns_200_and_full_contract() -> None:
    r = post(make_request())
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "scenario_id",
        "directive_interpretation",
        "hourly_plan",
        "total_grid_kwh",
        "total_cost_bdt",
        "peak_grid_kwh",
        "plan_summary",
    }


def test_scenario_id_is_echoed() -> None:
    assert post(make_request(scenario_id="GRID-XYZ")).json()["scenario_id"] == "GRID-XYZ"


@pytest.mark.parametrize("n_notes", [1, 2, 3])
def test_one_interpretation_per_note_in_order(n_notes: int) -> None:
    entries = post(make_request(n_notes=n_notes)).json()["directive_interpretation"]
    assert [e["note_index"] for e in entries] == list(range(n_notes))
    for e in entries:
        assert set(e) == {
            "note_index",
            "applies",
            "directive_type",
            "structured_adjustment",
            "explanation",
        }


def test_hourly_plan_covers_24_hours_with_exact_fields() -> None:
    plan = post(make_request()).json()["hourly_plan"]
    assert len(plan) == 24
    assert [p["hour"] for p in plan] == list(range(24))
    for p in plan:
        assert set(p) == {
            "hour",
            "grid_kwh",
            "solar_used_kwh",
            "battery_action",
            "battery_kwh",
            "battery_energy_after_kwh",
        }
        assert p["battery_action"] in ("charge", "discharge", "idle")


def test_totals_match_the_returned_plan() -> None:
    req = make_request()
    body = post(req).json()
    plan = body["hourly_plan"]
    tariff = {h["hour"]: h["tariff_bdt_per_kwh"] for h in req["hours"]}
    assert body["total_grid_kwh"] == pytest.approx(
        sum(p["grid_kwh"] for p in plan), abs=0.01
    )
    assert body["total_cost_bdt"] == pytest.approx(
        sum(p["grid_kwh"] * tariff[p["hour"]] for p in plan), abs=0.01
    )
    assert body["peak_grid_kwh"] == pytest.approx(
        max(p["grid_kwh"] for p in plan), abs=0.01
    )


def test_returned_plan_satisfies_the_energy_rules() -> None:
    """Whatever the optimizer does, the schedule must obey the GridWise rules."""
    req = make_request()
    body = post(req).json()
    plan = {p["hour"]: p for p in body["hourly_plan"]}
    for h in req["hours"]:
        p = plan[h["hour"]]
        chg = p["battery_kwh"] if p["battery_action"] == "charge" else 0.0
        dis = p["battery_kwh"] if p["battery_action"] == "discharge" else 0.0
        assert p["grid_kwh"] + p["solar_used_kwh"] + dis == pytest.approx(
            h["demand_kwh"] + chg, abs=0.01
        )
        assert p["solar_used_kwh"] <= h["solar_kwh"] + 0.01
        assert p["grid_kwh"] >= 0
    # End-of-day battery neutrality.
    assert plan[23]["battery_energy_after_kwh"] == pytest.approx(
        req["battery"]["initial_energy_kwh"], abs=0.01
    )


def test_idle_hours_report_zero_battery_kwh() -> None:
    for p in post(make_request()).json()["hourly_plan"]:
        if p["battery_action"] == "idle":
            assert p["battery_kwh"] == 0


def test_extra_request_fields_are_ignored_not_rejected() -> None:
    assert post(make_request(submitted_at="2026-09-18T19:00:00Z")).status_code == 200


def test_out_of_order_hours_are_accepted_and_sorted() -> None:
    req = make_request()
    req["hours"] = list(reversed(req["hours"]))
    body = post(req).json()
    assert [p["hour"] for p in body["hourly_plan"]] == list(range(24))


# ---------------------------------------------------------------------------
# Invalid input -> 400
# ---------------------------------------------------------------------------


def test_malformed_json_returns_400() -> None:
    r = client.post(
        "/optimize-energy",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.pop("scenario_id"),
        lambda b: b.pop("battery"),
        lambda b: b.update(operator_notes=[]),
        lambda b: b.update(operator_notes=["a", "b", "c", "d"]),
        lambda b: b.update(hours=b["hours"][:23]),
        lambda b: b.update(hours=[{**h, "hour": h["hour"]} for h in b["hours"][:23]]
                           + [{**b["hours"][0]}]),
        lambda b: b.update(hours=[{**b["hours"][0], "demand_kwh": -5}] + b["hours"][1:]),
        lambda b: b.update(battery={**b["battery"], "initial_energy_kwh": 9999}),
    ],
    ids=[
        "no scenario_id",
        "no battery",
        "zero notes",
        "four notes",
        "23 hours",
        "duplicate hour",
        "negative demand",
        "initial above capacity",
    ],
)
def test_invalid_requests_return_400(mutate) -> None:
    body = make_request()
    mutate(body)
    assert post(body).status_code == 400


def test_error_body_leaks_no_stack_trace() -> None:
    r = post(make_request(scenario_id=""))
    assert r.status_code == 400
    text = r.text.lower()
    assert "traceback" not in text
    assert "site-packages" not in text


# ---------------------------------------------------------------------------
# Real public sample case
# ---------------------------------------------------------------------------


def test_public_sample_case_round_trips() -> None:
    path = "/home/user/uploads/BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
    try:
        cases = json.load(open(path))["cases"]
    except FileNotFoundError:
        pytest.skip("public sample pack not present")
    for case in cases:
        r = post(case["input"])
        assert r.status_code == 200, case["id"]
        body = r.json()
        assert body["scenario_id"] == case["input"]["scenario_id"]
        assert len(body["hourly_plan"]) == 24
        assert len(body["directive_interpretation"]) == len(
            case["input"]["operator_notes"]
        )
