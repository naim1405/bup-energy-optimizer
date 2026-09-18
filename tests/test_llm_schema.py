"""Tests for the LLM structured-output schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.common import DirectiveType
from app.schemas.llm import NoteInterpretation, NoteInterpretationResult

CAPACITY = 200.0


def test_solar_reduction_round_trips() -> None:
    n = NoteInterpretation(
        note_index=0,
        applies=True,
        directive_type="solar_reduction",
        hours=[13, 14],
        factor=0.2,
        explanation="panel cleaning",
    )
    api = n.to_api_model(CAPACITY)
    assert api.applies is True
    assert api.directive_type is DirectiveType.SOLAR_REDUCTION
    assert api.structured_adjustment.model_dump(exclude_none=True) == {
        "hours": [13, 14],
        "factor": 0.2,
    }


def test_no_op_round_trips() -> None:
    n = NoteInterpretation(
        note_index=1, applies=False, directive_type="no_op", explanation="unrelated"
    )
    api = n.to_api_model(CAPACITY)
    assert api.applies is False
    assert api.structured_adjustment is None


def test_hours_are_deduped_and_sorted() -> None:
    n = NoteInterpretation(
        note_index=0, applies=True, directive_type="no_charge_window",
        hours=[15, 13, 13, 14],
    )
    assert n.hours == [13, 14, 15]
    assert n.to_api_model(CAPACITY).structured_adjustment.hours == [13, 14, 15]


def test_window_directives_carry_no_stray_numeric_fields() -> None:
    """A no_discharge_window adjustment is hours-only -- no null riders."""
    adj = NoteInterpretation(
        note_index=0, applies=True, directive_type="no_discharge_window", hours=[18, 19]
    ).to_api_model(CAPACITY).structured_adjustment
    assert adj.model_dump() == {"hours": [18, 19]}
    assert adj.factor is None and adj.minimum_energy_kwh is None
    assert adj.max_grid_kwh is None


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(hours=[18, 19, 20], minimum_energy_kwh=100),
        dict(hours=[19, 20], max_grid_kwh=180),
    ],
)
def test_numeric_directives_round_trip(kwargs) -> None:
    dtype = ("minimum_battery_reserve" if "minimum_energy_kwh" in kwargs
             else "max_grid_window")
    n = NoteInterpretation(
        note_index=0, applies=True, directive_type=dtype, **kwargs
    )
    adj = n.to_api_model(CAPACITY).structured_adjustment
    assert adj.model_dump(exclude_none=True) == {"hours": kwargs["hours"],
                                                 **{k: v for k, v in kwargs.items()
                                                    if k != "hours"}}


# --- the percentage-of-capacity case (public SAMPLE-03) ---------------------


def test_percentage_reserve_resolves_to_absolute_kwh() -> None:
    n = NoteInterpretation(
        note_index=0, applies=True, directive_type="minimum_battery_reserve",
        hours=[18, 19, 20], reserve_percent_of_capacity=50,
    )
    assert n.resolved_reserve_kwh(CAPACITY) == 100.0
    adj = n.to_api_model(CAPACITY).structured_adjustment
    assert adj.minimum_energy_kwh == 100.0
    # The API contract only carries the absolute figure.
    assert "reserve_percent_of_capacity" not in adj.model_dump()


def test_absolute_kwh_wins_over_the_percentage() -> None:
    n = NoteInterpretation(
        note_index=0, applies=True, directive_type="minimum_battery_reserve",
        hours=[18], minimum_energy_kwh=90, reserve_percent_of_capacity=50,
    )
    assert n.resolved_reserve_kwh(CAPACITY) == 90.0


# --- applies is derived, never trusted --------------------------------------


def test_llm_cannot_mark_a_real_directive_inapplicable() -> None:
    n = NoteInterpretation.model_construct(
        note_index=0, applies=False, directive_type=DirectiveType.MAX_GRID_WINDOW,
        hours=[19, 20], factor=None, minimum_energy_kwh=None,
        reserve_percent_of_capacity=None, max_grid_kwh=180, explanation="",
    )
    assert n.to_api_model(CAPACITY).applies is True


def test_llm_cannot_mark_no_op_applicable() -> None:
    n = NoteInterpretation.model_construct(
        note_index=0, applies=True, directive_type=DirectiveType.NO_OP, hours=[],
        factor=None, minimum_energy_kwh=None, reserve_percent_of_capacity=None,
        max_grid_kwh=None, explanation="",
    )
    api = n.to_api_model(CAPACITY)
    assert api.applies is False
    assert api.structured_adjustment is None


# --- validation failures (what the call site must catch) --------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(directive_type="solar_reduction", hours=[13]), "requires factor"),
        (dict(directive_type="max_grid_window", hours=[13]), "requires max_grid_kwh"),
        (dict(directive_type="minimum_battery_reserve", hours=[13]),
         "minimum_energy_kwh"),
        (dict(directive_type="no_charge_window", hours=[]), "non-empty hours"),
        (dict(directive_type="no_op", applies=True), "applies=false"),
        (dict(directive_type="no_op", applies=False, hours=[3]), "no_op must not"),
        (dict(directive_type="solar_reduction", hours=[13], factor=1.7),
         "less than or equal to 1"),
        (dict(directive_type="solar_reduction", hours=[24], factor=0.5),
         "less than or equal to 23"),
    ],
)
def test_bad_llm_output_raises_validation_error(kwargs, match) -> None:
    # kwargs may override `applies`, so merge rather than pass both.
    with pytest.raises(ValidationError, match=match):
        NoteInterpretation(**{"note_index": 0, "applies": True, **kwargs})


def test_unknown_directive_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        NoteInterpretation(note_index=0, applies=True,
                           directive_type="warp_core_offline", hours=[1])


def test_root_model_reindexes_by_note_index() -> None:
    res = NoteInterpretationResult(interpretations=[
        {"note_index": 2, "applies": False, "directive_type": "no_op"},
        {"note_index": 0, "applies": True, "directive_type": "no_charge_window",
         "hours": [3, 4]},
        {"note_index": 1, "applies": True, "directive_type": "solar_reduction",
         "hours": [12], "factor": 0.25},
    ])
    assert [i.note_index for i in res.interpretations] == [0, 1, 2]


def test_every_entry_projects_to_a_valid_api_model() -> None:
    res = NoteInterpretationResult(interpretations=[
        {"note_index": 0, "applies": True, "directive_type": "solar_reduction",
         "hours": [13, 14], "factor": 0.2},
        {"note_index": 1, "applies": True, "directive_type": "minimum_battery_reserve",
         "hours": [18, 19, 20], "reserve_percent_of_capacity": 50},
        {"note_index": 2, "applies": False, "directive_type": "no_op"},
    ])
    api = [i.to_api_model(CAPACITY) for i in res.interpretations]
    assert [a.note_index for a in api] == [0, 1, 2]
    assert [a.applies for a in api] == [True, True, False]
    assert api[1].structured_adjustment.minimum_energy_kwh == 100.0


# --- window bounds -> derived hour list -------------------------------------
# The model reads a window correctly but miscounts the enumeration, dropping
# the final hour. Deriving the list from the bounds removes that class of
# error; these tests pin the arithmetic, including the awkward edges.


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (18, 21, [18, 19, 20]),      # "6 PM until 9 PM"
        (13, 15, [13, 14]),          # "1 PM to 3 PM"
        (11, 14, [11, 12, 13]),      # "11 AM to 2 PM"
        (22, 24, [22, 23]),          # "...until midnight"
        (23, 2, [0, 1, 23]),         # wraps past midnight, sorted ascending
        (9, 10, [9]),                # single hour
        (0, 24, list(range(24))),    # whole day
    ],
)
def test_window_bounds_derive_the_hour_list(start: int, end: int,
                                            expected: list[int]) -> None:
    n = NoteInterpretation(
        note_index=0, applies=True, directive_type="no_discharge_window",
        window_start_hour=start, window_end_hour=end,
    )
    assert n.hours == expected


def test_window_bounds_reach_the_api_contract() -> None:
    n = NoteInterpretation(
        note_index=0, applies=True, directive_type="minimum_battery_reserve",
        window_start_hour=18, window_end_hour=21, reserve_percent_of_capacity=50,
    )
    adj = n.to_api_model(200.0).model_dump()["structured_adjustment"]
    assert adj == {"hours": [18, 19, 20], "minimum_energy_kwh": 100.0}


def test_explicit_hours_survive_when_no_bounds_are_given() -> None:
    """A non-contiguous window is still listable by hand."""
    n = NoteInterpretation(
        note_index=0, applies=True, directive_type="no_charge_window",
        hours=[9, 14, 15],
    )
    assert n.hours == [9, 14, 15]


def test_bounds_do_not_resurrect_a_no_op() -> None:
    n = NoteInterpretation(note_index=0, applies=False, directive_type="no_op")
    assert n.hours == []
    assert n.to_api_model(200.0).model_dump()["structured_adjustment"] is None


def test_nonsense_bounds_fall_back_without_crashing() -> None:
    """Out-of-range bounds must not produce a bogus plan."""
    n = NoteInterpretation(
        note_index=0, applies=True, directive_type="no_charge_window",
        window_start_hour=18, window_end_hour=21, hours=[18, 19, 20],
    )
    assert n.hours == [18, 19, 20]
