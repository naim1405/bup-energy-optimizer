"""Tests for the LLM interpretation layer.

None of these call OpenAI. The model is injected as a fake, which is also how
the failure paths get exercised -- a missing key, a rate limit and a
malformed response all have to degrade to ``no_op`` rather than raise.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.config import Settings
from app.llm.interpreter import get_structured_llm, interpret_notes, reconcile
from app.llm.prompts import SYSTEM_PROMPT, build_messages
from app.main import app
from app.schemas.common import DirectiveType
from app.schemas.llm import NoteInterpretation, NoteInterpretationResult

client = TestClient(app)
CAPACITY = 200.0


class FakeLLM:
    """Minimal stand-in exposing only the ``invoke`` surface we use."""

    def __init__(self, results):
        self._results = list(results)
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        result = self._results.pop(0) if self._results else self._results_default
        if isinstance(result, Exception):
            raise result
        return result

    _results_default = None


def good_result():
    return NoteInterpretationResult(interpretations=[
        {"note_index": 0, "applies": True, "directive_type": "solar_reduction",
         "hours": [13, 14], "factor": 0.2},
        {"note_index": 1, "applies": False, "directive_type": "no_op"},
    ])


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_model_output_is_returned_unchanged() -> None:
    llm = FakeLLM([good_result()])
    out = interpret_notes(["note a", "unrelated"], CAPACITY, llm=llm)
    assert [e.note_index for e in out] == [0, 1]
    assert out[0].directive_type is DirectiveType.SOLAR_REDUCTION
    assert out[0].hours == [13, 14]
    assert out[0].factor == 0.2
    assert out[1].directive_type is DirectiveType.NO_OP
    assert len(llm.calls) == 1


def test_percentage_reserve_is_resolved_against_capacity() -> None:
    llm = FakeLLM([NoteInterpretationResult(interpretations=[
        {"note_index": 0, "applies": True,
         "directive_type": "minimum_battery_reserve",
         "hours": [18, 19, 20], "reserve_percent_of_capacity": 50},
    ])])
    out = interpret_notes(["keep half the pack"], CAPACITY, llm=llm)
    api = out[0].to_api_model(CAPACITY)
    assert api.structured_adjustment.minimum_energy_kwh == 100.0


# ---------------------------------------------------------------------------
# Reconcile: messy model output
# ---------------------------------------------------------------------------


def test_missing_note_index_is_filled_with_no_op() -> None:
    rec = reconcile(NoteInterpretationResult(interpretations=[
        {"note_index": 1, "applies": True, "directive_type": "no_charge_window",
         "hours": [3]},
    ]), n_notes=3)
    out = rec.entries
    assert [e.note_index for e in out] == [0, 1, 2]
    assert out[0].directive_type is DirectiveType.NO_OP
    assert out[2].directive_type is DirectiveType.NO_OP
    assert out[1].directive_type is DirectiveType.NO_CHARGE_WINDOW
    assert rec.complete is False, "a filled gap must be reported as incomplete"


def test_out_of_range_and_duplicate_indices_are_dropped() -> None:
    rec = reconcile(NoteInterpretationResult(interpretations=[
        {"note_index": 9, "applies": True, "directive_type": "no_charge_window",
         "hours": [3]},
        {"note_index": 0, "applies": True, "directive_type": "solar_reduction",
         "hours": [1], "factor": 0.5},
        {"note_index": 0, "applies": True, "directive_type": "no_charge_window",
         "hours": [2]},
    ]), n_notes=2)
    assert [e.note_index for e in rec.entries] == [0, 1]
    assert rec.entries[0].directive_type is DirectiveType.SOLAR_REDUCTION
    assert rec.complete is False


def test_none_result_signals_a_retry() -> None:
    rec = reconcile(None, n_notes=2)
    assert rec.entries is None and rec.complete is False


def test_empty_interpretations_signals_a_retry() -> None:
    rec = reconcile(NoteInterpretationResult(interpretations=[]), n_notes=2)
    assert rec.entries is None and rec.complete is False


def test_plain_dicts_are_validated() -> None:
    """Some providers hand back dicts rather than model instances."""
    rec = reconcile({"interpretations": [
        {"note_index": 0, "applies": True, "directive_type": "no_charge_window",
         "hours": [4, 5]},
    ]}, n_notes=1)
    assert rec.entries is not None
    assert rec.complete is True
    assert rec.entries[0].directive_type is DirectiveType.NO_CHARGE_WINDOW


# ---------------------------------------------------------------------------
# Failure handling -- the part that must never raise
# ---------------------------------------------------------------------------


def test_provider_error_falls_back_to_no_op() -> None:
    llm = FakeLLM([RuntimeError("rate limited"), RuntimeError("still down")])
    out = interpret_notes(["a", "b"], CAPACITY, llm=llm)
    assert len(out) == 2
    assert all(e.directive_type is DirectiveType.NO_OP for e in out)
    assert all(e.applies is False for e in out)


def test_retries_once_then_succeeds() -> None:
    llm = FakeLLM([RuntimeError("transient"), good_result()])
    out = interpret_notes(["a", "b"], CAPACITY, llm=llm)
    assert out[0].directive_type is DirectiveType.SOLAR_REDUCTION
    assert len(llm.calls) == 2


def test_incomplete_result_triggers_a_retry() -> None:
    partial = NoteInterpretationResult(interpretations=[
        {"note_index": 0, "applies": True, "directive_type": "no_charge_window",
         "hours": [3]},
    ])
    llm = FakeLLM([partial, good_result()])
    out = interpret_notes(["a", "b"], CAPACITY, llm=llm)
    assert len(llm.calls) == 2
    assert out[0].directive_type is DirectiveType.SOLAR_REDUCTION


def test_partial_result_is_kept_when_the_retry_also_fails() -> None:
    """Some interpretation beats none, even if it only covers one note."""
    partial = NoteInterpretationResult(interpretations=[
        {"note_index": 0, "applies": True, "directive_type": "no_charge_window",
         "hours": [3]},
    ])
    llm = FakeLLM([partial, RuntimeError("second call died")])
    out = interpret_notes(["a", "b"], CAPACITY, llm=llm)
    assert len(out) == 2
    assert out[0].directive_type is DirectiveType.NO_CHARGE_WINDOW, "kept"
    assert out[1].directive_type is DirectiveType.NO_OP, "filled"


def test_budget_exhaustion_stops_retrying(monkeypatch) -> None:
    """A hung provider must not eat the whole 30s request budget."""
    from app.core.config import settings
    from app.llm import interpreter

    monkeypatch.setattr(settings, "llm_budget_seconds", 0.0)
    llm = FakeLLM([RuntimeError("slow")] * 5)
    out = interpret_notes(["a", "b"], CAPACITY, llm=llm)
    assert len(llm.calls) == 1, "no retry once the budget is gone"
    assert all(e.directive_type is DirectiveType.NO_OP for e in out)
    assert interpreter  # module imported cleanly


def test_client_construction_failure_falls_back(monkeypatch) -> None:
    """No API key at all must still produce a valid, if unapplied, plan."""
    from app.llm import interpreter

    def boom():
        raise RuntimeError("api_key not set")

    monkeypatch.setattr(interpreter, "get_structured_llm", boom)
    out = interpret_notes(["a"], CAPACITY, llm=None)
    assert out[0].directive_type is DirectiveType.NO_OP


def test_endpoint_survives_a_dead_provider(monkeypatch, cases_input) -> None:
    """End-to-end: a dead LLM must not turn into a 500."""
    import app.services.energy_optimizer as svc

    monkeypatch.setattr(
        svc, "interpret_notes",
        lambda notes, capacity_kwh: [
            NoteInterpretation(note_index=i, applies=False,
                               directive_type=DirectiveType.NO_OP,
                               explanation="unavailable")
            for i in range(len(notes))
        ],
    )
    r = client.post("/optimize-energy", json=cases_input)
    assert r.status_code == 200
    body = r.json()
    assert len(body["hourly_plan"]) == 24
    assert all(e["applies"] is False for e in body["directive_interpretation"])


@pytest.fixture
def cases_input():
    import json
    from pathlib import Path

    cases = json.loads(
        (Path(__file__).parent / "fixtures" / "public_sample_cases.json").read_text()
    )["cases"]
    return cases[0]["input"]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def test_prompt_states_the_two_rules_models_get_wrong() -> None:
    # Windows: half-open, and expressed as bounds rather than a hand-counted
    # list, because the model enumerates hour lists unreliably.
    assert "the end is NOT" in SYSTEM_PROMPT
    assert "window_start_hour" in SYSTEM_PROMPT
    assert "window_end_hour" in SYSTEM_PROMPT
    assert "start 18, end 21" in SYSTEM_PROMPT      # covers 18, 19 and 20
    assert "until midnight" in SYSTEM_PROMPT        # end 24, not 23
    # Factor: the fraction remaining, not the amount lost.
    assert "REMAINS" in SYSTEM_PROMPT
    assert "factor 0.2" in SYSTEM_PROMPT
    assert "falls by two-fifths" in SYSTEM_PROMPT   # -> 0.6, the "by" trap


def test_prompt_covers_every_directive_type() -> None:
    for name in ("solar_reduction", "minimum_battery_reserve",
                 "no_charge_window", "no_discharge_window",
                 "max_grid_window", "no_op"):
        assert name in SYSTEM_PROMPT


def test_prompt_warns_about_distractors() -> None:
    assert "no_op" in SYSTEM_PROMPT
    assert "distractor" in SYSTEM_PROMPT.lower()


def test_user_message_carries_every_note_and_the_capacity() -> None:
    messages = build_messages(["first note", "second note"], 200.0)
    assert len(messages) == 2
    user = messages[1].content
    assert "[0] first note" in user
    assert "[1] second note" in user
    assert "200 kWh" in user
    assert "exactly 2 entries" in user


def test_one_call_covers_all_notes() -> None:
    """Three notes must be one round-trip, not three -- p95 latency is scored."""
    llm = FakeLLM([NoteInterpretationResult(interpretations=[
        {"note_index": i, "applies": False, "directive_type": "no_op"}
        for i in range(3)
    ])])
    interpret_notes(["a", "b", "c"], CAPACITY, llm=llm)
    assert len(llm.calls) == 1


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_model_name_comes_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    assert Settings().llm_model == "gpt-4o"


def test_model_name_defaults_to_gpt_4o_mini(monkeypatch) -> None:
    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert Settings().llm_model == "gpt-4o-mini"


def test_api_key_is_never_repr_in_plaintext(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-value")
    s = Settings()
    assert isinstance(s.openai_api_key, SecretStr)
    assert "super-secret" not in repr(s)
    assert "super-secret" not in s.model_dump().__str__()
    # ...but is still retrievable for the client.
    assert s.openai_api_key.get_secret_value() == "sk-super-secret-value"


def test_structured_llm_builds_from_settings(monkeypatch) -> None:
    """Construction must not require a network call."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    get_structured_llm.cache_clear()
    try:
        llm = get_structured_llm()
        assert llm is not None
    finally:
        get_structured_llm.cache_clear()


def test_health_works_with_no_key_configured(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    get_structured_llm.cache_clear()
    try:
        assert client.get("/health").json() == {"status": "ok"}
    finally:
        get_structured_llm.cache_clear()
