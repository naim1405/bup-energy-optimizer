"""Structured-output schema for the operator-note interpretation LLM.

This is the model you will hand to LangChain::

    llm = init_chat_model(...).with_structured_output(NoteInterpretationResult)
    result: NoteInterpretationResult = llm.invoke(messages)

Design notes
------------
* Deliberately FLAT per note. Nested optional objects are noticeably harder
  for models to fill reliably, and a flat schema lets the per-type required
  field be enforced by a validator instead of by the model's judgement.
* The ``Field(description=...)`` strings are not decoration -- LangChain puts
  them in the JSON schema the model sees, so they are part of the prompt. The
  two rules models get wrong most often (factor = fraction REMAINING, and
  half-open hour windows) are stated there explicitly.
* LLM output is untrusted. Validation errors raised here are the signal your
  call site should catch, retry once, and then fall back to ``no_op`` -- never
  let one bad note take the request down.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.common import DirectiveType
from app.schemas.energy import DirectiveInterpretation, StructuredAdjustment

#: How the LLM should express a reserve given as a percentage of capacity.
PCT_KEYS = ("percent_of_capacity",)


class NoteInterpretation(BaseModel):
    """The model's reading of ONE operator note.

    Exactly one of these must be produced per note, in note_index order.
    """

    model_config = ConfigDict(extra="ignore")

    note_index: Annotated[int, Field(ge=0)] = Field(
        description="Zero-based index of the operator note this entry describes."
    )
    applies: bool = Field(
        description="True if the note changes today's 24-hour energy schedule. "
        "False only when the note is irrelevant (a distractor) and "
        "directive_type is no_op."
    )
    directive_type: DirectiveType = Field(
        description="Exactly one supported directive type. Use no_op for any "
        "note that does not affect the current 24-hour energy schedule."
    )

    hours: list[Annotated[int, Field(ge=0, le=23)]] = Field(
        default_factory=list,
        description="Affected hours as integers 0-23. Time windows are "
        "start-inclusive and end-EXCLUSIVE: '1 PM to 3 PM' is [13, 14], "
        "'6 PM until 9 PM' is [18, 19, 20]. Empty only for no_op.",
    )

    factor: Annotated[float, Field(ge=0, le=1)] | None = Field(
        default=None,
        description="solar_reduction only. The usable fraction of solar that "
        "REMAINS, not the amount lost: an 80% reduction means 0.2, 'drops to "
        "about 20%' means 0.2, 'one-fifth of normal' means 0.2.",
    )
    minimum_energy_kwh: Annotated[float, Field(ge=0)] | None = Field(
        default=None,
        description="minimum_battery_reserve only. Absolute kWh that must stay "
        "in the battery. If the note states a percentage of capacity, convert "
        "it using the battery capacity given in the request.",
    )
    reserve_percent_of_capacity: Annotated[float, Field(ge=0, le=100)] | None = (
        Field(
            default=None,
            description="Fallback for minimum_battery_reserve when the note "
            "gives a percentage rather than an absolute kWh figure, e.g. "
            "'keep at least 50% of the battery capacity' -> 50. Leave null if "
            "minimum_energy_kwh is already filled in.",
        )
    )
    max_grid_kwh: Annotated[float, Field(ge=0)] | None = Field(
        default=None,
        description="max_grid_window only. The maximum grid import permitted "
        "in each listed hour, in kWh.",
    )

    explanation: str = Field(
        default="",
        description="One short sentence explaining the interpretation. "
        "Free-text wording is not judged.",
    )

    @field_validator("hours")
    @classmethod
    def _normalize_hours(cls, v: list[int]) -> list[int]:
        """Coerce to unique ascending integers, as the contract requires."""
        return sorted(set(v))

    @model_validator(mode="after")
    def _check_type_coupling(self) -> Self:
        t = self.directive_type
        if t is DirectiveType.NO_OP:
            if self.applies:
                raise ValueError("no_op must use applies=false")
            if self.hours or any(
                getattr(self, k) is not None
                for k in ("factor", "minimum_energy_kwh",
                          "reserve_percent_of_capacity", "max_grid_kwh")
            ):
                raise ValueError("no_op must not carry hours or numeric fields")
            return self

        if not self.hours:
            raise ValueError(f"{t.value} requires a non-empty hours list")

        if t is DirectiveType.SOLAR_REDUCTION and self.factor is None:
            raise ValueError("solar_reduction requires factor")
        if t is DirectiveType.MINIMUM_BATTERY_RESERVE and (
            self.minimum_energy_kwh is None
            and self.reserve_percent_of_capacity is None
        ):
            raise ValueError(
                "minimum_battery_reserve requires minimum_energy_kwh or "
                "reserve_percent_of_capacity"
            )
        if t is DirectiveType.MAX_GRID_WINDOW and self.max_grid_kwh is None:
            raise ValueError("max_grid_window requires max_grid_kwh")
        return self

    def resolved_reserve_kwh(self, battery_capacity_kwh: float) -> float | None:
        """Absolute reserve in kWh, resolving a percentage against capacity.

        The response contract only accepts absolute kWh, so a percentage note
        must be converted before it leaves this layer.
        """
        if self.minimum_energy_kwh is not None:
            return self.minimum_energy_kwh
        if self.reserve_percent_of_capacity is not None:
            return battery_capacity_kwh * self.reserve_percent_of_capacity / 100.0
        return None

    def to_api_model(self, battery_capacity_kwh: float) -> DirectiveInterpretation:
        """Project this into the API contract's interpretation entry.

        ``applies`` and the adjustment shape are DERIVED here rather than
        trusted, so an inconsistent model answer cannot produce a response
        that violates section 05.1.
        """
        if self.directive_type is DirectiveType.NO_OP:
            return DirectiveInterpretation(
                note_index=self.note_index,
                applies=False,
                directive_type=DirectiveType.NO_OP,
                structured_adjustment=None,
                explanation=self.explanation
                or "This note does not affect today's energy schedule.",
            )

        payload: dict[str, object] = {"hours": self.hours}
        if self.directive_type is DirectiveType.SOLAR_REDUCTION:
            payload["factor"] = self.factor
        elif self.directive_type is DirectiveType.MINIMUM_BATTERY_RESERVE:
            payload["minimum_energy_kwh"] = self.resolved_reserve_kwh(
                battery_capacity_kwh
            )
        elif self.directive_type is DirectiveType.MAX_GRID_WINDOW:
            payload["max_grid_kwh"] = self.max_grid_kwh

        return DirectiveInterpretation(
            note_index=self.note_index,
            applies=True,
            directive_type=self.directive_type,
            structured_adjustment=StructuredAdjustment(**payload),
            explanation=self.explanation,
        )


class NoteInterpretationResult(BaseModel):
    """Root object for ``with_structured_output``.

    LangChain needs a single top-level model; this is it. Ask for one entry
    per operator note, ordered by note_index.
    """

    model_config = ConfigDict(extra="ignore")

    interpretations: list[NoteInterpretation] = Field(
        description="Exactly one entry per operator note, in note_index order "
        "0..N-1. Never skip a note and never emit duplicates."
    )

    @field_validator("interpretations")
    @classmethod
    def _reindex(cls, v: list[NoteInterpretation]) -> list[NoteInterpretation]:
        """Sort by note_index so downstream ordering is deterministic."""
        return sorted(v, key=lambda i: i.note_index)
