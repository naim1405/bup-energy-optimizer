"""Request and response models for POST /optimize-energy.

Field names and shapes follow Problem Statement sections 07 and 10 exactly --
the judge harness checks them mechanically, so they are not free to drift.

Input models are lenient (``extra="ignore"``): a judge sending an additional
metadata field must not cause a 422. Output models are strict
(``extra="forbid"``) so we can never emit a field outside the contract.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from app.schemas.common import (
    HOURS_PER_DAY,
    TYPE_REQUIRED_FIELD,
    BatteryAction,
    DirectiveType,
)

NonNegative = Annotated[float, Field(ge=0)]


# ===========================================================================
# Request (Problem Statement 07)
# ===========================================================================


class HourInput(BaseModel):
    """One hour of the 24-hour scenario (section 07.2)."""

    model_config = ConfigDict(extra="ignore")

    hour: Annotated[int, Field(ge=0, le=23, description="Unique integer 0-23.")]
    demand_kwh: NonNegative = Field(description="Campus demand that must be supplied.")
    solar_kwh: NonNegative = Field(
        description="Base solar energy available, before operator-note adjustments."
    )
    tariff_bdt_per_kwh: NonNegative = Field(description="Grid price for this hour.")


class BatterySpec(BaseModel):
    """Battery parameters (section 07.3)."""

    model_config = ConfigDict(extra="ignore")

    capacity_kwh: NonNegative = Field(description="Maximum storable energy.")
    initial_energy_kwh: NonNegative = Field(description="Energy at the start of hour 0.")
    minimum_energy_kwh: NonNegative = Field(
        description="Base reserve level the battery must never go below."
    )
    max_charge_kwh_per_hour: NonNegative
    max_discharge_kwh_per_hour: NonNegative

    @model_validator(mode="after")
    def _check_against_capacity(self) -> Self:
        if self.initial_energy_kwh > self.capacity_kwh:
            raise ValueError("initial_energy_kwh exceeds capacity_kwh")
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError("minimum_energy_kwh exceeds capacity_kwh")
        return self


class OptimizeEnergyRequest(BaseModel):
    """Body of POST /optimize-energy (section 07.1)."""

    model_config = ConfigDict(extra="ignore")

    scenario_id: str = Field(min_length=1, description="Unique synthetic scenario id.")
    operator_notes: Annotated[
        list[Annotated[str, Field(min_length=1)]],
        Field(min_length=1, max_length=3),
    ] = Field(description="1-3 natural-language campus operator notes to interpret.")
    hours: list[HourInput] = Field(description="Exactly 24 hourly entries, hours 0-23.")
    battery: BatterySpec

    @field_validator("hours")
    @classmethod
    def _normalize_hours(cls, v: list[HourInput]) -> list[HourInput]:
        """Sort by hour and require full 0-23 coverage with no duplicates."""
        if len(v) != HOURS_PER_DAY:
            raise ValueError(f"hours must contain exactly {HOURS_PER_DAY} entries")
        ordered = sorted(v, key=lambda h: h.hour)
        seen = [h.hour for h in ordered]
        if seen != list(range(HOURS_PER_DAY)):
            raise ValueError("hours must cover 0-23 exactly once each")
        return ordered


# ===========================================================================
# Response (Problem Statement 10)
# ===========================================================================


class StructuredAdjustment(BaseModel):
    """The machine-checkable directive payload (section 04.1).

    `hours` is always present; exactly one numeric field is present depending
    on the parent's directive_type. That coupling is enforced by
    :class:`DirectiveInterpretation`, which can see the type.
    """

    model_config = ConfigDict(extra="forbid")

    hours: list[Annotated[int, Field(ge=0, le=23)]] = Field(
        description="Unique integers 0-23 in ascending order."
    )
    factor: Annotated[float, Field(ge=0, le=1)] | None = Field(
        default=None,
        description="solar_reduction only: the usable fraction REMAINING. "
        "An 80% reduction means 0.2.",
    )
    minimum_energy_kwh: NonNegative | None = Field(
        default=None, description="minimum_battery_reserve only."
    )
    max_grid_kwh: NonNegative | None = Field(
        default=None, description="max_grid_window only."
    )

    @field_validator("hours")
    @classmethod
    def _normalize_hours(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("hours must not be empty")
        return sorted(set(v))

    @model_serializer
    def _drop_unused_fields(self) -> dict:
        """Emit only the keys this directive type actually uses.

        The contract requires an exact shape per type -- the published
        reference for a reserve is ``{"hours": [...], "minimum_energy_kwh": n}``
        with no ``"factor": null`` riding along. `response_model_exclude_none`
        is not an option here because `no_op` genuinely needs
        ``structured_adjustment: null`` at the parent level.
        """
        return {k: v for k, v in self.__dict__.items() if v is not None}


class DirectiveInterpretation(BaseModel):
    """One machine-checkable interpretation entry per operator note (10.2)."""

    model_config = ConfigDict(extra="forbid")

    note_index: Annotated[int, Field(ge=0)] = Field(
        description="Zero-based index into the request's operator_notes."
    )
    applies: bool = Field(
        description="True for every applicable directive; false only for no_op."
    )
    directive_type: DirectiveType
    structured_adjustment: StructuredAdjustment | None = Field(
        default=None, description="Required object, or null only for no_op."
    )
    explanation: str = Field(
        default="", description="Short explanation of the interpretation."
    )

    @model_validator(mode="after")
    def _check_type_coupling(self) -> Self:
        """Enforce section 05.1: no_op <=> applies=false <=> adjustment=null."""
        t = self.directive_type
        adj = self.structured_adjustment

        if t is DirectiveType.NO_OP:
            if self.applies:
                raise ValueError("no_op must use applies=false")
            if adj is not None:
                raise ValueError("no_op must use structured_adjustment=null")
            return self

        if not self.applies:
            raise ValueError(f"{t.value} must use applies=true")
        if adj is None:
            raise ValueError(f"{t.value} requires a structured_adjustment object")

        required = TYPE_REQUIRED_FIELD[t]
        if required is not None and getattr(adj, required) is None:
            raise ValueError(f"{t.value} requires structured_adjustment.{required}")

        # Window-only directives must not carry stray numeric fields.
        allowed = {"hours"} | ({required} if required else set())
        for name in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
            if name not in allowed and getattr(adj, name) is not None:
                raise ValueError(f"{t.value} must not set structured_adjustment.{name}")
        return self


class HourlyPlanEntry(BaseModel):
    """One hour of the final schedule (section 10.3)."""

    model_config = ConfigDict(extra="forbid")

    hour: Annotated[int, Field(ge=0, le=23)]
    grid_kwh: NonNegative = Field(description="Grid energy purchased this hour.")
    solar_used_kwh: NonNegative = Field(
        description="Solar used; must not exceed effective solar."
    )
    battery_action: BatteryAction
    battery_kwh: NonNegative = Field(
        description="Magnitude of the battery action; 0 when idle."
    )
    battery_energy_after_kwh: NonNegative = Field(
        description="Battery energy immediately after this hour."
    )

    @model_validator(mode="after")
    def _idle_means_zero(self) -> Self:
        if self.battery_action is BatteryAction.IDLE and self.battery_kwh != 0:
            raise ValueError("battery_kwh must be 0 when battery_action is idle")
        return self


class OptimizeEnergyResponse(BaseModel):
    """Body of a successful POST /optimize-energy (section 10.1)."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(description="Echoes the request scenario_id.")
    directive_interpretation: list[DirectiveInterpretation] = Field(
        description="One entry for every operator note, in note_index order."
    )
    hourly_plan: list[HourlyPlanEntry] = Field(
        description="One entry for every hour 0 through 23."
    )
    total_grid_kwh: NonNegative = Field(description="Sum of grid_kwh over 24 hours.")
    total_cost_bdt: NonNegative = Field(description="Total grid electricity cost.")
    peak_grid_kwh: NonNegative = Field(description="Maximum hourly grid_kwh.")
    plan_summary: str = Field(
        description="Short human-readable explanation of the final strategy."
    )

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        if len(self.hourly_plan) != HOURS_PER_DAY:
            raise ValueError(f"hourly_plan must contain {HOURS_PER_DAY} entries")
        if [p.hour for p in self.hourly_plan] != list(range(HOURS_PER_DAY)):
            raise ValueError("hourly_plan must cover hours 0-23 in order")
        indices = [d.note_index for d in self.directive_interpretation]
        if indices != list(range(len(indices))):
            raise ValueError(
                "directive_interpretation must be one entry per note, in "
                "note_index order 0..N-1 with no gaps or duplicates"
            )
        return self


class HealthResponse(BaseModel):
    """GET /health body -- exactly the shape in Problem Statement 06.2."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="ok", pattern="^ok$")
