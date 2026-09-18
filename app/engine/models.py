"""Plain dataclasses for the optimization engine.

The engine is deliberately framework-free: no pydantic, no FastAPI, no I/O.
Request validation belongs to the pydantic layer in ``app/schemas``; this
module only describes the already-validated data the optimizer consumes and
produces. ``app/services/energy_optimizer.py`` adapts between the two.

Keeping it pure is what makes it unit-testable without a network, which
matters because the LLM call upstream of it may be slow or unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --- Directive vocabulary (Problem Statement 04.1) --------------------------
SOLAR_REDUCTION = "solar_reduction"
MIN_BATTERY_RESERVE = "minimum_battery_reserve"
NO_CHARGE_WINDOW = "no_charge_window"
NO_DISCHARGE_WINDOW = "no_discharge_window"
MAX_GRID_WINDOW = "max_grid_window"
NO_OP = "no_op"

DIRECTIVE_TYPES = frozenset({
    SOLAR_REDUCTION, MIN_BATTERY_RESERVE, NO_CHARGE_WINDOW,
    NO_DISCHARGE_WINDOW, MAX_GRID_WINDOW, NO_OP,
})

HOURS_PER_DAY = 24

#: Judge tolerance is 0.01 absolute (section 11.5); 6dp sits far inside it.
ROUND_DP = 6


@dataclass(frozen=True)
class HourInput:
    hour: int
    demand_kwh: float
    solar_kwh: float
    tariff_bdt_per_kwh: float


@dataclass(frozen=True)
class BatterySpec:
    capacity_kwh: float
    initial_energy_kwh: float
    minimum_energy_kwh: float
    max_charge_kwh_per_hour: float
    max_discharge_kwh_per_hour: float


@dataclass(frozen=True)
class Directive:
    """A validated directive -- the only thing that crosses into the math."""

    note_index: int
    directive_type: str
    hours: tuple[int, ...] = ()
    factor: float | None = None
    minimum_energy_kwh: float | None = None
    max_grid_kwh: float | None = None
    explanation: str = ""

    @property
    def applies(self) -> bool:
        return self.directive_type != NO_OP

    def adjustment(self) -> dict[str, Any] | None:
        """The exact ``structured_adjustment`` object, derived from the same
        fields the engine reads, so reported and applied cannot drift apart."""
        if not self.applies:
            return None
        if self.directive_type == SOLAR_REDUCTION:
            return {"hours": list(self.hours), "factor": self.factor}
        if self.directive_type == MIN_BATTERY_RESERVE:
            return {"hours": list(self.hours),
                    "minimum_energy_kwh": self.minimum_energy_kwh}
        if self.directive_type == MAX_GRID_WINDOW:
            return {"hours": list(self.hours), "max_grid_kwh": self.max_grid_kwh}
        return {"hours": list(self.hours)}


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    hours: tuple[HourInput, ...]
    battery: BatterySpec
    directives: tuple[Directive, ...] = ()

    @property
    def demand(self) -> list[float]:
        return [h.demand_kwh for h in self.hours]

    @property
    def solar(self) -> list[float]:
        return [h.solar_kwh for h in self.hours]

    @property
    def tariff(self) -> list[float]:
        return [h.tariff_bdt_per_kwh for h in self.hours]


@dataclass(frozen=True)
class HourPlan:
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: str            # charge | discharge | idle
    battery_kwh: float
    battery_energy_after_kwh: float


@dataclass
class ScheduleResult:
    plan: list[HourPlan]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    feasible: bool                 # False => relaxed fallback plan in use
    violations: list[str] = field(default_factory=list)
    overlay_notes: list[str] = field(default_factory=list)
