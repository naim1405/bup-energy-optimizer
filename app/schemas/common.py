"""Shared enums for the API and LLM schemas."""

from __future__ import annotations

from enum import StrEnum


class DirectiveType(StrEnum):
    """The six supported operator-directive types (Problem Statement 04.1).

    `no_op` is the only type allowed with ``applies = false``.
    """

    SOLAR_REDUCTION = "solar_reduction"
    MINIMUM_BATTERY_RESERVE = "minimum_battery_reserve"
    NO_CHARGE_WINDOW = "no_charge_window"
    NO_DISCHARGE_WINDOW = "no_discharge_window"
    MAX_GRID_WINDOW = "max_grid_window"
    NO_OP = "no_op"

    @property
    def applies(self) -> bool:
        """Whether this directive type can ever be marked applicable."""
        return self is not DirectiveType.NO_OP


#: Directive types whose structured_adjustment carries an extra numeric field.
TYPE_REQUIRED_FIELD: dict[DirectiveType, str | None] = {
    DirectiveType.SOLAR_REDUCTION: "factor",
    DirectiveType.MINIMUM_BATTERY_RESERVE: "minimum_energy_kwh",
    DirectiveType.NO_CHARGE_WINDOW: None,
    DirectiveType.NO_DISCHARGE_WINDOW: None,
    DirectiveType.MAX_GRID_WINDOW: "max_grid_kwh",
    DirectiveType.NO_OP: None,
}


class BatteryAction(StrEnum):
    """Exactly one of these per hour (Problem Statement 10.3)."""

    CHARGE = "charge"
    DISCHARGE = "discharge"
    IDLE = "idle"


HOURS_PER_DAY = 24
