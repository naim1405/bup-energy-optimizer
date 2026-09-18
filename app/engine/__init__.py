"""The GridWise math engine.

Pure functions over plain dataclasses: normalized scenario in, schedule out.
No pydantic, no FastAPI, no I/O -- see ``app/services/energy_optimizer.py``
for the adapter that bridges this to the API schemas.

Model
-----
A plain linear program. Charge and discharge are 1:1 (Problem Statement 9.1
has no efficiency loss), so no integer variables are needed even though
``battery_action`` looks categorical.

Variables, 5 per hour (120 total):
    grid[h]        >= 0        grid energy purchased
    solar[h]       >= 0        solar energy actually used
    charge[h]      >= 0        energy into the battery
    discharge[h]   >= 0        energy out of the battery
    E[h]                       battery energy AFTER hour h

Minimize  sum(grid[h] * tariff[h])

Subject to
    grid + solar + discharge - charge = demand            (balance, 9.5)
    E[h] = E[h-1] + charge - discharge                    (state,   9.1)
    E[23] = initial_energy_kwh                            (neutrality, 9.6)
    0 <= solar[h] <= effective_solar[h]                   (9.4)
    min_floor[h] <= E[h] <= capacity                      (9.2)
    charge[h]    <= max_charge_kwh_per_hour               (9.3)
    discharge[h] <= max_discharge_kwh_per_hour            (9.3)
    grid[h]      <= grid_cap[h]                           (max_grid_window)
"""

from app.engine.models import (
    HOURS_PER_DAY,
    ROUND_DP,
    BatterySpec,
    Directive,
    HourInput,
    HourPlan,
    Scenario,
    ScheduleResult,
)
from app.engine.optimizer import Overlay, build_overlay, optimize
from app.engine.validate import make_validator, replay

__all__ = [
    "HOURS_PER_DAY",
    "ROUND_DP",
    "BatterySpec",
    "Directive",
    "HourInput",
    "HourPlan",
    "Overlay",
    "Scenario",
    "ScheduleResult",
    "build_overlay",
    "make_validator",
    "optimize",
    "replay",
]
