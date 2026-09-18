"""MOCK response builder for POST /optimize-energy.

Placeholder only. It returns a structurally valid, contract-conformant
response so the API surface can be exercised end to end before the real
pipeline exists. It performs NO interpretation and NO optimization:

  * every operator note is reported as ``no_op`` -- nothing has actually
    been interpreted yet, so claiming a directive would be a lie;
  * the battery stays idle all day, which trivially satisfies end-of-day
    neutrality;
  * solar is used up to demand and the remainder is bought from the grid.

That schedule is valid under the GridWise energy rules, but it is NOT
cost-optimal and it ignores every directive. Replace the body of
:func:`build_mock_response` with the real LLM + optimizer pipeline; the
route and the schemas do not need to change.
"""

from __future__ import annotations

from app.schemas.common import BatteryAction, DirectiveType
from app.schemas.energy import (
    DirectiveInterpretation,
    HourlyPlanEntry,
    OptimizeEnergyRequest,
    OptimizeEnergyResponse,
)

_DP = 6


def build_mock_response(request: OptimizeEnergyRequest) -> OptimizeEnergyResponse:
    """Return a mock, schema-valid response for the given request."""
    plan: list[HourlyPlanEntry] = []
    for hour in request.hours:
        solar_used = min(hour.solar_kwh, hour.demand_kwh)
        grid = max(0.0, hour.demand_kwh - solar_used)
        plan.append(
            HourlyPlanEntry(
                hour=hour.hour,
                grid_kwh=round(grid, _DP),
                solar_used_kwh=round(solar_used, _DP),
                battery_action=BatteryAction.IDLE,
                battery_kwh=0.0,
                # Idle all day, so the level never moves off its start value.
                battery_energy_after_kwh=request.battery.initial_energy_kwh,
            )
        )

    total_grid = round(sum(p.grid_kwh for p in plan), _DP)
    total_cost = round(
        sum(
            p.grid_kwh * request.hours[p.hour].tariff_bdt_per_kwh for p in plan
        ),
        _DP,
    )
    peak = round(max(p.grid_kwh for p in plan), _DP)

    return OptimizeEnergyResponse(
        scenario_id=request.scenario_id,
        directive_interpretation=[
            DirectiveInterpretation(
                note_index=i,
                applies=False,
                directive_type=DirectiveType.NO_OP,
                structured_adjustment=None,
                explanation="Stub: operator-note interpretation is not implemented yet.",
            )
            for i in range(len(request.operator_notes))
        ],
        hourly_plan=plan,
        total_grid_kwh=total_grid,
        total_cost_bdt=total_cost,
        peak_grid_kwh=peak,
        plan_summary=(
            "MOCK RESPONSE - no interpretation or optimization performed. "
            "Solar is used up to demand, the shortfall is bought from the "
            "grid, and the battery is left idle. Not cost-optimal."
        ),
    )
