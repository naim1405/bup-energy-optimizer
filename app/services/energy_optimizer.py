"""Bridge between the API schemas and the pure math engine.

Pipeline::

    request (pydantic)
      -> interpret_notes()                 [app/llm, gpt-4o-mini]
      -> NoteInterpretation.to_api_model() -> DirectiveInterpretation
      -> engine Directive
      -> engine.optimize()                 -> ScheduleResult
      -> OptimizeEnergyResponse

Everything downstream of interpretation -- the ``to_api_model`` projection,
the engine call and the response assembly -- is unchanged by which model
produces the directives. Tests inject a fake at this seam.
"""

from __future__ import annotations

import logging

from app.engine.models import (
    ROUND_DP,
    BatterySpec,
    Directive,
    HourInput,
    Scenario,
    ScheduleResult,
)
from app.engine.optimizer import optimize
from app.engine.validate import make_validator
# Imported by name so tests can monkeypatch
# ``app.services.energy_optimizer.interpret_notes`` without touching app/llm.
from app.llm.interpreter import interpret_notes
from app.schemas.common import DirectiveType
from app.schemas.energy import (
    DirectiveInterpretation,
    HourlyPlanEntry,
    OptimizeEnergyRequest,
    OptimizeEnergyResponse,
)

logger = logging.getLogger(__name__)

_SUMMARY_BY_TYPE = {
    DirectiveType.SOLAR_REDUCTION: "reduced usable solar",
    DirectiveType.MINIMUM_BATTERY_RESERVE: "a raised battery reserve floor",
    DirectiveType.NO_CHARGE_WINDOW: "charging blocked in a maintenance window",
    DirectiveType.NO_DISCHARGE_WINDOW: "discharging blocked in a testing window",
    DirectiveType.MAX_GRID_WINDOW: "a capped grid import",
}


# ===========================================================================
# Adapt pydantic <-> engine dataclasses
# ===========================================================================


def to_engine_directive(entry: DirectiveInterpretation) -> Directive:
    """Project a contract interpretation entry into the engine's directive."""
    adj = entry.structured_adjustment
    return Directive(
        note_index=entry.note_index,
        directive_type=entry.directive_type.value,
        hours=tuple(adj.hours) if adj else (),
        factor=adj.factor if adj else None,
        minimum_energy_kwh=adj.minimum_energy_kwh if adj else None,
        max_grid_kwh=adj.max_grid_kwh if adj else None,
        explanation=entry.explanation,
    )


def to_scenario(
    request: OptimizeEnergyRequest, entries: list[DirectiveInterpretation]
) -> Scenario:
    """Build the engine's Scenario from a validated request."""
    b = request.battery
    return Scenario(
        scenario_id=request.scenario_id,
        hours=tuple(
            HourInput(
                hour=h.hour,
                demand_kwh=h.demand_kwh,
                solar_kwh=h.solar_kwh,
                tariff_bdt_per_kwh=h.tariff_bdt_per_kwh,
            )
            for h in request.hours
        ),
        battery=BatterySpec(
            capacity_kwh=b.capacity_kwh,
            initial_energy_kwh=b.initial_energy_kwh,
            minimum_energy_kwh=b.minimum_energy_kwh,
            max_charge_kwh_per_hour=b.max_charge_kwh_per_hour,
            max_discharge_kwh_per_hour=b.max_discharge_kwh_per_hour,
        ),
        directives=tuple(to_engine_directive(e) for e in entries),
    )


# ===========================================================================
# Response assembly
# ===========================================================================


def _plan_summary(
    entries: list[DirectiveInterpretation], result: ScheduleResult, n_notes: int
) -> str:
    applied = [e for e in entries if e.applies]
    ignored = n_notes - len(applied)
    if applied:
        bits = ", ".join(
            _SUMMARY_BY_TYPE.get(e.directive_type, e.directive_type.value)
            for e in applied
        )
        head = f"Applied {len(applied)} operator directive(s): {bits}."
    else:
        head = "No operator directive affected this schedule."
    tail = (
        " Battery energy was shifted toward high-tariff hours and returned to "
        f"its starting level by end of day; peak grid draw "
        f"{result.peak_grid_kwh:g} kWh, total cost {result.total_cost_bdt:g} BDT."
    )
    if ignored:
        tail += f" {ignored} note(s) were irrelevant and ignored."
    if not result.feasible:
        tail += (
            " WARNING: the interpreted directives were not simultaneously "
            "satisfiable, so this is the closest feasible plan."
        )
    return head + tail


def to_response(
    request: OptimizeEnergyRequest,
    entries: list[DirectiveInterpretation],
    result: ScheduleResult,
) -> OptimizeEnergyResponse:
    """Assemble the contract response from the engine's schedule."""
    return OptimizeEnergyResponse(
        scenario_id=request.scenario_id,
        directive_interpretation=entries,
        hourly_plan=[
            HourlyPlanEntry(
                hour=p.hour,
                grid_kwh=p.grid_kwh,
                solar_used_kwh=p.solar_used_kwh,
                battery_action=p.battery_action,
                battery_kwh=round(p.battery_kwh, ROUND_DP),
                battery_energy_after_kwh=p.battery_energy_after_kwh,
            )
            for p in result.plan
        ],
        total_grid_kwh=result.total_grid_kwh,
        total_cost_bdt=result.total_cost_bdt,
        peak_grid_kwh=result.peak_grid_kwh,
        plan_summary=_plan_summary(entries, result, len(request.operator_notes)),
    )


# ===========================================================================
# Entry point
# ===========================================================================


def optimize_scenario(request: OptimizeEnergyRequest) -> OptimizeEnergyResponse:
    """Run the full math path for one request.

    The engine validates its own output before returning; any violation is
    logged rather than sent to the client, because the response schema is
    closed (``extra="forbid"``) and the judge recalculates everything from
    ``hourly_plan`` anyway.
    """
    capacity = request.battery.capacity_kwh
    notes = interpret_notes(request.operator_notes, capacity)
    entries = [n.to_api_model(capacity) for n in notes]

    scenario = to_scenario(request, entries)
    result = optimize(scenario, validate_fn=make_validator())

    if result.violations:
        logger.warning(
            "scenario %s produced %d self-validation violation(s): %s",
            request.scenario_id, len(result.violations), result.violations[:5],
        )
    if not result.feasible:
        logger.warning(
            "scenario %s was infeasible under the interpreted directives; "
            "returned the relaxed fallback plan", request.scenario_id,
        )
    return to_response(request, entries, result)
