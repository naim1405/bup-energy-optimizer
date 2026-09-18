"""POST /optimize-energy -- operator-note interpretation + 24-hour schedule."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.schemas.energy import OptimizeEnergyRequest, OptimizeEnergyResponse
from app.services.mock_optimizer import build_mock_response

router = APIRouter(tags=["optimize"])

_EXAMPLE_RESPONSE = {
    "scenario_id": "GRID-101",
    "directive_interpretation": [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "solar_reduction",
            "structured_adjustment": {"hours": [13, 14], "factor": 0.2},
            "explanation": "Solar availability is reduced during panel cleaning.",
        },
        {
            "note_index": 1,
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": "This note does not affect today's energy schedule.",
        },
    ],
    "hourly_plan": [
        {
            "hour": 0,
            "grid_kwh": 130.0,
            "solar_used_kwh": 0.0,
            "battery_action": "idle",
            "battery_kwh": 0.0,
            "battery_energy_after_kwh": 200.0,
        },
        {
            "hour": 1,
            "grid_kwh": 95.0,
            "solar_used_kwh": 30.0,
            "battery_action": "discharge",
            "battery_kwh": 25.0,
            "battery_energy_after_kwh": 175.0,
        },
    ],
    "total_grid_kwh": 2430.0,
    "total_cost_bdt": 35480.0,
    "peak_grid_kwh": 205.0,
    "plan_summary": (
        "Applied one solar-reduction directive, ignored an unrelated note, and "
        "shifted battery energy toward high-tariff hours while restoring the "
        "initial battery level."
    ),
}


@router.post(
    "/optimize-energy",
    response_model=OptimizeEnergyResponse,
    status_code=status.HTTP_200_OK,
    summary="Interpret operator notes and return a 24-hour energy schedule",
    responses={
        200: {
            "description": "Interpretation plus the final 24-hour plan.",
            "content": {"application/json": {"example": _EXAMPLE_RESPONSE}},
        },
        400: {
            "description": "Malformed JSON or structurally invalid request.",
        },
    },
)
def optimize_energy(
    request: OptimizeEnergyRequest,
) -> OptimizeEnergyResponse:
    """Return the operator-note interpretation and the final 24-hour schedule.

    NOTE: the body is a stub. It currently returns a mock plan and reports
    every note as ``no_op``. The LLM interpretation layer and the optimizer
    are not wired in yet.
    """
    return build_mock_response(request)
