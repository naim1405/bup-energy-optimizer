"""Health-check endpoint."""

from fastapi import APIRouter

from app.schemas.energy import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Readiness probe used by the judging harness, Docker and nginx.

    Returns exactly ``{"status": "ok"}`` as specified in Problem Statement
    06.2. Service name and version are intentionally not included here -- a
    strict equality check on the health body must pass. They remain available
    at ``/`` and in the OpenAPI document.
    """
    return HealthResponse(status="ok")
