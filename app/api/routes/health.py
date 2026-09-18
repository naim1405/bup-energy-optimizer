"""Health-check endpoint."""

from fastapi import APIRouter

from app.core.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    """Liveness/readiness probe used by Docker and nginx."""
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.version,
    }
