"""Application entry point.

Creates the FastAPI app, wires up configuration and mounts API routers.
"""

from fastapi import FastAPI

from app.api.routes import health
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
)

app.include_router(health.router)


@app.get("/")
def root() -> dict:
    """Simple index with pointers to the health endpoint and docs."""
    return {
        "message": f"{settings.app_name} is 100% Fully Automated! ⚡🚀",
        "pipeline": "GitHub Actions -> Docker Hub -> Watchtower -> DigitalOcean",
        "health": "/health",
        "docs": "/docs",
    }


