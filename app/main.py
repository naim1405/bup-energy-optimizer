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
        "message": f"Welcome to {settings.app_name}! 🚀 Auto-Deploy is LIVE",
        "deployment": "Automated via GitHub Actions + Watchtower",
        "health": "/health",
        "docs": "/docs",
    }

