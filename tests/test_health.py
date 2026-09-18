"""Tests for the /health endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_body_is_exactly_the_specified_shape() -> None:
    """Problem Statement 06.2 shows the body as exactly {"status": "ok"}."""
    body = client.get("/health").json()
    assert set(body) == {"status"}
    assert body["status"] == "ok"
