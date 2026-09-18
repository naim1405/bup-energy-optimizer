"""Shared test configuration.

The one job of this file is to make the suite hermetic: **no test may reach
the network, and no test may depend on the developer's environment.**

This exists because of a real failure. ``Settings`` is configured with
``env_file=".env"``, so a developer with a key in their ``.env`` (or exported
in their shell) ran the suite down the live-model path while CI and a clean
checkout ran it down the no-key fallback. Five integration tests passed in
one environment and failed in the other, and the suite took 86s instead of
3s. The tests were not wrong about the engine -- they were silently measuring
two different systems.

So every test starts with the key removed from both places it can come from:
the process environment and the already-constructed settings singleton. With
no key, ``get_structured_llm()`` raises at construction, ``interpret_notes``
falls back to ``no_op`` immediately, and nothing touches the network.

Tests that specifically want a key construct their own ``Settings()`` or
``monkeypatch.setenv`` -- pydantic-settings gives the environment priority
over ``.env``, so those keep working. Language understanding is covered by
``tests/test_llm.py`` with fake LLMs, and against the real model by
``scripts/eval_interpretation.py``, which is a dev tool rather than a test.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_ambient_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip the API key from the environment and the settings singleton."""
    from app.core.config import settings
    from app.llm.interpreter import get_structured_llm

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", None, raising=False)

    # Construction is cached, so a client built before this fixture ran would
    # otherwise survive into tests that expect the fallback.
    get_structured_llm.cache_clear()
    try:
        yield
    finally:
        get_structured_llm.cache_clear()
