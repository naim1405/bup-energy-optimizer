"""The suite must not depend on the developer's environment.

This exists because of a real failure: ``Settings`` loads ``.env``, so a key
in the environment sent the integration tests down the live-model path while
a clean checkout ran them down the no-key fallback. The same suite passed in
one place and failed in another, and took 86s instead of 3s.

``conftest.py`` strips the key for every test. This module asserts that
actually happened. It lives here rather than in ``conftest.py`` because
pytest never collects ``conftest.py`` as a test module -- a guard placed
there silently never runs.
"""

from __future__ import annotations

import os


def test_no_api_key_is_visible_to_tests() -> None:
    from app.core.config import settings

    assert os.environ.get("OPENAI_API_KEY") is None
    assert settings.openai_api_key is None


def test_interpretation_falls_back_without_reaching_the_network() -> None:
    """No key means an immediate no_op fallback, not a hanging request."""
    import time

    from app.llm.interpreter import interpret_notes

    t0 = time.perf_counter()
    out = interpret_notes(["Keep the battery above 50% from 6 PM to 9 PM."], 200.0)
    elapsed = time.perf_counter() - t0

    assert [e.directive_type.value for e in out] == ["no_op"]
    assert elapsed < 1.0, f"{elapsed:.2f}s suggests a network call was attempted"
