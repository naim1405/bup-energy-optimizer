"""LLM layer: operator-note interpretation.

The model lives here and nowhere else. Its structured output is validated by
``app/schemas/llm.py`` before any of it reaches the optimizer.
"""

from app.llm.interpreter import get_structured_llm, interpret_notes, reconcile
from app.llm.prompts import SYSTEM_PROMPT, build_messages

__all__ = [
    "SYSTEM_PROMPT",
    "build_messages",
    "get_structured_llm",
    "interpret_notes",
    "reconcile",
]
