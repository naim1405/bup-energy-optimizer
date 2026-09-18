"""Operator-note interpretation via LangChain.

This is the module the Problem Statement's LLM requirement points at: a
language-capable model produces the structured interpretation that the
optimizer then consumes. It is not cosmetic -- the directives returned here
become hard constraints in the linear program.

Failure handling is the important part. The model is untrusted and the
provider can be slow, rate-limited, or down, so this module never raises.
It retries once inside a hard time budget and then falls back to ``no_op``
for every note, which keeps the request valid instead of returning a 500.
Losing interpretation credit is recoverable; an unavailable service is not.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any, NamedTuple

from app.core.config import settings
from app.llm.prompts import build_messages
from app.schemas.common import DirectiveType
from app.schemas.llm import NoteInterpretation, NoteInterpretationResult

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_structured_llm() -> Any:
    """Build (once) a chat model bound to our structured-output schema.

    Cached because constructing a client per request would add latency for no
    benefit. ``cache_clear()`` in tests resets it.
    """
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "timeout": settings.llm_timeout_seconds,
        "max_retries": settings.llm_max_retries,
    }
    # Passed explicitly so a `.env` file alone configures the service; the
    # SDK's own environment lookup would miss keys that only pydantic loaded.
    if settings.openai_api_key is not None:
        kwargs["api_key"] = settings.openai_api_key.get_secret_value()

    return ChatOpenAI(**kwargs).with_structured_output(
        NoteInterpretationResult,
        include_raw=False,
    )


def _no_op(note_index: int, why: str) -> NoteInterpretation:
    return NoteInterpretation(
        note_index=note_index,
        applies=False,
        directive_type=DirectiveType.NO_OP,
        explanation=why,
    )


def _all_no_op(n_notes: int, why: str) -> list[NoteInterpretation]:
    return [_no_op(i, why) for i in range(n_notes)]


class Reconciled(NamedTuple):
    """Result of forcing model output into the required one-entry-per-note shape.

    ``complete`` is False when any note had to be filled in locally, which is
    the caller's signal that a retry may recover the missing interpretation.
    """

    entries: list[NoteInterpretation] | None
    complete: bool


def _extract_raw(result: Any) -> Any:
    if isinstance(result, NoteInterpretationResult):
        return result.interpretations
    if isinstance(result, dict):
        return result.get("interpretations")
    return getattr(result, "interpretations", None)


def reconcile(result: Any, n_notes: int) -> Reconciled:
    """Force the model's output into exactly one entry per note, in order.

    ``entries`` is ``None`` if nothing usable came back. Missing indices are
    filled with ``no_op``; out-of-range indices and duplicates are dropped.
    Entries that arrive as plain dicts (some providers skip schema
    enforcement) are validated here.
    """
    raw = _extract_raw(result)
    if not isinstance(raw, (list, tuple)):
        return Reconciled(None, False)

    by_index: dict[int, NoteInterpretation] = {}
    for item in raw:
        try:
            entry = (item if isinstance(item, NoteInterpretation)
                     else NoteInterpretation.model_validate(item))
        except Exception:                                   # noqa: BLE001
            logger.warning("dropping an unparseable interpretation entry: %r", item)
            continue
        if not 0 <= entry.note_index < n_notes:
            logger.warning("dropping note_index %s (out of range for %d notes)",
                           entry.note_index, n_notes)
            continue
        if entry.note_index in by_index:
            logger.warning("duplicate note_index %s; keeping the first",
                           entry.note_index)
            continue
        by_index[entry.note_index] = entry

    if not by_index:
        return Reconciled(None, False)

    entries = [
        by_index.get(i)
        or _no_op(i, "The model returned no usable interpretation for this note.")
        for i in range(n_notes)
    ]
    return Reconciled(entries, len(by_index) == n_notes)


def interpret_notes(
    notes: list[str],
    battery_capacity_kwh: float,
    llm: Any | None = None,
) -> list[NoteInterpretation]:
    """Interpret every operator note. Always returns one entry per note.

    ``llm`` is injectable for tests; in production it is the cached
    structured-output model from :func:`get_structured_llm`.
    """
    n = len(notes)
    messages = build_messages(notes, battery_capacity_kwh)
    deadline = time.monotonic() + settings.llm_budget_seconds
    attempts = 2
    best: list[NoteInterpretation] | None = None

    for attempt in range(1, attempts + 1):
        if llm is None:
            try:
                llm = get_structured_llm()
            except Exception as exc:                        # noqa: BLE001
                logger.error("could not construct the LLM client: %s", exc)
                break
        try:
            result = llm.invoke(messages)
        except Exception as exc:                            # noqa: BLE001
            logger.warning(
                "interpretation attempt %d/%d failed: %s: %s",
                attempt, attempts, type(exc).__name__, exc,
            )
        else:
            rec = reconcile(result, n)
            if rec.entries is not None:
                # Keep the best partial result: some interpretation beats none.
                best = rec.entries
                if rec.complete:
                    if attempt > 1:
                        logger.info("interpretation recovered on attempt %d", attempt)
                    return rec.entries
                logger.warning(
                    "interpretation attempt %d/%d covered %d of %d notes",
                    attempt, attempts,
                    sum(1 for e in rec.entries
                        if e.directive_type is not DirectiveType.NO_OP
                        or e.applies),
                    n,
                )
            else:
                logger.warning(
                    "interpretation attempt %d/%d produced nothing usable",
                    attempt, attempts,
                )

        if attempt < attempts and time.monotonic() >= deadline:
            logger.warning("interpretation budget exhausted; not retrying")
            break

    if best is not None:
        logger.error(
            "returning a partial interpretation for %d note(s)", n,
        )
        return best

    logger.error(
        "falling back to no_op for all %d note(s); the schedule will be valid "
        "but will not honour any operator directive", n,
    )
    return _all_no_op(
        n,
        "Interpretation unavailable; this note was not applied to the schedule.",
    )
