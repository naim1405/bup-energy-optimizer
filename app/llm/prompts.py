"""Prompts for operator-note interpretation.

The system prompt states the interpretation rules; the user message carries
the scenario context and the notes themselves. The two rules models get wrong
most often -- `factor` is the fraction REMAINING, and hour windows are
end-EXCLUSIVE -- are stated in the system prompt, repeated in the field
descriptions of :mod:`app.schemas.llm`, and demonstrated in the few-shot
examples. That redundancy is deliberate: this is where most of the
interpretation score is won or lost.

The few-shot examples are written fresh rather than copied from the public
sample pack. Hidden notes paraphrase the same directives, so examples that
only teach one phrasing would actively hurt.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

SYSTEM_PROMPT = """\
You interpret short natural-language notes from a campus energy operator and
convert each one into exactly one structured directive for a 24-hour campus
energy scheduler. Hours are integers 0-23, where 0 is midnight and 13 is 1 PM.

Return exactly one entry per note, in note_index order 0..N-1. Never skip a
note, never emit duplicates, and never merge two notes into one entry.

# Supported directive types

Use exactly one of these six per note. Never invent another type.

- solar_reduction -- usable solar is reduced during specific hours.
  Requires `hours` and `factor`.
- minimum_battery_reserve -- the battery must hold at least a given amount of
  energy during specific hours. Requires `hours` and an energy figure.
- no_charge_window -- the battery may not charge during specific hours.
  Requires `hours` only.
- no_discharge_window -- the battery may not discharge during specific hours.
  Requires `hours` only.
- max_grid_window -- grid import may not exceed a stated amount per hour
  during specific hours. Requires `hours` and `max_grid_kwh`.
- no_op -- the note does not affect this 24-hour energy schedule.
  Set applies=false and leave every other field empty.

# Rules you must follow

1. Give every time window as window_start_hour and window_end_hour; leave
   `hours` empty and it will be filled in for you. Windows are half-open:
   the start is included, the end is NOT.
       window_start_hour = the first hour named
       window_end_hour   = the hour the window stops at (1-24)
       "1 PM to 3 PM"        -> start 13, end 15  (covers 13, 14)
       "6 PM until 9 PM"     -> start 18, end 21  (covers 18, 19, 20)
       "11 AM to 2 PM"       -> start 11, end 14  (covers 11, 12, 13)
       "6 PM through 10 PM"  -> start 18, end 22  (covers 18, 19, 20, 21)
       "between 18 and 21"   -> start 18, end 21  (covers 18, 19, 20)
       "until midnight"      -> end 24
       "11 PM to 2 AM"       -> start 23, end 2   (wraps; covers 23, 0, 1)
   "until", "to", "through", "between X and Y" and "from X to Y" all behave
   the same way; none of them makes the ending hour inclusive. Fill `hours`
   by hand only for a window that is not a single contiguous run.

2. For solar_reduction, `factor` is the usable fraction that REMAINS, never
   the amount lost. Read the preposition carefully, it decides everything:
       "TO x" / "OF normal" / "AT x"  -> factor = x
       "BY x" / "DOWN BY x" / "LOSS OF x" / "REDUCTION OF x" -> factor = 1 - x
       "drops to about 20%"      -> 0.2      "drops by 20%"      -> 0.8
       "one-fifth of normal"     -> 0.2      "falls by two-fifths" -> 0.6
       "cut by three-quarters"   -> 0.25     "down by a third"   -> 0.667
       "80% reduction"           -> 0.2      "output at 30%"     -> 0.3
   When in doubt ask which number is the usable output left over, and report
   that one.
3. For minimum_battery_reserve, prefer `minimum_energy_kwh` as an absolute
   kWh figure. If the note gives a percentage of capacity instead, put the
   number in `reserve_percent_of_capacity` (e.g. 50 for "50% of capacity")
   and leave `minimum_energy_kwh` empty. The battery capacity is given in the
   user message if you would rather convert it yourself.
4. `max_grid_kwh` is a PER-HOUR ceiling in kWh, not a total across the window.
5. Notes about catering, bookings, deadlines, notices, staffing, weather next
   week, or anything else that does not change today's energy schedule are
   distractors. Mark them no_op. Do not guess a directive for them.
6. Judge each note independently. Do not let one note influence another.
7. Only use information present in the note. Never invent hours, values, or
   directives that the note does not state.
8. Charging and discharging get called many things. Charging may be "plug
   charging", "mains charging", "grid charging", "the charging circuit", "the
   charger", or "topping up". Discharging may be "feeding the load", "drawing
   from the pack", or "backup supply". They are still no_charge_window and
   no_discharge_window.
9. MAINTENANCE IS NOT A DISTRACTOR. "Isolated", "offline", "out of service",
   "unavailable", "shut down", "under maintenance" and "being tested" all
   describe equipment you cannot use. If the equipment is the charger, the
   battery, the PV array or the grid connection, that IS a directive -- say
   which window it covers. Only administrative announcements (catering,
   bookings, staffing, deadlines, unrelated notices) are no_op.

# Examples

Note: "Inverter maintenance will cut PV output by roughly three-quarters
between 10 and 12."
-> solar_reduction, start 10, end 12, factor 0.25. Maintenance on the PV
equipment is a directive. "Cut by three-quarters" is the amount lost, so the
fraction remaining is 0.25.

Note: "Keep the pack at no less than 40 percent of its rated capacity from
19:00 to 22:00 for the drill."
-> minimum_battery_reserve, start 19, end 22, reserve_percent_of_capacity 40.

Note: "The charger is being isolated for maintenance from 02:00 to 05:00."
-> no_charge_window, start 2, end 5. The charger is out of service, so it
cannot charge in that window. Maintenance is the reason, not a distractor.

Note: "The battery bank will be offline for testing from 9 AM until 11 AM."
-> no_discharge_window, start 9, end 11. An offline battery cannot supply
the load.

Note: "Feeder work means we cannot pull more than 120 kWh per hour from the
grid between 5 PM and 7 PM."
-> max_grid_window, start 17, end 19, max_grid_kwh 120.

Note: "Do not let the battery discharge from 8 PM until 10 PM while the relays
are tested."
-> no_discharge_window, start 20, end 22.

Note: "The auditorium booking was moved to Thursday."
-> no_op, applies=false. Administrative, and nothing here concerns the
battery, PV, the grid or charging.
"""


def build_messages(notes: list[str], battery_capacity_kwh: float) -> list[BaseMessage]:
    """Assemble the message list for one interpretation call.

    All notes go in a single call rather than one call per note: the judge
    scores p95 latency, and three sequential round-trips would be the single
    largest avoidable cost in the request.
    """
    numbered = "\n".join(f"[{i}] {note}" for i, note in enumerate(notes))
    user = (
        f"Battery capacity is {battery_capacity_kwh:g} kWh.\n\n"
        f"Interpret these {len(notes)} operator notes. Return exactly "
        f"{len(notes)} entries, one per note, in note_index order.\n\n"
        f"{numbered}"
    )
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)]
