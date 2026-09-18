#!/usr/bin/env python3
"""Score operator-note interpretation against the public sample pack.

Mirrors the 25-point interpretation rubric: 5 for relevance/no_op, 5 for
directive_type, 5 for affected hours, 5 for numeric values and adjustment
shape. Paraphrase robustness (the fifth 5) needs notes the pack does not
contain, so `--paraphrase` runs a separate set of reworded notes.

    export OPENAI_API_KEY=sk-...
    uv run python scripts/eval_interpretation.py
    uv run python scripts/eval_interpretation.py --paraphrase
    uv run python scripts/eval_interpretation.py --runs 3

Nothing here is imported by the service; it is a development tool.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm.interpreter import get_structured_llm, interpret_notes  # noqa: E402

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "public_sample_cases.json"
TOL = 0.01

#: Capacity the paraphrase expectations are written against.
PARA_CAPACITY = 200.0

# Reworded variants of the public directives. Same meaning, different words,
# which is what the hidden set does.
PARAPHRASES = [
    (["PV production will fall to roughly a fifth of forecast between 13:00 "
      "and 15:00 while the inverters are serviced."],
     [{"directive_type": "solar_reduction", "hours": [13, 14], "factor": 0.2}]),
    (["Panel washing is scheduled from noon until 2 PM; treat usable output as "
      "about a quarter of normal during that time."],
     [{"directive_type": "solar_reduction", "hours": [12, 13], "factor": 0.25}]),
    (["We lose four-fifths of rooftop generation from 11 AM to 2 PM."],
     [{"directive_type": "solar_reduction", "hours": [11, 12, 13], "factor": 0.2}]),
    (["The charger is being isolated for maintenance between 02:00 and 05:00."],
     [{"directive_type": "no_charge_window", "hours": [2, 3, 4]}]),
    (["No charging after 2 in the afternoon until 4; the circuit is down."],
     [{"directive_type": "no_charge_window", "hours": [14, 15]}]),
    (["Relay testing means the battery must not discharge from 6 PM until 8 PM."],
     [{"directive_type": "no_discharge_window", "hours": [18, 19]}]),
    (["Half the battery has to stay in reserve from 18:00 to 21:00 for the drill."],
     [{"directive_type": "minimum_battery_reserve", "hours": [18, 19, 20],
       "minimum_energy_kwh": 100}]),   # 50% of the 200 kWh test battery
    (["Keep a minimum of 90 kWh stored from 6 PM through 10 PM for the hospital feed."],
     [{"directive_type": "minimum_battery_reserve", "hours": [18, 19, 20, 21],
       "minimum_energy_kwh": 90}]),
    (["The feeder is limited, so grid draw cannot go above 155 kWh in any hour "
      "from 6 PM until 9 PM."],
     [{"directive_type": "max_grid_window", "hours": [18, 19, 20],
       "max_grid_kwh": 155}]),
    (["The cafeteria menu changes tomorrow."], [{"directive_type": "no_op"}]),
    (["The library is extending book-return hours next week."],
     [{"directive_type": "no_op"}]),
    (["Cloud cover during the panel inspection will leave about half the "
      "forecast output from 10 AM until noon.",
      "The charging circuit will be unavailable from 2 PM until 4 PM.",
      "The library is extending book-return hours next week."],
     [{"directive_type": "solar_reduction", "hours": [10, 11], "factor": 0.5},
      {"directive_type": "no_charge_window", "hours": [14, 15]},
      {"directive_type": "no_op"}]),
]


def flatten(entry: dict) -> dict:
    """Normalize either shape into one flat dict for comparison.

    A response entry nests its payload under `structured_adjustment`; the
    paraphrase expectations are written flat. Comparing without flattening
    silently scores hours and numerics as trivially equal -- both sides read
    as None -- which is exactly the bug this guards against.
    """
    adj = entry.get("structured_adjustment")
    flat = {
        "directive_type": entry.get("directive_type"),
        "applies": entry.get("applies"),
    }
    src = adj if isinstance(adj, dict) else entry
    for k in ("hours", "factor", "minimum_energy_kwh", "max_grid_kwh"):
        if src.get(k) is not None:
            flat[k] = src[k]
    return flat


def score_note(got: dict, want: dict) -> tuple[int, int, list[str]]:
    """Return (points, max_points, problems) for one note."""
    got, want = flatten(got), flatten(want)
    problems: list[str] = []
    pts = 0

    want_no_op = want["directive_type"] == "no_op"
    got_no_op = got["directive_type"] == "no_op"

    # 1. relevance / no_op  (mirrors `applies`)
    if got_no_op == want_no_op:
        pts += 1
    else:
        problems.append(f"relevance: got no_op={got_no_op}, want {want_no_op}")

    # 2. directive_type
    if got["directive_type"] == want["directive_type"]:
        pts += 1
    else:
        problems.append(f"type: got {got['directive_type']}, want {want['directive_type']}")
        return pts, 4, problems   # hours/values are meaningless after this

    if want_no_op:
        return pts + 2, 4, problems   # hours and values are trivially correct

    # 3. hours
    if got.get("hours") == want.get("hours"):
        pts += 1
    else:
        problems.append(f"hours: got {got.get('hours')}, want {want.get('hours')}")

    # 4. numeric value
    key = next((k for k in ("factor", "minimum_energy_kwh", "max_grid_kwh")
                if k in want), None)
    if key is None:
        pts += 1
    else:
        gv, wv = got.get(key), want[key]
        ok = (isinstance(gv, (int, float)) and isinstance(wv, (int, float))
              and abs(gv - wv) <= TOL)
        if ok:
            pts += 1
        else:
            problems.append(f"{key}: got {gv!r}, want {wv!r}")
    return pts, 4, problems


def _self_check() -> None:
    """Fail loudly if the scorer ever stops comparing hours and values."""
    good = {"directive_type": "solar_reduction", "applies": True,
            "structured_adjustment": {"hours": [13, 14], "factor": 0.2}}
    bad = {"directive_type": "solar_reduction", "applies": True,
           "structured_adjustment": {"hours": [13, 15], "factor": 0.9}}
    want = {"directive_type": "solar_reduction", "hours": [13, 14], "factor": 0.2}
    assert score_note(good, want)[0] == 4, "scorer is not checking hours/values"
    assert score_note(bad, want)[0] == 2, "scorer is not checking hours/values"


def run_case(notes: list[str], capacity: float, expected: list[dict],
             label: str) -> tuple[int, int, float, list[str]]:
    t0 = time.perf_counter()
    out = interpret_notes(notes, capacity)
    elapsed = time.perf_counter() - t0

    got = [e.to_api_model(capacity).model_dump() for e in out]
    pts = total = 0
    problems: list[str] = []
    for i, want in enumerate(expected):
        if i >= len(got):
            problems.append(f"note {i}: missing from the response")
            total += 4
            continue
        p, t, pr = score_note(got[i], want)
        pts += p
        total += t
        problems += [f"note {i}: {x}" for x in pr]
    return pts, total, elapsed, [f"{label}: {x}" for x in problems]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paraphrase", action="store_true",
                    help="run the reworded variants instead of the public pack")
    ap.add_argument("--runs", type=int, default=1,
                    help="repeat each case to check stability")
    args = ap.parse_args()

    _self_check()
    get_structured_llm.cache_clear()
    cases = json.loads(FIXTURE.read_text())["cases"]

    # Each job carries its own battery capacity. Passing one shared value
    # makes every percentage reserve wrong, and the model looks broken when
    # it is the harness that lied to it.
    jobs: list[tuple[str, list[str], list[dict], float]] = []
    if args.paraphrase:
        for i, (notes, expected) in enumerate(PARAPHRASES):
            jobs.append((f"PARA-{i + 1:02d}", notes, expected, PARA_CAPACITY))
    else:
        for c in cases:
            jobs.append((
                c["id"],
                c["input"]["operator_notes"],
                c["expected_output"]["directive_interpretation"],
                float(c["input"]["battery"]["capacity_kwh"]),
            ))

    print(f"model: gpt-4o-mini | cases: {len(jobs)} | runs: {args.runs}\n")
    pts = total = 0
    latencies: list[float] = []
    all_problems: list[str] = []

    for label, notes, expected, capacity in jobs:
        for run in range(args.runs):
            p, t, elapsed, problems = run_case(notes, capacity, expected, label)
            pts += p
            total += t
            latencies.append(elapsed)
            all_problems += problems
            tag = "OK " if not problems else "ERR"
            print(f"  {tag} {label}{'' if args.runs == 1 else f' run{run + 1}'} "
                  f"{p}/{t}  {elapsed:.2f}s")
            for pr in problems:
                print(f"        ! {pr}")

    pct = pts / total * 100 if total else 0.0
    print(f"\nscore: {pts}/{total} ({pct:.1f}%)")
    if latencies:
        lat = sorted(latencies)
        print(f"latency: mean {statistics.mean(lat):.2f}s  "
              f"p95 {lat[max(0, int(len(lat) * 0.95) - 1)]:.2f}s  "
              f"max {max(lat):.2f}s   (budget: p95 <= 5s, hard 30s)")
    return 0 if pct == 100 else 1


if __name__ == "__main__":
    sys.exit(main())
