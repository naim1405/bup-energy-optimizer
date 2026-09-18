# BUP Energy Optimizer

A modular **FastAPI** backend for the BUP CSE Fest 2026 GridWise challenge:
it receives a 24-hour campus energy scenario plus 1-3 natural-language
operator notes, and returns an interpretation of those notes along with a
24-hour operating schedule.

> **Current status — optimizer wired in, interpretation stubbed.**
> `POST /optimize-energy` runs the **real optimizer**: a linear program over
> grid, solar, charge, discharge and battery energy that minimizes total grid
> cost subject to every GridWise rule. Operator-note interpretation is still
> **stubbed to `no_op`** for every note, so the plan is optimal for the base
> scenario but does not yet respond to directives. Wiring in the LLM means
> replacing one function — see [Swapping in the LLM](#swapping-in-the-llm).

## Project structure

```
bup-energy-optimizer/
├── app/
│   ├── main.py                  # app factory, router mounts, 400 error handler
│   ├── core/
│   │   └── config.py            # Settings (env vars / .env via pydantic-settings)
│   ├── engine/                  # the math: pure, framework-free
│   │   ├── models.py            # dataclasses (Scenario, Directive, HourPlan...)
│   │   ├── optimizer.py         # directive overlay + LP + post-processing
│   │   └── validate.py          # independent hour-by-hour rule replay
│   ├── schemas/
│   │   ├── common.py            # DirectiveType, BatteryAction enums
│   │   ├── energy.py            # API request + response models
│   │   └── llm.py               # LLM structured-output model (LangChain target)
│   ├── services/
│   │   └── energy_optimizer.py  # pydantic <-> engine adapter + interpreter stub
│   └── api/
│       └── routes/
│           ├── health.py        # GET  /health
│           └── optimize.py      # POST /optimize-energy
├── tests/
│   └── fixtures/
│       └── public_sample_cases.json   # organizer-provided public cases
├── pyproject.toml               # deps (uv), incl. langchain + redis
├── uv.lock
├── Dockerfile
├── docker-compose.yml           # api + redis
└── .env.example
```

## Local development

```bash
uv sync                # install pinned dependencies
uv run uvicorn app.main:app --reload
# → http://localhost:8000/health
# → http://localhost:8000/docs
```

Run the tests:

```bash
uv run pytest
```

## Endpoints

| Method | Path                | Description                                              |
| ------ | ------------------- | -------------------------------------------------------- |
| GET    | `/`                 | Index with pointers to docs and the endpoints            |
| GET    | `/health`           | Readiness probe — exactly `{"status":"ok"}`              |
| POST   | `/optimize-energy`  | Interpretation + optimized 24-hour plan                  |
| GET    | `/docs`             | Swagger UI                                               |

### `GET /health`

Returns exactly `{"status": "ok"}`. Service name and version are deliberately
not in this body so a strict equality check passes; they are still available at
`/` and in the OpenAPI document.

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### `POST /optimize-energy`

```bash
curl -X POST http://localhost:8000/optimize-energy \
     -H 'Content-Type: application/json' \
     -d '{
           "scenario_id": "GRID-101",
           "operator_notes": ["Solar output will drop to about 20% from 1 PM to 3 PM."],
           "hours": [ {"hour":0,"demand_kwh":180,"solar_kwh":0,"tariff_bdt_per_kwh":7}, ... 23 more ... ],
           "battery": {"capacity_kwh":500,"initial_energy_kwh":200,
                       "minimum_energy_kwh":50,"max_charge_kwh_per_hour":100,
                       "max_discharge_kwh_per_hour":100}
         }'
```

Response shape: `scenario_id`, `directive_interpretation[]`, `hourly_plan[24]`,
`total_grid_kwh`, `total_cost_bdt`, `peak_grid_kwh`, `plan_summary`.
See `/docs` for the full schema and a worked example.

**Note on current costs:** because interpretation is stubbed to `no_op`, the
returned cost can be *lower* than the published reference for the same case —
the reference plan obeys directives that restrict the schedule, and we are not
applying them yet. That is expected, not a bug. Every returned plan is still
fully valid under the energy rules.

**Status codes**

| Code | Meaning                                                        |
| ---- | -------------------------------------------------------------- |
| 200  | Successful response                                            |
| 400  | Malformed JSON or structurally invalid request                 |

Request validation errors are mapped to **400** rather than FastAPI's default
422, matching the problem statement's status-code table. The error body
contains only `loc` / `msg` / `type` — no stack traces or internals.

## Schemas

`app/schemas/energy.py` holds the API contract. Input models use
`extra="ignore"` so an unexpected metadata field cannot cause a 400; output
models use `extra="forbid"` so the service can never emit a field outside the
contract.

Two rules are enforced by validators rather than left to callers:

- **`applies` is derived from `directive_type`** — `no_op` is the only type
  permitted with `applies=false` and a `null` adjustment, so an inconsistent
  pair cannot be constructed.
- **`structured_adjustment` must match its directive type** — `solar_reduction`
  requires `factor` (0-1), `minimum_battery_reserve` requires
  `minimum_energy_kwh`, `max_grid_window` requires `max_grid_kwh`, and the
  window directives take `hours` only.

### LLM structured output (`app/schemas/llm.py`)

`NoteInterpretationResult` is the model to hand to LangChain:

```python
from app.schemas.llm import NoteInterpretationResult

llm = init_chat_model(...).with_structured_output(NoteInterpretationResult)
result = llm.invoke(messages)                    # -> NoteInterpretationResult
entries = [i.to_api_model(battery.capacity_kwh)
           for i in result.interpretations]      # -> list[DirectiveInterpretation]
```

Design notes:

- The per-note model is **flat** — nested optional objects are harder for
  models to fill reliably, and a flat schema lets the per-type required field
  be checked by a validator.
- The `Field(description=...)` strings end up in the JSON schema the model
  sees, so they act as prompt guidance. The two most commonly misread rules
  (`factor` is the fraction **remaining**, and hour windows are
  **end-exclusive**) are spelled out there.
- `reserve_percent_of_capacity` handles notes like *"keep at least 50% of the
  battery capacity"*. `to_api_model(capacity)` resolves it to absolute kWh,
  which is what the response contract requires.
- LLM output is untrusted. A `ValidationError` here is the signal to retry
  once and then fall back to `no_op` — never let one bad note fail a request.

**Not implemented yet:** the prompt, the LangChain call, the guardrail/fallback
chain, and the optimizer.

## Docker

### Option A — dev / single service (API only)

```bash
docker build -t bup-energy-optimizer .
docker run -p 8000:8000 bup-energy-optimizer
```

The app listens on `0.0.0.0:8000` inside the container and exposes port 8000.

### Option B — full stack (API + Redis) via docker compose

```bash
docker compose up --build -d
```

Starts `api` on `8000` and `redis` on `6379`. Redis is wired in for the future
queue but is not required for either endpoint.

> **BuildKit note:** the Dockerfile deliberately avoids BuildKit-only features
> (no `RUN --mount`), so it also builds with the classic builder on hosts
> without the `buildx` plugin.

## Nginx reverse proxy (port forwarding)

Deploy the container so it publishes port `8000`, then point nginx at it:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Reload nginx (`sudo nginx -t && sudo systemctl reload nginx`) and health-check
via `curl http://your-domain.com/health`.

## The optimizer

`app/engine/` is deliberately **pure**: no pydantic, no FastAPI, no I/O. It
takes dataclasses and returns dataclasses, so it can be unit-tested without a
network — which matters because the LLM call upstream of it may be slow or
unavailable. `app/services/energy_optimizer.py` adapts between the two worlds.

```
request (pydantic)
  -> interpret_notes()                 [STUB: all no_op]
  -> NoteInterpretation.to_api_model() -> DirectiveInterpretation
  -> engine Directive
  -> engine.optimize()                 -> ScheduleResult
  -> OptimizeEnergyResponse
```

**Model.** A plain linear program — no integer variables, even though
`battery_action` looks categorical. Charge and discharge are 1:1 (no
round-trip loss), so they are modelled as two non-negative continuous
variables and the action is derived afterwards.

Five variables per hour (120 total): `grid`, `solar`, `charge`, `discharge`,
`E_after`. Minimize `sum(grid[h] * tariff[h])` subject to:

| Rule | Constraint |
| ---- | ---------- |
| Energy balance (9.5) | `grid + solar + discharge - charge = demand` |
| Battery state (9.1) | `E[h] = E[h-1] + charge - discharge` |
| End-of-day neutrality (9.6) | `E[23] = initial_energy_kwh` |
| Solar usage (9.4) | `0 <= solar[h] <= effective_solar[h]` |
| Battery bounds (9.2) | `min_floor[h] <= E[h] <= capacity` |
| Rate limits (9.3) | `charge <= max_charge`, `discharge <= max_discharge` |
| Grid cap | `grid[h] <= grid_cap[h]` |

Solved with scipy's HiGHS binding (~5 ms per request).

**Directives become per-hour arrays** in one place, and only there:

```python
effective_solar[h] *= factor                 # solar_reduction
min_floor[h]      = max(min_floor[h], n)     # minimum_battery_reserve
charge_ok[h]      = False                    # no_charge_window
discharge_ok[h]   = False                    # no_discharge_window
grid_cap[h]       = min(grid_cap[h], n)      # max_grid_window
```

**Post-processing.** Net `charge - discharge` into a single action, round to
6dp, then *derive* `grid` from the balance equation and *replay* `E` forward.
Both are computed rather than trusted, so they cannot disagree with the plan
the judge recalculates from.

**Self-validation.** `app/engine/validate.py` re-implements every rule from
the problem statement, independently of the optimizer, and runs on our own
output before it is returned. Violations are logged, never sent to the client.

**Infeasibility.** A misread directive can make the model infeasible even
though the organizer's ground truth never is. On LP failure the engine
re-solves with slack variables on the soft directives (grid caps, raised
reserve floors), so it always returns a structurally valid 24-hour plan,
sets `feasible=False`, and flags it in `plan_summary`. Never a 500.

## Swapping in the LLM

Everything downstream of `interpret_notes()` is already exercised by the stub.
Replace that one function in `app/services/energy_optimizer.py`:

```python
from app.schemas.llm import NoteInterpretationResult

def interpret_notes(notes: list[str]) -> list[NoteInterpretation]:
    llm = init_chat_model(...).with_structured_output(NoteInterpretationResult)
    result = llm.invoke(build_messages(notes))     # your prompt goes here
    return result.interpretations
```

`app/schemas/llm.py` is the guardrail layer: it validates the model's output
and `to_api_model()` derives `applies` and the adjustment shape rather than
trusting them. Remember the output is untrusted — catch `ValidationError`,
retry once, then fall back to a `no_op` entry for any note that still fails.

## Tests

```bash
uv run pytest
```

137 tests across four suites:

| File | Covers |
| ---- | ------ |
| `test_engine.py` | All 10 public cases: the engine reproduces every published optimum exactly, passes an independent rule replay, and reports self-consistent totals. |
| `test_engine_edges.py` | Infeasibility fallback, the reconcile/repair path, each directive honoured in isolation, stacked directives, and 120 random unseen scenarios cross-checked against a second LP algorithm. |
| `test_integration.py` | HTTP end-to-end, including directives injected by monkeypatching `interpret_notes` to prove they travel the whole path. |
| `test_optimize.py`, `test_llm_schema.py`, `test_health.py` | Contract shape, invalid-input handling, schema coupling. |

`tests/fixtures/public_sample_cases.json` is the organizer-provided public
pack, vendored so the suite is self-contained.

## Roadmap

- **LLM interpretation** — replace `interpret_notes()`; add the prompt and the
  retry/fallback chain.
- **Job queue** — backend will be Redis (`redis` already a dependency).
