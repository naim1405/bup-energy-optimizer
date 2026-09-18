# BUP Energy Optimizer

A modular **FastAPI** backend for the BUP CSE Fest 2026 GridWise challenge:
it receives a 24-hour campus energy scenario plus 1-3 natural-language
operator notes, and returns an interpretation of those notes along with a
24-hour operating schedule.

> **Current status — scaffold + mock.** `POST /optimize-energy` accepts the
> full request contract and returns a **schema-valid mock response**. It does
> **not** interpret operator notes and does **not** optimize: every note is
> reported as `no_op`, the battery is left idle, and the plan is not
> cost-optimal. The LLM layer and the optimizer are not wired in yet.

## Project structure

```
bup-energy-optimizer/
├── app/
│   ├── main.py                  # app factory, router mounts, 400 error handler
│   ├── core/
│   │   └── config.py            # Settings (env vars / .env via pydantic-settings)
│   ├── schemas/
│   │   ├── common.py            # DirectiveType, BatteryAction enums
│   │   ├── energy.py            # API request + response models
│   │   └── llm.py               # LLM structured-output model (LangChain target)
│   ├── services/
│   │   └── mock_optimizer.py    # MOCK response builder (replace with real pipeline)
│   └── api/
│       └── routes/
│           ├── health.py        # GET  /health
│           └── optimize.py      # POST /optimize-energy
├── tests/
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
| POST   | `/optimize-energy`  | Interpretation + 24-hour plan (**mock** at present)      |
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

## Roadmap

- **LLM interpretation** — `app/schemas/llm.py` is ready; add an `app/llm/`
  module with the prompt, the LangChain call, and the guardrail/fallback chain.
- **Optimizer** — replace `app/services/mock_optimizer.py` with the real
  24-hour scheduler (an LP over grid / solar / charge / discharge / energy).
- **Self-validation** — replay the finished plan against the energy rules
  before returning it.
- **Job queue** — backend will be Redis (`redis` already a dependency).
