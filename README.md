# BUP Energy Optimizer

A minimal, modular **FastAPI** backend for the BUP energy optimizer.
Currently it exposes a single `/health` endpoint. Dependencies for the
planned LLM layer (`langchain`) and job queue (`redis`) are already declared
but intentionally have no code behind them yet.

## Project structure

```
bup-energy-optimizer/
├── app/
│   ├── main.py                 # FastAPI app factory + entry point
│   ├── core/
│   │   └── config.py           # Settings (env vars / .env via pydantic-settings)
│   └── api/
│       └── routes/
│           └── health.py       # GET /health
├── tests/
│   └── test_health.py
├── pyproject.toml              # deps (uv), incl. langchain + redis
├── uv.lock                     # locked dependency versions
├── Dockerfile
├── docker-compose.yml          # api + redis
└── .env.example
```

## Local development

```bash
uv sync                # install pinned dependencies
uv run uvicorn app.main:app --reload
# → http://localhost:8000/health
```

Run the tests:

```bash
uv run pytest
```

## Endpoints

| Method | Path      | Description                                  |
| ------ | --------- | -------------------------------------------- |
| GET    | `/`       | Index with pointers to docs and health       |
| GET    | `/health` | Liveness/readiness probe (`{"status":"ok"}`) |
| GET    | `/docs`   | Swagger UI                                   |

## Docker

### Option A — dev / single service (API only)

```bash
docker build -t bup-energy-optimizer .
docker run -p 8000:8000 bup-energy-optimizer
```

The app **listens on 0.0.0.0:8000** inside the container and exposes port
8000, so on the host it is reachable at `http://localhost:8000`.

### Option B — full stack (API + Redis) via docker compose

```bash
docker compose up --build -d
```

This starts `api` on `8000` and `redis` on `6379`. `redis` is wired in now
(ready for the future queue) but is not required for `/health` to work.

> **BuildKit note:** the Dockerfile deliberately avoids BuildKit-only features
> (no `RUN --mount`), so it also builds with the classic builder on hosts
> without the `buildx` plugin. If `docker compose` prints a "requires buildx
> plugin" warning, it is harmless — or enable BuildKit anyway with
> `sudo pacman -S docker-buildx` (Arch Linux).

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

- **LLM integration** — `langchain` is in `pyproject.toml`; add an `app/llm/`
  module when ready.
- **Job queue** — backend will be Redis (`redis` already a dependency); add a
  queue module (e.g. RQ / Celery / arq) when ready.
