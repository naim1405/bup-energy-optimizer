# syntax=docker/dockerfile:1

# uv-managed Python 3.12 image (slim Debian base)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install pinned dependencies first (better layer caching; pyproject/uv.lock
# change far less often than the app source).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Application source
COPY app ./app

# Make the virtualenv's binaries available on PATH
ENV PATH="/app/.venv/bin:$PATH"

# Run as an unprivileged user
RUN useradd --create-home appuser
USER appuser

# The port nginx (or the host) will forward to
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
