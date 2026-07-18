FROM python:3.12-slim-bookworm AS base
COPY --from=ghcr.io/astral-sh/uv:bookworm /usr/local/bin/uv /usr/local/bin/uv

FROM base AS venv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

FROM base AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=venv /app/.venv /app/.venv
COPY src/ /app/src/
COPY pyproject.toml /app/
ENV PATH="/app/.venv/bin:$PATH"
WORKDIR /app
RUN useradd -m trader
USER trader
CMD ["python", "-m", "src.main"]
