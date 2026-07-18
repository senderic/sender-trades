FROM ghcr.io/astral-sh/uv:bookworm AS uv
WORKDIR /app
COPY pyproject.toml .
RUN uv sync --no-dev --frozen

FROM python:3.12-slim-bookworm AS build
COPY --from=uv /app/.venv /app/.venv
COPY src/ /app/src/
COPY pyproject.toml /app/
RUN useradd -m trader

FROM build AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/app/.venv/bin:$PATH"
WORKDIR /app
USER trader
CMD ["python", "-m", "src.main"]
