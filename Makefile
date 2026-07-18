.PHONY: install test lint format check clean

install:
	uv sync --all-extras

test:
	uv run pytest tests/ -v --tb=short

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

check: lint format-check test

format-check:
	uv run ruff format --check src/ tests/

fix:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

clean:
	rm -rf .venv/ .pytest_cache/ __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
