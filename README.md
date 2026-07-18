# sender-trades

0DTE intraday options trading recommendation system for SPY and QQQ.

Ingests market data, runs multiple strategies, aggregates decisions, applies risk guardrails, and executes trades through an MCP broker.

## Features

- **3 strategies**: momentum (gap detection), mean-reversion (RSI), event-driven (catalyst-aware)
- **Decision aggregator**: picks highest-confidence recommendation or merges near-ties
- **Risk engine**: 8 guardrails including max loss, spread check, time windows, consensus, OCC chain validation
- **Multi-source ingestion**: Finnhub (quotes), Brave Search (news), Reddit (WSB/options sentiment), Unusual Whales (options flow), RSS feeds
- **Atlas briefing parser**: structured morning briefings with tickers, news, and macro sentiment
- **MCP broker abstraction**: subprocess stdio transport for Alpaca and Robinhood
- **Structured logging**: JSON logs with per-run correlation IDs
- **Dry-run mode**: safe by default — simulate without executing trades
- **Docker**: multi-stage build, `docker compose up` for app and Sphinx docs

## Architecture

```
Atlas Briefing ──> Parser ──┐
Finnhub ────────────────────┤
Brave ──────────────────────┤
Reddit ─────────────────────┤
Unusual Whales ─────────────┤
RSS Feeds ──────────────────┤
                             v
                      MarketSnapshot ──> Strategy A (momentum)
                                       ──> Strategy B (mean-reversion)
                                       ──> Strategy C (event-driven)
                                              │
                                              v
                                       DecisionAggregator
                                              │
                                              v
                                         RiskEngine
                                              │
                                              v
                                    MCPBrokerClient ──> Alpaca / Robinhood
```

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)
- Docker (optional, for containerized runs + docs)

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url> && cd sender-trades
uv sync --all-extras

# 2. Configure secrets
cp .env.example .env
# Edit .env with your API keys (see Configuration section)

# 3. Run a dry run (no trades executed)
python -m src.main --dry-run
```

## Configuration

### config.yaml (tracked, no secrets)

Application structure is in `config.yaml`. All secrets use `${VAR}` substitution resolved from `.env` at runtime.

```yaml
general:
  env_mode: PAPER_ALPACA      # or LIVE_ROBINHOOD
  target_assets: [SPY, QQQ]
  execute: false               # dry-run by default

mcp:
  alpaca:                      # uvx alpaca-mcp-server
    command: uvx
    args: ["alpaca-mcp-server"]
  robinhood:
    command: uvx
    args: ["robinhood-mcp-server"]
    auth:
      username: ${ROBINHOOD_USERNAME}
      password: ${ROBINHOOD_PASSWORD}
```

Override settings at runtime:

```bash
# Override config file
python -m src.main --config path/to/config.yaml

# Force execute mode (overrides config)
python -m src.main --execute

# Force dry-run (overrides config)
python -m src.main --dry-run
```

### .env (untracked, secrets)

Copy `.env.example` to `.env` and fill in:

```env
FINNHUB_API_KEY=             # Real-time quotes (SPY/QQQ)
BRAVE_API_KEY=               # News search
APCA_API_KEY_ID=             # Alpaca paper trading
APCA_API_SECRET_KEY=
APCA_API_BASE_URL=https://paper-api.alpaca.markets
UNUSUAL_WHALES_API_KEY=      # Options flow (paid subscription)
ROBINHOOD_USERNAME=          # Robinhood (if using LIVE_ROBINHOOD)
ROBINHOOD_PASSWORD=
ROBINHOOD_MFA_CODE=
```

## Running

```bash
# Dry run (default — simulate only)
python -m src.main --dry-run

# Live run (requires API keys + execute: true in config)
python -m src.main --execute

# Dry run with custom config
python -m src.main --config my-config.yaml
```

The pipeline prints a summary on completion:

```
Pipeline Complete — a1b2c3d4e5f6
Duration: 0.2s
Errors:   0
Trade:    NONE — no recommendation passed all gates
```

## Docker

```bash
# Build and run the app
docker compose build app
docker compose run app

# Build and run Sphinx docs
docker compose up docs
# Open http://localhost:8080
```

## Development

### Commands

| Command | Description |
|---|---|
| `make install` | Install dependencies |
| `make test` | Run pytest (63 tests) |
| `make lint` | Run ruff check |
| `make format` | Run ruff format |
| `make fix` | Auto-fix + format |
| `make check` | Lint + format-check + test |
| `make clean` | Remove venv and caches |

### Pre-commit

```bash
uv run pre-commit install
# Ruff lint + format run automatically on every commit
```

### CI

GitHub Actions runs on every push/PR:

- **lint**: `ruff check` + `ruff format --check`
- **test**: `pytest` (63 tests)

### Sphinx Docs

```bash
# Via Docker (recommended)
docker compose up docs    # http://localhost:8080

# Or locally
uv run sphinx-autobuild docs/source docs/build --host 0.0.0.0 --port 8080
```

## Project Structure

```
src/
├── config.py                  # Settings (pydantic-settings)
├── main.py                    # CLI entry point
├── pipeline.py                # Pipeline orchestrator
├── logging_setup.py           # structlog JSON logging
├── ingestion/
│   ├── fetcher.py             # Finnhub, Brave, Reddit, Unusual Whales, RSS
│   └── parser.py              # Atlas briefing parser
├── engine/
│   ├── base.py                # Base strategy
│   ├── strategy_a.py          # Momentum
│   ├── strategy_b.py          # Mean-reversion
│   ├── strategy_c.py          # Event-driven
│   ├── decision.py            # Decision aggregator
│   └── risk.py                # Risk engine
├── mcp/
│   ├── client.py              # MCP broker client (subprocess stdio)
│   └── schemas.py             # AlpacaOrderPayload, etc.
└── models/
    ├── market.py              # Quote, NewsHeadline, MarketSnapshot
    ├── briefing.py            # BriefingData
    └── recommendation.py      # StrategyResult, DecisionOutput
tests/
├── test_pipeline.py
├── test_risk.py
├── test_strategies.py
└── test_models.py
config.yaml                    # Application config (tracked)
.env.example                   # Secret template (copy to .env)
Dockerfile                     # Multi-stage build
compose.yml                    # app + docs services
Makefile                       # dev commands
pyproject.toml                 # Project metadata + tool config
.pre-commit-config.yaml        # ruff hooks
.github/
└── workflows/ci.yml           # GitHub Actions
```
