# sender-trades — Agent Context

## What This Is

An intraday directional prediction engine with an optional 0DTE options execution module for SPY and QQQ. Ingests the Atlas Morning Briefing + market snapshots, runs an LLM research pass via opencode, produces per-asset directional predictions with estimated move %, confidence, and cited evidence, and optionally executes trades via Alpaca with automatic exit management.

## Schedule

Cron (America/Los_Angeles): `15 6 * * 1-5` — 6:15 AM Mon-Fri.
Runs ~42 min after upstream `~/atlas-morning-briefing/` (5:30 AM).
Pipeline runs at 9:15 AM ET, submits day limit orders to Alpaca queued for 9:30 AM ET open.

## Key Commands

```bash
# Full run (dry-run + email forecast)
./run_trades.sh

# Manual dry run
uv run python -m src.main --dry-run --email

# Paper execution (set execute:true in config)
uv run python -m src.main --dry-run --execute

# All tests (skip integration)
uv run pytest tests/ -q --tb=short -m "not integration"

# Execution module tests
uv run pytest tests/test_execution_*.py -v --tb=short

# Lint + format
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Dependencies

**Upstream:** Requires `~/atlas-morning-briefing/` with:
- Briefing markdown at `briefings/Atlas-Briefing-YYYY.MM.DD.md`
- `status.json` at root with `intelligence_enabled` flag
- Snapshots at `snapshots/YYYY-MM-DD/` (optional — falls back to live API)
- `.env` at root (API keys sourced by `run_trades.sh`)

**Binaries:**
- `uv` at `~/.local/bin/uv` (v0.10.12) — runs the pipeline
- `opencode` at `/home/linuxbrew/.linuxbrew/bin/opencode` (v1.18.3) — LLM resynthesis + trade signal
- `npx` (optional) — options-chain MCP for token-efficient chain lookups

**Runtime Python deps:** alpaca-py, httpx, pydantic, structlog, tenacity, etc.
**Dev deps:** pytest, pytest-asyncio, pytest-cov, pytest-httpx, ruff

## Pipeline

1. **Ingest briefing** — parse markdown + status.json, detect degraded quality
2. **Ingest market** — load snapshots or live API (Finnhub, Brave, RSS, Reddit, UW)
3. **Analyze** — LLM prediction (primary) + 3 deterministic strategies in parallel
4. **Forecast** — build per-asset directional forecast table
5. **Decide** — aggregate strategies, apply risk checks
6. **Execute** — submit order via Alpaca, manage exits, close by deadline
7. **Email** — styled HTML via Gmail SMTP

## Execution Module (`src/execution/`)

| File | Responsibility |
|------|---------------|
| `client.py` | AlpacaBrokerClient — direct API via `alpaca-py` + Tenacity retries |
| `engine.py` | ExecutionEngine — orchestrates full trade lifecycle |
| `exit_manager.py` | ExitManager — TP, SL, trailing stop, time-based hard close |
| `lifecycle.py` | TradeLifecycle — state machine: CREATED → FILLED → CLOSED |
| `models.py` | Pydantic models: ExecutionConfig, OrderResult, TradeState, etc. |
| `retry.py` | Tenacity config: 3 attempts, expo backoff 1s→30s, 5xx only |
| `context.py` | TradeContext — writes audit JSON to `logs/<date>/trade-<id>.json` |

### Trade Lifecycle

```
CREATED → VALIDATING → SUBMITTED → ACKNOWLEDGED → FILLED → EXITS_PLACED → CLOSED
    ↓         ↓            ↓              ↓          ↓            ↓
                                        REJECTED   PARTIALLY   TP_FILLED
                                         EXPIRED    FILLED     SL_FILLED
                                                              FORCE_CLOSED
Terminal: CLOSED, REJECTED, EXPIRED, FAILED
```

### Exit Strategies

- **Take-profit**: limit sell at `entry_price * (1 + take_profit_pct/100)`
- **Stop-loss**: stop sell at `entry_price * (1 + stop_loss_pct/100)`
- **Trailing stop**: activates after `activate_after_pct`, trails `trail_pct` below peak
- **Time deadline**: force-close market order at `time_deadline_est`

### Entry Flow

1. Pipeline produces `TradeRecommendation` → submitted at 9:15 AM ET
2. Day limit order sits queued at Alpaca (`accepted` status)
3. Routes to exchange at 9:30 AM ET open
4. Engine polls for fill (5 min window, then cancel if unfilled)

## Degraded Briefing Handling

When atlas LLM layer fails, the briefing markdown contains `"Synthesis unavailable for today's briefing"`. The pipeline detects this (`src/ingestion/parser.py:DEGRADED_SUMMARY_PREFIX`), classifies quality as `DEGRADED`, and re-synthesizes the executive summary via opencode's free-tier models as a fallback.

## Known Non-Blocking Issues

| Issue | Location | Impact |
|-------|----------|--------|
| Reddit 403 blocked | `src/ingestion/fetcher.py` | Logged error, pipeline continues |
| Unusual Whales no API key | `src/ingestion/fetcher.py` | Logged warning, pipeline continues |
| MCP Alpaca keys not configured | `src/mcp/client.py` | Expected — cron passes `--dry-run` |
| RSS 301 redirects | `src/ingestion/fetcher.py` | Auto-followed by httpx |
| Alpaca no native bracket orders for options | `src/execution/exit_manager.py` | Exit management built in-app — TP limit + SL stop, cancel unfilled |
| MCP subprocess too slow for execution | `src/mcp/client.py` | Direct `alpaca-py` used instead; MCP kept for chain/quote queries |
| `pytest_httpx` incompatible with `alpaca-py` | `tests/test_execution_client.py` | Mocks at SDK level with `MagicMock/AsyncMock` instead |

## Key Architecture Notes

- **Prediction engine + optional executor** — redesigned 2026-07-18, extended 2026-07-28 with execution module. Outputs per-asset directional forecasts. The `best_trade` field feeds the execution engine when `execute: true`.
- **Dry-run safe by default** — `run_trades.sh` passes `--dry-run --email`. No live execution without explicit `--execute` flag.
- **Snapshot priority** — tries atlas snapshots first before live API calls (reduces API usage).
- **Config** — `config.yaml` uses `${VAR}` interpolation. Full settings in `src/config.py::Settings`. Execution settings in `src/execution/models.py::ExecutionConfig`.
- **Alpaca direct for execution, MCP for discovery** — `alpaca-py` SDK used for all order actions (fast, retryable). MCP subprocess (alpaca-mcp-server) kept for option chain/quote lookups.
- **Full audit trail** — every trade lifecycle event written to structured JSON at `logs/<date>/trade-<id>.json`, linked back to pipeline `correlation_id`.

## MCP Servers (config.yaml)

| Server | Command | Purpose |
|--------|---------|---------|
| Alpaca | `uvx alpaca-mcp-server` | Chain lookups, quotes, account info |
| Robinhood | `uvx robinhood-mcp-server` | Alternative broker (not active) |
| Options Chain | `npx -y @blake365/options-chain` | Token-efficient chain filtering (~15 strikes vs ~400) |

## Learning Resources

- `LESSONS_LEARNED.md` — incident log and design rationale
- `.agents/skills/trade-execution-skill.md` — debug & run the execution engine
- `.agents/skills/trade-audit-skill.md` — post-trade forensics
- `src/pipeline.py` — orchestration
- `src/execution/engine.py` — trade lifecycle
- `src/execution/exit_manager.py` — exit logic
- `logs/<date>/trade-<id>.json` — full audit trail examples
