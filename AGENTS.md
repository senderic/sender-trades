# sender-trades — Agent Context

## What This Is

An intraday directional prediction engine for SPY and QQQ. Ingests the Atlas Morning Briefing + market snapshots, runs an LLM research pass via opencode, and produces per-asset directional predictions with estimated move %, confidence, and cited evidence.

## Schedule

Cron (America/Los_Angeles): `30 6 * * 1-5` — 6:30 AM Mon-Fri.
Runs ~42 min after upstream `~/atlas-morning-briefing/` (5:30 AM).

## Key Commands

```bash
# Full run (dry-run + email forecast)
./run_trades.sh

# Manual dry run
uv run python -m src.main --dry-run --email

# Live test (no MCP execution)
uv run python -m src.main --dry-run

# Tests
uv run pytest -v --tb=short
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

## Pipeline

1. **Ingest briefing** — parse markdown + status.json, detect degraded quality
2. **Ingest market** — load snapshots or live API (Finnhub, Brave, RSS, Reddit, UW)
3. **Analyze** — LLM prediction (primary) + 3 deterministic strategies in parallel
4. **Forecast** — build per-asset directional forecast table
5. **Optional execute** — MCP broker (disabled by `--dry-run`)
6. **Email forecast** — styled HTML via Gmail SMTP

## Degraded Briefing Handling

When atlas LLM layer fails, the briefing markdown contains `"Synthesis unavailable for today's briefing"`. The pipeline detects this (`src/ingestion/parser.py:DEGRADED_SUMMARY_PREFIX`), classifies quality as `DEGRADED`, and re-synthesizes the executive summary via opencode's free-tier models as a fallback.

## Known Non-Blocking Issues

| Issue | Location | Impact |
|-------|----------|--------|
| Reddit 403 blocked | `src/ingestion/fetcher.py` | Logged error, pipeline continues |
| Unusual Whales no API key | `src/ingestion/fetcher.py` | Logged warning, pipeline continues |
| MCP Alpaca keys not configured | `src/mcp/client.py` | Expected — cron passes `--dry-run` |
| RSS 301 redirects | `src/ingestion/fetcher.py` | Auto-followed by httpx |

## Key Architecture Notes

- **Prediction engine, not trade executor** — redesigned 2026-07-18. Outputs per-asset directional forecasts with confidence and move %. The `best_trade` field is optional and only fires for strong signals.
- **Dry-run safe by default** — `run_trades.sh` passes `--dry-run --email`. No live execution without explicit `--execute` flag.
- **Snapshot priority** — tries atlas snapshots first before live API calls (reduces API usage).
- **Config** — `config.yaml` uses `${VAR}` interpolation. Full settings in `src/config.py::Settings`.

## Learning Resources

- `LESSONS_LEARNED.md` — incident log and design rationale
- `src/ingestion/parser.py` — briefing markdown parsing
- `src/pipeline.py` — orchestration
