# sender-trades

0DTE intraday directional prediction engine for SPY and QQQ.

Ingests the Atlas Morning Briefing + market snapshots, runs an LLM research pass via opencode, and produces per-asset directional predictions with estimated move %, confidence, and cited evidence.

## Features

- **LLM-powered prediction**: opencode CLI emits per-asset predictions (direction, confidence, predicted move %, rationale) with root-provenance citations tracing back to original news publishers and market data
- **Clear forecast table**: Direction (UP/DOWN), Confidence, Predicted Move %, Key Drivers of the Prediction, and Market Vibe — no confusing UP/DOWN/SIDE columns
- **3 deterministic strategies**: momentum (gap detection), mean-reversion (RSI-estimate), event-driven (catalyst-aware) run alongside the LLM for cross-reference
- **Multi-source ingestion**: Finnhub (quotes), Brave Search (news), RSS feeds (Reuters, Dow Jones, Seeking Alpha, Investing.com)
- **Snapshot-aware**: reuses cached data from atlas-morning-briefing to avoid duplicate API calls
- **Realistic 0DTE strikes**: ~0.6% OTM for 30-delta — strikes that actually exist in the chain
- **Optional trade execution**: when signal is strong, produces a best-trade recommendation through MCP broker (Alpaca/Robinhood)
- **Email reports**: styled HTML forecast via Gmail SMTP
- **Structured logging**: JSON logs with per-run correlation IDs
- **Dry-run safe by default**

## Quick start

```bash
git clone <repo> && cd sender-trades
cp .env.example .env          # configure GMAIL_USER, GMAIL_APP_PASSWORD
uv sync --all-extras
uv run pytest                 # 127 tests
uv run python -m src.main --dry-run --email   # dry-run + email forecast
```

Requires `~/atlas-morning-briefing` with briefings and snapshots (see atlas-morning-briefing repo).

## Output example

```
────────────────────────────────────────────────────────────────────────────────────────────────────
Asset  Direction   Confidence  Pred. Move   Key Drivers of the Prediction
────────────────────────────────────────────────────────────────────────────────────────────────────
SPY    DOWN              82%       -1.2%   Broad-based selloff led by mega-cap tech (NVDA -2.2%,
                                           META -2.8%, MSFT -1.8%) and a -1.15% gap-down open
                                           signals capital rotation out of consumer-AI compute into
                                           defense/space infrastructure.
QQQ    DOWN              88%       -1.8%   QQQ is the epicenter of the rotation: gapped -2.02%,
                                           now -1.50%, with every major tech holding selling off as
                                           capital rotates from consumer-AI compute into
                                           defense/space infrastructure.
────────────────────────────────────────────────────────────────────────────────────────────────────

Market Vibe: Rotation from consumer-AI compute into defense/space infrastructure
```

## Pipeline

1. **Ingest briefing** — parse Atlas Morning Briefing markdown + status.json
2. **Ingest market** — load snapshots or live API (Finnhub, Brave, RSS)
3. **Analyze** — LLM prediction (primary) + 3 deterministic strategies in parallel
4. **Forecast** — build per-asset directional forecast table
5. **Optional execute** — MCP broker subprocess (dry-run by default)
