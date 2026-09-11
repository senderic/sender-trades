# sender-trades — Agent Context

## Working Principles

When asked to do anything, check the **quality and spirit** of the request —
not just the literal instruction. Look for the underlying goal the user is
trying to achieve (e.g. "trades should actually make money", "emails should
reflect reality"), and verify the work delivers that, not just that it
compiles or matches the letter of the ask. Push back or flag gaps when the
literal request would undermine its own intent.

## What This Is

An intraday directional prediction engine with an optional 0DTE options execution module for SPY and QQQ. Ingests the Atlas Morning Briefing + market snapshots, runs an LLM research pass via opencode, produces per-asset directional predictions with estimated move %, confidence, and cited evidence, and optionally executes trades via Alpaca with automatic exit management.

## Schedule

Cron (America/Los_Angeles): `28 6 * * 1-5` — 6:28 AM Mon-Fri.
Runs ~58 min after upstream `~/atlas-morning-briefing/` (5:30 AM).
Pipeline runs at 9:28 AM ET, 2 min before market open (9:30 AM ET), to catch pre-market gap moves.

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
2. **Ingest market** — load snapshots or live API (Finnhub, Brave, RSS, Reddit, UW), then `Pipeline._enrich_premarket` attaches live pre-market price/volume/reliability per asset (see "Pre-Market Data Fix" below)
3. **Analyze** — LLM prediction (primary) + 3 deterministic strategies in parallel
4. **Forecast** — build per-asset directional forecast table
5. **Decide** — aggregate strategies, apply risk checks
6. **Execute** — submit order via Alpaca, manage exits, close by deadline
7. **Email** — styled HTML via Gmail SMTP

### Pre-Market Data Fix (2026-09-11)

The atlas Finnhub snapshot is captured ~06:02 PT, before Finnhub has any pre-market price for SPY/QQQ — outside market hours its `/quote` returns the PRIOR completed session's close as `current_price` and the session before *that* as `previous_close`. Every `Quote` this pipeline builds pre-market is therefore one session stale; `Quote.prior_session_close` (== `current_price`) is the correct anchor for today's gap, never `previous_close`. Verified on the 2026-09-11 live run: the snapshot's QQQ `current_price` (708.69) was exactly Thursday's close; the pipeline predicted DOWN on that stale "-1.06%" gap while QQQ actually opened Friday +0.95%, and the strike was ~1.6% OTM at the open as a result.

`src/ingestion/premarket.py` (`fetch_premarket_quote`) fixes this by pulling live 1-minute bars from Alpaca (IEX feed — this account has no SIP subscription) from `premarket.session_start_et` to `premarket.cutoff_et`, producing a `PremarketQuote` per asset (`market.premarket[asset]`) with the live last-trade price, VWAP, cumulative volume, gap vs `prior_session_close`, and a `reliable` flag (today's volume ≥ `premarket.min_volume_ratio` × the trailing `premarket.lookback_days`-day median at the same clock time — relative, not absolute, since IEX pre-market volume for SPY/QQQ is a small, noisy slice of the real tape). Two distinct consumers:

- **MECHANICS** (strike selection, the forecast target strike, the gap-fade gap) — `MarketSnapshot.mechanics_price`/`mechanics_gap_pct` always use the live pre-market price regardless of `reliable`, falling back to `prior_session_close` (logging a WARNING) only when no live quote exists at all.
- **SIGNAL** (what the LLM prompt says) — the prompt's PRE-MARKET block states the reliability flag explicitly and tells the model to weight a thin-volume move lightly; when pre-market data is unavailable it says so explicitly rather than silently presenting the stale prior-session quote as "today".

`scripts/replay_models.py --premarket` reruns the historical replay with this same reconstruction (no look-ahead — bars are bounded at the cutoff for the historical date), caching LLM responses under a separate `<model>__premarket/` directory so the original stale-input results stay intact for comparison.

## Graph Engineering (LLM Analysis)

The daily LLM research pass is a **graph of narrow-scope opencode subagents** (`src/llm/graph.py::GraphOrchestrator`) instead of one monolithic call. Each node is a named agent in `.opencode/agent/`, invoked via `OpencodeLLMClient.invoke_agent()` with per-node timeouts, and the deterministic strategies feed in alongside the LLM.

```
[Research SPY] ──→ [Predict SPY] ──┐
[Research QQQ] ──→ [Predict QQQ] ──┤
                                    ├──→ [Checker] ──→ [Pick Trade]
[Momentum / MeanRev / EventDriven] ─┘
```

### Nodes (`.opencode/agent/`)

| Agent | Role | Runs |
|-------|------|------|
| `research-spy` / `research-qqq` | Parse briefing + market into structured catalysts, gap, sentiment, news | In parallel |
| `predict-spy` / `predict-qqq` | Turn research output into direction, confidence, move %, evidence | In parallel, after research |
| `checker` | Validates ALL outputs (LLM + deterministic), flags contradictions, adjusts confidence | Serial, after all outputs |
| `pick-trade` | Chooses best trade or passes, given validated predictions, history, trade outcomes | Serial, last |

### Behavior

- **Enabled** via `graph.enabled` in `config.yaml` (currently `true`). When false, the legacy monolithic `LLMTradeStrategy` call is used.
- **Ordering** — `src/pipeline.py::_phase_analyze` runs the three deterministic strategies **first** (pure local computation), serializes their results, and injects them into `LLMTradeStrategy(deterministic_results=...)`. The graph therefore validates against real deterministic signals, and the checker runs **exactly once** per pipeline run, inside the graph.
- **Timeout budget** — per *attempt*: research 45s, predict 30s, checker 30s, pick-trade 30s. Separately, `graph.total_deadline_sec` (240s) is a wall-clock ceiling on the whole graph, checked before each phase and threaded into every `invoke_agent` call as an absolute deadline. This matters because the model fallback chain is 7 models deep, so an unbounded node's worst case (7 × 45s) would overrun the 15-minute `entry_window_minutes` on a pipeline that starts ~2 min before the open.
- **Call budget** — `graph.reserved_calls_for_fallback` (1) is held back from `llm.max_calls_per_run`, so the graph path (`invoke_agent`) cannot exhaust the budget that the monolithic fallback (`invoke`) needs.
- **Checker authority** — the checker returns `can_proceed`. When it is false and `checker_contradiction_action: veto`, `LLMTradeStrategy` sets **both** `recommendation=None` and `StrategyResult.confidence=0.0`. Both are required: `DecisionAggregator.aggregate` filters on `recommendation is not None` and ranks/thresholds on `StrategyResult.confidence`, and never reads `recommendation.confidence`. Under `penalize`, `checker_confidence_penalty` (0.15) is subtracted from both fields together. The checker's `adjusted_confidence` values are applied to the per-asset predictions.
- **Gap-fade thresholds** — single source of truth in `config.gap_fade` (`threshold_for(asset)`, `sentiment_magnitude_max`), consumed by the risk engine, the strategies, and the prompt builders. The agent `.md` files deliberately contain **no** threshold numbers; the applicable values are stated in the prompt text at call time. The gap itself (`_gap_pct` in `trade_signal.py`, mirrored in `graph.py`) is the LIVE pre-market price vs `Quote.prior_session_close` (see "Pre-Market Data Fix" above) — it returns `None` for prompt/narrative purposes when no live pre-market quote exists, so the model is told explicitly rather than shown a gap computed from stale fields; `MarketSnapshot.mechanics_gap_pct` is the separate, always-numeric variant deterministic strategies and the risk engine use.
- **History awareness** — pick-trade sees prediction history + trade outcomes (`format_history_for_prompt`, `format_outcomes_for_prompt`) to avoid repeating losing calls.
- **Failure fallback** — if both research nodes, the checker, or the pick-trade node fails, or the wall-clock deadline expires, or any unexpected exception is raised, the orchestrator returns `graph_failed: true` in its trace and `trade_signal.py` falls back to the monolithic call (`graph.fallback_to_monolithic: true`). A pick-trade node that *returns* `best_trade: null` is a legitimate pass, not a failure.
- **Malformed output is never fatal** — every value read out of parsed checker JSON is defensively coerced and containers are type-checked before iteration. A bad `adjusted_confidence` degrades to "no adjustment applied"; it must never raise, because nothing above `Pipeline.run` catches (see `src/main.py`) and a crash means no forecast and no email.
- **JSON contracts** — each agent returns a JSON object (extracted by `src/llm/trade_signal.py::_parse_pick`, a balanced-brace scanner re-exported as `graph._extract_json`, tolerant of fenced/mixed text). Node contracts, matching `.opencode/agent/*.md` exactly:
  - research → `{asset, catalysts[], risks[], sentiment{}, technical_context{}, watchlist_signals[], key_theme}`
  - predict → `{asset, direction, confidence, predicted_move_pct, rationale, sources[]}`
  - checker → `{validated_predictions[], contradictions[], flags[], overall_assessment, can_proceed}`
  - pick-trade → `{best_trade|null, rationale, pass_reason, alternatives_considered}`
- **Prediction identity** — predictions are keyed by the **node** that produced them, never by the agent's self-reported `asset` field; a mismatch is rejected and logged, so a confused `predict-spy` cannot overwrite the QQQ prediction.

### Extending the Graph

To add a node: create `.opencode/agent/<name>.md` (the system prompt for that node, instructing it to emit the JSON contract), add a `_run_<name>()` method + prompt builder in `src/llm/graph.py`, wire it into `run()` in the right phase, then add a test in `tests/test_graph.py`.

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
| `intraday_monitor.py` | Cron SL enforcer — polls option mark, closes trades at `sl_level`, writes `exit_reason: stop_loss`; also wires in the profit-exit advisor below |
| `exit_advisor.py` | Profit-exit advisor — LLM-driven HOLD/EXIT decisions at event cadence, trailing-stop fallback, persisted per-trade state |

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

- **Stop-loss**: enforced by the **intraday monitor cron** (`intraday_monitor.sh`, `*/3 6-12 * * 1-5`), NOT in-process — always the FIRST check on every monitor pass, ahead of everything below, and never overridable. The pipeline's `_monitor_exits` / `ExitManager.evaluate` are dead code within the pipeline process. The monitor (`src/execution/intraday_monitor.py`) polls the live option mark, and force-closes any open trade whose mark ≤ `sl_level`, idempotently writing `exit_reason: stop_loss` to the audit JSON. Without it, every loser bled ~-86% to the once-daily safety-close sweep instead of -50%.
- **Take-profit**: when `execution.exit_advisor.enabled` is false (today's default off, but **on** in `config.yaml` per owner request — see below), a fixed limit sell at `entry_price * (1 + take_profit_pct/100)` is the only exit placed at Alpaca, and the pipeline exits (`exit_via_cron: true`). When the advisor is enabled, this fixed TP is replaced by a far `exit_advisor.safety_cap_pct` limit order (or none at all if null) — see next.
- **Profit-exit advisor** (`src/execution/exit_advisor.py`, new): a 2026-09-11 review of 72 asset-days found this strategy's OTM 0DTE options pay off on only a handful of big days (avg ~+71%/trade in premium units holding to expiry, top 3 days ~45% of total gains), and the fixed +100% TP was cutting off exactly those days (fell to ~+7-9% under fixed TP/SL). When `execution.exit_advisor.enabled` is true, `ExecutionEngine` skips the fixed TP and the intraday monitor instead consults the heavy model (`execution.exit_advisor.model`, falling back to `llm.primary_model` — Muse Spark 1.3, no further fallback chain) via `.opencode/agent/exit-advisor.md` at event-driven cadence, capped at `max_calls_per_trade` (12) per trade: a profit milestone (`profit_step_pct`, every further step), a give-back from peak (`giveback_from_peak_pct`, re-firing at each further step down the same peak — not just once — so a winner sliding all the way back to the stop keeps getting consulted), a one-shot consult if P&L round-trips from a real peak back below zero, a ~15-min periodic check-in while in profit, or the final ~30 min before the deadline — never every 3-min poll. State (peak P&L, consult/cadence bookkeeping, any advisor-tightened trailing stop) persists in the trade's own audit JSON under `advisor_state` since each cron run is a fresh process. The stop-loss above and the safety-close sweep below are unaffected either way. A close (from any rail) polls for an actual fill before marking the trade resolved — a 0DTE spread can be wider than the aggressive limit cushion, so an unfilled sell cancels itself, records the attempt (`close_attempts`), and leaves the trade `pending` for the next run to retry more aggressively, instead of orphaning an open Alpaca position behind a closed-looking audit. `scripts/simulate_exits.py` replays this against historical fills (Black-Scholes-reconstructed marks) to compare policies offline.
- **Trailing stop** (`execution.exit_strategy.trailing`): enforcement depends on `exit_advisor.enabled` — (1) disabled: never enforced by the monitor at all (dead config; `ExitManager.evaluate` doesn't run in-process, see above). (2) enabled, advisor healthy: fallback-only — enforced deterministically when a consult fails/times out/is unparseable, or the per-trade call budget is spent (`exit_reason: trailing_stop`). (3) enabled, and the advisor has itself tightened the trail via its `trail_stop_pct` output: that tightened trail is enforced on **every poll**, since it's the advisor's own deliberate standing order rather than a fallback — it can only ever tighten (never loosen) over a trade's life. Time deadline backstop is `safety_close.sh` (12:20 PM PT).

### Sizing (conservative 2-stage)

`DecisionAggregator._apply_sizing` scales a selected trade 1 → 2 contracts only when confidence ≥ `risk.sizing_tier2_min_confidence` (0.60) AND the pick is LLM-backed or corroborated by a second strategy. Hardcoded `contracts=1` remains in the strategy files as the base.

### De-risking (2026-09-07 review)

- **Momentum confidence capped** (`strategies.momentum.max_confidence: 0.50`) — momentum was 0/4 for -$182 with formula-inflated 0.65-0.91 confidence outranking the LLM.
- **Min-move gate** (`risk.min_predicted_move_pct: 0.30`) — `DecisionAggregator._min_move_gate` blocks LLM picks whose predicted move is too small to survive theta (skipped when no LLM prediction exists).
- **QQQ-first tie-break** (`risk.preferred_asset: QQQ`) — aggregation sort prefers QQQ at equal confidence (QQQ +$86 vs SPY -$26, 70% vs 67% accuracy).
- **Honest open→close metric** — `PredictionOutcome.open_close_correct` surfaces end-of-day direction correctness in the email recap alongside the flattering target-strike "HIT".

### De-risking (2026-09-10 review)

- **Premium-aware entry gate** (`risk.predicted_move_shrink: 0.4`, `risk.breakeven_margin_pct: 0.03`) — `DecisionAggregator.premium_gate` blocks a trade unless the LLM's predicted move, *shrunk* by `predicted_move_shrink` (predicted moves ran ~3x hotter than realized moves across 2026-07-29..09-09, e.g. -0.7/-0.8% predicted vs ~-0.2% actual on 09-09), clears the option's true expiry breakeven plus a small margin. `min_predicted_move_pct` compared the model's raw (inflated) number against a flat floor and never looked at what the option cost — a correct-direction call on a 0.2% move still expires worthless when the premium paid exceeds the payoff. Breakeven is OTM-distance-aware: the underlying must first cover the gap from spot to strike, then the premium — PUT `((underlying − strike) + ask) / underlying × 100`, CALL `((strike − underlying) + ask) / underlying × 100` (an ITM strike gives a negative distance, correctly *reducing* the requirement). A 2026-09-10 follow-up review found the first version used `ask / strike × 100` with the strike standing in for a live underlying price it didn't have — dropping the distance term entirely and understating breakeven by roughly the ~0.6% OTM offset every strike here is chosen at (`compute_otm_strike`); `breakeven_margin_pct` dropped from 0.15 to 0.03 accordingly, since it no longer needs to stand in for that distance, only for spread/slippage (not fitted from data — the audits don't retain the quoted ask, only the fill price). Runs in `ExecutionEngine.execute` (not at decision time) because neither the option ask nor a fresh underlying quote exist until after market open; falls back to the old ask/strike approximation (logged via `premium_gate_fallback_formula`) when a live underlying quote can't be fetched. `TradeRecommendation.predicted_move_pct` carries the LLM's per-asset move from `DecisionAggregator.aggregate` through to execution for this purpose. **Backtest (n=22 filled trades, 2026-07-29..09-09, strike/fill as ask, open price as spot proxy)**: keeps 3 trades (net +$145.00: two losers plus the single biggest winner, +$282 on 08-18), blocks 19 (net −$142.50, including 8 winning trades totaling +$452 — e.g. +$91 on 09-03, +$74 on 08-13, +$73 on 08-07). At this sample size the gate mainly reduces trade frequency (22 → 3) rather than cleanly separating winners from losers: with ~30-delta (~0.6% OTM) strikes and typical shrunk predicted moves of 0.1-0.3%, true (distance + premium)/spot breakeven (~0.3-1.2%) is almost never cleared by construction, independent of shrink/margin tuning. Treat this as a frequency dial, not a proven edge filter, until more trades accumulate.

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

## Live-Test Gotchas (2026-07-29)

Discovered during paper-trading verification. Do NOT repeat.

| Gotcha | Root Cause | Fix Location |
|--------|-----------|-------------|
| OCC symbol 422 "not found" | `occ_option_symbol()` used 8-digit date (`20260729`); Alpaca requires 6-digit (`260729`) | `src/mcp/schemas.py` — strips century if 8-char date part |
| Options market orders rejected pre-market | Alpaca rejects market orders before 9:30 AM ET: "options market orders are only allowed during market hours" | `src/execution/client.py` — always uses config's `order_type: limit`, ignores LLM's "market" |
| Pre-market gap inverts prediction | Briefing at 5:30 AM PT misses overnight earnings/catalysts. Model needs live pre-market quotes. | Cron moved to 9:28 AM ET (2 min before open). Model reruns with real pre-market data. |
| SDK returns `datetime`, not `str` | `alpaca-py` Order model has `datetime` fields; our `OrderResult` uses `str` — Pydantic crash | `src/execution/client.py:_order_to_result()` — `datetime.isoformat()` |
| Two simultaneous sell orders rejected | Alpaca prohibits selling same contract twice; TP fill "held" the position | Engine places only TP at Alpaca; SL managed in-app |
| Monitor loop hangs forever | No time-based exit in `_monitor_exits` when quotes unavailable | Time-deadline check at top of loop → force-close |
| Finnhub 502 → no market data → no trade | Atlas snapshots can have Finnhub errors for all symbols | Yahoo Finance fallback in `src/ingestion/snapshot_loader.py` |
| Paper chain truncates contracts | Default chain endpoint returns only calls (100 limit); puts exist separately | Always pass `type: put` filter when looking for puts |
| 0DTE entry limit never fills (expired) | Options don't trade pre-market, so no option ask at 9:28 AM ET submission; old delta fallback priced a $1.00 limit for an ATM option → never filled | `src/execution/engine.py:_await_option_quote()` waits for 9:30 AM ET open then retries the live option ask; `build_entry_order()` prices buy limit as a generous ceiling (intrinsic + 0.5%·spot, +`limit_offset_pct`) from `get_underlying_quote()` fallback so it fills at the open |
| Quote endpoints 404 / return None | `/v2/options/snapshots` and `/v2/stocks/snapshots` are **data-API** endpoints on `data.alpaca.markets`, not the trading host; options snapshots live under `/v1beta1/`; field is `latestQuote`/`bp`/`ap` (capital Q), and stocks response is `{SYMBOL: {...}}` with no `snapshots` wrapper | `src/execution/client.py` — separate `data_http` client (`https://data.alpaca.markets`); option quote hits `/v1beta1/options/snapshots?symbols=`, underlying hits `/v2/stocks/snapshots?symbols=` and reads `{symbol}.latestQuote` |

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
