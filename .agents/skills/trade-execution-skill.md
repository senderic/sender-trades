# Trade Execution Skill

## Purpose
Run, debug, and manually intervene in the 0DTE trade execution engine.
Understand the trade lifecycle, exit strategies, and audit trail so you can
diagnose failed trades or adjust execution parameters.

## Architecture

```
src/execution/
├── client.py         — AlpacaBrokerClient (alpaca-py + Tenacity retries)
├── engine.py         — ExecutionEngine (full lifecycle orchestrator)
├── exit_manager.py   — ExitManager (TP, SL, trailing stop, time-based exit)
├── lifecycle.py      — TradeLifecycle (state machine)
├── models.py         — Pydantic models (OrderResult, ExecutionConfig, TradeState)
├── retry.py          — Tenacity retry config (3 attempts, expo backoff)
└── context.py        — TradeContext (audit JSON to logs/<date>/trade-<id>.json)
```

## Trade Lifecycle

```
CREATED → VALIDATING → SUBMITTED → ACKNOWLEDGED → FILLED → EXITS_PLACED → CLOSED
                                                             → TP_FILLED
                                                             → SL_FILLED
                                                             → FORCE_CLOSED
Terminal states: CLOSED, REJECTED, EXPIRED, FAILED
```

## Running the Engine

```bash
# Paper trading (dry-run by default — no execution)
uv run python -m src.main --dry-run --email

# Paper trading WITH execution (set execute:true in config or use --execute)
uv run python -m src.main --dry-run --execute

# Live trading
uv run python -m src.main --execute
```

## Checking Trade Status

1. **Audit trail**: `logs/YYYY-MM-DD/trade-<trade_id>.json`
   - Full lifecycle events, entry/strike/exit data, PnL, monitoring snapshots
   - Links back to pipeline `correlation_id`
2. **Pipeline logs**: `logs/YYYY-MM-DD/run-<correlation_id>.json`
   - Decision rationale, strategy results, risk checks passed

## Diagnosing Failed Trades

### Entry Never Filled
- Check `logs/<date>/trade-<id>.json` → look for `expired` exit_reason
- Verify OCC symbol exists in chain: `cat logs/<date>/run-*.json | jq 'select(.event_type=="entry_submitted")'`
- Check bid/ask spread: wide spread may block limit orders
- Reduce `max_bid_ask_spread_pct` or switch to market orders

### Order Rejected
- Check `rejected` exit_reason in audit file
- Verify Alpaca keys in `.env` (APCA_API_KEY_ID, APCA_API_SECRET_KEY)
- Paper account may need options trading enabled (should be auto-enabled on paper)
- Check that OCC symbol format is correct

### Exit Orders Didn't Fire
- TP/SL may be too far from price: adjust `take_profit_pct` and `stop_loss_pct`
- Force close at deadline: check `force_close` exit_reason
- Trailing stop may have missed: check `trailing_active` snapshots in audit

### Tenacity Exhausted
- 3 consecutive API failures → trade marked FAILED
- Check network, Alpaca API status, rate limits
- Increase `max_attempts` or `max_wait_sec` in `execution.tenacity`

## Manual Intervention

### Cancel an open order
```bash
# Via alpaca-py
uv run python -c "
from src.execution.client import AlpacaBrokerClient
import asyncio
c = AlpacaBrokerClient('key', 'secret', paper=True)
asyncio.run(c.cancel_order('ORDER_ID'))
"
```

### Check a position
```bash
# Via alpaca-mcp-server (uvx)
# Or use Alpaca dashboard at https://app.alpaca.markets/
```

### Force-close all positions
```bash
uv run python -c "
from alpaca.trading.client import TradingClient
c = TradingClient('key', 'secret', paper=True)
c.close_all_positions()
"
```

## Config Quick Reference

```yaml
execution:
  entry:
    order_type: limit            # market or limit
    limit_offset_pct: 5          # bid + offset for limit entries
    entry_window_minutes: 5      # cancel if unfilled after N minutes
  exit_strategy:
    take_profit_pct: 100         # close at +100% of entry premium
    stop_loss_pct: -50           # close at -50% of entry premium
    trailing:
      enabled: true
      activate_after_pct: 30     # start trailing after +30% PnL
      trail_pct: 15              # trail @ 15% below peak
    time_deadline_est: "15:25"   # hard close deadline
  tenacity:
    max_attempts: 3
    min_wait_sec: 1
    max_wait_sec: 30
```

## Common Mistakes (Do NOT repeat)

These were discovered during 2026-07-29 paper-trading verification.

1. **OCC symbol wrong date format**: `occ_option_symbol()` produces `SPY20260729C00750000` (8-digit year) but Alpaca requires `SPY260729C00750000` (6-digit). Already fixed — function strips century prefix.

2. **Datetime serialization**: `alpaca-py` returns `datetime` objects for `created_at`/`updated_at`. Pydantic `OrderResult` expects `str`. Fixed — `_order_to_result()` converts with `.isoformat()`.

3. **Cannot place two sell orders**: Alpaca rejects a second sell order for the same contract. Engine now places only the TP limit order at Alpaca. Stop-loss and trailing stops are managed in-app by the monitoring loop.

4. **Monitor loop must have time guard**: `_monitor_exits` runs `while not is_terminal`. Without a time-deadline escape, it hangs forever if quotes are unavailable (paper accounts lack market data subscriptions). Fixed — deadline check at top of each iteration.

5. **Paper chain missing puts**: Default `GET /v2/options/contracts` returns only calls. Always pass `type=put` filter to find put contracts. Individual contracts DO exist even when the unfiltered chain doesn't show them.

6. **Finnhub snapshots unreliable**: Upstream snapshots can have 502 errors for every symbol. Pipeline now falls back to Yahoo Finance via `snapshot_loader._fetch_yahoo_quote()` for any symbol with errors.

## Guardrails
- Never set `execute: true` unless explicitly asked by a human
- Never trade live without first verifying paper fills
- Never modify `max_loss_per_trade_usd` above $500
- All 0DTE positions must close by 15:30 ET
- Bid/ask spread > 20% of mid = do not trade
