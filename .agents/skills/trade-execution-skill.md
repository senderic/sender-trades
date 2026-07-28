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

## Guardrails
- Never set `execute: true` unless explicitly asked by a human
- Never trade live without first verifying paper fills
- Never modify `max_loss_per_trade_usd` above $500
- All 0DTE positions must close by 15:30 ET
- Bid/ask spread > 20% of mid = do not trade
