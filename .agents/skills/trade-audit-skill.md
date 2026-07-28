# Trade Audit Skill

## Purpose
Read and analyze trade audit files (`logs/<date>/trade-<trade_id>.json`) to
determine what happened with a specific trade, calculate PnL, and diagnose
why the trade won or lost.

## Audit File Location
```
logs/YYYY-MM-DD/trade-<trade_id>.json
```

## Audit File Structure

```json
{
  "trade_id": "abc123",
  "correlation_id": "def456",
  "asset": "SPY",
  "direction": "CALL",
  "contracts": 1,
  "entry_strike": 600.0,
  "exit_reason": "take_profit",
  "entry_price": 0.50,
  "exit_price": 1.00,
  "final_pnl": 50.0,
  "final_pnl_pct": 100.0,
  "duration_seconds": 3600,
  "started_at": "...",
  "ended_at": "...",
  "events": [
    {"timestamp": "...", "state_from": null, "state_to": "created", "metadata": {}},
    {"timestamp": "...", "state_from": "created", "state_to": "validating", "metadata": {}},
    {"timestamp": "...", "state_from": "filling", "state_to": "filled", "metadata": {"avg_price": 0.50}}
  ],
  "entries": [
    {"event_type": "entry_submitted", "occ_symbol": "SPY250728C00600000", ...},
    {"event_type": "entry_filled", "fill_price": 0.50, ...},
    {"event_type": "monitoring_snapshot", "current_pnl_pct": 45.0, ...}
  ],
  "recommendation": {...}
}
```

## Analysis Workflow

### 1. Open the audit file
Find the latest trade:
```bash
ls -t logs/*/trade-*.json | head -1
```

### 2. Check the high-level result
```bash
cat logs/2026-07-28/trade-abc123.json | jq '{exit_reason, final_pnl, final_pnl_pct, duration_seconds}'
```

### 3. Trace the lifecycle
```bash
cat logs/2026-07-28/trade-abc123.json | jq '.events[] | {state_to, timestamp}'
```

### 4. Review monitoring snapshots
```bash
cat logs/2026-07-28/trade-abc123.json | jq '.entries[] | select(.event_type=="monitoring_snapshot") | {timestamp, current_pnl_pct, tp_level, sl_level}'
```

### 5. Cross-reference with pipeline decision
```bash
# Find the correlation_id from the trade
trade_cid=$(cat logs/*/trade-abc123.json | jq -r '.correlation_id')
# Find the pipeline run
cat logs/*/summary-$trade_cid.json | jq '.decision'
```

## Diagnosing Outcomes

### Win (take_profit)
- Check if TP was hit cleanly or barely
- Check duration — fast win (~5-30 min) means good entry timing
- Compare predicted move % vs actual

### Loss (stop_loss)
- Check if SL was hit on a whip-saw (then reversed)
- Check if trailing stop would have saved the trade
- Compare stop distance vs typical intraday volatility

### Force Close (time)
- Did the trade have unrealized gains that evaporated?
- Was the directional call correct but timing wrong?
- Consider adjusting time_deadline_est earlier or later

### Expired (never filled)
- Check the spread at entry time
- Was the strike too far OTM?
- Consider adjusting entry `limit_offset_pct` or switching to market

### Rejected
- Check `entries[].event_type == "entry_order_response"` for the rejection reason
- Likely an OCC symbol format issue or authorization problem

## Calculating Metrics

```bash
# Extract trade data
trade=$(cat logs/*/trade-abc123.json)

# PnL
echo $(echo $trade | jq '.final_pnl')

# PnL %
echo $(echo $trade | jq '.final_pnl_pct')

# Duration in minutes
echo $(echo $trade | jq '.duration_seconds / 60')

# Max drawdown (from monitoring snapshots)
echo $trade | jq '[.entries[] | select(.event_type=="monitoring_snapshot") | .current_pnl_pct] | min'
```

## Suggesting Config Adjustments

| Observation | Config Change |
|---|---|
| Consistently hits TP in < 1 hour | Consider raising `take_profit_pct` |
| Hits SL frequently before reversing | Enable/widen trailing stop |
| Never fills | Reduce `limit_offset_pct` or use `market` orders |
| Expires before filling | Increase `entry_window_minutes` |
| Force-closed while profitable | Later `time_deadline_est` |
| API errors | Increase `tenacity.max_attempts` |

## Guardrails
- Never fabricate results — all data must come from the actual audit file
- Always cross-reference with pipeline logs before concluding
- PnL calculations must use `final_pnl` from the audit file, not manual math
