# 0DTE Intraday Options Trader Skill

## Purpose
Analyze structured pipeline logs from the sender-trades system to evaluate
strategy performance, diagnose failures, and suggest config improvements for
the next trading day.

## Input Sources
- `logs/YYYY-MM-DD/run-{correlation_id}.json` — Per-event structured log
- `logs/YYYY-MM-DD/summary-{correlation_id}.json` — Run summary with decision
- `config.yaml` — Current strategy configuration
- `~/atlas-morning-briefing/Atlas-Briefing-{date}.md` — The briefing that was ingested

## Log Format
Each log entry is a JSON line with keys:
- `correlation_id` — Run identifier
- `phase` — One of: `ingest_briefing`, `ingest_market`, `analyze`, `decide`, `execute`
- `strategy` — Present in analyze phase: `momentum`, `mean_reversion`, `event_driven`
- `confidence` — Float 0-1
- `has_recommendation` — Whether strategy produced a trade idea
- `rationale` — Decision reasoning when present
- `error` — Error details when present

## Analysis Workflow

### 1. Read Summary First
Open the latest `summary-{correlation_id}.json` to get the overall outcome:
- Was a trade recommended? (`decision.recommendation` not null)
- Which strategy was selected? (`decision.selected_label`)
- What was the confidence score?
- Were there errors?

### 2. Trace the Decision Path
If a trade was recommended:
1. Read the `analyze` entries for the selected strategy to see its signals
2. Read the `decide` entry to see why it was chosen over alternatives
3. Read the `execute` entry to see what MCP command was sent

If NO trade was recommended:
1. Check each strategy's `analyze` entry for `skip_reason` in debug_trace
2. Check if `risk_check_failed` appears in the `decide` entry
3. Check `consensus_check_failed` — indicates insufficient data source agreement

### 3. Evaluate Yesterday's Idea vs Today's Reality
Compare the trade that was (or was not) made against what actually happened:
- If a CALL was recommended but the asset fell, what went wrong?
  - Did sentiment reverse? Check executed trade's `rationale` for entry price vs day's high
- If no trade was recommended but the asset moved 1%+, what did the models miss?
  - Check each strategy's `debug_trace` for the signal thresholds that weren't met

### 4. Suggest Config Adjustments
Based on win/loss analysis:

| Observation | Config Tweak |
|---|---|
| Strategy consistently undershoots | Lower `min_confidence` for that strategy |
| Strategy triggers on noise | Raise `gap_threshold_pct` or `min_confidence` |
| Spread too wide blocks execution | Raise `max_bid_ask_spread_pct` |
| Too many no-trade days | Lower `min_data_sources_for_direction` to 1 |
| Losses exceed comfort | Lower `max_loss_per_trade_usd` |

## Guardrails to Respect
- Never modify `risk.max_loss_per_trade_usd` above a responsible level
- Never set `execute: true` unless explicitly asked by a human
- Never fabricate price data — all signals must trace to real `MarketSnapshot` data
- Always verify that an OCC option symbol exists in the chain before suggesting it

## 0DTE-Specific Constraints
- All positions must be closed by 3:30 PM EST (configurable as `close_deadline_est`)
- No trades recommended after 2:00 PM EST
- Bid/ask spread >20% of mid = illiquid, do not trade
- Maximum loss per trade is hard-capped in config
- 0DTE theta accelerates rapidly after 10:30 AM EST — factor this into confidence
