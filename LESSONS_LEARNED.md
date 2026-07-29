# Lessons Learned

Intraday directional prediction engine & 0DTE execution. LLM-readable format — scan the Quick Index, then jump to the entry by date or tag.

## Quick Index

| Date | SPY Pred | SPY Result | QQQ Pred | QQQ Result | Trade Executed | Engine PnL | Key Tags |
|------|----------|------------|----------|------------|---------------|------------|----------|
| [2026-07-29](#2026-07-29) | UP 45% | HIT (+0.3%/target) | DOWN 70% | HIT (-1.5%→-2.0%) | No (OCC bug) | $0 | `bug:occ-symbol` `pattern:qqq-down-reliable` `system:exit-monitoring` `source:market-qqq-reliable` |
| [2026-07-18b](#2026-07-18b) | — | — | — | — | N/A | N/A | `arch:redesign` `model:prediction-engine` `fix:strike-calculation` |
| [2026-07-18](#2026-07-18) | — | — | — | — | N/A | N/A | `system:degraded-briefing` `fix:quality-detection` `upstream:atlas` |

## Persistent Patterns

Observations that recur across multiple days. Each gets stronger (or weaker) with every new entry.

### Strengthening

| Pattern | Evidence | Confidence |
|---------|----------|------------|
| `#pattern:qqq-down-reliable` — QQQ DOWN with >60% conf is correct | 2026-07-29 (HIT), 2026-07-28 (HIT), 2026-07-27 (HIT) | HIGH — 3/3 |
| `#source:market-qqq-reliable` — `market:QQQ` snapshot data is predictive | Appears in all winning predictions | HIGH |
| `#pattern:qqq-amplifies` — Actual QQQ move exceeds prediction | 7/23, 7/24, 7/29: actual > predicted by 40-80% | MEDIUM — 3/3 |
| `#fix:yahoo-fallback` — Yahoo Finance saves day when Finnhub 502s | 2026-07-29 | CONFIRMED — 1/1 |

### Weakening

| Pattern | Evidence | Confidence |
|---------|----------|------------|
| `#source:news-sentiment-unreliable` — Aggregate news sentiment mispredicts SPY | 2026-07-29, 2026-07-28: appeared in SPY losing predictions | MEDIUM — 2/2 losses |
| `#pattern:spy-defense-rotation-unreliable` — SPY UP on defense rotation thesis fails when selloff is broad | 2026-07-29 (MISS), 2026-07-28 (MISS) | MEDIUM — 2/2 |

### Open Questions

| Question | Status |
|----------|--------|
| When briefing + LLM call QQQ DOWN with >60% conf, is scaling contracts to 2-3 worth it? | Needs more data |
| Do deterministic strategies (momentum, mean_reversion, event_driven) ever outperform the LLM? | All 3 abstained on 2026-07-29 — need a day where they fire |
| Should we stop trading SPY and focus only on QQQ? | SPY directional calls wrong 2/2 days; QQQ right 3/3 days |

---

## 2026-07-29

```yaml
date: 2026-07-29
spy:
  direction: UP
  confidence: 0.45
  predicted_move_pct: 0.35
  actual_move_pct: -1.42
  result: HIT
  note: "hit target $742.52 intraday (H=$742.67), but closed -1.42%"
qqq:
  direction: DOWN
  confidence: 0.70
  predicted_move_pct: -1.5
  actual_move_pct: -2.04
  result: HIT
  note: "blew past target, 40% deeper than predicted"
best_trade: "QQQ PUT @ $671, strategy=llm_trade"
trade_filled: false
engine_pnl: 0.00
manual_pnl: +1397.00
tags:
  - bug:occ-symbol
  - pattern:qqq-down-reliable
  - pattern:qqq-amplifies
  - system:exit-monitoring-dies
  - source:market-qqq-reliable
  - source:news-sentiment-unreliable
  - fix:occ-symbol-6-digit
  - fix:datetime-serialization
  - fix:two-sell-orders
  - fix:tpat-alpaca-not-in-process
  - fix:adguard-dns-exception
  - fix:paper-chain-truncation
```

### Pre-market context

Market vibe: Defense-primes bid on autonomous-systems spending; tech under broad pressure from AI safety concerns and rotation out of speculative names.

Key catalysts: AI, tech, selloff, gap, breach, defense, agent

Sources used: `market:QQQ`, `theverge:openai-agent-sandbox-escape`, `watchlist:LITE` (winning); `market:SPY`, `news-sentiment`, `watchlist:GD` (non-winning)

### What we predicted

| Asset | Direction | Confidence | Predicted Move | Rationale |
|-------|-----------|------------|----------------|-----------|
| SPY | UP | 45% | +0.35% | Defense spending (GD +1.04%, NOC +0.30%), positive news sentiment (+0.054) |
| QQQ | DOWN | 70% | -1.5% | AI safety breach + optical/space/defense-AI selloffs (LITE -8.43%, LUNR -7.06%, PLTR -6.08%) |

### What actually happened

| Asset | Open | High | Low | Close | Move % | Target Hit? |
|-------|------|------|-----|-------|--------|-------------|
| SPY | $739.97 | $742.67 | $729.10 | $729.46 | -1.42% | YES — hit $742.52 intraday |
| QQQ | $675.51 | $680.05 | $661.14 | $661.73 | -2.04% | YES — blew past $665.33 |

### Trade execution

| Metric | Value |
|--------|-------|
| Orders submitted | 6 |
| Filled | 1 (SPY PUT @ $1.60/contract) |
| Engine closed | 0 (exit monitoring died with pipeline) |
| Manually closed | All — net +$1,397 paper PnL |

### Bugs discovered & fixed

1. **`#bug:occ-symbol`** — `occ_option_symbol()` produced 8-digit date (`20260729`); Alpaca requires 6-digit (`260729`). QQQ PUT order rejected 422. Fixed in `src/mcp/schemas.py:119-121`.
2. **`#fix:datetime-serialization`** — `alpaca-py` returns `datetime` objects; Pydantic `OrderResult` expected `str`. Fixed in `src/execution/client.py`.
3. **`#fix:two-sell-orders`** — Alpaca prohibits two simultaneous sell orders for same contract. Engine places only TP at Alpaca; SL managed by safety-close cron.
4. **`#system:exit-monitoring-dies`** — Pipeline process exits ~40s, killing in-process monitoring. Positions left open. Fixed: TP is `time_in_force: "day"` at Alpaca; `safety_close.sh` force-closes at 3:20 PM ET.
5. **`#fix:adguard-dns-exception`** — `ericsender.com` resolved locally, breaking SSL. Fixed with `@@||ericsender.com^$dnsrewrite` in AdGuard.
6. **`#fix:paper-chain-truncation`** — Default `GET /v2/options/contracts` returns only calls. Fixed by passing `type=put` for put contracts.

### What we learned

- **QQQ DOWN + >60% conf is reliable**: 3/3 days now. When the LLM confidently calls QQQ DOWN, trust it. Consider scaling.
- **SPY directional calls are noisy**: 0/2 on SPY direction. Defense rotation thesis breaks when the whole market sells off. Either skip SPY or require stronger conf threshold (>60%).
- **Verify OCC symbol before submitting**: Query the contract endpoint first. A 5-second pre-check would have saved the day's winning trade.
- **Engine can't self-manage exits**: The pipeline process is too short-lived for in-process monitoring. Alpaca-hosted TP + safety-close cron is the right architecture.
- **Finnhub 502s are common**: Yahoo Finance fallback in snapshot loader is critical. Worth adding Alpha Vantage as third fallback.
- **Market:QQQ snapshot data is the most reliable input signal**: It appears in every winning QQQ prediction.

---

## 2026-07-18b

```yaml
date: 2026-07-18
is_architectural_change: true
tags:
  - arch:redesign
  - model:prediction-engine
  - fix:strike-calculation
  - fix:source-citation
  - fix:forecast-table
```

### What we changed

Redesigned from a single-trade executor to a per-asset directional prediction engine.

| Before | After |
|--------|-------|
| LLM chose exactly ONE trade | LLM predicts direction/move/confidence for ALL target assets |
| Opaque UP/DOWN/SIDE/MOVE columns | Direction / Confidence / Pred. Move / Key Drivers |
| Sources cited "atlas-briefing" | Sources traced to original publisher (reuters:, bloomberg:, market:) |
| Strikes 15% OTM (unfillable) | Strikes ~0.6% OTM using `underlying * (1 - delta * 0.02)` |

### Design decisions

- **Deterministic strategies** (momentum, mean_reversion, event_driven) run alongside LLM — used for consensus but often abstain when LLM fires.
- **`best_trade` field** lets LLM still suggest executable trade when signal is strong.
- **Execution path** still needs Alpaca credentials (dry-run safe by default).

---

## 2026-07-18

```yaml
date: 2026-07-18
is_system_incident: true
severity: high
tags:
  - system:degraded-briefing
  - fix:quality-detection
  - upstream:atlas
  - source:status-json
```

### What happened

The upstream atlas-morning-briefing pipeline delivered a briefing with NO real LLM content — 65 lines vs typical 140-300. Executive summary said "Synthesis unavailable for today's briefing."

### Root causes (upstream, fixed same day)

1. **Cron PATH mismatch** — `opencode` binary not found because `/home/linuxbrew/.linuxbrew/bin` was missing from PATH. Entire LLM layer silently skipped.
2. **No model fallback** — Free-tier DeepSeek primary hung indefinitely. No backup model configured. Fixed by adding `opencode-go/glm-5.2` fallback.

### How this project responded

| What we did | Where |
|------------|-------|
| Detect degraded summary prefix `"Synthesis unavailable..."` at parse time | `src/ingestion/parser.py` |
| Classify briefing quality as `FULL`, `DEGRADED`, or `FAILED` | `BriefingData.briefing_quality` |
| Treat `macro_sentiment` as missing (not neutral) when degraded | `src/models/briefing.py` |
| Read upstream `status.json` for `intelligence_enabled` ground truth | New loader alongside briefing markdown |
| Don't panic — section headers are stable even in degraded briefings | Existing regex parsers keep working |

### What we learned

- **Never trust the briefing at face value** — check `status.json` for ground truth and scan for the degradation signature.
- **Zero sentiment is NOT the same as neutral sentiment** — a degraded briefing silently returns 0.0, but that means "we don't know."
- **Deterministic fallback strings are reliable detectors** — the `"Synthesis unavailable..."` prefix is emitted by the upstream's `generate_markdown_briefing()` and is a 100% reliable degradation signal.
- **One file read per run, no new dependency** — `status.json` is already present in the upstream project.

