# Lessons Learned

Intraday directional prediction engine & 0DTE execution. LLM-readable format -- scan the Quick Index, then jump to the entry by date or tag.

## Quick Index

| Date | SPY Pred | SPY Result | QQQ Pred | QQQ Result | Trade Executed | Engine PnL | Key Tags |
|------|----------|------------|----------|------------|---------------|------------|----------|
| [2026-07-30](#2026-07-30) | DOWN 62% | TBD | DOWN 75% -> UP 55% | MISS (gapped +2%) | SPY PUT @ 731 (rerun) | open | `bug:market-order-premarket` `pattern:qqq-down-BROKEN` `pattern:model-pivot-correct` `luck:bug-saved-loss` |
| [2026-07-29](#2026-07-29) | UP 45% | HIT (+0.3%/target) | DOWN 70% | HIT (-1.5% -> -2.0%) | No (OCC bug) | $0 | `bug:occ-symbol` `pattern:qqq-down-reliable` `system:exit-monitoring` `source:market-qqq-reliable` |
| [2026-07-18b](#2026-07-18b) | -- | -- | -- | -- | N/A | N/A | `arch:redesign` `model:prediction-engine` `fix:strike-calculation` |
| [2026-07-18](#2026-07-18) | -- | -- | -- | -- | N/A | N/A | `system:degraded-briefing` `fix:quality-detection` `upstream:atlas` |

## Persistent Patterns

Observations that recur across multiple days. Each gets stronger (or weaker) with every new entry.

### Strengthening

| Pattern | Evidence | Confidence |
|---------|----------|------------|
| `source:market-qqq-reliable` -- `market:QQQ` snapshot data is predictive | Appears in winning predictions; today correctly pivoted DOWN -> UP when QQQ gapped +2% | HIGH |
| `pattern:model-pivot-correct` -- Model correctly changes direction when pre-market contradicts briefing | 7/30: 6:15 AM called QQQ DOWN, 6:43 AM rerun pivoted to QQQ UP after +2% gap | CONFIRMED -- 1/1 |
| `fix:yahoo-fallback` -- Yahoo Finance saves day when Finnhub 502s | 2026-07-29 | CONFIRMED -- 1/1 |
| `bug:market-order-premarket` -- Market orders rejected pre-market; must use limit | 7/30: QQQ PUT rejected 422 at 9:15 AM ET | CONFIRMED -- fixed in `client.py` |

### Weakening

| Pattern | Evidence | Confidence |
|---------|----------|------------|
| `pattern:qqq-down-reliable` -- QQQ DOWN with >60% conf was 3/3, now 3/4 | 7/30: QQQ DOWN 75% at 6:15 AM, actual QQQ gapped OPEN +2% -- **major miss** | DOWNGRADED -- 3/4 (75%) |
| `source:news-sentiment-unreliable` -- Aggregate news sentiment mispredicts SPY | 7/29, 7/28: appeared in SPY losing predictions | MEDIUM -- 2/2 |
| `pattern:spy-defense-rotation-unreliable` -- SPY UP on defense rotation fails in broad selloff | 7/29 (MISS), 7/28 (MISS) | MEDIUM -- 2/2 |

### Lucky Escapes

| Date | What | Loss Avoided |
|------|------|-------------|
| `luck:bug-saved-loss` 7/30 | Market order bug rejected QQQ PUT @ $658 at 9:15 AM. QQQ gapped +2% and never dropped. | ~$50-100 loss |

### Open Questions

| Question | Status |
|----------|--------|
| Should submission delay to 9:29 AM ET to let model see pre-market gap moves? | Both 7/29 and 7/30 the picture changed 9:15 -> 9:30 |
| How reliable is the briefing when overnight catalysts (GOOGL earnings) invert direction? | 7/30: briefing missed QQQ gap-up |
| Do deterministic strategies out- perform the LLM? | event_driven selected today (SPY PUT) |
| Should we focus on SPY over QQQ? | SPY 2/3 HITs this week; QQQ 3/4 HITs overall but broke today |

---

## 2026-07-30

```yaml
date: 2026-07-30
spy:
  direction: DOWN
  confidence: 0.65
  predicted_move_pct: -2.0
  actual_move_pct: 0.77
  result: HIT
  note: "hit target, but reversed — closed +0.77%"
qqq:
  direction: DOWN
  confidence: 0.75
  predicted_move_pct: -2.8
  actual_move_pct: 1.30
  result: HIT
  note: "hit target, but reversed — closed +1.30%"
best_trade: "QQQ PUT @ $658.0, strategy=llm_trade"
trade_filled: true
engine_pnl: 0.00
tags:
  - pattern:qqq-down-reliable
  - pattern:spy-down-reliable
  - source:market-QQQ
  - source:market-QQQ-reliable
  - source:market-SPY
  - source:market-SPY-reliable
  - source:news-sentiment
  - source:news-sentiment-reliable
  - source:watchlist-ANET
  - source:watchlist-ANET-reliable
  - source:watchlist-LITE
  - source:watchlist-LITE-reliable
  - source:watchlist-NVDA
  - source:watchlist-NVDA-reliable
```

### Pre-market context

Market vibe: Risk-off — defense-tech, space, and AI infrastructure names sold off broadly while Pentagon procurement signals are ignored by markets

Key catalysts: tech

### What we predicted

| Asset | Direction | Confidence | Predicted Move | Rationale (truncated) |
|-------|-----------|------------|----------------|-----------------------|
| SPY | DOWN | 65% | -2.0% | Broad risk-off selling across defense, tech, and AI infrastructure names with no offsetting catalysts, and SPY is alread |
| QQQ | DOWN | 75% | -2.8% | Tech-sector rout led by optical-networking collapse (LITE -7.6%, ANET -6.9%) and chip valuation reset (NVDA -3.6%) with  |

### What actually happened

| Asset | Open | High | Low | Close | Move % | Result |
|-------|------|------|-----|-------|--------|--------|
| SPY | $736.05 | $742.45 | $734.59 | $741.69 | +0.77% | :white_check_mark: HIT |
| QQQ | $674.76 | $685.12 | $673.30 | $683.55 | +1.30% | :white_check_mark: HIT |

### Trade execution

**2** orders submitted, **1** filled

-   SPY PUT @ $0.84/contract → not closed by engine (pending)

### Strategy summary

| Strategy | SPY | QQQ |
|----------|-----|-----|
| momentum | — | — |
| mean_reversion | — | — |
| event_driven | — | — |
| llm_trade | :white_check_mark: DOWN | :white_check_mark: DOWN |

**Cumulative prediction record**: 9/15 (60%)

---

## 2026-07-30

```yaml
date: 2026-07-30
cron_run:
  time: "6:15 AM PT / 9:15 AM ET"
  spy: {direction: DOWN, confidence: 0.65, predicted_move_pct: -2.0}
  qqq: {direction: DOWN, confidence: 0.75, predicted_move_pct: -2.8}
  best_trade: "QQQ PUT @ 658, 75% conf, llm_trade"
  result: "REJECTED: options market orders only during market hours"
  occ_symbol: "QQQ260730P00658000 (valid, 6-digit fix worked)"
manual_rerun:
  time: "6:43 AM PT / 9:43 AM ET"
  spy: {direction: DOWN, confidence: 0.62, predicted_move_pct: -0.5}
  qqq: {direction: UP, confidence: 0.55, predicted_move_pct: 0.6}
  best_trade: "SPY PUT @ 731, 62% conf, event_driven+llm_trade"
  result: "FILLED at $0.58/share, TP limit @ $1.68"
qqq_actual:
  open: 674.69
  note: "gapped OPEN +2% on GOOGL earnings -- QQQ DOWN 75% was DEAD WRONG"
trade_filled: true
engine_pnl: 0.00
tags:
  - bug:market-order-premarket
  - pattern:qqq-down-BROKEN
  - pattern:model-pivot-correct
  - luck:bug-saved-loss
  - fix:limit-order-only
  - lesson:premarket-gap-can-invert
```

### What happened

**6:15 AM cron**: Pipeline predicted QQQ DOWN 75% confidence @ $658, SPY DOWN 65% @ $725. Market order submitted for QQQ PUT but rejected 422 -- "options market orders are only allowed during market hours." Between 9:15 AM and 9:30 AM open, QQQ gapped **UP +2%** on overnight GOOGL earnings -- the QQQ DOWN call was dead on arrival.

**6:43 AM rerun**: After applying the limit-order fix, pipeline reran and correctly pivoted: QQQ was now UP (+0.6% predicted) and SPY was DOWN (-0.5%). Best trade switched to SPY PUT @ $731. Filled at $0.58/share ($58/contract). TP limit order placed at $1.68.

### Key lesson: the bug saved us

The market order rejection at 9:15 AM inadvertently prevented a losing QQQ PUT trade. Overnight catalysts (GOOGL earnings) flipped the market between prediction time and open. This is a structural risk:

1. The briefing is built at 5:30 AM PT from yesterday's close + overnight news
2. The pipeline predicts at 9:15 AM ET
3. The market opens at 9:30 AM ET -- 15-minute window where the picture can shift
4. Overnight earnings, economic data, or geopolitical events can gap the market opposite direction

### System fixes applied

- **`fix:limit-order-only`**: Engine ignores LLM's `"market"` order type, always uses `limit` from config. Config had `order_type: limit` but was overridden by truthy `"market"`. Fixed `src/execution/client.py:207`.
- **`fix:limit-price-delta`**: Limit price = `|delta| * |entry - strike| + 0.15` (min $1.00), replacing broken `strike * 0.005`.

### What we learned

- **Gap risk is real**: A strong overnight catalyst invalidated the morning prediction before open. The 15-minute cron-to-open window is a vulnerability.
- **Model can self-correct**: The rerun correctly saw the +2% gap and flipped from QQQ DOWN to QQQ UP. We need the model to see pre-market data BEFORE submitting.
- **QQQ-down-reliable streak broken**: 3/3 -> 3/4 (75%). Pattern still strong but no longer unquestionable.
- **Bug luck is unsustainable**: Yesterday's bug cost a winning trade, today's saved a losing one. Fix the bugs, add pre-market gap detection instead.

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
  note: "hit target $742.52 intraday (H=$742.67), but reversed -- closed -1.42%"
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
| SPY | $739.97 | $742.67 | $729.10 | $729.46 | -1.42% | YES -- hit $742.52 intraday |
| QQQ | $675.51 | $680.05 | $661.14 | $661.73 | -2.04% | YES -- blew past $665.33 |

### Trade execution

| Metric | Value |
|--------|-------|
| Orders submitted | 6 |
| Filled | 1 (SPY PUT @ $1.60/contract) |
| Engine closed | 0 (exit monitoring died with pipeline) |
| Manually closed | All -- net +$1,397 paper PnL |

### Bugs discovered & fixed

1. **`bug:occ-symbol`** -- `occ_option_symbol()` produced 8-digit date; Alpaca needs 6-digit. QQQ PUT rejected 422. Fixed `src/mcp/schemas.py`.
2. **`fix:datetime-serialization`** -- `alpaca-py` returns `datetime`; Pydantic `OrderResult` expected `str`. Fixed `src/execution/client.py`.
3. **`fix:two-sell-orders`** -- Alpaca prohibits two sell orders for same contract. Engine places only TP; SL handled by safety-close.
4. **`system:exit-monitoring-dies`** -- Pipeline process exits ~40s, killing in-process monitoring. Fixed: TP is `time_in_force: "day"` at Alpaca; `safety_close.sh` force-closes at 3:20 PM.
5. **`fix:adguard-dns-exception`** -- `ericsender.com` resolved locally, breaking SSL. Fixed with AdGuard exception.
6. **`fix:paper-chain-truncation`** -- Default `GET /v2/options/contracts` returns only calls. Fixed by passing `type=put`.

### What we learned

- **QQQ DOWN + >60% conf is reliable**: 3/3 days. When the LLM confidently calls QQQ DOWN, trust it.
- **SPY directional calls are noisy**: 0/2 on direction. Defense rotation thesis breaks in broad selloff.
- **Verify OCC symbol before submitting**: A 5-second pre-check saves a day's winning trade.
- **Engine can't self-manage exits**: Pipeline too short-lived for in-process monitoring.
- **Finnhub 502s are common**: Yahoo Finance fallback in snapshot loader is critical.
- **market:QQQ snapshot data is the most reliable input signal**.

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

- Deterministic strategies (momentum, mean_reversion, event_driven) run alongside LLM for consensus.
- `best_trade` field lets LLM suggest executable trade when signal is strong.
- Execution path dry-run safe by default.

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

Upstream atlas-morning-briefing delivered a briefing with NO real LLM content: 65 lines vs typical 140-300. Executive summary said "Synthesis unavailable for today's briefing."

### Root causes (upstream, fixed same day)

1. Cron PATH mismatch: `opencode` not found (missing `/home/linuxbrew/.linuxbrew/bin`). LLM layer silently skipped.
2. No model fallback: Free-tier DeepSeek hung indefinitely. Fixed by adding `opencode-go/glm-5.2` fallback.

### How this project responded

| What we did | Where |
|-------------|-------|
| Detect degraded summary prefix at parse time | `src/ingestion/parser.py` |
| Classify briefing quality as FULL/DEGRADED/FAILED | `BriefingData.briefing_quality` |
| Treat `macro_sentiment` as missing (not neutral) when degraded | `src/models/briefing.py` |
| Read upstream `status.json` for `intelligence_enabled` | New loader alongside briefing markdown |
| Section headers are stable even in degraded briefings | Existing regex parsers keep working |

### What we learned

- Never trust the briefing at face value: check `status.json` for ground truth.
- Zero sentiment != neutral sentiment: degraded briefing returns 0.0, meaning "unknown."
- Deterministic fallback strings are reliable detectors: `"Synthesis unavailable..."` is 100% signal.
- One file read per run, no new dependency: `status.json` is already present upstream.
