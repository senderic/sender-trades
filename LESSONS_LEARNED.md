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

## 2026-08-17

```yaml
date: 2026-08-17
spy:
  direction: UP
  confidence: 0.58
  predicted_move_pct: 0.2
  actual_move_pct: -0.45
  result: MISS
  note: "never hit target, closed -0.45%"
qqq:
  direction: UP
  confidence: 0.55
  predicted_move_pct: 0.15
  actual_move_pct: -0.42
  result: HIT
  note: "hit target, but reversed — closed -0.42%"
best_trade: "QQQ CALL @ $735.0, strategy=llm_trade"
trade_filled: true
engine_pnl: 0.00
tags:
  - pattern:qqq-amplifies
  - pattern:spy-up-unreliable
  - source:market-QQQ
  - source:market-QQQ-reliable
  - source:news-sentiment
  - source:news-sentiment-reliable
  - source:news-sentiment-unreliable
  - source:watchlist-EQIX
  - source:watchlist-EQIX-unreliable
  - source:watchlist-LITE
  - source:watchlist-LITE-reliable
  - source:watchlist-VRT
  - source:watchlist-VRT-unreliable
```

### Pre-market context

Market vibe: Mildly constructive: positive news polarity (+0.154) with AI infrastructure and defense strength (EQIX +2.6%, LMT +1.8%, LITE +5.2%) offsetting a broad-tech slip (ANET -2.4%, AMZN -0.9%); tiny gaps opening flat-to-slightly-negative with no gap-fade alarm.

Key catalysts: AI

### What we predicted

| Asset | Direction | Confidence | Predicted Move | Rationale (truncated) |
|-------|-----------|------------|----------------|-----------------------|
| SPY | UP | 58% | 0.2% | Positive aggregate news sentiment (+0.154, historically the most reliable signal at 11/11) and resilient AI data-center  |
| QQQ | UP | 55% | 0.1% | Positive sentiment backdrop and surging AI optical/data-center names (LITE +5.19%, EQIX +2.64%) counter a mild broad-tec |

### What actually happened

| Asset | Open | High | Low | Close | Move % | Result |
|-------|------|------|-----|-------|--------|--------|
| SPY | $776.18 | $776.78 | $772.51 | $772.67 | -0.45% | :x: MISS |
| QQQ | $732.95 | $734.58 | $729.27 | $729.87 | -0.42% | :white_check_mark: HIT |

### Trade execution

**1** orders submitted, **1** filled

-   QQQ CALL @ $1.14/contract → not closed by engine (pending)

### Strategy summary

| Strategy | SPY | QQQ |
|----------|-----|-----|
| momentum | — | — |
| mean_reversion | — | — |
| event_driven | — | — |
| llm_trade | :x: UP | :white_check_mark: UP |

**Cumulative prediction record**: 24/33 (73%)

---

## 2026-08-14

```yaml
date: 2026-08-14
spy:
  direction: UP
  confidence: 0.55
  predicted_move_pct: 0.35
  actual_move_pct: -0.28
  result: MISS
  note: "never hit target, closed -0.28%"
qqq:
  direction: UP
  confidence: 0.68
  predicted_move_pct: 0.65
  actual_move_pct: -0.32
  result: MISS
  note: "never hit target, closed -0.32%"
best_trade: "QQQ CALL @ $736.0, strategy=llm_trade"
trade_filled: true
engine_pnl: 0.00
tags:
  - pattern:qqq-up-unreliable
  - pattern:spy-up-unreliable
  - source:market-SPY
  - source:market-SPY-unreliable
  - source:news-sentiment
  - source:news-sentiment-unreliable
  - source:watchlist-META
  - source:watchlist-META-unreliable
  - source:watchlist-QQQ
  - source:watchlist-QQQ-unreliable
  - source:watchlist-SPY
  - source:watchlist-SPY-unreliable
```

### Pre-market context

Market vibe: Mildly risk-on with a narrow AI/autonomy leadership bid: megacap tech and defense-AI names (GOOGL, META, PLTR) lift QQQ while defense primes and space stocks drift lower; broad indices grind up on modestly positive sentiment.

Key catalysts: AI, tech, gap

### What we predicted

| Asset | Direction | Confidence | Predicted Move | Rationale (truncated) |
|-------|-----------|------------|----------------|-----------------------|
| SPY | UP | 55% | 0.3% | SPY gapped +0.7% and is holding above its close on mildly positive (+0.107) aggregate sentiment, but mixed sector action |
| QQQ | UP | 68% | 0.7% | QQQ gapped +1.2% and held, backed by broad AI optimism across GOOGL +0.8%, META +2.8%, PLTR +4.7% and NVDA +0.5%, with a |

### What actually happened

| Asset | Open | High | Low | Close | Move % | Result |
|-------|------|------|-----|-------|--------|--------|
| SPY | $778.54 | $778.80 | $775.43 | $776.34 | -0.28% | :x: MISS |
| QQQ | $733.41 | $734.39 | $728.32 | $731.07 | -0.32% | :x: MISS |

### Trade execution

**1** orders submitted, **1** filled

-   QQQ CALL @ $0.87/contract → not closed by engine (pending)

### Strategy summary

| Strategy | SPY | QQQ |
|----------|-----|-----|
| momentum | — | — |
| mean_reversion | — | — |
| event_driven | — | — |
| llm_trade | :x: UP | :x: UP |

**Cumulative prediction record**: 24/31 (77%)

---

## 2026-08-13

```yaml
date: 2026-08-13
spy:
  direction: UP
  confidence: 0.55
  predicted_move_pct: 0.3
  actual_move_pct: 0.39
  result: HIT
  note: "hit target (H=$779.37)"
qqq:
  direction: UP
  confidence: 0.58
  predicted_move_pct: 0.8
  actual_move_pct: 0.95
  result: HIT
  note: "hit target (H=$733.96)"
best_trade: "QQQ CALL @ $728.0, strategy=llm_trade"
trade_filled: true
engine_pnl: 74.00
tags:
  - source:market-QQQ
  - source:market-QQQ-reliable
  - source:market-SPY
  - source:market-SPY-reliable
  - source:news-sentiment
  - source:news-sentiment-reliable
  - source:watchlist-ANET
  - source:watchlist-ANET-reliable
  - source:watchlist-NVDA
  - source:watchlist-NVDA-reliable
```

### Pre-market context

Market vibe: Mildly positive, rotated tape: AI chip/networking/data-center infrastructure (LITE +13.6%, ANET +6.4%, NVDA +3.0%, EQIX, VRT) is leading while mega-cap tech (META -3.4%, MSFT -2.3%, AMZN -1.8%) sells off; defense/AI-defense themes in focus with sentiment at +0.126.

Key catalysts: AI, gap

### What we predicted

| Asset | Direction | Confidence | Predicted Move | Rationale (truncated) |
|-------|-----------|------------|----------------|-----------------------|
| SPY | UP | 55% | 0.3% | Broad market holds green on mild +0.126 sentiment with AI-infrastructure strength offsetting a mega-cap tech selloff (ME |
| QQQ | UP | 58% | 0.8% | QQQ stays positive despite mega-cap drag because AI chip and networking demand is surging (NVDA +3.0%, ANET +6.4%, LITE  |

### What actually happened

| Asset | Open | High | Low | Close | Move % | Result |
|-------|------|------|-----|-------|--------|--------|
| SPY | $774.87 | $779.37 | $774.11 | $777.88 | +0.39% | :white_check_mark: HIT |
| QQQ | $725.15 | $733.96 | $724.04 | $732.07 | +0.95% | :white_check_mark: HIT |

### Trade execution

**1** orders submitted, **1** filled

-   QQQ CALL @ $0.73/contract → +$74.00 (take_profit)

Engine-tracked PnL: +$74.00

### Strategy summary

| Strategy | SPY | QQQ |
|----------|-----|-----|
| momentum | — | — |
| mean_reversion | — | — |
| event_driven | — | — |
| llm_trade | :white_check_mark: UP | :white_check_mark: UP |

**Cumulative prediction record**: 22/29 (76%)

---

## 2026-08-12

```yaml
date: 2026-08-12
spy:
  direction: DOWN
  confidence: 0.6
  predicted_move_pct: -0.35
  actual_move_pct: -0.29
  result: HIT
  note: "hit target (L=$771.29)"
qqq:
  direction: DOWN
  confidence: 0.62
  predicted_move_pct: -0.4
  actual_move_pct: -0.46
  result: HIT
  note: "hit target (L=$722.92)"
best_trade: "SPY CALL @ $775.0, strategy=momentum"
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
  - source:watchlist-GOOGL
  - source:watchlist-GOOGL-reliable
```

### Pre-market context

Market vibe: Broad tech selloff on China open-model AI competition fears; defense/space names diverge higher (BKSY +4.5%, LUNR +4.4%) while legacy mega-caps AMZN/GOOGL/MSFT weaken; red-banded sentiment at +0.112 too weak to support equity indexes.

### What we predicted

| Asset | Direction | Confidence | Predicted Move | Rationale (truncated) |
|-------|-----------|------------|----------------|-----------------------|
| SPY | DOWN | 60% | -0.3% | SPY gapped only +0.19% but has faded to -0.32% on the day amid a broad market dip and mega-cap weakness (AMZN -2.09%, GO |
| QQQ | DOWN | 62% | -0.4% | QQQ opened +0.34% but reversed to -0.34% as AI competition fears from China's open-weight models hit mega-caps (GOOGL -3 |

### What actually happened

| Asset | Open | High | Low | Close | Move % | Result |
|-------|------|------|-----|-------|--------|--------|
| SPY | $774.71 | $774.87 | $771.29 | $772.49 | -0.29% | :white_check_mark: HIT |
| QQQ | $727.08 | $727.25 | $722.92 | $723.70 | -0.46% | :white_check_mark: HIT |

### Trade execution

**1** orders submitted, **1** filled

-   SPY CALL @ $0.69/contract → not closed by engine (pending)

### Strategy summary

| Strategy | SPY | QQQ |
|----------|-----|-----|
| momentum | — | — |
| mean_reversion | — | — |
| event_driven | — | — |
| llm_trade | :white_check_mark: DOWN | :white_check_mark: DOWN |

**Cumulative prediction record**: 20/27 (74%)

---

## 2026-08-11

```yaml
date: 2026-08-11
spy:
  direction: DOWN
  confidence: 0.52
  predicted_move_pct: -0.3
  actual_move_pct: -0.51
  result: HIT
  note: "hit target (L=$769.34)"
qqq:
  direction: DOWN
  confidence: 0.6
  predicted_move_pct: -0.55
  actual_move_pct: -0.67
  result: HIT
  note: "hit target (L=$715.79)"
best_trade: "SPY CALL @ $778.0, strategy=event_driven"
trade_filled: true
engine_pnl: 0.00
tags:
  - pattern:qqq-down-reliable
  - pattern:spy-amplifies
  - source:market-QQQ
  - source:market-QQQ-reliable
  - source:market-SPY
  - source:market-SPY-reliable
  - source:news-sentiment
  - source:news-sentiment-reliable
  - source:watchlist-LITE
  - source:watchlist-LITE-reliable
  - source:watchlist-NVDA
  - source:watchlist-NVDA-reliable
```

### Pre-market context

Market vibe: Rotational tape: defense/AI (LMT, GD, SPCX, PLTR) rallies on Ukraine-combat-validated autonomy and Pentagon AI diffusion news while big tech takes profits; indices flat to slightly down despite mildly positive news sentiment (+0.101).

### What we predicted

| Asset | Direction | Confidence | Predicted Move | Rationale (truncated) |
|-------|-----------|------------|----------------|-----------------------|
| SPY | DOWN | 52% | -0.3% | SPY sits flat at -0.03% with no positive catalyst, as NVDA profit-taking (-2.86%) and broad tech weakness offset the def |
| QQQ | DOWN | 60% | -0.6% | QQQ trades at -0.30% and is structurally weakest today: NVDA -2.86%, LITE -8.61% on earnings disappointment, and VGT -0. |

### What actually happened

| Asset | Open | High | Low | Close | Move % | Result |
|-------|------|------|-----|-------|--------|--------|
| SPY | $774.53 | $774.61 | $769.34 | $770.56 | -0.51% | :white_check_mark: HIT |
| QQQ | $723.29 | $723.30 | $715.79 | $718.45 | -0.67% | :white_check_mark: HIT |

### Trade execution

**1** orders submitted, **1** filled

-   SPY CALL @ $0.20/contract → not closed by engine (pending)

### Strategy summary

| Strategy | SPY | QQQ |
|----------|-----|-----|
| momentum | — | — |
| mean_reversion | — | — |
| event_driven | — | — |
| llm_trade | :white_check_mark: DOWN | :white_check_mark: DOWN |

**Cumulative prediction record**: 18/25 (72%)

---

## 2026-08-10

```yaml
date: 2026-08-10
spy:
  direction: ?
  confidence: 0.0
  predicted_move_pct: 0.0
  actual_move_pct: 0.06
  result: MISS
  note: "never hit target, closed +0.06%"
qqq:
  direction: UP
  confidence: 1.0
  predicted_move_pct: 0.3
  actual_move_pct: -0.21
  result: HIT
  note: "hit target, but reversed — closed -0.21%"
best_trade: "QQQ CALL @ $727.0, strategy=momentum"
trade_filled: true
engine_pnl: 0.00
tags:
  - pattern:qqq-up-reliable
```

### Pre-market context

### What we predicted

| Asset | Direction | Confidence | Predicted Move | Rationale (truncated) |
|-------|-----------|------------|----------------|-----------------------|
| SPY | ? | 0% | 0.0% |  |
| QQQ | UP | 100% | 0.3% |  |

### What actually happened

| Asset | Open | High | Low | Close | Move % | Result |
|-------|------|------|-----|-------|--------|--------|
| SPY | $772.60 | $775.03 | $771.89 | $773.03 | +0.06% | :x: MISS |
| QQQ | $722.39 | $724.67 | $720.33 | $720.87 | -0.21% | :white_check_mark: HIT |

### Trade execution

**1** orders submitted, **1** filled

-   QQQ CALL @ $0.73/contract → not closed by engine (pending)

### Strategy summary

| Strategy | SPY | QQQ |
|----------|-----|-----|
| momentum | — | — |
| mean_reversion | — | — |
| event_driven | — | — |
| llm_trade | — | — |

**Cumulative prediction record**: 17/24 (71%)

---

## 2026-08-07

```yaml
date: 2026-08-07
spy:
  direction: UP
  confidence: 0.58
  predicted_move_pct: 0.25
  actual_move_pct: 0.29
  result: HIT
  note: "hit target (H=$773.91)"
qqq:
  direction: DOWN
  confidence: 0.55
  predicted_move_pct: -0.3
  actual_move_pct: 0.40
  result: HIT
  note: "hit target, but reversed — closed +0.40%"
best_trade: "SPY CALL @ $773.0, strategy=llm_trade"
trade_filled: true
engine_pnl: 0.00
tags:
  - pattern:qqq-amplifies
  - source:market-QQQ
  - source:market-QQQ-reliable
  - source:market-SPY
  - source:market-SPY-reliable
  - source:news-sentiment
  - source:news-sentiment-reliable
  - source:watchlist-ANET
  - source:watchlist-ANET-reliable
  - source:watchlist-GD
  - source:watchlist-GD-reliable
  - source:watchlist-GOOGL
  - source:watchlist-GOOGL-reliable
```

### Pre-market context

Market vibe: Defense and space stocks rally (BKSY, RDW, LUNR, RKLB) on AI-agent/edge-deployment and counter-drone themes, while broad tech trades mixed-to-weak with sector rotation and mild positive news sentiment overall.

Key catalysts: defense

### What we predicted

| Asset | Direction | Confidence | Predicted Move | Rationale (truncated) |
|-------|-----------|------------|----------------|-----------------------|
| SPY | UP | 58% | 0.2% | Shallow dip (-0.16%) with a positive gap, slightly positive aggregate news sentiment (+0.088), and broad defense/sector  |
| QQQ | DOWN | 55% | -0.3% | QQQ gapped down -0.90% and remains -0.37% on the day amid persistent tech selling in heavyweights (GOOGL -1.29%, PLTR -1 |

### What actually happened

| Asset | Open | High | Low | Close | Move % | Result |
|-------|------|------|-----|-------|--------|--------|
| SPY | $771.02 | $773.91 | $769.61 | $773.26 | +0.29% | :white_check_mark: HIT |
| QQQ | $720.15 | $723.63 | $716.52 | $723.03 | +0.40% | :white_check_mark: HIT |

### Trade execution

**1** orders submitted, **1** filled

-   SPY CALL @ $0.73/contract → not closed by engine (pending)

### Strategy summary

| Strategy | SPY | QQQ |
|----------|-----|-----|
| momentum | — | — |
| mean_reversion | — | — |
| event_driven | — | — |
| llm_trade | :white_check_mark: UP | :white_check_mark: DOWN |

**Cumulative prediction record**: 15/22 (68%)

---

## 2026-08-05

```yaml
date: 2026-08-05
spy:
  direction: UP
  confidence: 0.62
  predicted_move_pct: 1.3
  actual_move_pct: -0.78
  result: MISS
  note: "never hit target, closed -0.78%"
qqq:
  direction: UP
  confidence: 0.67
  predicted_move_pct: 2.2
  actual_move_pct: -1.23
  result: MISS
  note: "never hit target, closed -1.23%"
best_trade: "SPY CALL @ $776.0, strategy=event_driven"
trade_filled: true
engine_pnl: 0.00
tags:
  - pattern:qqq-up-unreliable
  - pattern:spy-up-unreliable
  - source:market-QQQ
  - source:market-QQQ-unreliable
  - source:market-SPY
  - source:market-SPY-unreliable
  - source:market-VGT
  - source:market-VGT-unreliable
  - source:news-sentiment
  - source:news-sentiment-unreliable
```

### Pre-market context

Market vibe: Risk-on AI and space rally: inference-cost collapse (DeepSeek V4-Flash ~105x cheaper) around autonomous defense procurement spurring tech/space beta (RDW +11.8%, RKLB +8.4%, LUNR +6.2%), while defense primes stay flat. Positive news sentiment +0.128 with strong pre-market gaps.

### What we predicted

| Asset | Direction | Confidence | Predicted Move | Rationale (truncated) |
|-------|-----------|------------|----------------|-----------------------|
| SPY | UP | 62% | 1.3% | SPY gapped +1.8% pre-market to $771.33 and is tracking +1.80% on day amid AI-led broad market rally, but recent misses ( |
| QQQ | UP | 67% | 2.2% | QQQ gapped +3.4% pre-market to $723.85 and is +3.40% on day, riding breadth in mega-cap AI names (META +6.02%, GOOGL +4. |

### What actually happened

| Asset | Open | High | Low | Close | Move % | Result |
|-------|------|------|-----|-------|--------|--------|
| SPY | $775.85 | $776.85 | $769.52 | $769.79 | -0.78% | :x: MISS |
| QQQ | $726.25 | $728.54 | $716.92 | $717.30 | -1.23% | :x: MISS |

### Trade execution

**3** orders submitted, **2** filled

-   SPY CALL @ $0.09/contract → not closed by engine (pending)
-   SPY CALL @ $0.06/contract → not closed by engine (pending)

### Strategy summary

| Strategy | SPY | QQQ |
|----------|-----|-----|
| momentum | — | — |
| mean_reversion | — | — |
| event_driven | — | — |
| llm_trade | :x: UP | :x: UP |

**Cumulative prediction record**: 15/22 (68%)

---

## 2026-08-04

```yaml
date: 2026-08-04
spy:
  direction: UP
  confidence: 0.68
  predicted_move_pct: 0.85
  actual_move_pct: 1.41
  result: HIT
  note: "hit target (H=$773.41)"
qqq:
  direction: UP
  confidence: 0.72
  predicted_move_pct: 1.15
  actual_move_pct: 2.22
  result: HIT
  note: "hit target (H=$725.65)"
best_trade: "QQQ CALL @ $704.0, strategy=llm_trade"
trade_filled: false
engine_pnl: 0.00
tags:
  - pattern:qqq-amplifies
  - pattern:qqq-up-reliable
  - pattern:spy-amplifies
  - pattern:spy-up-reliable
  - source:market-QQQ
  - source:market-QQQ-reliable
  - source:market-SPY
  - source:market-SPY-reliable
  - source:news-sentiment
  - source:news-sentiment-reliable
  - source:reuters-ai-cost-collapse
  - source:reuters-ai-cost-collapse-reliable
  - source:watchlist-QQQ
  - source:watchlist-QQQ-reliable
  - source:watchlist-SPY
  - source:watchlist-SPY-reliable
  - system:no-trades-filled
```

### Pre-market context

Market vibe: Strong risk-on sentiment driven by AI cost breakthroughs and defense-tech procurement boom, with tech leading broader market rally.

Key catalysts: AI, tech

### What we predicted

| Asset | Direction | Confidence | Predicted Move | Rationale (truncated) |
|-------|-----------|------------|----------------|-----------------------|
| SPY | UP | 68% | 0.8% | Broad market rally with +1.42% gain and positive news sentiment (+0.179) supports continued upside despite gap-up. |
| QQQ | UP | 72% | 1.1% | Tech sector leadership with AI/defense-tech rally driving mega-caps (META +6%, GOOGL +4.88%, MSFT +4.93%) and +1.76% gai |

### What actually happened

| Asset | Open | High | Low | Close | Move % | Result |
|-------|------|------|-----|-------|--------|--------|
| SPY | $760.63 | $773.41 | $760.52 | $771.33 | +1.41% | :white_check_mark: HIT |
| QQQ | $708.16 | $725.65 | $707.59 | $723.85 | +2.22% | :white_check_mark: HIT |

### Trade execution

**1** orders submitted, **0** filled


### Strategy summary

| Strategy | SPY | QQQ |
|----------|-----|-----|
| momentum | — | — |
| mean_reversion | — | — |
| event_driven | — | — |
| llm_trade | :white_check_mark: UP | :white_check_mark: UP |

**Cumulative prediction record**: 13/20 (65%)

---

## 2026-08-03

```yaml
date: 2026-08-03
spy:
  direction: UP
  confidence: 0.62
  predicted_move_pct: 0.5
  actual_move_pct: 1.10
  result: HIT
  note: "hit target (H=$758.58)"
qqq:
  direction: UP
  confidence: 0.65
  predicted_move_pct: 0.7
  actual_move_pct: 1.71
  result: HIT
  note: "hit target (H=$701.59)"
best_trade: "SPY CALL @ $752.0, strategy=momentum+event_driven"
trade_filled: true
engine_pnl: 0.00
tags:
  - pattern:qqq-amplifies
  - pattern:qqq-up-reliable
  - pattern:spy-amplifies
  - pattern:spy-up-reliable
  - source:dowjones-market-wrap
  - source:dowjones-market-wrap-reliable
  - source:market-QQQ
  - source:market-QQQ-reliable
  - source:market-SPY
  - source:market-SPY-reliable
  - source:news-sentiment
  - source:news-sentiment-reliable
  - source:reuters-palantir-nvidia-battlefield-ai
  - source:reuters-palantir-nvidia-battlefield-ai-reliable
  - source:watchlist-NVDA
  - source:watchlist-NVDA-reliable
```

### Pre-market context

Market vibe: Risk-on tech bid; AI infrastructure via NVDA battlefield-AI pact and AWS-driven cloud surge fuels a positive backdrop, though data-center-vulnerability concerns (EQIX -2.7%) cap euphoria.

### What we predicted

| Asset | Direction | Confidence | Predicted Move | Rationale (truncated) |
|-------|-----------|------------|----------------|-----------------------|
| SPY | UP | 62% | 0.5% | Positive pre-market gap +0.4%, now +0.72% on day, with broad market strength (+0.72%) and positive news sentiment suppor |
| QQQ | UP | 65% | 0.7% | Largest gap (+1.25%) with NVDA +2.9% battlefield-AI deal and FANG+ cloud/frontier-model momentum (AMZN +15.3%, GOOGL +6. |

### What actually happened

| Asset | Open | High | Low | Close | Move % | Result |
|-------|------|------|-----|-------|--------|--------|
| SPY | $749.44 | $758.58 | $748.80 | $757.67 | +1.10% | :white_check_mark: HIT |
| QQQ | $688.30 | $701.59 | $685.82 | $700.07 | +1.71% | :white_check_mark: HIT |

### Trade execution

**1** orders submitted, **1** filled

-   SPY CALL @ $0.60/contract → not closed by engine (pending)

### Strategy summary

| Strategy | SPY | QQQ |
|----------|-----|-----|
| momentum | — | — |
| mean_reversion | — | — |
| event_driven | — | — |
| llm_trade | :white_check_mark: UP | :white_check_mark: UP |

**Cumulative prediction record**: 11/18 (61%)

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
