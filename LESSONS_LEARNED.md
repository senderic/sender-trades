# Lessons Learned

Cross-project notes about running an LLM-driven intraday prediction system.

Newest entries at the top.

---

## 2026-07-29 — Post-market analysis

### Prediction accuracy
- :x: MISS **SPY UP** | 45% conf | predicted 0.3% | actual -0.80%
  _SPY is recovering +0.24% intraday from the $735.87 low with defense-prime spending (GD +1.04%, NOC +0.30%) and slightly positive aggregate news sentiment (+0.054) providing support._
- :white_check_mark: HIT **QQQ DOWN** | 70% conf | predicted -1.5% | actual -1.30%
  _QQQ gapped down -0.87% and continues to -0.97% as the OpenAI agent-safety breach burdens sentiment while routs in opticals (LITE -8.43%), space (LUNR -7.06%), and defense AI (PLTR -6.08%) drive broad _

**Best trade**: QQQ PUT, strategy=llm_trade

### Trade execution
**6** orders submitted, **1** filled
**1** positions were not closed by the engine (engine exit error). These were closed manually or by safety-close.
  SPY PUT @ $1.60/contract → not closed by engine (error)

### System issues
- Engine failed to close 1 filled positions (exit monitoring died with pipeline)

### Daily market
  SPY: O=$739.97 H=$740.39 L=$731.74 C=$734.03 (-0.80%)
  QQQ: O=$675.46 H=$677.46 L=$663.30 C=$666.71 (-1.30%)

**Cumulative prediction record**: 7/13 (54%)

## 2026-07-29 — First paper-trading day: wins, failures, and live debugging

### What happened

The pipeline ran at 9:15 AM ET via cron and produced **QQQ PUT @ $671, 70% confidence**. The LLM correctly predicted a DOWN day for QQQ driven by China's open-source Kimi K3 AI model threatening US tech valuations, enterprise networking selloffs (ANET/EQIX/LITE), and a -0.87% gap down.

The trade was NOT executed because of a bug: `occ_option_symbol()` produced an 8-digit year date (`20260729`) but Alpaca requires 6-digit (`260729`). By the time the bug was fixed at ~9:44 AM ET, the pipeline produced **SPY PUT @ $735** which filled on paper. Later runs opened SPY PUT @ $736 and QQQ PUT @ $671.

### What we got right

- **Prediction accuracy**: QQQ DOWN call was correct. QQQ opened at $675.46 and hit a low of $663.30 (-1.80%). The predicted move was -0.9% — actual was double.
- **Source quality**: The LLM's reasoning about Kimi K3 and tech rotation was accurate. QQQ gapped down -0.87% and never recovered.
- **Yahoo Finance fallback**: When Finnhub 502'd every symbol in the snapshot, Yahoo Finance filled the gap and the pipeline still produced a trade. This saved the day.
- **Trade timing**: SPY PUT @ $735 was profitable (opened around $740, closed at $734). The system's directional calls were directionally right on both assets.

### What went wrong

1. **OCC symbol format** — `occ_option_symbol()` uses 8-digit date; Alpaca uses 6-digit. Fixed by stripping century prefix.
2. **`datetime` serialization** — `alpaca-py` returns `datetime` objects for `created_at`/`updated_at`. Pydantic `OrderResult` expected `str`. Fixed with `.isoformat()` conversion.
3. **Cannot place two sell orders** — Alpaca rejects simultaneous sell orders for the same option contract. Engine now places only TP at Alpaca; SL + force-close handled by safety-close cron.
4. **Monitoring loop dies with pipeline** — Pipeline process exits after ~40 seconds, killing the in-app monitoring loop. Positions left open with no exits. Fixed: TP order is `time_in_force: "day"` at Alpaca, safety-close cron at 3:20 PM ET force-closes everything.
5. **AdGuard DNS rewrite** — `ericsender.com` was being resolved locally, breaking SSL and the site. Fixed by adding `@@||ericsender.com^$dnsrewrite` exception in AdGuard.
6. **Paper chain truncation** — Default `GET /v2/options/contracts` returns only calls. Puts require explicit `type=put` filter.

### Paper PnL (test trades)

All positions were manually closed after market hours:
- QQQ 671 PUT: +$530
- SPY 735 PUT x2: +$545
- SPY 736 PUT: +$322
- **Total**: +$1,397 on paper

The QQQ PUT @ $671 would have been the intended morning trade. If it had been the only position, it would have yielded the largest single gain.

### What we'd do differently

- Verify OCC symbol against the chain BEFORE submitting (query contract endpoint first)
- Run a quick validation trade at system startup using the test credentials
- Don't trust in-process monitoring — put exit orders at Alpaca directly
- Always test with `type=put` filter for put contracts

### Market data note

Finnhub free tier 502'd every symbol in the atlas snapshot today. The Yahoo Finance fallback (`snapshot_loader._fetch_yahoo_quote()`) was essential. Consider adding Alpha Vantage as a third fallback.

---

## 2026-07-18b — Redesigned from trade-executor to prediction-engine

### What we changed

The system was originally designed to find a single trade (asset, direction,
strike, contracts) and execute it via MCP. Users found the output confusing
— "buy sell side" language, opaque UP/DOWN/SIDE columns, absurd strikes (15%
OTM), and no clear prediction of *how much* an asset would move.

### Changes made

1. **LLM prompt redesigned**: Instead of "Choose exactly ONE trade", the LLM
   now outputs per-asset predictions for ALL target assets: direction (UP/DOWN),
   confidence, predicted_move_pct, rationale, and root-provenance sources.

2. **Forecast table simplified**: Replaced UP/DOWN/SIDE/MOVE columns with
   Direction / Confidence / Pred. Move / Key Drivers of the Prediction — clear at a glance.

3. **Source citation improved**: The LLM is now instructed to trace evidence
   back to original publishers (reuters:, bloomberg:, market:) rather than
   citing "atlas-briefing" as a root source.

4. **Strike computation fixed**: Previously used `underlying * 0.85` for puts
   (15% OTM). Now uses `underlying * (1 - delta * 0.02)` — ~0.6% OTM for
   30-delta, producing strikes that actually exist in the chain
   (e.g. QQQ PUT @ 691 instead of 591).

### What to watch

- The optional `best_trade` field lets the LLM still suggest an executable
  trade when the signal is strong. The execution path (risk checks, MCP)
  still needs Alpaca credentials configured.
- Deterministic strategies (momentum, mean-reversion, event-driven) often
  abstain when the LLM fires — may want to reconsider their value.

---

## 2026-07-18 — Briefing can silently come back "empty" or degraded

### What happened upstream
The atlas-morning-briefing pipeline at `~/atlas-morning-briefing` ran at
06:00 cron and delivered a briefing with **no real LLM content**:

- `status.json` reported `"intelligence_enabled": false` despite
  `opencode.enabled: true` in upstream `config.yaml`.
- The briefing markdown `Atlas-Briefing-2026.07.18.md` was only 65
  lines (vs the typical 140–300), and its Executive Summary said:
  > *"Synthesis unavailable for today's briefing. Please see the
  > individual sections below for key updates in tech, defense, and
  > research."*
- Stock driver column was blank, blog summaries were absent, news
  section was just flattened raw headlines with no ranking.

Root causes (both fixed upstream same day, see
`~/atlas-morning-briefing/AI_LOG.md`):

1. **Cron PATH mismatch.** `run_briefing.sh` exported a PATH that did
   not include `/home/linuxbrew/.linuxbrew/bin`, so the `opencode`
   binary was not found and `OpencodeClient.available == False`. The
   entire LLM layer was silently skipped, replaced by deterministic
   fallbacks.
2. **No model fallback.** Even after PATH was fixed, the free-tier
   DeepSeek primary (`opencode/deepseek-v4-flash-free`) hung
   indefinitely on every call. Without a backup model, this would
   have re-degraded the briefing. Upstream added a per-tier fallback
   chain (`opencode-go/glm-5.2` first).

### How this project should respond

**1. Detect degraded briefings at parse time.**

`src/ingestion/parser.py` currently extracts `executive_summary` and
exposes it via `BriefingData.executive_summary`. The empty-briefing
signature is one of:

- `BriefingData.executive_summary` starts with the literal
  `"Synthesis unavailable for today's briefing"` — this is the
  deterministic fallback string in atlas-morning-briefing's
  `generate_markdown_briefing()`, and it is a 100% reliable
  signal that the LLM layer was skipped.
- `BriefingData.executive_summary == ""` — sections missing entirely.
- `len(BriefingData.blog_items) == 0` while
  `len(BriefingData.news_items) > 0` — blog summaries require an LLM
  pass; their absence with news present is a strong degradation signal.

**Recommended action:** add a `briefing_quality` field to
`BriefingData` (enum: `full`, `degraded`, `failed`) populated at parse
time, then have downstream strategies read it:

```python
class BriefingQuality(str, Enum):
    FULL = "full"
    DEGRADED = "degraded"   # LLM-skipped fallback markdown
    FAILED = "failed"       # missing or unparsable
```

**2. Stop trusting LLM-derived sentiment when the briefing is
degraded.**

`BriefingData.macro_sentiment` (in `src/models/briefing.py:58`)
counts bullish/bearish words in the executive summary. On a degraded
briefing the summary is the deterministic fallback string, which has
no sentiment words — so `macro_sentiment` returns `0.0`. That
"neutral" reading is semantically wrong: it means "we don't know,"
not "market is neutral." Downstream strategies (especially
`StrategyC` / event-driven) must distinguish these cases.

**Recommended action:** when `briefing_quality != FULL`, treat
`macro_sentiment` as missing rather than zero. Strategy C should
down-weight or abstain when briefing quality is degraded, not emit
a `Direction.FLAT` recommendation.

**3. Watch the upstream `status.json` — don't parse the briefing
alone.**

`~/atlas-morning-briefing/status.json` carries the upstream ground
truth for whether the briefing's AI layer was active. Fields of
interest:

- `intelligence_enabled: bool` — `False` means the entire LLM
  layer was skipped.
- `papers_found`, `blogs_found`, `news_found`, `stocks_fetched` —
  raw feed counts, useful as availability envelope even when the LLM
  is off.
- `errors: list[str]` — non-fatal upstream errors (scanner failures,
  etc.) appended here.

**Recommended action:** add a small loader that reads
`~/atlas-morning-briefing/status.json` alongside the briefing
markdown and folds `intelligence_enabled` into
`BriefingData.briefing_quality`. One file read per run; no new
dependency.

**4. Assume the briefing markdown grammar is stable but the
content depth varies.**

Section headers (`## Executive Summary`, `## Financial Market
Overview`, `## AI & Tech News`, `## Blog Updates`, etc.) are emitted
by `briefing_runner.generate_markdown_briefing()` whether or not the
LLM ran, so the existing regex parsers in `src/ingestion/parser.py`
keep working across degraded runs. What changes is content *depth*:
on a degraded run, tickers have empty `Driver` columns, news items
lack `relevance_score` and ranked ordering, blog summaries are raw
feed snippets rather than LLM-distilled takeaways. Downstream
strategies that read those fields must tolerate shallower data.

---

### Concrete parser change (sketched)

```python
# src/ingestion/parser.py

DEGRADED_SUMMARY_PREFIX = "Synthesis unavailable for today's briefing"

def _classify_quality(briefing: BriefingData) -> BriefingQuality:
    if not briefing.executive_summary and not briefing.news_items:
        return BriefingQuality.FAILED
    if briefing.executive_summary.startswith(DEGRADED_SUMMARY_PREFIX):
        return BriefingQuality.DEGRADED
    return BriefingQuality.FULL
```

And `BriefingData.macro_sentiment` should return `Optional[float]`,
with `None` standing in for "unknown" when quality is degraded.

---