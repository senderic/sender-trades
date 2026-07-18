# Lessons Learned — Atlas Morning Briefing as a Dependency

Cross-project notes about `~/atlas-morning-briefing`, the daily morning
briefing this project ingests via `src/ingestion/parser.py`. Each entry
documents a real production incident upstream and how this project
should respond.

Newest entries at the top.

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