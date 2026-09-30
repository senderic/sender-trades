"""Replay historical trading days against multiple candidate LLMs.

Answers "which LLM makes better predictions" by rebuilding each day's
inputs exactly as the pipeline would have seen them pre-market (no
look-ahead), asking each candidate model — pinned individually, with NO
fallback to another model — for a single monolithic prediction covering
all target assets, and scoring the result strictly against what actually
happened that session (open/high/low/close).

This script only READS the existing pipeline modules (``src/pipeline.py``,
``src/llm/trade_signal.py``, ``src/llm/graph.py``, ``src/ingestion/*``,
``src/prediction_tracker.py``) to reconstruct identical inputs and reuse
the exact same prompt-building / response-parsing code the live pipeline
uses. It never modifies those modules, never places orders, never sends
email, and never writes to ``logs/prediction-history.json`` or
``LESSONS_LEARNED.md``. All of its own outputs live under
``logs/replay/``.

Usage::

    uv run python scripts/replay_models.py \\
        --start 2026-08-26 --end 2026-09-09 \\
        --models opencode/muse-spark-1.3-contributor-free,nvidia-direct/nvidia/nemotron-3-ultra-550b-a55b,openrouter/deepseek/deepseek-v4-pro \\
        --concurrency 4

Caching: raw responses are cached at ``logs/replay/<model-slug>/<date>.json``.
A rerun with the same date range and models is free — cached entries
(successes AND failures) are reused unless ``--force`` or
``--retry-failures`` is passed.

Pre-market mode (``--premarket``): rebuilds each day's prompt with the
RECONSTRUCTED pre-market block (live price/gap/volume-reliability as of
the 09:28 ET cutoff, via ``src.ingestion.premarket`` against real Alpaca
history) and PRIOR SESSION labeling on the stale snapshot quote — see
``src.models.market.Quote.prior_session_close`` and
``src.config.PremarketConfig`` for the underlying fix. Requires
``APCA_API_KEY_ID``/``APCA_API_SECRET_KEY`` in the environment (loaded
from ``.env`` via ``python-dotenv`` if present). Responses cache under
``logs/replay/<model-slug>__premarket/<date>.json`` — a DIFFERENT
directory from the stale-input cache above, so this mode never reads or
overwrites the existing (stale-input) results, and the two can be
compared side by side. Alpaca's own historical-bar responses are cached
separately at ``logs/replay/premarket_cache/<symbol>/<date>.json`` so
reruns cost zero new Alpaca calls, only LLM calls.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

# Running this file directly (``uv run python scripts/replay_models.py``)
# puts scripts/ on sys.path[0], not the repo root, so ``import src.*``
# below would fail with ModuleNotFoundError. Insert the repo root explicitly.
_REPO_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORTS))

import structlog  # noqa: E402

# NOTE: at the time this script was written, other in-progress edits to
# src/config.py and src/execution/engine.py (concurrent work — see
# AGENTS.md-adjacent task notes) leave a transient circular import
# (src.config -> src.execution -> src.execution.engine -> src.config)
# that breaks depending on which module is imported first. Importing
# src.execution.engine before src.config sidesteps it without touching
# either file. Safe to remove once that edit lands cleanly.
import src.execution.engine as _execution_engine_import_order_workaround  # noqa: F401,E402
from src.config import (  # noqa: E402
    GapFadeConfig,
    GraphConfig,
    LLMConfig,
    PremarketConfig,
    Settings,
)
from src.execution.client import AlpacaBrokerClient  # noqa: E402
from src.ingestion.candle_providers import CandleProviderChain, build_candle_chain  # noqa: E402
from src.ingestion.parser import read_briefing  # noqa: E402
from src.ingestion.premarket import fetch_premarket_day, fetch_premarket_quote  # noqa: E402
from src.ingestion.snapshot_loader import SnapshotLoader  # noqa: E402
from src.llm.client import OpencodeLLMClient  # noqa: E402
from src.llm.trade_signal import (  # noqa: E402
    SYSTEM_PROMPT,
    _build_prompt,
    _normalise_sources,
    _parse_pick,
)
from src.models.briefing import BriefingData  # noqa: E402
from src.models.market import MarketSnapshot  # noqa: E402
from src.prediction_tracker import format_history_for_prompt, load_history  # noqa: E402
from src.trade_tracker import (  # noqa: E402
    TradeOutcome,
    format_outcomes_for_prompt,
    load_trade_outcomes,
)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - python-dotenv is a project dependency
    pass

logger = structlog.get_logger()

REPO_ROOT = Path(__file__).resolve().parents[1]
ATLAS_DIR = Path("~/atlas-morning-briefing").expanduser()
LOG_DIR = REPO_ROOT / "logs"
REPLAY_DIR = REPO_ROOT / "logs" / "replay"

TARGET_ASSETS = ["SPY", "QQQ"]

DEFAULT_MODELS = [
    "opencode/muse-spark-1.3-contributor-free",
    "nvidia-direct/nvidia/nemotron-3-ultra-550b-a55b",
    "openrouter/deepseek/deepseek-v4-pro",
]

# Documented fallbacks used ONLY when logs/*/trade-*.json has no fills to
# derive real numbers from (see estimate_option_pricing_from_fills). When
# fills ARE available, actual production runs price options
# out-of-the-money (src.engine.options_strategy.compute_otm_strike targets
# ~30-delta, roughly 0.6% OTM from the pre-market spot), never at-the-money
# — an earlier version of this script priced the payoff proxy as if every
# prediction bought an ATM option, which is not what production actually
# trades and made the payoff metrics read far too optimistic (every row,
# including the always-UP baseline, showed a positive mean return).
#
# Joining the 22 resolved fills in logs/*/trade-*.json against the day's
# open price in logs/prediction-history.json (entry_strike vs open_price,
# entry_price vs open_price) gives: median OTM distance ~0.47% of the
# underlying, median premium ~0.098% of the underlying (mean ~0.146%) —
# see estimate_option_pricing_from_fills for the exact computation. The
# fallbacks below are close approximations for when that join yields
# nothing (e.g. a fresh checkout with no trade history yet).
DEFAULT_OTM_PCT = 0.37
DEFAULT_PREMIUM_PCT = 0.10
# Documented assumption for the sensitivity table's ATM (0% OTM) column:
# no real ATM 0DTE fills exist in this account's history (production
# always trades OTM), so this is how ATM SPY/QQQ 0DTE options typically
# price as a % of spot in the last hour before/at the open, not a
# measurement from this system's own data.
ATM_PREMIUM_ASSUMPTION_PCT = 0.30
SENSITIVITY_OTM_GRID = (0.0, 0.2, 0.37, 0.6)

CALL_TIMEOUT_SEC = 150
# opencode CLI errors that mean "try again shortly", not "this model is
# broken" — retried with backoff before being recorded as a failure.
_RATE_LIMIT_MARKERS = ("rate limit", "429", "too many requests", "rate_limit")
_RETRY_BACKOFF_SEC = (5, 20, 60)
# Local opencode CLI/session errors (its own sqlite session store), not a
# model or network problem — seen when another process (e.g. a
# concurrently-running pipeline or the exit-advisor's own opencode calls)
# hits the same local opencode session store at once. Retried up to 2
# times (a subset of the attempts in _RETRY_BACKOFF_SEC above), then
# recorded as a failure like anything else.
_LOCAL_OPENCODE_ERROR_MARKERS = ("failed query: insert into", "session not found")
_LOCAL_OPENCODE_MAX_RETRIES = 2

CONFIDENCE_BUCKETS: list[tuple[float, float]] = [
    (0.0, 0.5),
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 1.01),
]

Status = Literal["predict", "abstain", "fail"]


# ─────────────────────────── input reconstruction ───────────────────────────


def load_briefing_for_day(day: date) -> BriefingData:
    """Load the briefing exactly as it existed for ``day``, no look-ahead.

    Mirrors ``Pipeline._phase_ingest_briefing`` minus the LLM
    re-synthesis step (deliberately skipped here: re-synthesis is itself
    a model-dependent LLM call, and every candidate must see byte-identical
    inputs). Falls back to an empty ``BriefingData`` — exactly like
    ``Pipeline._phase_analyze`` does when no briefing is found — rather
    than skipping the day, since the live pipeline would still run.
    """
    fname = f"Atlas-Briefing-{day.year}.{day.month:02d}.{day.day:02d}.md"
    path = ATLAS_DIR / "briefings" / fname
    if not path.is_file():
        return BriefingData(briefing_date=day)
    return read_briefing(path)


def load_market_for_day(day: date) -> MarketSnapshot | None:
    """Load the atlas-morning-briefing snapshot captured for ``day``.

    Uses only the pre-fetched snapshot (never the live-API fallback in
    ``Pipeline._phase_ingest_market``, which would hit today's real
    market data, not ``day``'s). Returns ``None`` when no snapshot
    directory exists for ``day`` — the caller skips such days entirely
    since there is no way to see what the pipeline would have seen.
    """
    loader = SnapshotLoader(ATLAS_DIR)
    loader.snapshot_dir = ATLAS_DIR / "snapshots" / day.isoformat()
    loader.today = day
    if not loader.is_available():
        return None
    return loader.load()


def history_before(day: date, log_dir: Path = LOG_DIR) -> list[dict]:
    """Prediction history strictly BEFORE ``day`` (no look-ahead)."""
    return [h for h in load_history(str(log_dir)) if h.get("date", "9999") < day.isoformat()]


def trade_outcomes_before(day: date, log_dir: Path = LOG_DIR) -> list[TradeOutcome]:
    """Resolved trade outcomes strictly BEFORE ``day`` (no look-ahead)."""
    return [o for o in load_trade_outcomes(str(log_dir)) if o.date < day.isoformat()]


def build_prompt_for_day(day: date, briefing: BriefingData, market: MarketSnapshot) -> str:
    """Build the exact monolithic prompt ``LLMTradeStrategy._evaluate_monolithic``
    would have built for ``day``, scoping prediction/trade history to what was
    actually known at the time.
    """
    history_str = format_history_for_prompt(history_before(day))
    trade_outcomes_str = format_outcomes_for_prompt(trade_outcomes_before(day))
    return _build_prompt(
        briefing,
        market,
        TARGET_ASSETS,
        history_str,
        trade_outcomes_str,
        gap_fade=GapFadeConfig(),
    )


def _alpaca_client_for_replay() -> AlpacaBrokerClient | None:
    """Build an AlpacaBrokerClient from env creds, or None if unset.

    ``--premarket`` needs a live/historical Alpaca data connection;
    without keys it fails loudly at startup (see ``main_async``) rather
    than silently falling back to the stale snapshot data this mode
    exists to fix.
    """
    api_key = os.environ.get("APCA_API_KEY_ID", "")
    api_secret = os.environ.get("APCA_API_SECRET_KEY", "")
    if not api_key or not api_secret:
        return None
    return AlpacaBrokerClient(api_key, api_secret, paper=True)


def _premarket_bars_cache_path(symbol: str, day: date) -> Path:
    return REPLAY_DIR / "premarket_cache" / symbol / f"{day.isoformat()}.json"


async def _cached_premarket_day(
    client: AlpacaBrokerClient,
    symbol: str,
    day: date,
    session_start_et: str,
    cutoff_et: str,
) -> list[dict[str, Any]]:
    """On-disk-cached day-fetcher for ``fetch_premarket_quote``.

    Overlapping trailing-median windows across many replayed days would
    otherwise re-fetch the same (symbol, day) bars from Alpaca on every
    script run; this caches each (symbol, day) exactly once regardless
    of whether it was fetched as a day's own pre-market data or as
    another day's lookback comparison. A day with zero bars (holiday, or
    a real bar-less session) is cached too, as an empty list, so it is
    not re-queried forever; a genuine fetch FAILURE (exception) is not
    cached and will retry on the next run.
    """
    path = _premarket_bars_cache_path(symbol, day)
    if path.exists():
        try:
            raw = json.loads(path.read_text())
            for bar in raw:
                if bar.get("timestamp"):
                    bar["timestamp"] = datetime.fromisoformat(bar["timestamp"])
            return raw
        except (json.JSONDecodeError, OSError, ValueError):
            pass  # corrupt cache entry — fall through and re-fetch

    bars = await fetch_premarket_day(client, symbol, day, session_start_et, cutoff_et)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [
        {**bar, "timestamp": bar["timestamp"].isoformat() if bar.get("timestamp") else None}
        for bar in bars
    ]
    path.write_text(json.dumps(serializable))
    return bars


async def enrich_market_with_premarket(
    day: date,
    market: MarketSnapshot,
    client: AlpacaBrokerClient,
    config: PremarketConfig,
) -> MarketSnapshot:
    """Attach live pre-market price/volume/reliability to ``market`` in place.

    Mirrors ``Pipeline._enrich_premarket``, except ``day`` is a
    historical replay date instead of "today", and two things the live
    pipeline does are deliberately NOT reproduced here (2026-09-11
    follow-up review):

    - ``quote_fetcher=None`` -- there is no cheap way to reconstruct a
      past bid/ask NBBO quote for a historical date, so replay always
      falls back straight to the bar-close / prior-close tiers (see
      ``fetch_premarket_quote``'s MECHANICS price resolution). This is a
      known, documented gap between replay and live, not an oversight.
    - ``range_fetcher=None`` -- keeps the existing per-day
      :func:`_cached_premarket_day` on-disk cache reuse across
      overlapping lookback windows (which already makes reruns free)
      instead of switching to the single-ranged-request optimization
      added for the live pipeline's latency. Re-deriving that caching
      against a bulk fetch is out of scope here: this replay's bar-based
      inputs are unchanged by the latency fix.
    """
    for asset in TARGET_ASSETS:
        quote = market.quotes.get(asset)
        prior_close = quote.prior_session_close if quote is not None else 0.0
        pm = await fetch_premarket_quote(
            client,
            asset,
            day,
            prior_close,
            config,
            day_fetcher=_cached_premarket_day,
            range_fetcher=None,
            quote_fetcher=None,
            source="replay",
        )
        market.premarket[asset] = pm
    return market


def enumerate_trading_days(start: date, end: date) -> list[date]:
    """Weekdays in ``[start, end]`` that have an atlas snapshot on disk."""
    days: list[date] = []
    d = start
    while d <= end:
        if d.weekday() < 5 and (ATLAS_DIR / "snapshots" / d.isoformat()).is_dir():
            days.append(d)
        d += timedelta(days=1)
    return days


# ─────────────────────────── actual outcomes ───────────────────────────


@dataclass
class DailyOutcome:
    """What actually happened to ``asset`` on a given day."""

    asset: str
    open: float
    high: float
    low: float
    close: float
    source: str  # "prediction-history" | "candle-chain"


def _prediction_history_outcomes(log_dir: Path = LOG_DIR) -> dict[tuple[str, str], DailyOutcome]:
    out: dict[tuple[str, str], DailyOutcome] = {}
    for h in load_history(str(log_dir)):
        d, a = h.get("date"), h.get("asset")
        if not d or not a:
            continue
        o, hi, lo, c = (
            h.get("open_price"),
            h.get("high_price"),
            h.get("low_price"),
            h.get("close_price"),
        )
        if None in (o, hi, lo, c):
            continue
        out[(d, a)] = DailyOutcome(
            asset=a,
            open=float(o),
            high=float(hi),
            low=float(lo),
            close=float(c),
            source="prediction-history",
        )
    return out


class OutcomeStore:
    """Resolves actual daily OHLC, preferring ``prediction-history.json``
    and falling back to the repo's own candle-provider chain.
    """

    def __init__(self, candle_chain: CandleProviderChain | None = None) -> None:
        self._history = _prediction_history_outcomes()
        self._chain = candle_chain or build_candle_chain()
        self._cache: dict[tuple[str, str], DailyOutcome | None] = {}

    async def get(self, day: date, asset: str) -> DailyOutcome | None:
        key = (day.isoformat(), asset)
        if key in self._history:
            return self._history[key]
        if key in self._cache:
            return self._cache[key]
        candle = await self._chain.fetch_daily_candle(asset, day)
        result: DailyOutcome | None = None
        if candle is not None:
            try:
                result = DailyOutcome(
                    asset=asset,
                    open=float(candle["o"][0]),
                    high=float(candle["h"][0]),
                    low=float(candle["l"][0]),
                    close=float(candle["c"][0]),
                    source="candle-chain",
                )
            except (KeyError, IndexError, TypeError, ValueError):
                result = None
        self._cache[key] = result
        return result

    def previous_trading_day(self, day: date) -> date:
        prev = day - timedelta(days=1)
        while prev.weekday() >= 5:
            prev -= timedelta(days=1)
        return prev


# ─────────────────────────── model invocation ───────────────────────────


def model_slug(model: str) -> str:
    return model.replace("/", "__")


def cache_path(model: str, day: date, cache_suffix: str = "") -> Path:
    """Cache file for one (model, day). ``cache_suffix`` (e.g.
    ``"__premarket"``) puts pre-market-mode results in a separate
    directory from the stale-input cache so reruns of one mode never
    read or clobber the other's results.
    """
    return REPLAY_DIR / (model_slug(model) + cache_suffix) / f"{day.isoformat()}.json"


def _is_rate_limited(error: str) -> bool:
    low = error.lower()
    return any(marker in low for marker in _RATE_LIMIT_MARKERS)


def _is_local_opencode_error(error: str) -> bool:
    low = error.lower()
    return any(marker in low for marker in _LOCAL_OPENCODE_ERROR_MARKERS)


def _write_cache(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Strip the transient, run-local bookkeeping flags before persisting —
    # they describe THIS run's relationship to the record (fresh vs. cache
    # hit), not the record itself, and would otherwise go stale on disk.
    to_write = {k: v for k, v in record.items() if not k.startswith("_")}
    path.write_text(json.dumps(to_write, indent=2))


@dataclass
class CallBudget:
    """Cross-task guardrail for a retry/rescore run: caps total fresh
    (non-cached) calls and aborts the whole run the moment any call fails
    with a billing/credits error — used when explicitly retrying a small,
    known set of failures rather than doing a fresh full sweep, so a
    surprise billing wall can't silently burn through many calls.
    """

    max_new_calls: int | None = None
    new_calls_made: int = 0
    aborted_reason: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def try_claim(self) -> bool:
        """Reserve one fresh-call slot. Returns False if the budget is
        exhausted or the run has already been aborted."""
        async with self.lock:
            if self.aborted_reason is not None:
                return False
            if self.max_new_calls is not None and self.new_calls_made >= self.max_new_calls:
                return False
            self.new_calls_made += 1
            return True

    async def maybe_abort_on(self, error: str) -> None:
        """Flag the run as aborted if ``error`` looks like a billing/credits
        failure. Idempotent — keeps the first reason seen."""
        if _is_credit_exhausted_error(error):
            async with self.lock:
                if self.aborted_reason is None:
                    self.aborted_reason = error


async def call_model_monolithic(
    model: str,
    day: date,
    prompt: str,
    force: bool = False,
    retry_failures: bool = False,
    offline: bool = False,
    budget: CallBudget | None = None,
    cache_suffix: str = "",
) -> dict[str, Any]:
    """Invoke ``model`` — pinned, with NO fallback chain — for one day.

    Cached at ``logs/replay/<model-slug><cache_suffix>/<date>.json``. A
    failed call is recorded with ``ok: False`` and the error text; it is
    never silently retried against a different model
    (``fallback_models=[]`` below is what guarantees that).

    ``offline=True`` makes this function NEVER place a network call: a
    cache miss is returned as a synthetic ``skipped`` record instead of
    being fetched. Used for a pure rescore-from-cache pass that must
    spend zero new calls, even for models with incomplete cache coverage.

    ``budget``, when given, caps total fresh calls across a whole retry
    run (:attr:`CallBudget.max_new_calls`) and aborts the run (skips all
    further fresh calls) the moment any call fails with a billing/credits
    error — see :class:`CallBudget`.

    ``cache_suffix`` (e.g. ``"__premarket"``) namespaces the cache
    directory for ``--premarket`` runs — see :func:`cache_path`.
    """
    path = cache_path(model, day, cache_suffix)
    if path.exists() and not force:
        try:
            cached = json.loads(path.read_text())
            if cached.get("ok") or not retry_failures:
                return {**cached, "_from_cache": True}
        except (json.JSONDecodeError, OSError):
            pass  # corrupt cache entry — fall through and re-call

    if offline:
        return {
            "model": model,
            "date": day.isoformat(),
            "ok": False,
            "skipped": True,
            "error": "offline mode: not cached, no network call made",
            "_from_cache": False,
        }

    if budget is not None and not await budget.try_claim():
        reason = budget.aborted_reason or "max_new_calls budget exhausted"
        return {
            "model": model,
            "date": day.isoformat(),
            "ok": False,
            "skipped": True,
            "error": f"call skipped: {reason}",
            "_from_cache": False,
        }

    client = OpencodeLLMClient(
        LLMConfig(
            enabled=True,
            primary_model=model,
            fallback_models=[],  # pinned: a failure must never be served by another model
            timeout_sec=CALL_TIMEOUT_SEC,
            max_calls_per_run=1,
        )
    )

    last_error = ""
    elapsed = 0.0
    attempt = 0
    for attempt, delay in enumerate([0, *_RETRY_BACKOFF_SEC]):
        if delay:
            logger.info("replay_rate_limit_backoff", model=model, date=str(day), delay=delay)
            await asyncio.sleep(delay)
        t0 = time.monotonic()
        response = await asyncio.to_thread(client.invoke, prompt, SYSTEM_PROMPT)
        elapsed = time.monotonic() - t0
        if response is not None:
            record = {
                "model": model,
                "date": day.isoformat(),
                "ok": True,
                "response": response,
                "elapsed_sec": round(elapsed, 2),
                "attempts": attempt + 1,
            }
            _write_cache(path, record)
            return {**record, "_from_cache": False}
        last_error = client.last_error
        if _is_rate_limited(last_error):
            continue
        if _is_local_opencode_error(last_error) and attempt < _LOCAL_OPENCODE_MAX_RETRIES:
            logger.info(
                "replay_local_opencode_error_retry",
                model=model,
                date=str(day),
                attempt=attempt + 1,
                error=last_error[:200],
            )
            continue
        break

    if budget is not None:
        await budget.maybe_abort_on(last_error)

    record = {
        "model": model,
        "date": day.isoformat(),
        "ok": False,
        "error": last_error,
        "elapsed_sec": round(elapsed, 2),
        "attempts": attempt + 1,
    }
    _write_cache(path, record)
    return {**record, "_from_cache": False}


async def call_model_graph(
    model: str,
    day: date,
    briefing: BriefingData,
    market: MarketSnapshot,
    force: bool = False,
    retry_failures: bool = False,
    offline: bool = False,
) -> dict[str, Any]:
    """Graph-path variant of :func:`call_model_monolithic` (``--graph``).

    Runs the full diamond graph (research + predict + checker + pick per
    ``src.llm.graph.GraphOrchestrator``) with the model chain pinned to a
    single candidate (no fallback). Costs several calls per day instead of
    one, so this is opt-in only. ``offline=True`` behaves as in
    :func:`call_model_monolithic`: a cache miss is skipped, never fetched.
    """
    from src.llm.graph import GraphOrchestrator

    path = cache_path(model, day)
    if path.exists() and not force:
        try:
            cached = json.loads(path.read_text())
            if cached.get("ok") or not retry_failures:
                return {**cached, "_from_cache": True}
        except (json.JSONDecodeError, OSError):
            pass

    if offline:
        return {
            "model": model,
            "date": day.isoformat(),
            "ok": False,
            "graph": True,
            "skipped": True,
            "error": "offline mode: not cached, no network call made",
            "_from_cache": False,
        }

    settings = Settings(
        llm=LLMConfig(
            enabled=True,
            primary_model=model,
            fallback_models=[],
            timeout_sec=CALL_TIMEOUT_SEC,
            max_calls_per_run=20,
        ),
        graph=GraphConfig(enabled=True, fallback_to_monolithic=False, total_deadline_sec=600),
    )
    client = OpencodeLLMClient(settings.llm)

    t0 = time.monotonic()
    try:
        result = await GraphOrchestrator(settings, client).run(
            briefing=briefing, market=market, deterministic_results=[]
        )
    except Exception as e:  # pragma: no cover - defensive, mirrors LLMTradeStrategy
        result = {"trace": {"graph_failed": True, "graph_fail_reason": f"{type(e).__name__}: {e}"}}
    elapsed = time.monotonic() - t0

    trace = result.get("trace", {})
    ok = not trace.get("graph_failed")
    if ok:
        record = {
            "model": model,
            "date": day.isoformat(),
            "ok": True,
            "graph": True,
            "predictions_raw": result.get("predictions", {}),
            "market_vibe": result.get("market_vibe", ""),
            "elapsed_sec": round(elapsed, 2),
        }
    else:
        record = {
            "model": model,
            "date": day.isoformat(),
            "ok": False,
            "graph": True,
            "error": trace.get("graph_fail_reason", "graph failed"),
            "elapsed_sec": round(elapsed, 2),
        }
    _write_cache(path, record)
    return {**record, "_from_cache": False}


# ─────────────────────────── response parsing ───────────────────────────


def normalize_prediction(pred_raw: Any) -> dict[str, Any] | None:
    """Normalize one asset's raw prediction dict the same way
    ``LLMTradeStrategy._evaluate_monolithic`` does.

    Returns ``None`` when the entry is missing or structurally invalid
    (not a dict, or an unrecognized direction) — that is an abstain, not
    a parse failure.
    """
    if not isinstance(pred_raw, dict):
        return None
    direction = pred_raw.get("direction")
    if direction not in ("UP", "DOWN"):
        return None
    try:
        confidence = float(pred_raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    try:
        move_pct = float(pred_raw.get("predicted_move_pct", 0.0))
    except (TypeError, ValueError):
        move_pct = 0.0
    return {
        "direction": direction,
        "confidence": round(confidence, 4),
        "predicted_move_pct": round(move_pct, 2),
        "rationale": str(pred_raw.get("rationale", "")),
        "sources": _normalise_sources(pred_raw.get("sources", [])),
    }


def parse_monolithic_predictions(response: str) -> dict[str, dict[str, Any]]:
    """Extract per-asset normalized predictions from a monolithic response.

    Returns a dict keyed by asset; assets that could not be extracted are
    simply absent (== abstain for that asset).
    """
    parsed = _parse_pick(response)
    if parsed is None:
        return {}
    predictions_raw = parsed.get("predictions")
    if not isinstance(predictions_raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for asset in TARGET_ASSETS:
        norm = normalize_prediction(predictions_raw.get(asset))
        if norm is not None:
            out[asset] = norm
    return out


def parse_graph_predictions(predictions_raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Same normalization as :func:`parse_monolithic_predictions` for the
    already-dict ``predictions`` field the graph path returns.
    """
    out: dict[str, dict[str, Any]] = {}
    for asset in TARGET_ASSETS:
        norm = normalize_prediction(predictions_raw.get(asset))
        if norm is not None:
            out[asset] = norm
    return out


# ─────────────────────────── scoring (unit-tested) ───────────────────────────


def score_direction(direction: str, open_: float, close: float) -> bool:
    """Open->close direction correctness — same rule as
    ``prediction_tracker.check_outcome``'s ``open_close_correct``.
    """
    return (close > open_) if direction == "UP" else (close < open_)


def score_target_hit(
    direction: str, predicted_move_pct: float, open_: float, high: float, low: float
) -> bool:
    """Whether the predicted direction reached ``|predicted_move_pct|`` from
    the open intraday, using the session's high/low. Mirrors
    ``prediction_tracker.check_outcome``: below a 0.1% magnitude the
    threshold degrades to "did it move past the open at all".
    """
    if abs(predicted_move_pct) >= 0.1:
        target = open_ * (1 + predicted_move_pct / 100)
        return high >= target if direction == "UP" else low <= target
    # Degenerate (near-zero) magnitude: matches check_outcome's fallback of
    # "did it move past the open at all" — strict inequality, since the
    # open itself is not a "move".
    return high > open_ if direction == "UP" else low < open_


def score_expiry_return(
    direction: str,
    open_: float,
    close: float,
    otm_pct: float = DEFAULT_OTM_PCT,
    premium_pct: float = DEFAULT_PREMIUM_PCT,
) -> float:
    """0DTE expiry return in premium units, pricing the option production
    actually buys: struck ``otm_pct`` away from the open in the predicted
    direction (``strike = open * (1 +/- otm_pct/100)``), costing
    ``premium_pct`` of the open, and settling at the session close.

    Intrinsic value at expiry is ``max(0, signed move % in the predicted
    direction - otm_pct)`` — the underlying must clear the OTM distance
    before the option is worth anything at all — and the return relative
    to the premium paid is ``intrinsic / premium_pct - 1`` (a wrong
    direction, a flat close, or a correct-but-insufficient move all give
    intrinsic 0, i.e. a full -100% loss of premium, matching a 0DTE OTM
    option expiring worthless). Direction-only otherwise, so this scores
    baselines too.

    An earlier version priced this as if every prediction bought an ATM
    option (``otm_pct=0``), which doesn't match what
    ``src.engine.options_strategy.compute_otm_strike`` actually trades and
    made every row — including the always-UP baseline — show a positive
    mean return; see ``estimate_option_pricing_from_fills`` for how the
    defaults are derived from real fills instead.
    """
    if open_ <= 0:
        return -1.0
    signed_move_pct = (
        (close - open_) / open_ * 100 if direction == "UP" else (open_ - close) / open_ * 100
    )
    intrinsic_pct = max(0.0, signed_move_pct - otm_pct)
    return intrinsic_pct / premium_pct - 1.0


def score_tp_touch(
    direction: str,
    open_: float,
    high: float,
    low: float,
    otm_pct: float = DEFAULT_OTM_PCT,
    premium_pct: float = DEFAULT_PREMIUM_PCT,
    tp_multiple: float = 2.0,
) -> bool:
    """Whether the intraday excursion in the predicted direction reached
    the point where the option's intrinsic value is ``tp_multiple``
    premiums (default 2x, roughly mirroring a +100% take-profit order):
    the underlying must move ``otm_pct + tp_multiple * premium_pct`` past
    the open. Direction-only, so this scores baselines too.
    """
    threshold_pct = otm_pct + tp_multiple * premium_pct
    threshold = open_ * (threshold_pct / 100)
    if direction == "UP":
        return (high - open_) >= threshold
    return (open_ - low) >= threshold


def confidence_bucket(confidence: float) -> str:
    for lo, hi in CONFIDENCE_BUCKETS:
        if lo <= confidence < hi:
            return f"{lo:.1f}-{hi:.1f}"
    return "unknown"


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% CI by default).

    Used instead of the normal approximation because it stays inside
    [0, 1] and remains sane at the small sample sizes this replay
    produces (n often well under 40 per model).
    """
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z**2 / n
    centre = phat + z**2 / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return (max(0.0, lo), min(1.0, hi))


def _load_history_opens(log_dir: Path) -> dict[tuple[str, str], float]:
    """(date, asset) -> session open, from prediction-history.json."""
    opens: dict[tuple[str, str], float] = {}
    for h in load_history(str(log_dir)):
        d, a, o = h.get("date"), h.get("asset"), h.get("open_price")
        if d and a and o:
            opens[(d, a)] = float(o)
    return opens


def estimate_option_pricing_from_fills(log_dir: Path = LOG_DIR) -> dict[str, Any]:
    """Empirical OTM distance and premium, both as a % of the day's OPEN,
    from real fills in ``logs/*/trade-*.json`` joined against
    ``logs/prediction-history.json``'s ``open_price`` for that (date,
    asset) — the same moneyness production actually trades
    (``src.engine.options_strategy.compute_otm_strike`` targets ~30-delta,
    roughly 0.6% OTM from the pre-market spot), not the ATM assumption an
    earlier version of this script used.

    Returns ``{"otm_pct": median, "premium_pct": median, "premium_pct_mean":
    mean, "n": n}``. Falls back to ``{"otm_pct": DEFAULT_OTM_PCT,
    "premium_pct": DEFAULT_PREMIUM_PCT, "n": 0}`` when no fills join
    successfully (e.g. a fresh checkout with no trade history yet). Used
    only to derive the scoring defaults / report the sensitivity table —
    never silently recomputed mid-run.
    """
    opens = _load_history_opens(log_dir)
    otm_pcts: list[float] = []
    premium_pcts: list[float] = []
    for path in glob.glob(str(log_dir / "*" / "trade-*.json")):
        if path.endswith(".bak"):
            continue
        try:
            data = json.loads(Path(path).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        strike = data.get("entry_strike")
        asset = data.get("asset")
        entry_price = None
        for entry in data.get("entries", []):
            if entry.get("event_type") == "entry_filled":
                entry_price = entry.get("avg_price")
                break
        if not (strike and asset and entry_price):
            continue
        date_str = Path(path).parent.name[:10]
        open_price = opens.get((date_str, asset))
        if not open_price or open_price <= 0:
            continue
        # Signed: positive = OTM, negative = ITM (abs() would count the
        # ITM fills on 2026-08-18 / 09-01 as ~0.66% OTM and skew the median).
        # A fill without a CALL/PUT direction can't be signed, so skip it.
        direction = data.get("direction")
        if direction not in ("CALL", "PUT"):
            continue
        distance = strike - open_price if direction == "CALL" else open_price - strike
        otm_pcts.append(distance / open_price * 100)
        premium_pcts.append(entry_price / open_price * 100)

    if not otm_pcts:
        return {
            "otm_pct": DEFAULT_OTM_PCT,
            "premium_pct": DEFAULT_PREMIUM_PCT,
            "premium_pct_mean": DEFAULT_PREMIUM_PCT,
            "n": 0,
        }
    return {
        "otm_pct": round(statistics.median(otm_pcts), 4),
        "premium_pct": round(statistics.median(premium_pcts), 4),
        "premium_pct_mean": round(statistics.mean(premium_pcts), 4),
        "n": len(otm_pcts),
    }


def premium_for_otm(
    otm_pct: float,
    atm_premium_pct: float = ATM_PREMIUM_ASSUMPTION_PCT,
    anchor_otm_pct: float = DEFAULT_OTM_PCT,
    anchor_premium_pct: float = DEFAULT_PREMIUM_PCT,
) -> float:
    """Premium (% of the open) for a hypothetical strike ``otm_pct`` away,
    for the sensitivity table.

    Linearly interpolates/extrapolates between two anchor points: ATM
    (``otm_pct=0`` -> ``atm_premium_pct``, a documented assumption — no
    real ATM 0DTE fills exist in this account's history since production
    always trades OTM) and the real, fills-derived
    (``anchor_otm_pct``, ``anchor_premium_pct``) point. This is a rough
    illustrative slope, not a volatility-surface fit — there are only 22
    real fills, clustered near ``anchor_otm_pct``, so anything more
    elaborate would be overfitting noise. Floored at a small epsilon so a
    large ``otm_pct`` can never divide-by-zero downstream.
    """
    if anchor_otm_pct <= 0:
        return max(atm_premium_pct, 1e-6)
    slope = (anchor_premium_pct - atm_premium_pct) / anchor_otm_pct
    premium = atm_premium_pct + slope * otm_pct
    return max(premium, 1e-6)


# ─────────────────────────── replay records + aggregation ───────────────────────────


@dataclass
class ReplayRecord:
    """One (day, asset, model) scored data point."""

    date: str
    asset: str
    model: str
    status: Status
    direction: str | None = None
    confidence: float | None = None
    predicted_move_pct: float | None = None
    direction_correct: bool | None = None
    target_hit: bool | None = None
    expiry_return: float | None = None
    tp_touch: bool | None = None
    error: str = ""
    # Raw session prices, stashed on "predict" records so the sensitivity
    # table (different OTM%/premium% assumptions) can be recomputed
    # straight from cached records without re-fetching outcomes.
    open_price: float | None = None
    close_price: float | None = None
    high_price: float | None = None
    low_price: float | None = None


def aggregate(records: list[ReplayRecord]) -> dict[str, dict[str, Any]]:
    """Roll per-(day, asset, model) records up into one leaderboard row
    per model/baseline label.
    """
    by_model: dict[str, list[ReplayRecord]] = {}
    for r in records:
        by_model.setdefault(r.model, []).append(r)

    rows: dict[str, dict[str, Any]] = {}
    for model, recs in by_model.items():
        n = len(recs)
        fails = sum(1 for r in recs if r.status == "fail")
        abstains = sum(1 for r in recs if r.status == "abstain")
        predicted = [r for r in recs if r.status == "predict"]

        dir_correct = sum(1 for r in predicted if r.direction_correct)
        target_scored = [r for r in predicted if r.target_hit is not None]
        target_correct = sum(1 for r in target_scored if r.target_hit)
        return_scored = [r for r in predicted if r.expiry_return is not None]
        expiry_wins = sum(1 for r in return_scored if r.expiry_return > 0)
        tp_scored = [r for r in predicted if r.tp_touch is not None]
        tp_touches = sum(1 for r in tp_scored if r.tp_touch)

        dir_lo, dir_hi = wilson_ci(dir_correct, len(predicted))

        calibration: dict[str, dict[str, Any]] = {}
        for r in predicted:
            if r.confidence is None:
                continue
            bucket = confidence_bucket(r.confidence)
            slot = calibration.setdefault(bucket, {"n": 0, "correct": 0})
            slot["n"] += 1
            if r.direction_correct:
                slot["correct"] += 1
        for slot in calibration.values():
            slot["accuracy"] = round(slot["correct"] / slot["n"], 3) if slot["n"] else None

        rows[model] = {
            "n": n,
            "predicted_n": len(predicted),
            "fail_rate": round(fails / n, 3) if n else 0.0,
            "abstain_rate": round(abstains / n, 3) if n else 0.0,
            "direction_accuracy": round(dir_correct / len(predicted), 3) if predicted else None,
            "direction_accuracy_ci95": (round(dir_lo, 3), round(dir_hi, 3)) if predicted else None,
            "target_hit_accuracy": round(target_correct / len(target_scored), 3)
            if target_scored
            else None,
            "mean_expiry_return": round(statistics.mean(r.expiry_return for r in return_scored), 3)
            if return_scored
            else None,
            "expiry_win_rate": round(expiry_wins / len(return_scored), 3)
            if return_scored
            else None,
            "tp_touch_rate": round(tp_touches / len(tp_scored), 3) if tp_scored else None,
            "calibration": calibration,
        }
    return rows


def compute_sensitivity_table(
    records: list[ReplayRecord],
    otm_grid: tuple[float, ...] = SENSITIVITY_OTM_GRID,
    atm_premium_pct: float = ATM_PREMIUM_ASSUMPTION_PCT,
    anchor_otm_pct: float = DEFAULT_OTM_PCT,
    anchor_premium_pct: float = DEFAULT_PREMIUM_PCT,
) -> tuple[dict[str, dict[float, float | None]], dict[float, float]]:
    """Mean expiry return per model/baseline at each OTM grid point, with
    the premium at each point coming from :func:`premium_for_otm`.

    Recomputed directly from each "predict" record's stored
    open/close price (see :class:`ReplayRecord`), so it costs zero new
    calls or outcome refetches — pure re-scoring of already-cached data.

    Returns ``(table, premiums)`` where ``table`` is
    ``{model: {otm: mean_return}}`` and ``premiums`` is ``{otm:
    premium_pct}`` (so the report can show what premium each column used).
    """
    premiums = {
        otm: premium_for_otm(otm, atm_premium_pct, anchor_otm_pct, anchor_premium_pct)
        for otm in otm_grid
    }

    by_model: dict[str, list[ReplayRecord]] = {}
    for r in records:
        if (
            r.status != "predict"
            or r.open_price is None
            or r.close_price is None
            or r.direction is None
        ):
            continue
        by_model.setdefault(r.model, []).append(r)

    table: dict[str, dict[float, float | None]] = {}
    for model, recs in by_model.items():
        row: dict[float, float | None] = {}
        for otm in otm_grid:
            premium = premiums[otm]
            returns = [
                score_expiry_return(r.direction, r.open_price, r.close_price, otm, premium)
                for r in recs
            ]
            row[otm] = round(statistics.mean(returns), 3) if returns else None
        table[model] = row
    return table, premiums


# ─────────────────────────── report writers ───────────────────────────


_CREDIT_EXHAUSTED_MARKERS = (
    "can only afford",
    "add more credits",
    "insufficient credit",
    "insufficient balance",
    "creditserror",
)


def _is_credit_exhausted_error(error: str) -> bool:
    low = error.lower()
    return any(marker in low for marker in _CREDIT_EXHAUSTED_MARKERS)


def find_excluded_models(records: list[ReplayRecord]) -> dict[str, str]:
    """Models hit by a provider credit/quota wall at some point during the
    run — not a real capability signal, so they get dropped from the
    scored leaderboard rather than ranked on a partial, non-random sample
    (whatever days happened to run before the account ran dry) or shown
    with a misleading 100% fail rate.

    Triggers on ANY credit-exhausted failure for a model, even if it also
    has some earlier successes: once an account runs out of credits, later
    dates are systematically missing, which biases (not just shrinks) the
    sample rather than merely reducing it.
    """
    by_model: dict[str, list[ReplayRecord]] = {}
    for r in records:
        by_model.setdefault(r.model, []).append(r)

    excluded: dict[str, str] = {}
    for model, recs in by_model.items():
        credit_fails = [
            r for r in recs if r.status == "fail" and _is_credit_exhausted_error(r.error)
        ]
        if not credit_fails:
            continue
        ok_n = sum(1 for r in recs if r.status == "predict")
        if ok_n == 0:
            excluded[model] = "excluded: provider out of credits"
        else:
            excluded[model] = f"excluded: provider credits exhausted (n={ok_n} ok)"
    return excluded


def write_csv(
    rows: dict[str, dict[str, Any]], path: Path, excluded: dict[str, str] | None = None
) -> None:
    excluded = excluded or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "n",
                "predicted_n",
                "direction_accuracy",
                "direction_accuracy_ci95_lo",
                "direction_accuracy_ci95_hi",
                "target_hit_accuracy",
                "abstain_rate",
                "fail_rate",
                "mean_expiry_return",
                "expiry_win_rate",
                "tp_touch_rate",
            ]
        )
        for model, r in rows.items():
            if model in excluded:
                continue
            ci = r["direction_accuracy_ci95"] or ("", "")
            writer.writerow(
                [
                    model,
                    r["n"],
                    r["predicted_n"],
                    r["direction_accuracy"],
                    ci[0],
                    ci[1],
                    r["target_hit_accuracy"],
                    r["abstain_rate"],
                    r["fail_rate"],
                    r["mean_expiry_return"],
                    r["expiry_win_rate"],
                    r["tp_touch_rate"],
                ]
            )


def _fmt_pct(v: float | None) -> str:
    return "N/A" if v is None else f"{v * 100:.1f}%"


def _fmt_return(v: float | None) -> str:
    return "N/A" if v is None else f"{v * 100:+.1f}%"


def write_markdown(
    rows: dict[str, dict[str, Any]],
    start: date,
    end: date,
    total_calls: int,
    otm_pct: float,
    premium_pct: float,
    premium_n: int,
    cutoff_notes: dict[str, str],
    graph_mode: bool,
    path: Path,
    excluded: dict[str, str] | None = None,
    sensitivity: dict[str, dict[float, float | None]] | None = None,
    sensitivity_premiums: dict[float, float] | None = None,
) -> None:
    excluded = excluded or {}
    lines: list[str] = []
    lines.append("# LLM Prediction Replay Report")
    lines.append("")
    lines.append(f"Date range: {start.isoformat()} .. {end.isoformat()}")
    lines.append(f"Mode: {'graph' if graph_mode else 'monolithic (one call/day/model)'}")
    lines.append(
        f"Fresh (non-cached) LLM calls made in THIS run: {total_calls} "
        "(cumulative calls across all prior runs are cached under logs/replay/<model>/*.json)"
    )
    lines.append(
        f"0DTE option pricing used for the payoff metrics below: struck {otm_pct:.3f}% "
        f"out-of-the-money in the predicted direction, costing {premium_pct:.3f}% of the "
        f"underlying's open "
        f"({'documented fallback (no fills found)' if premium_n == 0 else f'median of {premium_n} real fills, joined against logs/prediction-history.json open prices'}) "
        "— this is what production actually trades (see "
        "src.engine.options_strategy.compute_otm_strike), not an at-the-money option."
    )
    lines.append("")
    lines.append(
        "## Leaderboard\n\n"
        "| Model | n | Direction Acc. | 95% CI | Target-Hit Acc. | Abstain Rate | Fail Rate | "
        "Mean Expiry Return | Expiry Win Rate | TP-Touch Rate (2x) |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for model, r in rows.items():
        if model in excluded:
            continue
        ci = r["direction_accuracy_ci95"]
        ci_str = f"[{ci[0] * 100:.0f}%, {ci[1] * 100:.0f}%]" if ci else "N/A"
        lines.append(
            f"| {model} | {r['n']} | {_fmt_pct(r['direction_accuracy'])} | {ci_str} | "
            f"{_fmt_pct(r['target_hit_accuracy'])} | {_fmt_pct(r['abstain_rate'])} | "
            f"{_fmt_pct(r['fail_rate'])} | {_fmt_return(r['mean_expiry_return'])} | "
            f"{_fmt_pct(r['expiry_win_rate'])} | {_fmt_pct(r['tp_touch_rate'])} |"
        )
    lines.append("")

    if excluded:
        lines.append("## Excluded\n")
        for model, reason in excluded.items():
            r = rows.get(model, {})
            lines.append(f"- `{model}` ({reason}); n attempted = {r.get('n', '?')}.")
        lines.append("")

    if sensitivity:
        grid = sorted(sensitivity_premiums) if sensitivity_premiums else SENSITIVITY_OTM_GRID
        header_cells = " | ".join(
            f"{otm:.2f}% OTM (P={sensitivity_premiums.get(otm, float('nan')):.3f}%)" for otm in grid
        )
        lines.append(
            "## Sensitivity: mean expiry return by strike moneyness\n\n"
            "Same cached predictions, rescored at different OTM%/premium% "
            "assumptions — shows whether trading closer to the money would "
            "change the picture. Premiums are `premium_for_otm`'s "
            f"interpolation/extrapolation between the ATM assumption "
            f"({ATM_PREMIUM_ASSUMPTION_PCT:.2f}%, documented, not measured) and the "
            f"anchor point actually used for the leaderboard above "
            f"({otm_pct:.2f}% OTM / {premium_pct:.3f}% premium); not a "
            "volatility-surface fit. The 0.6% column extrapolates PAST the "
            "real-fills anchor (further OTM than any actual fill in this "
            "account's history) — a linear extrapolation likely "
            "underestimates real option premium at that moneyness (real "
            "premium decay flattens out, it doesn't go linearly toward "
            "zero), so that column's return is optimistic and should be "
            "read as directional, not a reliable point estimate.\n"
        )
        lines.append(f"| Model | {header_cells} |")
        lines.append("|---|" + "---:|" * len(grid))
        for model, row in sensitivity.items():
            if model in excluded:
                continue
            cells = " | ".join(_fmt_return(row.get(otm)) for otm in grid)
            lines.append(f"| {model} | {cells} |")
        lines.append("")

    lines.append("## Calibration (accuracy by confidence bucket)\n")
    for model, r in rows.items():
        if model in excluded:
            continue
        lines.append(f"### {model}")
        if not r["calibration"]:
            lines.append("No scored predictions.\n")
            continue
        lines.append("| Confidence bucket | n | Accuracy |")
        lines.append("|---|---:|---:|")
        for bucket in sorted(r["calibration"]):
            slot = r["calibration"][bucket]
            lines.append(f"| {bucket} | {slot['n']} | {_fmt_pct(slot['accuracy'])} |")
        lines.append("")

    lines.append("## Caveats\n")
    lines.append(
        "- **Training-data contamination risk**: these models may have been "
        "trained on data covering some or all of the replayed dates, which "
        "would inflate their scores relative to genuine out-of-sample "
        "forecasting. Likely cutoffs (self-reported / vendor-documented, "
        "not independently verified):"
    )
    for model, note in cutoff_notes.items():
        lines.append(f"  - `{model}`: {note}")
    lines.append(
        "- **Sample size**: with n in the tens per model, the 95% Wilson "
        "confidence intervals above are wide. Treat any leaderboard "
        "ranking as suggestive, not conclusive, unless the CIs are "
        "non-overlapping."
    )
    lines.append(
        "- **Mean Expiry Return** prices each prediction as the OTM 0DTE "
        "option production actually buys (strike `otm_pct` away from the "
        "open in the predicted direction, costing `premium_pct` of the "
        "open) settling at the session close: intrinsic value is "
        "`max(0, signed move % in the predicted direction - otm_pct)`, and "
        "the return is `intrinsic / premium_pct - 1` — a wrong direction, a "
        "flat close, or a correct-but-insufficient move are all a full "
        "-100% loss of premium. An earlier version of this metric priced "
        "every prediction as ATM (`otm_pct=0`), which is not what "
        "production trades and made every row — including the always-UP "
        "baseline — show a positive mean return."
    )
    lines.append(
        "- **TP-Touch Rate (2x)** is a separate, purely intraday metric: did "
        "the high/low reach `otm_pct + 2 * premium_pct` past the open in the "
        "predicted direction — the point where the option's intrinsic value "
        "equals 2x its cost (roughly mirroring a +100% take-profit order)? "
        "It ignores path/order and whether the position was actually still "
        "open at that moment."
    )
    lines.append(
        "- **Neither metric is a P&L backtest.** Both price the option only "
        "at expiry / by a static intraday threshold; they ignore that "
        "production actually exits at a +100% take-profit or a -50% "
        "stop-loss (see logs/*/trade-*.json `tp_level`/`sl_level`), and "
        "ignore path-dependence, bid/ask spread, and fill slippage. Treat "
        "them as a directional-edge proxy — is the model's direction call "
        "worth more than the premium it costs, in principle — not as an "
        "estimate of realized trading P&L."
    )
    lines.append(
        "- Baseline rows (`baseline:always-up`, `baseline:yesterday-direction`) "
        "have no per-asset move-magnitude estimate, so their `target_hit_accuracy` "
        "is reported as N/A; the expiry-return and TP-touch metrics need only a "
        "direction, so baselines are scored on those."
    )
    path.write_text("\n".join(lines) + "\n")


# ─────────────────────────── main replay loop ───────────────────────────


MODEL_CUTOFF_NOTES = {
    # Sources (checked 2026-09-10): opencode.ai/data/meta/muse-spark-1-3-contributor,
    # developer.meta.com/ai/models/muse-spark, DataCamp, Winbuzzer (2026-09-04 release).
    "opencode/muse-spark-1.3-contributor-free": (
        "Meta Muse Spark 1.3 (white-labeled via the opencode Zen 'Contributor Free' "
        "tier). Released ~2026-09-04 — AFTER this replay's date range ends "
        "(2026-09-09/10), so it plausibly has training exposure to nearly the "
        "entire replayed window. Meta has NOT publicly disclosed a knowledge "
        "cutoff date for this model. HIGH contamination risk — treat its scores "
        "on July/August 2026 dates with real suspicion."
    ),
    # Source: NVIDIA's own model card (huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16,
    # docs.api.nvidia.com, build.nvidia.com) — pre-training cutoff Sept 2025,
    # post-training cutoff May 2026, released 2026-06-04.
    "nvidia-direct/nvidia/nemotron-3-ultra-550b-a55b": (
        "NVIDIA Nemotron 3 Ultra 550B-A55B — vendor-documented cutoffs: "
        "pre-training data through Sept 2025, post-training through May 2026 "
        "(released 2026-06-04). LOW-to-MODERATE contamination risk for this "
        "replay's range (2026-07-22 onward is after the documented post-training "
        "cutoff), though later fine-tunes/updates to the served endpoint are not "
        "independently verifiable here."
    ),
    # Sources: openrouter.ai/deepseek/deepseek-v4-pro-0813, Unite.AI, Simon Willison
    # (GA release 2026-08-12). No official DeepSeek-disclosed cutoff; third-party
    # trackers disagree (one cites ~April 2026, another June 2025) — unconfirmed.
    "openrouter/deepseek/deepseek-v4-pro": (
        "DeepSeek V4 Pro ('0813' build, GA 2026-08-12). No official cutoff "
        "disclosed by DeepSeek; third-party estimates conflict (~April 2026 vs "
        "June 2025) and should not be trusted. Its GA date falls INSIDE this "
        "replay's range, so contamination risk for August/September 2026 dates "
        "cannot be ruled out. Separately: in this environment this model "
        "consistently failed with an OpenRouter 'insufficient credits' error "
        "(see logs/replay/*/*.json) — its scores here reflect that outage, not "
        "the model's actual forecasting capability."
    ),
}


async def run_replay(
    days: list[date],
    models: list[str],
    concurrency: int,
    graph_mode: bool,
    force: bool,
    retry_failures: bool,
    offline: bool = False,
    budget: CallBudget | None = None,
    otm_pct: float = DEFAULT_OTM_PCT,
    premium_pct: float = DEFAULT_PREMIUM_PCT,
    premarket: bool = False,
    premarket_config: PremarketConfig | None = None,
    alpaca_client: AlpacaBrokerClient | None = None,
) -> tuple[list[ReplayRecord], int, int]:
    """Returns ``(records, total_tasks, fresh_calls)`` — ``total_tasks`` is
    every (day, model) attempted (cache hit or not), ``fresh_calls`` is how
    many of those actually hit the network *this run* (0 when ``offline``).

    ``premarket=True`` rebuilds each day's ``MarketSnapshot`` with the
    reconstructed live pre-market block (see
    :func:`enrich_market_with_premarket`) before building the prompt, and
    routes LLM responses to a separate ``__premarket``-suffixed cache
    directory (see :func:`cache_path`) so this never collides with the
    stale-input cache. Requires ``alpaca_client``.
    """
    outcomes = OutcomeStore()
    sem = asyncio.Semaphore(concurrency)
    records: list[ReplayRecord] = []
    cache_suffix = "__premarket" if premarket else ""

    async def bounded_call(
        model: str, day: date, briefing: BriefingData, market: MarketSnapshot, prompt: str
    ):
        async with sem:
            if graph_mode:
                return await call_model_graph(
                    model,
                    day,
                    briefing,
                    market,
                    force=force,
                    retry_failures=retry_failures,
                    offline=offline,
                )
            return await call_model_monolithic(
                model,
                day,
                prompt,
                force=force,
                retry_failures=retry_failures,
                offline=offline,
                budget=budget,
                cache_suffix=cache_suffix,
            )

    day_inputs: dict[date, tuple[BriefingData, MarketSnapshot]] = {}
    for day in days:
        market = load_market_for_day(day)
        if market is None:
            logger.warning("replay_skip_day_no_market", date=str(day))
            continue
        if premarket:
            assert alpaca_client is not None, "--premarket requires an Alpaca client"
            market = await enrich_market_with_premarket(
                day, market, alpaca_client, premarket_config or PremarketConfig()
            )
        briefing = load_briefing_for_day(day)
        day_inputs[day] = (briefing, market)

    tasks = []
    task_meta = []
    for day, (briefing, market) in day_inputs.items():
        prompt = build_prompt_for_day(day, briefing, market) if not graph_mode else ""
        for model in models:
            tasks.append(bounded_call(model, day, briefing, market, prompt))
            task_meta.append((day, model))

    logger.info("replay_starting", days=len(day_inputs), models=len(models), calls=len(tasks))
    call_records = await asyncio.gather(*tasks)
    # Every task is one (day, model) attempt, whether served fresh or resumed
    # from cache; this is the "total calls" figure reported to the user.
    total_calls = len(call_records)
    fresh_calls = sum(1 for r in call_records if not r.get("_from_cache") and not r.get("skipped"))
    if budget is not None and budget.aborted_reason is not None:
        logger.warning(
            "replay_budget_aborted",
            reason=budget.aborted_reason,
            new_calls_made=budget.new_calls_made,
        )

    for (day, model), record in zip(task_meta, call_records, strict=True):
        briefing, market = day_inputs[day]
        if record.get("skipped"):
            # offline cache-miss or budget/abort skip — no data for this
            # (day, model), not a scored failure. Silently omitted.
            continue
        if record.get("ok"):
            if record.get("graph"):
                preds = parse_graph_predictions(record.get("predictions_raw", {}))
            else:
                preds = parse_monolithic_predictions(record.get("response", ""))
        else:
            preds = {}

        for asset in TARGET_ASSETS:
            outcome = await outcomes.get(day, asset)
            if outcome is None:
                continue  # can't score without ground truth

            if not record.get("ok"):
                records.append(
                    ReplayRecord(
                        date=day.isoformat(),
                        asset=asset,
                        model=model,
                        status="fail",
                        error=record.get("error", ""),
                    )
                )
                continue

            pred = preds.get(asset)
            if pred is None:
                records.append(
                    ReplayRecord(date=day.isoformat(), asset=asset, model=model, status="abstain")
                )
                continue

            direction = pred["direction"]
            move_pct = pred["predicted_move_pct"]
            dir_correct = score_direction(direction, outcome.open, outcome.close)
            target_hit = score_target_hit(
                direction, move_pct, outcome.open, outcome.high, outcome.low
            )
            expiry_return = score_expiry_return(
                direction, outcome.open, outcome.close, otm_pct, premium_pct
            )
            tp_touch = score_tp_touch(
                direction, outcome.open, outcome.high, outcome.low, otm_pct, premium_pct
            )
            records.append(
                ReplayRecord(
                    date=day.isoformat(),
                    asset=asset,
                    model=model,
                    status="predict",
                    direction=direction,
                    confidence=pred["confidence"],
                    predicted_move_pct=move_pct,
                    open_price=outcome.open,
                    close_price=outcome.close,
                    high_price=outcome.high,
                    low_price=outcome.low,
                    direction_correct=dir_correct,
                    target_hit=target_hit,
                    expiry_return=expiry_return,
                    tp_touch=tp_touch,
                )
            )

    # ── Baselines: always-UP and yesterday's-direction ──
    for day in day_inputs:
        prev_day = outcomes.previous_trading_day(day)
        for asset in TARGET_ASSETS:
            outcome = await outcomes.get(day, asset)
            if outcome is None:
                continue

            # always-UP baseline
            dir_correct = score_direction("UP", outcome.open, outcome.close)
            expiry_return = score_expiry_return(
                "UP", outcome.open, outcome.close, otm_pct, premium_pct
            )
            tp_touch = score_tp_touch(
                "UP", outcome.open, outcome.high, outcome.low, otm_pct, premium_pct
            )
            records.append(
                ReplayRecord(
                    date=day.isoformat(),
                    asset=asset,
                    model="baseline:always-up",
                    status="predict",
                    direction="UP",
                    open_price=outcome.open,
                    close_price=outcome.close,
                    high_price=outcome.high,
                    low_price=outcome.low,
                    direction_correct=dir_correct,
                    target_hit=None,
                    expiry_return=expiry_return,
                    tp_touch=tp_touch,
                )
            )

            # yesterday's-direction baseline
            prev_outcome = await outcomes.get(prev_day, asset)
            if prev_outcome is None:
                records.append(
                    ReplayRecord(
                        date=day.isoformat(),
                        asset=asset,
                        model="baseline:yesterday-direction",
                        status="abstain",
                    )
                )
                continue
            yest_direction = "UP" if prev_outcome.close > prev_outcome.open else "DOWN"
            dir_correct = score_direction(yest_direction, outcome.open, outcome.close)
            expiry_return = score_expiry_return(
                yest_direction, outcome.open, outcome.close, otm_pct, premium_pct
            )
            tp_touch = score_tp_touch(
                yest_direction, outcome.open, outcome.high, outcome.low, otm_pct, premium_pct
            )
            records.append(
                ReplayRecord(
                    date=day.isoformat(),
                    asset=asset,
                    model="baseline:yesterday-direction",
                    status="predict",
                    direction=yest_direction,
                    open_price=outcome.open,
                    close_price=outcome.close,
                    high_price=outcome.high,
                    low_price=outcome.low,
                    direction_correct=dir_correct,
                    target_hit=None,
                    expiry_return=expiry_return,
                    tp_touch=tp_touch,
                )
            )

    return records, total_calls, fresh_calls


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument(
        "--models", default=",".join(DEFAULT_MODELS), help="Comma-separated opencode model IDs"
    )
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument(
        "--graph",
        action="store_true",
        help="Use the graph orchestrator instead of one monolithic call/day",
    )
    p.add_argument(
        "--force", action="store_true", help="Ignore cache entirely and re-call every model/day"
    )
    p.add_argument(
        "--retry-failures",
        action="store_true",
        help="Reuse cached successes but re-attempt cached failures",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Where to write leaderboard.csv/report.md. Defaults to logs/replay/ "
        "(or logs/replay/premarket/ when --premarket is set, so the two modes' "
        "aggregate reports never clobber each other).",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Never place a network call — rescore purely from whatever is already cached",
    )
    p.add_argument(
        "--premarket",
        action="store_true",
        help="Rebuild each day's prompt with the reconstructed live pre-market block "
        "(price/gap/volume-reliability as of the cutoff, via real Alpaca history) and "
        "prior-session labeling, instead of the stale snapshot quote treated as today. "
        "Requires APCA_API_KEY_ID/APCA_API_SECRET_KEY. Caches LLM responses under a "
        "separate '<model>__premarket' directory — never touches the stale-input cache.",
    )
    p.add_argument(
        "--max-new-calls",
        type=int,
        default=None,
        help="Cap total fresh (non-cached) calls this run; also aborts immediately on a billing/credits error",
    )
    p.add_argument(
        "--otm-pct",
        type=float,
        default=None,
        help="Override the OTM strike distance (%% of open) used for the payoff metrics; "
        "default is the median from real fills, falling back to a documented 0.37%%",
    )
    p.add_argument(
        "--premium-pct",
        type=float,
        default=None,
        help="Override the option premium (%% of open) used for the payoff metrics; "
        "default is the median from real fills, falling back to a documented 0.10%%",
    )
    return p.parse_args(argv)


async def main_async(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    elif args.premarket:
        out_dir = REPLAY_DIR / "premarket"
    else:
        out_dir = REPLAY_DIR

    alpaca_client: AlpacaBrokerClient | None = None
    premarket_config = PremarketConfig()
    if args.premarket:
        alpaca_client = _alpaca_client_for_replay()
        if alpaca_client is None:
            print(  # noqa: T201
                "ERROR: --premarket requires APCA_API_KEY_ID/APCA_API_SECRET_KEY "
                "in the environment (or a .env file) — refusing to silently fall "
                "back to stale-input data."
            )
            sys.exit(1)

    pricing = estimate_option_pricing_from_fills()
    otm_pct = args.otm_pct if args.otm_pct is not None else pricing["otm_pct"]
    premium_pct = args.premium_pct if args.premium_pct is not None else pricing["premium_pct"]

    days = enumerate_trading_days(start, end)
    print(  # noqa: T201
        f"Replaying {len(days)} trading days x {len(models)} models "
        f"(graph={args.graph}, premarket={args.premarket}, offline={args.offline}, "
        f"max_new_calls={args.max_new_calls}, otm_pct={otm_pct:.3f}, "
        f"premium_pct={premium_pct:.3f}, priced from {pricing['n']} real fills)"
    )

    budget = (
        CallBudget(max_new_calls=args.max_new_calls) if args.max_new_calls is not None else None
    )

    t0 = time.monotonic()
    records, total_calls, fresh_calls = await run_replay(
        days,
        models,
        args.concurrency,
        args.graph,
        args.force,
        args.retry_failures,
        offline=args.offline,
        budget=budget,
        otm_pct=otm_pct,
        premium_pct=premium_pct,
        premarket=args.premarket,
        premarket_config=premarket_config,
        alpaca_client=alpaca_client,
    )
    elapsed = time.monotonic() - t0

    if budget is not None and budget.aborted_reason is not None:
        print(f"ABORTED after a billing/credits error: {budget.aborted_reason}")  # noqa: T201

    rows = aggregate(records)
    excluded = find_excluded_models(records)
    sensitivity, sensitivity_premiums = compute_sensitivity_table(
        records, anchor_otm_pct=otm_pct, anchor_premium_pct=premium_pct
    )

    csv_path = out_dir / "leaderboard.csv"
    md_path = out_dir / "report.md"
    write_csv(rows, csv_path, excluded=excluded)
    write_markdown(
        rows,
        start,
        end,
        fresh_calls,
        otm_pct,
        premium_pct,
        pricing["n"],
        {
            m: MODEL_CUTOFF_NOTES.get(m, "Unknown — not in the built-in cutoff notes table.")
            for m in models
        },
        args.graph,
        md_path,
        excluded=excluded,
        sensitivity=sensitivity,
        sensitivity_premiums=sensitivity_premiums,
    )

    print(  # noqa: T201
        f"Done in {elapsed:.1f}s. Total (day,model) tasks: {total_calls}. Fresh calls this run: {fresh_calls}"
    )
    print(f"CSV:      {csv_path}")  # noqa: T201
    print(f"Markdown: {md_path}")  # noqa: T201
    for model, r in rows.items():
        if model in excluded:
            print(f"  {model}: {excluded[model]} (n attempted={r['n']})")  # noqa: T201
            continue
        print(  # noqa: T201
            f"  {model}: n={r['n']} dir_acc={_fmt_pct(r['direction_accuracy'])} "
            f"target_hit={_fmt_pct(r['target_hit_accuracy'])} abstain={_fmt_pct(r['abstain_rate'])} "
            f"fail={_fmt_pct(r['fail_rate'])} mean_ret={_fmt_return(r['mean_expiry_return'])} "
            f"expiry_win={_fmt_pct(r['expiry_win_rate'])} tp_touch={_fmt_pct(r['tp_touch_rate'])}"
        )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
