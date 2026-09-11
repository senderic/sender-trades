"""Live pre-market price/volume reconstruction via Alpaca.

Fixes the 2026-09-11 incident: the atlas Finnhub snapshot is captured
~06:02 PT, before Finnhub has any pre-market price for SPY/QQQ, so the
pipeline was fed the PRIOR completed session's OHLC labeled as "today".
See ``src.models.market.Quote.prior_session_close`` for the mechanism
and ``src.config.PremarketConfig`` for the fix's config surface.

This module fetches, for one asset and one session date (works for the
LIVE run and for a historical replay date identically -- the caller
just supplies a different ``session_date``, and Alpaca serves historical
minute bars for any past date without any look-ahead risk since every
fetch window is bounded at ``[session_start, cutoff]`` for that date):

- A MECHANICS price (:attr:`~src.models.market.PremarketQuote.price`),
  resolved through three tiers (see :func:`fetch_premarket_quote`):
  live bid/ask midpoint, then a recent pre-market bar close, then the
  prior session's close. Used for strike selection, the forecast target
  strike, and the gap-fade gap regardless of volume.
- Cumulative pre-market volume up to the cutoff, compared against the
  trailing ``lookback_days`` median at the SAME clock cutoff, to flag
  whether today's move is backed by enough volume (and recent enough
  trading) to trust as ANALYSIS EVIDENCE (``reliable``). This account's
  data feed (IEX, no SIP subscription -- see ``AlpacaBrokerClient.
  get_minute_bars``) reports only a small, noisy slice of true volume,
  so the comparison is relative to this account's own trailing norm,
  never an absolute count.

2026-09-11 follow-up review: a zero- or thin-volume morning has no
recent TRADE, but Alpaca still returns a live bid/ask (a market-maker
quote exists even when nobody has traded), so the live quote is tried
first and a stale bar is no longer silently used as "the" price. The
historical replay has no cheap way to reconstruct a past bid/ask quote,
so it always skips that tier (see ``quote_fetcher=None`` in the replay
harness) and falls back to bar close / prior close only -- a documented
difference from the live path, not an oversight.
"""

from __future__ import annotations

import statistics
from collections.abc import Awaitable, Callable
from datetime import date as date_
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog

from src.config import PremarketConfig
from src.models.market import PremarketQuote
from src.timezone import ET_TZ

if TYPE_CHECKING:
    from src.execution.client import AlpacaBrokerClient

logger = structlog.get_logger()

DayFetcher = Callable[["AlpacaBrokerClient", str, date_, str, str], Awaitable[list[dict[str, Any]]]]
RangeFetcher = Callable[
    ["AlpacaBrokerClient", str, list[date_], str, str],
    Awaitable[dict[date_, list[dict[str, Any]]]],
]
QuoteFetcher = Callable[["AlpacaBrokerClient", str], Awaitable[dict[str, Any] | None]]


def _parse_hm(value: str) -> tuple[int, int]:
    hour_str, minute_str = value.split(":")
    return int(hour_str), int(minute_str)


def _session_bounds(
    session_date: date_, session_start_et: str, cutoff_et: str
) -> tuple[datetime, datetime]:
    """Return the ``[start, end]`` datetimes (ET) for one session's pre-market window."""
    sh, sm = _parse_hm(session_start_et)
    ch, cm = _parse_hm(cutoff_et)
    start = datetime(session_date.year, session_date.month, session_date.day, sh, sm, tzinfo=ET_TZ)
    end = datetime(session_date.year, session_date.month, session_date.day, ch, cm, tzinfo=ET_TZ)
    return start, end


def prior_trading_days(session_date: date_, n: int) -> list[date_]:
    """The ``n`` most recent weekdays strictly before ``session_date``.

    A simple weekday filter (no market-holiday calendar) -- a holiday in
    the lookback window just yields zero bars for that day, which
    ``fetch_premarket_quote`` already tolerates (dropped from the median
    rather than treated as a zero-volume day, see
    :attr:`PremarketQuote.lookback_days_used`).
    """
    days: list[date_] = []
    d = session_date - timedelta(days=1)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return days


async def fetch_premarket_day(
    client: AlpacaBrokerClient,
    symbol: str,
    session_date: date_,
    session_start_et: str,
    cutoff_et: str,
) -> list[dict[str, Any]]:
    """Fetch one session's raw 1-minute bars, ``[session_start_et, cutoff_et]`` ET.

    The default day-fetcher used by :func:`fetch_premarket_quote` for
    TODAY's own bars (always a single day). The replay harness
    (``scripts/replay_models.py``) injects a caching wrapper with the
    same signature instead, since it re-fetches heavily overlapping
    trailing windows across many days.
    """
    start, end = _session_bounds(session_date, session_start_et, cutoff_et)
    return await client.get_minute_bars(symbol, start, end)


async def fetch_premarket_range(
    client: AlpacaBrokerClient,
    symbol: str,
    days: list[date_],
    session_start_et: str,
    cutoff_et: str,
) -> dict[date_, list[dict[str, Any]]]:
    """Fetch ALL of ``days``' pre-market windows in ONE ranged Alpaca request.

    The naive approach -- one ``get_minute_bars`` call per lookback day
    -- costs ``lookback_days`` sequential round trips just to compute the
    trailing-volume median (11 calls/asset at the default 10-day
    lookback), which dominates the live pipeline's pre-market enrichment
    latency. Instead this fetches ``[04:00 ET of the earliest day, 09:28
    ET of the latest day]`` (or whatever ``session_start_et``/
    ``cutoff_et`` are configured to) in a single request -- necessarily
    including full regular-hours bars for every day strictly between
    them, since Alpaca has no way to request only a time-of-day slice
    across multiple dates -- and splits the result back into each day's
    EXACT ``[session_start_et, cutoff_et]`` window locally, so the
    per-day same-clock-cutoff comparison is unchanged.

    Used by :func:`fetch_premarket_quote`'s default ``range_fetcher``
    for the live pipeline. The replay harness passes ``range_fetcher=
    None`` to keep its existing per-day on-disk caching (see
    ``scripts/replay_models.py::_cached_premarket_day``) untouched --
    that caching already makes reruns free, and re-deriving it against a
    bulk fetch is out of scope for this pass (its bar-based inputs are
    unchanged).
    """
    if not days:
        return {}
    earliest, latest = min(days), max(days)
    start, _ = _session_bounds(earliest, session_start_et, cutoff_et)
    _, end = _session_bounds(latest, session_start_et, cutoff_et)
    all_bars = await client.get_minute_bars(symbol, start, end)

    sh, sm = _parse_hm(session_start_et)
    ch, cm = _parse_hm(cutoff_et)
    by_day: dict[date_, list[dict[str, Any]]] = {d: [] for d in days}
    for bar in all_bars:
        ts = bar.get("timestamp")
        if ts is None:
            continue
        ts_et = ts.astimezone(ET_TZ)
        d = ts_et.date()
        if d not in by_day:
            continue
        hm = (ts_et.hour, ts_et.minute)
        if hm < (sh, sm) or hm > (ch, cm):
            continue
        by_day[d].append(bar)
    return by_day


async def fetch_live_underlying_quote(
    client: AlpacaBrokerClient, symbol: str
) -> dict[str, Any] | None:
    """Default ``quote_fetcher``: ``AlpacaBrokerClient.get_underlying_quote``.

    Only meaningful live -- there is no cheap way to reconstruct a past
    bid/ask NBBO quote for a historical replay date, so
    ``scripts/replay_models.py`` passes ``quote_fetcher=None`` instead of
    this, and mechanics falls back to bar close / prior close there. See
    the module docstring.
    """
    return await client.get_underlying_quote(symbol)


def _cumulative_volume(bars: list[dict[str, Any]]) -> float:
    return sum(float(b.get("volume") or 0.0) for b in bars)


def _volume_weighted_price(bars: list[dict[str, Any]]) -> float | None:
    total_vol = _cumulative_volume(bars)
    if total_vol <= 0:
        return None
    num = sum(
        float(b.get("vwap") or b.get("close") or 0.0) * float(b.get("volume") or 0.0) for b in bars
    )
    return num / total_vol


def _quote_midpoint(underlying: dict[str, Any] | None, max_spread_pct: float) -> float | None:
    """Coalesce a live underlying quote dict into a bid/ask midpoint.

    Deliberately narrower than ``ExecutionEngine._extract_spot`` (which
    also accepts the quote's ``last`` field, or a lone ``ask``): ``last``
    is just the last TRADE, which goes stale on a thin morning exactly
    like a pre-market bar's close does, so this tier only trusts the
    bid/ask midpoint -- the one thing a live quote gives us that a bar
    can't (a market-maker quote exists even with zero trades). Returns
    ``None`` when either side is missing/non-positive, or the spread is
    too wide to trust the midpoint as a real price.
    """
    if not underlying:
        return None
    bid = underlying.get("bid")
    ask = underlying.get("ask")
    if not bid or not ask or bid <= 0 or ask <= 0:
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    spread_pct = (ask - bid) / mid * 100
    if spread_pct > max_spread_pct:
        return None
    return mid


def _bar_age_min(last_bar_time: datetime | None, cutoff_dt: datetime) -> float | None:
    if last_bar_time is None:
        return None
    return (cutoff_dt - last_bar_time).total_seconds() / 60.0


async def fetch_premarket_quote(
    client: AlpacaBrokerClient,
    symbol: str,
    session_date: date_,
    prior_session_close: float,
    config: PremarketConfig,
    day_fetcher: DayFetcher = fetch_premarket_day,
    range_fetcher: RangeFetcher | None = fetch_premarket_range,
    quote_fetcher: QuoteFetcher | None = fetch_live_underlying_quote,
    source: str = "live",
) -> PremarketQuote:
    """Fetch today's (or a historical day's) pre-market price/volume/reliability.

    MECHANICS price resolution (:attr:`PremarketQuote.price` /
    :attr:`PremarketQuote.price_source`), in order:

    1. ``quote_fetcher(client, symbol)``'s bid/ask midpoint, when both
       sides are present and the spread is within
       :attr:`PremarketConfig.max_quote_spread_pct` -- see
       :func:`_quote_midpoint`. Skipped entirely when ``quote_fetcher``
       is ``None`` (the replay harness's case).
    2. The last pre-market bar's close, when its age (vs the session's
       cutoff) is within :attr:`PremarketConfig.max_bar_age_min`.
    3. ``prior_session_close``, logged as a WARNING -- the exact stale-
       data bug this module fixes, used only when nothing live exists.

    :attr:`PremarketQuote.available` is ``True`` only for tiers 1-2 (a
    genuinely live read); :attr:`PremarketQuote.reliable` additionally
    requires the volume-ratio threshold AND a fresh last bar, regardless
    of which tier supplied the price -- a live quote with no recent
    trade is not evidence of a confirmed move.

    Args:
        client: An :class:`AlpacaBrokerClient` (used only via
            ``day_fetcher``/``range_fetcher``/``quote_fetcher``).
        symbol: Underlying symbol (e.g. ``QQQ``).
        session_date: The session to reconstruct -- today for the live
            pipeline, or a historical date for the replay harness.
        prior_session_close: The correct anchor for today's gap (see
            ``Quote.prior_session_close``) -- NOT ``Quote.previous_close``.
        config: :class:`PremarketConfig` (cutoff, lookback, thresholds).
        day_fetcher: Fetches ONE day's raw bars (used for ``session_date``
            itself). The replay harness overrides this with an on-disk
            caching wrapper.
        range_fetcher: Fetches the WHOLE lookback window in one request
            (see :func:`fetch_premarket_range`) for the trailing-volume
            median. Pass ``None`` to fall back to one ``day_fetcher`` call
            per lookback day instead -- used by the replay harness to
            keep its existing per-day cache reuse untouched.
        quote_fetcher: Fetches a live bid/ask quote (tier 1). Pass
            ``None`` to skip straight to bar/prior-close tiers -- the
            replay harness's case (see the module docstring).
        source: Recorded on the result as :attr:`PremarketQuote.source`
            (``"live"`` for the pipeline, ``"replay"`` for the harness).

    Returns:
        A :class:`PremarketQuote`. ``price`` is populated whenever ANY
        tier resolves (including the tier-3 fallback); ``available`` is
        ``False`` when only tier 3 (or nothing) resolved -- callers must
        not read ``available=False`` as "no price," only as "not a
        genuinely live one."
    """
    if not config.enabled:
        return PremarketQuote(symbol=symbol, available=False, source="disabled")

    try:
        bars = await day_fetcher(
            client, symbol, session_date, config.session_start_et, config.cutoff_et
        )
    except Exception as e:
        logger.warning(
            "premarket_fetch_error", symbol=symbol, date=session_date.isoformat(), error=str(e)
        )
        bars = []

    volume = _cumulative_volume(bars)
    last_bar = bars[-1] if bars else None
    first_bar = bars[0] if bars else None
    bar_price = float(last_bar["close"]) if last_bar and last_bar.get("close") else None
    vwap = _volume_weighted_price(bars) if bars else None
    if vwap is None:
        vwap = bar_price
    first_price = float(first_bar["open"]) if first_bar and first_bar.get("open") else bar_price

    _, cutoff_dt = _session_bounds(session_date, config.session_start_et, config.cutoff_et)
    last_bar_time = last_bar.get("timestamp") if last_bar else None
    bar_age = _bar_age_min(last_bar_time, cutoff_dt)
    bar_fresh = bar_age is not None and bar_age <= config.max_bar_age_min

    # --- Tier 1: live quote midpoint ---
    quote_mid: float | None = None
    if quote_fetcher is not None:
        try:
            underlying = await quote_fetcher(client, symbol)
        except Exception as e:
            logger.warning("premarket_quote_fetch_error", symbol=symbol, error=str(e))
            underlying = None
        quote_mid = _quote_midpoint(underlying, config.max_quote_spread_pct)

    price: float | None
    price_source: str | None
    available: bool
    if quote_mid is not None:
        price, price_source, available = quote_mid, "quote_midpoint", True
    elif bar_price is not None and bar_fresh:
        price, price_source, available = bar_price, "bar_close", True
    elif prior_session_close and prior_session_close > 0:
        logger.warning(
            "premarket_price_fallback_to_prior_close",
            symbol=symbol,
            date=session_date.isoformat(),
            prior_session_close=prior_session_close,
            bar_age_min=round(bar_age, 2) if bar_age is not None else None,
        )
        price, price_source, available = prior_session_close, "prior_close", False
    else:
        price, price_source, available = None, None, False

    gap_pct = None
    if price is not None and prior_session_close and prior_session_close > 0:
        gap_pct = (price - prior_session_close) / prior_session_close * 100

    # --- Trailing volume median (reliability) ---
    prior_days = prior_trading_days(session_date, config.lookback_days)
    prior_volumes: list[float] = []
    if range_fetcher is not None and prior_days:
        try:
            by_day = await range_fetcher(
                client, symbol, prior_days, config.session_start_et, config.cutoff_et
            )
        except Exception as e:
            logger.warning("premarket_range_fetch_error", symbol=symbol, error=str(e))
            by_day = {}
        for d in prior_days:
            b = by_day.get(d, [])
            if b:
                prior_volumes.append(_cumulative_volume(b))
    else:
        for d in prior_days:
            try:
                prior_bars = await day_fetcher(
                    client, symbol, d, config.session_start_et, config.cutoff_et
                )
            except Exception:
                continue
            if prior_bars:
                prior_volumes.append(_cumulative_volume(prior_bars))

    median_volume = statistics.median(prior_volumes) if prior_volumes else None
    volume_ratio = (volume / median_volume) if median_volume and median_volume > 0 else None
    volume_ok = volume_ratio is not None and volume_ratio >= config.min_volume_ratio
    # A stale (or absent) last bar can never count as reliable evidence,
    # regardless of volume_ratio or which tier supplied `price`.
    reliable = bool(volume_ok and bar_fresh)

    return PremarketQuote(
        symbol=symbol,
        available=available,
        price=price,
        price_source=price_source,
        vwap=round(vwap, 4) if vwap is not None else None,
        first_price=first_price,
        cumulative_volume=volume,
        last_bar_time=last_bar_time,
        bar_age_min=round(bar_age, 2) if bar_age is not None else None,
        gap_pct=round(gap_pct, 4) if gap_pct is not None else None,
        median_volume=round(median_volume, 2) if median_volume is not None else None,
        volume_ratio=round(volume_ratio, 4) if volume_ratio is not None else None,
        bar_fresh=bar_fresh,
        reliable=reliable,
        lookback_days_used=len(prior_volumes),
        source=source,
    )
