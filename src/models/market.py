from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class DataSource(StrEnum):
    """Enumeration of supported market data sources."""

    FINNHUB = "finnhub"
    MCP_CHAIN = "mcp_chain"
    BRIEFING = "briefing"
    BRAVE = "brave"
    REDDIT = "reddit"
    UNUSUAL_WHALES = "unusual_whales"


class Quote(BaseModel):
    """A market quote for a given symbol from a specific source.

    Every quote reaching this pipeline is captured PRE-MARKET (cron runs
    9:28 AM ET, 2 min before the open), and every free source it uses
    (Finnhub, and the Yahoo fallback in ``src.ingestion.snapshot_loader``)
    reports NO live pre-market price outside market hours -- they return
    the last completed session's close. See :attr:`prior_session_close`.
    """

    symbol: str
    current_price: float
    open_price: float
    high_price: float
    low_price: float
    previous_close: float
    change_pct: float
    volume: int
    source: DataSource
    timestamp: datetime

    @property
    def prior_session_close(self) -> float:
        """The most recently COMPLETED session's close.

        Verified on the 2026-09-11 incident: the atlas Finnhub snapshot
        taken ~06:02 PT reported QQQ ``current_price=708.69`` /
        ``previous_close=716.31`` -- exactly Thursday 9/10's session
        (Thursday's own close and Wednesday's close), a full two days
        stale relative to Friday 9/11. Outside market hours, Finnhub's
        (and Yahoo's) "current price" is the last trade, which is the
        PRIOR session's close, not today's; ``previous_close`` is the
        session BEFORE that. So the correct anchor for TODAY's gap is
        ``current_price`` here, never ``previous_close``. See
        ``src.ingestion.premarket`` and ``src.llm.trade_signal._gap_pct``
        for where this feeds the actual gap computation.
        """
        return self.current_price


class PremarketQuote(BaseModel):
    """Live pre-market price/volume for one asset, up to a cutoff time.

    Reconstructed from Alpaca 1-minute bars on the IEX feed, plus (live
    mode only) a live bid/ask quote -- see ``src.ingestion.premarket``
    and ``AlpacaBrokerClient.get_minute_bars``/``get_underlying_quote``.
    This account has no SIP/consolidated-tape subscription, so IEX
    reports only a small, noisy slice of true pre-market volume. Absolute
    volume is therefore not meaningful; :attr:`reliable` instead compares
    :attr:`cumulative_volume` to the trailing-day median AT THE SAME
    cutoff time (:attr:`median_volume`), which cancels out the fixed
    IEX/consolidated ratio.

    :attr:`price` is what strike selection and the gap-fade gap use --
    see ``MarketSnapshot.mechanics_price`` -- regardless of
    :attr:`reliable`. It is resolved in ``fetch_premarket_quote`` through
    three tiers, recorded in :attr:`price_source`:

    1. ``"quote_midpoint"`` -- the live bid/ask midpoint. A market-maker
       quote exists even on a morning with zero trades, unlike a bar's
       last-trade close, which goes stale exactly when volume is thin.
       Live mode only (see ``fetch_premarket_quote``'s ``quote_fetcher``).
    2. ``"bar_close"`` -- the last pre-market bar's close, only when
       within :attr:`PremarketConfig.max_bar_age_min` of the cutoff.
    3. ``"prior_close"`` -- ``Quote.prior_session_close``, logged as a
       WARNING. This is the exact stale-data bug being fixed, used only
       when nothing live is available at all.

    :attr:`reliable` (ANALYSIS EVIDENCE weight, never gates MECHANICS)
    additionally requires the last bar to be within
    :attr:`PremarketConfig.max_bar_age_min` of the cutoff, regardless of
    :attr:`volume_ratio` and regardless of which tier supplied
    :attr:`price` -- a live quote with no recent trade tells us WHERE the
    market is, not that real money has confirmed it.
    """

    symbol: str
    available: bool = False
    price: float | None = None
    price_source: Literal["quote_midpoint", "bar_close", "prior_close"] | None = None
    vwap: float | None = None
    first_price: float | None = None
    cumulative_volume: float = 0.0
    last_bar_time: datetime | None = None
    bar_age_min: float | None = None
    # Whether the last bar's age is within PremarketConfig.max_bar_age_min
    # of the cutoff -- exposed separately from `reliable` so callers (the
    # prompt builder) can say WHY a move is untrusted: a stale bar vs
    # merely thin volume are different messages to the model.
    bar_fresh: bool = False
    gap_pct: float | None = None
    median_volume: float | None = None
    volume_ratio: float | None = None
    reliable: bool = False
    lookback_days_used: int = 0
    source: Literal["live", "replay", "disabled", "unavailable"] = "unavailable"


class NewsHeadline(BaseModel):
    """A news headline with sentiment polarity."""

    title: str
    source: str
    url: str
    published_at: datetime | None = None
    snippet: str = ""
    polarity: float = 0.0


class RSSCacheItem(BaseModel):
    """An entry from an RSS feed."""

    title: str
    source: str
    url: str
    published_at: datetime | None = None
    summary: str = ""


class MarketSnapshot(BaseModel):
    """A point-in-time snapshot of market data: quotes, news, and RSS items.

    ``quotes`` is the (pre-market-stale, see ``Quote.prior_session_close``)
    snapshot/live-API data. ``premarket`` is the live pre-market price/
    volume reconstruction, keyed by asset, populated by
    ``Pipeline._enrich_premarket`` -- see ``src.ingestion.premarket``.
    """

    quotes: dict[str, Quote] = Field(default_factory=dict)
    premarket: dict[str, PremarketQuote] = Field(default_factory=dict)
    news: list[NewsHeadline] = Field(default_factory=list)
    rss_items: list[RSSCacheItem] = Field(default_factory=list)
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def avg_sentiment_polarity(self) -> float:
        """Compute the average sentiment polarity across all news headlines.

        Returns:
            Mean polarity, or 0.0 if there are no headlines.
        """
        if not self.news:
            return 0.0
        return round(sum(n.polarity for n in self.news) / len(self.news), 4)

    def mechanics_price(self, asset: str) -> float | None:
        """Live pre-market price for STRIKE/GAP MECHANICS, regardless of volume.

        ``PremarketQuote.price`` (see ``src.ingestion.premarket.
        fetch_premarket_quote``) is already resolved through three tiers
        -- live quote midpoint, then a recent bar close, then the prior
        session's close (with a WARNING logged there) -- REGARDLESS of
        :attr:`PremarketQuote.reliable`, which only governs how the LLM
        prompt is told to weight the move as analysis evidence, never
        whether mechanics gets a live number.

        This method trusts that resolved value whenever it exists. Its
        own fallback to ``Quote.prior_session_close`` (logging a WARNING
        here too) only fires when the pre-market subsystem produced NO
        entry at all for ``asset`` (disabled, no API keys, or an
        unhandled exception in ``Pipeline._enrich_premarket``) -- a
        second, independent safety net. Returns ``None`` when neither is
        available.
        """
        pm = self.premarket.get(asset)
        if pm is not None and pm.price is not None and pm.price > 0:
            return pm.price
        quote = self.quotes.get(asset)
        prior_close = quote.prior_session_close if quote is not None else 0.0
        if prior_close and prior_close > 0:
            logger.warning(
                "premarket_mechanics_fallback_to_prior_close",
                asset=asset,
                prior_session_close=prior_close,
                reason="no_premarket_entry" if pm is None else "premarket_price_unresolved",
            )
            return prior_close
        return None

    def mechanics_gap_pct(self, asset: str) -> float | None:
        """Today's pre-market gap for MECHANICS (gap-fade gate, deterministic
        strategies): live pre-market price vs the prior session's close,
        via :meth:`mechanics_price` (same fallback + WARNING behaviour).

        Returns ``None`` only when the prior session's close is unknown
        (no quote at all for ``asset``) -- when a live pre-market quote
        is unavailable but a prior close IS known, this returns 0.0 (the
        fallback price IS the prior close), the conservative "assume no
        gap" default rather than fabricating a number from stale fields.
        """
        quote = self.quotes.get(asset)
        prior_close = quote.prior_session_close if quote is not None else 0.0
        if not prior_close or prior_close <= 0:
            return None
        price = self.mechanics_price(asset)
        if price is None:
            return None
        return (price - prior_close) / prior_close * 100
