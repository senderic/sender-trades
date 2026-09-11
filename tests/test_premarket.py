from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from src.config import PremarketConfig
from src.ingestion.premarket import (
    _quote_midpoint,
    _session_bounds,
    fetch_premarket_day,
    fetch_premarket_quote,
    fetch_premarket_range,
    prior_trading_days,
)
from src.timezone import ET_TZ

# Default bar timestamp: 3 minutes before the standard 09:28 ET cutoff,
# i.e. clearly "fresh" by default. Tests that care about staleness pass
# an explicit `timestamp=`.
_DEFAULT_TS = datetime(2026, 9, 11, 9, 25, tzinfo=ET_TZ)


def _bar(
    close: float,
    volume: float,
    vwap: float | None = None,
    open_: float | None = None,
    timestamp: datetime | None = None,
) -> dict:
    return {
        "open": open_ if open_ is not None else close,
        "high": close,
        "low": close,
        "close": close,
        "volume": volume,
        "vwap": vwap if vwap is not None else close,
        "trade_count": 1,
        "timestamp": timestamp if timestamp is not None else _DEFAULT_TS,
    }


class TestSessionBounds:
    def test_bounds_are_et_on_the_session_date(self) -> None:
        start, end = _session_bounds(date(2026, 9, 11), "04:00", "09:28")
        assert start.tzinfo is not None
        assert (start.hour, start.minute) == (4, 0)
        assert (end.hour, end.minute) == (9, 28)
        assert start.date() == date(2026, 9, 11)
        assert end.date() == date(2026, 9, 11)


class TestPriorTradingDays:
    def test_returns_n_weekdays_before(self) -> None:
        # 2026-09-11 is a Friday.
        days = prior_trading_days(date(2026, 9, 11), 5)
        assert len(days) == 5
        assert all(d.weekday() < 5 for d in days)
        assert date(2026, 9, 11) not in days
        # Most recent first.
        assert days[0] == date(2026, 9, 10)

    def test_skips_weekends(self) -> None:
        # 2026-09-14 is a Monday; the immediate prior weekday is Friday 9/11.
        days = prior_trading_days(date(2026, 9, 14), 3)
        assert days[0] == date(2026, 9, 11)
        assert date(2026, 9, 12) not in days  # Saturday
        assert date(2026, 9, 13) not in days  # Sunday


class TestFetchPremarketDay:
    @pytest.mark.asyncio
    async def test_calls_client_with_session_bounds(self) -> None:
        client = AsyncMock()
        client.get_minute_bars.return_value = [_bar(700.0, 100.0)]
        bars = await fetch_premarket_day(client, "QQQ", date(2026, 9, 11), "04:00", "09:28")
        assert bars == [_bar(700.0, 100.0)]
        client.get_minute_bars.assert_awaited_once()
        args, _kwargs = client.get_minute_bars.call_args
        assert args[0] == "QQQ"


class TestQuoteMidpoint:
    def test_returns_midpoint_when_both_sides_present_and_spread_sane(self) -> None:
        underlying = {"bid": 715.40, "ask": 715.48}  # ~0.011% spread
        mid = _quote_midpoint(underlying, max_spread_pct=0.1)
        assert mid == pytest.approx((715.40 + 715.48) / 2.0)

    def test_none_when_underlying_is_none(self) -> None:
        assert _quote_midpoint(None, max_spread_pct=0.1) is None

    def test_none_when_bid_missing(self) -> None:
        assert _quote_midpoint({"bid": None, "ask": 715.48}, max_spread_pct=0.1) is None

    def test_none_when_ask_missing(self) -> None:
        assert _quote_midpoint({"bid": 715.40, "ask": None}, max_spread_pct=0.1) is None

    def test_none_when_side_non_positive(self) -> None:
        assert _quote_midpoint({"bid": 0.0, "ask": 715.48}, max_spread_pct=0.1) is None

    def test_none_when_spread_too_wide(self) -> None:
        # (716.0 - 715.0) / 715.5 * 100 ~= 0.14%, over a 0.1% cap.
        assert _quote_midpoint({"bid": 715.0, "ask": 716.0}, max_spread_pct=0.1) is None

    def test_allows_spread_exactly_at_cap(self) -> None:
        underlying = {"bid": 99.95, "ask": 100.05}  # exactly 0.1% spread
        mid = _quote_midpoint(underlying, max_spread_pct=0.1)
        assert mid == pytest.approx(100.0)


class TestFetchPremarketRange:
    @pytest.mark.asyncio
    async def test_splits_bulk_response_by_day_and_window(self) -> None:
        client = AsyncMock()
        # A bulk response spanning two days: pre-market bars for each,
        # PLUS a regular-hours bar that must be filtered out locally.
        client.get_minute_bars.return_value = [
            _bar(700.0, 100.0, timestamp=datetime(2026, 9, 9, 9, 0, tzinfo=ET_TZ)),
            _bar(
                701.0, 200.0, timestamp=datetime(2026, 9, 9, 11, 0, tzinfo=ET_TZ)
            ),  # RTH, excluded
            _bar(702.0, 300.0, timestamp=datetime(2026, 9, 10, 9, 20, tzinfo=ET_TZ)),
        ]
        by_day = await fetch_premarket_range(
            client, "QQQ", [date(2026, 9, 9), date(2026, 9, 10)], "04:00", "09:28"
        )
        assert [b["volume"] for b in by_day[date(2026, 9, 9)]] == [100.0]
        assert [b["volume"] for b in by_day[date(2026, 9, 10)]] == [300.0]
        # Exactly one network call regardless of how many days were requested.
        client.get_minute_bars.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_bulk_request_spans_earliest_to_latest_day(self) -> None:
        client = AsyncMock()
        client.get_minute_bars.return_value = []
        await fetch_premarket_range(
            client, "QQQ", [date(2026, 9, 10), date(2026, 9, 8), date(2026, 9, 9)], "04:00", "09:28"
        )
        args, _ = client.get_minute_bars.call_args
        start, end = args[1], args[2]
        assert start.date() == date(2026, 9, 8)
        assert end.date() == date(2026, 9, 10)

    @pytest.mark.asyncio
    async def test_empty_days_list_makes_no_call(self) -> None:
        client = AsyncMock()
        by_day = await fetch_premarket_range(client, "QQQ", [], "04:00", "09:28")
        assert by_day == {}
        client.get_minute_bars.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_bar_outside_requested_days_is_dropped(self) -> None:
        client = AsyncMock()
        client.get_minute_bars.return_value = [
            _bar(700.0, 100.0, timestamp=datetime(2026, 9, 9, 9, 0, tzinfo=ET_TZ)),
        ]
        by_day = await fetch_premarket_range(client, "QQQ", [date(2026, 9, 10)], "04:00", "09:28")
        assert by_day == {date(2026, 9, 10): []}


class TestFetchPremarketQuotePriceTiers:
    """The three-tier MECHANICS price resolution (2026-09-11 follow-up
    review): live quote midpoint -> recent bar close -> prior close.
    """

    @pytest.mark.asyncio
    async def test_disabled_returns_unavailable(self) -> None:
        config = PremarketConfig(enabled=False)
        result = await fetch_premarket_quote(
            client=object(),
            symbol="QQQ",
            session_date=date(2026, 9, 11),
            prior_session_close=708.69,
            config=config,
        )
        assert result.available is False
        assert result.source == "disabled"
        assert result.price is None

    @pytest.mark.asyncio
    async def test_tier1_quote_midpoint_used_even_with_zero_bars(self) -> None:
        """The exact zero-volume-morning case: no bars at all, but a live
        bid/ask quote exists (a market maker quotes without anyone
        trading) -- mechanics must use it, not fall back to prior close.
        """

        async def day_fetcher(client, symbol, day, start, cutoff):
            return []

        async def quote_fetcher(client, symbol):
            return {"bid": 715.40, "ask": 715.48}

        config = PremarketConfig(max_quote_spread_pct=0.1)
        result = await fetch_premarket_quote(
            client=object(),
            symbol="QQQ",
            session_date=date(2026, 9, 11),
            prior_session_close=708.69,
            config=config,
            day_fetcher=day_fetcher,
            range_fetcher=None,
            quote_fetcher=quote_fetcher,
        )
        assert result.price == pytest.approx((715.40 + 715.48) / 2.0)
        assert result.price_source == "quote_midpoint"
        assert result.available is True

    @pytest.mark.asyncio
    async def test_tier1_preferred_over_a_fresh_bar(self) -> None:
        async def day_fetcher(client, symbol, day, start, cutoff):
            return [_bar(700.0, 500.0)]  # fresh

        async def quote_fetcher(client, symbol):
            return {"bid": 715.40, "ask": 715.48}

        config = PremarketConfig()
        result = await fetch_premarket_quote(
            client=object(),
            symbol="QQQ",
            session_date=date(2026, 9, 11),
            prior_session_close=708.69,
            config=config,
            day_fetcher=day_fetcher,
            range_fetcher=None,
            quote_fetcher=quote_fetcher,
        )
        assert result.price_source == "quote_midpoint"

    @pytest.mark.asyncio
    async def test_tier2_fresh_bar_used_when_quote_unavailable(self) -> None:
        async def day_fetcher(client, symbol, day, start, cutoff):
            return [_bar(715.44, 500.0, open_=712.0)]  # fresh (default timestamp)

        async def quote_fetcher(client, symbol):
            return None

        config = PremarketConfig()
        result = await fetch_premarket_quote(
            client=object(),
            symbol="QQQ",
            session_date=date(2026, 9, 11),
            prior_session_close=708.69,
            config=config,
            day_fetcher=day_fetcher,
            range_fetcher=None,
            quote_fetcher=quote_fetcher,
        )
        assert result.price == 715.44
        assert result.price_source == "bar_close"
        assert result.available is True
        assert result.first_price == 712.0
        expected_gap = (715.44 - 708.69) / 708.69 * 100
        assert result.gap_pct == pytest.approx(round(expected_gap, 4))

    @pytest.mark.asyncio
    async def test_tier2_skipped_when_bar_is_stale(self) -> None:
        """A bar more than max_bar_age_min before the cutoff can't be
        used for mechanics -- falls through to prior close instead."""
        stale_ts = datetime(2026, 9, 11, 8, 0, tzinfo=ET_TZ)  # 88 min before 09:28 cutoff

        async def day_fetcher(client, symbol, day, start, cutoff):
            return [_bar(715.44, 500.0, timestamp=stale_ts)]

        async def quote_fetcher(client, symbol):
            return None

        config = PremarketConfig(max_bar_age_min=15.0)
        result = await fetch_premarket_quote(
            client=object(),
            symbol="QQQ",
            session_date=date(2026, 9, 11),
            prior_session_close=708.69,
            config=config,
            day_fetcher=day_fetcher,
            range_fetcher=None,
            quote_fetcher=quote_fetcher,
        )
        assert result.price == 708.69
        assert result.price_source == "prior_close"
        assert result.available is False
        assert result.bar_fresh is False
        assert result.bar_age_min == pytest.approx(88.0)

    @pytest.mark.asyncio
    async def test_tier3_prior_close_when_nothing_live_exists(self) -> None:
        async def day_fetcher(client, symbol, day, start, cutoff):
            return []

        async def quote_fetcher(client, symbol):
            return None

        config = PremarketConfig()
        result = await fetch_premarket_quote(
            client=object(),
            symbol="QQQ",
            session_date=date(2026, 9, 11),
            prior_session_close=708.69,
            config=config,
            day_fetcher=day_fetcher,
            range_fetcher=None,
            quote_fetcher=quote_fetcher,
        )
        assert result.price == 708.69
        assert result.price_source == "prior_close"
        assert result.available is False
        assert result.gap_pct == 0.0

    @pytest.mark.asyncio
    async def test_wide_spread_quote_is_skipped_in_favor_of_bar(self) -> None:
        async def day_fetcher(client, symbol, day, start, cutoff):
            return [_bar(700.0, 500.0)]  # fresh

        async def quote_fetcher(client, symbol):
            return {"bid": 690.0, "ask": 710.0}  # absurdly wide spread

        config = PremarketConfig(max_quote_spread_pct=0.1)
        result = await fetch_premarket_quote(
            client=object(),
            symbol="QQQ",
            session_date=date(2026, 9, 11),
            prior_session_close=690.0,
            config=config,
            day_fetcher=day_fetcher,
            range_fetcher=None,
            quote_fetcher=quote_fetcher,
        )
        assert result.price_source == "bar_close"

    @pytest.mark.asyncio
    async def test_replay_mode_skips_quote_tier_entirely(self) -> None:
        """quote_fetcher=None (the replay harness's case) must never try
        tier 1, even if a quote WOULD have been available."""
        called = {"n": 0}

        async def day_fetcher(client, symbol, day, start, cutoff):
            return [_bar(700.0, 500.0)]

        config = PremarketConfig()
        result = await fetch_premarket_quote(
            client=object(),
            symbol="QQQ",
            session_date=date(2026, 9, 11),
            prior_session_close=690.0,
            config=config,
            day_fetcher=day_fetcher,
            range_fetcher=None,
            quote_fetcher=None,
            source="replay",
        )
        assert called["n"] == 0
        assert result.price_source == "bar_close"
        assert result.source == "replay"

    @pytest.mark.asyncio
    async def test_quote_fetcher_exception_falls_back_gracefully(self) -> None:
        async def day_fetcher(client, symbol, day, start, cutoff):
            return [_bar(700.0, 500.0)]

        async def quote_fetcher(client, symbol):
            raise RuntimeError("network blip")

        config = PremarketConfig()
        result = await fetch_premarket_quote(
            client=object(),
            symbol="QQQ",
            session_date=date(2026, 9, 11),
            prior_session_close=690.0,
            config=config,
            day_fetcher=day_fetcher,
            range_fetcher=None,
            quote_fetcher=quote_fetcher,
        )
        assert result.price_source == "bar_close"

    @pytest.mark.asyncio
    async def test_day_fetcher_exception_still_tries_quote_tier(self) -> None:
        """A failed bars fetch (today) must not prevent trying the live
        quote -- the whole point of tier 1 existing."""

        async def day_fetcher(client, symbol, day, start, cutoff):
            raise RuntimeError("boom")

        async def quote_fetcher(client, symbol):
            return {"bid": 715.40, "ask": 715.48}

        config = PremarketConfig()
        result = await fetch_premarket_quote(
            client=object(),
            symbol="QQQ",
            session_date=date(2026, 9, 11),
            prior_session_close=708.69,
            config=config,
            day_fetcher=day_fetcher,
            range_fetcher=None,
            quote_fetcher=quote_fetcher,
        )
        assert result.price_source == "quote_midpoint"
        assert result.available is True


class TestFetchPremarketQuoteReliability:
    """Reliability = volume_ratio ok AND the last bar is fresh --
    independent of which tier ended up supplying the price.
    """

    @pytest.mark.asyncio
    async def test_reliable_when_volume_meets_threshold_and_bar_fresh(self) -> None:
        async def day_fetcher(client, symbol, day, start, cutoff):
            # Today's volume equals the trailing median exactly.
            return [_bar(700.0, 1000.0)]

        config = PremarketConfig(lookback_days=5, min_volume_ratio=0.5)
        result = await fetch_premarket_quote(
            client=object(),
            symbol="QQQ",
            session_date=date(2026, 9, 11),
            prior_session_close=690.0,
            config=config,
            day_fetcher=day_fetcher,
            range_fetcher=None,
            quote_fetcher=None,
        )
        assert result.volume_ratio == pytest.approx(1.0)
        assert result.bar_fresh is True
        assert result.reliable is True
        assert result.lookback_days_used == 5

    @pytest.mark.asyncio
    async def test_unreliable_when_volume_thin(self) -> None:
        async def day_fetcher(client, symbol, day, start, cutoff):
            if day == date(2026, 9, 11):
                return [_bar(700.0, 100.0)]  # thin today
            return [_bar(700.0, 1000.0)]  # trailing days much higher

        config = PremarketConfig(lookback_days=5, min_volume_ratio=0.5)
        result = await fetch_premarket_quote(
            client=object(),
            symbol="QQQ",
            session_date=date(2026, 9, 11),
            prior_session_close=690.0,
            config=config,
            day_fetcher=day_fetcher,
            range_fetcher=None,
            quote_fetcher=None,
        )
        assert result.volume_ratio == pytest.approx(0.1)
        assert result.reliable is False

    @pytest.mark.asyncio
    async def test_unreliable_when_bar_stale_even_with_ample_volume(self) -> None:
        """A stale last trade forces reliable=False regardless of
        volume_ratio -- staleness and thinness are independent gates."""
        stale_ts = datetime(2026, 9, 11, 8, 0, tzinfo=ET_TZ)

        async def day_fetcher(client, symbol, day, start, cutoff):
            if day == date(2026, 9, 11):
                return [_bar(700.0, 5000.0, timestamp=stale_ts)]  # huge volume, but stale
            return [_bar(700.0, 1000.0)]

        config = PremarketConfig(lookback_days=5, min_volume_ratio=0.5, max_bar_age_min=15.0)
        result = await fetch_premarket_quote(
            client=object(),
            symbol="QQQ",
            session_date=date(2026, 9, 11),
            prior_session_close=690.0,
            config=config,
            day_fetcher=day_fetcher,
            range_fetcher=None,
            quote_fetcher=None,
        )
        assert result.volume_ratio == pytest.approx(5.0)  # volume itself looks great
        assert result.bar_fresh is False
        assert result.reliable is False

    @pytest.mark.asyncio
    async def test_no_bars_today_is_never_reliable(self) -> None:
        async def day_fetcher(client, symbol, day, start, cutoff):
            if day == date(2026, 9, 11):
                return []
            return [_bar(700.0, 1000.0)]

        async def quote_fetcher(client, symbol):
            return {"bid": 690.0, "ask": 690.5}  # tier 1 supplies a price, but no trades today

        config = PremarketConfig(lookback_days=3)
        result = await fetch_premarket_quote(
            client=object(),
            symbol="QQQ",
            session_date=date(2026, 9, 11),
            prior_session_close=690.0,
            config=config,
            day_fetcher=day_fetcher,
            range_fetcher=None,
            quote_fetcher=quote_fetcher,
        )
        assert result.price_source == "quote_midpoint"
        assert result.bar_fresh is False
        assert result.reliable is False

    @pytest.mark.asyncio
    async def test_lookback_day_failure_is_dropped_not_zero(self) -> None:
        async def day_fetcher(client, symbol, day, start, cutoff):
            if day == date(2026, 9, 11):
                return [_bar(700.0, 500.0)]
            if day == date(2026, 9, 10):
                raise RuntimeError("transient")
            return [_bar(700.0, 500.0)]

        config = PremarketConfig(lookback_days=3)
        result = await fetch_premarket_quote(
            client=object(),
            symbol="QQQ",
            session_date=date(2026, 9, 11),
            prior_session_close=690.0,
            config=config,
            day_fetcher=day_fetcher,
            range_fetcher=None,
            quote_fetcher=None,
        )
        # Only 2 of the 3 lookback days contributed (one raised).
        assert result.lookback_days_used == 2

    @pytest.mark.asyncio
    async def test_no_median_when_no_lookback_data(self) -> None:
        async def day_fetcher(client, symbol, day, start, cutoff):
            if day == date(2026, 9, 11):
                return [_bar(700.0, 500.0)]
            return []

        config = PremarketConfig(lookback_days=3)
        result = await fetch_premarket_quote(
            client=object(),
            symbol="QQQ",
            session_date=date(2026, 9, 11),
            prior_session_close=690.0,
            config=config,
            day_fetcher=day_fetcher,
            range_fetcher=None,
            quote_fetcher=None,
        )
        assert result.median_volume is None
        assert result.volume_ratio is None
        assert result.reliable is False

    @pytest.mark.asyncio
    async def test_range_fetcher_used_by_default_for_lookback(self) -> None:
        """The default wiring uses ONE bulk range_fetcher call instead of
        one get_minute_bars call per lookback day."""
        client = AsyncMock()
        lookback = prior_trading_days(date(2026, 9, 11), 5)

        async def side_effect(symbol, start, end):
            # Today's own single-day fetch (fetch_premarket_day) and the
            # bulk lookback fetch (fetch_premarket_range) both go through
            # get_minute_bars; distinguish by the requested span. One bar
            # per lookback day so all 5 days contribute to the median.
            if (end - start) > timedelta(days=1):
                return [
                    _bar(
                        700.0, 500.0, timestamp=datetime(d.year, d.month, d.day, 5, 0, tzinfo=ET_TZ)
                    )
                    for d in lookback
                ]
            return [_bar(700.0, 500.0)]

        client.get_minute_bars.side_effect = side_effect

        async def quote_fetcher(client, symbol):
            return None

        config = PremarketConfig(lookback_days=5)
        result = await fetch_premarket_quote(
            client=client,
            symbol="QQQ",
            session_date=date(2026, 9, 11),
            prior_session_close=690.0,
            config=config,
            quote_fetcher=quote_fetcher,
        )
        # 1 call for today's bars + 1 bulk call for the whole lookback
        # window, never lookback_days separate calls.
        assert client.get_minute_bars.await_count == 2
        assert result.lookback_days_used == 5

    @pytest.mark.asyncio
    async def test_source_label_passthrough(self) -> None:
        async def day_fetcher(client, symbol, day, start, cutoff):
            return [_bar(700.0, 500.0)]

        config = PremarketConfig(lookback_days=1)
        result = await fetch_premarket_quote(
            client=object(),
            symbol="QQQ",
            session_date=date(2026, 9, 11),
            prior_session_close=690.0,
            config=config,
            day_fetcher=day_fetcher,
            range_fetcher=None,
            quote_fetcher=None,
            source="replay",
        )
        assert result.source == "replay"
