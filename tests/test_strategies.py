from datetime import date

import pytest

from src.config import Settings
from src.engine.strategy_a import MomentumStrategy
from src.engine.strategy_b import MeanReversionStrategy
from src.engine.strategy_c import EventDrivenStrategy
from src.models.briefing import BriefingData
from src.models.market import MarketSnapshot, PremarketQuote
from src.models.recommendation import Direction


class TestMomentumStrategy:
    @pytest.mark.asyncio
    async def test_gap_up_generates_call(
        self, sample_briefing_data, sample_market_snapshot
    ) -> None:
        config = Settings()
        config.strategies.momentum.gap_threshold_pct = 0.1
        strategy = MomentumStrategy(config)
        result = await strategy.evaluate(sample_briefing_data, sample_market_snapshot)
        assert result.recommendation is not None
        assert result.recommendation.direction == Direction.CALL

    @pytest.mark.asyncio
    async def test_low_confidence_returns_none(
        self, sample_briefing_data, sample_market_snapshot
    ) -> None:
        config = Settings()
        config.strategies.momentum.min_confidence = 1.0
        strategy = MomentumStrategy(config)
        result = await strategy.evaluate(sample_briefing_data, sample_market_snapshot)
        assert result.recommendation is None

    @pytest.mark.asyncio
    async def test_gap_sourced_from_live_premarket_not_stale_quote(
        self, sample_briefing_data, sample_market_snapshot
    ) -> None:
        """The gap driving momentum's direction/strike must come from
        ``market.premarket`` (today's live price), never from the
        pre-market-stale ``Quote.open_price``/``previous_close`` fields
        — see MarketSnapshot.mechanics_gap_pct."""
        config = Settings()
        config.strategies.momentum.gap_threshold_pct = 0.1
        config.strategies.momentum.min_confidence = 0.1
        # Stale quote fields would compute a *negative* gap; the live
        # pre-market quote says the opposite (a large positive gap), and
        # that must be what drives the recommendation.
        quote = sample_market_snapshot.quotes["SPY"]
        sample_market_snapshot.premarket["SPY"] = PremarketQuote(
            symbol="SPY",
            available=True,
            price=quote.prior_session_close * 1.02,
            gap_pct=2.0,
            reliable=True,
        )
        strategy = MomentumStrategy(config)
        result = await strategy.evaluate(sample_briefing_data, sample_market_snapshot)
        assert result.recommendation is not None
        assert result.debug_trace["SPY_gap_pct"] == pytest.approx(2.0, abs=0.01)
        assert result.recommendation.direction == Direction.CALL
        # Strike computed off the live pre-market price, not the stale quote.
        assert result.recommendation.target_strike != quote.open_price

    @pytest.mark.asyncio
    async def test_no_premarket_data_falls_back_to_zero_gap(
        self, sample_briefing_data, sample_market_snapshot
    ) -> None:
        """No live pre-market quote -> gap defaults to 0% (conservative
        "no gap known"), not a number derived from stale fields."""
        config = Settings()
        strategy = MomentumStrategy(config)
        result = await strategy.evaluate(sample_briefing_data, sample_market_snapshot)
        assert result.debug_trace["SPY_gap_pct"] == 0.0


class TestMeanReversionStrategy:
    @pytest.mark.asyncio
    async def test_evaluates_without_error(
        self, sample_briefing_data, sample_market_snapshot
    ) -> None:
        strategy = MeanReversionStrategy(Settings())
        result = await strategy.evaluate(sample_briefing_data, sample_market_snapshot)
        assert result.label == "mean_reversion"

    @pytest.mark.asyncio
    async def test_debug_trace_contains_expected_keys(
        self, sample_briefing_data, sample_market_snapshot
    ) -> None:
        strategy = MeanReversionStrategy(Settings())
        result = await strategy.evaluate(sample_briefing_data, sample_market_snapshot)
        assert "SPY_move_from_close_pct" in result.debug_trace
        assert "SPY_estimated_rsi" in result.debug_trace


class TestEventDrivenStrategy:
    @pytest.mark.asyncio
    async def test_detects_catalysts_from_briefing(
        self, sample_briefing_data, sample_market_snapshot
    ) -> None:
        strategy = EventDrivenStrategy(Settings())
        result = await strategy.evaluate(sample_briefing_data, sample_market_snapshot)
        assert "catalysts" in result.debug_trace

    @pytest.mark.asyncio
    async def test_no_catalysts_returns_no_recommendation(self) -> None:
        empty_briefing = BriefingData(briefing_date=date.today())
        empty_market = MarketSnapshot()
        strategy = EventDrivenStrategy(Settings())
        result = await strategy.evaluate(empty_briefing, empty_market)
        assert result.recommendation is None

    @pytest.mark.asyncio
    async def test_abstains_on_degraded_briefing(self, sample_market_snapshot) -> None:
        from src.models.briefing import BriefingQuality

        degraded = BriefingData(
            briefing_date=date.today(),
            executive_summary="Synthesis unavailable for today's briefing.",
            briefing_quality=BriefingQuality.DEGRADED,
        )
        strategy = EventDrivenStrategy(Settings())
        result = await strategy.evaluate(degraded, sample_market_snapshot)
        assert result.recommendation is None
        assert result.debug_trace.get("skip_reason") == "degraded_briefing"
        assert result.debug_trace.get("briefing_quality") == "degraded"
