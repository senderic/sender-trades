from datetime import date

import pytest

from src.config import Settings
from src.engine.strategy_a import MomentumStrategy
from src.engine.strategy_b import MeanReversionStrategy
from src.engine.strategy_c import EventDrivenStrategy
from src.models.briefing import BriefingData
from src.models.market import MarketSnapshot
from src.models.recommendation import Direction


class TestMomentumStrategy:
    @pytest.mark.asyncio
    async def test_gap_up_generates_call(self, sample_briefing_data, sample_market_snapshot) -> None:
        config = Settings()
        config.strategies.momentum.gap_threshold_pct = 0.1
        strategy = MomentumStrategy(config)
        result = await strategy.evaluate(sample_briefing_data, sample_market_snapshot)
        assert result.recommendation is not None
        assert result.recommendation.direction == Direction.CALL

    @pytest.mark.asyncio
    async def test_low_confidence_returns_none(self, sample_briefing_data, sample_market_snapshot) -> None:
        config = Settings()
        config.strategies.momentum.min_confidence = 1.0
        strategy = MomentumStrategy(config)
        result = await strategy.evaluate(sample_briefing_data, sample_market_snapshot)
        assert result.recommendation is None


class TestMeanReversionStrategy:
    @pytest.mark.asyncio
    async def test_evaluates_without_error(self, sample_briefing_data, sample_market_snapshot) -> None:
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
