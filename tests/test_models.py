from datetime import date

import pytest

from src.models.briefing import BriefingData
from src.models.market import MarketSnapshot, NewsHeadline
from src.models.recommendation import (
    AlpacaOrderPayload,
    Direction,
    PositionIntent,
    TradeRecommendation,
)


class TestBriefingData:
    def test_macro_sentiment_positive(self) -> None:
        bd = BriefingData(
            briefing_date=date.today(),
            executive_summary="Strong surge in market optimism with bullish growth",
        )
        assert bd.macro_sentiment > 0

    def test_macro_sentiment_negative(self) -> None:
        bd = BriefingData(
            briefing_date=date.today(),
            executive_summary="Sharp decline and bearish pessimism ahead",
        )
        assert bd.macro_sentiment < 0

    def test_macro_sentiment_neutral(self) -> None:
        bd = BriefingData(
            briefing_date=date.today(),
            executive_summary="The market opened at regular hours.",
        )
        assert bd.macro_sentiment == 0.0


class TestMarketSnapshot:
    def test_avg_sentiment_positive(self) -> None:
        ms = MarketSnapshot(
            news=[
                NewsHeadline(
                    title="Up",
                    source="a",
                    url="https://example.com",
                    snippet="Good news",
                    polarity=0.5,
                ),
                NewsHeadline(
                    title="Down",
                    source="b",
                    url="https://example.com",
                    snippet="Bad news",
                    polarity=-0.3,
                ),
            ],
        )
        avg = ms.avg_sentiment_polarity()
        assert avg == 0.1

    def test_avg_sentiment_empty(self) -> None:
        ms = MarketSnapshot(news=[])
        assert ms.avg_sentiment_polarity() == 0.0


class TestTradeRecommendation:
    def test_puts_confidence_between_zero_and_one(self) -> None:
        rec = TradeRecommendation(
            correlation_id="t1",
            strategy_label="a",
            asset="SPY",
            direction=Direction.CALL,
            confidence=0.5,
            target_strike=746.0,
            contracts=1,
            order_type="market",
            position_intent=PositionIntent.BUY_TO_OPEN,
            rationale={},
            expires_at="2026-07-17",
            must_close_before="15:30",
        )
        assert 0.0 <= rec.confidence <= 1.0


class TestAlpacaOrderPayload:
    def test_requires_symbol_or_legs(self) -> None:
        payload = AlpacaOrderPayload(
            qty="1", type="market", time_in_force="day", symbol="SPY250717C00746000"
        )
        assert payload.symbol == "SPY250717C00746000"

    def test_rejects_both_symbol_and_legs(self) -> None:
        with pytest.raises(ValueError):
            AlpacaOrderPayload(
                qty="1",
                type="market",
                time_in_force="day",
                symbol="SPY250717C00746000",
                legs=[{"symbol": "X", "ratio_qty": "1"}],
            )

    def test_rejects_neither(self) -> None:
        with pytest.raises(ValueError):
            AlpacaOrderPayload(qty="1", type="market", time_in_force="day")
