from datetime import date, datetime

import pytest

from src.models.briefing import BriefingData
from src.models.market import DataSource, MarketSnapshot, NewsHeadline, PremarketQuote, Quote
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


def _quote(symbol: str, current: float, previous_close: float) -> Quote:
    return Quote(
        symbol=symbol,
        current_price=current,
        open_price=current,
        high_price=current,
        low_price=current,
        previous_close=previous_close,
        change_pct=0.0,
        volume=0,
        source=DataSource.FINNHUB,
        timestamp=datetime.now(),
    )


class TestQuotePriorSessionClose:
    def test_prior_session_close_is_current_price(self) -> None:
        """Verified on the 2026-09-11 incident: a pre-market-stale Quote's
        ``current_price`` is actually the PRIOR session's close, not
        today's price."""
        q = _quote("QQQ", current=708.69, previous_close=716.31)
        assert q.prior_session_close == 708.69
        assert q.prior_session_close != q.previous_close


class TestMarketSnapshotMechanics:
    def test_mechanics_price_prefers_live_premarket(self) -> None:
        ms = MarketSnapshot(
            quotes={"QQQ": _quote("QQQ", current=708.69, previous_close=716.31)},
            premarket={
                "QQQ": PremarketQuote(symbol="QQQ", available=True, price=715.44, reliable=True)
            },
        )
        assert ms.mechanics_price("QQQ") == 715.44

    def test_mechanics_price_ignores_reliability_flag(self) -> None:
        """MECHANICS uses the live price regardless of volume/reliability
        -- only the LLM prompt narrative weighs reliability."""
        ms = MarketSnapshot(
            quotes={"QQQ": _quote("QQQ", current=708.69, previous_close=716.31)},
            premarket={
                "QQQ": PremarketQuote(symbol="QQQ", available=True, price=715.44, reliable=False)
            },
        )
        assert ms.mechanics_price("QQQ") == 715.44

    def test_mechanics_price_falls_back_to_prior_session_close(self, caplog) -> None:
        ms = MarketSnapshot(quotes={"QQQ": _quote("QQQ", current=708.69, previous_close=716.31)})
        assert ms.mechanics_price("QQQ") == 708.69

    def test_mechanics_price_none_without_any_data(self) -> None:
        ms = MarketSnapshot()
        assert ms.mechanics_price("QQQ") is None

    def test_mechanics_gap_pct_uses_live_price_vs_prior_session_close(self) -> None:
        ms = MarketSnapshot(
            quotes={"QQQ": _quote("QQQ", current=708.69, previous_close=716.31)},
            premarket={
                "QQQ": PremarketQuote(symbol="QQQ", available=True, price=715.44, reliable=True)
            },
        )
        gap = ms.mechanics_gap_pct("QQQ")
        assert gap == pytest.approx((715.44 - 708.69) / 708.69 * 100, abs=1e-6)

    def test_mechanics_gap_pct_is_zero_when_premarket_unavailable(self) -> None:
        """Fallback conservatively assumes no gap rather than fabricating
        one from the prior session's own open-vs-two-days-ago-close."""
        ms = MarketSnapshot(quotes={"QQQ": _quote("QQQ", current=708.69, previous_close=716.31)})
        assert ms.mechanics_gap_pct("QQQ") == 0.0

    def test_mechanics_gap_pct_none_without_any_quote(self) -> None:
        ms = MarketSnapshot()
        assert ms.mechanics_gap_pct("QQQ") is None


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
