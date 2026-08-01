from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from src.config import Settings
from src.engine.risk import RiskEngine
from src.models.market import DataSource, MarketSnapshot, Quote
from src.models.recommendation import Direction, PositionIntent, TradeRecommendation

_TS = __import__("datetime").datetime.now()

ET_TZ = ZoneInfo("America/New_York")
_9_30_AM_ET = datetime(2026, 1, 1, 9, 30, tzinfo=ET_TZ)


@pytest.fixture
def risk_engine() -> RiskEngine:
    return RiskEngine(Settings())


@pytest.fixture
def valid_rec() -> TradeRecommendation:
    return TradeRecommendation(
        correlation_id="test-1",
        strategy_label="momentum",
        asset="SPY",
        direction=Direction.CALL,
        confidence=0.75,
        target_strike=746.0,
        contracts=1,
        order_type="market",
        position_intent=PositionIntent.BUY_TO_OPEN,
        rationale={},
        expires_at=date.today().isoformat(),
        must_close_before="15:30",
    )


class TestRiskEngineTimeCheck:
    def test_passes_valid_time(
        self, risk_engine: RiskEngine, valid_rec: TradeRecommendation
    ) -> None:
        valid_rec.target_strike = 100.0
        valid_rec.contracts = 1
        result = risk_engine.validate(valid_rec, MarketSnapshot(), _now=_9_30_AM_ET)
        assert result is valid_rec

    def test_max_position_size_exceeded(
        self, risk_engine: RiskEngine, valid_rec: TradeRecommendation
    ) -> None:
        valid_rec.contracts = 100
        with pytest.raises(Exception):
            risk_engine.validate(valid_rec, MarketSnapshot())

    def test_max_loss_exceeded(
        self, risk_engine: RiskEngine, valid_rec: TradeRecommendation
    ) -> None:
        valid_rec.contracts = 1000
        with pytest.raises(Exception):
            risk_engine.validate(valid_rec, MarketSnapshot())


class TestRiskEngineConsensus:
    def test_consensus_positive(self) -> None:
        ok, avg = RiskEngine.check_consensus(0.5, 0.3, min_sources=2)
        assert ok is True
        assert avg > 0

    def test_consensus_negative(self) -> None:
        ok, avg = RiskEngine.check_consensus(-0.5, -0.3, min_sources=2)
        assert ok is True
        assert avg < 0

    def test_consensus_insufficient_sources(self) -> None:
        ok, avg = RiskEngine.check_consensus(0.02, 0.01, min_sources=2)
        assert ok is False
        assert avg == 0.0


def _market_with_quote(symbol: str, current: float, prev_close: float) -> MarketSnapshot:
    return MarketSnapshot(
        quotes={
            symbol: Quote(
                symbol=symbol,
                current_price=current,
                open_price=current,
                high_price=current,
                low_price=current,
                previous_close=prev_close,
                change_pct=(current - prev_close) / prev_close * 100 if prev_close > 0 else 0.0,
                volume=0,
                source=DataSource.FINNHUB,
                timestamp=_TS,
            ),
        },
    )


class TestRiskEnginePreMarketGap:
    def test_gap_opposite_direction_rejects(
        self, risk_engine: RiskEngine, valid_rec: TradeRecommendation
    ) -> None:
        valid_rec.asset = "SPY"
        valid_rec.direction = Direction.PUT
        valid_rec.target_strike = 100.0
        valid_rec.contracts = 1
        market = _market_with_quote("SPY", current=746.0, prev_close=735.0)
        with pytest.raises(Exception) as exc_info:
            risk_engine.validate(valid_rec, market, _now=_9_30_AM_ET)
        assert "gap" in str(exc_info.value)

    def test_gap_same_direction_passes(
        self, risk_engine: RiskEngine, valid_rec: TradeRecommendation
    ) -> None:
        valid_rec.asset = "SPY"
        valid_rec.direction = Direction.CALL
        valid_rec.target_strike = 100.0
        valid_rec.contracts = 1
        market = _market_with_quote("SPY", current=746.0, prev_close=735.0)
        result = risk_engine.validate(valid_rec, market, _now=_9_30_AM_ET)
        assert result is valid_rec

    def test_gap_below_threshold_passes(
        self, risk_engine: RiskEngine, valid_rec: TradeRecommendation
    ) -> None:
        valid_rec.asset = "SPY"
        valid_rec.direction = Direction.PUT
        valid_rec.target_strike = 100.0
        valid_rec.contracts = 1
        market = _market_with_quote("SPY", current=746.0, prev_close=744.0)
        result = risk_engine.validate(valid_rec, market, _now=_9_30_AM_ET)
        assert result is valid_rec

    def test_no_quote_skips_gracefully(
        self, risk_engine: RiskEngine, valid_rec: TradeRecommendation
    ) -> None:
        valid_rec.asset = "QQQ"
        valid_rec.direction = Direction.PUT
        valid_rec.target_strike = 100.0
        valid_rec.contracts = 1
        result = risk_engine.validate(valid_rec, MarketSnapshot(), _now=_9_30_AM_ET)
        assert result is valid_rec

    def test_previous_close_zero_skips(
        self, risk_engine: RiskEngine, valid_rec: TradeRecommendation
    ) -> None:
        valid_rec.asset = "SPY"
        valid_rec.direction = Direction.PUT
        valid_rec.target_strike = 100.0
        valid_rec.contracts = 1
        market = _market_with_quote("SPY", current=746.0, prev_close=0.0)
        result = risk_engine.validate(valid_rec, market, _now=_9_30_AM_ET)
        assert result is valid_rec
