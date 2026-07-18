from datetime import date

import pytest

from src.config import Settings
from src.engine.risk import RiskEngine
from src.models.market import MarketSnapshot
from src.models.recommendation import Direction, PositionIntent, TradeRecommendation


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
    def test_passes_valid_time(self, risk_engine: RiskEngine, valid_rec: TradeRecommendation) -> None:
        valid_rec.target_strike = 100.0
        valid_rec.contracts = 1
        result = risk_engine.validate(valid_rec, MarketSnapshot())
        assert result is valid_rec

    def test_max_position_size_exceeded(self, risk_engine: RiskEngine, valid_rec: TradeRecommendation) -> None:
        valid_rec.contracts = 100
        with pytest.raises(Exception):
            risk_engine.validate(valid_rec, MarketSnapshot())

    def test_max_loss_exceeded(self, risk_engine: RiskEngine, valid_rec: TradeRecommendation) -> None:
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
