from datetime import date

from src.config import Settings
from src.engine.decision import DecisionAggregator
from src.models.recommendation import (
    Direction,
    PositionIntent,
    StrategyResult,
    TradeRecommendation,
)


def _make_rec(strategy: str, asset: str = "SPY") -> TradeRecommendation:
    return TradeRecommendation(
        correlation_id="t1",
        strategy_label=strategy,
        asset=asset,
        direction=Direction.CALL,
        confidence=0.5,
        target_strike=746.0,
        contracts=1,
        order_type="market",
        position_intent=PositionIntent.BUY_TO_OPEN,
        rationale={},
        expires_at=date.today().isoformat(),
        must_close_before="15:30",
    )


class TestDecisionAggregator:
    def test_no_valid_results_returns_empty(self) -> None:
        agg = DecisionAggregator(Settings())
        result = agg.aggregate([])
        assert result.recommendation is None
        assert result.selected_label is None

    def test_picks_highest_confidence(self) -> None:
        agg = DecisionAggregator(Settings())
        results = [
            StrategyResult(label="a", recommendation=_make_rec("a"),
                           confidence=0.3, duration_ms=1.0),
            StrategyResult(label="b", recommendation=_make_rec("b"),
                           confidence=0.8, duration_ms=1.0),
        ]
        result = agg.aggregate(results)
        assert result.selected_label == "b"

    def test_below_threshold_returns_none(self) -> None:
        agg = DecisionAggregator(Settings())
        results = [
            StrategyResult(label="a", recommendation=_make_rec("a"),
                           confidence=0.1, duration_ms=1.0),
        ]
        result = agg.aggregate(results)
        assert result.recommendation is None
