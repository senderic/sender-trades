from datetime import date

import pytest

from src.config import Settings
from src.engine.decision import DecisionAggregator
from src.models.recommendation import (
    AssetPrediction,
    Direction,
    PositionIntent,
    StrategyResult,
    TradeRecommendation,
)


def _make_settings(tmp_path) -> Settings:
    """Build Settings isolated from real logs so streak dampening has no data."""
    settings = Settings()
    settings.logging.json_dir = str(tmp_path)
    return settings


def _make_rec(
    strategy: str, asset: str = "SPY", direction: Direction = Direction.CALL
) -> TradeRecommendation:
    return TradeRecommendation(
        correlation_id="t1",
        strategy_label=strategy,
        asset=asset,
        direction=direction,
        confidence=0.5,
        target_strike=746.0,
        contracts=1,
        order_type="market",
        position_intent=PositionIntent.BUY_TO_OPEN,
        rationale={},
        expires_at=date.today().isoformat(),
        must_close_before="15:30",
    )


def _make_result(label: str, rec: TradeRecommendation, confidence: float) -> StrategyResult:
    return StrategyResult(label=label, recommendation=rec, confidence=confidence, duration_ms=1.0)


class TestDecisionAggregator:
    def test_no_valid_results_returns_empty(self, tmp_path) -> None:
        agg = DecisionAggregator(_make_settings(tmp_path))
        result = agg.aggregate([])
        assert result.recommendation is None
        assert result.selected_label is None

    def test_picks_highest_confidence(self, tmp_path) -> None:
        agg = DecisionAggregator(_make_settings(tmp_path))
        results = [
            StrategyResult(
                label="a", recommendation=_make_rec("a"), confidence=0.3, duration_ms=1.0
            ),
            StrategyResult(
                label="b", recommendation=_make_rec("b"), confidence=0.8, duration_ms=1.0
            ),
        ]
        result = agg.aggregate(results)
        assert result.selected_label == "b"

    def test_below_threshold_returns_none(self, tmp_path) -> None:
        agg = DecisionAggregator(_make_settings(tmp_path))
        results = [
            StrategyResult(
                label="a", recommendation=_make_rec("a"), confidence=0.1, duration_ms=1.0
            ),
        ]
        result = agg.aggregate(results)
        assert result.recommendation is None


class TestConsensusScoring:
    def test_three_way_consensus_boosts_confidence(self, tmp_path) -> None:
        agg = DecisionAggregator(_make_settings(tmp_path))
        qqq_put = _make_rec("event", asset="QQQ", direction=Direction.PUT)
        qqq_put.confidence = 0.65
        results = [
            _make_result("momentum", _make_rec("m", asset="QQQ", direction=Direction.PUT), 0.45),
            _make_result(
                "mean_reversion", _make_rec("mr", asset="QQQ", direction=Direction.PUT), 0.50
            ),
            _make_result("event_driven", qqq_put, 0.65),
            _make_result(
                "llm_trade", _make_rec("llm", asset="QQQ", direction=Direction.CALL), 0.40
            ),
        ]
        decision = agg.aggregate(results)
        assert decision.recommendation is not None
        assert decision.recommendation.asset == "QQQ"
        assert decision.recommendation.direction == Direction.PUT
        assert decision.recommendation.confidence == pytest.approx(0.70)

    def test_four_way_split_penalizes_confidence(self, tmp_path) -> None:
        agg = DecisionAggregator(_make_settings(tmp_path))
        spy_call = _make_rec("event", asset="SPY", direction=Direction.CALL)
        spy_call.confidence = 0.55
        results = [
            _make_result("momentum", _make_rec("m", asset="SPY", direction=Direction.PUT), 0.45),
            _make_result(
                "mean_reversion", _make_rec("mr", asset="SPY", direction=Direction.PUT), 0.50
            ),
            _make_result("event_driven", spy_call, 0.55),
            _make_result(
                "llm_trade", _make_rec("llm", asset="SPY", direction=Direction.CALL), 0.40
            ),
        ]
        decision = agg.aggregate(results)
        assert decision.recommendation is not None
        assert decision.recommendation.confidence == pytest.approx(0.50)

    def test_single_strategy_no_consensus_effect(self, tmp_path) -> None:
        agg = DecisionAggregator(_make_settings(tmp_path))
        rec = _make_rec("event", asset="SPY", direction=Direction.CALL)
        rec.confidence = 0.60
        results = [
            _make_result("event_driven", rec, 0.60),
        ]
        decision = agg.aggregate(results)
        assert decision.recommendation is not None
        assert decision.recommendation.confidence == pytest.approx(0.60)

    def test_different_assets_no_split_penalty(self, tmp_path) -> None:
        agg = DecisionAggregator(_make_settings(tmp_path))
        spy_call = _make_rec("event", asset="SPY", direction=Direction.CALL)
        spy_call.confidence = 0.65
        results = [
            _make_result("momentum", _make_rec("m", asset="SPY", direction=Direction.CALL), 0.45),
            _make_result(
                "mean_reversion", _make_rec("mr", asset="SPY", direction=Direction.CALL), 0.50
            ),
            _make_result("event_driven", spy_call, 0.65),
            _make_result("llm_trade", _make_rec("llm", asset="QQQ", direction=Direction.PUT), 0.40),
        ]
        decision = agg.aggregate(results)
        assert decision.recommendation is not None
        assert decision.recommendation.asset == "SPY"
        assert decision.recommendation.direction == Direction.CALL
        assert decision.recommendation.confidence == pytest.approx(0.70)


class TestForecastAlignment:
    def _llm_prediction(
        self, asset: str = "SPY", direction: str = "UP", confidence: float = 0.6
    ) -> StrategyResult:
        return StrategyResult(
            label="llm_trade",
            recommendation=None,
            predictions={
                asset: AssetPrediction(
                    asset=asset,
                    direction=direction,
                    confidence=confidence,
                    predicted_move_pct=0.3 if direction == "UP" else -0.3,
                    rationale="test",
                    sources=["news-sentiment"],
                )
            },
            confidence=0.0,
            duration_ms=1.0,
        )

    def test_call_blocked_when_llm_forecasts_down(self, tmp_path) -> None:
        agg = DecisionAggregator(_make_settings(tmp_path))
        results = [
            _make_result("event_driven", _make_rec("event", asset="SPY", direction=Direction.CALL), 0.77),
            self._llm_prediction(asset="SPY", direction="DOWN", confidence=0.52),
        ]
        decision = agg.aggregate(results)
        assert decision.recommendation is None
        assert "conflicts with LLM forecast" in decision.rationale

    def test_put_blocked_when_llm_forecasts_up(self, tmp_path) -> None:
        agg = DecisionAggregator(_make_settings(tmp_path))
        results = [
            _make_result("event_driven", _make_rec("event", asset="QQQ", direction=Direction.PUT), 0.70),
            self._llm_prediction(asset="QQQ", direction="UP", confidence=0.60),
        ]
        decision = agg.aggregate(results)
        assert decision.recommendation is None

    def test_call_allowed_when_llm_forecasts_up(self, tmp_path) -> None:
        agg = DecisionAggregator(_make_settings(tmp_path))
        results = [
            _make_result("event_driven", _make_rec("event", asset="SPY", direction=Direction.CALL), 0.77),
            self._llm_prediction(asset="SPY", direction="UP", confidence=0.60),
        ]
        decision = agg.aggregate(results)
        assert decision.recommendation is not None
        assert decision.recommendation.direction == Direction.CALL

    def test_no_llm_prediction_allows_trade(self, tmp_path) -> None:
        agg = DecisionAggregator(_make_settings(tmp_path))
        results = [
            _make_result("event_driven", _make_rec("event", asset="SPY", direction=Direction.CALL), 0.77),
        ]
        decision = agg.aggregate(results)
        assert decision.recommendation is not None

    def test_different_asset_prediction_does_not_conflict(self, tmp_path) -> None:
        agg = DecisionAggregator(_make_settings(tmp_path))
        results = [
            _make_result("event_driven", _make_rec("event", asset="SPY", direction=Direction.CALL), 0.77),
            self._llm_prediction(asset="QQQ", direction="DOWN", confidence=0.60),
        ]
        decision = agg.aggregate(results)
        assert decision.recommendation is not None

    def test_alignment_can_be_disabled(self, tmp_path) -> None:
        settings = _make_settings(tmp_path)
        settings.general.require_forecast_alignment = False
        agg = DecisionAggregator(settings)
        results = [
            _make_result("event_driven", _make_rec("event", asset="SPY", direction=Direction.CALL), 0.77),
            self._llm_prediction(asset="SPY", direction="DOWN", confidence=0.52),
        ]
        decision = agg.aggregate(results)
        assert decision.recommendation is not None
