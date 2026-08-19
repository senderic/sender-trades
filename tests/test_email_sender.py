from datetime import datetime

from src.email_sender import _render_decision_text, render_forecast_html
from src.models.recommendation import (
    DecisionOutput,
    Direction,
    DirectionalForecast,
    PositionIntent,
    StrategyResult,
    TradeRecommendation,
)
from src.timezone import today_local


def _make_rec(
    strategy: str, asset: str = "SPY", direction: Direction = Direction.CALL
) -> TradeRecommendation:
    return TradeRecommendation(
        correlation_id="t1",
        strategy_label=strategy,
        asset=asset,
        direction=direction,
        confidence=0.82,
        target_strike=772.0,
        contracts=1,
        order_type="market",
        position_intent=PositionIntent.BUY_TO_OPEN,
        rationale={},
        expires_at=today_local().isoformat(),
        must_close_before="15:30",
    )


def _make_forecast() -> DirectionalForecast:
    return DirectionalForecast(
        forecasts=[],
        market_vibe="",
        generated_at=datetime(2026, 8, 19, 6, 31, 0),
    )


def _make_vetoed_decision() -> DecisionOutput:
    blocked = _make_rec("momentum+event_driven", "SPY", Direction.CALL)
    results = [
        StrategyResult(
            label="momentum+event_driven", recommendation=blocked, confidence=0.82, duration_ms=1.0
        ),
        StrategyResult(
            label="llm_trade",
            recommendation=_make_rec("llm_trade", "QQQ", Direction.PUT),
            confidence=0.7,
            duration_ms=1.0,
        ),
    ]
    return DecisionOutput(
        selected_label=None,
        recommendation=None,
        all_results=results,
        rationale=(
            "Blocked momentum+event_driven CALL on SPY: trade direction CALL "
            "(implying UP) conflicts with LLM forecast DOWN (confidence 58%)"
        ),
    )


def _make_dry_run_decision() -> DecisionOutput:
    rec = _make_rec("llm_trade", "QQQ", Direction.PUT)
    return DecisionOutput(
        selected_label="llm_trade",
        recommendation=rec,
        all_results=[
            StrategyResult(label="llm_trade", recommendation=rec, confidence=0.7, duration_ms=1.0)
        ],
        rationale="Selected strategy llm_trade with confidence 0.70.",
    )


class TestTradeDecisionEmailSection:
    def test_vetoed_no_trade_shows_rationale(self) -> None:
        html = render_forecast_html(
            _make_forecast(),
            decision=_make_vetoed_decision(),
        )
        assert "Trade Decision" in html
        assert "No trade placed" in html
        assert "conflicts with LLM forecast DOWN" in html
        assert "Top signal considered" in html

    def test_dry_run_recommended_not_executed(self) -> None:
        html = render_forecast_html(
            _make_forecast(),
            decision=_make_dry_run_decision(),
        )
        assert "Trade Decision" in html
        assert "recommended but not executed" in html
        assert "dry run" in html

    def test_executed_trade_hides_decision_section(self) -> None:
        html = render_forecast_html(
            _make_forecast(),
            execution_result={"exit_reason": "take_profit"},
            decision=_make_dry_run_decision(),
        )
        assert "Trade Decision" not in html
        assert "Trade Execution" in html

    def test_no_decision_renders_nothing(self) -> None:
        html = render_forecast_html(_make_forecast())
        assert "Trade Decision" not in html

    def test_plain_text_vetoed(self) -> None:
        text = _render_decision_text(_make_vetoed_decision(), None)
        assert "No trade placed" in text
        assert "conflicts with LLM forecast DOWN" in text
        assert "Top signal considered" in text

    def test_plain_text_empty_when_executed(self) -> None:
        text = _render_decision_text(_make_dry_run_decision(), {"exit_reason": "take_profit"})
        assert text == ""
