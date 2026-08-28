from __future__ import annotations

import json

import pytest

from src.config import Settings
from src.engine.decision import DecisionAggregator
from src.models.recommendation import Direction, PositionIntent, StrategyResult, TradeRecommendation
from src.trade_tracker import (
    TradeOutcome,
    compute_direction_stats,
    compute_strategy_stats,
    format_outcomes_for_prompt,
    load_trade_outcomes,
)


def _write_trade(
    log_dir,
    date_str: str,
    trade_id: str,
    *,
    asset: str = "SPY",
    direction: str = "CALL",
    strategy: str = "momentum",
    entry_price: float = 0.5,
    exit_price: float = 1.0,
    exit_reason: str = "take_profit",
    pnl: float = 50.0,
    pnl_pct: float = 100.0,
) -> None:
    day_dir = log_dir / date_str
    day_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "trade_id": trade_id,
        "correlation_id": "c-" + trade_id,
        "asset": asset,
        "direction": direction,
        "entry_strike": 700.0,
        "exit_reason": exit_reason,
        "exit_price": exit_price,
        "final_pnl": pnl,
        "final_pnl_pct": pnl_pct,
        "entries": [
            {
                "event_type": "entry_filled",
                "avg_price": entry_price,
                "filled_qty": 1,
            }
        ],
        "recommendation": {"strategy_label": strategy},
    }
    (day_dir / f"trade-{trade_id}.json").write_text(json.dumps(data))


class TestTradeOutcome:
    def test_resolved_win(self) -> None:
        raw = {
            "trade_id": "t1",
            "asset": "SPY",
            "direction": "CALL",
            "exit_reason": "take_profit",
            "exit_price": 1.0,
            "final_pnl": 50.0,
            "final_pnl_pct": 100.0,
            "entries": [{"event_type": "entry_filled", "avg_price": 0.5}],
            "recommendation": {"strategy_label": "momentum"},
        }
        o = TradeOutcome(raw, "2026-08-07")
        assert o.won is True
        assert o.entry_price == pytest.approx(0.5)
        assert o.strategy == "momentum"

    def test_unresolved_excluded(self, tmp_path) -> None:
        _write_trade(
            tmp_path,
            "2026-08-11",
            "pending1",
            exit_reason="pending",
            exit_price=0.0,
            pnl=0.0,
            pnl_pct=0.0,
        )
        _write_trade(
            tmp_path,
            "2026-08-10",
            "expired1",
            exit_reason="expired",
            exit_price=0.0,
            pnl=0.0,
            pnl_pct=0.0,
        )
        _write_trade(
            tmp_path,
            "2026-08-09",
            "unfilled1",
            exit_reason="unfilled",
            exit_price=0.0,
            pnl=0.0,
            pnl_pct=0.0,
        )
        outcomes = load_trade_outcomes(tmp_path)
        assert outcomes == []

    def test_expired_worthless_is_a_learnable_loss(self, tmp_path) -> None:
        # Bought a 0DTE option that expired OTM — a real loss, must be counted.
        _write_trade(
            tmp_path,
            "2026-08-11",
            "ew1",
            entry_price=0.73,
            exit_price=0.0,
            exit_reason="expired_worthless",
            pnl=-73.0,
            pnl_pct=-100.0,
        )
        outcomes = load_trade_outcomes(tmp_path)
        assert len(outcomes) == 1
        assert outcomes[0].lost is True
        assert outcomes[0].pnl == pytest.approx(-73.0)


class TestStrategyStats:
    def test_streak_negative_two(self, tmp_path) -> None:
        _write_trade(
            tmp_path,
            "2026-08-11",
            "a",
            pnl=-65.0,
            pnl_pct=-94.2,
            exit_price=0.04,
            exit_reason="safety_close",
        )
        _write_trade(
            tmp_path,
            "2026-08-12",
            "b",
            pnl=-20.0,
            pnl_pct=-50.0,
            exit_price=0.1,
            exit_reason="safety_close",
        )
        outcomes = load_trade_outcomes(tmp_path)
        stats = compute_strategy_stats(outcomes)
        assert stats["momentum"]["wins"] == 0
        assert stats["momentum"]["losses"] == 2
        assert stats["momentum"]["current_streak"] == -2
        assert stats["momentum"]["total_pnl"] == pytest.approx(-85.0)

    def test_streak_win_then_loss(self, tmp_path) -> None:
        _write_trade(
            tmp_path, "2026-08-07", "w", pnl=73.0, exit_price=1.46, exit_reason="take_profit"
        )
        _write_trade(
            tmp_path,
            "2026-08-11",
            "l",
            pnl=-65.0,
            exit_price=0.04,
            exit_reason="safety_close",
            pnl_pct=-94.2,
        )
        outcomes = load_trade_outcomes(tmp_path)
        stats = compute_strategy_stats(outcomes)
        assert stats["momentum"]["wins"] == 1
        assert stats["momentum"]["losses"] == 1
        assert stats["momentum"]["current_streak"] == -1


class TestDirectionStats:
    def test_direction_streak(self, tmp_path) -> None:
        _write_trade(
            tmp_path,
            "2026-08-10",
            "q1",
            asset="QQQ",
            direction="CALL",
            pnl=-73.0,
            exit_price=0.0,
            exit_reason="safety_close",
        )
        _write_trade(
            tmp_path,
            "2026-08-11",
            "s1",
            asset="SPY",
            direction="CALL",
            pnl=-20.0,
            exit_price=0.0,
            exit_reason="safety_close",
        )
        outcomes = load_trade_outcomes(tmp_path)
        stats = compute_direction_stats(outcomes)
        assert stats["SPY:CALL"]["current_streak"] == -1
        assert stats["QQQ:CALL"]["current_streak"] == -1


class TestFormatForPrompt:
    def test_format_includes_learning_directive(self, tmp_path) -> None:
        _write_trade(
            tmp_path,
            "2026-08-11",
            "a",
            pnl=-65.0,
            pnl_pct=-94.2,
            exit_price=0.04,
            exit_reason="safety_close",
        )
        outcomes = load_trade_outcomes(tmp_path)
        text = format_outcomes_for_prompt(outcomes)
        assert "Actual trade results" in text
        assert "Learning directive" in text
        assert "-$65.00" in text

    def test_format_empty(self, tmp_path) -> None:
        assert format_outcomes_for_prompt(load_trade_outcomes(tmp_path)) == ""


class TestStreakDampening:
    def _rec(self, strategy="momentum", asset="SPY", direction=Direction.CALL, conf=0.8):
        return TradeRecommendation(
            correlation_id="t1",
            strategy_label=strategy,
            asset=asset,
            direction=direction,
            confidence=conf,
            target_strike=700.0,
            contracts=1,
            order_type="market",
            position_intent=PositionIntent.BUY_TO_OPEN,
            rationale={},
            expires_at="2026-08-13",
            must_close_before="15:30",
        )

    def test_dampens_losing_streak(self, tmp_path) -> None:
        _write_trade(
            tmp_path, "2026-08-10", "a", pnl=-65.0, exit_price=0.04, exit_reason="safety_close"
        )
        _write_trade(
            tmp_path, "2026-08-11", "b", pnl=-20.0, exit_price=0.1, exit_reason="safety_close"
        )

        settings = Settings()
        settings.logging.json_dir = str(tmp_path)
        settings.general.require_forecast_alignment = False

        agg = DecisionAggregator(settings)
        rec = self._rec(conf=0.8)
        results = [
            StrategyResult(label="momentum", recommendation=rec, confidence=0.8, duration_ms=1.0),
            # Corroborating strategy on the same asset+direction. Without it
            # the unsupported-signal cap blocks the trade outright and this
            # test can no longer observe the streak penalty it is asserting.
            StrategyResult(
                label="event_driven",
                recommendation=self._rec(conf=0.5),
                confidence=0.5,
                duration_ms=1.0,
            ),
        ]
        decision = agg.aggregate(results)
        # 2-loss streak → -0.10 penalty on 0.80 → 0.70
        assert decision.recommendation is not None
        assert decision.recommendation.confidence == pytest.approx(0.70, abs=0.01)

    def test_no_penalty_without_history(self, tmp_path) -> None:
        settings = Settings()
        settings.logging.json_dir = str(tmp_path)
        settings.general.require_forecast_alignment = False

        agg = DecisionAggregator(settings)
        rec = self._rec(conf=0.8)
        results = [
            StrategyResult(label="momentum", recommendation=rec, confidence=0.8, duration_ms=1.0),
            # Corroborating strategy on the same asset+direction. Without it
            # the unsupported-signal cap blocks the trade outright and this
            # test can no longer observe the streak penalty it is asserting.
            StrategyResult(
                label="event_driven",
                recommendation=self._rec(conf=0.5),
                confidence=0.5,
                duration_ms=1.0,
            ),
        ]
        decision = agg.aggregate(results)
        assert decision.recommendation.confidence == pytest.approx(0.80)
