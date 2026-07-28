"""Unit tests for ExitManager — take-profit, stop-loss, trailing, time exits."""

from __future__ import annotations

from datetime import datetime

from src.execution.exit_manager import ExitManager
from src.execution.models import ExitConfig, TrailingConfig
from src.timezone import ET_TZ


class TestExitManagerBasic:
    def test_calculates_take_profit_correctly(self) -> None:
        config = ExitConfig(take_profit_pct=100.0, stop_loss_pct=-50.0)
        mgr = ExitManager(config)
        mgr.on_entry_filled(0.50)
        assert mgr.tp_level == 1.00
        assert mgr.sl_level == 0.25

    def test_calculates_tp_sl_with_custom_thresholds(self) -> None:
        config = ExitConfig(take_profit_pct=50.0, stop_loss_pct=-30.0)
        mgr = ExitManager(config)
        mgr.on_entry_filled(1.00)
        assert mgr.tp_level == 1.50
        assert mgr.sl_level == 0.70

    def test_rejects_zero_entry_price(self) -> None:
        config = ExitConfig()
        mgr = ExitManager(config)
        try:
            mgr.on_entry_filled(0.0)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_rejects_negative_entry_price(self) -> None:
        config = ExitConfig()
        mgr = ExitManager(config)
        try:
            mgr.on_entry_filled(-1.0)
            assert False, "Expected ValueError"
        except ValueError:
            pass


class TestExitManagerEvaluate:
    def test_no_trigger_when_price_between_levels(self) -> None:
        config = ExitConfig(take_profit_pct=100.0, stop_loss_pct=-50.0)
        mgr = ExitManager(config)
        mgr.on_entry_filled(0.50)
        result = mgr.evaluate(0.75)
        assert result["triggered"] is False
        assert result["trigger_type"] is None
        assert result["current_pnl_pct"] == 50.0

    def test_triggers_take_profit(self) -> None:
        config = ExitConfig(take_profit_pct=100.0, stop_loss_pct=-50.0)
        mgr = ExitManager(config)
        mgr.on_entry_filled(0.50)
        result = mgr.evaluate(1.50)
        assert result["triggered"] is True
        assert result["trigger_type"] == "take_profit"
        assert result["current_pnl_pct"] == 200.0

    def test_triggers_stop_loss(self) -> None:
        config = ExitConfig(take_profit_pct=100.0, stop_loss_pct=-50.0)
        mgr = ExitManager(config)
        mgr.on_entry_filled(0.50)
        result = mgr.evaluate(0.20)
        assert result["triggered"] is True
        assert result["trigger_type"] == "stop_loss"
        assert result["current_pnl_pct"] == -60.0

    def test_no_entry_price_yields_safe_result(self) -> None:
        config = ExitConfig()
        mgr = ExitManager(config)
        result = mgr.evaluate(1.00)
        assert result["triggered"] is False
        assert result["trigger_type"] is None
        assert result["current_pnl_pct"] == 0.0


class TestExitManagerTrailing:
    def test_trailing_activates_after_threshold(self) -> None:
        config = ExitConfig(
            take_profit_pct=200.0,
            stop_loss_pct=-50.0,
            trailing=TrailingConfig(enabled=True, activate_after_pct=30.0, trail_pct=15.0),
        )
        mgr = ExitManager(config)
        mgr.on_entry_filled(0.50)
        mgr.evaluate(0.70)
        assert mgr.trailing_active is True
        assert mgr.trail_level is not None

    def test_trailing_does_not_activate_below_threshold(self) -> None:
        config = ExitConfig(
            trailing=TrailingConfig(enabled=True, activate_after_pct=30.0, trail_pct=15.0),
        )
        mgr = ExitManager(config)
        mgr.on_entry_filled(0.50)
        mgr.evaluate(0.60)
        assert mgr.trailing_active is False
        assert mgr.trail_level is None

    def test_trailing_stop_fires_on_drop(self) -> None:
        config = ExitConfig(
            take_profit_pct=200.0,
            stop_loss_pct=-90.0,
            trailing=TrailingConfig(enabled=True, activate_after_pct=30.0, trail_pct=15.0),
        )
        mgr = ExitManager(config)
        mgr.on_entry_filled(0.50)
        mgr.evaluate(0.80)
        assert mgr.trailing_active
        result = mgr.evaluate(0.68)
        assert result["triggered"] is True
        assert result["trigger_type"] == "trailing_stop"

    def test_trailing_disabled_when_config_off(self) -> None:
        config = ExitConfig(
            trailing=TrailingConfig(enabled=False, activate_after_pct=30.0, trail_pct=15.0),
        )
        mgr = ExitManager(config)
        mgr.on_entry_filled(0.50)
        mgr.evaluate(0.80)
        assert mgr.trailing_active is False


class TestExitManagerTimeDeadline:
    def test_deadline_not_reached_early(self) -> None:
        config = ExitConfig(time_deadline_est="15:25")
        mgr = ExitManager(config)
        early = datetime(2026, 7, 28, 10, 0, tzinfo=ET_TZ)
        assert mgr.is_time_deadline_approaching(now=early) is False

    def test_deadline_reached_at_cutoff(self) -> None:
        config = ExitConfig(time_deadline_est="15:25")
        mgr = ExitManager(config)
        cutoff = datetime(2026, 7, 28, 15, 25, tzinfo=ET_TZ)
        assert mgr.is_time_deadline_approaching(now=cutoff) is True

    def test_deadline_triggered_in_evaluate(self) -> None:
        config = ExitConfig(time_deadline_est="10:00")
        mgr = ExitManager(config)
        mgr.on_entry_filled(0.50)
        after = datetime(2026, 7, 28, 15, 30, tzinfo=ET_TZ)
        result = mgr.evaluate(0.60)
        if mgr.is_time_deadline_approaching(now=after):
            assert result["triggered"] is True
            assert result["trigger_type"] == "time_deadline"


class TestExitManagerBuildOrders:
    def test_builds_tp_limit_order(self) -> None:
        config = ExitConfig(take_profit_pct=100.0)
        mgr = ExitManager(config)
        mgr.on_entry_filled(0.50)
        order = mgr.build_tp_order("SPY250728C00600000", 2)
        assert order["symbol"] == "SPY250728C00600000"
        assert order["side"] == "sell"
        assert order["type"] == "limit"
        assert order["limit_price"] == 1.00
        assert order["qty"] == 2

    def test_builds_sl_stop_order(self) -> None:
        config = ExitConfig(stop_loss_pct=-50.0)
        mgr = ExitManager(config)
        mgr.on_entry_filled(0.50)
        order = mgr.build_sl_order("SPY250728C00600000", 1)
        assert order["symbol"] == "SPY250728C00600000"
        assert order["side"] == "sell"
        assert order["type"] == "stop"
        assert order["stop_price"] == 0.25
        assert order["qty"] == 1

    def test_builds_market_close_order(self) -> None:
        config = ExitConfig()
        mgr = ExitManager(config)
        order = mgr.build_market_close_order(
            "SPY250728C00600000", 3, client_order_id="force-close-123"
        )
        assert order["symbol"] == "SPY250728C00600000"
        assert order["type"] == "market"
        assert order["qty"] == 3
        assert order["client_order_id"] == "force-close-123"
