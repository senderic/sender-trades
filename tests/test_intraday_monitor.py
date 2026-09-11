"""Unit tests for the intraday stop-loss monitor."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from src.execution.intraday_monitor import (
    _close_position,
    _load_exec_config,
    _poll_fill,
    compute_pnl,
    extract_open_trade,
    find_open_trades,
    in_market_hours,
    load_audit_file,
    mark_from_quote,
    monitor_trade,
    should_trigger,
    write_close_attempt,
)
from src.timezone import ET_TZ


def _audit_data(
    exit_reason: str | None = "pending",
    tp_level: float = 1.0,
    sl_level: float = 0.5,
    entry_price: float = 0.8,
    occ_symbol: str = "QQQ26090C00713000",
) -> dict:
    return {
        "trade_id": "abc123",
        "correlation_id": "corr",
        "asset": "QQQ",
        "direction": "CALL",
        "contracts": 1,
        "entry_strike": 713.0,
        "exit_reason": exit_reason,
        "exit_price": None,
        "final_pnl": None,
        "final_pnl_pct": None,
        "started_at": "2026-09-03T13:30:01+00:00",
        "ended_at": None,
        "entries": [
            {
                "event_type": "entry_submitted",
                "occ_symbol": occ_symbol,
                "contracts": 1,
                "order_type": "limit",
            },
            {
                "event_type": "entry_filled",
                "order_id": "buy-order",
                "filled_qty": 1,
                "avg_price": entry_price,
            },
            {
                "event_type": "exits_placed",
                "tp_order_id": "tp-order",
                "tp_level": tp_level,
                "sl_level": sl_level,
            },
        ],
        "recommendation": {
            "strategy_label": "llm_trade",
            "asset": "QQQ",
            "direction": "CALL",
            "confidence": 0.5,
        },
    }


class TestParsing:
    def test_extract_open_trade(self) -> None:
        data = _audit_data()
        trade = extract_open_trade(data, __import__("pathlib").Path("x.json"))
        assert trade is not None
        assert trade.occ_symbol == "QQQ26090C00713000"
        assert trade.tp_level == 1.0
        assert trade.sl_level == 0.5
        assert trade.entry_price == 0.8
        assert trade.tp_order_id == "tp-order"
        assert trade.contracts == 1

    def test_extract_skips_resolved(self) -> None:
        for reason in ("take_profit", "stop_loss", "safety_close", "expired_worthless"):
            data = _audit_data(exit_reason=reason)
            assert extract_open_trade(data, __import__("pathlib").Path("x.json")) is None

    def test_extract_returns_none_without_symbol(self) -> None:
        data = _audit_data(occ_symbol=None)
        data["entries"][0]["occ_symbol"] = None
        assert extract_open_trade(data, __import__("pathlib").Path("x.json")) is None

    def test_extract_returns_none_without_sl(self) -> None:
        data = _audit_data()
        data["entries"][2]["sl_level"] = None
        assert extract_open_trade(data, __import__("pathlib").Path("x.json")) is None

    def test_find_open_trades_scans_day_dir(self, tmp_path) -> None:
        day_dir = tmp_path / "2026-09-03"
        day_dir.mkdir()
        # One pending, one resolved
        (day_dir / "trade-a.json").write_text(json.dumps(_audit_data(exit_reason="pending")))
        (day_dir / "trade-b.json").write_text(json.dumps(_audit_data(exit_reason="take_profit")))
        # Non-trade files must be ignored
        (day_dir / "summary.json").write_text("{}")
        (day_dir / "trade-c.json.bak").write_text(json.dumps(_audit_data()))

        open_trades = find_open_trades(tmp_path, trade_date="2026-09-03")
        assert len(open_trades) == 1
        assert open_trades[0].trade_id == "abc123"

    def test_find_open_trades_missing_day_dir(self, tmp_path) -> None:
        assert find_open_trades(tmp_path, trade_date="2026-09-03") == []


class TestLoadAuditFile:
    """`TradeContext` now rewrites the whole audit file atomically on
    every event (see src.execution.context), so a current file is
    always a single JSON object. These tests cover the legacy shape --
    several JSON objects concatenated in one file by the prior
    append-only writer -- that `load_audit_file` must still tolerate."""

    def test_single_object_file(self, tmp_path) -> None:
        path = tmp_path / "trade-a.json"
        path.write_text(json.dumps(_audit_data()))
        data = load_audit_file(path)
        assert data["trade_id"] == "abc123"
        assert data["exit_reason"] == "pending"

    def test_legacy_concatenated_file_is_parsed_and_merged(self, tmp_path) -> None:
        path = tmp_path / "trade-legacy.json"
        lines = [
            {"trade_id": "legacy1", "event_type": "entry_submitted", "occ_symbol": "SPY"},
            {"trade_id": "legacy1", "event_type": "entry_filled", "avg_price": 0.6},
            {"trade_id": "legacy1", "event_type": "exits_placed", "tp_level": 1.0, "sl_level": 0.4},
        ]
        path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

        data = load_audit_file(path)
        trade = extract_open_trade(data, path)
        assert trade is not None
        assert trade.occ_symbol == "SPY"
        assert trade.entry_price == 0.6
        assert trade.sl_level == 0.4
        assert trade.tp_level == 1.0

    def test_empty_file_returns_empty_dict(self, tmp_path) -> None:
        path = tmp_path / "trade-empty.json"
        path.write_text("")
        assert load_audit_file(path) == {}


class TestSignals:
    def test_should_trigger(self) -> None:
        assert should_trigger(0.40, 0.50) is True
        assert should_trigger(0.50, 0.50) is True
        assert should_trigger(0.51, 0.50) is False

    def test_mark_from_quote_midpoint(self) -> None:
        mark = mark_from_quote({"bid": 0.4, "ask": 0.6})
        assert mark == 0.5

    def test_mark_from_quote_ask_only(self) -> None:
        assert mark_from_quote({"bid": 0.0, "ask": 0.6}) == 0.6

    def test_mark_from_quote_none(self) -> None:
        assert mark_from_quote(None) is None
        assert mark_from_quote({"bid": 0.0, "ask": 0.0}) is None

    def test_compute_pnl(self) -> None:
        pnl, pct = compute_pnl(0.8, 0.4, 1)
        assert pnl == -40.0
        assert pct == -50.0

    def test_compute_pnl_scales_with_contracts(self) -> None:
        pnl, _ = compute_pnl(0.8, 0.4, 2)
        assert pnl == -80.0

    def test_compute_pnl_zero_entry(self) -> None:
        pnl, pct = compute_pnl(None, 0.4, 1)
        assert pnl == 0.0
        assert pct == 0.0


class TestMarketHours:
    def test_in_hours(self) -> None:
        assert in_market_hours(datetime(2026, 9, 3, 10, 0, tzinfo=ET_TZ)) is True
        assert in_market_hours(datetime(2026, 9, 3, 15, 0, tzinfo=ET_TZ)) is True

    def test_out_of_hours(self) -> None:
        assert in_market_hours(datetime(2026, 9, 3, 9, 0, tzinfo=ET_TZ)) is False
        assert in_market_hours(datetime(2026, 9, 3, 16, 0, tzinfo=ET_TZ)) is False


class TestLoadExecConfig:
    def test_loads_real_config_yaml(self) -> None:
        exec_config, primary_model = _load_exec_config("config.yaml")
        assert primary_model  # non-empty, whatever the configured primary is
        assert isinstance(exec_config.exit_advisor.enabled, bool)

    def test_missing_file_falls_back_to_defaults(self) -> None:
        exec_config, primary_model = _load_exec_config("/nonexistent/path/config.yaml")
        assert exec_config.exit_advisor.enabled is False
        assert primary_model == "opencode/muse-spark-1.3-contributor-free"

    def test_malformed_yaml_falls_back_to_defaults(self, tmp_path) -> None:
        bad = tmp_path / "config.yaml"
        bad.write_text("not: [valid, yaml: structure")
        exec_config, primary_model = _load_exec_config(str(bad))
        assert exec_config.exit_advisor.enabled is False
        assert primary_model


def _order(status: str, order_id: str = "sell-1", filled_avg_price: str | None = None):
    return type(
        "FakeOrder",
        (),
        {"status": status, "order_id": order_id, "filled_avg_price": filled_avg_price},
    )()


class TestPollFill:
    @pytest.mark.asyncio
    async def test_returns_order_immediately_when_filled(self) -> None:
        client = AsyncMock()
        client.get_order = AsyncMock(return_value=_order("filled", filled_avg_price="0.45"))
        result = await _poll_fill(client, "sell-1")
        assert result is not None
        assert result.status == "filled"
        client.get_order.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_rejected(self) -> None:
        client = AsyncMock()
        client.get_order = AsyncMock(return_value=_order("rejected"))
        result = await _poll_fill(client, "sell-1")
        assert result is None
        client.get_order.assert_awaited_once()  # terminal non-fill -- no need to keep polling

    @pytest.mark.asyncio
    async def test_returns_none_after_exhausting_attempts_while_still_open(self) -> None:
        client = AsyncMock()
        client.get_order = AsyncMock(return_value=_order("new"))
        with patch("src.execution.intraday_monitor.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            result = await _poll_fill(client, "sell-1")
        assert result is None
        assert client.get_order.await_count == 5  # CLOSE_FILL_POLL_ATTEMPTS
        assert mock_sleep.await_count == 5

    @pytest.mark.asyncio
    async def test_returns_none_on_client_error(self) -> None:
        client = AsyncMock()
        client.get_order = AsyncMock(side_effect=RuntimeError("boom"))
        result = await _poll_fill(client, "sell-1")
        assert result is None


class TestWriteCloseAttempt:
    def test_leaves_exit_reason_pending_and_records_attempt(self, tmp_path) -> None:
        data = _audit_data(exit_reason="pending")
        path = tmp_path / "trade-abc123.json"
        path.write_text(json.dumps(data))
        trade = extract_open_trade(data, path)
        assert trade is not None

        attempts = write_close_attempt(trade, "stop_loss", 0.32, "sell-1")
        assert attempts == 1

        on_disk = json.loads(path.read_text())
        assert on_disk["exit_reason"] == "pending"
        assert on_disk["close_attempts"] == 1
        unfilled = [e for e in on_disk["entries"] if e["event_type"] == "close_attempt_unfilled"]
        assert len(unfilled) == 1
        assert unfilled[0]["attempted_exit_reason"] == "stop_loss"
        assert unfilled[0]["attempted_limit_price"] == 0.32

        # A still-pending trade with the same file must still be picked up
        # as open on the next scan -- this is the whole point of leaving
        # exit_reason untouched.
        reloaded = extract_open_trade(on_disk, path)
        assert reloaded is not None

    def test_second_attempt_increments_counter(self, tmp_path) -> None:
        data = _audit_data(exit_reason="pending")
        data["close_attempts"] = 1
        path = tmp_path / "trade-abc123.json"
        path.write_text(json.dumps(data))
        trade = extract_open_trade(data, path)
        assert trade is not None

        attempts = write_close_attempt(trade, "stop_loss", 0.30, "sell-2")
        assert attempts == 2


class TestClosePositionFillConfirmation:
    """Fix (2026-09-11 review): a close must CONFIRM the fill before the
    trade is marked resolved -- an unfilled sell must never orphan an
    open Alpaca position behind a closed-looking audit file."""

    @pytest.mark.asyncio
    async def test_unfilled_close_stays_pending_and_cancels_order(self, tmp_path) -> None:
        data = _audit_data(exit_reason="pending", sl_level=0.5)
        path = tmp_path / "trade-abc123.json"
        path.write_text(json.dumps(data))
        trade = extract_open_trade(data, path)
        assert trade is not None

        client = AsyncMock()
        client.cancel_order = AsyncMock()
        client.submit_order = AsyncMock(return_value=_order("accepted", order_id="sell-1"))
        client.get_order = AsyncMock(return_value=_order("new", order_id="sell-1"))

        with patch("src.execution.intraday_monitor.asyncio.sleep", new=AsyncMock()):
            result = await _close_position(
                trade, client, mark=0.45, exit_reason="stop_loss", limit_mult=0.7, now=None
            )

        assert result is None
        # Both the resting TP (tp-order) and the never-filled sell (sell-1)
        # must be cancelled so neither lingers into the next run.
        cancelled_ids = {call.args[0] for call in client.cancel_order.await_args_list}
        assert "tp-order" in cancelled_ids
        assert "sell-1" in cancelled_ids

        on_disk = json.loads(path.read_text())
        assert on_disk["exit_reason"] == "pending"
        assert on_disk["close_attempts"] == 1

    @pytest.mark.asyncio
    async def test_filled_close_writes_terminal_result(self, tmp_path) -> None:
        data = _audit_data(exit_reason="pending", sl_level=0.5, entry_price=1.0)
        path = tmp_path / "trade-abc123.json"
        path.write_text(json.dumps(data))
        trade = extract_open_trade(data, path)
        assert trade is not None

        client = AsyncMock()
        client.cancel_order = AsyncMock()
        client.submit_order = AsyncMock(return_value=_order("accepted", order_id="sell-1"))
        client.get_order = AsyncMock(
            return_value=_order("filled", order_id="sell-1", filled_avg_price="0.44")
        )

        result = await _close_position(
            trade, client, mark=0.45, exit_reason="stop_loss", limit_mult=0.7, now=None
        )

        assert result is not None
        assert result["exit_reason"] == "stop_loss"
        assert result["exit_price"] == 0.44  # actual fill price, not the quoted mark

        on_disk = json.loads(path.read_text())
        assert on_disk["exit_reason"] == "stop_loss"
        assert on_disk["exit_price"] == 0.44

    @pytest.mark.asyncio
    async def test_retry_escalates_limit_price(self, tmp_path) -> None:
        data = _audit_data(exit_reason="pending", sl_level=0.5)
        data["close_attempts"] = 2  # this is the 3rd attempt
        path = tmp_path / "trade-abc123.json"
        path.write_text(json.dumps(data))
        trade = extract_open_trade(data, path)
        assert trade is not None

        client = AsyncMock()
        client.cancel_order = AsyncMock()
        client.submit_order = AsyncMock(return_value=_order("accepted", order_id="sell-1"))
        client.get_order = AsyncMock(return_value=_order("new", order_id="sell-1"))

        with patch("src.execution.intraday_monitor.asyncio.sleep", new=AsyncMock()):
            await _close_position(
                trade, client, mark=1.0, exit_reason="stop_loss", limit_mult=0.7, now=None
            )

        submitted = client.submit_order.await_args_list[0].args[0]
        # base limit_mult 0.7 - 2 prior attempts * 0.1 = 0.5 (the floor)
        assert submitted["limit_price"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_monitor_trade_returns_none_and_stays_open_when_sl_close_unfilled(
        self, tmp_path
    ) -> None:
        data = _audit_data(exit_reason="pending", sl_level=0.5)
        path = tmp_path / "trade-abc123.json"
        path.write_text(json.dumps(data))
        trade = extract_open_trade(data, path)
        assert trade is not None

        client = AsyncMock()
        client.get_option_quote = AsyncMock(
            return_value={"bid": 0.4, "ask": 0.45}
        )  # mark 0.425 <= sl 0.5
        client.cancel_order = AsyncMock()
        client.submit_order = AsyncMock(return_value=_order("accepted", order_id="sell-1"))
        client.get_order = AsyncMock(return_value=_order("new", order_id="sell-1"))

        with patch("src.execution.intraday_monitor.asyncio.sleep", new=AsyncMock()):
            outcome = await monitor_trade(trade, client, now=None)

        assert outcome is None
        on_disk = json.loads(path.read_text())
        assert on_disk["exit_reason"] == "pending"
        assert on_disk["close_attempts"] == 1

        # A fresh scan of the log directory still finds this trade open --
        # the next cron run will retry it.
        reloaded = extract_open_trade(on_disk, path)
        assert reloaded is not None
