"""Unit tests for the intraday stop-loss monitor."""

from __future__ import annotations

import json
from datetime import datetime

from src.execution.intraday_monitor import (
    compute_pnl,
    extract_open_trade,
    find_open_trades,
    in_market_hours,
    load_audit_file,
    mark_from_quote,
    should_trigger,
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
