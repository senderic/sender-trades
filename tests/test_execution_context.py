"""Unit tests for TradeContext audit trail."""

from __future__ import annotations

import json
from pathlib import Path

from src.execution.context import TradeContext
from src.models.recommendation import Direction, PositionIntent, TradeRecommendation


def _make_rec() -> TradeRecommendation:
    return TradeRecommendation(
        correlation_id="corr-abc",
        strategy_label="momentum",
        asset="SPY",
        direction=Direction.CALL,
        confidence=0.75,
        target_strike=600.0,
        contracts=2,
        order_type="market",
        position_intent=PositionIntent.BUY_TO_OPEN,
        rationale={"source": "test"},
        expires_at="2026-07-28",
        must_close_before="15:30",
    )


class TestTradeContextEntryRecording:
    def test_records_entry_to_file(self, tmp_path: Path) -> None:
        rec = _make_rec()
        ctx = TradeContext("trade-test-1", "corr-abc", rec, log_dir=str(tmp_path))
        ctx.record_entry("order_submitted", occ_symbol="SPY250728C00600000")

        assert len(ctx.entries) == 1
        assert ctx.entries[0]["event_type"] == "order_submitted"
        assert ctx.entries[0]["occ_symbol"] == "SPY250728C00600000"
        assert ctx.entries[0]["correlation_id"] == "corr-abc"
        assert ctx.entries[0]["trade_id"] == "trade-test-1"
        assert "timestamp" in ctx.entries[0]

        assert ctx.audit_path.exists()

    def test_pending_file_is_single_valid_json_document(self, tmp_path: Path) -> None:
        """The audit file must be a single parseable JSON object after every
        `record_entry` call, not just after `finalize()` -- a prior
        append-only writer left several concatenated JSON objects in the
        file whenever a trade never reached `finalize()` (crash, kill,
        etc.), which plain `json.load` cannot parse."""
        rec = _make_rec()
        ctx = TradeContext("trade-pending", "corr-pending", rec, log_dir=str(tmp_path))
        ctx.record_entry("entry_submitted", occ_symbol="SPY250728C00600000")
        ctx.record_entry("entry_filled", fill_price=0.50)

        with open(ctx.audit_path) as f:
            data = json.load(f)  # raises if more than one object is concatenated

        assert data["exit_reason"] == "pending"
        assert data["trade_id"] == "trade-pending"
        assert [e["event_type"] for e in data["entries"]] == [
            "entry_submitted",
            "entry_filled",
        ]
        # No leftover temp file from the atomic-write dance.
        assert not ctx.audit_path.with_suffix(".tmp").exists()

    def test_multiple_entries(self, tmp_path: Path) -> None:
        rec = _make_rec()
        ctx = TradeContext("trade-xyz", "corr-xyz", rec, log_dir=str(tmp_path))
        ctx.record_entry("entry_submitted", symbol="SPY")
        ctx.record_entry("entry_filled", fill_price=0.50)
        ctx.record_entry("exits_placed", tp=1.00, sl=0.25)
        assert len(ctx.entries) == 3


class TestTradeContextMonitoring:
    def test_records_snapshot(self, tmp_path: Path) -> None:
        rec = _make_rec()
        ctx = TradeContext("trade-m", "corr-m", rec, log_dir=str(tmp_path))
        ctx.record_monitoring_snapshot(
            current_price=0.75,
            current_pnl_pct=50.0,
            tp_level=1.00,
            sl_level=0.25,
            trailing_active=True,
            trail_level=0.65,
        )
        entry = ctx.entries[0]
        assert entry["event_type"] == "monitoring_snapshot"
        assert entry["current_price"] == 0.75
        assert entry["current_pnl_pct"] == 50.0
        assert entry["trailing_active"] is True
        assert entry["trail_level"] == 0.65

    def test_records_adjustment(self, tmp_path: Path) -> None:
        rec = _make_rec()
        ctx = TradeContext("trade-adj", "corr-adj", rec, log_dir=str(tmp_path))
        ctx.record_adjustment("trail_raised", old_level=0.60, new_level=0.65)
        entry = ctx.entries[0]
        assert entry["event_type"] == "adjustment"
        assert entry["reason"] == "trail_raised"
        assert entry["old_level"] == 0.60
        assert entry["new_level"] == 0.65


class TestTradeContextFinalize:
    def test_finalize_writes_complete_record(self, tmp_path: Path) -> None:
        rec = _make_rec()
        ctx = TradeContext("trade-fin", "corr-fin", rec, log_dir=str(tmp_path))
        ctx.record_entry("entry_filled", fill_price=0.50)

        summary = ctx.finalize(
            exit_reason="take_profit",
            exit_price=1.00,
            final_pnl=100.0,
            final_pnl_pct=100.0,
            lifecycle_events=[{"state_from": "filled", "state_to": "closed"}],
        )
        assert summary["trade_id"] == "trade-fin"
        assert summary["correlation_id"] == "corr-fin"
        assert summary["exit_reason"] == "take_profit"
        assert summary["final_pnl"] == 100.0
        assert summary["final_pnl_pct"] == 100.0
        assert summary["asset"] == "SPY"
        assert summary["direction"] == "CALL"
        assert summary["contracts"] == 2
        assert summary["duration_seconds"] >= 0
        assert "started_at" in summary
        assert "ended_at" in summary
        assert "events" in summary

    def test_finalize_includes_recommendation(self, tmp_path: Path) -> None:
        rec = _make_rec()
        ctx = TradeContext("trade-rec", "corr-rec", rec, log_dir=str(tmp_path))
        summary = ctx.finalize(
            exit_reason="stop_loss",
            exit_price=0.25,
            final_pnl=-50.0,
            final_pnl_pct=-50.0,
        )
        assert "recommendation" in summary
        assert summary["recommendation"]["asset"] == "SPY"

    def test_finalize_writes_to_disk(self, tmp_path: Path) -> None:
        rec = _make_rec()
        ctx = TradeContext("trade-disk", "corr-disk", rec, log_dir=str(tmp_path))
        ctx.finalize(
            exit_reason="force_close",
            exit_price=0.30,
            final_pnl=-40.0,
            final_pnl_pct=-40.0,
        )

        audit_path = ctx.audit_path
        assert audit_path.exists()
        with open(audit_path) as f:
            data = json.load(f)
        assert data["exit_reason"] == "force_close"
        assert data["trade_id"] == "trade-disk"

    def test_finalize_after_pending_entries_leaves_one_valid_document(
        self, tmp_path: Path
    ) -> None:
        rec = _make_rec()
        ctx = TradeContext("trade-seq", "corr-seq", rec, log_dir=str(tmp_path))
        ctx.record_entry("entry_submitted", occ_symbol="SPY250728C00600000")
        ctx.record_entry("entry_filled", fill_price=0.50)
        ctx.finalize(
            exit_reason="take_profit",
            exit_price=1.00,
            final_pnl=100.0,
            final_pnl_pct=100.0,
        )

        raw = ctx.audit_path.read_text()
        data = json.loads(raw)  # raises on concatenated/extra data
        assert data["exit_reason"] == "take_profit"
        assert not ctx.audit_path.with_suffix(".tmp").exists()


class TestTradeContextLinks:
    def test_correlation_id_links_pipeline_decision(self, tmp_path: Path) -> None:
        rec = _make_rec()
        ctx = TradeContext("trade-link", "corr-pipeline-abc", rec, log_dir=str(tmp_path))
        ctx.record_entry("entry_submitted")
        assert ctx.entries[0]["correlation_id"] == "corr-pipeline-abc"
        assert ctx.entries[0]["trade_id"] == "trade-link"

    def test_error_reporting(self, tmp_path: Path) -> None:
        rec = _make_rec()
        ctx = TradeContext("trade-err", "corr-err", rec, log_dir=str(tmp_path))
        ctx.record_entry("execution_error", error="API timeout")
        assert ctx.entries[0]["event_type"] == "execution_error"
        assert ctx.entries[0]["error"] == "API timeout"
