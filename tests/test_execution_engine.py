"""Integration tests for ExecutionEngine with mocked client methods."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.execution.client import AlpacaBrokerClient
from src.execution.engine import ExecutionEngine
from src.execution.models import ExecutionConfig, OrderResult, TenacityConfig
from src.models.recommendation import Direction, PositionIntent, TradeRecommendation


def _rec(**overrides: Any) -> TradeRecommendation:
    kw: dict[str, Any] = {
        "correlation_id": "test-engine-corr",
        "strategy_label": "momentum",
        "asset": "SPY",
        "direction": Direction.CALL,
        "confidence": 0.75,
        "target_strike": 600.0,
        "contracts": 1,
        "order_type": "market",
        "position_intent": PositionIntent.BUY_TO_OPEN,
        "rationale": {"source": "test"},
        "expires_at": "2026-07-28",
        "must_close_before": "15:30",
    }
    kw.update(overrides)
    return TradeRecommendation(**kw)


ORD_ID = "ab12cd34-ab12-cd34-ab12-cd34ab12cd34"


def _make_result(**overrides: Any) -> OrderResult:
    defaults: dict[str, Any] = {
        "order_id": ORD_ID,
        "status": "new",
        "symbol": "SPY250728C00600000",
        "side": "buy",
        "order_type": "market",
        "qty": "1",
        "filled_qty": "0",
        "filled_avg_price": None,
    }
    defaults.update(overrides)
    return OrderResult(**defaults)


def _make_client() -> AlpacaBrokerClient:
    config = ExecutionConfig(tenacity=TenacityConfig(min_wait_sec=0.01, max_wait_sec=0.05))
    config.entry.entry_window_minutes = 1
    return AlpacaBrokerClient("key", "secret", paper=True, config=config)


def _make_engine(client: AlpacaBrokerClient, tmp_path: Path) -> ExecutionEngine:
    config = ExecutionConfig(tenacity=TenacityConfig(min_wait_sec=0.01, max_wait_sec=0.05))
    config.entry.entry_window_minutes = 1
    config.exit_strategy.time_deadline_est = "23:59"
    engine = ExecutionEngine(client, config, log_dir=str(tmp_path))
    engine._monitor_interval = 0.01
    engine._wait_for_option_open = False
    return engine


class TestExecutionEngineEntryRejected:
    @pytest.mark.asyncio
    async def test_rejected_yields_rejected_exit(self, tmp_path: Path) -> None:
        client = _make_client()
        client.submit_order = AsyncMock(return_value=_make_result(status="rejected"))
        engine = _make_engine(client, tmp_path)

        result = await engine.execute(_rec(), "test-corr")
        assert result["exit_reason"] == "rejected"
        assert result["final_pnl"] == 0.0


class TestExecutionEngineAwaitOptionQuote:
    @pytest.mark.asyncio
    async def test_returns_quote_when_ask_present_immediately(self, tmp_path: Path) -> None:
        client = _make_client()
        client.get_option_quote = AsyncMock(
            return_value={"symbol": "SPY250728C00600000", "bid": 0.48, "ask": 0.52}
        )
        engine = _make_engine(client, tmp_path)

        quote = await engine._await_option_quote("SPY250728C00600000")
        assert quote is not None
        assert quote["ask"] == 0.52
        assert client.get_option_quote.await_count == 1

    @pytest.mark.asyncio
    async def test_retries_after_open_when_quote_missing(self, tmp_path: Path) -> None:
        client = _make_client()
        client.get_option_quote = AsyncMock(
            side_effect=[
                None,
                {"symbol": "SPY250728C00600000", "bid": 0.48, "ask": 0.52},
            ]
        )
        engine = _make_engine(client, tmp_path)
        engine._wait_for_option_open = True
        engine._option_open_wait_cap_sec = 60.0
        engine._option_open_buffer_sec = 0.0

        from unittest.mock import patch

        from src.timezone import ET_TZ

        fixed_now = datetime(2026, 8, 5, 9, 29, 40, tzinfo=ET_TZ)
        with (
            patch("src.execution.engine.datetime") as mock_dt,
            patch("src.execution.engine.asyncio.sleep", new=AsyncMock()) as mock_sleep,
        ):
            mock_dt.now.side_effect = lambda tz=None: fixed_now if tz else fixed_now.astimezone(UTC)
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
            quote = await engine._await_option_quote("SPY250728C00600000")

        assert quote is not None
        assert quote["ask"] == 0.52
        assert client.get_option_quote.await_count == 2
        assert mock_sleep.await_count == 1

    @pytest.mark.asyncio
    async def test_returns_none_when_still_missing_after_open(self, tmp_path: Path) -> None:
        client = _make_client()
        client.get_option_quote = AsyncMock(side_effect=[None, None])
        engine = _make_engine(client, tmp_path)
        engine._wait_for_option_open = True
        engine._option_open_wait_cap_sec = 60.0
        engine._option_open_buffer_sec = 0.0

        from unittest.mock import patch

        from src.timezone import ET_TZ

        fixed_now = datetime(2026, 8, 5, 9, 29, 40, tzinfo=ET_TZ)
        with (
            patch("src.execution.engine.datetime") as mock_dt,
            patch("src.execution.engine.asyncio.sleep", new=AsyncMock()) as mock_sleep,
        ):
            mock_dt.now.side_effect = lambda tz=None: fixed_now if tz else fixed_now.astimezone(UTC)
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
            quote = await engine._await_option_quote("SPY250728C00600000")

        assert quote is None
        assert client.get_option_quote.await_count == 2
        assert mock_sleep.await_count == 1


class TestExecutionEngineEntryExpired:
    @pytest.mark.asyncio
    async def test_expired_when_wait_for_fill_returns_none(self, tmp_path: Path) -> None:
        client = _make_client()
        client.submit_order = AsyncMock(return_value=_make_result(status="new"))
        client.cancel_order = AsyncMock(return_value=_make_result(status="canceled"))
        engine = _make_engine(client, tmp_path)
        engine._wait_for_fill = AsyncMock(return_value=None)

        result = await engine.execute(_rec(), "test-corr")
        assert result["exit_reason"] == "expired"
        assert client.cancel_order.called


class TestExecutionEngineEntryFilled:
    @pytest.mark.asyncio
    async def test_filled_transitions_correctly(self, tmp_path: Path) -> None:
        client = _make_client()
        fill_result = _make_result(status="filled", filled_qty="1", filled_avg_price="0.50")
        client.submit_order = AsyncMock(return_value=_make_result(status="new"))
        client.cancel_order = AsyncMock()
        client.get_option_quote = AsyncMock(return_value=None)
        client.get_underlying_quote = AsyncMock(
            return_value={"symbol": "SPY", "last": 600.0, "bid": 599.0, "ask": 600.1}
        )
        engine = _make_engine(client, tmp_path)
        engine._wait_for_fill = AsyncMock(return_value=fill_result)
        engine._monitor_exits = AsyncMock(
            return_value={"trade_id": "abc", "exit_reason": "force_close", "final_pnl": 0.0}
        )

        result = await engine.execute(_rec(), "test-corr")
        assert result is not None
        assert result["trade_id"] is not None


class TestExecutionEngineAuditTrail:
    @pytest.mark.asyncio
    async def test_writes_audit_file(self, tmp_path: Path) -> None:
        client = _make_client()
        client.submit_order = AsyncMock(return_value=_make_result(status="rejected"))
        engine = _make_engine(client, tmp_path)

        result = await engine.execute(_rec(), "test-corr")
        trade_id = result["trade_id"]
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")
        audit_path = tmp_path / date_str / f"trade-{trade_id}.json"
        assert audit_path.exists()
        data = json.loads(audit_path.read_text())
        assert data["trade_id"] == trade_id
        assert data["correlation_id"] == "test-corr"
        assert data["exit_reason"] == "rejected"


class TestExecutionEngineExceptionHandling:
    @pytest.mark.asyncio
    async def test_client_error_returns_error_exit(self, tmp_path: Path) -> None:
        client = _make_client()
        client.submit_order = AsyncMock(side_effect=RuntimeError("API unavailable"))
        engine = _make_engine(client, tmp_path)

        result = await engine.execute(_rec(), "test-corr")
        assert result["exit_reason"] == "error"
        assert result["final_pnl"] == 0.0

    @pytest.mark.asyncio
    async def test_partially_filled_handled(self, tmp_path: Path) -> None:
        client = _make_client()
        client.submit_order = AsyncMock(return_value=_make_result(status="new"))
        client.cancel_order = AsyncMock()
        client.get_option_quote = AsyncMock(return_value=None)
        client.get_underlying_quote = AsyncMock(
            return_value={"symbol": "SPY", "last": 600.0, "bid": 599.0, "ask": 600.1}
        )
        engine = _make_engine(client, tmp_path)
        partial = _make_result(status="partially_filled", filled_qty="1", filled_avg_price="0.50")
        engine._wait_for_fill = AsyncMock(return_value=partial)
        engine._monitor_exits = AsyncMock(
            return_value={"trade_id": "abc", "exit_reason": "stop_loss", "final_pnl": -25.0}
        )

        result = await engine.execute(_rec(), "test-corr")
        assert result is not None
        assert result["trade_id"] is not None
