"""Unit tests for AlpacaBrokerClient using mock SDK."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.execution.client import AlpacaBrokerClient
from src.models.recommendation import Direction, PositionIntent, TradeRecommendation


def _rec() -> TradeRecommendation:
    return TradeRecommendation(
        correlation_id="test",
        strategy_label="momentum",
        asset="SPY",
        direction=Direction.CALL,
        confidence=0.75,
        target_strike=600.0,
        contracts=2,
        order_type="limit",
        position_intent=PositionIntent.BUY_TO_OPEN,
        rationale={},
        expires_at="2026-07-28",
        must_close_before="15:30",
    )


def _mock_order(**overrides: Any) -> MagicMock:
    now = datetime.now(UTC).isoformat()
    defaults = {
        "id": "ab12cd34-ab12-cd34-ab12-cd34ab12cd34",
        "client_order_id": "",
        "created_at": now,
        "updated_at": now,
        "submitted_at": now,
        "filled_at": None,
        "expired_at": None,
        "canceled_at": None,
        "failed_at": None,
        "replaced_at": None,
        "replaced_by": None,
        "replaces": None,
        "asset_id": "abc-def",
        "symbol": "SPY250728C00600000",
        "asset_class": "us_option",
        "notional": None,
        "qty": "2",
        "filled_qty": "0",
        "filled_avg_price": None,
        "order_class": "simple",
        "order_type": "limit",
        "type": "limit",
        "side": "buy",
        "time_in_force": "day",
        "limit_price": "0.50",
        "stop_price": None,
        "status": "new",
        "extended_hours": False,
        "legs": None,
        "trail_percent": None,
        "trail_price": None,
        "hwm": None,
        "subtag": None,
        "source": None,
    }
    defaults.update(overrides)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    mock.model_dump.return_value = defaults
    return mock


class TestBuildEntryOrder:
    def test_builds_entry_order_from_recommendation(self) -> None:
        client = AlpacaBrokerClient("key", "secret")
        rec = _rec()
        order = client.build_entry_order(rec, limit_price=0.55)
        assert order["symbol"].startswith("SPY")
        assert "C" in order["symbol"]
        assert order["qty"] == 2
        assert order["side"] == "buy"
        assert order["type"] == "limit"
        assert order["limit_price"] == 0.55
        assert order["time_in_force"] == "day"

    def test_builds_put_order(self) -> None:
        client = AlpacaBrokerClient("key", "secret")
        rec = _rec()
        rec.direction = Direction.PUT
        order = client.build_entry_order(rec)
        assert "P" in order["symbol"]


class TestAlpacaBrokerClientSubmit:
    @pytest.mark.asyncio
    async def test_submit_limit_order(self) -> None:
        client = AlpacaBrokerClient("key", "secret", paper=True)
        mock_trading = MagicMock()
        mock_trading.submit_order.return_value = _mock_order(status="new")
        client._trading = mock_trading

        result = await client.submit_order(
            {
                "symbol": "SPY250728C00600000",
                "qty": 2,
                "side": "buy",
                "type": "limit",
                "limit_price": 0.50,
                "time_in_force": "day",
            }
        )
        assert result.status == "new"
        assert result.symbol == "SPY250728C00600000"

    @pytest.mark.asyncio
    async def test_submit_market_order(self) -> None:
        client = AlpacaBrokerClient("key", "secret", paper=True)
        mock_trading = MagicMock()
        mock_trading.submit_order.return_value = _mock_order(status="new", type="market")
        client._trading = mock_trading

        result = await client.submit_order(
            {
                "symbol": "SPY250728C00600000",
                "qty": 1,
                "side": "sell",
                "type": "market",
                "time_in_force": "day",
            }
        )
        assert result.status == "new"

    @pytest.mark.asyncio
    async def test_get_order(self) -> None:
        client = AlpacaBrokerClient("key", "secret", paper=True)
        mock_trading = MagicMock()
        mock_trading.get_order_by_id.return_value = _mock_order(
            status="filled", filled_qty="2", filled_avg_price="0.52"
        )
        client._trading = mock_trading

        result = await client.get_order("ab12cd34-ab12-cd34-ab12-cd34ab12cd34")
        assert result.status == "filled"
        assert result.filled_qty == "2"
        assert result.filled_avg_price == "0.52"

    @pytest.mark.asyncio
    async def test_cancel_order(self) -> None:
        client = AlpacaBrokerClient("key", "secret", paper=True)
        mock_trading = MagicMock()
        mock_trading.get_order_by_id.return_value = _mock_order(status="canceled")
        client._trading = mock_trading

        result = await client.cancel_order("ab12cd34-ab12-cd34-ab12-cd34ab12cd34")
        assert result.status == "canceled"

    @pytest.mark.asyncio
    async def test_get_option_chain(self) -> None:
        client = AlpacaBrokerClient("key", "secret", paper=True)
        mock_chain = MagicMock()
        mock_contract = MagicMock()
        mock_contract.symbol = "SPY250728C00600000"
        mock_contract.strike_price = "600"
        mock_contract.type = "call"
        mock_contract.expiration_date = "2026-07-28"
        mock_chain.option_contracts = [mock_contract]
        mock_data = MagicMock()
        mock_data.get_option_chain.return_value = mock_chain
        client._data = mock_data

        result = await client.get_option_chain("SPY", "2026-07-28")
        assert len(result) == 1
        assert result[0]["symbol"] == "SPY250728C00600000"

    @pytest.mark.asyncio
    async def test_get_option_quote(self) -> None:
        client = AlpacaBrokerClient("key", "secret", paper=True)
        mock_quote = MagicMock()
        mock_quote.bid_price = "0.48"
        mock_quote.ask_price = "0.52"
        mock_quote.bid_size = 100
        mock_quote.ask_size = 100
        mock_data = MagicMock()
        mock_data.get_option_latest_quote.return_value = {"SPY250728C00600000": mock_quote}
        client._data = mock_data

        result = await client.get_option_quote("SPY250728C00600000")
        assert result is not None
        assert result["bid"] == 0.48
        assert result["ask"] == 0.52
