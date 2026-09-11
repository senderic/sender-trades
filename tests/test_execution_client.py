"""Unit tests for AlpacaBrokerClient using mock SDK."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

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

    def test_limit_priced_off_live_ask_with_offset(self) -> None:
        client = AlpacaBrokerClient("key", "secret")
        rec = _rec()
        quote = {"symbol": "SPY250728C00600000", "bid": 0.48, "ask": 0.52}
        order = client.build_entry_order(rec, quote=quote)
        offset = 5.0 / 100.0
        expected = round(0.52 * (1 + offset), 2)
        assert order["limit_price"] == expected
        assert order["limit_price"] > 0.52

    def test_limit_uses_passed_limit_price_over_quote(self) -> None:
        client = AlpacaBrokerClient("key", "secret")
        rec = _rec()
        quote = {"symbol": "SPY250728C00600000", "bid": 0.48, "ask": 0.52}
        order = client.build_entry_order(rec, limit_price=0.55, quote=quote)
        assert order["limit_price"] == 0.55

    def test_limit_falls_back_to_delta_estimate_without_quote(self) -> None:
        client = AlpacaBrokerClient("key", "secret")
        rec = _rec()
        order = client.build_entry_order(rec)
        assert order["limit_price"] >= 1.0

    def test_limit_priced_from_spot_when_no_option_quote(self) -> None:
        client = AlpacaBrokerClient("key", "secret")
        rec = _rec()
        spot = 600.0
        order = client.build_entry_order(rec, spot=spot)
        offset = 5.0 / 100.0
        intrinsic = max(0.0, spot - rec.target_strike)
        expected = round((intrinsic + spot * 0.005) * (1 + offset), 2)
        assert order["limit_price"] == expected
        assert order["limit_price"] > 0

    def test_spot_estimate_is_generous_ceiling(self) -> None:
        client = AlpacaBrokerClient("key", "secret")
        rec = _rec()
        order = client.build_entry_order(rec, spot=600.0)
        assert order["limit_price"] >= 600.0 * 0.005

    def test_spot_estimate_for_itm_call_adds_intrinsic(self) -> None:
        client = AlpacaBrokerClient("key", "secret")
        rec = _rec()
        order = client.build_entry_order(rec, spot=610.0)
        offset = 5.0 / 100.0
        intrinsic = 610.0 - rec.target_strike
        expected = round((intrinsic + 610.0 * 0.005) * (1 + offset), 2)
        assert order["limit_price"] == expected


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
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "option_contracts": [
                {
                    "symbol": "SPY250728C00600000",
                    "strike_price": "600",
                    "type": "call",
                    "expiration_date": "2026-07-28",
                }
            ]
        }
        mock_http = AsyncMock()
        mock_http.get.return_value = mock_resp
        client._http = mock_http

        result = await client.get_option_chain("SPY", "2026-07-28")
        assert len(result) == 1
        assert result[0]["symbol"] == "SPY250728C00600000"

    @pytest.mark.asyncio
    async def test_get_option_quote(self) -> None:
        client = AlpacaBrokerClient("key", "secret", paper=True)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "snapshots": {
                "SPY250728C00600000": {
                    "latestQuote": {"bp": "0.48", "ap": "0.52", "bs": 100, "as": 100},
                    "latestTrade": {},
                }
            }
        }
        mock_http = AsyncMock()
        mock_http.get.return_value = mock_resp
        client._data_http = mock_http

        result = await client.get_option_quote("SPY250728C00600000")
        assert result is not None
        assert result["bid"] == 0.48
        assert result["ask"] == 0.52

    @pytest.mark.asyncio
    async def test_get_underlying_quote(self) -> None:
        client = AlpacaBrokerClient("key", "secret", paper=True)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "SPY": {
                "latestQuote": {"bp": "599.0", "ap": "600.1", "bs": 100, "as": 100},
                "latestTrade": {"p": "600.0", "s": 100, "t": "2026-08-05T09:28:00Z"},
            }
        }
        mock_http = AsyncMock()
        mock_http.get.return_value = mock_resp
        client._data_http = mock_http

        result = await client.get_underlying_quote("SPY")
        assert result is not None
        assert result["symbol"] == "SPY"
        assert result["last"] == 600.0
        assert result["bid"] == 599.0
        assert result["ask"] == 600.1

    @pytest.mark.asyncio
    async def test_get_underlying_quote_missing_symbol_returns_none(self) -> None:
        client = AlpacaBrokerClient("key", "secret", paper=True)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"SPY": {}}
        mock_http = AsyncMock()
        mock_http.get.return_value = mock_resp
        client._data_http = mock_http

        result = await client.get_underlying_quote("SPY")
        assert result is None


def _mock_bar(**overrides: Any) -> MagicMock:
    bar = MagicMock()
    bar.open = overrides.get("open", 700.0)
    bar.high = overrides.get("high", 701.0)
    bar.low = overrides.get("low", 699.0)
    bar.close = overrides.get("close", 700.5)
    bar.volume = overrides.get("volume", 500.0)
    bar.vwap = overrides.get("vwap", 700.4)
    bar.trade_count = overrides.get("trade_count", 3.0)
    bar.timestamp = overrides.get("timestamp", datetime(2026, 9, 11, 13, 28, tzinfo=UTC))
    return bar


class TestGetMinuteBars:
    @pytest.mark.asyncio
    async def test_returns_bar_dicts_oldest_first(self) -> None:
        client = AlpacaBrokerClient("key", "secret", paper=True)
        mock_historical = MagicMock()
        mock_bars = MagicMock()
        mock_bars.data = {"QQQ": [_mock_bar(close=700.0), _mock_bar(close=701.0)]}
        mock_historical.get_stock_bars.return_value = mock_bars
        client._historical_data = mock_historical

        rows = await client.get_minute_bars(
            "QQQ",
            datetime(2026, 9, 11, 8, 0, tzinfo=UTC),
            datetime(2026, 9, 11, 13, 28, tzinfo=UTC),
        )
        assert len(rows) == 2
        assert rows[0]["close"] == 700.0
        assert rows[1]["close"] == 701.0
        assert rows[0]["volume"] == 500.0

    @pytest.mark.asyncio
    async def test_uses_iex_feed_by_default(self) -> None:
        from alpaca.data.enums import DataFeed

        client = AlpacaBrokerClient("key", "secret", paper=True)
        mock_historical = MagicMock()
        mock_bars = MagicMock()
        mock_bars.data = {"QQQ": []}
        mock_historical.get_stock_bars.return_value = mock_bars
        client._historical_data = mock_historical

        await client.get_minute_bars(
            "QQQ",
            datetime(2026, 9, 11, 8, 0, tzinfo=UTC),
            datetime(2026, 9, 11, 13, 28, tzinfo=UTC),
        )
        request = mock_historical.get_stock_bars.call_args[0][0]
        assert request.feed == DataFeed.IEX

    @pytest.mark.asyncio
    async def test_missing_symbol_in_response_returns_empty(self) -> None:
        client = AlpacaBrokerClient("key", "secret", paper=True)
        mock_historical = MagicMock()
        mock_bars = MagicMock()
        mock_bars.data = {}
        mock_historical.get_stock_bars.return_value = mock_bars
        client._historical_data = mock_historical

        rows = await client.get_minute_bars(
            "QQQ",
            datetime(2026, 9, 11, 8, 0, tzinfo=UTC),
            datetime(2026, 9, 11, 13, 28, tzinfo=UTC),
        )
        assert rows == []

    @pytest.mark.asyncio
    async def test_sdk_exception_returns_empty_list(self) -> None:
        client = AlpacaBrokerClient("key", "secret", paper=True)
        mock_historical = MagicMock()
        mock_historical.get_stock_bars.side_effect = RuntimeError("network error")
        client._historical_data = mock_historical

        rows = await client.get_minute_bars(
            "QQQ",
            datetime(2026, 9, 11, 8, 0, tzinfo=UTC),
            datetime(2026, 9, 11, 13, 28, tzinfo=UTC),
        )
        assert rows == []
