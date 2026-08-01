"""Integration tests for candle providers.

These tests verify that each provider correctly fetches and normalises
candle data.  When required API keys are available, they test against
live APIs.  When keys are missing (e.g. on CI), they use fixture data
injected at the HTTP client boundary so the full parsing/conversion
pipeline is still exercised.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from unittest.mock import AsyncMock, patch

import pytest

from src.ingestion.candle_providers import (
    AlphaVantageProvider,
    FinnhubProvider,
    YahooFinanceProvider,
    build_candle_chain,
)
from src.timezone import ET_TZ

KNOWN_DATE = date(2026, 7, 17)
_TS = int(datetime(2026, 7, 17, tzinfo=ET_TZ).timestamp())


def _alphavantage_daily_json(symbol: str, date_str: str) -> dict:
    """Well-formed Alpha Vantage TIME_SERIES_DAILY response."""
    return {
        "Time Series (Daily)": {
            date_str: {
                "1. open": "745.00",
                "2. high": "748.50",
                "3. low": "740.00",
                "4. close": "742.00",
                "5. volume": "45000000",
            }
        }
    }


def _finnhub_candle_json() -> dict:
    """Well-formed Finnhub /stock/candle response."""
    return {
        "c": [742.00],
        "h": [748.50],
        "l": [740.00],
        "o": [745.00],
        "s": "ok",
        "t": [_TS],
        "v": [45_000_000],
    }


# ── Yahoo Finance (no API key needed) ────────────────────────────────


class TestYahooFinanceIntegration:
    @pytest.mark.asyncio
    async def test_daily_live(self):
        provider = YahooFinanceProvider(timeout=30)
        result = await provider.fetch_daily_candle("SPY", KNOWN_DATE)
        assert result is not None, "Yahoo Finance should return data for SPY on a trading day"
        assert result["s"] == "ok"
        assert len(result["o"]) == 1
        assert result["o"][0] > 0
        assert result["h"][0] >= result["l"][0]

    @pytest.mark.asyncio
    async def test_intraday_live(self):
        provider = YahooFinanceProvider(timeout=30)
        result = await provider.fetch_intraday_candles("SPY", KNOWN_DATE)
        assert result is not None
        assert len(result) > 0
        assert result[0]["open"] > 0
        assert result[0]["high"] >= result[0]["low"]

    @pytest.mark.asyncio
    async def test_daily_qqq(self):
        provider = YahooFinanceProvider(timeout=30)
        result = await provider.fetch_daily_candle("QQQ", KNOWN_DATE)
        assert result is not None
        assert result["s"] == "ok"
        assert result["o"][0] > 0


# ── Alpha Vantage (mock when no key) ────────────────────────────────


class TestAlphaVantageIntegration:
    @pytest.mark.asyncio
    async def test_daily(self):
        key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        if key:
            provider = AlphaVantageProvider(api_key=key)
            result = await provider.fetch_daily_candle("SPY", KNOWN_DATE)
        else:
            mock = AsyncMock()
            mock.return_value = _alphavantage_daily_json("SPY", KNOWN_DATE.isoformat())
            with patch.object(AlphaVantageProvider, "_get", mock):
                provider = AlphaVantageProvider(api_key="ci-mock-key")
                result = await provider.fetch_daily_candle("SPY", KNOWN_DATE)

        assert result is not None
        assert result["s"] == "ok"
        assert result["o"][0] > 0


# ── Finnhub (mock when no key) ──────────────────────────────────────


def _mock_httpx_get(return_json: dict):
    """Return an async mock for ``httpx.AsyncClient.get``."""

    class _MockResponse:
        status_code = 200

        @staticmethod
        def raise_for_status():
            pass

        @staticmethod
        def json():
            return return_json

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(return_value=_MockResponse())
    return mock_client


class TestFinnhubIntegration:
    @pytest.mark.asyncio
    async def test_daily(self):
        key = os.environ.get("FINNHUB_API_KEY", "")
        if key:
            provider = FinnhubProvider(api_key=key)
            result = await provider.fetch_daily_candle("SPY", KNOWN_DATE)
        else:
            with patch("src.ingestion.candle_providers.httpx.AsyncClient", return_value=_mock_httpx_get(_finnhub_candle_json())):
                provider = FinnhubProvider(api_key="ci-mock-key")
                result = await provider.fetch_daily_candle("SPY", KNOWN_DATE)

        assert result is not None
        assert result["s"] == "ok"
        assert result["o"][0] > 0


# ── Chain integration ─────────────────────────────────────────────


class TestChainIntegration:
    @pytest.mark.asyncio
    async def test_chain_yahoo_succeeds(self):
        """Chain should succeed via Yahoo Finance (no keys needed)."""
        chain = build_candle_chain()
        result = await chain.fetch_daily_candle("SPY", KNOWN_DATE)
        assert result is not None
        assert result["s"] == "ok"
        assert result["o"][0] > 0

    @pytest.mark.asyncio
    async def test_chain_intraday(self):
        chain = build_candle_chain()
        result = await chain.fetch_intraday_candles("SPY", KNOWN_DATE)
        assert result is not None
        assert len(result) > 0
