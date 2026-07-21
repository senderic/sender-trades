"""Integration tests for candle providers against live APIs.

These tests make real network calls and are skipped when the required
API keys are not set in the environment.  Yahoo Finance requires no key
and is always tested.
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import pytest

from src.ingestion.candle_providers import (
    AlphaVantageProvider,
    CandleProviderChain,
    FinnhubProvider,
    YahooFinanceProvider,
    build_candle_chain,
)

pytestmark = [pytest.mark.integration]

# Use a known recent trading day for reproducible results.
# Friday 2026-07-17 is a recent trading day.
KNOWN_DATE = date(2026, 7, 17)


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


# ── Alpha Vantage (skipped if no key) ─────────────────────────────────


class TestAlphaVantageIntegration:
    @pytest.mark.asyncio
    async def test_daily_live(self):
        key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        if not key:
            pytest.skip("ALPHA_VANTAGE_API_KEY not set")
        provider = AlphaVantageProvider(api_key=key)
        result = await provider.fetch_daily_candle("SPY", KNOWN_DATE)
        assert result is not None
        assert result["s"] == "ok"
        assert result["o"][0] > 0


# ── Finnhub (skipped if no key) ──────────────────────────────────────


class TestFinnhubIntegration:
    @pytest.mark.asyncio
    async def test_daily_live(self):
        key = os.environ.get("FINNHUB_API_KEY", "")
        if not key:
            pytest.skip("FINNHUB_API_KEY not set")
        provider = FinnhubProvider(api_key=key)
        result = await provider.fetch_daily_candle("SPY", KNOWN_DATE)
        assert result is not None
        assert result["s"] == "ok"
        assert result["o"][0] > 0


# ── Chain integration ─────────────────────────────────────────────────


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
