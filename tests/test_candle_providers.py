"""Unit tests for candle providers with mocked HTTP / yfinance calls."""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd

from src.ingestion.candle_providers import (
    AlphaVantageProvider,
    CandleProviderChain,
    FinnhubProvider,
    YahooFinanceProvider,
    build_candle_chain,
)
from src.timezone import ET_TZ

# ── Helpers ───────────────────────────────────────────────────────────

TARGET_DATE = date(2026, 7, 17)
TARGET_TS = int(datetime(2026, 7, 17, tzinfo=ET_TZ).timestamp())


def _make_daily_df(open_p=100.0, high=105.0, low=99.0, close=103.0, volume=1_000_000):

    idx = pd.DatetimeIndex([datetime(2026, 7, 17, 16, 0, tzinfo=ET_TZ)])
    return pd.DataFrame(
        {"Open": [open_p], "High": [high], "Low": [low], "Close": [close], "Volume": [volume]},
        index=idx,
    )


def _make_hourly_df():

    rows = []
    for h in range(10, 16):
        rows.append(
            {
                "Datetime": datetime(2026, 7, 17, h, 0, tzinfo=ET_TZ),
                "Open": 100.0 + h,
                "High": 101.0 + h,
                "Low": 99.0 + h,
                "Close": 100.5 + h,
                "Volume": 1_000_000 + h * 1000,
            }
        )
    df = pd.DataFrame(rows).set_index("Datetime")
    df.index.name = "Datetime"
    return df


# ── YahooFinanceProvider ──────────────────────────────────────────────


class TestYahooFinanceProvider:
    @patch("yfinance.Ticker")
    async def test_daily_success(self, mock_ticker):
        df = _make_daily_df()
        mock_ticker.return_value.history.return_value = df
        provider = YahooFinanceProvider()
        result = await provider.fetch_daily_candle("SPY", date(2026, 7, 17))
        assert result is not None
        assert result["s"] == "ok"
        assert result["o"] == [100.0]
        assert result["h"] == [105.0]
        assert result["l"] == [99.0]
        assert result["c"] == [103.0]
        assert result["v"] == [1_000_000]

    @patch("yfinance.Ticker")
    async def test_daily_no_data(self, mock_ticker):

        mock_ticker.return_value.history.return_value = pd.DataFrame()
        provider = YahooFinanceProvider()
        result = await provider.fetch_daily_candle("SPY", date(2026, 7, 17))
        assert result is None

    @patch("yfinance.Ticker")
    async def test_intraday_success(self, mock_ticker):
        df = _make_hourly_df()
        mock_ticker.return_value.history.return_value = df
        provider = YahooFinanceProvider()
        result = await provider.fetch_intraday_candles("SPY", date(2026, 7, 17))
        assert result is not None
        assert len(result) == 6
        assert result[0]["open"] == 110.0
        assert result[0]["high"] == 111.0
        assert "timestamp" in result[0]

    @patch("yfinance.Ticker")
    async def test_intraday_no_data(self, mock_ticker):

        mock_ticker.return_value.history.return_value = pd.DataFrame()
        provider = YahooFinanceProvider()
        result = await provider.fetch_intraday_candles("SPY", date(2026, 7, 17))
        assert result is None


# ── AlphaVantageProvider ──────────────────────────────────────────────


class TestAlphaVantageProvider:
    @patch("src.ingestion.candle_providers.httpx.AsyncClient")
    async def test_daily_success(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "Time Series (Daily)": {
                "2026-07-17": {
                    "1. open": "100.0",
                    "2. high": "105.0",
                    "3. low": "99.0",
                    "4. close": "103.0",
                    "5. volume": "1000000",
                }
            }
        }
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_resp
        provider = AlphaVantageProvider(api_key="test_key")
        result = await provider.fetch_daily_candle("SPY", date(2026, 7, 17))
        assert result is not None
        assert result["s"] == "ok"
        assert result["o"] == [100.0]
        assert result["h"] == [105.0]
        assert result["l"] == [99.0]
        assert result["c"] == [103.0]
        assert result["v"] == [1000000]

    async def test_daily_no_api_key(self):
        provider = AlphaVantageProvider(api_key="")
        result = await provider.fetch_daily_candle("SPY", date(2026, 7, 17))
        assert result is None

    @patch("src.ingestion.candle_providers.httpx.AsyncClient")
    async def test_daily_api_error(self, mock_client):
        mock_client.return_value.__aenter__.return_value.get.side_effect = httpx.HTTPStatusError(
            "403", request=MagicMock(), response=MagicMock(status_code=403)
        )
        provider = AlphaVantageProvider(api_key="test_key")
        result = await provider.fetch_daily_candle("SPY", date(2026, 7, 17))
        assert result is None

    @patch("src.ingestion.candle_providers.httpx.AsyncClient")
    async def test_intraday_success(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "Time Series (60min)": {
                "2026-07-17 09:30:00": {
                    "1. open": "100.0",
                    "2. high": "101.0",
                    "3. low": "99.0",
                    "4. close": "100.5",
                    "5. volume": "500000",
                },
                "2026-07-17 10:30:00": {
                    "1. open": "100.5",
                    "2. high": "102.0",
                    "3. low": "100.0",
                    "4. close": "101.0",
                    "5. volume": "600000",
                },
            }
        }
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_resp
        provider = AlphaVantageProvider(api_key="test_key")
        result = await provider.fetch_intraday_candles("SPY", date(2026, 7, 17))
        assert result is not None
        assert len(result) == 2
        assert result[0]["open"] == 100.0
        assert result[1]["open"] == 100.5


# ── FinnhubProvider ──────────────────────────────────────────────────


class TestFinnhubProvider:
    @patch("src.ingestion.candle_providers.httpx.AsyncClient")
    async def test_daily_success(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "o": [100.0],
            "h": [105.0],
            "l": [99.0],
            "c": [103.0],
            "v": [1000000],
            "t": [TARGET_TS],
            "s": "ok",
        }
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_resp
        provider = FinnhubProvider(api_key="test_key")
        result = await provider.fetch_daily_candle("SPY", date(2026, 7, 17))
        assert result is not None
        assert result["s"] == "ok"
        assert result["o"] == [100.0]

    async def test_daily_no_api_key(self):
        provider = FinnhubProvider(api_key="")
        result = await provider.fetch_daily_candle("SPY", date(2026, 7, 17))
        assert result is None

    @patch("src.ingestion.candle_providers.httpx.AsyncClient")
    async def test_daily_no_data(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"s": "no_data"}
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_resp
        provider = FinnhubProvider(api_key="test_key")
        result = await provider.fetch_daily_candle("SPY", date(2026, 7, 17))
        assert result is None

    @patch("src.ingestion.candle_providers.httpx.AsyncClient")
    async def test_intraday_success(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "o": [100.0, 101.0],
            "h": [102.0, 103.0],
            "l": [99.0, 100.0],
            "c": [101.0, 102.0],
            "v": [500000, 600000],
            "t": [TARGET_TS, TARGET_TS + 3600],
            "s": "ok",
        }
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_resp
        provider = FinnhubProvider(api_key="test_key")
        result = await provider.fetch_intraday_candles("SPY", date(2026, 7, 17))
        assert result is not None
        assert len(result) == 2
        assert result[0]["open"] == 100.0
        assert result[1]["open"] == 101.0


# ── CandleProviderChain ───────────────────────────────────────────────


class _FakeProvider:
    def __init__(self, name: str, daily_result=None, intraday_result=None):
        self.name = name
        self.daily_result = daily_result
        self.intraday_result = intraday_result

    async def fetch_daily_candle(self, symbol, target_date):
        return self.daily_result

    async def fetch_intraday_candles(self, symbol, target_date, resolution=60):
        return self.intraday_result


class TestCandleProviderChain:
    async def test_first_provider_succeeds(self):
        data = {"s": "ok", "o": [100.0]}
        chain = CandleProviderChain([
            _FakeProvider("A", daily_result=data),
            _FakeProvider("B", daily_result={"s": "ok", "o": [200.0]}),
        ])
        result = await chain.fetch_daily_candle("SPY", date(2026, 7, 17))
        assert result == data

    async def test_first_fails_second_succeeds(self):
        chain = CandleProviderChain([
            _FakeProvider("A", daily_result=None),
            _FakeProvider("B", daily_result={"s": "ok", "o": [200.0]}),
        ])
        result = await chain.fetch_daily_candle("SPY", date(2026, 7, 17))
        assert result == {"s": "ok", "o": [200.0]}

    async def test_all_fail(self):
        chain = CandleProviderChain([
            _FakeProvider("A", daily_result=None),
            _FakeProvider("B", daily_result=None),
        ])
        result = await chain.fetch_daily_candle("SPY", date(2026, 7, 17))
        assert result is None

    async def test_empty_chain(self):
        chain = CandleProviderChain([])
        result = await chain.fetch_daily_candle("SPY", date(2026, 7, 17))
        assert result is None

    async def test_intraday_fallback(self):
        chain = CandleProviderChain([
            _FakeProvider("A", intraday_result=None),
            _FakeProvider("B", intraday_result=[{"timestamp": 0, "open": 100.0}]),
        ])
        result = await chain.fetch_intraday_candles("SPY", date(2026, 7, 17))
        assert result == [{"timestamp": 0, "open": 100.0}]


# ── build_candle_chain ───────────────────────────────────────────────


class TestBuildCandleChain:
    def test_yahoo_only_when_no_keys(self):
        chain = build_candle_chain()
        assert len(chain._providers) == 1
        assert isinstance(chain._providers[0], YahooFinanceProvider)

    def test_includes_alphavantage_when_key_provided(self):
        chain = build_candle_chain(alpha_vantage_api_key="av_key")
        assert len(chain._providers) == 2
        assert isinstance(chain._providers[0], YahooFinanceProvider)
        assert isinstance(chain._providers[1], AlphaVantageProvider)

    def test_includes_finnhub_when_key_provided(self):
        chain = build_candle_chain(finnhub_api_key="fh_key")
        assert len(chain._providers) == 2
        assert isinstance(chain._providers[1], FinnhubProvider)

    def test_all_three_when_all_keys_provided(self):
        chain = build_candle_chain(
            finnhub_api_key="fh_key",
            alpha_vantage_api_key="av_key",
        )
        assert len(chain._providers) == 3
        assert isinstance(chain._providers[0], YahooFinanceProvider)
        assert isinstance(chain._providers[1], AlphaVantageProvider)
        assert isinstance(chain._providers[2], FinnhubProvider)
