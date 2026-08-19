"""Fallback candle providers for historical price data.

Provides a chain-of-responsibility pattern over three providers:
  1. YahooFinanceProvider  — free, no API key (primary)
  2. AlphaVantageProvider  — free tier, needs API key (first fallback)
  3. FinnhubProvider       — free tier, needs API key (final fallback)

Each provider normalises to the same Finnhub-compatible dict format so
that :func:`src.prediction_tracker.check_outcome` works unchanged.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Protocol, runtime_checkable

import httpx
import structlog
import yfinance as yf

from src.timezone import ET_TZ

logger = structlog.get_logger()

# ── Protocol ──────────────────────────────────────────────────────────


@runtime_checkable
class CandleProvider(Protocol):
    """Protocol for providers of historical stock candle data.

    All providers normalise to the same Finnhub-compatible dict format
    so that :func:`src.prediction_tracker.check_outcome` works unchanged.
    """

    async def fetch_daily_candle(self, symbol: str, target_date: date) -> dict | None:
        """Return a Finnhub-style daily candle dict or ``None``.

        The returned dict has the shape::

            {"o": [open], "h": [high], "l": [low], "c": [close],
             "v": [volume], "t": [timestamp], "s": "ok"}
        """

    async def fetch_intraday_candles(
        self, symbol: str, target_date: date, resolution: int = 60
    ) -> list[dict] | None:
        """Return a list of hourly candle dicts or ``None``.

        Each dict has the shape::

            {"timestamp": t, "open": o, "high": h,
             "low": l, "close": c, "volume": v}
        """


# ── Yahoo Finance (primary — no API key needed) ──────────────────────


class YahooFinanceProvider:
    """Candle provider backed by Yahoo Finance via the ``yfinance`` library.

    Requires no API key.  Runs the synchronous ``yfinance`` calls in a
    thread-pool executor so the async event loop is not blocked.
    """

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout

    async def fetch_daily_candle(self, symbol: str, target_date: date) -> dict | None:
        start = target_date.isoformat()
        end = (target_date + timedelta(days=1)).isoformat()

        def _fetch() -> dict | None:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval="1d")
            if df.empty:
                return None
            row = df.iloc[-1]
            ts = int(row.name.timestamp()) if hasattr(row.name, "timestamp") else 0
            return {
                "o": [float(row["Open"])],
                "h": [float(row["High"])],
                "l": [float(row["Low"])],
                "c": [float(row["Close"])],
                "v": [int(row["Volume"])],
                "t": [ts],
                "s": "ok",
            }

        try:
            return await asyncio.to_thread(_fetch)
        except Exception:
            return None

    async def fetch_intraday_candles(
        self, symbol: str, target_date: date, resolution: int = 60
    ) -> list[dict] | None:
        start = target_date.isoformat()
        end = (target_date + timedelta(days=1)).isoformat()
        interval = f"{resolution}m"

        def _fetch() -> list[dict] | None:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval=interval)
            if df.empty:
                return None
            candles: list[dict] = []
            for idx, row in df.iterrows():
                ts = int(idx.timestamp()) if hasattr(idx, "timestamp") else 0
                candles.append(
                    {
                        "timestamp": ts,
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": int(row["Volume"]),
                    }
                )
            return candles

        try:
            return await asyncio.to_thread(_fetch)
        except Exception:
            return None


# ── Alpha Vantage (first fallback — needs API key) ───────────────────


ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"


class AlphaVantageProvider:
    """Candle provider backed by the Alpha Vantage free-tier API.

    Requires ``api_key``.  When the key is empty the provider silently
    returns ``None`` from every call (no-op fallback).
    """

    def __init__(self, api_key: str, timeout: int = 10) -> None:
        self.api_key = api_key
        self.timeout = timeout

    async def fetch_daily_candle(self, symbol: str, target_date: date) -> dict | None:
        if not self.api_key:
            return None

        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "compact",
            "apikey": self.api_key,
        }
        data = await self._get(params)
        if data is None:
            return None

        series = data.get("Time Series (Daily)")
        if not isinstance(series, dict):
            return None

        key = target_date.isoformat()
        day = series.get(key)
        if not isinstance(day, dict):
            return None

        try:
            o = float(day["1. open"])
            h = float(day["2. high"])
            l = float(day["3. low"])
            c = float(day["4. close"])
            v = int(day["5. volume"])
        except (KeyError, TypeError, ValueError):
            return None

        ts = int(
            datetime(target_date.year, target_date.month, target_date.day, tzinfo=ET_TZ).timestamp()
        )
        return {
            "o": [o],
            "h": [h],
            "l": [l],
            "c": [c],
            "v": [v],
            "t": [ts],
            "s": "ok",
        }

    async def fetch_intraday_candles(
        self, symbol: str, target_date: date, resolution: int = 60
    ) -> list[dict] | None:
        if not self.api_key:
            return None

        interval = f"{resolution}min"
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": "full",
            "apikey": self.api_key,
        }
        data = await self._get(params)
        if data is None:
            return None

        series_key = f"Time Series ({interval})"
        series = data.get(series_key)
        if not isinstance(series, dict):
            return None

        target_prefix = target_date.isoformat()
        candles: list[dict] = []
        for ts_str in sorted(series.keys()):
            if not ts_str.startswith(target_prefix):
                continue
            entry = series[ts_str]
            try:
                candles.append(
                    {
                        "timestamp": int(
                            datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").timestamp()
                        ),
                        "open": float(entry["1. open"]),
                        "high": float(entry["2. high"]),
                        "low": float(entry["3. low"]),
                        "close": float(entry["4. close"]),
                        "volume": int(entry["5. volume"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue

        return candles if candles else None

    async def _get(self, params: dict) -> dict | None:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(ALPHA_VANTAGE_BASE, params=params)
                resp.raise_for_status()
                data = resp.json()
                if "Error Message" in data or "Note" in data:
                    logger.warning(
                        "alphavantage_api_error",
                        detail=data.get("Note", data.get("Error Message", "")),
                    )
                    return None
                return data
        except httpx.TimeoutException:
            logger.error("alphavantage_timeout")
            return None
        except httpx.HTTPStatusError as e:
            logger.error("alphavantage_http_error", status=e.response.status_code)
            return None
        except Exception as e:
            logger.error("alphavantage_error", error=str(e))
            return None


# ── Finnhub (final fallback — needs API key) ──────────────────────────


class FinnhubProvider:
    """Candle provider wrapping the existing Finnhub candle endpoints.

    Requires ``api_key``.  When the key is empty the provider silently
    returns ``None`` from every call.
    """

    def __init__(self, api_key: str, timeout: int = 10) -> None:
        self.api_key = api_key
        self.timeout = timeout

    async def fetch_daily_candle(self, symbol: str, target_date: date) -> dict | None:
        if not self.api_key:
            return None
        start_dt = datetime(target_date.year, target_date.month, target_date.day, tzinfo=ET_TZ)
        end_dt = start_dt + timedelta(days=1) - timedelta(seconds=1)
        url = "https://finnhub.io/api/v1/stock/candle"
        params = {
            "symbol": symbol,
            "resolution": "D",
            "from": int(start_dt.timestamp()),
            "to": int(end_dt.timestamp()),
            "token": self.api_key,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                if data.get("s") != "ok":
                    return None
                return data
        except httpx.TimeoutException:
            logger.error("finnhub_candle_timeout", symbol=symbol)
            return None
        except httpx.HTTPStatusError as e:
            logger.error("finnhub_candle_http_error", symbol=symbol, status=e.response.status_code)
            return None
        except Exception as e:
            logger.error("finnhub_candle_error", symbol=symbol, error=str(e))
            return None

    async def fetch_intraday_candles(
        self, symbol: str, target_date: date, resolution: int = 60
    ) -> list[dict] | None:
        if not self.api_key:
            return None
        start_dt = datetime(target_date.year, target_date.month, target_date.day, tzinfo=ET_TZ)
        end_dt = start_dt + timedelta(days=1) - timedelta(seconds=1)
        url = "https://finnhub.io/api/v1/stock/candle"
        params = {
            "symbol": symbol,
            "resolution": str(resolution),
            "from": int(start_dt.timestamp()),
            "to": int(end_dt.timestamp()),
            "token": self.api_key,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                if data.get("s") != "ok":
                    return None
                candles: list[dict] = []
                for i in range(len(data.get("t", []))):
                    candles.append(
                        {
                            "timestamp": data["t"][i],
                            "open": data["o"][i],
                            "high": data["h"][i],
                            "low": data["l"][i],
                            "close": data["c"][i],
                            "volume": data["v"][i],
                        }
                    )
                return candles
        except httpx.TimeoutException:
            logger.error("finnhub_intraday_timeout", symbol=symbol)
            return None
        except httpx.HTTPStatusError as e:
            logger.error(
                "finnhub_intraday_http_error", symbol=symbol, status=e.response.status_code
            )
            return None
        except Exception as e:
            logger.error("finnhub_intraday_error", symbol=symbol, error=str(e))
            return None


# ── Chain of Responsibility ───────────────────────────────────────────


class CandleProviderChain:
    """Chain of candle providers tried in order until one returns data.

    Each provider is tried sequentially.  The first non-``None`` result
    is returned immediately.  If every provider fails the chain returns
    ``None``.

    Logs which provider served each request for observability.
    """

    def __init__(self, providers: list[CandleProvider]) -> None:
        self._providers = providers

    async def fetch_daily_candle(self, symbol: str, target_date: date) -> dict | None:
        for provider in self._providers:
            label = _provider_label(provider)
            try:
                result = await provider.fetch_daily_candle(symbol, target_date)
            except Exception:
                logger.warning("candle_provider_error", provider=label, symbol=symbol)
                continue
            if result is not None:
                logger.info("candle_served_by", provider=label, symbol=symbol, kind="daily")
                return result
        logger.warning("candle_no_provider", symbol=symbol, kind="daily")
        return None

    async def fetch_intraday_candles(
        self, symbol: str, target_date: date, resolution: int = 60
    ) -> list[dict] | None:
        for provider in self._providers:
            label = _provider_label(provider)
            try:
                result = await provider.fetch_intraday_candles(symbol, target_date, resolution)
            except Exception:
                logger.warning("candle_provider_error", provider=label, symbol=symbol)
                continue
            if result is not None:
                logger.info("candle_served_by", provider=label, symbol=symbol, kind="intraday")
                return result
        logger.warning("candle_no_provider", symbol=symbol, kind="intraday")
        return None


def _provider_label(provider: CandleProvider) -> str:
    return type(provider).__name__.replace("Provider", "")


def build_candle_chain(
    finnhub_api_key: str = "",
    alpha_vantage_api_key: str = "",
    timeout: int = 10,
) -> CandleProviderChain:
    """Build the default candle provider chain.

    Providers are ordered by reliability / cost:
      1. Yahoo Finance (free, no key)
      2. Alpha Vantage  (free tier, needs key)
      3. Finnhub        (free tier, needs key)
    """
    providers: list[CandleProvider] = [YahooFinanceProvider(timeout=timeout)]
    if alpha_vantage_api_key:
        providers.append(AlphaVantageProvider(alpha_vantage_api_key, timeout=timeout))
    if finnhub_api_key:
        providers.append(FinnhubProvider(finnhub_api_key, timeout=timeout))
    return CandleProviderChain(providers)
