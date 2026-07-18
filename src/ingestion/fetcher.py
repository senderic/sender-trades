from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import httpx
import structlog

from src.models.market import (
    DataSource,
    MarketSnapshot,
    NewsHeadline,
    Quote,
    RSSCacheItem,
)

logger = structlog.get_logger()


class FinnhubFetcher:
    """Fetches real-time stock quotes from the Finnhub API."""

    def __init__(self, api_key: str, timeout: int = 10):
        self.api_key = api_key
        self.timeout = timeout

    async def fetch_quote(self, symbol: str) -> Optional[Quote]:
        """Fetch the latest quote for a given symbol from Finnhub.

        Args:
            symbol: The ticker symbol to fetch.

        Returns:
            A Quote object if successful, or None on failure.
        """
        if not self.api_key:
            logger.warning("finnhub_no_api_key", symbol=symbol)
            return None
        url = f"https://finnhub.io/api/v1/quote"
        params = {"symbol": symbol, "token": self.api_key}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                if "c" not in data or data["c"] == 0:
                    logger.warning("finnhub_empty_response", symbol=symbol)
                    return None
                return Quote(
                    symbol=symbol,
                    current_price=float(data["c"]),
                    open_price=float(data["o"]),
                    high_price=float(data["h"]),
                    low_price=float(data["l"]),
                    previous_close=float(data["pc"]),
                    change_pct=(
                        (float(data["c"]) - float(data["pc"])) / float(data["pc"]) * 100
                        if float(data["pc"]) else 0.0
                    ),
                    volume=int(data.get("v", 0)),
                    source=DataSource.FINNHUB,
                    timestamp=datetime.now(timezone.utc),
                )
        except httpx.TimeoutException:
            logger.error("finnhub_timeout", symbol=symbol)
            return None
        except httpx.HTTPStatusError as e:
            logger.error("finnhub_http_error", symbol=symbol, status=e.response.status_code)
            return None
        except Exception as e:
            logger.error("finnhub_error", symbol=symbol, error=str(e))
            return None


class BraveFetcher:
    """Fetches financial news headlines from the Brave Search API."""

    def __init__(self, api_key: str, query: str = "", timeout: int = 10):
        self.api_key = api_key
        self.query = query
        self.timeout = timeout

    async def fetch_news(self) -> list[NewsHeadline]:
        """Fetch recent news headlines from Brave Search.

        Returns:
            A list of NewsHeadline objects, or an empty list on failure.
        """
        if not self.api_key:
            logger.warning("brave_no_api_key")
            return []
        url = "https://api.search.brave.com/res/v1/news/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }
        params = {
            "q": self.query or "stock market",
            "count": 20,
            "freshness": "pd",
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url, headers=headers, params=params)
                resp.raise_for_status()
                data = resp.json()
                headlines: list[NewsHeadline] = []
                for result in data.get("results", []):
                    published = None
                    if result.get("published_time"):
                        try:
                            published = datetime.fromisoformat(result["published_time"].replace("Z", "+00:00"))
                        except (ValueError, TypeError):
                            published = datetime.now(timezone.utc)
                    headlines.append(NewsHeadline(
                        title=result.get("title", ""),
                        source=result.get("source", ""),
                        url=result.get("url", ""),
                        published_at=published,
                        snippet=result.get("description", ""),
                        polarity=_estimate_polarity(result.get("description", "") or result.get("title", "")),
                    ))
                return headlines
        except httpx.TimeoutException:
            logger.error("brave_timeout")
            return []
        except httpx.HTTPStatusError as e:
            logger.error("brave_http_error", status=e.response.status_code)
            return []
        except Exception as e:
            logger.error("brave_error", error=str(e))
            return []


class RSSFetcher:
    """Fetches entries from a list of RSS/Atom feeds."""

    def __init__(self, feed_urls: list[str], timeout: int = 10):
        self.feed_urls = feed_urls
        self.timeout = timeout

    async def fetch_all(self) -> list[RSSCacheItem]:
        """Concurrently fetch all configured RSS feeds.

        Returns:
            Combined list of RSSCacheItem from all feeds.
        """
        items: list[RSSCacheItem] = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [self._fetch_feed(client, url) for url in self.feed_urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    items.extend(result)
                elif isinstance(result, Exception):
                    logger.warning("rss_fetch_error", error=str(result))
        return items

    async def _fetch_feed(self, client: httpx.AsyncClient, url: str) -> list[RSSCacheItem]:
        """Fetch and parse a single RSS feed.

        Args:
            client: Shared httpx async client.
            url: Feed URL to fetch.

        Returns:
            List of RSSCacheItem parsed from the feed.
        """
        try:
            import feedparser
            resp = await client.get(url)
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
            items: list[RSSCacheItem] = []
            for entry in feed.entries[:10]:
                published = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    import time
                    published = datetime.fromtimestamp(time.mktime(entry.published_parsed), tz=timezone.utc)
                items.append(RSSCacheItem(
                    title=entry.get("title", ""),
                    source=url,
                    url=entry.get("link", ""),
                    published_at=published,
                    summary=entry.get("summary", "")[:500],
                ))
            return items
        except ImportError:
            logger.error("rss_missing_dependency", detail="feedparser not installed")
            return []
        except Exception as e:
            logger.warning("rss_feed_error", url=url, error=str(e))
            return []


async def fetch_market_data(
    symbols: list[str],
    finnhub_key: str,
    brave_key: str,
    brave_query: str,
    rss_urls: list[str],
    timeout: int = 10,
) -> MarketSnapshot:
    """Fetch all market data concurrently: quotes, news, and RSS feeds.

    Args:
        symbols: List of ticker symbols for which to fetch quotes.
        finnhub_key: Finnhub API key.
        brave_key: Brave Search API key.
        brave_query: Brave news search query.
        rss_urls: List of RSS feed URLs.
        timeout: HTTP request timeout in seconds.

    Returns:
        A MarketSnapshot containing quotes, news, and RSS items.
    """
    finnhub = FinnhubFetcher(finnhub_key, timeout)
    brave = BraveFetcher(brave_key, brave_query, timeout)
    rss = RSSFetcher(rss_urls, timeout)

    quote_tasks = [finnhub.fetch_quote(sym) for sym in symbols]
    news_task = brave.fetch_news()
    rss_task = rss.fetch_all()

    quotes_results, news_results, rss_results = await asyncio.gather(
        asyncio.gather(*quote_tasks),
        news_task,
        rss_task,
    )

    quotes: dict[str, Quote] = {}
    for q in quotes_results:
        if q is not None:
            quotes[q.symbol] = q

    return MarketSnapshot(
        quotes=quotes,
        news=news_results,
        rss_items=rss_results,
        captured_at=datetime.now(timezone.utc),
    )


POSITIVE_WORDS = {"surge", "gain", "rally", "bullish", "optimism", "growth",
                  "positive", "strength", "breakout", "upside", "beat", "up",
                  "higher", "soar", "climb", "jump", "rise"}
NEGATIVE_WORDS = {"decline", "drop", "loss", "bearish", "pessimism", "slowdown",
                  "negative", "weakness", "selloff", "downside", "miss", "down",
                  "lower", "plunge", "slide", "fall", "slump", "crash"}


def _estimate_polarity(text: str) -> float:
    """Estimate sentiment polarity of a text string using keyword matching.

    Args:
        text: Input text to analyse.

    Returns:
        A float between -1.0 and 1.0, where positive is bullish.
    """
    words = text.lower().split()
    pos = sum(1 for w in words if w.strip(".,!?") in POSITIVE_WORDS)
    neg = sum(1 for w in words if w.strip(".,!?") in NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 4)
