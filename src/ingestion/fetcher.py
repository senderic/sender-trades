"""Data fetchers for market data, news, and RSS feeds."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
import structlog

from src.models.market import (
    DataSource,
    MarketSnapshot,
    NewsHeadline,
    Quote,
    RSSCacheItem,
)

ET_TZ = ZoneInfo("America/New_York")

logger = structlog.get_logger()


class FinnhubFetcher:
    """Fetches real-time stock quotes from the Finnhub API."""

    def __init__(self, api_key: str, timeout: int = 10):
        """Initialize FinnhubFetcher with API credentials.

        Args:
            api_key: Finnhub API key.
            timeout: HTTP request timeout in seconds.
        """
        self.api_key = api_key
        self.timeout = timeout

    async def fetch_quote(self, symbol: str) -> Quote | None:
        """Fetch the latest quote for a given symbol from Finnhub.

        Args:
            symbol: The ticker symbol to fetch.

        Returns:
            A Quote object if successful, or None on failure.
        """
        if not self.api_key:
            logger.warning("finnhub_no_api_key", symbol=symbol)
            return None
        url = "https://finnhub.io/api/v1/quote"
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
                        if float(data["pc"])
                        else 0.0
                    ),
                    volume=int(data.get("v", 0)),
                    source=DataSource.FINNHUB,
                    timestamp=datetime.now(UTC),
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


    async def fetch_daily_candle(
        self, symbol: str, target_date: date
    ) -> dict | None:
        if not self.api_key:
            logger.warning("finnhub_no_api_key", symbol=symbol)
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
                    logger.warning("finnhub_candle_no_data", symbol=symbol, target=target_date.isoformat())
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
            logger.warning("finnhub_no_api_key", symbol=symbol)
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
                    candles.append({
                        "timestamp": data["t"][i],
                        "open": data["o"][i],
                        "high": data["h"][i],
                        "low": data["l"][i],
                        "close": data["c"][i],
                        "volume": data["v"][i],
                    })
                return candles
        except httpx.TimeoutException:
            logger.error("finnhub_intraday_timeout", symbol=symbol)
            return None
        except httpx.HTTPStatusError as e:
            logger.error("finnhub_intraday_http_error", symbol=symbol, status=e.response.status_code)
            return None
        except Exception as e:
            logger.error("finnhub_intraday_error", symbol=symbol, error=str(e))
            return None


class BraveFetcher:
    """Fetches financial news headlines from the Brave Search API."""

    def __init__(self, api_key: str, query: str = "", timeout: int = 10):
        """Initialize BraveFetcher with API credentials and search query.

        Args:
            api_key: Brave Search API key.
            query: Default search query string.
            timeout: HTTP request timeout in seconds.
        """
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
                            published = datetime.fromisoformat(
                                result["published_time"].replace("Z", "+00:00")
                            )
                        except (ValueError, TypeError):
                            published = datetime.now(UTC)
                    headlines.append(
                        NewsHeadline(
                            title=result.get("title", ""),
                            source=result.get("source", ""),
                            url=result.get("url", ""),
                            published_at=published,
                            snippet=result.get("description", ""),
                            polarity=_estimate_polarity(
                                result.get("description", "") or result.get("title", "")
                            ),
                        )
                    )
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


class RedditFetcher:
    """Fetches hot posts from configured subreddits via Reddit's public JSON API."""

    def __init__(
        self,
        subreddits: list[str],
        post_limit: int = 25,
        user_agent: str = "sender-trades/1.0",
        timeout: int = 10,
    ):
        self.subreddits = subreddits
        self.post_limit = post_limit
        self.user_agent = user_agent
        self.timeout = timeout

    async def fetch_posts(self) -> list[NewsHeadline]:
        headlines: list[NewsHeadline] = []
        headers = {"User-Agent": self.user_agent}
        async with httpx.AsyncClient(timeout=self.timeout, headers=headers) as client:
            for sub in self.subreddits:
                try:
                    url = f"https://www.reddit.com/r/{sub}/hot.json?limit={self.post_limit}"
                    resp = await client.get(url)
                    if resp.status_code == 429:
                        logger.warning("reddit_rate_limited", subreddit=sub)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    for child in data.get("data", {}).get("children", []):
                        post = child.get("data", {})
                        if post.get("stickied"):
                            continue
                        created = (
                            datetime.fromtimestamp(post.get("created_utc", 0), tz=UTC)
                            if post.get("created_utc")
                            else None
                        )
                        snippet = (post.get("selftext", "") or "")[:500]
                        headlines.append(
                            NewsHeadline(
                                title=post.get("title", ""),
                                source=f"r/{sub}",
                                url=f"https://www.reddit.com{post.get('permalink', '')}",
                                published_at=created,
                                snippet=snippet,
                                polarity=_estimate_polarity(f"{post.get('title', '')} {snippet}"),
                            )
                        )
                except httpx.TimeoutException:
                    logger.error("reddit_timeout", subreddit=sub)
                except httpx.HTTPStatusError as e:
                    logger.error("reddit_http_error", subreddit=sub, status=e.response.status_code)
                except Exception as e:
                    logger.error("reddit_error", subreddit=sub, error=str(e))
        return headlines


class UnusualWhalesFetcher:
    """Fetches options flow data from Unusual Whales API (paid subscription required)."""

    BASE_URL = "https://api.unusualwhales.com/v1"

    def __init__(self, api_key: str, timeout: int = 10):
        self.api_key = api_key
        self.timeout = timeout

    async def fetch_flow_news(self) -> list[NewsHeadline]:
        if not self.api_key:
            logger.warning("unusual_whales_no_api_key")
            return []
        headers = {"Authorization": f"Bearer {self.api_key}"}
        headlines: list[NewsHeadline] = []
        try:
            async with httpx.AsyncClient(timeout=self.timeout, headers=headers) as client:
                resp = await client.get(f"{self.BASE_URL}/flow/tickers")
                resp.raise_for_status()
                data = resp.json()
                for item in data.get("data", [])[:20]:
                    ticker = item.get("ticker", "")
                    sentiment = item.get("flow_sentiment", "neutral")
                    headline_text = (
                        f"{ticker} unusual options flow: {sentiment} "
                        f"- {item.get('option_type', '')} {item.get('strike', '')}"
                    )
                    headlines.append(
                        NewsHeadline(
                            title=headline_text,
                            source="unusualwhales",
                            url=item.get("url", f"https://unusualwhales.com/{ticker}"),
                            published_at=datetime.now(UTC),
                            snippet=headline_text,
                            polarity=0.3
                            if sentiment == "bullish"
                            else (-0.3 if sentiment == "bearish" else 0.0),
                        )
                    )
        except httpx.TimeoutException:
            logger.error("unusual_whales_timeout")
        except httpx.HTTPStatusError as e:
            logger.error("unusual_whales_http_error", status=e.response.status_code)
        except Exception as e:
            logger.error("unusual_whales_error", error=str(e))
        return headlines


class RSSFetcher:
    """Fetches entries from a list of RSS/Atom feeds."""

    def __init__(self, feed_urls: list[str], timeout: int = 10):
        """Initialize RSSFetcher with feed URLs.

        Args:
            feed_urls: List of RSS feed URLs to fetch.
            timeout: HTTP request timeout in seconds.
        """
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

                    published = datetime.fromtimestamp(time.mktime(entry.published_parsed), tz=UTC)
                items.append(
                    RSSCacheItem(
                        title=entry.get("title", ""),
                        source=url,
                        url=entry.get("link", ""),
                        published_at=published,
                        summary=entry.get("summary", "")[:500],
                    )
                )
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
    reddit_subreddits: list[str] | None = None,
    reddit_post_limit: int = 25,
    unusual_whales_key: str = "",
    timeout: int = 10,
) -> MarketSnapshot:
    """Fetch all market data concurrently: quotes, news, RSS, Reddit, and Unusual Whales.

    Args:
        symbols: List of ticker symbols for which to fetch quotes.
        finnhub_key: Finnhub API key.
        brave_key: Brave Search API key.
        brave_query: Brave news search query.
        rss_urls: List of RSS feed URLs.
        reddit_subreddits: List of subreddits to scrape, or None to skip.
        reddit_post_limit: Max posts per subreddit.
        unusual_whales_key: Unusual Whales API key (empty to skip).
        timeout: HTTP request timeout in seconds.

    Returns:
        A MarketSnapshot containing quotes, news, RSS items, and social sentiment.
    """
    finnhub = FinnhubFetcher(finnhub_key, timeout)
    brave = BraveFetcher(brave_key, brave_query, timeout)
    rss = RSSFetcher(rss_urls, timeout)
    reddit = RedditFetcher(
        subreddits=reddit_subreddits or [],
        post_limit=reddit_post_limit,
        timeout=timeout,
    )
    uw = UnusualWhalesFetcher(unusual_whales_key, timeout)

    quote_tasks = [finnhub.fetch_quote(sym) for sym in symbols]
    news_task = brave.fetch_news()
    rss_task = rss.fetch_all()
    reddit_task = reddit.fetch_posts()
    uw_task = uw.fetch_flow_news()

    quotes_results, news_results, rss_results, reddit_results, uw_results = await asyncio.gather(
        asyncio.gather(*quote_tasks),
        news_task,
        rss_task,
        reddit_task,
        uw_task,
    )

    quotes: dict[str, Quote] = {}
    for q in quotes_results:
        if q is not None:
            quotes[q.symbol] = q

    all_news = news_results + reddit_results + uw_results

    return MarketSnapshot(
        quotes=quotes,
        news=all_news,
        rss_items=rss_results,
        captured_at=datetime.now(UTC),
    )


POSITIVE_WORDS = {
    "surge",
    "gain",
    "rally",
    "bullish",
    "optimism",
    "growth",
    "positive",
    "strength",
    "breakout",
    "upside",
    "beat",
    "up",
    "higher",
    "soar",
    "climb",
    "jump",
    "rise",
}
NEGATIVE_WORDS = {
    "decline",
    "drop",
    "loss",
    "bearish",
    "pessimism",
    "slowdown",
    "negative",
    "weakness",
    "selloff",
    "downside",
    "miss",
    "down",
    "lower",
    "plunge",
    "slide",
    "fall",
    "slump",
    "crash",
}


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
