from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class DataSource(StrEnum):
    """Enumeration of supported market data sources."""

    FINNHUB = "finnhub"
    MCP_CHAIN = "mcp_chain"
    BRIEFING = "briefing"
    BRAVE = "brave"
    REDDIT = "reddit"
    UNUSUAL_WHALES = "unusual_whales"


class Quote(BaseModel):
    """A market quote for a given symbol from a specific source."""

    symbol: str
    current_price: float
    open_price: float
    high_price: float
    low_price: float
    previous_close: float
    change_pct: float
    volume: int
    source: DataSource
    timestamp: datetime


class NewsHeadline(BaseModel):
    """A news headline with sentiment polarity."""

    title: str
    source: str
    url: str
    published_at: datetime | None = None
    snippet: str = ""
    polarity: float = 0.0


class RSSCacheItem(BaseModel):
    """An entry from an RSS feed."""

    title: str
    source: str
    url: str
    published_at: datetime | None = None
    summary: str = ""


class MarketSnapshot(BaseModel):
    """A point-in-time snapshot of market data: quotes, news, and RSS items."""

    quotes: dict[str, Quote] = Field(default_factory=dict)
    news: list[NewsHeadline] = Field(default_factory=list)
    rss_items: list[RSSCacheItem] = Field(default_factory=list)
    captured_at: datetime = Field(default_factory=datetime.utcnow)

    def avg_sentiment_polarity(self) -> float:
        """Compute the average sentiment polarity across all news headlines.

        Returns:
            Mean polarity, or 0.0 if there are no headlines.
        """
        if not self.news:
            return 0.0
        return round(sum(n.polarity for n in self.news) / len(self.news), 4)
