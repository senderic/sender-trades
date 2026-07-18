from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class TickerRow(BaseModel):
    """A single ticker entry from the briefing watchlist."""

    symbol: str
    price: float
    change_pct: float
    likely_driver: str = ""


class NewsItem(BaseModel):
    """A news item extracted from the briefing."""

    title: str
    source: str
    url: str
    snippet: str = ""
    relevance_score: float = 0.0


class BlogItem(BaseModel):
    """A blog update entry from the briefing."""

    title: str
    author: str = ""
    summary: str = ""
    url: str = ""
    rating: int = 0


class PaperItem(BaseModel):
    """A research paper entry from the briefing."""

    title: str
    authors: str = ""
    arxiv_url: str = ""
    reproduction_score: float = 0.0


class BriefingData(BaseModel):
    """Structured representation of a parsed morning briefing document."""

    briefing_date: date
    executive_summary: str = ""
    key_connections: str = ""
    tickers: list[TickerRow] = Field(default_factory=list)
    news_items: list[NewsItem] = Field(default_factory=list)
    blog_items: list[BlogItem] = Field(default_factory=list)
    papers: list[PaperItem] = Field(default_factory=list)
    raw_markdown: str = ""

    @property
    def macro_sentiment(self) -> float:
        """Estimate macro sentiment from the executive summary text.

        Returns:
            A polarity score between -1.0 (bearish) and 1.0 (bullish).
        """
        bullish_words = {"surge", "gain", "rally", "bullish", "optimism",
                         "growth", "positive", "strength", "breakout", "upside"}
        bearish_words = {"decline", "drop", "loss", "bearish", "pessimism",
                         "slowdown", "negative", "weakness", "selloff", "downside"}
        summary_lower = self.executive_summary.lower()
        words = summary_lower.split()
        bullish_count = sum(1 for w in words if w.strip(".,!?") in bullish_words)
        bearish_count = sum(1 for w in words if w.strip(".,!?") in bearish_words)
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        return round((bullish_count - bearish_count) / total, 4)
