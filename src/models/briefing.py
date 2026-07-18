from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field


class BriefingQuality(StrEnum):
    """Observed quality of an Atlas morning briefing.

    Attributes:
        FULL: LLM layer ran upstream; executive summary is real LLM output.
        DEGRADED: Upstream LLM layer was skipped or failed; the markdown
            is the deterministic fallback (e.g. starts with
            ``"Synthesis unavailable for today's briefing"``).
        FAILED: Briefing is missing or unparsable — no executive summary
            and no news items.
    """

    FULL = "full"
    DEGRADED = "degraded"
    FAILED = "failed"


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
    briefing_quality: BriefingQuality = BriefingQuality.FULL

    @property
    def macro_sentiment(self) -> float | None:
        """Estimate macro sentiment from the executive summary text.

        Returns ``None`` when the briefing is degraded or failed — the
        absence of sentiment words on a degraded summary means "we don't
        know," not "market is neutral," and downstream strategies must
        not treat it as a neutral signal.

        Returns:
            A polarity score between -1.0 (bearish) and 1.0 (bullish),
            or ``None`` when briefing quality is not ``FULL``.
        """
        if self.briefing_quality != BriefingQuality.FULL:
            return None
        bullish_words = {
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
        }
        bearish_words = {
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
        }
        summary_lower = self.executive_summary.lower()
        words = summary_lower.split()
        bullish_count = sum(1 for w in words if w.strip(".,!?") in bullish_words)
        bearish_count = sum(1 for w in words if w.strip(".,!?") in bearish_words)
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        return round((bullish_count - bearish_count) / total, 4)
