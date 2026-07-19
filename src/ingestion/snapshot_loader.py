"""Load raw data snapshots from atlas-morning-briefing instead of making API calls."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from src.models.market import (
    DataSource,
    MarketSnapshot,
    NewsHeadline,
    Quote,
    RSSCacheItem,
)
from src.timezone import today_local

logger = structlog.get_logger()


def _estimate_polarity(text: str) -> float:
    """Simple keyword-based polarity estimation matching fetcher.py."""
    words = text.lower().split()
    pos_words = {
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
    neg_words = {
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
    pos = sum(1 for w in words if w.strip(".,!?") in pos_words)
    neg = sum(1 for w in words if w.strip(".,!?") in neg_words)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 4)


class SnapshotLoader:
    """Load raw data snapshots saved by atlas-morning-briefing.

    Snapshots live at ``{atlas_dir}/snapshots/YYYY-MM-DD/`` and contain
    pre-fetched Finnhub quotes, Brave news, and RSS feed items.
    """

    def __init__(self, atlas_briefing_dir: str | Path) -> None:
        """Initialize SnapshotLoader.

        Args:
            atlas_briefing_dir: Path to the atlas-morning-briefing project root
                (e.g. ``~/atlas-morning-briefing``).
        """
        self.atlas_dir = Path(atlas_briefing_dir).expanduser().resolve()
        self.today = today_local()
        self.snapshot_dir = self.atlas_dir / "snapshots" / self.today.isoformat()

    def is_available(self) -> bool:
        """Check whether a snapshot exists for today.

        Returns:
            True if the today's snapshot directory exists.
        """
        return self.snapshot_dir.is_dir()

    def load(self) -> MarketSnapshot | None:
        """Load today's snapshot data, if available.

        Returns:
            A ``MarketSnapshot`` built from snapshot files, or ``None`` if
            no snapshot directory exists for today.
        """
        if not self.is_available():
            logger.info("snapshot_not_found", path=str(self.snapshot_dir))
            return None

        quotes = self._load_quotes()
        news = self._load_news()
        rss_items = self._load_rss()

        snapshot = MarketSnapshot(
            quotes=quotes,
            news=news,
            rss_items=rss_items,
            captured_at=datetime.now(UTC),
        )

        logger.info(
            "snapshot_loaded",
            path=str(self.snapshot_dir),
            quote_symbols=list(quotes.keys()),
            news_count=len(news),
            rss_count=len(rss_items),
        )
        return snapshot

    def is_complete(self, symbols: list[str]) -> bool:
        """Check whether the snapshot has all required data for the given symbols.

        Args:
            symbols: List of ticker symbols that must have quotes.

        Returns:
            True if all symbols have quotes and at least some news or RSS
            items are present.
        """
        if not self.is_available():
            return False
        quotes = self._load_quotes()
        has_all_quotes = all(sym in quotes for sym in symbols)
        has_content = self._snapshot_has_content()
        return has_all_quotes and has_content

    def _snapshot_has_content(self) -> bool:
        """Check if snapshot has any news or RSS files with data."""
        news_file = self.snapshot_dir / "brave_news.json"
        rss_file = self.snapshot_dir / "rss_feeds.json"
        if not news_file.exists() and not rss_file.exists():
            return False
        try:
            if news_file.exists() and news_file.stat().st_size > 10:
                return True
            if rss_file.exists() and rss_file.stat().st_size > 10:
                return True
        except OSError:
            pass
        return False

    def _load_json(self, filename: str) -> list[dict[str, Any]]:
        """Load and return a JSON array from the snapshot directory.

        Args:
            filename: Base name of the JSON file (e.g. ``finnhub_data.json``).

        Returns:
            Parsed list of dicts, or empty list on failure.
        """
        path = self.snapshot_dir / filename
        if not path.exists():
            logger.debug("snapshot_file_missing", file=filename)
            return []
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                return data
            logger.warning("snapshot_unexpected_format", file=filename, type=type(data).__name__)
            return []
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("snapshot_file_error", file=filename, error=str(e))
            return []

    def _load_quotes(self) -> dict[str, Quote]:
        """Load Finnhub quote data from snapshot.

        Returns:
            Dict mapping symbol to Quote object.
        """
        rows = self._load_json("finnhub_data.json")
        quotes: dict[str, Quote] = {}
        for row in rows:
            symbol = row.get("symbol", "")
            if not symbol:
                continue
            try:
                quotes[symbol] = Quote(
                    symbol=symbol,
                    current_price=float(row.get("current_price", 0)),
                    open_price=float(row.get("open", 0)),
                    high_price=float(row.get("high", 0)),
                    low_price=float(row.get("low", 0)),
                    previous_close=float(row.get("previous_close", 0)),
                    change_pct=float(row.get("percent_change", 0)),
                    volume=int(row.get("volume", 0)),
                    source=DataSource.BRIEFING,
                    timestamp=datetime.now(UTC),
                )
            except (ValueError, TypeError) as e:
                logger.warning("snapshot_quote_parse_error", symbol=symbol, error=str(e))
        return quotes

    def _load_news(self) -> list[NewsHeadline]:
        """Load Brave news data from snapshot.

        Returns:
            List of NewsHeadline objects.
        """
        rows = self._load_json("brave_news.json")
        headlines: list[NewsHeadline] = []
        for row in rows:
            title = row.get("title", "") or ""
            snippet = row.get("description", row.get("snippet", "")) or ""
            headlines.append(
                NewsHeadline(
                    title=title,
                    source=row.get("source", row.get("query", "brave_snapshot")),
                    url=row.get("url", ""),
                    published_at=None,
                    snippet=snippet,
                    polarity=_estimate_polarity(f"{title} {snippet}"),
                )
            )
        return headlines

    def _load_rss(self) -> list[RSSCacheItem]:
        """Load RSS feed data from snapshot.

        Returns:
            List of RSSCacheItem objects.
        """
        rows = self._load_json("rss_feeds.json")
        items: list[RSSCacheItem] = []
        for row in rows:
            items.append(
                RSSCacheItem(
                    title=row.get("title", "") or "",
                    source=row.get("source", ""),
                    url=row.get("link", row.get("url", "")),
                    published_at=None,
                    summary=(row.get("summary", row.get("description", "")) or "")[:500],
                )
            )
        return items
