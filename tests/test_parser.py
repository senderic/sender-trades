from pathlib import Path

import pytest

from src.ingestion.parser import (
    DEGRADED_SUMMARY_PREFIX,
    _classify_quality,
    _extract_tickers,
    find_todays_briefing,
    parse_briefing_date,
    read_briefing,
)
from src.models.briefing import BlogItem, BriefingData, BriefingQuality, NewsItem


class TestParseBriefingDate:
    def test_extracts_date_from_filename(self) -> None:
        result = parse_briefing_date("Atlas-Briefing-2026.07.17.md")
        assert result is not None
        assert result.year == 2026
        assert result.month == 7
        assert result.day == 17

    def test_returns_none_for_no_date(self) -> None:
        assert parse_briefing_date("no-date-here.md") is None


class TestFindTodaysBriefing:
    def test_finds_in_root_when_no_subdir(self, tmp_path: Path) -> None:
        today_file = tmp_path / "Atlas-Briefing-2026.07.17.md"
        today_file.write_text("# Test")
        # Default briefings_subdir="briefings" but no briefings/ dir exists
        # -> should fall back to searching root.
        result = find_todays_briefing(tmp_path)
        assert result is not None
        assert result.name == "Atlas-Briefing-2026.07.17.md"

    def test_finds_in_subdir_first(self, tmp_path: Path) -> None:
        briefings = tmp_path / "briefings"
        briefings.mkdir()
        (briefings / "Atlas-Briefing-2026.07.18.md").write_text("# New")
        (tmp_path / "Atlas-Briefing-2026.07.17.md").write_text("# Legacy")
        result = find_todays_briefing(tmp_path, briefings_subdir="briefings")
        assert result is not None
        assert result.name == "Atlas-Briefing-2026.07.18.md"
        assert result.parent == briefings

    def test_falls_back_to_root_when_subdir_empty(self, tmp_path: Path) -> None:
        briefings = tmp_path / "briefings"
        briefings.mkdir()  # empty
        (tmp_path / "Atlas-Briefing-2026.07.17.md").write_text("# Legacy")
        result = find_todays_briefing(tmp_path, briefings_subdir="briefings")
        assert result is not None
        assert result.name == "Atlas-Briefing-2026.07.17.md"

    def test_empty_subdir_string_searches_root(self, tmp_path: Path) -> None:
        today_file = tmp_path / "Atlas-Briefing-2026.07.17.md"
        today_file.write_text("# Test")
        result = find_todays_briefing(tmp_path, briefings_subdir="")
        assert result is not None

    def test_returns_none_for_empty_dir(self, tmp_path: Path) -> None:
        result = find_todays_briefing(tmp_path)
        assert result is None

    def test_returns_none_for_nonexistent_dir(self) -> None:
        result = find_todays_briefing("/nonexistent/path")
        assert result is None


class TestExtractTickers:
    def test_extracts_from_bold_markers(self) -> None:
        text = """**SPY** | $745.20 | +0.85% | Broad market strength
**QQQ** | $715.30 | +1.20% | Tech sector rally"""
        tickers = _extract_tickers(text)
        assert len(tickers) == 2
        assert tickers[0].symbol == "SPY"
        assert tickers[0].price == 745.20
        assert tickers[0].change_pct == 0.85

    def test_extracts_change_pct(self) -> None:
        text = """**AAPL** $250.30 -2.50%"""
        tickers = _extract_tickers(text)
        assert len(tickers) == 1
        assert tickers[0].change_pct == -2.50

    def test_extracts_likely_driver(self) -> None:
        text = """**NVDA** $235.40 +2.10% Likely driver: AI earnings optimism"""
        tickers = _extract_tickers(text)
        assert len(tickers) == 1
        assert "AI earnings" in tickers[0].likely_driver


class TestReadBriefing:
    def test_reads_complete_briefing(self, tmp_briefing_file: Path) -> None:
        briefing = read_briefing(tmp_briefing_file)
        assert briefing.briefing_date.year == 2026
        assert len(briefing.tickers) > 0
        assert len(briefing.news_items) > 0
        assert briefing.executive_summary != ""
        # The fixture has blogs present (a "### Understanding 0DTE..." item),
        # so classification should land on FULL.
        assert briefing.briefing_quality == BriefingQuality.FULL
        assert briefing.macro_sentiment is not None

    def test_macro_sentiment_positive(self, tmp_briefing_file: Path) -> None:
        briefing = read_briefing(tmp_briefing_file)
        assert briefing.macro_sentiment is not None
        assert briefing.macro_sentiment != 0.0

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            read_briefing("/nonexistent/file.md")


class TestClassifyQuality:
    def test_full_briefing_is_full(self) -> None:
        briefing = BriefingData(
            briefing_date=__import__("datetime").date.today(),
            executive_summary="Market sentiment is bullish following earnings.",
            news_items=[],
            blog_items=[BlogItem(title="x", author="a")],
        )
        # news_items is empty and blog_items non-empty, so neither degradation
        # rule fires; should return FULL.
        assert _classify_quality(briefing) == BriefingQuality.FULL

    def test_degraded_summary_string_is_degraded(self) -> None:
        briefing = BriefingData(
            briefing_date=__import__("datetime").date.today(),
            executive_summary=DEGRADED_SUMMARY_PREFIX + ". Please see the sections.",
            news_items=[NewsItem(title="x", source="a", url="https://example.com")],
            blog_items=[],
        )
        assert _classify_quality(briefing) == BriefingQuality.DEGRADED

    def test_news_present_but_no_blogs_is_degraded(self) -> None:
        briefing = BriefingData(
            briefing_date=__import__("datetime").date.today(),
            executive_summary="Bullish market sentiment today.",
            news_items=[NewsItem(title="x", source="a", url="https://example.com")],
            blog_items=[],
        )
        assert _classify_quality(briefing) == BriefingQuality.DEGRADED

    def test_no_news_no_summary_is_failed(self) -> None:
        briefing = BriefingData(briefing_date=__import__("datetime").date.today())
        assert _classify_quality(briefing) == BriefingQuality.FAILED

    def test_read_briefing_classifies_degraded_fixture(self, tmp_path: Path) -> None:
        degraded_md = (
            "# Atlas Morning Briefing\n\n"
            "## Executive Summary\n"
            + DEGRADED_SUMMARY_PREFIX
            + ". See the individual sections below.\n\n"
            "## AI & Tech News\n\n"
            "### Some raw headline\n"
            "*Source: reuters.com*\n\n"
            "[Read more](https://example.com/x)\n"
        )
        path = tmp_path / "Atlas-Briefing-2026.07.18.md"
        path.write_text(degraded_md)
        briefing = read_briefing(path)
        assert briefing.briefing_quality == BriefingQuality.DEGRADED
        assert briefing.macro_sentiment is None
