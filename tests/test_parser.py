from pathlib import Path

import pytest

from src.ingestion.parser import (
    _extract_tickers,
    find_todays_briefing,
    parse_briefing_date,
    read_briefing,
)


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
    def test_finds_exact_match(self, tmp_path: Path) -> None:
        today_file = tmp_path / "Atlas-Briefing-2026.07.17.md"
        today_file.write_text("# Test")
        result = find_todays_briefing(tmp_path)
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

    def test_macro_sentiment_positive(self, tmp_briefing_file: Path) -> None:
        briefing = read_briefing(tmp_briefing_file)
        assert briefing.macro_sentiment != 0.0

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            read_briefing("/nonexistent/file.md")
