from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ingestion.status import BriefingStatus, read_briefing_status


class TestReadBriefingStatus:
    def test_reads_valid_status(self, tmp_path: Path) -> None:
        (tmp_path / "status.json").write_text(
            json.dumps(
                {
                    "timestamp": "2026-07-18T09:22:16.370029",
                    "papers_found": 0,
                    "blogs_found": 67,
                    "stocks_fetched": 24,
                    "news_found": 241,
                    "intelligence_enabled": True,
                    "errors": [],
                }
            )
        )
        status = read_briefing_status(tmp_path)
        assert status is not None
        assert isinstance(status, BriefingStatus)
        assert status.intelligence_enabled is True
        assert status.blogs_found == 67
        assert status.errors == []

    def test_intelligence_disabled(self, tmp_path: Path) -> None:
        (tmp_path / "status.json").write_text(
            json.dumps({"intelligence_enabled": False, "errors": ["scanner timeout"]})
        )
        status = read_briefing_status(tmp_path)
        assert status is not None
        assert status.intelligence_enabled is False
        assert status.errors == ["scanner timeout"]

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert read_briefing_status(tmp_path) is None

    def test_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "status.json").write_text("{not valid json")
        assert read_briefing_status(tmp_path) is None

    def test_non_object_json_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "status.json").write_text("[1, 2, 3]")
        assert read_briefing_status(tmp_path) is None

    def test_defaults_filled_for_missing_fields(self, tmp_path: Path) -> None:
        (tmp_path / "status.json").write_text(json.dumps({}))
        status = read_briefing_status(tmp_path)
        assert status is not None
        # intelligence_enabled defaults to True per BriefingStatus model.
        assert status.intelligence_enabled is True
        assert status.news_found == 0
        assert status.errors == []

    def test_expands_tilde(self) -> None:
        # Function should not raise on a tilde path that does not exist.
        result = read_briefing_status("~/definitely-not-a-real-dir-12345")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__])
