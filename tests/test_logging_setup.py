"""Unit tests for JSONFileLogger's run-log file writer."""

from __future__ import annotations

import json
from pathlib import Path

from src.logging_setup import JSONFileLogger
from src.timezone import today_local


class TestJSONFileLoggerWriteEntry:
    def test_writes_to_jsonl_extension(self, tmp_path: Path) -> None:
        logger = JSONFileLogger(str(tmp_path), "corr123")
        logger.write_entry({"event": "pipeline_start"})

        day_dir = tmp_path / today_local().isoformat()
        run_files = list(day_dir.glob("run-corr123.*"))
        assert len(run_files) == 1
        assert run_files[0].suffix == ".jsonl"

    def test_multiple_entries_form_valid_jsonl(self, tmp_path: Path) -> None:
        logger = JSONFileLogger(str(tmp_path), "corr456")
        logger.write_entry({"event": "a"})
        logger.write_entry({"event": "b"})
        logger.write_entry({"event": "c"})

        run_file = next(logger.ensure_directory().glob("run-corr456.jsonl"))
        lines = run_file.read_text().strip().split("\n")
        assert len(lines) == 3
        events = [json.loads(line)["event"] for line in lines]
        assert events == ["a", "b", "c"]

    def test_write_summary_is_a_single_valid_json_object(self, tmp_path: Path) -> None:
        logger = JSONFileLogger(str(tmp_path), "corr789")
        logger.write_summary({"decision": "no_trade"})

        summary_file = next(logger.ensure_directory().glob("summary-corr789.json"))
        data = json.loads(summary_file.read_text())
        assert data == {"decision": "no_trade"}
