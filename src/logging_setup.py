from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from src.config import Settings


class JSONFileLogger:
    """Writes structured JSON log entries and summaries to dated directories."""

    def __init__(self, log_dir: str | Path, correlation_id: str):
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.correlation_id = correlation_id
        self.entries: list[dict[str, Any]] = []

    def ensure_directory(self) -> Path:
        """Create and return the date-stamped log subdirectory.

        Returns:
            Path to today's log directory.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        day_dir = self.log_dir / today
        day_dir.mkdir(parents=True, exist_ok=True)
        return day_dir

    def write_entry(self, entry: dict[str, Any]) -> None:
        """Append a JSON log entry and persist it to the daily run file.

        Args:
            entry: Dictionary of structured log data.
        """
        self.entries.append(entry)
        day_dir = self.ensure_directory()
        log_file = day_dir / f"run-{self.correlation_id}.json"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def write_summary(self, summary: dict[str, Any]) -> None:
        """Write a final pipeline summary JSON file for the run.

        Args:
            summary: Dictionary of summary data to persist.
        """
        day_dir = self.ensure_directory()
        summary_file = day_dir / f"summary-{self.correlation_id}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)


def _json_serializer(obj: object, *args: Any, **kwargs: Any) -> bytes:
    kwargs.pop("default", None)
    return json.dumps(obj, default=str, *args, **kwargs).encode("utf-8")


def setup_logging(settings: Settings, correlation_id: str) -> JSONFileLogger:
    """Configure structlog and return a JSON file logger for the run.

    Args:
        settings: Application settings containing logging configuration.
        correlation_id: Unique identifier for the current pipeline run.

    Returns:
        Initialised JSONFileLogger instance.
    """
    log_level = getattr(logging, settings.logging.level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.dev.ConsoleRenderer()
            if sys.stderr.isatty()
            else structlog.processors.JSONRenderer(serializer=_json_serializer),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    file_logger = JSONFileLogger(settings.logging.json_dir, correlation_id)

    def log_event(
        logger: structlog.stdlib.BoundLogger,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        event_dict.setdefault("correlation_id", correlation_id)
        event_dict.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        file_logger.write_entry(event_dict)
        return event_dict

    structlog.configure(processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        log_event,
        structlog.dev.ConsoleRenderer()
        if sys.stderr.isatty()
        else structlog.processors.JSONRenderer(serializer=_json_serializer),
    ])

    return file_logger
