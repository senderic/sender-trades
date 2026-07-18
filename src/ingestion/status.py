"""Load upstream atlas-morning-briefing ``status.json`` ground truth.

The briefing markdown alone cannot always tell us whether the upstream
LLM layer ran: a future upstream fix could change the deterministic
fallback string that :func:`src.ingestion.parser._classify_quality`
keys off. The companion ``status.json`` file written by the upstream
pipeline carries the authoritative ``intelligence_enabled`` flag and
should be consulted alongside the parsed briefing.

See ``LESSONS_LEARNED.md`` (2026-07-18 incident) for motivation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class BriefingStatus(BaseModel):
    """Subset of the upstream ``status.json`` fields this project cares about."""

    timestamp: str = ""
    intelligence_enabled: bool = True
    papers_found: int = 0
    blogs_found: int = 0
    news_found: int = 0
    stocks_fetched: int = 0
    errors: list[str] = Field(default_factory=list)


def read_briefing_status(directory: str | Path) -> BriefingStatus | None:
    """Read ``status.json`` from an atlas-morning-briefing directory.

    A single file read per run; safe to call on the same directory used
    by :func:`src.ingestion.parser.find_todays_briefing`.

    Args:
        directory: Directory containing ``status.json`` (typically the
            atlas-morning-briefing checkout, e.g. ``~/atlas-morning-briefing``).

    Returns:
        Parsed BriefingStatus, or ``None`` if the file is missing or
        unreadable. Missing status never raises — callers should treat
        ``None`` as "unknown" and fall back to markdown-based detection.
    """
    status_path = Path(directory).expanduser() / "status.json"
    if not status_path.is_file():
        return None
    try:
        raw: Any = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("status_json_unreadable", path=str(status_path), error=str(e))
        return None
    if not isinstance(raw, dict):
        logger.warning("status_json_not_object", path=str(status_path))
        return None
    return BriefingStatus(**raw)
