import json
import subprocess
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import Settings
from src.logging_setup import setup_logging
from src.pipeline import Pipeline


@pytest.fixture
def pipeline(tmp_path) -> Pipeline:
    config = Settings()
    config.logging.json_dir = str(tmp_path / "logs")
    config.atlas_briefing.directory = str(tmp_path / "briefings")
    cid = uuid.uuid4().hex[:12]
    logger = setup_logging(config, cid)
    return Pipeline(config, cid, logger)


@pytest.mark.asyncio
async def test_pipeline_handles_missing_briefing(pipeline: Pipeline) -> None:
    result = await pipeline.run()
    assert result.briefing is None
    assert result.correlation_id != ""
    assert hasattr(result, "duration_seconds")
    assert result.duration_seconds > 0


@pytest.mark.asyncio
async def test_pipeline_handles_market_data_empty(pipeline: Pipeline) -> None:
    result = await pipeline.run()
    assert result.market is not None
    assert result.decision is not None
    assert result.decision.recommendation is None


def _write_degraded_briefing(directory: Path) -> Path:
    """Write a degraded-format Atlas briefing fixture to disk."""
    directory.mkdir(parents=True, exist_ok=True)
    degraded_md = (
        "# Atlas Morning Briefing\n\n"
        "## Executive Summary\n"
        "Synthesis unavailable for today's briefing. See sections below.\n\n"
        "## AI & Tech News\n\n"
        "### Some raw headline about AI surge\n"
        "*Source: reuters.com*\n\n"
        "[Read more](https://example.com/x)\n"
    )
    path = directory / "Atlas-Briefing-2026.07.18.md"
    path.write_text(degraded_md)
    return path


def _ndjson(text: str) -> str:
    return json.dumps({"type": "text", "part": {"text": text}}) + "\n"


@pytest.mark.asyncio
async def test_pipeline_resynthesizes_degraded_briefing(tmp_path) -> None:
    config = Settings()
    config.logging.json_dir = str(tmp_path / "logs")
    briefing_dir = tmp_path / "briefings"
    config.atlas_briefing.directory = str(briefing_dir)
    config.atlas_briefing.snapshot_enabled = False
    config.llm.enabled = True
    config.llm.opencode_path = "opencode"

    _write_degraded_briefing(briefing_dir)

    cid = uuid.uuid4().hex[:12]
    logger = setup_logging(config, cid)
    pipeline = Pipeline(config, cid, logger)

    completed = subprocess.CompletedProcess(
        ["opencode"], 0, _ndjson("Markets lean bullish on AI earnings beats."), ""
    )
    with (
        patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
        patch("src.llm.client.subprocess.run", return_value=completed),
    ):
        briefing = await pipeline._phase_ingest_briefing()

    assert briefing is not None
    assert briefing.briefing_quality.value == "full"
    assert "bullish" in briefing.executive_summary.lower()
    assert briefing.macro_sentiment is not None


@pytest.mark.asyncio
async def test_pipeline_upstream_intelligence_disabled_forces_degraded(tmp_path) -> None:
    config = Settings()
    config.logging.json_dir = str(tmp_path / "logs")
    briefing_dir = tmp_path / "briefings"
    config.atlas_briefing.directory = str(briefing_dir)
    config.atlas_briefing.snapshot_enabled = False
    # Disable LLM re-synthesis so we observe only the status.json effect.
    config.llm.enabled = False

    _write_degraded_briefing(briefing_dir)
    (briefing_dir / "status.json").write_text(json.dumps({"intelligence_enabled": False}))

    cid = uuid.uuid4().hex[:12]
    logger = setup_logging(config, cid)
    pipeline = Pipeline(config, cid, logger)
    briefing = await pipeline._phase_ingest_briefing()

    assert briefing is not None
    assert briefing.briefing_quality.value == "degraded"
    assert briefing.macro_sentiment is None
