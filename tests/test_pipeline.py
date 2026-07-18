import uuid

import pytest

from src.config import Settings
from src.logging_setup import JSONFileLogger, setup_logging
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
