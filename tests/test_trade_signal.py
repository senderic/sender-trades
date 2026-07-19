from __future__ import annotations

import json
import subprocess
from datetime import date
from typing import Any
from unittest.mock import patch

import pytest

from src.config import Settings
from src.llm.trade_signal import LLMTradeStrategy, _parse_pick
from src.models.briefing import BriefingData, BriefingQuality
from src.models.market import DataSource, MarketSnapshot, Quote
from src.models.recommendation import Direction


def _ndjson(text: str) -> str:
    return json.dumps({"type": "text", "part": {"text": text}}) + "\n"


def _completed(stdout: str, rc: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(["opencode"], rc, stdout, "")


def _stock_response(pick: dict[str, Any]) -> str:
    """Wrap a pick dict in the NDJSON stream the opencode CLI emits."""
    return _ndjson(json.dumps(pick))


@pytest.fixture
def briefing_with_sentiment() -> BriefingData:
    return BriefingData(
        briefing_date=date(2026, 7, 18),
        executive_summary=("Market is bearish today following weak earnings and a hawkish Fed."),
        key_connections="Tech selloff broad; catalysts skew negative.",
        briefing_quality=BriefingQuality.FULL,
    )


@pytest.fixture
def market_with_quotes() -> MarketSnapshot:
    from datetime import datetime

    ts = datetime.now()
    return MarketSnapshot(
        quotes={
            "SPY": Quote(
                symbol="SPY",
                current_price=743.29,
                open_price=736.00,
                high_price=744.00,
                low_price=735.00,
                previous_close=739.00,
                change_pct=+0.58,
                volume=45_000_000,
                source=DataSource.FINNHUB,
                timestamp=ts,
            ),
            "QQQ": Quote(
                symbol="QQQ",
                current_price=695.33,
                open_price=691.00,
                high_price=697.00,
                low_price=690.00,
                previous_close=706.00,
                change_pct=-1.51,
                volume=32_000_000,
                source=DataSource.FINNHUB,
                timestamp=ts,
            ),
        },
    )


class TestParsePick:
    def test_bare_json(self) -> None:
        resp = '{"asset": "QQQ", "direction": "PUT", "confidence": 0.72, "rationale": "x"}'
        pick = _parse_pick(resp)
        assert pick is not None
        assert pick["asset"] == "QQQ"
        assert pick["direction"] == "PUT"

    def test_markdown_code_fence(self) -> None:
        resp = (
            "```json\n"
            + json.dumps({"asset": "SPY", "direction": "CALL", "confidence": 0.5, "rationale": "x"})
            + "\n```"
        )
        pick = _parse_pick(resp)
        assert pick is not None
        assert pick["asset"] == "SPY"

    def test_json_with_prose_around(self) -> None:
        resp = (
            "Here is my pick:\n"
            '{"asset": "SPY", "direction": "CALL", "confidence": 0.6, "rationale": "r"}\n'
            "Hope this helps."
        )
        pick = _parse_pick(resp)
        assert pick is not None
        assert pick["direction"] == "CALL"

    def test_unparseable_returns_none(self) -> None:
        assert _parse_pick("no json here") is None
        assert _parse_pick("") is None
        assert _parse_pick("{ broken json:") is None


class TestLLMTradeStrategy:
    def _strategy(self, config: Settings | None = None) -> LLMTradeStrategy:
        cfg = config or Settings()
        # Inject a fake opencode path so .available is False until patched.
        cfg.llm.opencode_path = "opencode"
        return LLMTradeStrategy(cfg)

    @pytest.mark.asyncio
    async def test_opencode_unavailable_abstains(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        cfg = Settings()
        cfg.llm.opencode_path = "opencode-not-on-path"
        strategy = LLMTradeStrategy(cfg)
        result = await strategy.evaluate(briefing_with_sentiment, market_with_quotes)
        assert result.recommendation is None
        assert result.debug_trace["skip_reason"] == "opencode_unavailable"

    @pytest.mark.asyncio
    async def test_valid_pick_builds_recommendation(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        strategy = self._strategy()
        pick = {
            "asset": "QQQ",
            "direction": "PUT",
            "confidence": 0.78,
            "rationale": "Tech selloff; weak QQQ pre-market.",
        }
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=_completed(_stock_response(pick)),
            ),
        ):
            result = await strategy.evaluate(briefing_with_sentiment, market_with_quotes)
        assert result.recommendation is not None
        assert result.recommendation.asset == "QQQ"
        assert result.recommendation.direction == Direction.PUT
        assert result.recommendation.confidence == 0.78
        assert result.recommendation.strategy_label == "llm_trade"
        assert result.debug_trace["served_by"] == "opencode/deepseek-v4-flash-free"
        assert result.debug_trace["paid_used"] is False
        # Rationale preserves the LLM's reasoning.
        assert "tech selloff" in result.recommendation.rationale["llm_rationale"].lower()

    @pytest.mark.asyncio
    async def test_confidence_clamped_to_range(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        strategy = self._strategy()
        # Confidence above 1.0 -> clamp to 1.0.
        pick = {"asset": "SPY", "direction": "CALL", "confidence": 1.5, "rationale": "x"}
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=_completed(_stock_response(pick)),
            ),
        ):
            result = await strategy.evaluate(briefing_with_sentiment, market_with_quotes)
        assert result.recommendation is not None
        assert result.recommendation.confidence == 1.0

    @pytest.mark.asyncio
    async def test_low_confidence_abstains(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        cfg = Settings()
        cfg.llm.trade_signal_min_confidence = 0.60
        strategy = self._strategy(cfg)
        pick = {"asset": "SPY", "direction": "CALL", "confidence": 0.42, "rationale": "x"}
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=_completed(_stock_response(pick)),
            ),
        ):
            result = await strategy.evaluate(briefing_with_sentiment, market_with_quotes)
        assert result.recommendation is None
        assert result.debug_trace["skip_reason"] == "below_min_confidence"

    @pytest.mark.asyncio
    async def test_asset_out_of_universe_abstains(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        strategy = self._strategy()
        pick = {
            "asset": "AAPL",  # not in target_assets
            "direction": "CALL",
            "confidence": 0.8,
            "rationale": "x",
        }
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=_completed(_stock_response(pick)),
            ),
        ):
            result = await strategy.evaluate(briefing_with_sentiment, market_with_quotes)
        assert result.recommendation is None
        assert result.debug_trace["skip_reason"] == "llm_asset_out_of_universe"

    @pytest.mark.asyncio
    async def test_invalid_direction_abstains(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        strategy = self._strategy()
        pick = {"asset": "SPY", "direction": "HOLD", "confidence": 0.7, "rationale": "x"}
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=_completed(_stock_response(pick)),
            ),
        ):
            result = await strategy.evaluate(briefing_with_sentiment, market_with_quotes)
        assert result.recommendation is None
        assert result.debug_trace["skip_reason"] == "llm_invalid_direction"

    @pytest.mark.asyncio
    async def test_unparseable_response_abstains(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        strategy = self._strategy()
        # LLM prose with no JSON.
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=_completed(_ndjson("I cannot decide today.")),
            ),
        ):
            result = await strategy.evaluate(briefing_with_sentiment, market_with_quotes)
        assert result.recommendation is None
        assert result.debug_trace["skip_reason"] == "llm_unparseable"

    @pytest.mark.asyncio
    async def test_no_response_abstains(self, briefing_with_sentiment, market_with_quotes) -> None:
        strategy = self._strategy()
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=_completed(""),
            ),
        ):
            result = await strategy.evaluate(briefing_with_sentiment, market_with_quotes)
        assert result.recommendation is None
        assert result.debug_trace["skip_reason"] == "llm_no_response"

    @pytest.mark.asyncio
    async def test_all_models_fail_abstains(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        strategy = self._strategy()
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=_completed("", rc=1),
            ),
        ):
            result = await strategy.evaluate(briefing_with_sentiment, market_with_quotes)
        assert result.recommendation is None
        # Subprocess failed for every model -> client.invoke returned None
        # -> strategy hit the no-response branch.
        assert result.debug_trace["skip_reason"] == "llm_no_response"

    @pytest.mark.asyncio
    async def test_paid_model_serves_pick_paid_used_flag(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        # Force primary Zen to fail so chain falls through to paid glm-5.2.
        cfg = Settings()
        cfg.llm.zen_models = ["opencode/deepseek-v4-flash-free"]
        cfg.llm.paid_go_models = ["opencode-go/glm-5.2"]
        cfg.llm.opencode_path = "opencode"
        strategy = LLMTradeStrategy(cfg)

        pick = {
            "asset": "SPY",
            "direction": "CALL",
            "confidence": 0.66,
            "rationale": "Rotation into defense supports broad market.",
        }

        def run_side_effect(cmd, **kwargs):
            if "opencode/deepseek-v4-flash-free" in cmd:
                return subprocess.CompletedProcess(cmd, 1, "", "fail")
            return _completed(_stock_response(pick))

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
        ):
            result = await strategy.evaluate(briefing_with_sentiment, market_with_quotes)
        assert result.recommendation is not None
        assert result.debug_trace["paid_used"] is True
        assert result.debug_trace["served_by"] == "opencode-go/glm-5.2"
        assert result.recommendation.rationale["llm_paid_used"] is True


if __name__ == "__main__":
    pytest.main([__file__])
