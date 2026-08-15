from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from src.config import GraphConfig, Settings
from src.llm.graph import (
    _build_checker_prompt,
    _build_predict_prompt,
    _build_research_prompt,
    _extract_json,
)
from src.models.market import DataSource, MarketSnapshot, NewsHeadline, Quote


class TestExtractJson:
    def test_bare_json(self) -> None:
        result = _extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_prose(self) -> None:
        result = _extract_json('Here is data: {"a": 1, "b": 2} and more.')
        assert result == {"a": 1, "b": 2}

    def test_code_fence(self) -> None:
        result = _extract_json('```json\n{"x": [1,2,3]}\n```')
        assert result == {"x": [1, 2, 3]}

    def test_empty(self) -> None:
        assert _extract_json("") is None
        assert _extract_json("   ") is None

    def test_unparseable(self) -> None:
        assert _extract_json("{broken}") is None
        assert _extract_json("not json at all") is None

    def test_non_dict_json(self) -> None:
        assert _extract_json("[1, 2, 3]") is None
        assert _extract_json('"string"') is None


class TestBuildResearchPrompt:
    def test_includes_asset_and_quote(self) -> None:
        from datetime import datetime

        market = MarketSnapshot(
            quotes={
                "SPY": Quote(
                    symbol="SPY",
                    current_price=745.0,
                    open_price=742.0,
                    high_price=746.0,
                    low_price=741.0,
                    previous_close=740.0,
                    change_pct=0.68,
                    volume=45_000_000,
                    source=DataSource.FINNHUB,
                    timestamp=datetime.now(),
                ),
            },
        )
        from datetime import date as dt_date

        from src.models.briefing import BriefingData

        briefing = BriefingData(briefing_date=dt_date.today(), executive_summary="Bullish market")
        prompt = _build_research_prompt(briefing, market, "SPY")
        assert "Research target: SPY" in prompt
        assert "Bullish market" in prompt
        assert "$745.00" in prompt
        assert "+0.68%" in prompt

    def test_includes_news(self) -> None:

        market = MarketSnapshot(
            news=[
                NewsHeadline(
                    title="Tech Stocks Rally",
                    source="reuters.com",
                    url="https://example.com",
                    snippet="Tech up",
                    polarity=0.6,
                ),
            ],
        )
        from datetime import date as dt_date3

        from src.models.briefing import BriefingData

        briefing = BriefingData(briefing_date=dt_date3.today())
        prompt = _build_research_prompt(briefing, market, "QQQ")
        assert "Tech Stocks Rally" in prompt


class TestBuildPredictPrompt:
    def test_embedds_research_json(self) -> None:
        research = {
            "asset": "SPY",
            "catalysts": [
                {
                    "type": "bullish",
                    "description": "AI rally",
                    "source": "reuters:ai",
                    "strength": 0.8,
                }
            ],
            "key_theme": "AI momentum",
        }
        prompt = _build_predict_prompt(research, "SPY")
        assert '"asset": "SPY"' in prompt
        assert '"AI rally"' in prompt
        assert "directional prediction JSON" in prompt


class TestBuildCheckerPrompt:
    def test_includes_all_predictions(self) -> None:
        predict_spy = {
            "asset": "SPY",
            "direction": "UP",
            "confidence": 0.65,
            "predicted_move_pct": 0.5,
            "rationale": "Bullish sentiment",
            "sources": ["reuters:bullish"],
        }
        predict_qqq = {
            "asset": "QQQ",
            "direction": "UP",
            "confidence": 0.72,
            "predicted_move_pct": 1.2,
            "rationale": "AI optimism",
            "sources": ["watchlist:NVDA"],
        }
        deterministic = [
            {
                "label": "event_driven",
                "recommendation": {
                    "asset": "SPY",
                    "direction": "PUT",
                    "confidence": 0.58,
                },
                "confidence": 0.58,
                "debug_trace": {"catalyst_polarity": -0.5},
            }
        ]
        from datetime import datetime

        market = MarketSnapshot(
            quotes={
                "SPY": Quote(
                    symbol="SPY",
                    current_price=743.0,
                    open_price=741.0,
                    high_price=744.0,
                    low_price=740.0,
                    previous_close=739.0,
                    change_pct=0.54,
                    volume=45_000_000,
                    source=DataSource.FINNHUB,
                    timestamp=datetime.now(),
                ),
                "QQQ": Quote(
                    symbol="QQQ",
                    current_price=715.0,
                    open_price=710.0,
                    high_price=717.0,
                    low_price=709.0,
                    previous_close=712.0,
                    change_pct=0.42,
                    volume=32_000_000,
                    source=DataSource.FINNHUB,
                    timestamp=datetime.now(),
                ),
            },
        )
        graph_config = GraphConfig()
        prompt = _build_checker_prompt(
            predict_spy, predict_qqq, deterministic, market, graph_config
        )
        assert "SPY prediction" in prompt
        assert "QQQ prediction" in prompt
        assert "event_driven" in prompt
        assert "PUT" in prompt
        assert "gap-fade threshold" in prompt

    def test_handles_none_predictions(self) -> None:
        deterministic: list[dict[str, Any]] = []
        from datetime import datetime

        market = MarketSnapshot(
            quotes={
                "SPY": Quote(
                    symbol="SPY",
                    current_price=740.0,
                    open_price=739.0,
                    high_price=741.0,
                    low_price=738.0,
                    previous_close=738.0,
                    change_pct=0.27,
                    volume=40_000_000,
                    source=DataSource.FINNHUB,
                    timestamp=datetime.now(),
                ),
            },
        )
        graph_config = GraphConfig()
        prompt = _build_checker_prompt(None, None, deterministic, market, graph_config)
        assert "FAILED" in prompt


class TestGraphOrchestratorInit:
    def test_creates_with_valid_config(self) -> None:
        from src.llm.client import OpencodeLLMClient
        from src.llm.graph import GraphOrchestrator

        config = Settings()
        client = OpencodeLLMClient(config.llm)
        orchestrator = GraphOrchestrator(config, client)
        assert orchestrator.AGENT_RESEARCH_SPY == "research-spy"
        assert orchestrator.AGENT_CHECKER == "checker"


class TestGraphFailureResult:
    def test_graph_failure_sets_flags(self) -> None:
        from src.llm.graph import _graph_failure

        trace: dict[str, Any] = {}
        result = _graph_failure(trace, "test failure")
        assert result["graph_failed"] is True
        assert result["predictions"] == {}
        assert result["best_trade"] is None
        assert trace["graph_failed"] is True


class TestGraphOrchestratorRun:
    @pytest.mark.asyncio
    async def test_all_research_fails_returns_failure(self) -> None:
        from src.llm.client import OpencodeLLMClient
        from src.llm.graph import GraphOrchestrator

        config = Settings()
        config.llm.opencode_path = "opencode"
        client = OpencodeLLMClient(config.llm)
        orchestrator = GraphOrchestrator(config, client)

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", return_value=_empty_failed()),
        ):
            result = await orchestrator.run(
                briefing=_empty_briefing(),
                market=_empty_market(),
                deterministic_results=[],
            )
        assert "predictions" in result
        assert result["trace"].get("graph_failed") is True

    @pytest.mark.asyncio
    async def test_research_succeeds_predict_fails(self) -> None:
        from src.llm.client import OpencodeLLMClient
        from src.llm.graph import GraphOrchestrator

        config = Settings()
        config.llm.opencode_path = "opencode"
        client = OpencodeLLMClient(config.llm)

        research_spy = json.dumps(
            {
                "asset": "SPY",
                "catalysts": [
                    {
                        "type": "bullish",
                        "description": "Tech rally",
                        "source": "reuters:tech",
                        "strength": 0.7,
                    }
                ],
                "risks": [],
                "sentiment": {
                    "aggregate_polarity": 0.5,
                    "briefing_level": 1.0,
                    "news_consensus": "bullish",
                },
                "technical_context": {
                    "gap_from_previous_close_pct": 0.5,
                    "gap_direction": "UP",
                    "gap_significance": "minor",
                    "pre_market_momentum": "holding",
                },
                "watchlist_signals": [],
                "key_theme": "Bullish",
            }
        )
        research_qqq = json.dumps(
            {
                "asset": "QQQ",
                "catalysts": [
                    {
                        "type": "bullish",
                        "description": "AI demand",
                        "source": "watchlist:NVDA",
                        "strength": 0.8,
                    }
                ],
                "risks": [],
                "sentiment": {
                    "aggregate_polarity": 0.6,
                    "briefing_level": 1.0,
                    "news_consensus": "bullish",
                },
                "technical_context": {
                    "gap_from_previous_close_pct": 0.3,
                    "gap_direction": "UP",
                    "gap_significance": "minor",
                    "pre_market_momentum": "strengthening",
                },
                "watchlist_signals": [],
                "key_theme": "AI momentum",
            }
        )

        predict_spy_ok = json.dumps(
            {
                "asset": "SPY",
                "direction": "UP",
                "confidence": 0.65,
                "predicted_move_pct": 0.5,
                "rationale": "Bullish sentiment",
                "sources": ["reuters:tech"],
            }
        )

        # Predict QQQ: valid prediction but no best_trade
        predict_qqq_ok = json.dumps(
            {
                "asset": "QQQ",
                "direction": "DOWN",
                "confidence": 0.45,
                "predicted_move_pct": -0.8,
                "rationale": "Valuation concerns despite AI",
                "sources": ["watchlist:NVDA"],
            }
        )

        # Checker response: clean pass
        checker_ok = json.dumps(
            {
                "validated_predictions": [
                    {
                        "asset": "SPY",
                        "original_direction": "UP",
                        "original_confidence": 0.65,
                        "adjusted_confidence": 0.65,
                        "adjustment_reasons": [],
                        "issues": [],
                    },
                    {
                        "asset": "QQQ",
                        "original_direction": "DOWN",
                        "original_confidence": 0.45,
                        "adjusted_confidence": 0.45,
                        "adjustment_reasons": [],
                        "issues": [],
                    },
                ],
                "contradictions": [],
                "flags": [],
                "overall_assessment": "Both predictions are internally consistent and well-supported.",
                "can_proceed": True,
            }
        )

        # Pick trade: SPY CALL
        pick_trade_ok = json.dumps(
            {
                "best_trade": {
                    "asset": "SPY",
                    "direction": "CALL",
                    "confidence": 0.65,
                    "rationale": "SPY has the stronger bullish signal with clean evidence.",
                    "sources": ["reuters:tech"],
                },
                "rationale": "Selected SPY CALL based on highest adjusted confidence.",
            }
        )

        orchestrator = GraphOrchestrator(config, client)

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                side_effect=_build_mock_subprocess(
                    research_spy,
                    research_qqq,
                    predict_spy_ok,
                    predict_qqq_ok,
                    checker_ok,
                    pick_trade_ok,
                ),
            ) as _,
        ):
            result = await orchestrator.run(
                briefing=_empty_briefing(),
                market=_market_with_spy_qqq(),
                deterministic_results=[],
            )

        assert result["predictions"]
        assert "best_trade" in result
        assert result["trace"].get("graph_failed") is not True


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _empty_briefing():
    from datetime import date

    from src.models.briefing import BriefingData

    return BriefingData(briefing_date=date.today())


def _empty_market():
    from src.models.market import MarketSnapshot

    return MarketSnapshot()


def _market_with_spy_qqq():
    from datetime import datetime

    from src.models.market import DataSource, MarketSnapshot, Quote

    ts = datetime.now()
    return MarketSnapshot(
        quotes={
            "SPY": Quote(
                symbol="SPY",
                current_price=745.0,
                open_price=743.0,
                high_price=746.0,
                low_price=742.0,
                previous_close=740.0,
                change_pct=0.68,
                volume=45000000,
                source=DataSource.FINNHUB,
                timestamp=ts,
            ),
            "QQQ": Quote(
                symbol="QQQ",
                current_price=715.0,
                open_price=712.0,
                high_price=717.0,
                low_price=711.0,
                previous_close=712.0,
                change_pct=0.42,
                volume=32000000,
                source=DataSource.FINNHUB,
                timestamp=ts,
            ),
        },
    )


def _subprocess_ok(stdout_text: str) -> Any:
    import subprocess

    return subprocess.CompletedProcess(["opencode"], 0, _ndjson_wrap(stdout_text), "")


def _ndjson_wrap(text: str) -> str:
    return json.dumps({"type": "text", "part": {"text": text}}) + "\n"


def _empty_failed():
    import subprocess as sp

    return sp.CompletedProcess(["opencode"], 1, "", "failed")


def _build_mock_subprocess(
    research_spy: str,
    research_qqq: str,
    predict_spy: str,
    predict_qqq: str,
    checker: str,
    pick_trade: str,
):
    """Build a side_effect function for subprocess.run that dispatches by --agent arg."""
    import subprocess as sp

    def side_effect(cmd, **kwargs):
        cmd_str = " ".join(cmd)
        if "research-spy" in cmd_str:
            return sp.CompletedProcess(cmd, 0, _ndjson_wrap(research_spy), "")
        if "research-qqq" in cmd_str:
            return sp.CompletedProcess(cmd, 0, _ndjson_wrap(research_qqq), "")
        if "predict-spy" in cmd_str:
            return sp.CompletedProcess(cmd, 0, _ndjson_wrap(predict_spy), "")
        if "predict-qqq" in cmd_str:
            return sp.CompletedProcess(cmd, 0, _ndjson_wrap(predict_qqq), "")
        if "checker" in cmd_str:
            return sp.CompletedProcess(cmd, 0, _ndjson_wrap(checker), "")
        if "pick-trade" in cmd_str:
            return sp.CompletedProcess(cmd, 0, _ndjson_wrap(pick_trade), "")
        return sp.CompletedProcess(cmd, 1, "", "unknown agent")

    return side_effect
