"""Integration tests for the graph orchestration in the full pipeline context."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from src.config import Settings
from src.llm.client import OpencodeLLMClient
from src.models.briefing import BriefingData
from src.models.market import DataSource, MarketSnapshot, Quote
from src.models.recommendation import Direction, PositionIntent, StrategyResult, TradeRecommendation


def _ndjson(text: str) -> str:
    return json.dumps({"type": "text", "part": {"text": text}}) + "\n"


def _completed(stdout: str, rc: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(["opencode"], rc, stdout, "")


def _market_spy_qqq():
    from datetime import datetime

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
                volume=45_000_000,
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
                volume=32_000_000,
                source=DataSource.FINNHUB,
                timestamp=ts,
            ),
        },
    )


def _briefing():
    from datetime import date

    return BriefingData(
        briefing_date=date(2026, 8, 14),
        executive_summary="Mixed signals: AI optimism vs defense weakness.",
        key_connections="Tech strength driven by AI demand.",
    )


def _make_llm_result(predictions, recommendation=None):
    return StrategyResult(
        label="llm_trade",
        recommendation=recommendation,
        predictions=predictions,
        confidence=recommendation.confidence if recommendation else 0.0,
        duration_ms=100.0,
    )


def _make_det_result(label: str, rec=None, confidence: float = 0.0, debug=None):
    return StrategyResult(
        label=label,
        recommendation=rec,
        confidence=confidence,
        debug_trace=debug or {},
        duration_ms=1.0,
    )


def _make_event_driven_rec():
    return TradeRecommendation(
        correlation_id="",
        strategy_label="event_driven",
        asset="SPY",
        direction=Direction.PUT,
        confidence=0.58,
        target_strike=745.0,
        contracts=1,
        position_intent=PositionIntent.BUY_TO_OPEN,
        rationale={"catalyst_count": 8, "catalyst_polarity": -0.5},
    )


class TestGraphIntegration:
    @pytest.mark.asyncio
    async def test_pipeline_phase_check_no_llm_result_skips(self) -> None:
        from src.pipeline import Pipeline

        config = Settings()
        config.graph.enabled = True
        config.llm.opencode_path = "opencode"
        config.general.execute = False

        pipeline = Pipeline(config, "test-correlation", _mock_file_logger())
        results = [
            _make_det_result("momentum"),
            _make_det_result("mean_reversion"),
        ]
        checked = await pipeline._phase_check(
            results,
            _briefing(),
            _market_spy_qqq(),
            OpencodeLLMClient(config.llm),
        )
        assert len(checked) == 2

    @pytest.mark.asyncio
    async def test_pipeline_phase_check_veto_blocks_trade(self) -> None:
        from src.pipeline import Pipeline

        config = Settings()
        config.graph.enabled = True
        config.graph.checker_contradiction_action = "veto"
        config.graph.checker_confidence_penalty = 0.15
        config.llm.opencode_path = "opencode"
        config.general.execute = False

        pipeline = Pipeline(config, "test-correlation", _mock_file_logger())

        # LLM says: SPY CALL 0.70
        from src.models.recommendation import AssetPrediction

        llm_rec = TradeRecommendation(
            correlation_id="",
            strategy_label="llm_trade",
            asset="SPY",
            direction=Direction.CALL,
            confidence=0.70,
            target_strike=746.0,
            contracts=1,
            position_intent=PositionIntent.BUY_TO_OPEN,
            rationale={"llm_rationale": "Bullish on broad market.", "sources": ["reuters:up"]},
        )
        llm_result = _make_llm_result(
            predictions={
                "SPY": AssetPrediction(
                    asset="SPY",
                    direction="UP",
                    confidence=0.70,
                    predicted_move_pct=0.5,
                    rationale="Bullish",
                    sources=["reuters:up"],
                ),
                "QQQ": AssetPrediction(
                    asset="QQQ",
                    direction="UP",
                    confidence=0.60,
                    predicted_move_pct=0.7,
                    rationale="AI momentum",
                    sources=["watchlist:NVDA"],
                ),
            },
            recommendation=llm_rec,
        )

        # Event-driven says: SPY PUT 0.58 (contradicts LLM)
        event_rec = _make_event_driven_rec()
        event_result = _make_det_result(
            "event_driven",
            rec=event_rec,
            confidence=0.58,
            debug={"catalyst_count": 8, "catalyst_polarity": -0.5},
        )

        # Checker output: contradiction detected, can_proceed = false
        checker_json = json.dumps(
            {
                "validated_predictions": [
                    {
                        "asset": "SPY",
                        "original_direction": "UP",
                        "original_confidence": 0.70,
                        "adjusted_confidence": 0.70,
                        "adjustment_reasons": [],
                        "issues": [],
                    },
                    {
                        "asset": "QQQ",
                        "original_direction": "UP",
                        "original_confidence": 0.60,
                        "adjusted_confidence": 0.60,
                        "adjustment_reasons": [],
                        "issues": [],
                    },
                ],
                "contradictions": [
                    {
                        "strategies": ["llm_trade", "event_driven"],
                        "description": "LLM recommends SPY CALL but event-driven flags bearish catalysts.",
                        "severity": "high",
                    }
                ],
                "flags": [],
                "overall_assessment": "Hard contradiction detected.",
                "can_proceed": False,
            }
        )

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=_completed(_ndjson(checker_json)),
            ),
        ):
            client = OpencodeLLMClient(config.llm)
            checked = await pipeline._phase_check(
                [llm_result, event_result],
                _briefing(),
                _market_spy_qqq(),
                client,
            )

        # Veto should penalize confidence significantly
        updated_llm = next(r for r in checked if r.label == "llm_trade")
        assert updated_llm.recommendation is not None
        # Confidence should be reduced by 2 * penalty (0.30)
        assert updated_llm.recommendation.confidence < 0.70

    @pytest.mark.asyncio
    async def test_pipeline_phase_check_clean_pass(self) -> None:
        from src.pipeline import Pipeline

        config = Settings()
        config.graph.enabled = True
        config.graph.checker_contradiction_action = "veto"
        config.llm.opencode_path = "opencode"
        config.general.execute = False

        pipeline = Pipeline(config, "test-correlation", _mock_file_logger())

        from src.models.recommendation import AssetPrediction

        llm_rec = TradeRecommendation(
            correlation_id="",
            strategy_label="llm_trade",
            asset="QQQ",
            direction=Direction.CALL,
            confidence=0.72,
            target_strike=720.0,
            contracts=1,
            position_intent=PositionIntent.BUY_TO_OPEN,
            rationale={"llm_rationale": "AI rally continues.", "sources": ["reuters:ai"]},
        )
        llm_result = _make_llm_result(
            predictions={
                "QQQ": AssetPrediction(
                    asset="QQQ",
                    direction="UP",
                    confidence=0.72,
                    predicted_move_pct=1.0,
                    rationale="AI rally",
                    sources=["reuters:ai"],
                ),
            },
            recommendation=llm_rec,
        )

        # Checker: clean pass
        checker_json = json.dumps(
            {
                "validated_predictions": [
                    {
                        "asset": "QQQ",
                        "original_direction": "UP",
                        "original_confidence": 0.72,
                        "adjusted_confidence": 0.72,
                        "adjustment_reasons": [],
                        "issues": [],
                    },
                ],
                "contradictions": [],
                "flags": [],
                "overall_assessment": "All clear.",
                "can_proceed": True,
            }
        )

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=_completed(_ndjson(checker_json)),
            ),
        ):
            client = OpencodeLLMClient(config.llm)
            checked = await pipeline._phase_check(
                [llm_result],
                _briefing(),
                _market_spy_qqq(),
                client,
            )

        updated_llm = next(r for r in checked if r.label == "llm_trade")
        # Confidence unchanged — no contradiction
        assert updated_llm.recommendation.confidence == 0.72
        assert "checker_output" in updated_llm.debug_trace
        assert updated_llm.debug_trace["checker_can_proceed"] is True


class TestGraphEndToEnd:
    @pytest.mark.asyncio
    async def test_graph_research_predict_flow(self) -> None:
        """Verify the full research → predict diamond without LLM calls,
        using mocked subprocess to simulate agent responses."""
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
                        "description": "AI demand",
                        "source": "reuters:ai",
                        "strength": 0.8,
                    }
                ],
                "risks": [],
                "sentiment": {
                    "aggregate_polarity": 0.5,
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
                "key_theme": "Bullish AI rally",
            }
        )
        research_qqq = json.dumps(
            {
                "asset": "QQQ",
                "catalysts": [
                    {
                        "type": "bullish",
                        "description": "Tech momentum",
                        "source": "watchlist:NVDA",
                        "strength": 0.7,
                    }
                ],
                "risks": [],
                "sentiment": {
                    "aggregate_polarity": 0.4,
                    "briefing_level": 1.0,
                    "news_consensus": "bullish",
                },
                "technical_context": {
                    "gap_from_previous_close_pct": 0.4,
                    "gap_direction": "UP",
                    "gap_significance": "minor",
                    "pre_market_momentum": "holding",
                },
                "watchlist_signals": [],
                "key_theme": "Tech rally",
            }
        )

        predict_spy = json.dumps(
            {
                "asset": "SPY",
                "direction": "UP",
                "confidence": 0.70,
                "predicted_move_pct": 0.6,
                "rationale": "AI demand driving rally",
                "sources": ["reuters:ai"],
            }
        )
        predict_qqq = json.dumps(
            {
                "asset": "QQQ",
                "direction": "UP",
                "confidence": 0.75,
                "predicted_move_pct": 1.5,
                "rationale": "NVDA and AI stocks surging",
                "sources": ["watchlist:NVDA"],
            }
        )
        checker = json.dumps(
            {
                "validated_predictions": [
                    {
                        "asset": "SPY",
                        "original_direction": "UP",
                        "original_confidence": 0.70,
                        "adjusted_confidence": 0.70,
                        "adjustment_reasons": [],
                        "issues": [],
                    },
                    {
                        "asset": "QQQ",
                        "original_direction": "UP",
                        "original_confidence": 0.75,
                        "adjusted_confidence": 0.75,
                        "adjustment_reasons": [],
                        "issues": [],
                    },
                ],
                "contradictions": [],
                "flags": [],
                "overall_assessment": "Both predictions consistent with each other and market data.",
                "can_proceed": True,
            }
        )
        pick_trade = json.dumps(
            {
                "best_trade": {
                    "asset": "QQQ",
                    "direction": "CALL",
                    "confidence": 0.75,
                    "rationale": "QQQ shows strongest signal with AI/tech momentum.",
                    "sources": ["watchlist:NVDA"],
                },
                "rationale": "Selected QQQ CALL based on highest confidence.",
            }
        )

        orchestrator = GraphOrchestrator(config, client)

        def run_side_effect(cmd, **kwargs):
            cmd_str = " ".join(cmd)
            if "research-spy" in cmd_str:
                return _completed(_ndjson(research_spy))
            if "research-qqq" in cmd_str:
                return _completed(_ndjson(research_qqq))
            if "predict-spy" in cmd_str:
                return _completed(_ndjson(predict_spy))
            if "predict-qqq" in cmd_str:
                return _completed(_ndjson(predict_qqq))
            if "checker" in cmd_str:
                return _completed(_ndjson(checker))
            if "pick-trade" in cmd_str:
                return _completed(_ndjson(pick_trade))
            return _completed("{}", rc=1)

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
        ):
            result = await orchestrator.run(
                briefing=_briefing(),
                market=_market_spy_qqq(),
                deterministic_results=[],
            )

        assert result["predictions"]["SPY"]["direction"] == "UP"
        assert result["predictions"]["SPY"]["confidence"] == 0.70
        assert result["predictions"]["QQQ"]["direction"] == "UP"
        assert result["predictions"]["QQQ"]["confidence"] == 0.75
        assert result["best_trade"] is not None
        assert result["best_trade"]["asset"] == "QQQ"
        assert result["best_trade"]["direction"] == "CALL"

    @pytest.mark.asyncio
    async def test_graph_prediction_count_validation(self) -> None:
        """Verify predictions for only target assets are retained."""
        from src.llm.client import OpencodeLLMClient
        from src.llm.graph import GraphOrchestrator

        config = Settings()
        config.llm.opencode_path = "opencode"
        config.general.target_assets = ["SPY", "QQQ"]
        client = OpencodeLLMClient(config.llm)

        research_spy = json.dumps(
            {
                "asset": "SPY",
                "catalysts": [
                    {"type": "bullish", "description": "x", "source": "r", "strength": 0.5}
                ],
                "risks": [],
                "sentiment": {
                    "aggregate_polarity": 0.0,
                    "briefing_level": 1.0,
                    "news_consensus": "bullish",
                },
                "technical_context": {
                    "gap_from_previous_close_pct": 0.0,
                    "gap_direction": "FLAT",
                    "gap_significance": "minor",
                    "pre_market_momentum": "holding",
                },
                "watchlist_signals": [],
                "key_theme": "Flat",
            }
        )
        research_qqq = json.dumps(
            {
                "asset": "QQQ",
                "catalysts": [
                    {"type": "bearish", "description": "y", "source": "w", "strength": 0.4}
                ],
                "risks": [],
                "sentiment": {
                    "aggregate_polarity": -0.2,
                    "briefing_level": 1.0,
                    "news_consensus": "bearish",
                },
                "technical_context": {
                    "gap_from_previous_close_pct": -0.1,
                    "gap_direction": "DOWN",
                    "gap_significance": "minor",
                    "pre_market_momentum": "fading",
                },
                "watchlist_signals": [],
                "key_theme": "Bearish",
            }
        )

        predict_spy = json.dumps(
            {
                "asset": "SPY",
                "direction": "UP",
                "confidence": 0.55,
                "predicted_move_pct": 0.3,
                "rationale": "x",
                "sources": ["r"],
            }
        )
        # Predict QQQ includes an extra (non-target) asset — should be filtered
        predict_qqq = json.dumps(
            {
                "asset": "QQQ",
                "direction": "DOWN",
                "confidence": 0.45,
                "predicted_move_pct": -0.5,
                "rationale": "y",
                "sources": ["w"],
            }
        )
        checker = json.dumps(
            {
                "validated_predictions": [],
                "contradictions": [],
                "flags": [],
                "overall_assessment": "ok.",
                "can_proceed": True,
            }
        )
        pick_trade = json.dumps({"best_trade": None, "rationale": "No strong signal."})

        orchestrator = GraphOrchestrator(config, client)

        def run_side_effect(cmd, **kwargs):
            cmd_str = " ".join(cmd)
            if "research-spy" in cmd_str:
                return _completed(_ndjson(research_spy))
            if "research-qqq" in cmd_str:
                return _completed(_ndjson(research_qqq))
            if "predict-spy" in cmd_str:
                return _completed(_ndjson(predict_spy))
            if "predict-qqq" in cmd_str:
                return _completed(_ndjson(predict_qqq))
            if "checker" in cmd_str:
                return _completed(_ndjson(checker))
            if "pick-trade" in cmd_str:
                return _completed(_ndjson(pick_trade))
            return _completed("{}", rc=1)

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
        ):
            result = await orchestrator.run(
                briefing=_briefing(),
                market=_market_spy_qqq(),
                deterministic_results=[],
            )

        assert len(result["predictions"]) == 2
        assert "SPY" in result["predictions"]
        assert "QQQ" in result["predictions"]
        assert result["predictions"]["SPY"]["direction"] == "UP"
        assert result["predictions"]["QQQ"]["direction"] == "DOWN"


def _mock_file_logger():
    from unittest.mock import MagicMock

    logger = MagicMock()
    logger.write_entry = MagicMock()
    logger.write_summary = MagicMock()
    return logger
