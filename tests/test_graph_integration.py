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


def _research_doc(asset: str, polarity: float = 0.3) -> dict:
    """Build a minimal well-formed research-node JSON document."""
    return {
        "asset": asset,
        "catalysts": [
            {"type": "bullish", "description": "x", "source": "reuters:x", "strength": 0.6}
        ],
        "risks": [],
        "sentiment": {
            "aggregate_polarity": polarity,
            "briefing_level": 1.0,
            "news_consensus": "bullish" if polarity >= 0 else "bearish",
        },
        "technical_context": {
            "gap_from_previous_close_pct": 0.3,
            "gap_direction": "UP",
            "gap_significance": "minor",
            "pre_market_momentum": "holding",
        },
        "watchlist_signals": [],
        "key_theme": "theme",
    }


def _predict_doc(asset: str, direction: str, confidence: float) -> dict:
    """Build a minimal well-formed predict-node JSON document."""
    return {
        "asset": asset,
        "direction": direction,
        "confidence": confidence,
        "predicted_move_pct": 0.5 if direction == "UP" else -0.5,
        "rationale": "evidence",
        "sources": ["reuters:x"],
    }


def _validated(asset: str, direction: str, original: float, adjusted: float) -> dict:
    return {
        "asset": asset,
        "original_direction": direction,
        "original_confidence": original,
        "adjusted_confidence": adjusted,
        "adjustment_reasons": [],
        "issues": [],
    }


def _clean_checker(validated: list) -> dict:
    return {
        "validated_predictions": validated,
        "contradictions": [],
        "flags": [],
        "overall_assessment": "Clean.",
        "can_proceed": True,
    }


def _dispatch(research_spy, research_qqq, predict_spy, predict_qqq, checker, pick_trade):
    """Build a subprocess.run side_effect dispatching on the --agent name."""

    def side_effect(cmd, **kwargs):
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

    return side_effect


class TestVetoBlocksTradeAtDecisionLevel:
    """The single most important regression test for the graph-engineering
    fix: a checker veto must make the trade UNSELECTABLE by
    DecisionAggregator (recommendation=None), not merely dent a
    confidence field the aggregator never reads. The old
    ``_phase_check`` only mutated ``recommendation.confidence``, which
    the aggregator's threshold check (``StrategyResult.confidence``)
    never looked at — so a veto changed nothing about which trade
    actually executed. This test exercises the real path: LLMTradeStrategy
    -> GraphOrchestrator -> checker veto -> DecisionAggregator.aggregate.
    """

    @pytest.mark.asyncio
    async def test_veto_blocks_trade_at_decision_level(self) -> None:
        from src.engine.decision import DecisionAggregator
        from src.llm.trade_signal import LLMTradeStrategy

        config = Settings()
        config.graph.enabled = True
        config.graph.checker_contradiction_action = "veto"
        config.graph.checker_confidence_penalty = 0.15
        config.llm.opencode_path = "opencode"

        research_spy = json.dumps(_research_doc("SPY", polarity=0.5))
        research_qqq = json.dumps(_research_doc("QQQ", polarity=0.4))
        predict_spy = json.dumps(_predict_doc("SPY", "UP", 0.70))
        predict_qqq = json.dumps(_predict_doc("QQQ", "UP", 0.60))
        checker = json.dumps(
            {
                "validated_predictions": [
                    _validated("SPY", "UP", 0.70, 0.70),
                    _validated("QQQ", "UP", 0.60, 0.60),
                ],
                "contradictions": [
                    {
                        "strategies": ["llm_trade", "event_driven"],
                        "description": (
                            "LLM recommends SPY CALL but event-driven flags bearish catalysts."
                        ),
                        "severity": "high",
                    }
                ],
                "flags": [],
                "overall_assessment": "Hard contradiction detected.",
                "can_proceed": False,
            }
        )
        pick_trade = json.dumps(
            {
                "best_trade": {
                    "asset": "SPY",
                    "direction": "CALL",
                    "confidence": 0.70,
                    "rationale": "Bullish signal with clean evidence.",
                    "sources": ["reuters:x"],
                },
                "rationale": "Selected SPY CALL based on highest confidence.",
            }
        )

        strategy = LLMTradeStrategy(
            config,
            deterministic_results=[
                {
                    "label": "event_driven",
                    "recommendation": {"asset": "SPY", "direction": "PUT", "confidence": 0.58},
                    "confidence": 0.58,
                    "debug_trace": {"catalyst_count": 8, "catalyst_polarity": -0.5},
                }
            ],
        )

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                side_effect=_dispatch(
                    research_spy, research_qqq, predict_spy, predict_qqq, checker, pick_trade
                ),
            ),
        ):
            llm_result = await strategy.evaluate(_briefing(), _market_spy_qqq())

        # The graph ran, the checker vetoed, and the veto must be visible
        # as a nulled-out recommendation AND zeroed confidence — the two
        # fields DecisionAggregator actually reads.
        assert llm_result.debug_trace.get("checker_can_proceed") is False
        assert llm_result.debug_trace.get("checker_veto") is True
        assert llm_result.recommendation is None
        assert llm_result.confidence == 0.0

        # Now prove it at the decision level: with only the vetoed LLM
        # result in play, DecisionAggregator must produce no trade.
        aggregator = DecisionAggregator(config)
        decision = aggregator.aggregate([llm_result])
        assert decision.recommendation is None


class TestPenalizeActionKeepsBothConfidencesInSync:
    """``checker_contradiction_action: penalize`` is the other
    config-selectable branch. The bug the veto path had — updating
    ``recommendation.confidence`` while leaving ``StrategyResult.confidence``
    stale — is equally possible here, and only the latter is what
    ``DecisionAggregator`` ranks and thresholds on. So both must move
    together, and the penalty must actually reach the decision.
    """

    @staticmethod
    def _run_with_penalty(confidence: float, penalty: float):
        from src.llm.trade_signal import LLMTradeStrategy

        config = Settings()
        config.graph.enabled = True
        config.graph.checker_contradiction_action = "penalize"
        config.graph.checker_confidence_penalty = penalty
        config.llm.opencode_path = "opencode"

        checker = json.dumps(
            {
                "validated_predictions": [_validated("SPY", "UP", confidence, confidence)],
                "contradictions": [
                    {
                        "strategies": ["llm_trade", "momentum"],
                        "description": "Soft disagreement.",
                        "severity": "medium",
                    }
                ],
                "flags": [],
                "overall_assessment": "Contradiction, but not fatal.",
                "can_proceed": False,
            }
        )
        pick_trade = json.dumps(
            {
                "best_trade": {
                    "asset": "SPY",
                    "direction": "CALL",
                    "confidence": confidence,
                    "rationale": "Bullish.",
                    "sources": ["reuters:x"],
                },
                "rationale": "Selected SPY CALL.",
            }
        )
        strategy = LLMTradeStrategy(config, deterministic_results=[])
        dispatch = _dispatch(
            json.dumps(_research_doc("SPY", polarity=0.5)),
            json.dumps(_research_doc("QQQ", polarity=0.4)),
            json.dumps(_predict_doc("SPY", "UP", confidence)),
            json.dumps(_predict_doc("QQQ", "UP", 0.50)),
            checker,
            pick_trade,
        )
        return config, strategy, dispatch

    @pytest.mark.asyncio
    async def test_penalty_applied_to_both_confidence_fields(self) -> None:
        _config, strategy, dispatch = self._run_with_penalty(0.70, 0.15)

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=dispatch),
        ):
            result = await strategy.evaluate(_briefing(), _market_spy_qqq())

        assert result.debug_trace.get("checker_penalized") is True
        # Penalized, not vetoed: the trade survives but at lower conviction.
        assert result.recommendation is not None
        assert result.recommendation.confidence == pytest.approx(0.55, abs=1e-6)
        # The field the aggregator actually reads must match, not lag.
        assert result.confidence == pytest.approx(result.recommendation.confidence, abs=1e-6)

    @pytest.mark.asyncio
    async def test_penalty_can_push_a_trade_below_the_decision_threshold(self) -> None:
        """Proof the penalty reaches the decision, not just the trace."""
        from src.engine.decision import DecisionAggregator

        # 0.50 confidence with a 0.15 penalty lands at 0.35, under the
        # 0.40 momentum min_confidence gate the aggregator applies.
        config, strategy, dispatch = self._run_with_penalty(0.50, 0.15)

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=dispatch),
        ):
            result = await strategy.evaluate(_briefing(), _market_spy_qqq())

        assert result.confidence == pytest.approx(0.35, abs=1e-6)
        decision = DecisionAggregator(config).aggregate([result])
        assert decision.recommendation is None

    @pytest.mark.asyncio
    async def test_clean_checker_leaves_confidence_untouched(self) -> None:
        """Control: no contradiction means no penalty on either field."""
        from src.llm.trade_signal import LLMTradeStrategy

        config = Settings()
        config.graph.enabled = True
        config.graph.checker_contradiction_action = "penalize"
        config.graph.checker_confidence_penalty = 0.15
        config.llm.opencode_path = "opencode"

        checker = json.dumps(
            {
                "validated_predictions": [_validated("SPY", "UP", 0.70, 0.70)],
                "contradictions": [],
                "flags": [],
                "overall_assessment": "All clear.",
                "can_proceed": True,
            }
        )
        pick_trade = json.dumps(
            {
                "best_trade": {
                    "asset": "SPY",
                    "direction": "CALL",
                    "confidence": 0.70,
                    "rationale": "Bullish.",
                    "sources": ["reuters:x"],
                },
                "rationale": "Selected SPY CALL.",
            }
        )
        strategy = LLMTradeStrategy(config, deterministic_results=[])

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                side_effect=_dispatch(
                    json.dumps(_research_doc("SPY", polarity=0.5)),
                    json.dumps(_research_doc("QQQ", polarity=0.4)),
                    json.dumps(_predict_doc("SPY", "UP", 0.70)),
                    json.dumps(_predict_doc("QQQ", "UP", 0.50)),
                    checker,
                    pick_trade,
                ),
            ),
        ):
            result = await strategy.evaluate(_briefing(), _market_spy_qqq())

        assert result.debug_trace.get("checker_penalized") is not True
        assert result.recommendation is not None
        assert result.recommendation.confidence == pytest.approx(0.70, abs=1e-6)
        assert result.confidence == pytest.approx(0.70, abs=1e-6)


class TestCheckerInvokedOncePerRun:
    @pytest.mark.asyncio
    async def test_checker_invoked_exactly_once_and_sees_deterministic_results(self) -> None:
        """Regression test for the double-checker-call bug: the old code
        ran the in-graph checker against an empty deterministic list AND
        then ran Pipeline._phase_check a second time against the real
        results — two LLM calls, and the first one useless. After the
        fix, deterministic strategies run first and feed the single
        in-graph checker call."""
        from src.pipeline import Pipeline

        config = Settings()
        config.graph.enabled = True
        config.llm.opencode_path = "opencode"
        config.llm.enabled = True
        config.llm.trade_signal_enabled = True
        config.general.execute = False

        pipeline = Pipeline(config, "test-correlation", _mock_file_logger())

        research_spy = json.dumps(_research_doc("SPY"))
        research_qqq = json.dumps(_research_doc("QQQ"))
        predict_spy = json.dumps(_predict_doc("SPY", "UP", 0.55))
        predict_qqq = json.dumps(_predict_doc("QQQ", "DOWN", 0.50))
        checker = json.dumps(
            _clean_checker(
                [_validated("SPY", "UP", 0.55, 0.55), _validated("QQQ", "DOWN", 0.50, 0.50)]
            )
        )
        pick_trade = json.dumps(
            {"best_trade": None, "rationale": "No strong edge.", "pass_reason": "mixed signals"}
        )

        calls: list[str] = []
        base_side_effect = _dispatch(
            research_spy, research_qqq, predict_spy, predict_qqq, checker, pick_trade
        )

        def counting_side_effect(cmd, **kwargs):
            calls.append(" ".join(cmd))
            return base_side_effect(cmd, **kwargs)

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=counting_side_effect),
        ):
            llm_client = OpencodeLLMClient(config.llm)
            results = await pipeline._phase_analyze(
                _briefing(), _market_spy_qqq(), llm_client=llm_client
            )

        checker_calls = [c for c in calls if "checker" in c]
        assert len(checker_calls) == 1, f"expected exactly 1 checker call, got {len(checker_calls)}"

        # The deterministic strategies actually ran and their labels
        # reached the checker prompt (the prompt text is the trailing
        # positional arg of the invoked command).
        checker_prompt = checker_calls[0]
        assert "momentum" in checker_prompt
        assert "mean_reversion" in checker_prompt
        assert "event_driven" in checker_prompt

        labels = {r.label for r in results}
        assert {"momentum", "mean_reversion", "event_driven", "llm_trade"} <= labels


class TestMalformedCheckerResponseDegradesGracefully:
    @pytest.mark.asyncio
    async def test_non_numeric_adjustment_and_non_list_predictions_do_not_raise(self) -> None:
        from src.llm.graph import GraphOrchestrator

        config = Settings()
        config.llm.opencode_path = "opencode"
        client = OpencodeLLMClient(config.llm)

        research_spy = json.dumps(_research_doc("SPY"))
        research_qqq = json.dumps(_research_doc("QQQ"))
        predict_spy = json.dumps(_predict_doc("SPY", "UP", 0.60))
        predict_qqq = json.dumps(_predict_doc("QQQ", "DOWN", 0.45))
        # Malformed checker output: validated_predictions is a plain
        # string (not a list), and can_proceed / other fields are also
        # the wrong type. Must degrade to "no adjustment applied", not
        # raise and kill the whole run.
        malformed_checker = json.dumps(
            {
                "validated_predictions": "oops, not a list",
                "contradictions": "also not a list",
                "flags": None,
                "overall_assessment": 12345,
                "can_proceed": "yes",
            }
        )
        pick_trade = json.dumps({"best_trade": None, "rationale": "no edge"})

        orchestrator = GraphOrchestrator(config, client)

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                side_effect=_dispatch(
                    research_spy,
                    research_qqq,
                    predict_spy,
                    predict_qqq,
                    malformed_checker,
                    pick_trade,
                ),
            ),
        ):
            result = await orchestrator.run(
                briefing=_briefing(),
                market=_market_spy_qqq(),
                deterministic_results=[],
            )

        # No exception, and the run completed rather than being treated
        # as a hard failure.
        assert result["trace"].get("graph_failed") is not True
        # Confidences unchanged since the adjustment couldn't be parsed.
        assert result["predictions"]["SPY"]["confidence"] == 0.60
        assert result["predictions"]["QQQ"]["confidence"] == 0.45
        # Malformed can_proceed defaults to a safe value (True — "no
        # adjustment applied"), never raises.
        assert result["checker_can_proceed"] is True
        assert result["checker_contradictions"] == []
        assert result["checker_flags"] == []

    @pytest.mark.asyncio
    async def test_llm_strategy_evaluate_does_not_raise_on_malformed_checker(self) -> None:
        """End-to-end: LLMTradeStrategy.evaluate() must never raise even
        when the checker node returns malformed JSON types."""
        from src.llm.trade_signal import LLMTradeStrategy

        config = Settings()
        config.graph.enabled = True
        config.llm.opencode_path = "opencode"

        research_spy = json.dumps(_research_doc("SPY"))
        research_qqq = json.dumps(_research_doc("QQQ"))
        predict_spy = json.dumps(_predict_doc("SPY", "UP", 0.60))
        predict_qqq = json.dumps(_predict_doc("QQQ", "DOWN", 0.45))
        malformed_checker = json.dumps(
            {
                "validated_predictions": {"not": "a list"},
                "adjusted_confidence": "high",
                "can_proceed": None,
            }
        )
        pick_trade = json.dumps({"best_trade": None, "rationale": "no edge"})

        strategy = LLMTradeStrategy(config)

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                side_effect=_dispatch(
                    research_spy,
                    research_qqq,
                    predict_spy,
                    predict_qqq,
                    malformed_checker,
                    pick_trade,
                ),
            ),
        ):
            result = await strategy.evaluate(_briefing(), _market_spy_qqq())

        assert result is not None
        assert result.predictions is not None


class TestWallClockDeadline:
    @pytest.mark.asyncio
    async def test_exhausted_deadline_triggers_graph_failure_before_any_call(self) -> None:
        """A wall-clock budget of 0 must stop the orchestrator before it
        starts the first node — no opencode subprocess call at all — and
        report a graph failure so the caller falls back."""
        from src.llm.graph import GraphOrchestrator

        config = Settings()
        config.llm.opencode_path = "opencode"
        config.graph.total_deadline_sec = 0
        client = OpencodeLLMClient(config.llm)
        orchestrator = GraphOrchestrator(config, client)

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run") as mock_run,
        ):
            result = await orchestrator.run(
                briefing=_briefing(),
                market=_market_spy_qqq(),
                deterministic_results=[],
            )

        assert result["trace"].get("graph_failed") is True
        assert "deadline_exhausted" in result["trace"].get("skip_reason", "")
        mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_deadline_exhaustion_falls_back_to_monolithic(self) -> None:
        """LLMTradeStrategy must fall back to the monolithic prompt when
        the graph's wall-clock budget is already exhausted."""
        from src.llm.trade_signal import LLMTradeStrategy

        config = Settings()
        config.graph.enabled = True
        config.graph.fallback_to_monolithic = True
        config.graph.total_deadline_sec = 0
        config.llm.opencode_path = "opencode"

        strategy = LLMTradeStrategy(config)

        def side_effect(cmd, **kwargs):
            # Only the monolithic call (no --agent flag) should ever land here.
            assert "--agent" not in cmd
            resp = json.dumps(
                {
                    "predictions": {
                        "SPY": {
                            "direction": "UP",
                            "confidence": 0.55,
                            "predicted_move_pct": 0.4,
                            "rationale": "fallback",
                            "sources": ["reuters:x"],
                        },
                        "QQQ": {
                            "direction": "DOWN",
                            "confidence": 0.4,
                            "predicted_move_pct": -0.3,
                            "rationale": "fallback",
                            "sources": ["reuters:x"],
                        },
                    }
                }
            )
            return _completed(_ndjson(resp))

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=side_effect),
        ):
            result = await strategy.evaluate(_briefing(), _market_spy_qqq())

        assert result.debug_trace.get("graph_failed") is True
        assert result.predictions is not None
        assert "SPY" in result.predictions


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
