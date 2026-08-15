from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from src.config import GapFadeConfig, Settings
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

    def test_two_json_objects_extracts_first_only(self) -> None:
        """The old greedy `\\{[\\s\\S]*\\}` regex spanned from the first `{`
        to the LAST `}`, which would swallow a second JSON object (or
        trailing braced prose) into one unparseable blob. The balanced
        scan must isolate just the first valid object."""
        text = '{"a": 1} some separator text {"b": 2}'
        result = _extract_json(text)
        assert result == {"a": 1}

    def test_trailing_braced_prose_does_not_break_extraction(self) -> None:
        text = 'Here is the JSON: {"pick": "SPY"} — and a note {like this}.'
        result = _extract_json(text)
        assert result == {"pick": "SPY"}

    def test_nested_braces_stay_balanced(self) -> None:
        text = '{"outer": {"inner": {"deep": 1}}, "sibling": 2}'
        result = _extract_json(text)
        assert result == {"outer": {"inner": {"deep": 1}}, "sibling": 2}

    def test_brace_inside_string_value_does_not_confuse_scan(self) -> None:
        text = '{"note": "use a { in prose", "value": 1}'
        result = _extract_json(text)
        assert result == {"note": "use a { in prose", "value": 1}

    def test_shares_implementation_with_trade_signal_parse_pick(self) -> None:
        from src.llm.trade_signal import _parse_pick

        assert _extract_json is _parse_pick


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
        prompt = _build_research_prompt(briefing, market, "SPY", GapFadeConfig())
        assert "Research target: SPY" in prompt
        assert "Bullish market" in prompt
        assert "$745.00" in prompt
        assert "+0.68%" in prompt
        assert "Gap-fade threshold for SPY: 1.5%" in prompt

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
        prompt = _build_research_prompt(briefing, market, "QQQ", GapFadeConfig())
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
        prompt = _build_predict_prompt(research, "SPY", GapFadeConfig())
        assert '"asset": "SPY"' in prompt
        assert '"AI rally"' in prompt
        assert "directional prediction JSON" in prompt
        assert "Gap-fade threshold for SPY: 1.5%" in prompt


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
        config = Settings()
        prompt = _build_checker_prompt(predict_spy, predict_qqq, deterministic, market, config)
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
        config = Settings()
        prompt = _build_checker_prompt(None, None, deterministic, market, config)
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


class TestValidateNodeAsset:
    def test_matching_asset_passes_through(self) -> None:
        from src.llm.graph import _validate_node_asset

        raw = {"asset": "SPY", "direction": "UP"}
        assert _validate_node_asset(raw, "SPY") == raw

    def test_none_stays_none(self) -> None:
        from src.llm.graph import _validate_node_asset

        assert _validate_node_asset(None, "SPY") is None

    def test_missing_asset_field_passes_through(self) -> None:
        from src.llm.graph import _validate_node_asset

        raw = {"direction": "UP"}
        assert _validate_node_asset(raw, "SPY") == raw

    def test_mismatched_self_reported_asset_is_rejected(self) -> None:
        """Regression test: predict-spy echoing "asset": "QQQ" must not
        silently overwrite the real QQQ prediction slot. The dict is
        keyed by the NODE that produced it, and a disagreeing
        self-report is rejected rather than trusted."""
        from src.llm.graph import _validate_node_asset

        raw = {"asset": "QQQ", "direction": "UP"}
        assert _validate_node_asset(raw, "SPY") is None


class TestCoerceCheckerVerdict:
    def test_well_formed_values_pass_through(self) -> None:
        from src.llm.graph import _coerce_checker_verdict

        can_proceed, contradictions, flags, overall = _coerce_checker_verdict(
            {
                "can_proceed": False,
                "contradictions": [{"description": "x"}],
                "flags": ["low_evidence"],
                "overall_assessment": "Mixed.",
            }
        )
        assert can_proceed is False
        assert contradictions == [{"description": "x"}]
        assert flags == ["low_evidence"]
        assert overall == "Mixed."

    def test_malformed_types_degrade_to_safe_defaults(self) -> None:
        """A checker returning wrong-typed verdict fields must not raise
        — everything degrades to a safe default instead."""
        from src.llm.graph import _coerce_checker_verdict

        can_proceed, contradictions, flags, overall = _coerce_checker_verdict(
            {
                "can_proceed": "false",  # string, not bool
                "contradictions": {"not": "a list"},
                "flags": None,
                "overall_assessment": 12345,
            }
        )
        assert can_proceed is True
        assert contradictions == []
        assert flags == []
        assert overall == ""

    def test_missing_keys_default_can_proceed_true(self) -> None:
        from src.llm.graph import _coerce_checker_verdict

        can_proceed, contradictions, flags, overall = _coerce_checker_verdict({})
        assert can_proceed is True
        assert contradictions == []
        assert flags == []
        assert overall == ""


class TestApplyCheckerAdjustments:
    def test_numeric_adjustment_applied_and_clamped(self) -> None:
        from src.llm.graph import _apply_checker_adjustments

        predictions_by_node = {
            "SPY": {"asset": "SPY", "confidence": 0.6},
            "QQQ": {"asset": "QQQ", "confidence": 0.5},
        }
        checker_output = {
            "validated_predictions": [
                {"asset": "SPY", "adjusted_confidence": 0.3},
                {"asset": "QQQ", "adjusted_confidence": 1.5},  # out of range -> clamped
            ]
        }
        _apply_checker_adjustments(predictions_by_node, checker_output)
        assert predictions_by_node["SPY"]["confidence"] == 0.3
        assert predictions_by_node["QQQ"]["confidence"] == 1.0

    def test_non_numeric_adjustment_ignored(self) -> None:
        """A checker returning `"adjusted_confidence": "high"` must not
        raise — the original confidence is left untouched."""
        from src.llm.graph import _apply_checker_adjustments

        predictions_by_node = {"SPY": {"asset": "SPY", "confidence": 0.6}}
        checker_output = {
            "validated_predictions": [{"asset": "SPY", "adjusted_confidence": "high"}]
        }
        _apply_checker_adjustments(predictions_by_node, checker_output)
        assert predictions_by_node["SPY"]["confidence"] == 0.6

    def test_non_list_validated_predictions_ignored(self) -> None:
        """A checker returning `validated_predictions` as something
        other than a list must not raise."""
        from src.llm.graph import _apply_checker_adjustments

        predictions_by_node = {"SPY": {"asset": "SPY", "confidence": 0.6}}
        checker_output = {"validated_predictions": "not-a-list"}
        _apply_checker_adjustments(predictions_by_node, checker_output)
        assert predictions_by_node["SPY"]["confidence"] == 0.6

    def test_unknown_asset_entry_ignored(self) -> None:
        from src.llm.graph import _apply_checker_adjustments

        predictions_by_node = {"SPY": {"asset": "SPY", "confidence": 0.6}, "QQQ": None}
        checker_output = {
            "validated_predictions": [
                {"asset": "AAPL", "adjusted_confidence": 0.9},
                {"asset": "QQQ", "adjusted_confidence": 0.9},  # node failed -> stays None
            ]
        }
        _apply_checker_adjustments(predictions_by_node, checker_output)
        assert predictions_by_node["SPY"]["confidence"] == 0.6
        assert predictions_by_node["QQQ"] is None


class TestGraphOrchestratorRun:
    @pytest.mark.asyncio
    async def test_pick_trade_node_failure_is_a_graph_failure(self) -> None:
        """A dead pick-trade node (no parseable output at all) must be
        treated as a graph failure so the caller falls back — distinct
        from a working node that legitimately returns
        ``{"best_trade": null}`` to pass on a trade."""
        from src.llm.client import OpencodeLLMClient
        from src.llm.graph import GraphOrchestrator

        config = Settings()
        config.llm.opencode_path = "opencode"
        client = OpencodeLLMClient(config.llm)

        research_spy = json.dumps(
            {
                "asset": "SPY",
                "catalysts": [
                    {"type": "bullish", "description": "x", "source": "r", "strength": 0.5}
                ],
                "risks": [],
                "sentiment": {
                    "aggregate_polarity": 0.3,
                    "briefing_level": 1.0,
                    "news_consensus": "bullish",
                },
                "technical_context": {
                    "gap_from_previous_close_pct": 0.2,
                    "gap_direction": "UP",
                    "gap_significance": "minor",
                    "pre_market_momentum": "holding",
                },
                "watchlist_signals": [],
                "key_theme": "x",
            }
        )
        predict_spy = json.dumps(
            {
                "asset": "SPY",
                "direction": "UP",
                "confidence": 0.6,
                "predicted_move_pct": 0.4,
                "rationale": "x",
                "sources": ["r"],
            }
        )
        checker = json.dumps(
            {
                "validated_predictions": [],
                "contradictions": [],
                "flags": [],
                "overall_assessment": "ok",
                "can_proceed": True,
            }
        )

        orchestrator = GraphOrchestrator(config, client)

        def side_effect(cmd, **kwargs):
            import subprocess as sp

            cmd_str = " ".join(cmd)
            if "research-spy" in cmd_str:
                return sp.CompletedProcess(cmd, 0, _ndjson_wrap(research_spy), "")
            if "research-qqq" in cmd_str:
                return sp.CompletedProcess(cmd, 1, "", "fail")
            if "predict-spy" in cmd_str:
                return sp.CompletedProcess(cmd, 0, _ndjson_wrap(predict_spy), "")
            if "checker" in cmd_str:
                return sp.CompletedProcess(cmd, 0, _ndjson_wrap(checker), "")
            if "pick-trade" in cmd_str:
                # Dead node: every model fails to produce a response.
                return sp.CompletedProcess(cmd, 1, "", "boom")
            return sp.CompletedProcess(cmd, 1, "", "unknown agent")

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=side_effect),
        ):
            result = await orchestrator.run(
                briefing=_empty_briefing(),
                market=_market_with_spy_qqq(),
                deterministic_results=[],
            )

        assert result["trace"].get("graph_failed") is True
        assert result["trace"].get("skip_reason") == "pick_trade_failed"

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

    @pytest.mark.asyncio
    async def test_unexpected_exception_becomes_a_graph_failure(self) -> None:
        """The crash-safety net: nothing above ``Pipeline.run`` catches
        (see ``src/main.py``), so an unexpected exception anywhere in the
        graph must degrade to a fallback, not kill the morning run."""
        from src.llm.client import OpencodeLLMClient
        from src.llm.graph import GraphOrchestrator

        config = Settings()
        config.llm.opencode_path = "opencode"
        orchestrator = GraphOrchestrator(config, OpencodeLLMClient(config.llm))

        with patch.object(
            GraphOrchestrator,
            "_run_inner",
            side_effect=RuntimeError("something nobody anticipated"),
        ):
            result = await orchestrator.run(
                briefing=_empty_briefing(),
                market=_market_with_spy_qqq(),
                deterministic_results=[],
            )

        assert result["graph_failed"] is True
        assert result["trace"]["skip_reason"] == "unexpected_exception"
        assert "RuntimeError" in result["trace"]["graph_fail_reason"]
        # And the shape stays consumable by the caller.
        assert result["predictions"] == {}
        assert result["best_trade"] is None

    @pytest.mark.asyncio
    async def test_all_predictions_failing_is_a_graph_failure(self) -> None:
        """Research can succeed while both predict nodes die."""
        from src.llm.client import OpencodeLLMClient
        from src.llm.graph import GraphOrchestrator

        config = Settings()
        config.llm.opencode_path = "opencode"
        orchestrator = GraphOrchestrator(config, OpencodeLLMClient(config.llm))

        def side_effect(cmd, **kwargs):
            import subprocess as sp

            cmd_str = " ".join(cmd)
            if "research-" in cmd_str:
                return sp.CompletedProcess(cmd, 0, _ndjson_wrap(_research_json("SPY")), "")
            return sp.CompletedProcess(cmd, 1, "", "predict died")

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=side_effect),
        ):
            result = await orchestrator.run(
                briefing=_empty_briefing(),
                market=_market_with_spy_qqq(),
                deterministic_results=[],
            )

        assert result["trace"].get("graph_failed") is True
        assert result["trace"].get("skip_reason") == "all_predictions_failed"

    @pytest.mark.asyncio
    async def test_unparseable_checker_is_a_graph_failure(self) -> None:
        """A checker that responds with pure prose (no JSON at all) is a
        dead node — distinct from a checker returning malformed *fields*,
        which degrades to 'no adjustment' instead."""
        from src.llm.client import OpencodeLLMClient
        from src.llm.graph import GraphOrchestrator

        config = Settings()
        config.llm.opencode_path = "opencode"
        orchestrator = GraphOrchestrator(config, OpencodeLLMClient(config.llm))

        def side_effect(cmd, **kwargs):
            import subprocess as sp

            cmd_str = " ".join(cmd)
            if "research-" in cmd_str:
                return sp.CompletedProcess(cmd, 0, _ndjson_wrap(_research_json("SPY")), "")
            if "predict-" in cmd_str:
                return sp.CompletedProcess(cmd, 0, _ndjson_wrap(_predict_json("SPY")), "")
            if "checker" in cmd_str:
                return sp.CompletedProcess(cmd, 0, _ndjson_wrap("I cannot comply."), "")
            return sp.CompletedProcess(cmd, 1, "", "unexpected")

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=side_effect),
        ):
            result = await orchestrator.run(
                briefing=_empty_briefing(),
                market=_market_with_spy_qqq(),
                deterministic_results=[],
            )

        assert result["trace"].get("graph_failed") is True
        assert result["trace"].get("skip_reason") == "checker_failed"

    @pytest.mark.asyncio
    async def test_deadline_expiring_mid_graph_stops_before_the_next_phase(self) -> None:
        """The budget is re-checked before every phase, not just at entry.

        Research succeeds, then the clock runs out, so the checker and
        pick-trade nodes must never be invoked.
        """
        from src.llm.client import OpencodeLLMClient
        from src.llm.graph import GraphOrchestrator

        config = Settings()
        config.llm.opencode_path = "opencode"
        config.graph.total_deadline_sec = 60
        orchestrator = GraphOrchestrator(config, OpencodeLLMClient(config.llm))

        agents_called: list[str] = []
        clock = {"t": 1000.0}

        def fake_monotonic() -> float:
            return clock["t"]

        def side_effect(cmd, **kwargs):
            import subprocess as sp

            cmd_str = " ".join(cmd)
            for name in ("research-spy", "research-qqq", "predict-spy", "predict-qqq"):
                if name in cmd_str:
                    agents_called.append(name)
                    payload = (
                        _research_json(name[-3:].upper())
                        if name.startswith("research")
                        else _predict_json(name[-3:].upper())
                    )
                    # Burn the entire budget during the predict phase.
                    if name.startswith("predict"):
                        clock["t"] += 120.0
                    return sp.CompletedProcess(cmd, 0, _ndjson_wrap(payload), "")
            agents_called.append(cmd_str)
            return sp.CompletedProcess(cmd, 0, _ndjson_wrap("{}"), "")

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=side_effect),
            patch("src.llm.graph.time.monotonic", side_effect=fake_monotonic),
        ):
            result = await orchestrator.run(
                briefing=_empty_briefing(),
                market=_market_with_spy_qqq(),
                deterministic_results=[],
            )

        assert result["trace"].get("graph_failed") is True
        assert result["trace"].get("skip_reason") == "deadline_exhausted_before_checker"
        assert not any("checker" in a for a in agents_called)
        assert not any("pick-trade" in a for a in agents_called)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _research_json(asset: str) -> str:
    """Minimal well-formed research-node payload."""
    return json.dumps(
        {
            "asset": asset,
            "catalysts": [
                {"type": "bullish", "description": "x", "source": "reuters:x", "strength": 0.5}
            ],
            "risks": [],
            "sentiment": {
                "aggregate_polarity": 0.3,
                "briefing_level": 1.0,
                "news_consensus": "bullish",
            },
            "technical_context": {
                "gap_from_previous_close_pct": 0.2,
                "gap_direction": "UP",
                "gap_significance": "minor",
                "pre_market_momentum": "holding",
            },
            "watchlist_signals": [],
            "key_theme": "x",
        }
    )


def _predict_json(asset: str) -> str:
    """Minimal well-formed predict-node payload."""
    return json.dumps(
        {
            "asset": asset,
            "direction": "UP",
            "confidence": 0.6,
            "predicted_move_pct": 0.4,
            "rationale": "x",
            "sources": ["reuters:x"],
        }
    )


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
