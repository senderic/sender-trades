from __future__ import annotations

import json
import subprocess
from datetime import date
from typing import Any
from unittest.mock import patch

import pytest

from src.config import Settings
from src.llm.trade_signal import LLMTradeStrategy, _build_prompt, _normalise_sources, _parse_pick
from src.models.briefing import BriefingData, BriefingQuality
from src.models.market import DataSource, MarketSnapshot, PremarketQuote, Quote
from src.models.recommendation import Direction

# Distinguishing phrases from each agent's inlined system-prompt body.
# ``invoke_agent`` no longer passes ``--agent``, so subprocess mocks route
# on these markers instead of the agent name.
_AGENT_MARKERS = {
    "research-spy": "market research analyst focused exclusively on SPY",
    "research-qqq": "market research analyst focused exclusively on QQQ",
    "predict-spy": "directional prediction specialist for SPY",
    "predict-qqq": "directional prediction specialist for QQQ",
    "checker": "validation checker",
    "pick-trade": "trade selector",
}


def _ndjson(text: str) -> str:
    return json.dumps({"type": "text", "part": {"text": text}}) + "\n"


def _completed(stdout: str, rc: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(["opencode"], rc, stdout, "")


def _stock_response(pick: dict[str, Any]) -> str:
    """Wrap a pick dict in the NDJSON stream the opencode CLI emits."""
    return _ndjson(json.dumps(pick))


def _default_predictions() -> dict[str, Any]:
    """Standard per-asset predictions for use across tests."""
    return {
        "SPY": {
            "direction": "DOWN",
            "confidence": 0.65,
            "predicted_move_pct": -1.1,
            "rationale": "Hawkish Fed comments weigh on broad market.",
            "sources": ["atlas-briefing:executive_summary"],
        },
        "QQQ": {
            "direction": "DOWN",
            "confidence": 0.72,
            "predicted_move_pct": -2.5,
            "rationale": "Tech selloff led by NVDA and META.",
            "sources": [
                "atlas-briefing:executive_summary",
                "reuters:tech-ai-selloff",
                "watchlist:NVDA",
            ],
        },
    }


def _full_response(
    predictions: dict[str, Any] | None = None,
    market_vibe: str = "",
    best_trade: dict[str, Any] | None = None,
) -> str:
    """Build a complete new-format prediction response."""
    data: dict[str, Any] = {}
    if predictions is None:
        predictions = _default_predictions()
    data["predictions"] = predictions
    if market_vibe:
        data["market_vibe"] = market_vibe
    if best_trade is not None:
        data["best_trade"] = best_trade
    return _ndjson(json.dumps(data))


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


class TestNormaliseSources:
    def test_list_passthrough(self) -> None:
        assert _normalise_sources(["reuters:kimi", "watchlist:NVDA"]) == [
            "reuters:kimi",
            "watchlist:NVDA",
        ]

    def test_single_string_wrapped(self) -> None:
        assert _normalise_sources("atlas-briefing:executive_summary") == [
            "atlas-briefing:executive_summary"
        ]

    def test_clamps_to_three(self) -> None:
        result = _normalise_sources(["a", "b", "c", "d", "e"])
        assert result == ["a", "b", "c"]

    def test_strips_whitespace_and_drops_empty(self) -> None:
        assert _normalise_sources(["  reuters:x  ", "", "   ", "watchlist:SPY"]) == [
            "reuters:x",
            "watchlist:SPY",
        ]

    def test_drops_non_string_entries(self) -> None:
        assert _normalise_sources(["ok", 42, None, {"a": "b"}, "watchlist:QQQ"]) == [
            "ok",
            "watchlist:QQQ",
        ]

    def test_drops_overlong_citations(self) -> None:
        huge = "x" * 200
        assert _normalise_sources([huge, "reuters:ok"]) == ["reuters:ok"]

    def test_missing_returns_atlas_briefing_fallback(self) -> None:
        assert _normalise_sources([]) == ["news-sentiment"]
        assert _normalise_sources(None) == ["news-sentiment"]
        assert _normalise_sources({}) == ["news-sentiment"]
        assert _normalise_sources([None, 1, {"x": "y"}]) == ["news-sentiment"]


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
    async def test_valid_pick_builds_predictions_and_best_trade(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        strategy = self._strategy()
        best_trade = {
            "asset": "QQQ",
            "direction": "PUT",
            "confidence": 0.78,
            "rationale": "Tech selloff; weak QQQ pre-market.",
            "sources": [
                "atlas-briefing:executive_summary",
                "reuters:tech-ai-selloff",
                "watchlist:NVDA",
            ],
        }
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=_completed(_full_response(best_trade=best_trade)),
            ),
        ):
            result = await strategy.evaluate(briefing_with_sentiment, market_with_quotes)

        # Predictions for both assets
        assert result.predictions is not None
        assert "SPY" in result.predictions
        assert "QQQ" in result.predictions
        assert result.predictions["QQQ"].direction == "DOWN"
        assert result.predictions["QQQ"].predicted_move_pct == -2.5

        # Best trade recommendation
        assert result.recommendation is not None
        assert result.recommendation.asset == "QQQ"
        assert result.recommendation.direction == Direction.PUT
        assert result.recommendation.confidence == 0.78
        assert result.recommendation.strategy_label == "llm_trade"
        assert result.debug_trace["served_by"] == "opencode/muse-spark-1.3-contributor-free"
        assert result.debug_trace["paid_used"] is False
        assert "tech selloff" in result.recommendation.rationale["llm_rationale"].lower()
        assert result.recommendation.rationale["llm_sources"] == [
            "atlas-briefing:executive_summary",
            "reuters:tech-ai-selloff",
            "watchlist:NVDA",
        ]
        # Forecast source labels come from all prediction sources
        assert "llm:atlas-briefing:executive_summary" in result.forecast_source_labels
        assert "llm:reuters:tech-ai-selloff" in result.forecast_source_labels

    @pytest.mark.asyncio
    async def test_predictions_populated_without_best_trade(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        strategy = self._strategy()
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=_completed(_full_response()),
            ),
        ):
            result = await strategy.evaluate(briefing_with_sentiment, market_with_quotes)

        assert result.predictions is not None
        assert result.predictions["SPY"].direction == "DOWN"
        assert result.predictions["SPY"].confidence == 0.65
        assert result.predictions["SPY"].predicted_move_pct == -1.1
        assert result.predictions["QQQ"].direction == "DOWN"
        assert result.predictions["QQQ"].confidence == 0.72
        assert result.predictions["QQQ"].predicted_move_pct == -2.5
        assert result.recommendation is None  # no best_trade

    @pytest.mark.asyncio
    async def test_missing_sources_defaults_to_news_sentiment_in_best_trade(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        strategy = self._strategy()
        best_trade = {
            "asset": "SPY",
            "direction": "CALL",
            "confidence": 0.6,
            "rationale": "broad market strength",
        }
        resp = _full_response(best_trade=best_trade)
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=_completed(resp),
            ),
        ):
            result = await strategy.evaluate(briefing_with_sentiment, market_with_quotes)
        assert result.recommendation is not None
        assert result.recommendation.rationale["llm_sources"] == ["news-sentiment"]

    @pytest.mark.asyncio
    async def test_sources_clamped_to_three(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        strategy = self._strategy()
        best_trade = {
            "asset": "SPY",
            "direction": "CALL",
            "confidence": 0.7,
            "rationale": "x",
            "sources": ["a", "b", "c", "d", "e"],
        }
        resp = _full_response(best_trade=best_trade)
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=_completed(resp),
            ),
        ):
            result = await strategy.evaluate(briefing_with_sentiment, market_with_quotes)
        assert result.recommendation is not None
        assert len(result.recommendation.rationale["llm_sources"]) == 3

    @pytest.mark.asyncio
    async def test_best_trade_confidence_clamped_to_range(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        strategy = self._strategy()
        best_trade = {"asset": "SPY", "direction": "CALL", "confidence": 1.5, "rationale": "x"}
        resp = _full_response(best_trade=best_trade)
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=_completed(resp),
            ),
        ):
            result = await strategy.evaluate(briefing_with_sentiment, market_with_quotes)
        assert result.recommendation is not None
        assert result.recommendation.confidence == 1.0

    @pytest.mark.asyncio
    async def test_low_confidence_best_trade_abstains_trade(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        cfg = Settings()
        cfg.llm.trade_signal_min_confidence = 0.60
        strategy = self._strategy(cfg)
        best_trade = {"asset": "SPY", "direction": "CALL", "confidence": 0.42, "rationale": "x"}
        resp = _full_response(best_trade=best_trade)
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=_completed(resp),
            ),
        ):
            result = await strategy.evaluate(briefing_with_sentiment, market_with_quotes)
        # Predictions still present, but no recommendation (best_trade rejected).
        assert result.predictions is not None
        assert result.recommendation is None
        assert result.debug_trace["best_trade_skip"] == "below_min_confidence"

    @pytest.mark.asyncio
    async def test_asset_out_of_universe_best_trade_skips(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        strategy = self._strategy()
        best_trade = {
            "asset": "AAPL",
            "direction": "CALL",
            "confidence": 0.8,
            "rationale": "x",
        }
        resp = _full_response(best_trade=best_trade)
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=_completed(resp),
            ),
        ):
            result = await strategy.evaluate(briefing_with_sentiment, market_with_quotes)
        # Predictions still present, best_trade skipped.
        assert result.predictions is not None
        assert result.recommendation is None
        assert result.debug_trace["best_trade_skip"] == "asset_out_of_universe"

    @pytest.mark.asyncio
    async def test_invalid_direction_in_best_trade_skips(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        strategy = self._strategy()
        best_trade = {"asset": "SPY", "direction": "HOLD", "confidence": 0.7, "rationale": "x"}
        resp = _full_response(best_trade=best_trade)
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch(
                "src.llm.client.subprocess.run",
                return_value=_completed(resp),
            ),
        ):
            result = await strategy.evaluate(briefing_with_sentiment, market_with_quotes)
        assert result.predictions is not None
        assert result.recommendation is None
        assert result.debug_trace["best_trade_skip"] == "invalid_direction"

    @pytest.mark.asyncio
    async def test_unparseable_response_abstains(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        strategy = self._strategy()
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
        assert result.debug_trace["skip_reason"] == "llm_no_response"

    @pytest.mark.asyncio
    async def test_paid_model_serves_pick_paid_used_flag(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        cfg = Settings()
        cfg.llm.primary_model = "opencode-go/deepseek-v4-pro"
        cfg.llm.fallback_models = ["openrouter/deepseek/deepseek-v4-pro"]
        cfg.llm.opencode_path = "opencode"
        strategy = LLMTradeStrategy(cfg)

        best_trade = {
            "asset": "SPY",
            "direction": "CALL",
            "confidence": 0.66,
            "rationale": "Rotation into defense supports broad market.",
        }
        resp = _full_response(best_trade=best_trade)

        def run_side_effect(cmd, **kwargs):
            if "opencode-go/deepseek-v4-pro" in cmd:
                return subprocess.CompletedProcess(cmd, 1, "", "fail")
            return _completed(resp)

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
        ):
            result = await strategy.evaluate(briefing_with_sentiment, market_with_quotes)
        assert result.recommendation is not None
        assert result.debug_trace["paid_used"] is True
        assert result.debug_trace["served_by"] == "openrouter/deepseek/deepseek-v4-pro"


def _premarket(symbol: str, gap_pct: float, reliable: bool = True) -> PremarketQuote:
    return PremarketQuote(
        symbol=symbol,
        available=True,
        price=700.0,
        vwap=699.5,
        first_price=695.0,
        cumulative_volume=2000.0,
        gap_pct=gap_pct,
        median_volume=2000.0,
        volume_ratio=1.0 if reliable else 0.1,
        reliable=reliable,
        lookback_days_used=10,
        source="live",
    )


class TestGapAwarenessInPrompt:
    def test_prompt_includes_gap_warning_when_large_gap(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        # Gap is now the LIVE pre-market price vs the PRIOR SESSION close
        # (see Quote.prior_session_close / _gap_pct) -- driven by
        # market.premarket, not the stale open_price/previous_close fields.
        market_with_quotes.premarket["QQQ"] = _premarket("QQQ", gap_pct=2.0)
        prompt = _build_prompt(briefing_with_sentiment, market_with_quotes, ["SPY", "QQQ"])
        assert "Pre-market gap alert" in prompt
        assert "QQQ has gapped +2.0%" in prompt

    def test_prompt_omits_gap_warning_when_gap_small(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        market_with_quotes.premarket["SPY"] = _premarket("SPY", gap_pct=0.16)
        market_with_quotes.premarket["QQQ"] = _premarket("QQQ", gap_pct=0.12)
        prompt = _build_prompt(briefing_with_sentiment, market_with_quotes, ["SPY", "QQQ"])
        assert "Pre-market gap alert" not in prompt

    def test_prompt_states_gap_fade_threshold_explicitly(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        """Prompt builders must state the applicable threshold in the
        prompt text itself — the .opencode/agent/*.md files expect the
        threshold to arrive in the prompt rather than being hardcoded."""
        prompt = _build_prompt(briefing_with_sentiment, market_with_quotes, ["SPY", "QQQ"])
        assert "Gap-fade threshold for SPY: 1.5%" in prompt
        assert "Gap-fade threshold for QQQ: 2.0%" in prompt

    def test_prompt_labels_quote_as_prior_session(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        prompt = _build_prompt(briefing_with_sentiment, market_with_quotes, ["SPY", "QQQ"])
        assert "PRIOR SESSION" in prompt
        assert "NOT today" in prompt

    def test_prompt_discloses_unavailable_premarket_explicitly(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        """When no live pre-market quote exists, the prompt must say so
        explicitly rather than silently treating the stale quote as
        today's price (the exact 2026-09-11 bug)."""
        prompt = _build_prompt(briefing_with_sentiment, market_with_quotes, ["SPY", "QQQ"])
        assert "UNAVAILABLE as of the cutoff" in prompt
        assert "do NOT treat" in prompt

    def test_prompt_flags_thin_premarket_volume(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        market_with_quotes.premarket["QQQ"] = _premarket("QQQ", gap_pct=0.3, reliable=False)
        prompt = _build_prompt(briefing_with_sentiment, market_with_quotes, ["SPY", "QQQ"])
        assert "THIN" in prompt
        assert "weight this QQQ pre-market move LIGHTLY" in prompt


class TestGraphEnabledLLMTradeStrategy:
    """Tests that the graph orchestrator path integrates correctly with LLMTradeStrategy."""

    def _strategy(
        self, graph_enabled: bool = True, config: Settings | None = None
    ) -> LLMTradeStrategy:
        cfg = config or Settings()
        cfg.graph.enabled = graph_enabled
        cfg.graph.fallback_to_monolithic = True
        cfg.llm.opencode_path = "opencode"
        return LLMTradeStrategy(cfg)

    def _predictions_json(self) -> str:
        """Build the graph research output JSON for both assets."""
        return json.dumps(
            {
                "predictions": {
                    "SPY": {
                        "asset": "SPY",
                        "direction": "UP",
                        "confidence": 0.62,
                        "predicted_move_pct": 0.4,
                        "rationale": "Broad strength.",
                        "sources": ["reuters:bullish"],
                    },
                    "QQQ": {
                        "asset": "QQQ",
                        "direction": "DOWN",
                        "confidence": 0.48,
                        "predicted_move_pct": -1.0,
                        "rationale": "Tech selloff.",
                        "sources": ["watchlist:NVDA"],
                    },
                },
                "market_vibe": "Mixed",
            }
        )

    @pytest.mark.asyncio
    async def test_graph_disabled_uses_monolithic(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        strategy = self._strategy(graph_enabled=False)
        best_trade = {
            "asset": "SPY",
            "direction": "CALL",
            "confidence": 0.66,
            "rationale": "x",
        }
        resp = _full_response(best_trade=best_trade)
        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", return_value=_completed(resp)),
        ):
            result = await strategy.evaluate(briefing_with_sentiment, market_with_quotes)
        assert result.recommendation is not None
        assert result.recommendation.asset == "SPY"
        assert "llm_raw" in result.debug_trace

    @pytest.mark.asyncio
    async def test_graph_enabled_runs_research_and_predict(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        strategy = self._strategy(graph_enabled=True)

        def research_response(text: str) -> str:
            return _ndjson(text)

        research_spy = json.dumps(
            {
                "asset": "SPY",
                "catalysts": [
                    {
                        "type": "bullish",
                        "description": "Broad strength",
                        "source": "reuters:up",
                        "strength": 0.6,
                    }
                ],
                "risks": [],
                "sentiment": {
                    "aggregate_polarity": 0.4,
                    "briefing_level": 1.0,
                    "news_consensus": "bullish",
                },
                "technical_context": {
                    "gap_from_previous_close_pct": 0.3,
                    "gap_direction": "UP",
                    "gap_significance": "minor",
                    "pre_market_momentum": "holding",
                },
                "watchlist_signals": [],
                "key_theme": "Mildly bullish",
            }
        )
        research_qqq = json.dumps(
            {
                "asset": "QQQ",
                "catalysts": [
                    {
                        "type": "bearish",
                        "description": "Tech weakness",
                        "source": "watchlist:QQQ",
                        "strength": 0.5,
                    }
                ],
                "risks": [],
                "sentiment": {
                    "aggregate_polarity": -0.3,
                    "briefing_level": 1.0,
                    "news_consensus": "bearish",
                },
                "technical_context": {
                    "gap_from_previous_close_pct": -0.2,
                    "gap_direction": "DOWN",
                    "gap_significance": "minor",
                    "pre_market_momentum": "fading",
                },
                "watchlist_signals": [],
                "key_theme": "Tech pressure",
            }
        )

        predict_spy = json.dumps(
            {
                "asset": "SPY",
                "direction": "UP",
                "confidence": 0.62,
                "predicted_move_pct": 0.4,
                "rationale": "Broad strength",
                "sources": ["reuters:up"],
            }
        )
        predict_qqq = json.dumps(
            {
                "asset": "QQQ",
                "direction": "DOWN",
                "confidence": 0.48,
                "predicted_move_pct": -1.0,
                "rationale": "Tech weakness",
                "sources": ["watchlist:QQQ"],
            }
        )

        def run_side_effect(cmd, **kwargs):
            cmd_str = " ".join(cmd)
            if _AGENT_MARKERS["research-spy"] in cmd_str:
                return _completed(research_response(research_spy))
            if _AGENT_MARKERS["research-qqq"] in cmd_str:
                return _completed(research_response(research_qqq))
            if _AGENT_MARKERS["predict-spy"] in cmd_str:
                return _completed(research_response(predict_spy))
            if _AGENT_MARKERS["predict-qqq"] in cmd_str:
                return _completed(research_response(predict_qqq))
            return _completed(research_response("{}"))

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
        ):
            result = await strategy.evaluate(briefing_with_sentiment, market_with_quotes)

        assert result.predictions is not None
        assert "SPY" in result.predictions
        assert result.predictions["SPY"].direction == "UP"
        assert result.predictions["QQQ"].direction == "DOWN"
        assert result.debug_trace.get("graph_run") is True

    @pytest.mark.asyncio
    async def test_graph_falls_back_to_monolithic_on_failure(
        self, briefing_with_sentiment, market_with_quotes
    ) -> None:
        strategy = self._strategy(graph_enabled=True)

        call_count = [0]

        def run_side_effect(cmd, **kwargs):
            call_count[0] += 1
            cmd_str = " ".join(cmd)
            # Graph attempt: fail all research nodes.
            if (
                _AGENT_MARKERS["research-spy"] in cmd_str
                or _AGENT_MARKERS["research-qqq"] in cmd_str
            ):
                return _completed("", rc=1)
            # Fallback monolithic: none of the agent bodies are inlined.
            if not any(marker in cmd_str for marker in _AGENT_MARKERS.values()):
                resp = _full_response(
                    best_trade={
                        "asset": "SPY",
                        "direction": "CALL",
                        "confidence": 0.55,
                        "rationale": "fallback",
                    }
                )
                return _completed(resp)
            return _completed("{}")

        with (
            patch("src.llm.client.shutil.which", return_value="/usr/bin/opencode"),
            patch("src.llm.client.subprocess.run", side_effect=run_side_effect),
        ):
            result = await strategy.evaluate(briefing_with_sentiment, market_with_quotes)

        assert result.recommendation is not None
        assert result.recommendation.asset == "SPY"
        assert call_count[0] >= 2  # At least one graph call + one monolithic call


if __name__ == "__main__":
    pytest.main([__file__])
