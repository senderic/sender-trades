"""Graph orchestrator that runs the diamond-shaped LLM prediction pipeline.

Replaces the monolithic single-call :class:`LLMTradeStrategy` with six
narrow-scope subagent nodes arranged in a directed acyclic graph:

    [Research SPY] ──→ [Predict SPY] ──┐
    [Research QQQ] ──→ [Predict QQQ] ──┤
                                        ├──→ [Checker] ──→ [Pick Trade]
    [Momentum / MeanRev / EventDriven] ─┘

Each node is a named opencode subagent (``.opencode/agent/<name>.md``)
invoked via :meth:`OpencodeLLMClient.invoke_agent`. Nodes with no mutual
dependencies run in parallel via :func:`asyncio.to_thread`.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import structlog

from src.config import GraphConfig, Settings
from src.llm.client import OpencodeLLMClient
from src.models.briefing import BriefingData
from src.models.market import MarketSnapshot
from src.prediction_tracker import format_history_for_prompt, load_history
from src.trade_tracker import format_outcomes_for_prompt, load_trade_outcomes

logger = structlog.get_logger()

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a text response."""
    if not text or not text.strip():
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", stripped)
        if fenced:
            stripped = fenced.group(1)
    match = _JSON_BLOCK_RE.search(stripped)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


class GraphOrchestrator:
    """Runs the diamond-shaped LLM graph for per-asset predictions + best trade.

    When the graph is disabled or fails, the caller should fall back to
    the legacy monolithic :class:`LLMTradeStrategy` call.
    """

    AGENT_RESEARCH_SPY = "research-spy"
    AGENT_RESEARCH_QQQ = "research-qqq"
    AGENT_PREDICT_SPY = "predict-spy"
    AGENT_PREDICT_QQQ = "predict-qqq"
    AGENT_CHECKER = "checker"
    AGENT_PICK_TRADE = "pick-trade"

    def __init__(self, config: Settings, client: OpencodeLLMClient):
        self.config = config
        self.graph_config: GraphConfig = config.graph
        self.client = client

    async def run(
        self,
        briefing: BriefingData,
        market: MarketSnapshot,
        deterministic_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute the full graph pipeline.

        Args:
            briefing: Parsed morning briefing.
            market: Current market snapshot.
            deterministic_results: Serialized outputs from Momentum,
                MeanReversion, and EventDriven strategies. Each dict has
                ``label``, ``recommendation``, ``confidence``,
                ``debug_trace``.

        Returns:
            A dict with keys matching the legacy monolithic output:
            ``predictions`` (dict[str, AssetPrediction]),
            ``market_vibe`` (str), ``best_trade`` (dict or None),
            ``trace`` (dict with debug info), ``served_by`` (str),
            ``paid_used`` (bool).
        """
        trace: dict[str, Any] = {
            "graph_enabled": True,
            "nodes": {},
        }

        # Phase 1: Research SPY + QQQ in parallel
        research_spy, research_qqq = await asyncio.gather(
            asyncio.to_thread(self._research_spy, briefing, market),
            asyncio.to_thread(self._research_qqq, briefing, market),
        )
        trace["nodes"]["research_spy"] = research_spy is not None
        trace["nodes"]["research_qqq"] = research_qqq is not None

        if research_spy is None and research_qqq is None:
            trace["skip_reason"] = "all_research_failed"
            return _graph_failure(trace, "all research nodes failed")

        # Phase 2: Predict SPY + QQQ in parallel
        predict_spy_task = (
            asyncio.to_thread(self._predict_spy, research_spy) if research_spy else None
        )
        predict_qqq_task = (
            asyncio.to_thread(self._predict_qqq, research_qqq) if research_qqq else None
        )

        tasks: list[asyncio.Future] = []
        task_labels: list[str] = []
        if predict_spy_task:
            tasks.append(predict_spy_task)
            task_labels.append("predict_spy")
        if predict_qqq_task:
            tasks.append(predict_qqq_task)
            task_labels.append("predict_qqq")

        predict_results: list[dict[str, Any] | None] = [None, None]
        if tasks:
            gathered = await asyncio.gather(*tasks)
            for i, label in enumerate(task_labels):
                if label == "predict_spy":
                    predict_results[0] = gathered[i]
                elif label == "predict_qqq":
                    predict_results[1] = gathered[i]

        trace["nodes"]["predict_spy"] = predict_results[0] is not None
        trace["nodes"]["predict_qqq"] = predict_results[1] is not None

        # Phase 3: Checker — validates all outputs
        checker_output = await asyncio.to_thread(
            self._run_checker,
            predict_results[0],
            predict_results[1],
            deterministic_results,
            market,
        )
        trace["nodes"]["checker"] = checker_output is not None

        if checker_output is None:
            trace["skip_reason"] = "checker_failed"
            return _graph_failure(trace, "checker node failed")

        # Phase 4: Pick trade
        history_str = format_history_for_prompt(
            load_history(self.config.logging.json_dir),
        )
        trade_outcomes_str = format_outcomes_for_prompt(
            load_trade_outcomes(self.config.logging.json_dir),
        )
        pick_output = await asyncio.to_thread(
            self._run_pick_trade,
            checker_output,
            predict_results,
            history_str,
            trade_outcomes_str,
        )
        trace["nodes"]["pick_trade"] = pick_output is not None

        # Phase 5: Assemble result
        predictions: dict[str, Any] = {}
        for pred_data in predict_results:
            if pred_data is None:
                continue
            asset = pred_data.get("asset", "")
            if asset in ("SPY", "QQQ"):
                predictions[asset] = pred_data

        best_trade = None
        if pick_output and isinstance(pick_output, dict):
            best_trade = pick_output.get("best_trade")
            trace["pick_rationale"] = pick_output.get("rationale", "")
            trace["pass_reason"] = pick_output.get("pass_reason", "")

        served_by = self.client.last_served_by or ""
        paid_used = self.client.paid_used

        return {
            "predictions": predictions,
            "market_vibe": checker_output.get("overall_assessment", "") if checker_output else "",
            "best_trade": best_trade,
            "trace": trace,
            "served_by": served_by,
            "paid_used": paid_used,
        }

    # ------------------------------------------------------------------
    # Research nodes
    # ------------------------------------------------------------------

    def _research_spy(
        self, briefing: BriefingData, market: MarketSnapshot
    ) -> dict[str, Any] | None:
        prompt = _build_research_prompt(briefing, market, "SPY")
        response = self.client.invoke_agent(
            self.AGENT_RESEARCH_SPY, prompt, timeout_sec=self.graph_config.research_timeout_sec
        )
        return _extract_json(response) if response else None

    def _research_qqq(
        self, briefing: BriefingData, market: MarketSnapshot
    ) -> dict[str, Any] | None:
        prompt = _build_research_prompt(briefing, market, "QQQ")
        response = self.client.invoke_agent(
            self.AGENT_RESEARCH_QQQ, prompt, timeout_sec=self.graph_config.research_timeout_sec
        )
        return _extract_json(response) if response else None

    # ------------------------------------------------------------------
    # Prediction nodes
    # ------------------------------------------------------------------

    def _predict_spy(self, research: dict[str, Any]) -> dict[str, Any] | None:
        prompt = _build_predict_prompt(research, "SPY")
        response = self.client.invoke_agent(
            self.AGENT_PREDICT_SPY, prompt, timeout_sec=self.graph_config.prediction_timeout_sec
        )
        return _extract_json(response) if response else None

    def _predict_qqq(self, research: dict[str, Any]) -> dict[str, Any] | None:
        prompt = _build_predict_prompt(research, "QQQ")
        response = self.client.invoke_agent(
            self.AGENT_PREDICT_QQQ, prompt, timeout_sec=self.graph_config.prediction_timeout_sec
        )
        return _extract_json(response) if response else None

    # ------------------------------------------------------------------
    # Checker node
    # ------------------------------------------------------------------

    def _run_checker(
        self,
        predict_spy: dict[str, Any] | None,
        predict_qqq: dict[str, Any] | None,
        deterministic_results: list[dict[str, Any]],
        market: MarketSnapshot,
    ) -> dict[str, Any] | None:
        prompt = _build_checker_prompt(
            predict_spy, predict_qqq, deterministic_results, market, self.graph_config
        )
        response = self.client.invoke_agent(
            self.AGENT_CHECKER, prompt, timeout_sec=self.graph_config.checker_timeout_sec
        )
        return _extract_json(response) if response else None

    # ------------------------------------------------------------------
    # Pick-trade node
    # ------------------------------------------------------------------

    def _run_pick_trade(
        self,
        checker_output: dict[str, Any],
        predict_results: list[dict[str, Any] | None],
        history_str: str,
        trade_outcomes_str: str,
    ) -> dict[str, Any] | None:
        prompt = _build_pick_trade_prompt(
            checker_output, predict_results, history_str, trade_outcomes_str, self.config
        )
        response = self.client.invoke_agent(
            self.AGENT_PICK_TRADE, prompt, timeout_sec=self.graph_config.pick_trade_timeout_sec
        )
        return _extract_json(response) if response else None


# ------------------------------------------------------------------
# Prompt builders
# ------------------------------------------------------------------


def _build_research_prompt(briefing: BriefingData, market: MarketSnapshot, asset: str) -> str:
    """Build the research node prompt for a single asset."""
    sections: list[str] = []
    sections.append(f"Research target: {asset}")
    sections.append("Analyze the following data and produce the structured JSON output.")

    if briefing.executive_summary:
        sections.append(f"Executive summary:\n{briefing.executive_summary}")
    if briefing.key_connections:
        sections.append(f"Key connections:\n{briefing.key_connections}")

    if briefing.tickers:
        ticker_lines = [
            f"- {t.symbol}: ${t.price:.2f} ({t.change_pct:+.2f}%)"
            + (f" — {t.likely_driver}" if t.likely_driver else "")
            for t in briefing.tickers[:25]
        ]
        sections.append("Watchlist tickers:\n" + "\n".join(ticker_lines))

    q = market.quotes.get(asset)
    if q:
        gap_pct = (
            (q.current_price - q.previous_close) / q.previous_close * 100
            if q.previous_close > 0
            else 0.0
        )
        sections.append(
            f"{asset} quote: ${q.current_price:.2f} "
            f"(gap {gap_pct:+.2f}% from prev close ${q.previous_close:.2f})"
        )

    polarity = market.avg_sentiment_polarity()
    sections.append(f"Market-average news sentiment polarity: {polarity:+.3f}")

    if market.news:
        top_news = market.news[:10]
        news_lines = [
            f"- [{n.source}] {n.title}"
            + (f" (polarity: {n.polarity:+.2f})" if n.polarity != 0 else "")
            + (f" — {n.snippet}" if n.snippet else "")
            for n in top_news
        ]
        sections.append("Top market news:\n" + "\n".join(news_lines))

    return "\n\n".join(sections)


def _build_predict_prompt(research: dict[str, Any], asset: str) -> str:
    """Build the prediction node prompt from research output."""
    research_json = json.dumps(research, indent=2)
    return (
        f"Below is a structured research document for {asset}. "
        f"Use it to produce your directional prediction JSON.\n\n"
        f"Research input:\n{research_json}"
    )


def _build_checker_prompt(
    predict_spy: dict[str, Any] | None,
    predict_qqq: dict[str, Any] | None,
    deterministic_results: list[dict[str, Any]],
    market: MarketSnapshot,
    graph_config: GraphConfig,
) -> str:
    """Build the checker node prompt."""
    sections: list[str] = []

    sections.append("Validate the following strategy outputs and flag any issues.")
    sections.append(f"Contradiction action: {graph_config.checker_contradiction_action}")
    sections.append(f"Contradiction confidence penalty: {graph_config.checker_confidence_penalty}")

    sections.append("\n== LLM PREDICTIONS ==")
    if predict_spy:
        sections.append(f"\nSPY prediction:\n{json.dumps(predict_spy, indent=2)}")
    else:
        sections.append("\nSPY prediction: FAILED (no output)")

    if predict_qqq:
        sections.append(f"\nQQQ prediction:\n{json.dumps(predict_qqq, indent=2)}")
    else:
        sections.append("\nQQQ prediction: FAILED (no output)")

    sections.append("\n== DETERMINISTIC STRATEGY RESULTS ==")
    for dr in deterministic_results:
        label = dr.get("label", "unknown")
        rec = dr.get("recommendation")
        conf = dr.get("confidence", 0.0)
        trace_d = dr.get("debug_trace", {})
        sections.append(f"\nStrategy: {label} (confidence: {conf:.4f})")
        if rec:
            sections.append(
                f"  Recommendation: {rec.get('asset', '')} {rec.get('direction', '')} "
                f"conf {rec.get('confidence', 0):.4f}"
            )
            rationale = rec.get("rationale", {})
            if isinstance(rationale, dict):
                for k, v in rationale.items():
                    if k not in ("llm_sources", "merged_from", "llm_rationale"):
                        sections.append(f"  {k}: {v}")
        if trace_d:
            sections.append(f"  Debug: {json.dumps(trace_d)}")

    sections.append("\n== MARKET CONTEXT ==")
    for asset in ("SPY", "QQQ"):
        q = market.quotes.get(asset)
        if q:
            gap_pct = (
                (q.current_price - q.previous_close) / q.previous_close * 100
                if q.previous_close > 0
                else 0.0
            )
            gap_threshold = 1.5 if asset == "SPY" else 2.0
            sentiment_mag = abs(market.avg_sentiment_polarity())
            sections.append(
                f"{asset}: ${q.current_price:.2f} gap {gap_pct:+.2f}% "
                f"(gap-fade threshold: {gap_threshold}%, "
                f"catalyst strength: {sentiment_mag:+.3f})"
            )

    return "\n".join(sections)


def _build_pick_trade_prompt(
    checker_output: dict[str, Any],
    predict_results: list[dict[str, Any] | None],
    history_str: str,
    trade_outcomes_str: str,
    config: Settings,
) -> str:
    """Build the pick-trade node prompt."""
    sections: list[str] = []

    sections.append("Select the best trade for today based on the validated predictions below.")
    sections.append(f"Minimum confidence threshold: {config.llm.trade_signal_min_confidence}")

    sections.append("\n== CHECKER OUTPUT ==")
    sections.append(json.dumps(checker_output, indent=2))

    sections.append("\n== RAW PREDICTIONS ==")
    for pred in predict_results:
        if pred:
            sections.append(json.dumps(pred, indent=2))

    if history_str:
        sections.append(f"\n== PREDICTION HISTORY ==\n{history_str}")

    if trade_outcomes_str:
        sections.append(f"\n== TRADE OUTCOMES ==\n{trade_outcomes_str}")

    return "\n".join(sections)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _graph_failure(trace: dict[str, Any], reason: str) -> dict[str, Any]:
    """Build a failure result that signals the caller to fall back."""
    trace["graph_failed"] = True
    trace["graph_fail_reason"] = reason
    logger.warning("graph_orchestrator_failed", reason=reason)
    return {
        "predictions": {},
        "market_vibe": "",
        "best_trade": None,
        "trace": trace,
        "served_by": "",
        "paid_used": False,
        "graph_failed": True,
    }
