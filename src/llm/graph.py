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

The whole graph runs against a wall-clock budget
(:attr:`GraphConfig.total_deadline_sec`): per-node timeouts are per
*attempt*, and with a multi-model fallback chain a single node's worst
case can run to several minutes, which would overrun the pipeline's
entry window. :meth:`GraphOrchestrator.run` checks the remaining budget
before starting each phase and shrinks the per-node timeout to whatever
budget remains, so a single node can never overrun the whole graph.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import structlog

from src.config import GapFadeConfig, GraphConfig, Settings
from src.llm.client import OpencodeLLMClient
from src.llm.trade_signal import _day_move_pct, _gap_pct, _premarket_prompt_block
from src.llm.trade_signal import _parse_pick as _extract_json
from src.models.briefing import BriefingData
from src.models.market import MarketSnapshot
from src.prediction_tracker import format_history_for_prompt, load_history
from src.trade_tracker import format_outcomes_for_prompt, load_trade_outcomes

logger = structlog.get_logger()

__all__ = ["GraphOrchestrator", "_extract_json"]


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

        Any unexpected exception raised while running the graph is caught
        here, logged, and converted into a graph-failure result so the
        caller falls back to the monolithic path instead of the whole
        pipeline run dying.

        Args:
            briefing: Parsed morning briefing.
            market: Current market snapshot.
            deterministic_results: Serialized outputs from Momentum,
                MeanReversion, and EventDriven strategies. Each dict has
                ``label``, ``recommendation``, ``confidence``,
                ``debug_trace``.

        Returns:
            A dict with keys matching the legacy monolithic output:
            ``predictions`` (dict[str, dict]), ``market_vibe`` (str),
            ``best_trade`` (dict or None), ``trace`` (dict with debug
            info), ``served_by`` (str), ``paid_used`` (bool), plus the
            checker verdict surfaced as ``checker_can_proceed``,
            ``checker_contradictions``, ``checker_flags``, and
            ``checker_overall_assessment``.
        """
        start = time.monotonic()
        # Absolute deadline for the whole graph. Handed to every
        # ``invoke_agent`` call so a single node's model-fallback chain
        # (7 models x per-attempt timeout) cannot overrun the budget
        # between phase checks.
        self._deadline_ts = start + self.graph_config.total_deadline_sec
        trace: dict[str, Any] = {"graph_enabled": True, "nodes": {}}
        try:
            return await self._run_inner(briefing, market, deterministic_results, trace, start)
        except Exception as e:
            logger.error(
                "graph_orchestrator_unexpected_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            trace["skip_reason"] = "unexpected_exception"
            return _graph_failure(trace, f"unexpected exception: {type(e).__name__}: {e}")

    async def _run_inner(
        self,
        briefing: BriefingData,
        market: MarketSnapshot,
        deterministic_results: list[dict[str, Any]],
        trace: dict[str, Any],
        start: float,
    ) -> dict[str, Any]:
        """Execute each graph phase in turn, enforcing the wall-clock budget."""
        deadline_sec = self.graph_config.total_deadline_sec

        def remaining_sec() -> float:
            return deadline_sec - (time.monotonic() - start)

        def check_budget(phase: str) -> dict[str, Any] | None:
            if remaining_sec() <= 0:
                trace["skip_reason"] = f"deadline_exhausted_before_{phase}"
                trace["elapsed_sec"] = round(time.monotonic() - start, 2)
                return _graph_failure(trace, f"wall-clock deadline exhausted before {phase}")
            return None

        def node_timeout(configured: int) -> int:
            return max(1, min(configured, int(remaining_sec())))

        # Phase 1: Research SPY + QQQ in parallel
        failure = check_budget("research")
        if failure is not None:
            return failure
        research_timeout = node_timeout(self.graph_config.research_timeout_sec)
        research_spy, research_qqq = await asyncio.gather(
            asyncio.to_thread(self._research_spy, briefing, market, research_timeout),
            asyncio.to_thread(self._research_qqq, briefing, market, research_timeout),
        )
        trace["nodes"]["research_spy"] = research_spy is not None
        trace["nodes"]["research_qqq"] = research_qqq is not None

        if research_spy is None and research_qqq is None:
            trace["skip_reason"] = "all_research_failed"
            return _graph_failure(trace, "all research nodes failed")

        # Phase 2: Predict SPY + QQQ in parallel, keyed by the NODE that
        # produced them (never by the agent's self-reported "asset"
        # field — see _validate_node_asset).
        failure = check_budget("prediction")
        if failure is not None:
            return failure
        predict_timeout = node_timeout(self.graph_config.prediction_timeout_sec)

        predict_jobs: dict[str, Any] = {}
        if research_spy is not None:
            predict_jobs["SPY"] = asyncio.to_thread(
                self._predict_spy, research_spy, predict_timeout
            )
        if research_qqq is not None:
            predict_jobs["QQQ"] = asyncio.to_thread(
                self._predict_qqq, research_qqq, predict_timeout
            )

        predictions_by_node: dict[str, dict[str, Any] | None] = {"SPY": None, "QQQ": None}
        if predict_jobs:
            gathered = await asyncio.gather(*predict_jobs.values())
            for node_asset, raw in zip(predict_jobs.keys(), gathered, strict=True):
                predictions_by_node[node_asset] = _validate_node_asset(raw, node_asset)

        trace["nodes"]["predict_spy"] = predictions_by_node["SPY"] is not None
        trace["nodes"]["predict_qqq"] = predictions_by_node["QQQ"] is not None

        if predictions_by_node["SPY"] is None and predictions_by_node["QQQ"] is None:
            trace["skip_reason"] = "all_predictions_failed"
            return _graph_failure(trace, "all prediction nodes failed")

        # Phase 3: Checker — validates all outputs
        failure = check_budget("checker")
        if failure is not None:
            return failure
        checker_timeout = node_timeout(self.graph_config.checker_timeout_sec)
        checker_output = await asyncio.to_thread(
            self._run_checker,
            predictions_by_node["SPY"],
            predictions_by_node["QQQ"],
            deterministic_results,
            market,
            checker_timeout,
        )
        trace["nodes"]["checker"] = checker_output is not None

        if checker_output is None:
            trace["skip_reason"] = "checker_failed"
            return _graph_failure(trace, "checker node failed")

        can_proceed, contradictions, flags, overall_assessment = _coerce_checker_verdict(
            checker_output
        )
        _apply_checker_adjustments(predictions_by_node, checker_output)
        trace["checker_can_proceed"] = can_proceed
        trace["checker_contradictions"] = contradictions
        trace["checker_flags"] = flags

        # Phase 4: Pick trade
        failure = check_budget("pick_trade")
        if failure is not None:
            return failure
        history_str = format_history_for_prompt(
            load_history(self.config.logging.json_dir),
        )
        trade_outcomes_str = format_outcomes_for_prompt(
            load_trade_outcomes(self.config.logging.json_dir),
        )
        pick_timeout = node_timeout(self.graph_config.pick_trade_timeout_sec)
        pick_output = await asyncio.to_thread(
            self._run_pick_trade,
            checker_output,
            predictions_by_node,
            history_str,
            trade_outcomes_str,
            pick_timeout,
        )
        trace["nodes"]["pick_trade"] = pick_output is not None

        if pick_output is None:
            # A node that never returned parseable output is a dead node,
            # not a deliberate pass — fall back rather than silently
            # treating it as "no trade today". A genuine `best_trade: null`
            # from a WORKING node is handled below as a legitimate pass.
            trace["skip_reason"] = "pick_trade_failed"
            return _graph_failure(trace, "pick_trade node failed")

        # Phase 5: Assemble result
        predictions: dict[str, Any] = {
            asset: pred for asset, pred in predictions_by_node.items() if pred is not None
        }

        best_trade = pick_output.get("best_trade")
        trace["pick_rationale"] = pick_output.get("rationale", "")
        trace["pass_reason"] = pick_output.get("pass_reason", "")

        served_by = self.client.last_served_by or ""
        paid_used = self.client.paid_used

        return {
            "predictions": predictions,
            "market_vibe": overall_assessment,
            "best_trade": best_trade,
            "trace": trace,
            "served_by": served_by,
            "paid_used": paid_used,
            "checker_can_proceed": can_proceed,
            "checker_contradictions": contradictions,
            "checker_flags": flags,
            "checker_overall_assessment": overall_assessment,
        }

    # ------------------------------------------------------------------
    # Research nodes
    # ------------------------------------------------------------------

    def _research_spy(
        self, briefing: BriefingData, market: MarketSnapshot, timeout_sec: int
    ) -> dict[str, Any] | None:
        prompt = _build_research_prompt(briefing, market, "SPY", self.config.gap_fade)
        response = self.client.invoke_agent(
            self.AGENT_RESEARCH_SPY, prompt, timeout_sec=timeout_sec, deadline_ts=self._deadline_ts
        )
        return _extract_json(response) if response else None

    def _research_qqq(
        self, briefing: BriefingData, market: MarketSnapshot, timeout_sec: int
    ) -> dict[str, Any] | None:
        prompt = _build_research_prompt(briefing, market, "QQQ", self.config.gap_fade)
        response = self.client.invoke_agent(
            self.AGENT_RESEARCH_QQQ, prompt, timeout_sec=timeout_sec, deadline_ts=self._deadline_ts
        )
        return _extract_json(response) if response else None

    # ------------------------------------------------------------------
    # Prediction nodes
    # ------------------------------------------------------------------

    def _predict_spy(self, research: dict[str, Any], timeout_sec: int) -> dict[str, Any] | None:
        prompt = _build_predict_prompt(research, "SPY", self.config.gap_fade)
        response = self.client.invoke_agent(
            self.AGENT_PREDICT_SPY, prompt, timeout_sec=timeout_sec, deadline_ts=self._deadline_ts
        )
        return _extract_json(response) if response else None

    def _predict_qqq(self, research: dict[str, Any], timeout_sec: int) -> dict[str, Any] | None:
        prompt = _build_predict_prompt(research, "QQQ", self.config.gap_fade)
        response = self.client.invoke_agent(
            self.AGENT_PREDICT_QQQ, prompt, timeout_sec=timeout_sec, deadline_ts=self._deadline_ts
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
        timeout_sec: int,
    ) -> dict[str, Any] | None:
        prompt = _build_checker_prompt(
            predict_spy, predict_qqq, deterministic_results, market, self.config
        )
        response = self.client.invoke_agent(
            self.AGENT_CHECKER, prompt, timeout_sec=timeout_sec, deadline_ts=self._deadline_ts
        )
        return _extract_json(response) if response else None

    # ------------------------------------------------------------------
    # Pick-trade node
    # ------------------------------------------------------------------

    def _run_pick_trade(
        self,
        checker_output: dict[str, Any],
        predictions_by_node: dict[str, dict[str, Any] | None],
        history_str: str,
        trade_outcomes_str: str,
        timeout_sec: int,
    ) -> dict[str, Any] | None:
        prompt = _build_pick_trade_prompt(
            checker_output, predictions_by_node, history_str, trade_outcomes_str, self.config
        )
        response = self.client.invoke_agent(
            self.AGENT_PICK_TRADE, prompt, timeout_sec=timeout_sec, deadline_ts=self._deadline_ts
        )
        return _extract_json(response) if response else None


# ------------------------------------------------------------------
# Prompt builders
# ------------------------------------------------------------------


def _build_research_prompt(
    briefing: BriefingData, market: MarketSnapshot, asset: str, gap_fade: GapFadeConfig
) -> str:
    """Build the research node prompt for a single asset."""
    sections: list[str] = []
    sections.append(f"Research target: {asset}")
    sections.append("Analyze the following data and produce the structured JSON output.")
    sections.append(
        f"Gap-fade threshold for {asset}: {gap_fade.threshold_for(asset)}% "
        f"(flag when |gap| exceeds this AND sentiment magnitude < "
        f"{gap_fade.sentiment_magnitude_max})"
    )

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
    pm = market.premarket.get(asset)
    if q:
        # PRIOR SESSION, not today — see Quote.prior_session_close.
        sections.append(
            f"{asset} PRIOR SESSION: closed ${q.prior_session_close:.2f} "
            f"(open ${q.open_price:.2f}, high ${q.high_price:.2f}, low ${q.low_price:.2f})."
        )
        sections.append(_premarket_prompt_block(asset, pm))
        gap_pct = _gap_pct(q, pm)
        day_move_pct = _day_move_pct(q, pm)
        if gap_pct is not None and day_move_pct is not None:
            sections.append(
                f"{asset} pre-market drift: {day_move_pct:+.2f}% from the session's "
                f"first print to the cutoff price (building vs fading momentum)."
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


def _build_predict_prompt(research: dict[str, Any], asset: str, gap_fade: GapFadeConfig) -> str:
    """Build the prediction node prompt from research output."""
    research_json = json.dumps(research, indent=2)
    return (
        f"Below is a structured research document for {asset}. "
        f"Use it to produce your directional prediction JSON.\n\n"
        f"Gap-fade threshold for {asset}: {gap_fade.threshold_for(asset)}% "
        f"(flag when |gap| exceeds this AND sentiment magnitude < "
        f"{gap_fade.sentiment_magnitude_max}).\n\n"
        f"Research input:\n{research_json}"
    )


def _build_checker_prompt(
    predict_spy: dict[str, Any] | None,
    predict_qqq: dict[str, Any] | None,
    deterministic_results: list[dict[str, Any]],
    market: MarketSnapshot,
    config: Settings,
) -> str:
    """Build the checker node prompt."""
    graph_config = config.graph
    gap_fade = config.gap_fade
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
            pm = market.premarket.get(asset)
            gap_pct = _gap_pct(q, pm)
            gap_threshold = gap_fade.threshold_for(asset)
            sentiment_mag = abs(market.avg_sentiment_polarity())
            gap_str = (
                f"{gap_pct:+.2f}%"
                if gap_pct is not None
                else "UNAVAILABLE (no live pre-market quote)"
            )
            sections.append(
                f"{asset}: prior session close ${q.prior_session_close:.2f}, "
                f"today's pre-market gap {gap_str} "
                f"(gap-fade threshold: {gap_threshold}%, "
                f"sentiment cutoff: {gap_fade.sentiment_magnitude_max}, "
                f"catalyst strength: {sentiment_mag:+.3f})"
            )

    return "\n".join(sections)


def _build_pick_trade_prompt(
    checker_output: dict[str, Any],
    predictions_by_node: dict[str, dict[str, Any] | None],
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
    for pred in predictions_by_node.values():
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


def _validate_node_asset(raw: dict[str, Any] | None, expected_asset: str) -> dict[str, Any] | None:
    """Reject a prediction whose self-reported asset disagrees with its node.

    Predictions are keyed into ``predictions_by_node`` by the NODE that
    produced them (``predict-spy`` always maps to ``"SPY"``), never by
    the agent's own ``"asset"`` field in its JSON response — an agent
    that echoes or hallucinates the wrong ticker must not silently
    overwrite the other asset's slot.

    Args:
        raw: The parsed JSON response from the predict node, or ``None``
            if the node failed outright.
        expected_asset: The asset this node was invoked for.

    Returns:
        ``raw`` unchanged when it agrees (or is silent) about the asset,
        else ``None``.
    """
    if raw is None:
        return None
    reported = raw.get("asset")
    if reported is not None and reported != expected_asset:
        logger.warning(
            "predict_node_asset_mismatch",
            node=expected_asset,
            reported_asset=reported,
        )
        return None
    return raw


def _coerce_checker_verdict(
    checker_output: dict[str, Any],
) -> tuple[bool, list[Any], list[Any], str]:
    """Defensively coerce the checker's top-level verdict fields.

    A checker returning malformed types (``can_proceed`` as a string,
    ``contradictions`` as a dict, etc.) must degrade to safe defaults
    instead of raising and killing the whole morning run.

    Args:
        checker_output: Parsed JSON from the checker node.

    Returns:
        A ``(can_proceed, contradictions, flags, overall_assessment)``
        tuple with well-typed values.
    """
    can_proceed = checker_output.get("can_proceed")
    if not isinstance(can_proceed, bool):
        can_proceed = True

    contradictions = checker_output.get("contradictions")
    if not isinstance(contradictions, list):
        contradictions = []

    flags = checker_output.get("flags")
    if not isinstance(flags, list):
        flags = []

    overall_assessment = checker_output.get("overall_assessment")
    if not isinstance(overall_assessment, str):
        overall_assessment = ""

    return can_proceed, contradictions, flags, overall_assessment


def _apply_checker_adjustments(
    predictions_by_node: dict[str, dict[str, Any] | None],
    checker_output: dict[str, Any],
) -> None:
    """Apply the checker's ``adjusted_confidence`` onto matching predictions.

    Mutates ``predictions_by_node`` in place. Every value pulled from the
    parsed checker JSON is defensively coerced — a checker returning
    ``"adjusted_confidence": "high"`` or a non-list
    ``validated_predictions`` must degrade to "no adjustment applied"
    rather than raising.

    Args:
        predictions_by_node: Per-asset raw prediction dicts (mutated).
        checker_output: Parsed JSON from the checker node.
    """
    validated = checker_output.get("validated_predictions")
    if not isinstance(validated, list):
        logger.warning(
            "checker_validated_predictions_malformed",
            type_name=type(validated).__name__,
        )
        return

    for entry in validated:
        if not isinstance(entry, dict):
            continue
        asset = entry.get("asset")
        if not isinstance(asset, str) or predictions_by_node.get(asset) is None:
            continue
        try:
            adjusted = float(entry.get("adjusted_confidence"))
        except (TypeError, ValueError):
            continue
        predictions_by_node[asset]["confidence"] = max(0.0, min(1.0, adjusted))


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
