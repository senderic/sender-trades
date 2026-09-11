"""LLM-driven directional prediction strategy.

Asks the opencode LLM (with the same Zen-first / paid-Go fallback chain
as the briefing re-synthesiser) to emit a structured JSON prediction for
today's session covering all target assets:

    {"predictions": {
       "SPY":  {"direction": "UP"|"DOWN", "confidence": 0.0-1.0,
                "predicted_move_pct": -1.5, "rationale": "...", "sources": [...]},
       "QQQ":  { ... }
     },
     "market_vibe": "overall sentiment string",
     "best_trade": {  // optional
        "asset": "SPY"|"QQQ", "direction": "CALL"|"PUT",
        "confidence": 0.0-1.0, "rationale": "...", "sources": [...]
     }}

The ``predictions`` dict feeds the :class:`DirectionalForecast` table
directly, giving the reader a clear per-asset directional view with
estimated move percentage and cited evidence. The optional
``best_trade`` can drive a :class:`TradeRecommendation` for execution
if the signal is strong enough.

Each prediction's ``sources`` array carries *root provenance* -- the LLM
must cite which upstream inputs drove its conclusion, using one of:

- ``"atlas-briefing:<section>"`` for content originally distilled by
  the upstream atlas-morning-briefing LLM (e.g.
  ``"atlas-briefing:executive_summary"``).
- ``"<publisher>:<slug-or-headline>"`` for market-news feed items
  (e.g. ``"reuters:kimi-k3-open-weight"``).
- ``"watchlist:<TICKER>"`` for watchlist ticker drivers.
- ``"market:<TICKER>"`` for live / snapshot quote metrics.
- ``"news-sentiment"`` for the aggregate-polarity signal.

These citations are surfaced verbatim (prefixed with ``llm:``) in the
forecast table, so the reader sees the evidence that drove each
prediction.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import structlog

from src.config import GapFadeConfig, Settings
from src.engine.base import TradingStrategy
from src.engine.options_strategy import compute_otm_strike, estimate_delta
from src.llm.client import OpencodeLLMClient
from src.models.briefing import BriefingData
from src.models.market import MarketSnapshot, PremarketQuote, Quote
from src.models.recommendation import (
    AssetPrediction,
    Direction,
    PositionIntent,
    StrategyResult,
    TradeRecommendation,
)
from src.prediction_tracker import format_history_for_prompt, load_history
from src.timezone import today_local
from src.trade_tracker import format_outcomes_for_prompt, load_trade_outcomes

logger = structlog.get_logger()


SYSTEM_PROMPT = (
    "You are the research and prediction engine for an intraday 0DTE "
    "options trading system. Your job is to analyze the morning briefing, "
    "market data, and news, then produce a directional prediction for "
    "each target asset.\n\n"
    "Emit a single JSON object with these keys:\n"
    '  - "predictions": a dict keyed by asset symbol. Each value is an '
    "object with:\n"
    '    - "direction": "UP" or "DOWN"\n'
    '    - "confidence": a float in [0.0, 1.0] reflecting how strongly '
    "the evidence supports this direction\n"
    '    - "predicted_move_pct": a float estimating the expected move '
    "percentage for today's session (positive for UP, negative for DOWN)\n"
    '    - "rationale": one short sentence citing the specific evidence '
    "that drove the prediction\n"
    '    - "sources": a list of 1-3 strings citing the ROOT provenance '
    "— where the evidence originally came from, not this LLM\n"
    '  - "market_vibe": a short string summarising the overall market '
    "backdrop / sentiment / key themes\n"
    '  - "best_trade": (optional) an object with exactly the same shape '
    'as the old single-trade format — asset, direction ("CALL" or '
    '"PUT"), confidence, rationale, sources — if the data clearly '
    "points to a specific executable trade today. Omit this key "
    "entirely when conditions are mixed, unclear, or when no trade "
    "has a strong edge. It is better to pass than to force a "
    "low-conviction trade.\n\n"
    "Gap-fade pattern (important): when a pre-market gap exceeds +1.5% "
    "for SPY or +2.0% for QQQ and the news sentiment magnitude is "
    "proportionally small (below 0.20 absolute), extreme caution is "
    "warranted. These large gaps often fade during the session — the "
    "overnight move exhausts before or shortly after the open and the "
    "market reverses. Aug 5 2026 was a textbook example: SPY gapped "
    "+1.8% and QQQ +3.4% on AI/space news with only +0.128 sentiment, "
    "both predicted UP, but both closed DOWN (-0.8% SPY, -1.2% QQQ). "
    "Do not blindly follow the gap direction when the catalyst strength "
    "is disproportionate to the gap size.\n\n"
    "Cite root provenance using these forms (prefer the MOST PRIMARY "
    "source available):\n"
    '  - "<publisher>:<short-slug>" for market news feed items (e.g. '
    '"reuters:kimi-k3-open-weight", "bloomberg:fed-cautious-stance", '
    '"seekingalpha:earnings-call", "dowjones:market-wrap"). '
    "This is your best option — use it whenever possible.\n"
    '  - "watchlist:<TICKER>" for a specific watchlist ticker that '
    'drove the call (e.g. "watchlist:NVDA" for NVDA\'s price action)\n'
    '  - "market:<TICKER>" for raw quote / price-action evidence\n'
    '  - "news-sentiment" for the aggregate news-polarity reading\n\n'
    'IMPORTANT: Do NOT cite "atlas-briefing" as a source. The atlas '
    "briefing is itself a distilled summary. Trace back to the original "
    "news publisher (reuters, bloomberg, seekingalpha, wsj, etc.) or "
    "market data point whenever possible. If you must reference the "
    "briefing's executive summary content, attribute it to the specific "
    "publisher or ticker that the briefing itself references.\n\n"
    "Do NOT cite this LLM or the llm_trade strategy. Cite the upstream "
    "source that produced the evidence.\n"
    "Output ONLY the JSON object. No prose, no code fence, no "
    "explanation outside the JSON."
)


class LLMTradeStrategy(TradingStrategy):
    """Trading strategy that delegates directional prediction to an LLM.

    Unlike Momentum / MeanReversion / EventDriven, which derive direction
    and confidence from deterministic price/sentiment math, this strategy
    asks the opencode LLM to emit a structured JSON prediction dict
    covering all target assets. The predictions feed the
    :class:`DirectionalForecast` table directly.

    The LLM may also optionally suggest a ``best_trade`` for execution;
    when present and strong enough, it is wrapped in a
    :class:`TradeRecommendation` and passed through the existing
    :class:`DecisionAggregator` / :class:`RiskEngine` pipeline.

    The LLM call goes through :class:`OpencodeLLMClient`, which tries
    :attr:`~src.config.LLMConfig.primary_model` first and then the
    :attr:`~src.config.LLMConfig.fallback_models` chain (see
    :class:`src.config.LLMConfig`). When the LLM is unavailable, returns
    junk, or emits an unparseable response, the strategy abstains
    (returns ``recommendation=None``) rather than guessing.
    """

    def __init__(
        self,
        config: Settings,
        client: OpencodeLLMClient | None = None,
        deterministic_results: list[dict[str, Any]] | None = None,
    ):
        """Initialize LLMTradeStrategy.

        Args:
            config: Application settings.
            client: Shared opencode client (constructed if omitted).
            deterministic_results: Serialized Momentum / MeanReversion /
                EventDriven outputs (``{"label", "recommendation",
                "confidence", "debug_trace"}`` dicts), already evaluated
                by the caller. Injected into the in-graph checker node so
                it validates the LLM against real deterministic signals
                instead of an empty list. Ignored on the monolithic path.
        """
        assert set(config.general.target_assets).issubset({"SPY", "QQQ"}), (
            f"LLMTradeStrategy only supports SPY/QQQ, got {config.general.target_assets}"
        )
        super().__init__(label="llm_trade", config=config)
        self._client = client or OpencodeLLMClient(config.llm)
        self._deterministic_results: list[dict[str, Any]] = deterministic_results or []

    async def evaluate(
        self,
        briefing: BriefingData,
        market: MarketSnapshot,
    ) -> StrategyResult:
        """Evaluate the LLM prediction strategy.

        When :attr:`GraphConfig.enabled` is True, delegates to the
        :class:`GraphOrchestrator` diamond (research + predict) and falls
        back to the monolithic call on failure. When graph is disabled
        (default), uses the legacy single-call LLM prompt.

        Args:
            briefing: Parsed morning briefing data.
            market: Current market snapshot with quotes and news.

        Returns:
            A :class:`StrategyResult` with:
            - ``predictions`` populated with per-asset predictions for the
              forecast table.
            - ``recommendation`` set when the LLM suggests a best_trade
              and it passes validation.
        """
        start = time.perf_counter()
        trace: dict[str, Any] = {}

        if not self._client.available:
            trace["skip_reason"] = "opencode_unavailable"
            return StrategyResult(
                label=self.label,
                recommendation=None,
                confidence=0.0,
                debug_trace=trace,
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
            )

        # --- Graph path ---
        graph_enabled = self.config.graph.enabled
        if graph_enabled:
            graph_result = await self._evaluate_via_graph(briefing, market, trace, start)
            if graph_result is not None:
                return graph_result
            if self.config.graph.fallback_to_monolithic:
                logger.info("graph_fell_back_to_monolithic")

        # --- Legacy monolithic path ---
        return await self._evaluate_monolithic(briefing, market, trace, start)

    async def _evaluate_via_graph(
        self,
        briefing: BriefingData,
        market: MarketSnapshot,
        trace: dict[str, Any],
        start: float,
    ) -> StrategyResult | None:
        """Run the graph orchestrator and convert output to a StrategyResult.

        Returns None when the graph fails, signalling the caller to fall
        back to the monolithic path.
        """
        from src.llm.graph import GraphOrchestrator

        orchestrator = GraphOrchestrator(self.config, self._client)
        try:
            result = await orchestrator.run(
                briefing=briefing,
                market=market,
                deterministic_results=self._deterministic_results,
            )
        except Exception as e:
            logger.warning("graph_orchestrator_exception", error=str(e))
            return None

        graph_trace = result.get("trace", {})
        if graph_trace.get("graph_failed"):
            trace["graph_failed"] = True
            trace["graph_fail_reason"] = graph_trace.get("graph_fail_reason", "")
            return None

        trace["served_by"] = result.get("served_by", self._client.last_served_by)
        trace["paid_used"] = result.get("paid_used", self._client.paid_used)
        trace["graph_run"] = True
        trace["graph_nodes"] = graph_trace.get("nodes", {})

        raw_predictions = result.get("predictions", {})
        predictions: dict[str, AssetPrediction] = {}
        target_set = set(self.config.general.target_assets)

        for asset, pred_raw in raw_predictions.items():
            if asset not in target_set:
                continue
            if not isinstance(pred_raw, dict):
                continue
            direction_raw = pred_raw.get("direction")
            if direction_raw not in ("UP", "DOWN"):
                continue
            try:
                confidence = float(pred_raw.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            try:
                predicted_move_pct = float(pred_raw.get("predicted_move_pct", 0.0))
            except (TypeError, ValueError):
                predicted_move_pct = 0.0
            rationale = str(pred_raw.get("rationale", ""))
            sources = _normalise_sources(pred_raw.get("sources", []))
            predictions[asset] = AssetPrediction(
                asset=asset,  # type: ignore
                direction=direction_raw,  # type: ignore
                confidence=round(confidence, 4),
                predicted_move_pct=round(predicted_move_pct, 2),
                rationale=rationale,
                sources=sources,
            )

        trace["predictions"] = {k: v.model_dump() for k, v in predictions.items()}
        trace["graph_predictions"] = raw_predictions

        best_trade = result.get("best_trade")
        market_vibe = result.get("market_vibe", "")
        trace["market_vibe"] = market_vibe
        trace["best_trade_raw"] = best_trade

        recommendation = None
        if isinstance(best_trade, dict):
            recommendation = self._parse_best_trade(best_trade, market, trace)

        if not predictions:
            trace["skip_reason"] = "graph_no_valid_predictions"
            return self._abstain(trace, start)

        # --- Apply the checker's verdict ---
        # A veto must make the trade unselectable at the decision-
        # aggregation level. DecisionAggregator.aggregate() filters on
        # `StrategyResult.recommendation is not None` and ranks/thresholds
        # on `StrategyResult.confidence` — it never looks at
        # `recommendation.confidence` directly — so both must be nulled
        # out together, not just the nested recommendation field.
        can_proceed = result.get("checker_can_proceed", True)
        contradictions = result.get("checker_contradictions", [])
        action = self.config.graph.checker_contradiction_action
        penalty = self.config.graph.checker_confidence_penalty

        trace["checker_can_proceed"] = can_proceed
        trace["checker_contradictions"] = contradictions
        trace["checker_flags"] = result.get("checker_flags", [])

        strategy_confidence = recommendation.confidence if recommendation else 0.0

        if not can_proceed and action == "veto":
            trace["checker_veto"] = True
            trace["checker_veto_reason"] = f"checker can_proceed=False: {contradictions}"
            recommendation = None
            strategy_confidence = 0.0
        elif not can_proceed and action == "penalize":
            trace["checker_penalized"] = True
            if recommendation is not None:
                recommendation.confidence = max(0.0, round(recommendation.confidence - penalty, 4))
            strategy_confidence = recommendation.confidence if recommendation else 0.0

        all_sources: list[str] = []
        for p in predictions.values():
            for s in p.sources:
                label = f"llm:{s}"
                if label not in all_sources:
                    all_sources.append(label)

        return StrategyResult(
            label=self.label,
            recommendation=recommendation,
            predictions=predictions,
            confidence=strategy_confidence,
            debug_trace=trace,
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
            forecast_source_labels=all_sources if all_sources else None,
        )

    async def _evaluate_monolithic(
        self,
        briefing: BriefingData,
        market: MarketSnapshot,
        trace: dict[str, Any],
        start: float,
    ) -> StrategyResult:
        """Run the legacy single-call LLM prediction (monolithic prompt)."""
        history_str = format_history_for_prompt(
            load_history(self.config.logging.json_dir),
        )
        trade_outcomes_str = format_outcomes_for_prompt(
            load_trade_outcomes(self.config.logging.json_dir),
        )
        prompt = _build_prompt(
            briefing,
            market,
            self.config.general.target_assets,
            history_str,
            trade_outcomes_str,
            gap_fade=self.config.gap_fade,
        )
        response = self._client.invoke(prompt=prompt, system_prompt=SYSTEM_PROMPT)

        trace["served_by"] = self._client.last_served_by
        trace["paid_used"] = self._client.paid_used
        trace["fallback_hit"] = self._client.last_fallback_hit
        trace["last_error"] = self._client.last_error

        if not response or not response.strip():
            trace["skip_reason"] = "llm_no_response"
            return self._abstain(trace, start)

        parsed = _parse_pick(response)
        if parsed is None:
            trace["skip_reason"] = "llm_unparseable"
            trace["raw_response"] = response[:300]
            return self._abstain(trace, start)

        trace["llm_raw"] = parsed

        # --- Parse predictions dict (primary output) ---
        predictions_raw = parsed.get("predictions")
        if not isinstance(predictions_raw, dict) or len(predictions_raw) == 0:
            trace["skip_reason"] = "llm_no_predictions"
            return self._abstain(trace, start)

        predictions: dict[str, AssetPrediction] = {}
        target_set = set(self.config.general.target_assets)
        for asset, pred_raw in predictions_raw.items():
            if asset not in target_set:
                continue
            if not isinstance(pred_raw, dict):
                continue
            direction_raw = pred_raw.get("direction")
            if direction_raw not in ("UP", "DOWN"):
                continue
            try:
                confidence = float(pred_raw.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            try:
                predicted_move_pct = float(pred_raw.get("predicted_move_pct", 0.0))
            except (TypeError, ValueError):
                predicted_move_pct = 0.0

            rationale = str(pred_raw.get("rationale", ""))
            sources = _normalise_sources(pred_raw.get("sources", []))

            predictions[asset] = AssetPrediction(
                asset=asset,  # type: ignore
                direction=direction_raw,  # type: ignore
                confidence=round(confidence, 4),
                predicted_move_pct=round(predicted_move_pct, 2),
                rationale=rationale,
                sources=sources,
            )

        if not predictions:
            trace["skip_reason"] = "llm_no_valid_predictions"
            return self._abstain(trace, start)

        trace["predictions"] = {k: v.model_dump() for k, v in predictions.items()}

        # --- Parse optional best_trade ---
        market_vibe = str(parsed.get("market_vibe", ""))
        trace["market_vibe"] = market_vibe

        recommendation: TradeRecommendation | None = None
        best_trade = parsed.get("best_trade")
        if isinstance(best_trade, dict):
            recommendation = self._parse_best_trade(best_trade, market, trace)

        # Build forecast_source_labels from all prediction sources
        all_sources: list[str] = []
        for p in predictions.values():
            for s in p.sources:
                label = f"llm:{s}"
                if label not in all_sources:
                    all_sources.append(label)

        return StrategyResult(
            label=self.label,
            recommendation=recommendation,
            predictions=predictions,
            confidence=recommendation.confidence if recommendation else 0.0,
            debug_trace=trace,
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
            forecast_source_labels=all_sources if all_sources else None,
        )

    def _parse_best_trade(
        self,
        best_trade: dict[str, Any],
        market: MarketSnapshot,
        trace: dict[str, Any],
    ) -> TradeRecommendation | None:
        """Parse the optional ``best_trade`` into a TradeRecommendation.

        Args:
            best_trade: Dict from the LLM with asset, direction, etc.
            market: Current market snapshot (for pricing).
            trace: Debug trace dict (mutated in-place).

        Returns:
            A TradeRecommendation or None if invalid / below threshold.
        """
        asset = best_trade.get("asset")
        direction_raw = best_trade.get("direction")
        confidence_raw = best_trade.get("confidence", 0.0)
        rationale_text = best_trade.get("rationale", "")
        sources_raw = best_trade.get("sources", [])

        if asset not in self.config.general.target_assets:
            trace["best_trade_skip"] = "asset_out_of_universe"
            return None
        if direction_raw not in ("CALL", "PUT"):
            trace["best_trade_skip"] = "invalid_direction"
            return None

        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            trace["best_trade_skip"] = "confidence_not_numeric"
            return None
        confidence = max(0.0, min(1.0, confidence))

        min_conf = self.config.llm.trade_signal_min_confidence
        if confidence < min_conf:
            trace["best_trade_skip"] = "below_min_confidence"
            trace["best_trade_min_confidence"] = min_conf
            return None

        quote = market.quotes.get(asset)
        if quote is None:
            trace["best_trade_skip"] = "no_quote"
            return None

        # MECHANICS: strike off the live pre-market price, never the
        # stale prior-session quote — see MarketSnapshot.mechanics_price.
        spot = market.mechanics_price(asset)
        if spot is None:
            trace["best_trade_skip"] = "no_mechanics_price"
            return None

        direction = Direction(direction_raw)
        strike = compute_otm_strike(spot, direction)
        delta = estimate_delta(spot, strike, 0, iv=0.20, direction=direction)
        sources = _normalise_sources(sources_raw)
        today_str = today_local().isoformat()

        rec = TradeRecommendation(
            correlation_id="",
            strategy_label=self.label,
            asset=asset,
            direction=direction,
            confidence=round(confidence, 4),
            target_strike=strike,
            contracts=min(self.config.risk.max_position_size_contracts, 1),
            order_type="market",
            position_intent=PositionIntent.BUY_TO_OPEN,
            rationale={
                "llm_rationale": rationale_text,
                "llm_sources": sources,
                "delta": round(delta, 4),
                "entry_price": spot,
                "strategy": "LLM best_trade from prediction analysis",
            },
            expires_at=today_str,
            must_close_before=self.config.risk.close_deadline_est,
        )
        trace["best_trade_parsed"] = rec.model_dump()
        return rec

    def _abstain(self, trace: dict, start: float) -> StrategyResult:
        """Build a no-recommendation :class:`StrategyResult` with trace."""
        return StrategyResult(
            label=self.label,
            recommendation=None,
            confidence=0.0,
            debug_trace=trace,
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
        )


def _gap_pct(quote: Quote, premarket: PremarketQuote | None) -> float | None:
    """Today's pre-market gap: live pre-market price vs the PRIOR SESSION close.

    This is the single "gap" definition used consistently across every
    LLM prompt builder in this module and in :mod:`src.llm.graph` — the
    overnight/pre-market move, not the move since the open. See
    :func:`_day_move_pct` for the latter.

    Before the 2026-09-11 fix this compared ``quote.open_price`` (the
    PRIOR session's own open) against ``quote.previous_close`` (the
    session before THAT) — i.e. yesterday's gap, not today's. ``quote``
    reaching this pipeline pre-market is one session stale (see
    ``Quote.prior_session_close``); the only correct source for TODAY's
    gap is a live pre-market quote (see ``src.ingestion.premarket``).

    For PROMPT/NARRATIVE purposes this deliberately returns ``None``
    (never a number silently derived from stale fields) when no live
    pre-market quote is available — the caller must disclose that
    explicitly rather than presenting a fabricated "today" gap. See
    ``MarketSnapshot.mechanics_gap_pct`` for the separate, always-numeric
    MECHANICS variant (strike selection, deterministic strategies) that
    intentionally falls back to a 0% ("no gap known") default instead.

    Args:
        quote: The (prior-session-stale) market quote.
        premarket: Live pre-market reconstruction for the same asset, or
            ``None``/unavailable.

    Returns:
        The signed gap percentage, or ``None`` when unavailable.
    """
    if premarket is None or not premarket.available:
        return None
    return premarket.gap_pct


def _day_move_pct(quote: Quote, premarket: PremarketQuote | None) -> float | None:
    """Drift WITHIN the pre-market session itself: first print vs the cutoff price.

    Distinct from :func:`_gap_pct` — that anchors to the PRIOR session's
    close (the overnight gap); this measures whether the pre-market move
    has been building or fading since it started (matches the
    ``pre_market_momentum`` field the research agents already produce).
    Before the 2026-09-11 fix this used ``quote.current_price`` vs
    ``quote.previous_close``, which pre-market are both prior-session-
    stale fields describing an entirely different day.

    Args:
        quote: The (prior-session-stale) market quote. Unused now that
            the drift is computed purely from live pre-market bars, kept
            in the signature for symmetry with :func:`_gap_pct` and so
            callers don't need to special-case which quote type feeds
            which function.
        premarket: Live pre-market reconstruction for the same asset.

    Returns:
        The signed percentage move from the session's first pre-market
        print to the cutoff price, or ``None`` when unavailable.
    """
    del quote  # unused — see docstring
    if premarket is None or not premarket.available:
        return None
    if not premarket.first_price or premarket.first_price <= 0 or premarket.price is None:
        return None
    return (premarket.price - premarket.first_price) / premarket.first_price * 100


def _premarket_prompt_block(asset: str, premarket: PremarketQuote | None) -> str:
    """Render the PRE-MARKET block for one asset's prompt section.

    Explicitly separate from the (prior-session, see
    ``Quote.prior_session_close``) quote line so the model is never left
    to guess which price is "today". When pre-market data could not be
    fetched at all, this states so explicitly rather than letting the
    model infer "today" from the stale snapshot quote — the exact
    silent-fallback failure mode this module fixes. When the move is
    real but backed by thin volume (relative to this account's own
    trailing norm — see ``PremarketConfig``), the block tells the model
    to weight it lightly, but the gap figure itself is never withheld or
    altered by that flag (MECHANICS still trades on it regardless).
    """
    if premarket is None or not premarket.available:
        return (
            f"{asset} PRE-MARKET: UNAVAILABLE as of the cutoff. No live pre-market "
            f"price could be fetched — do NOT treat the {asset} quote above (which "
            f"is the PRIOR SESSION's data) as today's price or infer today's gap "
            f"from it. Today's direction/move for {asset} is simply unknown from "
            f"price action; rely on news/catalysts instead."
        )
    price_str = f"${premarket.price:.2f}" if premarket.price is not None else "N/A"
    gap_str = f"{premarket.gap_pct:+.2f}%" if premarket.gap_pct is not None else "N/A"
    source_label = {
        "quote_midpoint": "live bid/ask midpoint",
        "bar_close": "last pre-market trade",
    }.get(premarket.price_source or "", "live")

    # Two independent reasons a move can be untrusted: no recent TRADE
    # (stale bar, regardless of a live quote existing) and thin volume
    # relative to this account's own trailing norm. Both are surfaced so
    # the model isn't told "thin volume" when the real issue is that
    # nothing has traded in a while.
    reasons: list[str] = []
    if not premarket.bar_fresh:
        age_str = (
            f"{premarket.bar_age_min:.0f} min" if premarket.bar_age_min is not None else "unknown"
        )
        reasons.append(f"the last pre-market trade is stale ({age_str} before the cutoff)")
    if premarket.volume_ratio is None:
        reasons.append("no trailing-volume baseline could be computed")
    elif not premarket.reliable and premarket.bar_fresh:
        # Only cite thin volume on its own when staleness isn't already
        # the reason above (avoids "stale AND thin" restating one cause).
        reasons.append(
            f"volume is only {premarket.volume_ratio:.2f}x the "
            f"{premarket.lookback_days_used}-day median at this time of day"
        )

    if premarket.reliable:
        reliability = (
            f"RELIABLE (volume is {premarket.volume_ratio:.2f}x the "
            f"{premarket.lookback_days_used}-day median at this time of day, "
            "last trade recent)"
            if premarket.volume_ratio is not None
            else "RELIABLE"
        )
        weight_note = ""
    else:
        reliability = "THIN/STALE (" + "; ".join(reasons) + ")"
        weight_note = (
            f" Because of this, weight this {asset} pre-market move LIGHTLY as "
            "analysis evidence — treat it as a weak signal, not confirmation."
        )

    return (
        f"{asset} PRE-MARKET (as of cutoff): last {price_str} ({source_label}), gap vs "
        f"prior session close {gap_str}. Volume/recency reliability: {reliability}.{weight_note}"
    )


def _build_prompt(
    briefing: BriefingData,
    market: MarketSnapshot,
    target_assets: list[str],
    prediction_history: str = "",
    trade_outcomes: str = "",
    gap_fade: GapFadeConfig | None = None,
) -> str:
    """Assemble the LLM prediction prompt from briefing + market data.

    Args:
        briefing: Parsed morning briefing.
        market: Current market snapshot.
        target_assets: Configured asset universe (e.g. ``["SPY", "QQQ"]``).
        prediction_history: Formatted string of past prediction outcomes.
        trade_outcomes: Formatted string of actual trade PnL outcomes.
        gap_fade: Gap-fade thresholds (per-asset threshold pct + sentiment
            magnitude cutoff). Defaults to :class:`GapFadeConfig` defaults
            when omitted.

    Returns:
        A prompt string suitable for the opencode CLI single-positional
        argument.
    """
    gap_fade = gap_fade or GapFadeConfig()
    sections: list[str] = []

    sections.append(
        f"Target assets for today's prediction: {', '.join(target_assets)}. "
        "You must produce a prediction for EACH asset."
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

    quote_lines = []
    for asset in target_assets:
        q = market.quotes.get(asset)
        if q is None:
            continue
        # PRIOR SESSION, not today — see Quote.prior_session_close. Every
        # free source this pipeline uses reports no live price pre-market,
        # so this whole block describes the most recently COMPLETED
        # session, labeled explicitly so the model never mistakes it for
        # today's price. Today's actual price/gap is the PRE-MARKET block
        # below, built separately from live Alpaca data.
        quote_lines.append(
            f"- {q.symbol} PRIOR SESSION: closed ${q.prior_session_close:.2f} "
            f"(open ${q.open_price:.2f}, high ${q.high_price:.2f}, low ${q.low_price:.2f})."
        )
    if quote_lines:
        sections.append(
            "Target-asset prior-session data (NOT today — see the PRE-MARKET "
            "block below for today's actual price):\n" + "\n".join(quote_lines)
        )

    premarket_lines = [
        _premarket_prompt_block(asset, market.premarket.get(asset))
        for asset in target_assets
        if market.quotes.get(asset) is not None
    ]
    if premarket_lines:
        sections.append("Pre-market data:\n" + "\n".join(premarket_lines))

    threshold_lines = [
        f"Gap-fade threshold for {asset}: {gap_fade.threshold_for(asset)}% "
        f"(flag when |gap| exceeds this AND sentiment magnitude < "
        f"{gap_fade.sentiment_magnitude_max})"
        for asset in target_assets
        if market.quotes.get(asset) is not None
    ]
    if threshold_lines:
        sections.append("Gap-fade thresholds:\n" + "\n".join(threshold_lines))

    gap_alerts: list[str] = []
    gap_fade_risk: list[str] = []
    for asset in target_assets:
        q = market.quotes.get(asset)
        if q is None:
            continue
        pm = market.premarket.get(asset)
        gap_pct = _gap_pct(q, pm)
        if gap_pct is None:
            continue
        if abs(gap_pct) > 0.5:
            gap_alerts.append(
                f"  - {q.symbol} has gapped {gap_pct:+.1f}% pre-market from "
                f"the prior session's close (${q.prior_session_close:.2f})."
            )
        gap_threshold = gap_fade.threshold_for(q.symbol)
        sentiment_mag = abs(market.avg_sentiment_polarity())
        if abs(gap_pct) > gap_threshold and sentiment_mag < gap_fade.sentiment_magnitude_max:
            gap_fade_risk.append(
                f"  - {q.symbol} gap ({gap_pct:+.1f}%) is large relative to "
                f"catalyst strength (sentiment {sentiment_mag:+.3f}). "
                f"This is a gap-fade risk — the overnight move may exhaust and "
                f"reverse during the session, as it did on Aug 5 2026 "
                f"(SPY +1.8% gap -> -0.8% close, QQQ +3.4% gap -> -1.2% close)."
            )
    if gap_fade_risk:
        sections.append(
            "Gap-fade risk warning: the following assets show pre-market gaps "
            "that are disproportionately large compared to news catalyst "
            "strength. Consider whether the gap is sustainable or likely to "
            "fade. On Aug 5 2026, both SPY and QQQ showed this pattern and "
            "reversed hard.\n" + "\n".join(gap_fade_risk)
        )
    if gap_alerts:
        alert_block = (
            "Pre-market gap alert: the following assets show significant "
            "gaps from yesterday's close. The briefing sections below were "
            "built before these gaps were fully visible. Consider whether "
            "each gap confirms or contradicts the briefing's thesis.\n" + "\n".join(gap_alerts)
        )
        sections.append(alert_block)

    if market.news:
        top_news = market.news[:10]
        news_lines = [
            f"- [{n.source}] {n.title}" + (f" — {n.snippet}" if n.snippet else "") for n in top_news
        ]
        sections.append("Top market news:\n" + "\n".join(news_lines))

    polarity = market.avg_sentiment_polarity()
    sections.append(f"Market-average news sentiment polarity: {polarity:+.3f}")

    if prediction_history:
        sections.append(
            f"Recent prediction history (learn from past outcomes):\n{prediction_history}"
        )

    if trade_outcomes:
        sections.append(trade_outcomes)

    return "\n\n".join(sections)


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*\})\s*```")


def _balanced_object_at(text: str, start: int) -> str | None:
    """Return the substring of a balanced ``{...}`` object starting at ``start``.

    Tracks JSON string state so braces that appear inside string values
    (e.g. a rationale sentence containing ``"{"``) aren't mistaken for
    structural braces. Used by :func:`_parse_pick` to scan for the
    correct closing brace instead of greedily matching from the first
    ``{`` to the LAST ``}`` in the text, which swallows everything
    between two separate JSON objects or between the real object and
    trailing braced prose.

    Args:
        text: The full text being scanned.
        start: Index of the opening ``{``.

    Returns:
        The balanced ``{...}`` substring, or ``None`` if no matching
        close brace is found before the end of ``text``.
    """
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_pick(response: str) -> dict[str, Any] | None:
    """Extract a JSON object from an LLM response.

    The opencode CLI may wrap the JSON in prose or code fences despite
    the system prompt asking for bare JSON, so this scans for a
    BALANCED ``{...}`` span starting at each ``{`` in the text (rather
    than a greedy regex from the first ``{`` to the last ``}``, which
    would span two separate JSON objects or trailing braced prose) and
    returns the first one that parses as a JSON object. Accepts either
    bare JSON or JSON embedded in markdown code fences.

    Args:
        response: Raw LLM response text.

    Returns:
        Parsed dict or ``None`` when no JSON object can be extracted.
    """
    if not response or not response.strip():
        return None
    text = response.strip()
    if text.startswith("```"):
        fenced = _FENCE_RE.search(text)
        if fenced:
            text = fenced.group(1)

    for i, ch in enumerate(text):
        if ch != "{":
            continue
        candidate = _balanced_object_at(text, i)
        if candidate is None:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalise_sources(sources_raw: Any) -> list[str]:
    """Normalise the LLM's ``sources`` array into 1-3 provenance strings.

    Accepts either a list of strings or a single string. Drops empty /
    overlong entries, clamps the list to 1-3 items (keeping order), and
    falls back to ``["atlas-briefing"]`` when the LLM omitted the field
    entirely or returned something unparseable.

    Args:
        sources_raw: Whatever the LLM put under the ``sources`` key.

    Returns:
        A list of 1-3 short citation strings.
    """
    if isinstance(sources_raw, str):
        candidates = [sources_raw]
    elif isinstance(sources_raw, list):
        candidates = [s for s in sources_raw if isinstance(s, str)]
    else:
        candidates = []

    cleaned: list[str] = []
    for s in candidates:
        s = s.strip()
        if not s or len(s) > 120:
            continue
        cleaned.append(s)
        if len(cleaned) == 3:
            break

    if not cleaned:
        return ["news-sentiment"]
    return cleaned
