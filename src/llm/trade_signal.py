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
from datetime import date
from typing import Any

import structlog

from src.config import Settings
from src.engine.base import TradingStrategy
from src.engine.options_strategy import compute_otm_strike, estimate_delta
from src.llm.client import OpencodeLLMClient
from src.models.briefing import BriefingData
from src.models.market import MarketSnapshot
from src.models.recommendation import (
    AssetPrediction,
    Direction,
    PositionIntent,
    StrategyResult,
    TradeRecommendation,
)

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
    "points to a specific executable trade today\n\n"
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
    every free Zen model before any paid Go model (see
    :class:`src.config.LLMConfig`). When the LLM is unavailable, returns
    junk, or emits an unparseable response, the strategy abstains
    (returns ``recommendation=None``) rather than guessing.
    """

    def __init__(self, config: Settings, client: OpencodeLLMClient | None = None):
        assert set(config.general.target_assets).issubset({"SPY", "QQQ"}), (
            f"LLMTradeStrategy only supports SPY/QQQ, got {config.general.target_assets}"
        )
        super().__init__(label="llm_trade", config=config)
        self._client = client or OpencodeLLMClient(config.llm)

    async def evaluate(
        self,
        briefing: BriefingData,
        market: MarketSnapshot,
    ) -> StrategyResult:
        """Evaluate the LLM prediction strategy.

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

        prompt = _build_prompt(briefing, market, self.config.general.target_assets)
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

        direction = Direction(direction_raw)
        strike = compute_otm_strike(quote.current_price, direction)
        delta = estimate_delta(quote.current_price, strike, 0, iv=0.20, direction=direction)
        sources = _normalise_sources(sources_raw)
        today_str = date.today().isoformat()

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
                "entry_price": quote.current_price,
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


def _build_prompt(
    briefing: BriefingData,
    market: MarketSnapshot,
    target_assets: list[str],
) -> str:
    """Assemble the LLM prediction prompt from briefing + market data.

    Args:
        briefing: Parsed morning briefing.
        market: Current market snapshot.
        target_assets: Configured asset universe (e.g. ``["SPY", "QQQ"]``).

    Returns:
        A prompt string suitable for the opencode CLI single-positional
        argument.
    """
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
        gap_pct = (
            (q.open_price - q.previous_close) / q.previous_close * 100
            if q.previous_close > 0
            else 0.0
        )
        move_pct = (
            (q.current_price - q.previous_close) / q.previous_close * 100
            if q.previous_close > 0
            else 0.0
        )
        quote_lines.append(
            f"- {q.symbol}: ${q.current_price:.2f} "
            f"(gap {gap_pct:+.2f}% from prev close {q.previous_close:.2f}, "
            f"now {move_pct:+.2f}% on day)"
        )
    if quote_lines:
        sections.append("Target-asset quotes:\n" + "\n".join(quote_lines))

    if market.news:
        top_news = market.news[:10]
        news_lines = [
            f"- [{n.source}] {n.title}" + (f" — {n.snippet}" if n.snippet else "") for n in top_news
        ]
        sections.append("Top market news:\n" + "\n".join(news_lines))

    polarity = market.avg_sentiment_polarity()
    sections.append(f"Market-average news sentiment polarity: {polarity:+.3f}")

    return "\n\n".join(sections)


_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _parse_pick(response: str) -> dict[str, Any] | None:
    """Extract the first JSON object from an LLM response.

    The opencode CLI may wrap the JSON in prose or code fences despite
    the system prompt asking for bare JSON, so we regex for the first
    ``{...}`` block and try to parse it. Accepts either bare JSON or
    JSON embedded in markdown code fences.

    Args:
        response: Raw LLM response text.

    Returns:
        Parsed dict or ``None`` when the response cannot be parsed.
    """
    text = response.strip()
    if text.startswith("```"):
        fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text)
        if fenced:
            text = fenced.group(1)
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


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
