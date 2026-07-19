"""LLM-driven trade-signal strategy.

Asks the opencode LLM (with the same Zen-first / paid-Go fallback chain
as the briefing re-synthesiser) to emit a single structured JSON pick
for today's 0DTE session:

    {"asset": "SPY" | "QQQ",
     "direction": "CALL" | "PUT",
     "confidence": 0.0-1.0,
     "rationale": "short one-sentence reason",
     "sources": [
        "atlas-briefing:executive_summary",
        "reuters:<headline>",
        "watchlist:<TICKER>",
        ...
     ]}

The pick is validated, clamped to the configured asset universe, and
returned as a :class:`StrategyResult` with label ``"llm_trade"`` so the
existing :class:`DecisionAggregator` folds it into the final decision
alongside Momentum / MeanReversion / EventDriven.

The ``sources`` array carries *root provenance* -- the LLM must cite
which upstream inputs drove its conclusion, using one of:

- ``"atlas-briefing:<section>"`` for content originally distilled by
  the upstream atlas-morning-briefing LLM (e.g.
  ``"atlas-briefing:executive_summary"``).
- ``"<publisher>:<slug-or-headline>"`` for market-news feed items
  (e.g. ``"reuters:kimi-k3-open-weight"``).
- ``"watchlist:<TICKER>"`` for watchlist ticker drivers.
- ``"market:<TICKER>"`` for live / snapshot quote metrics.
- ``"news-sentiment"`` for the aggregate-polarity signal.

These citations are surfaced verbatim (prefixed with ``llm:``) as the
forecast table's ``up_sources`` / ``down_sources`` entries, so the
reader sees ``↓llm:atlas-briefing+reuters+watchlist:NVDA`` rather than
the opaque ``↓llm_trade``.

This makes the LLM an explicit participant in the buy/sell decision
rather than only indirectly influencing it via the re-synthesised
executive summary's word-counted sentiment, and it preserves the
provenance of *why* the LLM picked what it picked.
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
    Direction,
    PositionIntent,
    StrategyResult,
    TradeRecommendation,
)

logger = structlog.get_logger()


SYSTEM_PROMPT = (
    "You are the decision layer of an intraday 0DTE options trading "
    "system. Choose exactly ONE trade for today's session and emit it "
    "as a single JSON object with these keys: "
    'asset ("SPY" or "QQQ"), direction ("CALL" or "PUT"), confidence '
    "(a float in [0.0, 1.0] reflecting how strongly the evidence "
    "supports the pick), rationale (one short sentence citing the "
    "specific evidence that drove the choice), and sources (a list "
    "of 1-3 strings citing the ROOT provenance -- where the evidence "
    "originally came from, not this LLM. Use these citation forms: "
    '"atlas-briefing:<section>" for content distilled by the upstream '
    "atlas-morning-briefing LLM (e.g. "
    '"atlas-briefing:executive_summary", "atlas-briefing:watchlist"); '
    '"<publisher>:<short-slug>" for market news feed items (e.g. '
    '"reuters:kimi-k3-release", "bloomberg:fed-cautious-stance"); '
    '"watchlist:<TICKER>" for a specific watchlist row that drove '
    "the call; "
    '"market:<TICKER>" for raw quote / price-action evidence; '
    '"news-sentiment" for the aggregate news-polarity reading). '
    "Do NOT cite this LLM itself or the trade-signal strategy. Cite "
    "the upstream source that produced the evidence. "
    "Output ONLY the JSON object. No prose, no code fence, no "
    "explanation outside the JSON."
)


class LLMTradeStrategy(TradingStrategy):
    """Trading strategy that delegates the buy/sell decision to an LLM.

    Unlike Momentum / MeanReversion / EventDriven, which derive direction
    and confidence from deterministic price/sentiment math, this strategy
    asks the opencode LLM to emit a structured JSON trade pick given the
    same briefing + market snapshot inputs. The LLM's pick is validated
    against the configured asset universe and clamped to a sane
    confidence range before being wrapped in a :class:`TradeRecommendation`.

    The LLM call goes through :class:`OpencodeLLMClient`, which tries
    every free Zen model before any paid Go model (see
    :class:`src.config.LLMConfig`). When the LLM is unavailable, returns
    junk, or emits an unparseable response, the strategy abstains
    (returns ``recommendation=None``) rather than guessing.
    """

    def __init__(self, config: Settings, client: OpencodeLLMClient | None = None):
        """Initialize LLMTradeStrategy.

        Args:
            config: Application settings. ``config.llm`` controls the
                opencode chain; ``config.llm.trade_signal_min_confidence``
                gates the strategy's confidence floor.
            client: Optional pre-constructed :class:`OpencodeLLMClient`
                (useful for tests). A new client is built from
                ``config.llm`` when omitted.
        """
        super().__init__(label="llm_trade", config=config)
        self._client = client or OpencodeLLMClient(config.llm)

    async def evaluate(
        self,
        briefing: BriefingData,
        market: MarketSnapshot,
    ) -> StrategyResult:
        """Evaluate the LLM trade-signal strategy.

        Args:
            briefing: Parsed morning briefing data.
            market: Current market snapshot with quotes and news.

        Returns:
            A :class:`StrategyResult` with a :class:`TradeRecommendation`
            built from the LLM's JSON pick, or ``recommendation=None``
            when the LLM was unavailable, abstained, or produced an
            unparseable / invalid response.
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

        pick = _parse_pick(response)
        if pick is None:
            trace["skip_reason"] = "llm_unparseable"
            trace["raw_response"] = response[:300]
            return self._abstain(trace, start)

        trace["llm_pick"] = pick

        asset = pick.get("asset")
        direction_raw = pick.get("direction")
        confidence_raw = pick.get("confidence", 0.0)
        rationale_text = pick.get("rationale", "")
        sources_raw = pick.get("sources", [])

        if asset not in self.config.general.target_assets:
            trace["skip_reason"] = "llm_asset_out_of_universe"
            return self._abstain(trace, start)
        if direction_raw not in ("CALL", "PUT"):
            trace["skip_reason"] = "llm_invalid_direction"
            return self._abstain(trace, start)

        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            trace["skip_reason"] = "llm_confidence_not_numeric"
            return self._abstain(trace, start)
        confidence = max(0.0, min(1.0, confidence))
        trace["llm_confidence_clamped"] = confidence

        # Extract and normalise provenance citations. The LLM is asked
        # to return 1-3 root-source strings (see SYSTEM_PROMPT). We
        # defensively default to ["llm:atlas-briefing"] when missing /
        # malformed so the forecast still surfaces something honest
        # ("we don't know which upstream input the LLM leaned on")
        # rather than the opaque "llm_trade" label.
        sources = _normalise_sources(sources_raw)
        trace["llm_sources_raw"] = sources_raw
        trace["llm_sources"] = sources
        # Forecast-table labels prefix each source with "llm:" so the
        # reader sees the LLM was the reasoner *and* what it read.
        forecast_source_labels = [f"llm:{s}" for s in sources]

        direction = Direction(direction_raw)
        quote = market.quotes.get(asset)
        if quote is None:
            trace["skip_reason"] = "no_quote_for_llm_asset"
            return self._abstain(trace, start)

        min_conf = self.config.llm.trade_signal_min_confidence
        if confidence < min_conf:
            trace["skip_reason"] = "below_min_confidence"
            trace["min_confidence"] = min_conf
            return self._abstain(trace, start)

        strike = compute_otm_strike(quote.current_price, direction)
        delta = estimate_delta(quote.current_price, strike, 0, iv=0.20, direction=direction)
        today_str = date.today().isoformat()

        recommendation = TradeRecommendation(
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
                "llm_served_by": self._client.last_served_by,
                "llm_paid_used": self._client.paid_used,
                "llm_direction": direction.value,
                "llm_confidence_raw": confidence_raw,
                "llm_rationale": rationale_text,
                "llm_sources": sources,
                "delta": round(delta, 4),
                "entry_price": quote.current_price,
                "strategy": "LLM trade signal: opencode JSON pick over briefing+market",
            },
            expires_at=today_str,
            must_close_before=self.config.risk.close_deadline_est,
        )
        return StrategyResult(
            label=self.label,
            recommendation=recommendation,
            confidence=recommendation.confidence,
            debug_trace=trace,
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
            forecast_source_labels=forecast_source_labels,
        )

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
    """Assemble the LLM trade-signal prompt from briefing + market data.

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
        f"Target asset universe for today's pick: {', '.join(target_assets)}. "
        "Choose exactly ONE asset from this list."
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
    """Extract the first JSON object from an LLM response and validate keys.

    The opencode CLI may wrap the JSON in prose or code fences despite
    the system prompt asking for bare JSON, so we regex for the first
    ``{...}`` block and try to parse it. Accepts either bare JSON or
    JSON embedded in markdown code fences.

    Args:
        response: Raw LLM response text.

    Returns:
        Parsed dict with at least ``asset`` / ``direction`` / ``confidence``
        / ``rationale`` keys, or ``None`` when the response cannot be
        parsed into a valid pick.
    """
    text = response.strip()
    # Strip markdown code fences if present.
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
    entirely or returned something unparseable. The fallback is honest:
    it tells the reader "the LLM was fed the atlas-briefing and may have
    leaned on it, but it did not specify which upstream source" -- far
    better than the opaque ``"llm_trade"`` strategy name.

    Args:
        sources_raw: Whatever the LLM put under the ``sources`` key.

    Returns:
        A list of 1-3 short citation strings ready to be prefixed with
        ``"llm:"`` for the forecast table.
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
        # Reject empty / whitespace-only / absurdly long citations.
        if not s or len(s) > 120:
            continue
        cleaned.append(s)
        if len(cleaned) == 3:
            break

    if not cleaned:
        return ["atlas-briefing"]
    return cleaned
