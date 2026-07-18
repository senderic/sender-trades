"""Event-driven trading strategy implementation."""

from __future__ import annotations

import time
from datetime import date

import structlog

from src.config import Settings
from src.engine.base import TradingStrategy
from src.engine.options_strategy import compute_otm_strike, estimate_delta
from src.models.briefing import BriefingData
from src.models.market import MarketSnapshot
from src.models.recommendation import Direction, PositionIntent, StrategyResult, TradeRecommendation

logger = structlog.get_logger()

CATALYST_KEYWORDS = [
    "fed",
    "interest rate",
    "cpi",
    "ppi",
    "unemployment",
    "jobs report",
    "earnings",
    "revenue",
    "guidance",
    "buyback",
    "dividend",
    "tariff",
    "trade deal",
    "regulation",
    "antitrust",
    "geopolitical",
    "sanctions",
    "conflict",
    "ceasefire",
    "inflation",
    "deflation",
    "recession",
    "gdp",
    "consumer sentiment",
    "retail sales",
    "manufacturing",
    "sp500",
    "nasdaq",
    "dow jones",
    "futures",
]


class EventDrivenStrategy(TradingStrategy):
    """Event-driven trading strategy — detects catalysts from briefing/news and trades on pre-market reaction."""

    def __init__(self, config: Settings):
        """Initialize EventDrivenStrategy with application settings.

        Args:
            config: Application settings.
        """
        super().__init__(label="event_driven", config=config)

    async def evaluate(
        self,
        briefing: BriefingData,
        market: MarketSnapshot,
    ) -> StrategyResult:
        """Evaluate event-driven signals by detecting catalysts and pre-market moves.

        Args:
            briefing: Parsed morning briefing data.
            market: Current market snapshot.

        Returns:
            StrategyResult containing an optional trade recommendation.
        """
        start = time.perf_counter()
        trace: dict = {}
        recommendation = None

        catalysts = self._detect_catalysts(briefing, market)
        trace["catalyst_count"] = len(catalysts)
        trace["catalysts"] = catalysts[:5]

        if not catalysts:
            return StrategyResult(
                label=self.label,
                recommendation=None,
                confidence=0.0,
                debug_trace={"skip_reason": "no_catalysts_detected", **trace},
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
            )

        catalyst_polarity = self._aggregate_catalyst_polarity(catalysts)
        trace["catalyst_polarity"] = catalyst_polarity

        for asset in self.config.general.target_assets:
            quote = market.quotes.get(asset)
            if quote is None:
                logger.warning("event_driven_no_quote", asset=asset)
                continue

            prior_close = quote.previous_close
            current = quote.current_price
            premarket_move_pct = (
                (current - prior_close) / prior_close * 100 if prior_close > 0 else 0.0
            )
            trace[f"{asset}_premarket_move_pct"] = premarket_move_pct

            sentiment = market.avg_sentiment_polarity()
            briefing_sent = briefing.macro_sentiment
            combined = (sentiment + briefing_sent + catalyst_polarity) / 3
            trace[f"{asset}_combined_sentiment"] = combined

            direction = None
            confidence = 0.0
            gap_pct = premarket_move_pct

            if combined > 0.1:
                direction = Direction.CALL
                confidence = min(0.85, 0.4 + abs(combined) + abs(gap_pct) / 15)
            elif combined < -0.1:
                direction = Direction.PUT
                confidence = min(0.85, 0.4 + abs(combined) + abs(gap_pct) / 15)
            else:
                trace[f"{asset}_skip_reason"] = "neutral_combined_sentiment"
                trace[f"{asset}_confidence_base"] = combined
                continue

            min_conf = self.config.strategies.event_driven.min_confidence
            if confidence < min_conf:
                trace[f"{asset}_skip_reason"] = "below_min_confidence"
                trace[f"{asset}_confidence"] = confidence
                continue

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
                    "catalyst_count": len(catalysts),
                    "catalyst_polarity": catalyst_polarity,
                    "premarket_move_pct": round(premarket_move_pct, 2),
                    "combined_sentiment": combined,
                    "delta": round(delta, 4),
                    "top_catalysts": catalysts[:3],
                    "strategy": "Event-driven: overnight catalyst + pre-market reaction",
                },
                expires_at=today_str,
                must_close_before=self.config.risk.close_deadline_est,
            )
            break

        duration = (time.perf_counter() - start) * 1000
        return StrategyResult(
            label=self.label,
            recommendation=recommendation,
            confidence=recommendation.confidence if recommendation else 0.0,
            debug_trace=trace,
            duration_ms=round(duration, 2),
        )

    def _detect_catalysts(
        self,
        briefing: BriefingData,
        market: MarketSnapshot,
    ) -> list[dict]:
        """Detect market catalysts from briefing text, news, and RSS items.

        Args:
            briefing: Parsed briefing data.
            market: Current market snapshot.

        Returns:
            List of catalyst dicts with keyword and source fields.
        """
        catalysts: list[dict] = []
        all_text = briefing.executive_summary + " " + briefing.key_connections + " "
        for item in briefing.news_items:
            all_text += f" {item.title} {item.snippet}" if item.snippet else f" {item.title}"
        for headline in market.news:
            all_text += f" {headline.title} {headline.snippet}"
        for rss in market.rss_items:
            all_text += f" {rss.title} {rss.summary}"

        all_text_lower = all_text.lower()
        for keyword in CATALYST_KEYWORDS:
            if keyword in all_text_lower:
                catalysts.append({"keyword": keyword, "source": "cross_source"})
        return catalysts

    def _aggregate_catalyst_polarity(self, catalysts: list[dict]) -> float:
        """Compute the net polarity of detected catalysts.

        Args:
            catalysts: List of catalyst dicts from _detect_catalysts.

        Returns:
            Signed polarity between -1.0 and 1.0.
        """
        positive_keywords = {
            "fed",
            "buyback",
            "dividend",
            "ceasefire",
            "consumer sentiment",
            "gdp",
            "retail sales",
        }
        negative_keywords = {
            "tariff",
            "sanctions",
            "conflict",
            "recession",
            "inflation",
            "antitrust",
            "unemployment",
        }
        pos = sum(1 for c in catalysts if c["keyword"] in positive_keywords)
        neg = sum(1 for c in catalysts if c["keyword"] in negative_keywords)
        total = pos + neg
        if total == 0:
            return 0.0
        return round((pos - neg) / total, 4)
