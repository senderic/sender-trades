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


class MomentumStrategy(TradingStrategy):
    """Momentum trading strategy — enters trades on gap moves with sentiment confirmation."""

    def __init__(self, config: Settings):
        super().__init__(label="momentum", config=config)

    async def evaluate(
        self,
        briefing: BriefingData,
        market: MarketSnapshot,
    ) -> StrategyResult:
        """Evaluate momentum signals for each target asset.

        Args:
            briefing: Parsed morning briefing data.
            market: Current market snapshot.

        Returns:
            StrategyResult containing an optional trade recommendation.
        """
        start = time.perf_counter()
        trace: dict = {}
        recommendation = None

        for asset in self.config.general.target_assets:
            quote = market.quotes.get(asset)
            if quote is None:
                logger.warning("momentum_no_quote", asset=asset)
                continue

            gap_pct = (
                (quote.open_price - quote.previous_close) / quote.previous_close * 100
                if quote.previous_close > 0 else 0.0
            )
            trace[f"{asset}_gap_pct"] = gap_pct

            news_sentiment = market.avg_sentiment_polarity()
            briefing_sentiment = briefing.macro_sentiment
            combined_sentiment = (news_sentiment + briefing_sentiment) / 2
            trace[f"{asset}_combined_sentiment"] = combined_sentiment

            direction = None
            confidence = 0.0
            gap_threshold = self.config.strategies.momentum.gap_threshold_pct
            min_conf = self.config.strategies.momentum.min_confidence

            if gap_pct > gap_threshold and combined_sentiment > 0.05:
                direction = Direction.CALL
                confidence = min(0.9, 0.4 + abs(gap_pct) / 10 + abs(combined_sentiment))
            elif gap_pct < -gap_threshold and combined_sentiment < -0.05:
                direction = Direction.PUT
                confidence = min(0.9, 0.4 + abs(gap_pct) / 10 + abs(combined_sentiment))
            elif combined_sentiment > 0.15:
                direction = Direction.CALL
                confidence = 0.35 + abs(combined_sentiment)
            elif combined_sentiment < -0.15:
                direction = Direction.PUT
                confidence = 0.35 + abs(combined_sentiment)

            if direction is None or confidence < min_conf:
                trace[f"{asset}_skip_reason"] = "low_confidence_or_no_direction"
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
                    "gap_pct": round(gap_pct, 2),
                    "combined_sentiment": combined_sentiment,
                    "news_sentiment": news_sentiment,
                    "briefing_sentiment": briefing_sentiment,
                    "delta": round(delta, 4),
                    "entry_price": quote.current_price,
                    "strategy": "Momentum: gap continuation + sentiment confirmation",
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
