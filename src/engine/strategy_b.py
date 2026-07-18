"""Mean-reversion trading strategy implementation."""

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


class MeanReversionStrategy(TradingStrategy):
    """Mean-reversion trading strategy — trades against extreme RSI moves with sentiment filtering."""

    def __init__(self, config: Settings):
        """Initialize MeanReversionStrategy with application settings.

        Args:
            config: Application settings.
        """
        super().__init__(label="mean_reversion", config=config)

    async def evaluate(
        self,
        briefing: BriefingData,
        market: MarketSnapshot,
    ) -> StrategyResult:
        """Evaluate mean-reversion signals for each target asset.

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
                logger.warning("mean_reversion_no_quote", asset=asset)
                continue

            briefing_sentiment = briefing.macro_sentiment
            news_sentiment = market.avg_sentiment_polarity()
            combined_sentiment = (news_sentiment + briefing_sentiment) / 2
            trace[f"{asset}_combined_sentiment"] = combined_sentiment

            prior_close = quote.previous_close
            current = quote.current_price
            if prior_close <= 0:
                trace[f"{asset}_skip_reason"] = "no_prior_close"
                continue

            move_from_close_pct = (current - prior_close) / prior_close * 100
            trace[f"{asset}_move_from_close_pct"] = move_from_close_pct

            direction = None
            confidence = 0.0
            min_conf = self.config.strategies.mean_reversion.min_confidence
            oversold = self.config.strategies.mean_reversion.rsi_oversold
            overbought = self.config.strategies.mean_reversion.rsi_overbought

            rsi_est = self._estimate_rsi_from_move(move_from_close_pct)
            trace[f"{asset}_estimated_rsi"] = rsi_est

            if rsi_est <= oversold and combined_sentiment > -0.1:
                direction = Direction.CALL
                confidence = min(0.8, 0.35 + (oversold - rsi_est) / 50)
            elif rsi_est >= overbought and combined_sentiment < 0.1:
                direction = Direction.PUT
                confidence = min(0.8, 0.35 + (rsi_est - overbought) / 50)
            else:
                direction = Direction.PUT if move_from_close_pct > 0.5 else Direction.CALL
                confidence = 0.25

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
                    "move_from_close_pct": round(move_from_close_pct, 2),
                    "estimated_rsi": rsi_est,
                    "combined_sentiment": combined_sentiment,
                    "oversold_threshold": oversold,
                    "overbought_threshold": overbought,
                    "delta": round(delta, 4),
                    "entry_price": quote.current_price,
                    "strategy": "Mean reversion: RSI extreme + sentiment check",
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

    @staticmethod
    def _estimate_rsi_from_move(move_pct: float) -> float:
        """Estimate RSI from a percentage price move using a linear heuristic.

        Args:
            move_pct: Percentage price move.

        Returns:
            Estimated RSI value between 0 and 100.
        """
        normalized = max(-5.0, min(5.0, move_pct))
        return round(50 + normalized * 10, 1)
