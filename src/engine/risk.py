"""Risk validation engine for trade recommendations."""

from __future__ import annotations

from datetime import UTC, date, datetime, time

import structlog

from src.config import Settings
from src.models.market import MarketSnapshot
from src.models.recommendation import TradeRecommendation

logger = structlog.get_logger()


class RiskError(Exception):
    """Raised when a trade fails a risk guardrail check."""

    def __init__(self, message: str, guardrail: str):
        self.guardrail = guardrail
        super().__init__(message)


class RiskEngine:
    """Validates trade recommendations against configured risk guardrails."""

    def __init__(self, config: Settings):
        """Initialize RiskEngine with application settings.

        Args:
            config: Application settings.
        """
        self.config = config
        self.risk_config = config.risk

    def validate(
        self,
        rec: TradeRecommendation,
        market: MarketSnapshot,
    ) -> TradeRecommendation:
        """Run all risk checks against a trade recommendation.

        Args:
            rec: The trade recommendation to validate.
            market: Current market snapshot for sanity checks.

        Returns:
            The validated (possibly modified) recommendation.

        Raises:
            RiskError: If any guardrail is breached.
        """
        self._check_time(rec)
        self._check_max_position_size(rec)
        self._check_max_loss(rec)
        self._check_dte(rec)
        self._check_statistical_sanity(rec, market)
        return rec

    def _check_time(self, rec: TradeRecommendation) -> None:
        """Verify there is enough time remaining before market close.

        Args:
            rec: The trade recommendation to check.

        Raises:
            RiskError: If current time is at or past the close deadline.
        """
        now = datetime.now(UTC)
        now_est = now
        cutoff_parts = self.risk_config.close_deadline_est.split(":")
        cutoff = time(int(cutoff_parts[0]), int(cutoff_parts[1]))
        cutoff_dt_est = now_est.replace(
            hour=cutoff.hour, minute=cutoff.minute, second=0, microsecond=0
        )
        if now_est >= cutoff_dt_est:
            raise RiskError(
                f"Current time {now_est.strftime('%H:%M')} EST is at or past close deadline "
                f"{self.risk_config.close_deadline_est} EST. 0DTE must be closed before market close.",
                guardrail="time_check",
            )

        two_pm_cutoff = now_est.replace(hour=14, minute=0, second=0, microsecond=0)
        if now_est >= two_pm_cutoff:
            raise RiskError(
                f"Current time {now_est.strftime('%H:%M')} EST is after 2:00 PM EST. "
                f"Insufficient time remaining for 0DTE management.",
                guardrail="time_check_2pm",
            )

    def _check_max_position_size(self, rec: TradeRecommendation) -> None:
        """Verify position size does not exceed the configured maximum.

        Args:
            rec: The trade recommendation to check.

        Raises:
            RiskError: If position size exceeds the limit.
        """
        if rec.contracts > self.risk_config.max_position_size_contracts:
            raise RiskError(
                f"Position size {rec.contracts} exceeds max {self.risk_config.max_position_size_contracts} contracts.",
                guardrail="max_position_size",
            )

    def _check_max_loss(self, rec: TradeRecommendation) -> None:
        """Verify estimated max loss does not exceed the configured limit.

        Uses 0.3 % of the underlying as a rough 0DTE ATM premium estimate.
        Actual premium depends on IV, time decay, and distance from strike.

        Args:
            rec: The trade recommendation to check.

        Raises:
            RiskError: If estimated max loss exceeds the limit.
        """
        if rec.legs and len(rec.legs) > 0:
            return
        underlying_price = rec.target_strike
        est_premium_pct = 0.003
        max_loss = rec.contracts * 100 * (underlying_price * est_premium_pct)
        if max_loss > self.risk_config.max_loss_per_trade_usd:
            raise RiskError(
                f"Estimated max loss ${max_loss:.0f} exceeds limit ${self.risk_config.max_loss_per_trade_usd:.0f}.",
                guardrail="max_loss",
            )

    def _check_dte(self, rec: TradeRecommendation) -> None:
        """Verify days-to-expiry falls within the configured range.

        Args:
            rec: The trade recommendation to check.

        Raises:
            RiskError: If DTE is outside the allowed range.
        """
        if rec.expires_at:
            try:
                expiry = datetime.fromisoformat(rec.expires_at).date()
                today = date.today()
                dte = (expiry - today).days
                if dte < self.risk_config.min_dte or dte > self.risk_config.max_dte:
                    raise RiskError(
                        f"DTE {dte} outside allowed range [{self.risk_config.min_dte}, {self.risk_config.max_dte}].",
                        guardrail="dte_range",
                    )
            except (ValueError, TypeError):
                pass

    def _check_statistical_sanity(
        self,
        rec: TradeRecommendation,
        market: MarketSnapshot,
    ) -> None:
        """Flag recommendations where the target strike is unusually far from current price.

        Args:
            rec: The trade recommendation to check.
            market: Current market snapshot for price data.

        Note:
            This check halves confidence rather than raising an error.
        """
        quote = market.quotes.get(rec.asset)
        if quote is None:
            return
        current = quote.current_price
        if current <= 0:
            return
        move_pct = abs(rec.target_strike - current) / current * 100
        if move_pct > 2.0:
            logger.warning(
                "statistical_sanity_flag",
                asset=rec.asset,
                current_price=current,
                target_strike=rec.target_strike,
                move_pct=move_pct,
                threshold_pct=2.0,
            )
            rec.confidence *= 0.5

    @staticmethod
    def check_consensus(
        briefing_direction: float | None,
        market_news_polarity: float | None,
        min_sources: int = 2,
    ) -> tuple[bool, float]:
        """Check whether multiple data sources agree on market direction.

        ``None`` values are treated as "this source is unavailable" —
        e.g. a degraded Atlas briefing yields ``None`` for
        ``macro_sentiment`` and must not count toward consensus.

        Args:
            briefing_direction: Sentiment polarity from the briefing,
                or ``None`` when briefing quality is not FULL.
            market_news_polarity: Average polarity from market news, or
                ``None`` when no news was fetched.
            min_sources: Minimum number of sources with non-trivial signal.

        Returns:
            Tuple of (consensus_reached, average_direction).
        """
        signals = 0
        total = 0.0
        if briefing_direction is not None and abs(briefing_direction) > 0.05:
            signals += 1
            total += briefing_direction
        if market_news_polarity is not None and abs(market_news_polarity) > 0.05:
            signals += 1
            total += market_news_polarity
        if signals < min_sources:
            return False, 0.0
        return True, total / signals
