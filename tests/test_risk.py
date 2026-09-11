from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.config import Settings
from src.engine.risk import RiskEngine
from src.models.market import DataSource, MarketSnapshot, PremarketQuote, Quote
from src.models.recommendation import Direction, PositionIntent, TradeRecommendation
from src.timezone import today_local

_TS = __import__("datetime").datetime.now()

ET_TZ = ZoneInfo("America/New_York")
_9_30_AM_ET = datetime(2026, 1, 1, 9, 30, tzinfo=ET_TZ)


@pytest.fixture
def risk_engine() -> RiskEngine:
    return RiskEngine(Settings())


@pytest.fixture
def valid_rec() -> TradeRecommendation:
    return TradeRecommendation(
        correlation_id="test-1",
        strategy_label="momentum",
        asset="SPY",
        direction=Direction.CALL,
        confidence=0.75,
        target_strike=746.0,
        contracts=1,
        order_type="market",
        position_intent=PositionIntent.BUY_TO_OPEN,
        rationale={},
        expires_at=today_local().isoformat(),
        must_close_before="15:30",
    )


class TestRiskEngineTimeCheck:
    def test_passes_valid_time(
        self, risk_engine: RiskEngine, valid_rec: TradeRecommendation
    ) -> None:
        valid_rec.target_strike = 100.0
        valid_rec.contracts = 1
        result = risk_engine.validate(valid_rec, MarketSnapshot(), _now=_9_30_AM_ET)
        assert result is valid_rec

    def test_max_position_size_exceeded(
        self, risk_engine: RiskEngine, valid_rec: TradeRecommendation
    ) -> None:
        valid_rec.contracts = 100
        with pytest.raises(Exception):
            risk_engine.validate(valid_rec, MarketSnapshot())

    def test_max_loss_exceeded(
        self, risk_engine: RiskEngine, valid_rec: TradeRecommendation
    ) -> None:
        valid_rec.contracts = 1000
        with pytest.raises(Exception):
            risk_engine.validate(valid_rec, MarketSnapshot())


class TestRiskEngineConsensus:
    def test_consensus_positive(self) -> None:
        ok, avg = RiskEngine.check_consensus(0.5, 0.3, min_sources=2)
        assert ok is True
        assert avg > 0

    def test_consensus_negative(self) -> None:
        ok, avg = RiskEngine.check_consensus(-0.5, -0.3, min_sources=2)
        assert ok is True
        assert avg < 0

    def test_consensus_insufficient_sources(self) -> None:
        ok, avg = RiskEngine.check_consensus(0.02, 0.01, min_sources=2)
        assert ok is False
        assert avg == 0.0


def _market_with_quote(symbol: str, current: float, prev_close: float) -> MarketSnapshot:
    """Build a market where ``current`` is today's LIVE pre-market price
    and ``prev_close`` is the prior session's close -- the correct gap
    anchor (see ``Quote.prior_session_close``). ``quote.current_price``
    plays the role of ``prior_session_close`` here, since that field (not
    ``previous_close``) is what a pre-market-stale Quote actually holds.
    """
    gap_pct = (current - prev_close) / prev_close * 100 if prev_close > 0 else None
    premarket = (
        {
            symbol: PremarketQuote(
                symbol=symbol,
                available=True,
                price=current,
                vwap=current,
                first_price=prev_close,
                cumulative_volume=2000.0,
                gap_pct=gap_pct,
                median_volume=2000.0,
                volume_ratio=1.0,
                reliable=True,
                lookback_days_used=10,
                source="live",
            ),
        }
        if prev_close > 0
        else {}
    )
    return MarketSnapshot(
        quotes={
            symbol: Quote(
                symbol=symbol,
                current_price=prev_close,
                open_price=prev_close,
                high_price=prev_close,
                low_price=prev_close,
                previous_close=prev_close,
                change_pct=0.0,
                volume=0,
                source=DataSource.FINNHUB,
                timestamp=_TS,
            ),
        },
        premarket=premarket,
    )


class TestRiskEnginePreMarketGap:
    def test_gap_opposite_direction_rejects(
        self, risk_engine: RiskEngine, valid_rec: TradeRecommendation
    ) -> None:
        valid_rec.asset = "SPY"
        valid_rec.direction = Direction.PUT
        valid_rec.target_strike = 100.0
        valid_rec.contracts = 1
        market = _market_with_quote("SPY", current=746.0, prev_close=735.0)
        with pytest.raises(Exception) as exc_info:
            risk_engine.validate(valid_rec, market, _now=_9_30_AM_ET)
        assert "gap" in str(exc_info.value)

    def test_gap_same_direction_passes(
        self, risk_engine: RiskEngine, valid_rec: TradeRecommendation
    ) -> None:
        valid_rec.asset = "SPY"
        valid_rec.direction = Direction.CALL
        valid_rec.target_strike = 100.0
        valid_rec.contracts = 1
        market = _market_with_quote("SPY", current=746.0, prev_close=735.0)
        result = risk_engine.validate(valid_rec, market, _now=_9_30_AM_ET)
        assert result is valid_rec

    def test_gap_below_threshold_passes(
        self, risk_engine: RiskEngine, valid_rec: TradeRecommendation
    ) -> None:
        valid_rec.asset = "SPY"
        valid_rec.direction = Direction.PUT
        valid_rec.target_strike = 100.0
        valid_rec.contracts = 1
        market = _market_with_quote("SPY", current=746.0, prev_close=744.0)
        result = risk_engine.validate(valid_rec, market, _now=_9_30_AM_ET)
        assert result is valid_rec

    def test_no_quote_skips_gracefully(
        self, risk_engine: RiskEngine, valid_rec: TradeRecommendation
    ) -> None:
        valid_rec.asset = "QQQ"
        valid_rec.direction = Direction.PUT
        valid_rec.target_strike = 100.0
        valid_rec.contracts = 1
        result = risk_engine.validate(valid_rec, MarketSnapshot(), _now=_9_30_AM_ET)
        assert result is valid_rec

    def test_previous_close_zero_skips(
        self, risk_engine: RiskEngine, valid_rec: TradeRecommendation
    ) -> None:
        valid_rec.asset = "SPY"
        valid_rec.direction = Direction.PUT
        valid_rec.target_strike = 100.0
        valid_rec.contracts = 1
        market = _market_with_quote("SPY", current=746.0, prev_close=0.0)
        result = risk_engine.validate(valid_rec, market, _now=_9_30_AM_ET)
        assert result is valid_rec

    def test_premarket_unavailable_skips_gracefully(
        self, risk_engine: RiskEngine, valid_rec: TradeRecommendation
    ) -> None:
        """No live pre-market quote at all (only the stale prior-session
        Quote) must never fabricate a gap from stale fields -- the check
        no-ops rather than rejecting or fading confidence on a number
        that describes a different day. See MarketSnapshot.mechanics_gap_pct.
        """
        valid_rec.asset = "SPY"
        valid_rec.direction = Direction.PUT
        valid_rec.target_strike = 100.0
        valid_rec.contracts = 1
        market = MarketSnapshot(
            quotes={
                "SPY": Quote(
                    symbol="SPY",
                    current_price=735.0,
                    open_price=735.0,
                    high_price=735.0,
                    low_price=735.0,
                    previous_close=735.0,
                    change_pct=0.0,
                    volume=0,
                    source=DataSource.FINNHUB,
                    timestamp=_TS,
                ),
            },
        )
        result = risk_engine.validate(valid_rec, market, _now=_9_30_AM_ET)
        assert result is valid_rec
        assert result.confidence == valid_rec.confidence


class TestRiskEngineGapFadeRisk:
    """``_check_gap_fade_risk`` halves-ish confidence on a large gap with a
    weak catalyst. Its thresholds now come from ``config.gap_fade`` rather
    than a hardcoded ``1.5 if asset == "SPY" else 2.0``, so these cover
    both the behaviour and the fact that config actually drives it.
    """

    @staticmethod
    def _call_rec(asset: str, current_price: float) -> TradeRecommendation:
        return TradeRecommendation(
            correlation_id="gap-fade",
            strategy_label="llm_trade",
            asset=asset,
            direction=Direction.CALL,
            confidence=0.80,
            # Keep the strike within the statistical-sanity band (2% of
            # spot) so this class isolates the gap-fade check.
            target_strike=round(current_price * 1.005, 2),
            contracts=1,
            order_type="market",
            position_intent=PositionIntent.BUY_TO_OPEN,
            rationale={},
            expires_at=today_local().isoformat(),
            must_close_before="15:30",
        )

    def test_large_gap_weak_catalyst_reduces_confidence(self) -> None:
        engine = RiskEngine(Settings())
        rec = self._call_rec("SPY", 750.0)
        # +2.0% gap, comfortably over SPY's 1.5% threshold, and the
        # snapshot has no news so sentiment magnitude is 0.0.
        market = _market_with_quote("SPY", current=750.0, prev_close=735.3)
        engine.validate(rec, market, _now=_9_30_AM_ET)
        assert rec.confidence < 0.80

    def test_gap_under_threshold_leaves_confidence_alone(self) -> None:
        engine = RiskEngine(Settings())
        rec = self._call_rec("SPY", 742.0)
        # +1.0% gap is below SPY's 1.5% threshold.
        market = _market_with_quote("SPY", current=742.0, prev_close=734.7)
        engine.validate(rec, market, _now=_9_30_AM_ET)
        assert rec.confidence == 0.80

    def test_qqq_tolerates_a_gap_that_would_flag_spy(self) -> None:
        """QQQ's threshold (2.0%) is looser than SPY's (1.5%)."""
        settings = Settings()
        # ~+1.7%: over SPY's threshold, under QQQ's.
        qqq_rec = self._call_rec("QQQ", 750.0)
        RiskEngine(settings).validate(
            qqq_rec, _market_with_quote("QQQ", current=750.0, prev_close=737.5), _now=_9_30_AM_ET
        )
        assert qqq_rec.confidence == 0.80

        spy_rec = self._call_rec("SPY", 750.0)
        RiskEngine(settings).validate(
            spy_rec, _market_with_quote("SPY", current=750.0, prev_close=737.5), _now=_9_30_AM_ET
        )
        assert spy_rec.confidence < 0.80

    def test_config_threshold_actually_drives_the_check(self) -> None:
        """Tightening the configured threshold must flag a gap the default
        lets through — proving the value is read from config rather than
        still hardcoded in the risk engine."""
        market = _market_with_quote("SPY", current=742.0, prev_close=734.7)  # ~+1.0%

        default_rec = self._call_rec("SPY", 742.0)
        RiskEngine(Settings()).validate(default_rec, market, _now=_9_30_AM_ET)
        assert default_rec.confidence == 0.80

        tightened = Settings()
        tightened.gap_fade.thresholds_pct = {"SPY": 0.5, "QQQ": 2.0}
        tight_rec = self._call_rec("SPY", 742.0)
        RiskEngine(tightened).validate(tight_rec, market, _now=_9_30_AM_ET)
        assert tight_rec.confidence < 0.80

    def test_strong_catalyst_suppresses_the_gap_fade_flag(self) -> None:
        """A large gap backed by strong sentiment is not a fade signal."""
        settings = Settings()
        settings.gap_fade.sentiment_magnitude_max = 0.0  # nothing counts as weak
        rec = self._call_rec("SPY", 750.0)
        market = _market_with_quote("SPY", current=750.0, prev_close=735.3)  # ~+2.0%
        RiskEngine(settings).validate(rec, market, _now=_9_30_AM_ET)
        assert rec.confidence == 0.80

    def test_downward_gap_is_not_flagged(self) -> None:
        """The fade pattern is specific to positive gaps."""
        engine = RiskEngine(Settings())
        rec = self._call_rec("SPY", 735.3)
        rec.direction = Direction.PUT
        market = _market_with_quote("SPY", current=735.3, prev_close=750.0)  # ~-2.0%
        engine.validate(rec, market, _now=_9_30_AM_ET)
        assert rec.confidence == 0.80
