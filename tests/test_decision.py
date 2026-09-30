import pytest

from src.config import Settings
from src.engine.decision import DecisionAggregator
from src.models.recommendation import (
    AssetPrediction,
    Direction,
    PositionIntent,
    StrategyResult,
    TradeRecommendation,
)
from src.timezone import today_local


def _make_settings(tmp_path) -> Settings:
    """Build Settings isolated from real logs so streak dampening has no data."""
    settings = Settings()
    settings.logging.json_dir = str(tmp_path)
    return settings


def _make_rec(
    strategy: str, asset: str = "SPY", direction: Direction = Direction.CALL
) -> TradeRecommendation:
    return TradeRecommendation(
        correlation_id="t1",
        strategy_label=strategy,
        asset=asset,
        direction=direction,
        confidence=0.5,
        target_strike=746.0,
        contracts=1,
        order_type="market",
        position_intent=PositionIntent.BUY_TO_OPEN,
        rationale={},
        expires_at=today_local().isoformat(),
        must_close_before="15:30",
    )


def _make_result(label: str, rec: TradeRecommendation, confidence: float) -> StrategyResult:
    return StrategyResult(label=label, recommendation=rec, confidence=confidence, duration_ms=1.0)


class TestDecisionAggregator:
    def test_no_valid_results_returns_empty(self, tmp_path) -> None:
        agg = DecisionAggregator(_make_settings(tmp_path))
        result = agg.aggregate([])
        assert result.recommendation is None
        assert result.selected_label is None

    def test_picks_highest_confidence(self, tmp_path) -> None:
        agg = DecisionAggregator(_make_settings(tmp_path))
        results = [
            StrategyResult(
                label="a", recommendation=_make_rec("a"), confidence=0.3, duration_ms=1.0
            ),
            StrategyResult(
                label="b", recommendation=_make_rec("b"), confidence=0.8, duration_ms=1.0
            ),
        ]
        result = agg.aggregate(results)
        assert result.selected_label == "b"

    def test_below_threshold_returns_none(self, tmp_path) -> None:
        agg = DecisionAggregator(_make_settings(tmp_path))
        results = [
            StrategyResult(
                label="a", recommendation=_make_rec("a"), confidence=0.1, duration_ms=1.0
            ),
        ]
        result = agg.aggregate(results)
        assert result.recommendation is None


class TestConsensusScoring:
    def test_three_way_consensus_boosts_confidence(self, tmp_path) -> None:
        agg = DecisionAggregator(_make_settings(tmp_path))
        qqq_put = _make_rec("event", asset="QQQ", direction=Direction.PUT)
        qqq_put.confidence = 0.65
        results = [
            _make_result("momentum", _make_rec("m", asset="QQQ", direction=Direction.PUT), 0.45),
            _make_result(
                "mean_reversion", _make_rec("mr", asset="QQQ", direction=Direction.PUT), 0.50
            ),
            _make_result("event_driven", qqq_put, 0.65),
            _make_result(
                "llm_trade", _make_rec("llm", asset="QQQ", direction=Direction.CALL), 0.40
            ),
        ]
        decision = agg.aggregate(results)
        assert decision.recommendation is not None
        assert decision.recommendation.asset == "QQQ"
        assert decision.recommendation.direction == Direction.PUT
        assert decision.recommendation.confidence == pytest.approx(0.70)

    def test_four_way_split_penalizes_confidence(self, tmp_path) -> None:
        agg = DecisionAggregator(_make_settings(tmp_path))
        spy_call = _make_rec("event", asset="SPY", direction=Direction.CALL)
        spy_call.confidence = 0.55
        results = [
            _make_result("momentum", _make_rec("m", asset="SPY", direction=Direction.PUT), 0.45),
            _make_result(
                "mean_reversion", _make_rec("mr", asset="SPY", direction=Direction.PUT), 0.50
            ),
            _make_result("event_driven", spy_call, 0.55),
            _make_result(
                "llm_trade", _make_rec("llm", asset="SPY", direction=Direction.CALL), 0.40
            ),
        ]
        decision = agg.aggregate(results)
        assert decision.recommendation is not None
        assert decision.recommendation.confidence == pytest.approx(0.50)

    def test_single_strategy_no_consensus_effect(self, tmp_path) -> None:
        # Uses llm_trade rather than a deterministic strategy: a lone
        # deterministic signal is now capped by the unsupported-signal guard,
        # which would mask the consensus behaviour under test here.
        agg = DecisionAggregator(_make_settings(tmp_path))
        rec = _make_rec("llm", asset="SPY", direction=Direction.CALL)
        rec.confidence = 0.60
        results = [
            _make_result("llm_trade", rec, 0.60),
        ]
        decision = agg.aggregate(results)
        assert decision.recommendation is not None
        assert decision.recommendation.confidence == pytest.approx(0.60)

    def test_different_assets_no_split_penalty(self, tmp_path) -> None:
        agg = DecisionAggregator(_make_settings(tmp_path))
        spy_call = _make_rec("event", asset="SPY", direction=Direction.CALL)
        spy_call.confidence = 0.65
        results = [
            _make_result("momentum", _make_rec("m", asset="SPY", direction=Direction.CALL), 0.45),
            _make_result(
                "mean_reversion", _make_rec("mr", asset="SPY", direction=Direction.CALL), 0.50
            ),
            _make_result("event_driven", spy_call, 0.65),
            _make_result("llm_trade", _make_rec("llm", asset="QQQ", direction=Direction.PUT), 0.40),
        ]
        decision = agg.aggregate(results)
        assert decision.recommendation is not None
        assert decision.recommendation.asset == "SPY"
        assert decision.recommendation.direction == Direction.CALL
        assert decision.recommendation.confidence == pytest.approx(0.70)


class TestForecastAlignment:
    def _llm_prediction(
        self, asset: str = "SPY", direction: str = "UP", confidence: float = 0.6
    ) -> StrategyResult:
        return StrategyResult(
            label="llm_trade",
            recommendation=None,
            predictions={
                asset: AssetPrediction(
                    asset=asset,
                    direction=direction,
                    confidence=confidence,
                    predicted_move_pct=0.3 if direction == "UP" else -0.3,
                    rationale="test",
                    sources=["news-sentiment"],
                )
            },
            confidence=0.0,
            duration_ms=1.0,
        )

    def test_call_blocked_when_llm_forecasts_down(self, tmp_path) -> None:
        agg = DecisionAggregator(_make_settings(tmp_path))
        results = [
            _make_result(
                "event_driven", _make_rec("event", asset="SPY", direction=Direction.CALL), 0.77
            ),
            self._llm_prediction(asset="SPY", direction="DOWN", confidence=0.52),
        ]
        decision = agg.aggregate(results)
        assert decision.recommendation is None
        assert "conflicts with LLM forecast" in decision.rationale

    def test_put_blocked_when_llm_forecasts_up(self, tmp_path) -> None:
        agg = DecisionAggregator(_make_settings(tmp_path))
        results = [
            _make_result(
                "event_driven", _make_rec("event", asset="QQQ", direction=Direction.PUT), 0.70
            ),
            self._llm_prediction(asset="QQQ", direction="UP", confidence=0.60),
        ]
        decision = agg.aggregate(results)
        assert decision.recommendation is None

    def test_call_allowed_when_llm_forecasts_up(self, tmp_path) -> None:
        agg = DecisionAggregator(_make_settings(tmp_path))
        results = [
            _make_result(
                "event_driven", _make_rec("event", asset="SPY", direction=Direction.CALL), 0.77
            ),
            self._llm_prediction(asset="SPY", direction="UP", confidence=0.60),
        ]
        decision = agg.aggregate(results)
        assert decision.recommendation is not None
        assert decision.recommendation.direction == Direction.CALL

    def test_no_llm_prediction_allows_trade(self, tmp_path) -> None:
        agg = DecisionAggregator(_make_settings(tmp_path))
        results = [
            _make_result(
                "event_driven", _make_rec("event", asset="SPY", direction=Direction.CALL), 0.77
            ),
            # Corroborating second strategy: without it the unsupported-signal
            # cap blocks the trade for its own reasons and this test would no
            # longer isolate the forecast-alignment behaviour it is named for.
            _make_result("momentum", _make_rec("mom", asset="SPY", direction=Direction.CALL), 0.60),
        ]
        decision = agg.aggregate(results)
        assert decision.recommendation is not None

    def test_different_asset_prediction_does_not_conflict(self, tmp_path) -> None:
        agg = DecisionAggregator(_make_settings(tmp_path))
        results = [
            _make_result(
                "event_driven", _make_rec("event", asset="SPY", direction=Direction.CALL), 0.77
            ),
            # Corroborating second strategy: without it the unsupported-signal
            # cap blocks the trade for its own reasons and this test would no
            # longer isolate the forecast-alignment behaviour it is named for.
            _make_result("momentum", _make_rec("mom", asset="SPY", direction=Direction.CALL), 0.60),
            self._llm_prediction(asset="QQQ", direction="DOWN", confidence=0.60),
        ]
        decision = agg.aggregate(results)
        assert decision.recommendation is not None

    def test_alignment_can_be_disabled(self, tmp_path) -> None:
        settings = _make_settings(tmp_path)
        settings.general.require_forecast_alignment = False
        agg = DecisionAggregator(settings)
        results = [
            _make_result(
                "event_driven", _make_rec("event", asset="SPY", direction=Direction.CALL), 0.77
            ),
            # Corroborating second strategy: without it the unsupported-signal
            # cap blocks the trade for its own reasons and this test would no
            # longer isolate the forecast-alignment behaviour it is named for.
            _make_result("momentum", _make_rec("mom", asset="SPY", direction=Direction.CALL), 0.60),
            self._llm_prediction(asset="SPY", direction="DOWN", confidence=0.52),
        ]
        decision = agg.aggregate(results)
        assert decision.recommendation is not None


class TestUnsupportedSignalCap:
    """A lone deterministic strategy must not be able to trade on its own.

    Replays the two real incidents: on 2026-08-20 event_driven traded alone
    at 0.75 (-$4) and on 2026-08-26 momentum traded alone at 0.80 (-$36),
    both while the LLM graph and its monolithic fallback produced nothing.
    Capping the forecast alone would not have stopped either, because
    `_compute_forecast` runs AFTER `_phase_decide` and the trading path
    never reads the forecast back.
    """

    @staticmethod
    def _result(label: str, confidence: float, asset: str = "SPY", direction=Direction.CALL):
        rec = _make_rec(label, asset=asset, direction=direction)
        rec.confidence = confidence
        return _make_result(label, rec, confidence)

    def test_solo_deterministic_is_capped_and_blocked(self, tmp_path) -> None:
        config = _make_settings(tmp_path)
        # 2026-08-20 shape: event_driven alone at 0.75, everyone else silent.
        results = [self._result("event_driven", 0.7519)]
        decision = DecisionAggregator(config).aggregate(results)
        assert decision.recommendation is None, "uncorroborated signal must not trade"
        assert results[0].confidence == config.graph.unsupported_confidence_cap

    def test_solo_momentum_is_blocked(self, tmp_path) -> None:
        config = _make_settings(tmp_path)
        # 2026-08-26 shape: momentum alone at 0.80.
        decision = DecisionAggregator(config).aggregate([self._result("momentum", 0.7983)])
        assert decision.recommendation is None

    def test_corroborating_strategy_prevents_the_cap(self, tmp_path) -> None:
        """Two deterministic strategies agreeing is real corroboration."""
        config = Settings()
        config.logging.json_dir = str(tmp_path)
        results = [
            self._result("event_driven", 0.75),
            self._result("momentum", 0.70),
        ]
        decision = DecisionAggregator(config).aggregate(results)
        assert decision.recommendation is not None
        assert decision.recommendation.confidence > config.graph.unsupported_confidence_cap

    def test_disagreeing_second_strategy_is_not_corroboration(self, tmp_path) -> None:
        """A second strategy pointing the OTHER way must not count as support."""
        config = Settings()
        config.logging.json_dir = str(tmp_path)
        results = [
            self._result("event_driven", 0.75, direction=Direction.CALL),
            self._result("momentum", 0.70, direction=Direction.PUT),
        ]
        decision = DecisionAggregator(config).aggregate(results)
        assert decision.recommendation is None

    def test_llm_trade_alone_is_not_capped(self, tmp_path) -> None:
        """The LLM path carries its own checker/veto validation upstream."""
        config = Settings()
        config.logging.json_dir = str(tmp_path)
        decision = DecisionAggregator(config).aggregate([self._result("llm_trade", 0.62)])
        assert decision.recommendation is not None
        assert decision.recommendation.confidence > config.graph.unsupported_confidence_cap

    def test_cap_sits_below_both_strategy_gates(self) -> None:
        """The cap only blocks if it is below the gates it must clear."""
        config = Settings()
        assert config.graph.unsupported_confidence_cap < config.strategies.momentum.min_confidence
        assert config.graph.unsupported_confidence_cap < config.llm.trade_signal_min_confidence


class TestSizing:
    """Conservative 1 -> 2 contract scaling.

    Two contracts only when confidence clears ``sizing_tier2_min_confidence``
    AND the pick is LLM-backed or corroborated by a second strategy.
    """

    @staticmethod
    def _result(label: str, confidence: float, asset: str = "SPY", direction=Direction.CALL):
        rec = _make_rec(label, asset=asset, direction=direction)
        rec.confidence = confidence
        return _make_result(label, rec, confidence)

    def test_llm_trade_scales_to_2_at_high_confidence(self, tmp_path) -> None:
        config = _make_settings(tmp_path)
        decision = DecisionAggregator(config).aggregate([self._result("llm_trade", 0.72)])
        assert decision.recommendation is not None
        assert decision.recommendation.contracts == 2

    def test_llm_trade_stays_1_below_threshold(self, tmp_path) -> None:
        config = _make_settings(tmp_path)
        decision = DecisionAggregator(config).aggregate([self._result("llm_trade", 0.55)])
        assert decision.recommendation is not None
        assert decision.recommendation.contracts == 1

    def test_corroborated_deterministic_scales_to_2(self, tmp_path) -> None:
        config = _make_settings(tmp_path)
        results = [
            self._result("event_driven", 0.75),
            self._result("momentum", 0.70),
        ]
        decision = DecisionAggregator(config).aggregate(results)
        assert decision.recommendation is not None
        assert decision.recommendation.contracts == 2

    def test_solo_deterministic_never_scales(self, tmp_path) -> None:
        config = _make_settings(tmp_path)
        results = [
            self._result("event_driven", 0.70),
        ]
        # Solo deterministic at 0.70 is uncorroborated -> blocked entirely.
        decision = DecisionAggregator(config).aggregate(results)
        assert decision.recommendation is None

    def test_merged_pick_scales_when_corroborated(self, tmp_path) -> None:
        config = _make_settings(tmp_path)
        results = [
            self._result("momentum", 0.66, asset="QQQ", direction=Direction.CALL),
            self._result("event_driven", 0.64, asset="QQQ", direction=Direction.CALL),
        ]
        decision = DecisionAggregator(config).aggregate(results)
        assert decision.recommendation is not None
        assert decision.recommendation.contracts == 2


class TestPredictedMoveCarry:
    """``aggregate()`` stashes the LLM's per-asset predicted move onto the
    selected recommendation (``TradeRecommendation.predicted_move_pct``) so
    ``premium_gate`` can use it later, once the option premium is known
    (the premium isn't available until ``ExecutionEngine.execute`` runs
    post-open -- see AGENTS.md De-risking, 2026-09-10 review).
    """

    @staticmethod
    def _llm_prediction(
        asset: str = "QQQ", direction: str = "DOWN", move: float = -0.8, confidence: float = 0.65
    ) -> StrategyResult:
        return StrategyResult(
            label="llm_trade",
            recommendation=None,
            predictions={
                asset: AssetPrediction(
                    asset=asset,
                    direction=direction,
                    confidence=confidence,
                    predicted_move_pct=move,
                    rationale="test",
                    sources=["test"],
                )
            },
            confidence=0.0,
            duration_ms=1.0,
        )

    def test_predicted_move_carried_onto_selected(self, tmp_path) -> None:
        config = _make_settings(tmp_path)
        rec = _make_rec("llm_trade", asset="QQQ", direction=Direction.PUT)
        results = [
            _make_result("llm_trade", rec, 0.65),
            self._llm_prediction(asset="QQQ", direction="DOWN", move=-0.8),
        ]
        decision = DecisionAggregator(config).aggregate(results)
        assert decision.recommendation is not None
        assert decision.recommendation.predicted_move_pct == -0.8

    def test_predicted_move_none_when_no_llm_prediction(self, tmp_path) -> None:
        config = _make_settings(tmp_path)
        # Two corroborating deterministic strategies (same asset+direction)
        # so the pick survives the unsupported-signal cap, with no LLM
        # prediction present at all.
        results = [
            _make_result("event_driven", _make_rec("event_driven"), 0.70),
            _make_result("momentum", _make_rec("momentum"), 0.68),
        ]
        decision = DecisionAggregator(config).aggregate(results)
        assert decision.recommendation is not None
        assert decision.recommendation.predicted_move_pct is None


class TestPremiumGate:
    """``DecisionAggregator.premium_gate`` -- the breakeven-aware entry gate.

    This only runs once the option's live ask (and, for the live formula,
    a live underlying quote) is known, which is after market open
    (``ExecutionEngine.execute``), not at decision time -- so it's tested
    directly as a static function rather than through ``aggregate()``.

    Breakeven is OTM-distance-aware (2026-09-10 follow-up review): the
    underlying must cover the distance from spot to strike, THEN the
    premium, before the position is above water --
    PUT: ``((underlying - strike) + ask) / underlying * 100``;
    CALL: ``((strike - underlying) + ask) / underlying * 100``. The first
    version of this gate used ``ask / strike * 100`` (strike standing in
    for a live underlying price it didn't have yet), which silently
    dropped the OTM-distance term -- since every strike here is chosen
    ~0.6% OTM (``compute_otm_strike``), that understated breakeven on
    every trade. That old formula is now only a fallback for when a live
    underlying quote can't be fetched (see ``TestFallbackFormula`` below).
    """

    @staticmethod
    def _rec(
        predicted_move_pct: float | None,
        direction: Direction = Direction.PUT,
        strike: float = 714.0,
    ) -> TradeRecommendation:
        rec = _make_rec("llm_trade", asset="QQQ", direction=direction)
        rec.target_strike = strike
        rec.predicted_move_pct = predicted_move_pct
        return rec

    def test_blocks_when_shrunk_move_below_breakeven(self, tmp_path) -> None:
        # PUT strike 714, underlying 716.40, ask 1.15: distance
        # (716.40-714)/716.40 = 0.335%, + ask/underlying 0.161% = breakeven
        # 0.495%, required (x1.03) 0.510%. Predicted -0.8% shrunk (x0.4) =
        # 0.32%, below required.
        config = _make_settings(tmp_path)
        rec = self._rec(-0.8)
        reason = DecisionAggregator.premium_gate(rec, 1.15, 716.40, config.risk)
        assert reason is not None
        assert "breakeven" in reason

    def test_allows_when_shrunk_move_clears_breakeven(self, tmp_path) -> None:
        config = _make_settings(tmp_path)
        rec = self._rec(-3.5)  # shrunk (x0.4) = 1.40%, well clear of the ~0.51% breakeven
        assert DecisionAggregator.premium_gate(rec, 1.15, 716.40, config.risk) is None

    def test_skips_when_no_predicted_move(self, tmp_path) -> None:
        config = _make_settings(tmp_path)
        rec = self._rec(None)
        assert DecisionAggregator.premium_gate(rec, 1.15, 716.40, config.risk) is None

    def test_skips_when_ask_missing_or_zero(self, tmp_path) -> None:
        config = _make_settings(tmp_path)
        rec = self._rec(-3.5)
        assert DecisionAggregator.premium_gate(rec, None, 716.40, config.risk) is None
        assert DecisionAggregator.premium_gate(rec, 0.0, 716.40, config.risk) is None

    def test_itm_strike_reduces_required_move(self, tmp_path) -> None:
        """An ITM strike gives a NEGATIVE OTM distance, correctly reducing
        (not increasing) the required move since intrinsic value already
        covers part of the premium."""
        config = _make_settings(tmp_path)
        # PUT strike 720 (ITM: underlying 716.40 < strike), ask 1.15:
        # distance (716.40-720)/716.40 = -0.503%, + ask/underlying 0.161%
        # = breakeven -0.343% (already profitable at the current spot) --
        # any non-zero predicted move clears a negative requirement.
        rec = self._rec(-0.2, strike=720.0)  # shrunk = 0.08%, tiny
        assert DecisionAggregator.premium_gate(rec, 1.15, 716.40, config.risk) is None

    def test_call_direction_uses_call_side_distance(self, tmp_path) -> None:
        # CALL strike 723, underlying 720.91, ask 1.13: distance
        # (723-720.91)/720.91 = 0.290%, + ask/underlying 0.157% =
        # breakeven 0.447%, required (x1.03) 0.460%. Predicted +0.4%
        # shrunk (x0.4) = 0.16%, below required.
        config = _make_settings(tmp_path)
        rec = self._rec(0.4, direction=Direction.CALL, strike=723.0)
        reason = DecisionAggregator.premium_gate(rec, 1.13, 720.91, config.risk)
        assert reason is not None

    def test_would_have_blocked_2026_09_09_trade(self, tmp_path) -> None:
        """Real 2026-09-09 QQQ PUT 714: predicted -0.8%, ask $1.15,
        underlying ~$716.40 (open). Under the ORIGINAL ask/strike gate
        this was not blocked (breakeven understated at 0.16%); with the
        OTM-distance-aware formula breakeven is ~0.50% and the shrunk
        expected move (0.32%) doesn't clear it -- this -$65.50 loser would
        now correctly be blocked too."""
        config = _make_settings(tmp_path)
        rec = self._rec(-0.8, strike=714.0)
        reason = DecisionAggregator.premium_gate(rec, 1.15, 716.40, config.risk)
        assert reason is not None

    def test_would_have_blocked_2026_09_08_trade(self, tmp_path) -> None:
        """Real 2026-09-08 QQQ CALL 723: predicted +0.4%, ask $1.13,
        underlying ~$720.91 (open) -- shrunk expected move 0.16% does not
        clear the ~0.46% breakeven requirement, so the gate would have
        blocked this -$117 loser (as it did under the original formula
        too, just at a smaller, understated threshold)."""
        config = _make_settings(tmp_path)
        rec = self._rec(0.4, direction=Direction.CALL, strike=723.0)
        reason = DecisionAggregator.premium_gate(rec, 1.13, 720.91, config.risk)
        assert reason is not None


class TestPremiumGateFallbackFormula:
    """When a live underlying quote isn't available, ``premium_gate`` falls
    back to the old ``ask / strike * 100`` approximation rather than
    skipping the gate outright -- logged via ``premium_gate_fallback_formula``
    so the (understated, less strict) approximation is visible in logs.
    """

    @staticmethod
    def _rec(predicted_move_pct: float | None, strike: float = 714.0) -> TradeRecommendation:
        rec = _make_rec("llm_trade", asset="QQQ", direction=Direction.PUT)
        rec.target_strike = strike
        rec.predicted_move_pct = predicted_move_pct
        return rec

    def test_fallback_used_when_underlying_none(self, tmp_path) -> None:
        # ask/strike = 1.15/714*100 = 0.161%, required (x1.03) = 0.166%.
        # Predicted -0.8% shrunk (x0.4) = 0.32%, clears the understated
        # fallback requirement even though the real (distance-aware)
        # breakeven would have blocked it (see TestPremiumGate).
        config = _make_settings(tmp_path)
        rec = self._rec(-0.8)
        assert DecisionAggregator.premium_gate(rec, 1.15, None, config.risk) is None

    def test_fallback_used_when_underlying_zero(self, tmp_path) -> None:
        config = _make_settings(tmp_path)
        rec = self._rec(-0.8)
        assert DecisionAggregator.premium_gate(rec, 1.15, 0.0, config.risk) is None

    def test_fallback_still_blocks_clearly_too_small_move(self, tmp_path) -> None:
        config = _make_settings(tmp_path)
        rec = self._rec(-0.2)  # shrunk = 0.08%, below the 0.166% fallback requirement
        reason = DecisionAggregator.premium_gate(rec, 1.15, None, config.risk)
        assert reason is not None

    def test_skips_when_strike_also_missing(self, tmp_path) -> None:
        config = _make_settings(tmp_path)
        rec = self._rec(-0.8, strike=0.0)
        assert DecisionAggregator.premium_gate(rec, 1.15, None, config.risk) is None
