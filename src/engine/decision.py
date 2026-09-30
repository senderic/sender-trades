"""Strategy result aggregation and final trade decision logic."""

from __future__ import annotations

import structlog

from src.config import RiskConfig, Settings
from src.models.recommendation import DecisionOutput, Direction, StrategyResult, TradeRecommendation
from src.trade_tracker import compute_direction_stats, compute_strategy_stats, load_trade_outcomes

logger = structlog.get_logger()


class DecisionAggregator:
    """Aggregates results from multiple strategies and selects the best trade recommendation."""

    def __init__(self, config: Settings):
        """Initialize DecisionAggregator with application settings.

        Args:
            config: Application settings.
        """
        self.config = config

    def aggregate(self, results: list[StrategyResult]) -> DecisionOutput:
        """Aggregate strategy results, select the best, and optionally merge top candidates.

        Args:
            results: List of StrategyResult from all enabled strategies.

        Returns:
            DecisionOutput with the selected recommendation and rationale.
        """
        valid = [r for r in results if r.recommendation is not None]

        if not valid:
            return DecisionOutput(
                selected_label=None,
                recommendation=None,
                all_results=results,
                rationale="No strategy produced a valid recommendation above confidence threshold.",
            )

        valid_sorted = sorted(
            valid,
            key=lambda r: (
                r.confidence,
                1 if r.recommendation.asset == self.config.risk.preferred_asset else 0,
            ),
            reverse=True,
        )
        best = valid_sorted[0]

        best = self._apply_consensus_scoring(best, results, valid_sorted)
        best = self._apply_streak_dampening(best)
        best = self._apply_unsupported_cap(best, valid_sorted, results)

        if best.confidence < self.config.strategies.momentum.min_confidence:
            return DecisionOutput(
                selected_label=None,
                recommendation=None,
                all_results=results,
                rationale=f"Highest confidence ({best.confidence:.2f}) is below minimum threshold.",
            )

        merged = None
        if len(valid_sorted) >= 2:
            second = valid_sorted[1]
            confidence_diff = best.confidence - second.confidence
            if confidence_diff < 0.1 and best.recommendation.asset == second.recommendation.asset:
                merged = self._merge_recommendations(best, second)

        selected = merged if merged else best.recommendation

        if self.config.general.require_forecast_alignment:
            conflict = self._forecast_conflict(selected, results)
            if conflict:
                logger.warning(
                    "decision_blocked_direction_mismatch",
                    asset=selected.asset,
                    trade_direction=selected.direction.value,
                    conflict=conflict,
                )
                return DecisionOutput(
                    selected_label=None,
                    recommendation=None,
                    all_results=results,
                    rationale=(
                        f"Blocked {selected.strategy_label} {selected.direction.value} "
                        f"on {selected.asset}: {conflict}"
                    ),
                )

        if merged:
            logger.info(
                "decision_merge",
                strategies=[best.label, second.label],
                confidence=selected.confidence,
            )
        else:
            logger.info(
                "decision_selected",
                strategy=best.label,
                label=selected.strategy_label,
                confidence=selected.confidence,
            )

        self._apply_sizing(selected, results)

        # Carried through to execution: the option premium isn't known until
        # ExecutionEngine fetches the live ask post-open, so premium_gate
        # runs there against this value rather than here.
        selected.predicted_move_pct = self._lookup_predicted_move(selected.asset, results)

        gate = self._min_move_gate(selected, results)
        if gate is not None:
            return DecisionOutput(
                selected_label=None,
                recommendation=None,
                all_results=results,
                rationale=gate,
            )

        return DecisionOutput(
            selected_label=selected.strategy_label,
            recommendation=selected,
            all_results=results,
            rationale=self._build_rationale(selected, best, merged),
        )

    def _merge_recommendations(
        self, a: StrategyResult, b: StrategyResult
    ) -> StrategyResult.recommendation:
        """Merge two strategy results into a single recommendation.

        Args:
            a: First (higher-confidence) strategy result.
            b: Second strategy result.

        Returns:
            The merged TradeRecommendation from result ``a`` with updated fields.
        """
        rec_a = a.recommendation
        rec_b = b.recommendation
        avg_confidence = (a.confidence + b.confidence) / 2
        direction = (
            rec_a.direction
            if rec_a.direction == rec_b.direction
            else (Direction.CALL if a.confidence > b.confidence else rec_b.direction)
        )
        merged_contracts = max(rec_a.contracts, rec_b.contracts)
        merged_rationale = {
            "merged_from": [a.label, b.label],
            "direction_source": direction.value,
            "confidence_a": a.confidence,
            "confidence_b": b.confidence,
            "merged_confidence": avg_confidence,
            "detail_a": rec_a.rationale,
            "detail_b": rec_b.rationale,
        }
        rec_a.confidence = round(avg_confidence, 4)
        rec_a.direction = direction
        rec_a.contracts = merged_contracts
        rec_a.rationale = merged_rationale
        rec_a.strategy_label = f"{a.label}+{b.label}"
        return rec_a

    def _apply_unsupported_cap(
        self,
        best: StrategyResult,
        valid_sorted: list[StrategyResult],
        all_results: list[StrategyResult],
    ) -> StrategyResult:
        """Cap a recommendation that no other signal corroborates.

        A trade resting on one deterministic strategy, with the LLM silent
        and no second strategy agreeing, is the weakest evidence this
        system can act on — yet nothing previously distinguished it from a
        consensus pick, because the aggregator only ever compared
        confidence numbers. On 2026-08-20 and 2026-08-26 the graph failed
        AND the monolithic fallback returned nothing, leaving a lone
        deterministic strategy to trade at 0.75 and 0.80 confidence. Both
        lost.

        Capping to :attr:`GraphConfig.unsupported_confidence_cap` (0.35,
        below the 0.40 gate applied immediately after this call) turns
        those into no-trade days rather than merely quieter ones.

        Both ``recommendation.confidence`` and ``StrategyResult.confidence``
        are set, because the gate downstream reads the latter and the
        execution path reads the former.

        Args:
            best: The leading strategy result (mutated in place).
            valid_sorted: Every result that produced a recommendation.
            all_results: Every result, including LLM predictions that
                carry no recommendation of their own.

        Returns:
            The (possibly capped) ``best``.
        """
        rec = best.recommendation
        if rec is None:
            return best
        if best.label == "llm_trade":
            return best

        # Corroboration means any OTHER strategy recommending the same
        # asset and direction. A second strategy pointing somewhere else is
        # not support, so agreement is checked rather than mere presence.
        corroborated = any(
            other is not best
            and other.recommendation is not None
            and other.recommendation.asset == rec.asset
            and other.recommendation.direction == rec.direction
            for other in valid_sorted
        )
        # An LLM directional prediction counts too, even when the LLM
        # declined to name a best_trade. That is the common case: the graph
        # produces per-asset predictions on most days but only sometimes
        # proposes a trade, and a deterministic pick moving the same way as
        # the LLM's forecast is corroborated in every sense that matters.
        # Only a run where the LLM produced nothing at all — the 08-20 and
        # 08-26 shape — leaves a deterministic strategy genuinely alone.
        if not corroborated:
            wanted = "UP" if rec.direction == Direction.CALL else "DOWN"
            corroborated = any(
                r.label == "llm_trade"
                and (pred := (r.predictions or {}).get(rec.asset)) is not None
                and pred.direction == wanted
                for r in all_results
            )
        if corroborated:
            return best

        # An LLM prediction that explicitly DISAGREES is a stronger and more
        # specific condition than "nothing corroborates this", and
        # `_forecast_conflict` already blocks it downstream with a rationale
        # naming the conflict. Capping here would pre-empt that check and
        # replace an actionable message with a generic "below threshold",
        # so defer — but only while that guard is actually enabled.
        if self.config.general.require_forecast_alignment:
            wanted = "UP" if rec.direction == Direction.CALL else "DOWN"
            disagrees = any(
                r.label == "llm_trade"
                and (pred := (r.predictions or {}).get(rec.asset)) is not None
                and pred.direction != wanted
                for r in all_results
            )
            if disagrees:
                return best

        cap = self.config.graph.unsupported_confidence_cap
        if best.confidence <= cap:
            return best

        original = best.confidence
        rec.confidence = cap
        best.confidence = cap
        logger.warning(
            "unsupported_signal_capped",
            strategy=best.label,
            asset=rec.asset,
            direction=rec.direction.value,
            original_confidence=round(original, 4),
            cap=cap,
        )
        return best

    def _apply_sizing(self, selected: TradeRecommendation, results: list[StrategyResult]) -> None:
        """Scale the selected trade conservatively to 2 contracts when justified.

        Two contracts only when the final (checker/streak-adjusted) confidence
        clears :attr:`RiskConfig.sizing_tier2_min_confidence` AND the signal is
        LLM-backed (``llm_trade``) or corroborated by another strategy agreeing
        on the same asset+direction. A lone deterministic strategy or a low-
        confidence pick stays at 1 contract.
        """
        if selected is None:
            return

        threshold = self.config.risk.sizing_tier2_min_confidence
        if selected.confidence < threshold:
            return

        llm_backed = selected.strategy_label == "llm_trade" or "llm_trade" in (
            selected.strategy_label or ""
        )

        corroborated = False
        if not llm_backed:
            corroborated = any(
                other.recommendation is not None
                and other.recommendation is not selected
                and other.recommendation.asset == selected.asset
                and other.recommendation.direction == selected.direction
                for other in results
            )

        if not (llm_backed or corroborated):
            return

        selected.contracts = max(selected.contracts, 2)
        logger.info(
            "decision_sizing_scaled",
            asset=selected.asset,
            direction=selected.direction.value,
            strategy=selected.strategy_label,
            confidence=round(selected.confidence, 2),
            contracts=selected.contracts,
        )

    @staticmethod
    def _lookup_predicted_move(asset: str, results: list[StrategyResult]) -> float | None:
        """Return the first LLM per-asset predicted_move_pct for ``asset``, if any.

        Shared by :meth:`_min_move_gate` (compares the raw predicted move
        against a floor) and :meth:`aggregate` (stashes it on the selected
        recommendation for :meth:`premium_gate`, which runs later once the
        option premium is known).
        """
        for result in results:
            pred = (result.predictions or {}).get(asset)
            if pred is not None:
                return pred.predicted_move_pct
        return None

    def _min_move_gate(
        self, selected: TradeRecommendation, results: list[StrategyResult]
    ) -> str | None:
        """Block a trade whose expected move is too small to survive theta.

        Reads the LLM per-asset prediction for the selected asset. A 68%-
        accurate direction call on a 0.1-0.2% move still expires worthless,
        so the trade is blocked below :attr:`RiskConfig.min_predicted_move_pct`.
        Returns a rationale string when blocked, else None. When no LLM
        prediction is present (graph down + monolithic silent), the gate is
        skipped so it cannot be the thing that silently kills a fallback trade.

        Note this compares the model's own (typically inflated, see
        :meth:`premium_gate`) predicted move against a flat floor -- it says
        nothing about whether the move, even if realized exactly as
        predicted, would cover the option's premium. That is what
        :meth:`premium_gate` checks, downstream, once the premium is known.
        """
        if selected is None:
            return None
        if selected.strategy_label and selected.strategy_label.startswith("llm"):
            move = self._lookup_predicted_move(selected.asset, results)
            if move is None:
                return None
            threshold = self.config.risk.min_predicted_move_pct
            if abs(move) < threshold:
                return (
                    f"Blocked {selected.strategy_label} {selected.direction.value} "
                    f"on {selected.asset}: predicted move {move:+.2f}% is below "
                    f"the {threshold:.2f}% minimum."
                )
        return None

    @staticmethod
    def premium_gate(
        selected: TradeRecommendation,
        ask_premium: float | None,
        underlying_price: float | None,
        risk_config: RiskConfig,
    ) -> str | None:
        """Block a trade whose realistic expected move can't clear the option's own cost.

        ``_min_move_gate`` compares the LLM's raw predicted move against a
        flat floor, but the 2026-09-10 audit of logs/ (2026-07-29..09-09)
        found predicted moves running roughly 3x hotter than what the
        underlying actually does intraday -- e.g. 2026-09-09 predicted
        SPY -0.7% / QQQ -0.8%, actual -0.22% / -0.01%. A correct DIRECTION
        call on that smaller real move still loses money on a 0DTE option
        once the premium paid exceeds the payoff, which
        ``min_predicted_move_pct`` cannot see because it never looks at
        what the option costs.

        This gate instead:

        1. Shrinks the predicted move by :attr:`RiskConfig.predicted_move_shrink`
           (default 0.4, ~1 / 2.5 -- the inverse of the observed ~3x
           overestimate) to get a realistic expected move.
        2. Computes the option's true expiry breakeven move as a % of the
           underlying: the underlying must first cover the OTM distance
           from spot to strike, THEN the premium paid, before the position
           is above water --

           - PUT:  ``((underlying - strike) + ask) / underlying * 100``
           - CALL: ``((strike - underlying) + ask) / underlying * 100``

           A 2026-09-10 follow-up review found the first version of this
           gate used ``ask / strike * 100`` with the STRIKE standing in for
           the underlying (see "Stand-in note" below) -- which drops the
           OTM-distance term entirely. Every strike this system trades is
           chosen ~0.6% OTM (see ``compute_otm_strike``), so that omission
           understated breakeven by roughly that amount on every trade:
           e.g. 2026-09-09 QQQ PUT 714 (spot ~716.40, ask $1.15) computed
           0.16% instead of the true ~0.50%. For an ITM strike the distance
           term goes negative, correctly REDUCING the required move since
           part of the premium is already covered by intrinsic value.
        3. Inflates that breakeven by :attr:`RiskConfig.breakeven_margin_pct`
           for spread/slippage margin (now small, since the OTM distance
           is explicit rather than folded into a large flat margin -- see
           the field's docstring), and blocks the trade if the shrunk
           expected move doesn't clear it.

        This can only run once the option's live ask is known, which is
        after market open -- pre-market at decision time
        (``DecisionAggregator.aggregate``) there is no option quote to gate
        against (see AGENTS.md "0DTE entry limit never fills" gotcha). It is
        therefore called from ``ExecutionEngine.execute`` right after the
        entry quote comes back, reading ``selected.predicted_move_pct`` (set
        by ``aggregate()``) and a freshly-fetched live underlying quote
        (``AlpacaBrokerClient.get_underlying_quote``).

        Stand-in note: when a live underlying quote is unavailable,
        ``underlying_price`` is ``None`` and this falls back to the old
        ``ask / strike * 100`` approximation (logged via
        ``premium_gate_fallback_formula``) rather than skipping the gate
        outright -- a same-side approximation is better than none, as long
        as callers know it runs less strict than the real thing.

        Args:
            selected: The recommendation being evaluated (not mutated).
            ask_premium: Live option ask price, per contract (not x100).
            underlying_price: Live underlying price, or ``None`` to use the
                ask/strike fallback approximation.
            risk_config: The app's ``RiskConfig`` (``config.risk``).

        Returns:
            A rationale string when blocked, else ``None`` -- including
            when premium, strike, or a predicted move isn't available,
            since an unmeasurable gate must never be the thing that
            silently kills a trade the other gates already let through.
        """
        if selected is None:
            return None
        if not ask_premium or ask_premium <= 0:
            return None
        if selected.predicted_move_pct is None:
            return None

        strike = selected.target_strike
        shrink = risk_config.predicted_move_shrink
        expected_move_pct = abs(selected.predicted_move_pct) * shrink

        if underlying_price and underlying_price > 0:
            if selected.direction == Direction.PUT:
                distance_pct = (underlying_price - strike) / underlying_price * 100
            else:
                distance_pct = (strike - underlying_price) / underlying_price * 100
            breakeven_pct = distance_pct + (ask_premium / underlying_price) * 100
            basis_label = f"underlying ${underlying_price:.2f}, strike ${strike:.2f}"
        else:
            if not strike or strike <= 0:
                return None
            breakeven_pct = (ask_premium / strike) * 100
            basis_label = f"strike ${strike:.2f} (no live underlying quote, using fallback formula)"
            logger.warning(
                "premium_gate_fallback_formula",
                asset=selected.asset,
                direction=selected.direction.value,
                strike=strike,
                ask=ask_premium,
            )

        required_pct = breakeven_pct * (1 + risk_config.breakeven_margin_pct)

        if expected_move_pct < required_pct:
            return (
                f"Blocked {selected.strategy_label} {selected.direction.value} on "
                f"{selected.asset}: shrunk expected move {expected_move_pct:.2f}% "
                f"(predicted {selected.predicted_move_pct:+.2f}% x shrink {shrink:.2f}) "
                f"doesn't clear breakeven {breakeven_pct:.2f}% (ask ${ask_premium:.2f}, "
                f"{basis_label}) + {risk_config.breakeven_margin_pct:.0%} margin = "
                f"{required_pct:.2f}% required."
            )
        return None

    def _apply_consensus_scoring(
        self,
        best: StrategyResult,
        all_results: list[StrategyResult],
        valid_sorted: list[StrategyResult],
    ) -> StrategyResult:
        """Adjust best strategy confidence based on cross-strategy consensus.

        If 3+ strategies agree on the same (asset, direction), boost
        confidence by 0.05. If all 4 strategies produced recommendations
        and the vote is an equal split (2-2), penalise confidence by 0.05.

        Args:
            best: The highest-confidence strategy result (mutated in-place).
            all_results: All strategy results (including those without recs).
            valid_sorted: All results that produced a recommendation.

        Returns:
            The (potentially modified) best StrategyResult.
        """
        votes: dict[tuple[str, str], int] = {}
        for r in valid_sorted:
            rec = r.recommendation
            if rec is None:
                continue
            key = (rec.asset, rec.direction.value)
            votes[key] = votes.get(key, 0) + 1

        if not votes:
            return best

        total_voting = sum(votes.values())
        majority_count = max(votes.values())

        if total_voting >= 2 and majority_count >= 3:
            old = best.recommendation.confidence
            new = min(1.0, old + 0.05)
            best.recommendation.confidence = new
            best.confidence = new
            logger.info(
                "consensus_boost",
                strategies_majority=majority_count,
                total_strategies=total_voting,
                old_confidence=round(old, 4),
                new_confidence=round(new, 4),
            )
        elif total_voting >= 3 and majority_count == total_voting // 2 and total_voting % 2 == 0:
            old = best.recommendation.confidence
            new = max(0.0, old - 0.05)
            best.recommendation.confidence = new
            best.confidence = new
            logger.info(
                "consensus_penalty",
                strategies_per_side=majority_count,
                total_strategies=total_voting,
                old_confidence=round(old, 4),
                new_confidence=round(new, 4),
            )

        return best

    def _apply_streak_dampening(self, best: StrategyResult) -> StrategyResult:
        """Reduce confidence for strategies on a losing streak.

        Loads resolved trade outcomes and applies a confidence penalty
        when the selected strategy (or asset + direction) has been losing
        consecutively. A longer losing streak produces a larger penalty,
        capped at -0.25. Strategies that are winning are left untouched
        (no boost — consensus scoring already handles that).

        Args:
            best: The highest-confidence strategy result (mutated in-place).

        Returns:
            The (potentially penalised) best StrategyResult.
        """
        rec = best.recommendation
        if rec is None:
            return best

        try:
            outcomes = load_trade_outcomes(self.config.logging.json_dir)
        except Exception:
            return best
        if not outcomes:
            return best

        penalties: list[tuple[float, str]] = []

        # Per-strategy streak
        strategy_stats = compute_strategy_stats(outcomes)
        sstat = strategy_stats.get(rec.strategy_label)
        if sstat and sstat["current_streak"] < 0:
            streak_len = abs(sstat["current_streak"])
            if streak_len >= 2:
                penalties.append(
                    (
                        min(0.25, 0.05 * streak_len),
                        f"{rec.strategy_label} on {streak_len}-loss streak",
                    )
                )

        # Per-asset + direction streak
        direction_stats = compute_direction_stats(outcomes)
        dkey = f"{rec.asset}:{rec.direction.value}"
        dstat = direction_stats.get(dkey)
        if dstat and dstat["current_streak"] < 0:
            streak_len = abs(dstat["current_streak"])
            if streak_len >= 2:
                penalties.append(
                    (min(0.25, 0.05 * streak_len), f"{dkey} on {streak_len}-loss streak")
                )

        if not penalties:
            return best

        # Use the MAX penalty, not the sum: strategy and direction streaks
        # usually describe the same underlying losing trades (e.g. every
        # "momentum" trade is also "SPY:CALL"), so summing would double-count.
        penalty = max(p for p, _ in penalties)
        reasons = [r for _, r in penalties]

        old = rec.confidence
        new = max(0.0, round(old - penalty, 4))
        rec.confidence = new
        best.confidence = new
        logger.warning(
            "streak_dampening",
            strategy=rec.strategy_label,
            asset=rec.asset,
            direction=rec.direction.value,
            old_confidence=round(old, 4),
            new_confidence=round(new, 4),
            penalty=round(penalty, 4),
            reasons=reasons,
        )
        return best

    @staticmethod
    def _forecast_conflict(
        selected: StrategyResult.recommendation,
        results: list[StrategyResult],
    ) -> str | None:
        """Return a conflict description if the selected trade direction
        disagrees with the LLM per-asset forecast, else None.

        Maps the option direction to a forecast direction (CALL→UP,
        PUT→DOWN) and compares against the LLM strategy's per-asset
        prediction for the same asset. A ``None`` result means either
        there is no LLM prediction for the asset (no conflict) or the
        directions agree.

        Args:
            selected: The selected TradeRecommendation.
            results: All strategy results (including LLM strategy).

        Returns:
            Human-readable conflict string, or None when aligned.
        """
        llm_result = next((r for r in results if r.predictions is not None), None)
        if llm_result is None or llm_result.predictions is None:
            return None

        pred = llm_result.predictions.get(selected.asset)
        if pred is None:
            return None

        trade_up = selected.direction == Direction.CALL
        pred_up = pred.direction == "UP"
        if trade_up == pred_up:
            return None

        trade_label = "UP" if trade_up else "DOWN"
        return (
            f"trade direction {selected.direction.value} (implying {trade_label}) "
            f"conflicts with LLM forecast {pred.direction} "
            f"(confidence {pred.confidence:.0%})"
        )

    @staticmethod
    def _build_rationale(
        selected: StrategyResult.recommendation,
        _best: StrategyResult,
        merged: StrategyResult.recommendation,
    ) -> str:
        """Build a human-readable rationale string for the final decision.

        Args:
            selected: The selected recommendation (may be merged or original).
            best: The highest-confidence strategy result.
            merged: The merged recommendation if one was created, else None.

        Returns:
            A formatted rationale string.
        """
        if merged:
            return (
                f"Merged strategies {merged.strategy_label} with confidence {merged.confidence:.2f}. "
                f"Asset: {merged.asset}, Direction: {merged.direction.value}, "
                f"Strike: {merged.target_strike}, Contracts: {merged.contracts}."
            )
        return (
            f"Selected strategy {selected.strategy_label} with confidence {selected.confidence:.2f}. "
            f"Asset: {selected.asset}, Direction: {selected.direction.value}, "
            f"Strike: {selected.target_strike}, Contracts: {selected.contracts}."
        )
