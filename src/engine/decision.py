"""Strategy result aggregation and final trade decision logic."""

from __future__ import annotations

import structlog

from src.config import Settings
from src.models.recommendation import DecisionOutput, Direction, StrategyResult
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

        valid_sorted = sorted(valid, key=lambda r: r.confidence, reverse=True)
        best = valid_sorted[0]

        best = self._apply_consensus_scoring(best, results, valid_sorted)
        best = self._apply_streak_dampening(best)

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
