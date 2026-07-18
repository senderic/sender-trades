from __future__ import annotations

import structlog

from src.config import Settings
from src.models.recommendation import DecisionOutput, Direction, StrategyResult

logger = structlog.get_logger()


class DecisionAggregator:
    """Aggregates results from multiple strategies and selects the best trade recommendation."""

    def __init__(self, config: Settings):
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
        if merged:
            logger.info("decision_merge", strategies=[best.label, second.label],
                        confidence=selected.confidence)
        else:
            logger.info("decision_selected", strategy=best.label,
                        label=selected.strategy_label, confidence=selected.confidence)

        return DecisionOutput(
            selected_label=selected.strategy_label,
            recommendation=selected,
            all_results=results,
            rationale=self._build_rationale(selected, best, merged),
        )

    def _merge_recommendations(self, a: StrategyResult, b: StrategyResult) -> StrategyResult.recommendation:
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
        direction = rec_a.direction if rec_a.direction == rec_b.direction else (
            Direction.CALL if a.confidence > b.confidence else rec_b.direction
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

    @staticmethod
    def _build_rationale(
        selected: StrategyResult.recommendation,
        best: StrategyResult,
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
