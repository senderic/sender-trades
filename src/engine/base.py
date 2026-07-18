from __future__ import annotations

from abc import ABC, abstractmethod

from src.config import Settings
from src.models.briefing import BriefingData
from src.models.market import MarketSnapshot
from src.models.recommendation import StrategyResult


class TradingStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, label: str, config: Settings):
        self.label = label
        self.config = config

    @abstractmethod
    async def evaluate(
        self,
        briefing: BriefingData,
        market: MarketSnapshot,
    ) -> StrategyResult:
        """Evaluate the strategy against current briefing and market data.

        Args:
            briefing: Parsed morning briefing data.
            market: Current market snapshot including quotes and news.

        Returns:
            A StrategyResult with an optional trade recommendation.
        """
        ...
