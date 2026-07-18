from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog

from src.config import Settings
from src.engine.decision import DecisionAggregator
from src.engine.risk import RiskEngine
from src.engine.strategy_a import MomentumStrategy
from src.engine.strategy_b import MeanReversionStrategy
from src.engine.strategy_c import EventDrivenStrategy
from src.ingestion.fetcher import fetch_market_data
from src.ingestion.parser import find_todays_briefing, read_briefing
from src.logging_setup import JSONFileLogger, setup_logging
from src.models.briefing import BriefingData
from src.models.market import MarketSnapshot
from src.models.recommendation import DecisionOutput, StrategyResult
from src.mcp.client import MCPBrokerClient

logger = structlog.get_logger()


class PipelineResult:
    """Container for the full pipeline execution result."""

    def __init__(self) -> None:
        self.correlation_id: str = ""
        self.briefing: Optional[BriefingData] = None
        self.market: Optional[MarketSnapshot] = None
        self.strategy_results: list[StrategyResult] = []
        self.decision: Optional[DecisionOutput] = None
        self.execution_result: Optional[dict] = None
        self.errors: list[str] = []
        self.start_time: datetime = datetime.now(timezone.utc)
        self.end_time: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        """Total wall-clock duration of the pipeline run in seconds."""
        end = self.end_time or datetime.now(timezone.utc)
        return (end - self.start_time).total_seconds()

    def to_summary(self) -> dict:
        """Summarise the pipeline result as a serialisable dictionary.

        Returns:
            Dict with correlation_id, timings, error count, and decision data.
        """
        return {
            "correlation_id": self.correlation_id,
            "duration_seconds": round(self.duration_seconds, 2),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else "",
            "error_count": len(self.errors),
            "errors": self.errors[:5],
            "briefing_found": self.briefing is not None,
            "market_data_symbols": list(self.market.quotes.keys()) if self.market else [],
            "strategies_evaluated": [r.label for r in self.strategy_results],
            "decision": self.decision.model_dump() if self.decision else None,
            "execution": self.execution_result,
        }


class Pipeline:
    """Orchestrates the full trading pipeline: ingest, analyse, decide, execute."""

    def __init__(self, config: Settings, correlation_id: str, file_logger: JSONFileLogger):
        self.config = config
        self.correlation_id = correlation_id
        self.file_logger = file_logger
        self.result = PipelineResult()
        self.result.correlation_id = correlation_id

    async def run(self) -> PipelineResult:
        """Execute the full pipeline: ingest, analyse, decide, and execute.

        Returns:
            A PipelineResult with all phase outputs.
        """
        logger.info("pipeline_start", correlation_id=self.correlation_id)

        briefing = await self._phase_ingest_briefing()
        market = await self._phase_ingest_market()
        strategy_results = await self._phase_analyze(briefing, market)
        decision = self._phase_decide(strategy_results)
        execution = None
        if decision.recommendation is not None:
            execution = await self._phase_execute(decision)
        else:
            logger.info("pipeline_no_trade", rationale=decision.rationale)

        self.result.end_time = datetime.now(timezone.utc)
        self.result.decision = decision
        self.result.execution_result = execution

        summary = self.result.to_summary()
        self.file_logger.write_summary(summary)
        logger.info("pipeline_complete", duration_seconds=self.result.duration_seconds,
                     trade_selected=decision.selected_label is not None)

        return self.result

    async def _phase_ingest_briefing(self) -> Optional[BriefingData]:
        """Phase 1: Find and parse today's morning briefing.

        Returns:
            Parsed BriefingData, or None if no briefing is found.
        """
        try:
            briefing_path = find_todays_briefing(self.config.atlas_briefing.directory)
            if briefing_path is None:
                logger.warning("briefing_not_found", directory=self.config.atlas_briefing.directory)
                return None
            logger.info("briefing_found", path=str(briefing_path))
            briefing = read_briefing(briefing_path)
            self.result.briefing = briefing
            self.file_logger.write_entry({
                "phase": "ingest_briefing",
                "status": "success",
                "path": str(briefing_path),
                "ticker_count": len(briefing.tickers),
                "news_count": len(briefing.news_items),
                "macro_sentiment": briefing.macro_sentiment,
            })
            return briefing
        except Exception as e:
            msg = f"Briefing ingestion failed: {e}"
            logger.error("briefing_ingest_error", error=str(e))
            self.result.errors.append(msg)
            return None

    async def _phase_ingest_market(self) -> MarketSnapshot:
        """Phase 2: Fetch live market data (quotes, news, RSS).

        Returns:
            MarketSnapshot with quotes, news, and RSS data.
        """
        try:
            market = await fetch_market_data(
                symbols=self.config.general.target_assets,
                finnhub_key=self.config.finnhub.api_key,
                brave_key=self.config.brave.api_key,
                brave_query=self.config.brave.news_query,
                rss_urls=[f.url for f in self.config.rss_feeds],
                timeout=self.config.finnhub.request_timeout_sec,
            )
            self.result.market = market
            self.file_logger.write_entry({
                "phase": "ingest_market",
                "status": "success",
                "quotes": {s: q.current_price for s, q in market.quotes.items()},
                "news_count": len(market.news),
                "rss_count": len(market.rss_items),
                "avg_news_polarity": market.avg_sentiment_polarity(),
            })
            return market
        except Exception as e:
            msg = f"Market data ingestion failed: {e}"
            logger.error("market_ingest_error", error=str(e))
            self.result.errors.append(msg)
            return MarketSnapshot()

    async def _phase_analyze(
        self, briefing: Optional[BriefingData], market: MarketSnapshot,
    ) -> list[StrategyResult]:
        """Phase 3: Run all enabled trading strategies.

        Args:
            briefing: Parsed briefing (may be None).
            market: Market snapshot with quotes and news.

        Returns:
            List of StrategyResult from each enabled strategy.
        """
        if briefing is None:
            briefing = BriefingData(briefing_date=datetime.now(timezone.utc).date())

        strategies = []
        if self.config.strategies.momentum.enabled:
            strategies.append(MomentumStrategy(self.config))
        if self.config.strategies.mean_reversion.enabled:
            strategies.append(MeanReversionStrategy(self.config))
        if self.config.strategies.event_driven.enabled:
            strategies.append(EventDrivenStrategy(self.config))

        results = await asyncio.gather(
            *[s.evaluate(briefing, market) for s in strategies],
            return_exceptions=True,
        )

        strategy_results: list[StrategyResult] = []
        for r in results:
            if isinstance(r, StrategyResult):
                strategy_results.append(r)
                self.file_logger.write_entry({
                    "phase": "analyze",
                    "strategy": r.label,
                    "confidence": r.confidence,
                    "has_recommendation": r.recommendation is not None,
                    "duration_ms": r.duration_ms,
                    "debug_trace": r.debug_trace,
                })
            elif isinstance(r, Exception):
                logger.error("strategy_error", error=str(r))
                self.result.errors.append(f"Strategy error: {r}")

        self.result.strategy_results = strategy_results
        return strategy_results

    def _phase_decide(self, strategy_results: list[StrategyResult]) -> DecisionOutput:
        """Phase 4: Aggregate strategy results, apply risk checks, and reach a decision.

        Args:
            strategy_results: Results from all strategies.

        Returns:
            DecisionOutput with the selected recommendation or None.
        """
        aggregator = DecisionAggregator(self.config)
        decision = aggregator.aggregate(strategy_results)
        risk_engine = RiskEngine(self.config)

        if decision.recommendation is not None:
            try:
                risk_engine.validate(
                    decision.recommendation,
                    self.result.market or MarketSnapshot(),
                )
                logger.info("risk_checks_passed", confidence=decision.recommendation.confidence)
            except Exception as e:
                logger.warning("risk_check_failed", error=str(e))
                decision.recommendation = None
                decision.selected_label = None
                decision.rationale = f"Risk check failed: {e}"

                consensus_ok, consensus_dir = RiskEngine.check_consensus(
                    self.result.briefing.macro_sentiment if self.result.briefing else 0.0,
                    self.result.market.avg_sentiment_polarity() if self.result.market else 0.0,
                    min_sources=self.config.risk.min_data_sources_for_direction,
                )
                if not consensus_ok:
                    logger.info("consensus_check_failed",
                                briefing_sent=self.result.briefing.macro_sentiment if self.result.briefing else 0.0,
                                news_polarity=self.result.market.avg_sentiment_polarity() if self.result.market else 0.0)
                    decision.rationale += " Insufficient data source consensus."

        self.file_logger.write_entry({
            "phase": "decide",
            "selected_strategy": decision.selected_label,
            "confidence": decision.recommendation.confidence if decision.recommendation else 0.0,
            "direction": decision.recommendation.direction.value if decision.recommendation else "none",
            "rationale": decision.rationale,
        })
        return decision

    async def _phase_execute(self, decision: DecisionOutput) -> Optional[dict]:
        """Phase 5: Execute the selected trade through the MCP broker client.

        Args:
            decision: The final decision output with a recommendation.

        Returns:
            Dict with execution result, or None if no recommendation.
        """
        rec = decision.recommendation
        if rec is None:
            return None

        rec.correlation_id = self.correlation_id
        mcp = MCPBrokerClient(self.config)
        try:
            result = await mcp.execute(rec)
            self.file_logger.write_entry({
                "phase": "execute",
                "status": result.get("status", "unknown"),
                "occ_symbol": result.get("occ_symbol", ""),
                "bid": result.get("bid"),
                "ask": result.get("ask"),
                "execution_command": result.get("execution_command"),
            })
            return result
        except Exception as e:
            msg = f"MCP execution failed: {e}"
            logger.error("execution_error", error=str(e))
            self.result.errors.append(msg)
            return {"error": msg}
