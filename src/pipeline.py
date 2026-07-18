"""Orchestrates the end-to-end trading pipeline: ingest, analyse, decide, execute."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import structlog

from src.config import Settings
from src.engine.decision import DecisionAggregator
from src.engine.risk import RiskEngine
from src.engine.strategy_a import MomentumStrategy
from src.engine.strategy_b import MeanReversionStrategy
from src.engine.strategy_c import EventDrivenStrategy
from src.ingestion.fetcher import fetch_market_data
from src.ingestion.parser import find_todays_briefing, read_briefing
from src.ingestion.snapshot_loader import SnapshotLoader
from src.ingestion.status import read_briefing_status
from src.llm.client import OpencodeLLMClient
from src.llm.resynthesizer import resynthesize_briefing
from src.logging_setup import JSONFileLogger
from src.mcp.client import MCPBrokerClient
from src.models.briefing import BriefingData, BriefingQuality
from src.models.market import MarketSnapshot
from src.models.recommendation import (
    AssetForecast,
    DecisionOutput,
    Direction,
    DirectionalForecast,
    StrategyResult,
)

logger = structlog.get_logger()


class PipelineResult:
    """Container for the full pipeline execution result."""

    def __init__(self) -> None:
        """Initialize an empty pipeline result container."""
        self.correlation_id: str = ""
        self.briefing: BriefingData | None = None
        self.market: MarketSnapshot | None = None
        self.strategy_results: list[StrategyResult] = []
        self.decision: DecisionOutput | None = None
        self.execution_result: dict | None = None
        self.errors: list[str] = []
        self.start_time: datetime = datetime.now(UTC)
        self.end_time: datetime | None = None

    @property
    def duration_seconds(self) -> float:
        """Total wall-clock duration of the pipeline run in seconds."""
        end = self.end_time or datetime.now(UTC)
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
        """Initialize Pipeline with configuration and logging.

        Args:
            config: Application settings.
            correlation_id: Unique identifier for this pipeline run.
            file_logger: JSON file logger instance.
        """
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
        decision.forecast = self._compute_forecast(strategy_results, decision)

        execution = None
        if decision.recommendation is not None:
            execution = await self._phase_execute(decision)
        else:
            logger.info("pipeline_no_trade", rationale=decision.rationale)

        self.result.end_time = datetime.now(UTC)
        self.result.decision = decision
        self.result.execution_result = execution

        summary = self.result.to_summary()
        self.file_logger.write_summary(summary)
        logger.info(
            "pipeline_complete",
            duration_seconds=self.result.duration_seconds,
            trade_selected=decision.selected_label is not None,
        )

        return self.result

    async def _phase_ingest_briefing(self) -> BriefingData | None:
        """Phase 1: Find and parse today's morning briefing.

        Also reads the upstream ``status.json`` to fold
        ``intelligence_enabled`` into :attr:`BriefingData.briefing_quality`,
        and locally re-synthesises the executive summary via an LLM
        when the briefing comes back degraded (upstream LLM layer
        failed). See ``LESSONS_LEARNED.md`` (2026-07-18 incident).

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

            status = read_briefing_status(self.config.atlas_briefing.directory)
            if status is not None and not status.intelligence_enabled:
                logger.warning(
                    "upstream_intelligence_disabled",
                    timestamp=status.timestamp,
                )
                briefing.briefing_quality = BriefingQuality.DEGRADED

            pre_resynth_quality = briefing.briefing_quality
            resynth_attempted = False
            resynth_served_by: str | None = None
            resynth_fallback_hit = False
            resynth_error = ""

            if briefing.briefing_quality != BriefingQuality.FULL and self.config.llm.enabled:
                resynth_attempted = True
                client = OpencodeLLMClient(self.config.llm)
                briefing = resynthesize_briefing(briefing, client)
                resynth_served_by = client.last_served_by
                resynth_fallback_hit = client.last_fallback_hit
                resynth_error = client.last_error

            self.result.briefing = briefing
            self.file_logger.write_entry(
                {
                    "phase": "ingest_briefing",
                    "status": "success",
                    "path": str(briefing_path),
                    "ticker_count": len(briefing.tickers),
                    "news_count": len(briefing.news_items),
                    "blog_count": len(briefing.blog_items),
                    "briefing_quality": briefing.briefing_quality.value,
                    "upstream_intelligence_enabled": (
                        status.intelligence_enabled if status else None
                    ),
                    "macro_sentiment": briefing.macro_sentiment,
                    "resynth_attempted": resynth_attempted,
                    "resynth_served_by": resynth_served_by,
                    "resynth_fallback_hit": resynth_fallback_hit,
                    "resynth_error": resynth_error,
                    "pre_resynth_quality": pre_resynth_quality.value,
                }
            )
            return briefing
        except Exception as e:
            msg = f"Briefing ingestion failed: {e}"
            logger.error("briefing_ingest_error", error=str(e))
            self.result.errors.append(msg)
            return None

    async def _phase_ingest_market(self) -> MarketSnapshot:
        """Phase 2: Ingest market data from snapshots or live API.

        Tries to load from atlas-morning-briefing snapshots first.
        Falls back to live API calls (Finnhub, Brave, RSS, Reddit, UW)
        when snapshot data is missing or incomplete.

        Returns:
            MarketSnapshot with quotes, news, and RSS data.
        """
        try:
            # --- Try snapshot first ---
            if self.config.atlas_briefing.snapshot_enabled:
                loader = SnapshotLoader(self.config.atlas_briefing.directory)
                if loader.is_complete(self.config.general.target_assets):
                    market = loader.load()
                    if market is not None:
                        logger.info(
                            "market_data_from_snapshots",
                            quotes=list(market.quotes.keys()),
                            news_count=len(market.news),
                        )
                        self.result.market = market
                        self.file_logger.write_entry(
                            {
                                "phase": "ingest_market",
                                "source": "snapshot",
                                "status": "success",
                                "quotes": {s: q.current_price for s, q in market.quotes.items()},
                                "news_count": len(market.news),
                                "rss_count": len(market.rss_items),
                                "avg_news_polarity": market.avg_sentiment_polarity(),
                            }
                        )
                        return market
                elif loader.is_available():
                    logger.info(
                        "snapshot_incomplete",
                        target_assets=self.config.general.target_assets,
                    )

            # --- Fall back to live API ---
            market = await fetch_market_data(
                symbols=self.config.general.target_assets,
                finnhub_key=self.config.finnhub.api_key,
                brave_key=self.config.brave.api_key,
                brave_query=self.config.brave.news_query,
                rss_urls=[f.url for f in self.config.rss_feeds],
                reddit_subreddits=self.config.reddit.subreddits
                if self.config.reddit.enabled
                else None,
                reddit_post_limit=self.config.reddit.post_limit,
                unusual_whales_key=self.config.unusual_whales.api_key
                if self.config.unusual_whales.enabled
                else "",
                timeout=self.config.finnhub.request_timeout_sec,
            )
            self.result.market = market
            self.file_logger.write_entry(
                {
                    "phase": "ingest_market",
                    "source": "live_api",
                    "status": "success",
                    "quotes": {s: q.current_price for s, q in market.quotes.items()},
                    "news_count": len(market.news),
                    "rss_count": len(market.rss_items),
                    "avg_news_polarity": market.avg_sentiment_polarity(),
                }
            )
            return market
        except Exception as e:
            msg = f"Market data ingestion failed: {e}"
            logger.error("market_ingest_error", error=str(e))
            self.result.errors.append(msg)
            return MarketSnapshot()

    async def _phase_analyze(
        self,
        briefing: BriefingData | None,
        market: MarketSnapshot,
    ) -> list[StrategyResult]:
        """Phase 3: Run all enabled trading strategies.

        Args:
            briefing: Parsed briefing (may be None).
            market: Market snapshot with quotes and news.

        Returns:
            List of StrategyResult from each enabled strategy.
        """
        if briefing is None:
            briefing = BriefingData(briefing_date=datetime.now(UTC).date())

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
                self.file_logger.write_entry(
                    {
                        "phase": "analyze",
                        "strategy": r.label,
                        "confidence": r.confidence,
                        "has_recommendation": r.recommendation is not None,
                        "duration_ms": r.duration_ms,
                        "debug_trace": r.debug_trace,
                    }
                )
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

                consensus_ok, _consensus_dir = RiskEngine.check_consensus(
                    self.result.briefing.macro_sentiment if self.result.briefing else 0.0,
                    self.result.market.avg_sentiment_polarity() if self.result.market else 0.0,
                    min_sources=self.config.risk.min_data_sources_for_direction,
                )
                if not consensus_ok:
                    logger.info(
                        "consensus_check_failed",
                        briefing_sent=self.result.briefing.macro_sentiment
                        if self.result.briefing
                        else 0.0,
                        news_polarity=self.result.market.avg_sentiment_polarity()
                        if self.result.market
                        else 0.0,
                    )
                    decision.rationale += " Insufficient data source consensus."

        self.file_logger.write_entry(
            {
                "phase": "decide",
                "selected_strategy": decision.selected_label,
                "confidence": decision.recommendation.confidence
                if decision.recommendation
                else 0.0,
                "direction": decision.recommendation.direction.value
                if decision.recommendation
                else "none",
                "rationale": decision.rationale,
            }
        )
        return decision

    def _compute_forecast(
        self,
        results: list[StrategyResult],
        decision: DecisionOutput,
    ) -> DirectionalForecast:
        """Build a directional forecast table from strategy results.

        Aggregates each strategy's direction and confidence into
        UP / DOWN / SIDEWAYS probabilities per asset.

        Args:
            results: Raw strategy results (pre-risk).
            decision: The final decision (for rationale / selected trade).

        Returns:
            A DirectionalForecast with per-asset confidence buckets.
        """
        assets = self.config.general.target_assets
        forecasts: list[AssetForecast] = []

        for asset in assets:
            up_sum = 0.0
            down_sum = 0.0
            weighted_magnitude = 0.0
            mag_count = 0
            up_sources: list[str] = []
            down_sources: list[str] = []

            for r in results:
                rec = r.recommendation
                if rec is None or rec.asset != asset:
                    continue
                if rec.direction == Direction.CALL:
                    up_sum += r.confidence
                    up_sources.append(r.label)
                else:
                    down_sum += r.confidence
                    down_sources.append(r.label)

                current = (
                    self.result.market.quotes.get(asset).current_price
                    if self.result.market and asset in self.result.market.quotes
                    else None
                )
                if current and current > 0:
                    expected_move = (rec.target_strike - current) / current
                    weighted_magnitude += expected_move * r.confidence
                    mag_count += 1

            total = up_sum + down_sum
            if total > 0:
                up_conf = round(up_sum / total, 4)
                down_conf = round(down_sum / total, 4)
            else:
                up_conf = 0.0
                down_conf = 0.0

            sideways_conf = round(max(0.0, 1.0 - up_conf - down_conf), 4)
            magnitude = round((weighted_magnitude / mag_count) * 100 if mag_count > 0 else 0.0, 2)
            forecasts.append(
                AssetForecast(
                    asset=asset,
                    up_confidence=up_conf,
                    down_confidence=down_conf,
                    sideways_confidence=sideways_conf,
                    expected_move_pct=magnitude,
                    up_sources=up_sources,
                    down_sources=down_sources,
                )
            )

        return DirectionalForecast(forecasts=forecasts)

    async def _phase_execute(self, decision: DecisionOutput) -> dict | None:
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
            self.file_logger.write_entry(
                {
                    "phase": "execute",
                    "status": result.get("status", "unknown"),
                    "occ_symbol": result.get("occ_symbol", ""),
                    "bid": result.get("bid"),
                    "ask": result.get("ask"),
                    "execution_command": result.get("execution_command"),
                }
            )
            return result
        except Exception as e:
            msg = f"MCP execution failed: {e}"
            logger.error("execution_error", error=str(e))
            self.result.errors.append(msg)
            return {"error": msg}
