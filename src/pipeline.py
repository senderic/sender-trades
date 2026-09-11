"""Orchestrates the end-to-end trading pipeline: ingest, analyse, decide, execute."""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime
from typing import Literal

import structlog

from src.config import Settings
from src.engine.base import TradingStrategy
from src.engine.decision import DecisionAggregator
from src.engine.risk import RiskEngine
from src.engine.strategy_a import MomentumStrategy
from src.engine.strategy_b import MeanReversionStrategy
from src.engine.strategy_c import EventDrivenStrategy
from src.execution.client import AlpacaBrokerClient
from src.execution.engine import ExecutionEngine
from src.execution.models import ExecutionConfig
from src.ingestion.candle_providers import build_candle_chain
from src.ingestion.fetcher import fetch_market_data
from src.ingestion.parser import find_todays_briefing, read_briefing
from src.ingestion.premarket import fetch_premarket_quote
from src.ingestion.snapshot_loader import SnapshotLoader
from src.ingestion.status import read_briefing_status
from src.llm.client import OpencodeLLMClient, validate_llm_config
from src.llm.resynthesizer import resynthesize_briefing
from src.llm.trade_signal import LLMTradeStrategy
from src.logging_setup import JSONFileLogger
from src.models.briefing import BriefingData, BriefingQuality
from src.models.market import MarketSnapshot, PremarketQuote
from src.models.recommendation import (
    AssetForecast,
    DecisionOutput,
    Direction,
    DirectionalForecast,
    PredictionOutcome,
    StrategyResult,
)
from src.prediction_tracker import (
    append_outcomes,
    check_outcome,
    find_previous_business_day,
    read_previous_forecasts,
)
from src.timezone import now_local, today_local

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
        self.start_time: datetime = now_local()
        self.end_time: datetime | None = None
        self.yesterday_outcomes: list[PredictionOutcome] = []
        self.model_usage_html: str = ""
        self.model_usage_text: str = ""

    @property
    def duration_seconds(self) -> float:
        """Total wall-clock duration of the pipeline run in seconds."""
        end = self.end_time or now_local()
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
            "model_usage": {
                "calls": 0,
                "failures": 0,
            }
            if not self.model_usage_html
            else {},
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

        # Fail fast on model ids that don't exist at all (e.g. the
        # 2026-09-05 Nemotron incident, where every call to a nonexistent
        # id failed in ~3s with an empty error and silently fell back
        # every run). Cheap: `opencode models` is cached to disk (see
        # `get_known_model_ids`), so this costs a live subprocess call at
        # most once per cache window, not once per pipeline run. Unknown
        # ids are dropped and logged loudly; never raises, never empties
        # the chain. Gated on `llm.preflight.enabled` -- the same flag
        # that opts into the availability probe this pairs with -- so an
        # operator (or a test using bare defaults) that doesn't want the
        # extra subprocess call at startup can opt out of both together.
        llm_config = (
            validate_llm_config(self.config.llm)
            if self.config.llm.preflight.enabled
            else self.config.llm
        )

        # The graph path (``invoke_agent``) is capped below the raw call
        # budget so a graph failure caused by exhausting the budget does
        # not also starve the monolithic fallback it triggers.
        llm_client = (
            OpencodeLLMClient(
                llm_config,
                reserved_calls_for_fallback=self.config.graph.reserved_calls_for_fallback,
            )
            if llm_config.enabled
            else None
        )

        briefing = await self._phase_ingest_briefing(llm_client=llm_client)
        market = await self._phase_ingest_market()
        strategy_results = await self._phase_analyze(briefing, market, llm_client=llm_client)

        decision = self._phase_decide(strategy_results)
        decision.forecast = self._compute_forecast(strategy_results, decision)

        if llm_client:
            self.result.model_usage_html = llm_client.get_usage_summary_html()
            self.result.model_usage_text = llm_client.get_usage_summary_text()

        execution = None
        if decision.recommendation is not None and self.config.general.execute:
            execution = await self._phase_execute(decision)
        elif decision.recommendation is not None:
            logger.info(
                "pipeline_execution_disabled",
                asset=decision.recommendation.asset,
                direction=decision.recommendation.direction.value,
            )
        else:
            logger.info("pipeline_no_trade", rationale=decision.rationale)

        self.result.decision = decision
        self.result.execution_result = execution

        await self._check_yesterday_prediction()

        self.result.end_time = now_local()

        summary = self.result.to_summary()
        self.file_logger.write_summary(summary)
        logger.info(
            "pipeline_complete",
            duration_seconds=self.result.duration_seconds,
            trade_selected=decision.selected_label is not None,
        )

        return self.result

    async def _phase_ingest_briefing(
        self, llm_client: OpencodeLLMClient | None = None
    ) -> BriefingData | None:
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
            briefing_path = find_todays_briefing(
                self.config.atlas_briefing.directory,
                briefings_subdir=self.config.atlas_briefing.briefings_subdir,
            )
            if briefing_path is None:
                logger.warning(
                    "briefing_not_found",
                    directory=self.config.atlas_briefing.directory,
                    briefings_subdir=self.config.atlas_briefing.briefings_subdir,
                )
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
            resynth_paid_used = False
            resynth_error = ""

            if briefing.briefing_quality != BriefingQuality.FULL and llm_client is not None:
                resynth_attempted = True
                briefing = resynthesize_briefing(briefing, llm_client)
                resynth_served_by = llm_client.last_served_by
                resynth_fallback_hit = llm_client.last_fallback_hit
                resynth_paid_used = llm_client.paid_used
                resynth_error = llm_client.last_error

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
                    "resynth_paid_used": resynth_paid_used,
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
                            note="quotes are PRIOR SESSION data, see Quote.prior_session_close",
                        )
                        self.result.market = market
                        await self._enrich_premarket(market)
                        self.file_logger.write_entry(
                            {
                                "phase": "ingest_market",
                                "source": "snapshot",
                                "status": "success",
                                "quotes": {s: q.current_price for s, q in market.quotes.items()},
                                "quotes_are_prior_session": True,
                                "news_count": len(market.news),
                                "rss_count": len(market.rss_items),
                                "avg_news_polarity": market.avg_sentiment_polarity(),
                                "premarket": {
                                    s: {
                                        "available": p.available,
                                        "price": p.price,
                                        "price_source": p.price_source,
                                        "bar_age_min": p.bar_age_min,
                                        "gap_pct": p.gap_pct,
                                        "volume_ratio": p.volume_ratio,
                                        "reliable": p.reliable,
                                    }
                                    for s, p in market.premarket.items()
                                },
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
            await self._enrich_premarket(market)
            self.file_logger.write_entry(
                {
                    "phase": "ingest_market",
                    "source": "live_api",
                    "status": "success",
                    "quotes": {s: q.current_price for s, q in market.quotes.items()},
                    "quotes_are_prior_session": True,
                    "news_count": len(market.news),
                    "rss_count": len(market.rss_items),
                    "avg_news_polarity": market.avg_sentiment_polarity(),
                    "premarket": {
                        s: {
                            "available": p.available,
                            "price": p.price,
                            "price_source": p.price_source,
                            "bar_age_min": p.bar_age_min,
                            "gap_pct": p.gap_pct,
                            "volume_ratio": p.volume_ratio,
                            "reliable": p.reliable,
                        }
                        for s, p in market.premarket.items()
                    },
                }
            )
            return market
        except Exception as e:
            msg = f"Market data ingestion failed: {e}"
            logger.error("market_ingest_error", error=str(e))
            self.result.errors.append(msg)
            return MarketSnapshot()

    async def _enrich_premarket(self, market: MarketSnapshot) -> None:
        """Attach live pre-market price/volume/reliability per target asset.

        Runs regardless of whether ``market`` came from the snapshot or
        the live-API fallback -- both are equally stale pre-market (see
        ``Quote.prior_session_close``) and need this to know what is
        actually happening TODAY. No-ops (leaving ``market.premarket``
        empty) when disabled or when Alpaca API keys are not configured
        -- downstream consumers already treat a missing/unavailable
        entry as "pre-market data unavailable" and must say so
        explicitly rather than silently trusting the stale quote.
        """
        if not self.config.premarket.enabled:
            return

        api_key = os.environ.get("APCA_API_KEY_ID", "")
        api_secret = os.environ.get("APCA_API_SECRET_KEY", "")
        if not api_key or not api_secret:
            logger.warning("premarket_enrich_skipped", reason="no_api_keys")
            for asset in self.config.general.target_assets:
                market.premarket[asset] = PremarketQuote(symbol=asset, available=False)
            return

        paper = self.config.general.env_mode == "PAPER_ALPACA"
        client = AlpacaBrokerClient(api_key, api_secret, paper=paper, config=self.config.execution)
        today = today_local()
        start = time.monotonic()

        async def _fetch_one(asset: str) -> PremarketQuote:
            quote = market.quotes.get(asset)
            prior_close = quote.prior_session_close if quote is not None else 0.0
            try:
                return await fetch_premarket_quote(
                    client, asset, today, prior_close, self.config.premarket
                )
            except Exception as e:
                logger.warning("premarket_enrich_error", asset=asset, error=str(e))
                return PremarketQuote(symbol=asset, available=False)

        # Per-asset fetches are independent (separate symbols) -- run
        # concurrently so total latency is ~max(asset) rather than
        # sum(asset), on top of the single-ranged-request lookback fetch
        # (see fetch_premarket_range) that already cut per-asset calls
        # from 11 to ~3 (today's bars + one live quote + one ranged
        # lookback request).
        assets = self.config.general.target_assets
        results = await asyncio.gather(*(_fetch_one(asset) for asset in assets))

        for asset, pm in zip(assets, results, strict=True):
            market.premarket[asset] = pm
            if not pm.available:
                logger.warning("premarket_unavailable", asset=asset, price_source=pm.price_source)
            elif not pm.reliable:
                logger.info(
                    "premarket_thin_or_stale",
                    asset=asset,
                    volume_ratio=pm.volume_ratio,
                    bar_fresh=pm.bar_fresh,
                    bar_age_min=pm.bar_age_min,
                    cumulative_volume=pm.cumulative_volume,
                )

        elapsed_sec = round(time.monotonic() - start, 2)
        logger.info("premarket_enrichment_complete", elapsed_sec=elapsed_sec, assets=assets)

    async def _phase_analyze(
        self,
        briefing: BriefingData | None,
        market: MarketSnapshot,
        llm_client: OpencodeLLMClient | None = None,
    ) -> list[StrategyResult]:
        """Phase 3: Run all enabled trading strategies.

        Deterministic strategies (Momentum, MeanReversion, EventDriven)
        run FIRST — they are pure local computation, so serializing their
        results into the checker prompt costs almost nothing. Their
        serialized outputs are injected into :class:`LLMTradeStrategy` so
        the in-graph checker node validates the LLM against REAL
        deterministic signals instead of an empty list. This also means
        the checker only runs once per pipeline run (inside the graph),
        rather than once inside the graph against nothing and a second
        time afterwards against the real results.

        Args:
            briefing: Parsed briefing (may be None).
            market: Market snapshot with quotes and news.
            llm_client: Shared opencode client for the LLM strategy.

        Returns:
            List of StrategyResult from each enabled strategy,
            deterministic strategies first.
        """
        if briefing is None:
            briefing = BriefingData(briefing_date=today_local())

        det_strategies: list[TradingStrategy] = []
        if self.config.strategies.momentum.enabled:
            det_strategies.append(MomentumStrategy(self.config))
        if self.config.strategies.mean_reversion.enabled:
            det_strategies.append(MeanReversionStrategy(self.config))
        if self.config.strategies.event_driven.enabled:
            det_strategies.append(EventDrivenStrategy(self.config))

        det_results = await self._run_strategies(det_strategies, briefing, market)

        deterministic_serialized = [
            {
                "label": r.label,
                "recommendation": r.recommendation.model_dump() if r.recommendation else None,
                "confidence": r.confidence,
                "debug_trace": r.debug_trace,
            }
            for r in det_results
        ]

        llm_strategies: list[TradingStrategy] = []
        if self.config.llm.enabled and self.config.llm.trade_signal_enabled:
            llm_strategies.append(
                LLMTradeStrategy(
                    self.config,
                    client=llm_client,
                    deterministic_results=deterministic_serialized,
                )
            )

        llm_results = await self._run_strategies(llm_strategies, briefing, market)

        strategy_results = det_results + llm_results
        for r in strategy_results:
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

        self.result.strategy_results = strategy_results
        return strategy_results

    async def _run_strategies(
        self,
        strategies: list[TradingStrategy],
        briefing: BriefingData,
        market: MarketSnapshot,
    ) -> list[StrategyResult]:
        """Evaluate a batch of strategies concurrently, tolerating individual failures.

        Args:
            strategies: Strategy instances to evaluate.
            briefing: Parsed briefing data.
            market: Market snapshot.

        Returns:
            A StrategyResult for each strategy that did not raise.
            Failures are logged and recorded in ``self.result.errors``
            rather than propagating.
        """
        if not strategies:
            return []

        results = await asyncio.gather(
            *[s.evaluate(briefing, market) for s in strategies],
            return_exceptions=True,
        )

        out: list[StrategyResult] = []
        for r in results:
            if isinstance(r, StrategyResult):
                out.append(r)
            elif isinstance(r, Exception):
                logger.error("strategy_error", error=str(r))
                self.result.errors.append(f"Strategy error: {r}")
        return out

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

        Uses the LLM strategy's per-asset predictions (when available)
        as the primary forecast data, falling back to the legacy
        CALL/PUT aggregation for deterministic strategies.

        In the fallback path, a per-asset forecast is "unsupported" when
        it draws on exactly one contributing strategy and that strategy
        is not ``llm_trade`` -- i.e. a single deterministic signal with no
        corroboration, either from a second deterministic strategy or
        from the LLM. Such forecasts have their confidence clamped to
        ``config.graph.unsupported_confidence_cap``. Two or more
        deterministic strategies that independently agree are NOT
        unsupported and are left untouched, since agreement across
        independent signals is itself a form of corroboration.

        Args:
            results: Raw strategy results (pre-risk).
            decision: The final decision (for rationale / selected trade).

        Returns:
            A DirectionalForecast with per-asset predictions.
        """
        assets = self.config.general.target_assets
        llm_result = next((r for r in results if r.predictions is not None), None)
        market_vibe = ""

        if llm_result is not None:
            # Primary path: use LLM predictions directly.
            trace = llm_result.debug_trace
            market_vibe = trace.get("market_vibe", "")
            forecasts: list[AssetForecast] = []
            for asset in assets:
                pred = llm_result.predictions.get(asset)
                if pred is not None:
                    # MECHANICS: the forecast target strike anchors to the
                    # live pre-market price, never the stale prior-session
                    # open (see MarketSnapshot.mechanics_price).
                    spot = self.result.market.mechanics_price(asset) if self.result.market else None
                    target_strike: float | None = None
                    if spot and spot > 0 and abs(pred.predicted_move_pct) >= 0.1:
                        target_strike = round(spot * (1 + pred.predicted_move_pct / 100), 2)
                    llm_sources = [f"llm:{s}" for s in pred.sources]
                    forecasts.append(
                        AssetForecast(
                            asset=pred.asset,
                            direction=pred.direction,
                            confidence=pred.confidence,
                            predicted_move_pct=pred.predicted_move_pct,
                            target_strike=target_strike,
                            rationale=pred.rationale,
                            sources=llm_sources,
                        )
                    )
                else:
                    forecasts.append(AssetForecast(asset=asset))  # type: ignore
            return DirectionalForecast(forecasts=forecasts, market_vibe=market_vibe)

        # Fallback: legacy aggregation from deterministic strategies.
        forecasts = []
        confidence_cap = self.config.graph.unsupported_confidence_cap
        for asset in assets:
            up_sum = 0.0
            down_sum = 0.0
            weighted_magnitude = 0.0
            mag_count = 0
            up_sources: list[str] = []
            down_sources: list[str] = []
            contributing_labels: set[str] = set()

            for r in results:
                rec = r.recommendation
                if rec is None or rec.asset != asset:
                    continue
                contributing_labels.add(r.label)
                src_labels = r.forecast_source_labels or [r.label]
                if rec.direction == Direction.CALL:
                    up_sum += r.confidence
                    up_sources.extend(src_labels)
                else:
                    down_sum += r.confidence
                    down_sources.extend(src_labels)

                current = self.result.market.mechanics_price(asset) if self.result.market else None
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

            direction: Literal["UP", "DOWN"] | None = (
                "UP" if up_conf > down_conf else ("DOWN" if down_conf > up_conf else None)
            )
            confidence = max(up_conf, down_conf)

            # Unsupported forecast: exactly one contributing strategy, and
            # it isn't the LLM. This DirectionalForecast is computed after
            # DecisionAggregator.aggregate() has already selected today's
            # trade from the raw (uncapped) StrategyResult confidences, so
            # this cap does not change today's execution -- it corrects
            # what gets published in the summary JSON and, via
            # prediction_tracker.read_previous_forecasts, what confidence
            # tomorrow's LLM prompt sees for this (date, asset) in its
            # prediction history.
            if (
                len(contributing_labels) == 1
                and "llm_trade" not in contributing_labels
                and confidence > confidence_cap
            ):
                (sole_strategy,) = contributing_labels
                logger.warning(
                    "forecast_confidence_capped",
                    asset=asset,
                    strategy=sole_strategy,
                    original_confidence=confidence,
                    cap=confidence_cap,
                )
                confidence = confidence_cap

            pct = round((weighted_magnitude / mag_count) * 100 if mag_count > 0 else 0.0, 2)
            target_strike: float | None = None
            if direction and abs(pct) >= 0.1:
                # MECHANICS: live pre-market price, never the stale
                # prior-session open (see MarketSnapshot.mechanics_price).
                spot = self.result.market.mechanics_price(asset) if self.result.market else None
                if spot and spot > 0:
                    target_strike = round(spot * (1 + pct / 100), 2)
            forecasts.append(
                AssetForecast(
                    asset=asset,
                    direction=direction,
                    confidence=confidence,
                    predicted_move_pct=pct if direction else 0.0,
                    target_strike=target_strike,
                    sources=up_sources + down_sources,
                )
            )

        return DirectionalForecast(forecasts=forecasts)

    async def _phase_execute(self, decision: DecisionOutput) -> dict | None:
        """Phase 5: Execute the selected trade through the Alpaca broker.

        Uses the :class:`ExecutionEngine` with direct ``alpaca-py``
        API calls and automatic exit management.

        Args:
            decision: The final decision output with a recommendation.

        Returns:
            Dict with execution result, or None if no recommendation.
        """
        rec = decision.recommendation
        if rec is None:
            return None

        rec.correlation_id = self.correlation_id
        api_key = os.environ.get("APCA_API_KEY_ID", "")
        api_secret = os.environ.get("APCA_API_SECRET_KEY", "")

        if not api_key or not api_secret:
            msg = "Alpaca API keys not set — skipping execution"
            logger.warning("execution_skipped", reason="no_api_keys")
            self.result.errors.append(msg)
            return {"error": msg}

        paper = self.config.general.env_mode == "PAPER_ALPACA"
        exec_config = (
            self.config.execution if hasattr(self.config, "execution") else ExecutionConfig()
        )

        client = AlpacaBrokerClient(api_key, api_secret, paper=paper, config=exec_config)
        engine = ExecutionEngine(
            client,
            exec_config,
            log_dir=self.config.logging.json_dir,
            risk_config=self.config.risk,
        )

        try:
            result = await engine.execute(rec, self.correlation_id)
            self.file_logger.write_entry(
                {
                    "phase": "execute",
                    "trade_id": result.get("trade_id", ""),
                    "exit_reason": result.get("exit_reason", ""),
                    "final_pnl": result.get("final_pnl", 0.0),
                    "final_pnl_pct": result.get("final_pnl_pct", 0.0),
                }
            )
            return result
        except Exception as e:
            msg = f"Execution failed: {e}"
            logger.error("execution_error", error=str(e))
            self.result.errors.append(msg)
            return {"error": msg}

    async def _check_yesterday_prediction(self) -> None:
        provider = build_candle_chain(
            finnhub_api_key=self.config.finnhub.api_key,
            alpha_vantage_api_key=self.config.alpha_vantage.api_key,
            timeout=self.config.finnhub.request_timeout_sec,
        )
        log_dir = self.config.logging.json_dir

        biz_date = await find_previous_business_day(log_dir, provider)
        if biz_date is None:
            logger.info("yesterday_check_skipped", reason="no_business_day_found")
            return

        forecasts = read_previous_forecasts(log_dir, biz_date)
        if not forecasts:
            logger.info(
                "yesterday_check_skipped",
                reason="no_forecasts_found",
                biz_date=biz_date.isoformat(),
            )
            return

        outcomes: list[PredictionOutcome] = []
        for pred in forecasts:
            daily = await provider.fetch_daily_candle(pred["asset"], biz_date)
            hourly = await provider.fetch_intraday_candles(pred["asset"], biz_date)
            outcome = check_outcome(pred, daily, hourly, biz_date=biz_date)
            outcomes.append(outcome)
            logger.info(
                "yesterday_outcome",
                asset=outcome.asset,
                predicted=outcome.predicted_direction,
                result=outcome.result,
                details=outcome.details,
            )

        self.result.yesterday_outcomes = outcomes
        append_outcomes(log_dir, outcomes)
