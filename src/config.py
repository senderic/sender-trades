"""Application configuration and settings management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings

from src.execution.models import ExecutionConfig


class AtlasBriefingConfig(BaseModel):
    """Configuration for the Atlas morning briefing directory.

    ``directory`` points at the upstream project root (typically
    ``~/atlas-morning-briefing``). The root holds ``status.json`` and
    the ``snapshots/`` tree; the briefing markdown files themselves
    live in a sub-directory (``briefings/`` in the current upstream
    layout) identified by ``briefings_subdir``.

    For legacy layouts where briefings sit directly in the root
    (mixed with stale files), set ``briefings_subdir`` to an empty
    string to search the root, or leave it defaulting to
    ``"briefings"`` -- :func:`src.ingestion.parser.find_todays_briefing`
    falls back to the root when the subdir contains no
    ``Atlas-Briefing-*.md`` files.
    """

    directory: str = "~/atlas-morning-briefing"
    briefings_subdir: str = "briefings"
    snapshot_enabled: bool = True

    @property
    def resolved_directory(self) -> Path:
        """Resolve and expand the project-root directory path.

        Returns:
            Absolute path with tilde and environment variables expanded.
        """
        return Path(self.directory).expanduser().resolve()

    @property
    def briefings_dir(self) -> Path:
        """Path to the directory containing ``Atlas-Briefing-*.md`` files.

        Returns ``resolved_directory`` when ``briefings_subdir`` is
        empty; otherwise ``<root>/<briefings_subdir>``.
        """
        root = self.resolved_directory
        if not self.briefings_subdir:
            return root
        return root / self.briefings_subdir


class FinnhubConfig(BaseModel):
    """Configuration for Finnhub API access."""

    api_key: str = ""
    request_timeout_sec: int = 10


class AlphaVantageConfig(BaseModel):
    """Configuration for Alpha Vantage API access (free tier)."""

    api_key: str = ""
    request_timeout_sec: int = 10


class BraveConfig(BaseModel):
    """Configuration for Brave Search API access."""

    api_key: str = ""
    news_query: str = "SPY QQQ stock market intraday trading"


class RedditConfig(BaseModel):
    """Configuration for Reddit source scraping."""

    enabled: bool = True
    subreddits: list[str] = Field(default_factory=lambda: ["wallstreetbets", "options"])
    post_limit: int = 25
    user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) sender-trades/1.0"


class UnusualWhalesConfig(BaseModel):
    """Configuration for Unusual Whales options flow API."""

    api_key: str = ""
    enabled: bool = True


class RSSFeedItem(BaseModel):
    """A single RSS feed URL to poll for market news."""

    url: str


class StrategyConfig(BaseModel):
    """Base configuration shared by all trading strategies."""

    enabled: bool = True
    min_confidence: float = 0.40


class MomentumConfig(StrategyConfig):
    """Configuration for the momentum trading strategy."""

    gap_threshold_pct: float = 0.5
    # Momentum has produced 0 wins in 4 trades (2026-08-10..26) for -$182,
    # and its raw confidence formula (0.4 + |gap|/10 + |sentiment|) inflated
    # picks to 0.65-0.91, outranking the LLM on several of its losses. Cap
    # the reported confidence so an uncorroborated momentum signal can no
    # longer claim conviction it has not demonstrated.
    max_confidence: float = 0.50


class MeanReversionConfig(StrategyConfig):
    """Configuration for the mean-reversion trading strategy."""

    rsi_oversold: int = 35
    rsi_overbought: int = 65


class EventDrivenConfig(StrategyConfig):
    """Configuration for the event-driven trading strategy."""

    min_confidence: float = 0.45
    catalyst_window_hours: int = 17


class GapFadeConfig(BaseModel):
    """Thresholds for the gap-fade reversal pattern.

    A large pre-market gap backed by a proportionally weak news catalyst
    often exhausts and reverses during the session (Aug 5 2026: SPY
    +1.8% gap closed -0.8%, QQQ +3.4% gap closed -1.2%). These
    thresholds are consumed by the risk engine, the deterministic
    strategies, and every LLM prompt that mentions gap-fade risk, so
    they live in one place rather than being restated per call site.

    ``thresholds_pct`` is keyed by asset symbol; assets absent from the
    mapping fall back to :attr:`default_threshold_pct`.
    """

    thresholds_pct: dict[str, float] = Field(default_factory=lambda: {"SPY": 1.5, "QQQ": 2.0})
    default_threshold_pct: float = 1.5
    sentiment_magnitude_max: float = 0.20

    def threshold_for(self, asset: str) -> float:
        """Return the gap-fade threshold percentage for ``asset``."""
        return self.thresholds_pct.get(asset, self.default_threshold_pct)


class StrategiesConfig(BaseModel):
    """Container holding configuration for all trading strategies."""

    momentum: MomentumConfig = MomentumConfig()
    mean_reversion: MeanReversionConfig = MeanReversionConfig()
    event_driven: EventDrivenConfig = EventDrivenConfig()


class RiskConfig(BaseModel):
    """Configuration for trade risk guardrails."""

    max_loss_per_trade_usd: float = 1000.0
    max_position_size_contracts: int = 10
    # Conservative 2-stage sizing: a selected trade scales to 2 contracts
    # only when its (checker/streak-adjusted) confidence clears this bar AND
    # it is LLM-backed or corroborated by a second strategy. Everything else
    # stays at 1. Sized below the $500 max-loss cap for typical premiums.
    sizing_tier2_min_confidence: float = 0.60
    # Minimum absolute predicted move (in %) for a trade to be executed.
    # 68%-accurate direction on a 0.1-0.2% move still dies to theta, so only
    # trade when the model expects a meaningful session move.
    min_predicted_move_pct: float = 0.30
    # Tie-break preference across assets at equal confidence. QQQ has shown
    # better direction accuracy and realized PnL than SPY across history.
    preferred_asset: str = "QQQ"
    # Premium-aware entry gate (2026-09-10 audit): predicted moves run ~3x
    # hotter than what the underlying actually does intraday (predicted
    # -0.7/-0.8%, actual ~-0.2% on 2026-09-09), so a shrink factor of ~0.4
    # (roughly 1 / 2.5, the inverse of that overestimate) converts the LLM's
    # predicted_move_pct into a realistic expected move before comparing it
    # against what the option itself costs. See
    # DecisionAggregator.premium_gate for the full breakeven calculation.
    predicted_move_shrink: float = 0.4
    # Extra cushion required above the breakeven computed by
    # DecisionAggregator.premium_gate, as a fraction. The breakeven itself
    # now includes the OTM distance from underlying to strike (2026-09-10
    # follow-up review -- the first version of this gate used ask/strike,
    # dropping the distance term and understating breakeven by ~0.6% on
    # every trade, since every strike here is chosen ~0.6% OTM). With that
    # distance now explicit, this margin only needs to cover spread/
    # slippage on exit, not stand in for the OTM gap -- hence 0.03, not the
    # 0.15 used previously. The trade audits don't retain the quoted ask
    # (only the fill price), so this can't be fitted precisely from
    # logs/; 0.03 is a reasoned default in the requested 0.02-0.05 band,
    # not a measured statistic.
    breakeven_margin_pct: float = 0.03
    close_deadline_est: str = "15:30"
    min_dte: int = 0
    max_dte: int = 0
    max_bid_ask_spread_pct: float = 20.0
    min_data_sources_for_direction: int = 2
    std_dev_threshold: float = 3.0


class MCPDaemonConfig(BaseModel):
    """Configuration for a single MCP daemon process."""

    command: str = "uvx"
    args: list[str] = ["alpaca-mcp-server"]
    timeout_sec: int = 30


class MCPConfig(BaseModel):
    """Configuration for MCP broker connections."""

    alpaca: MCPDaemonConfig = MCPDaemonConfig()
    robinhood: MCPDaemonConfig = MCPDaemonConfig(
        args=["robinhood-mcp-server"],
    )
    options_chain: MCPDaemonConfig | None = None


class LoggingConfig(BaseModel):
    """Configuration for logging behaviour."""

    level: str = "INFO"
    json_dir: str = "logs"


class GeneralConfig(BaseModel):
    """Top-level general configuration for the application."""

    env_mode: Literal["PAPER_ALPACA", "LIVE_ROBINHOOD"] = "PAPER_ALPACA"
    target_assets: list[str] = Field(default_factory=lambda: ["SPY", "QQQ"])
    execute: bool = False
    require_forecast_alignment: bool = True


class PreflightConfig(BaseModel):
    """Pre-flight model availability probing.

    Adapted from ``~/atlas-morning-briefing/scripts/preflight_model_check.py``.
    A probe runs ahead of the pipeline, walks the configured model chain,
    and writes :attr:`file_path`. The pipeline then pins a model already
    known to answer, instead of discovering a dead or degraded one
    mid-run at the cost of a full node timeout.

    Two rules carried over from atlas, both learned the hard way there:

    - The roster comes from config, never from a table in the probe. A
      local copy drifts from ``config.yaml`` and silently swaps models.
    - Stale results are worse than none. Model health changes hour to
      hour, so a file older than :attr:`max_age_sec` is ignored and the
      configured chain order is used.

    One deliberate divergence: atlas pins the *first* model that answers,
    because its failure mode is hard outages. Here the failure mode is
    per-node timeouts, so ``select_by: latency`` pins the *fastest*
    healthy model instead.
    """

    enabled: bool = False
    file_path: str = ".model-availability.json"
    max_age_sec: int = 6 * 3600
    probe_timeout_sec: int = 45
    select_by: Literal["latency", "order"] = "latency"


class LLMConfig(BaseModel):
    """Configuration for the LLM calls made via the ``opencode`` CLI.

    Covers every consumer in the pipeline: the graph nodes (research,
    predict, checker, pick-trade), the monolithic trade-signal fallback,
    and the re-synthesis of a degraded Atlas briefing. All of these are
    analysis tasks, so every call uses a capable reasoning model.

    Models are tried in order: :attr:`primary_model` first, then
    :attr:`fallback_models`. Muse Spark 1.3 via the Zen opencode route
    (``opencode/*``) is the model of record; Nemotron 3 Ultra via
    nvidia-direct (``nvidia-direct/*``) is the only fallback. DeepSeek
    was removed from the chain on 2026-09-11 (owner decision).
    """

    enabled: bool = True
    opencode_path: str = "opencode"
    primary_model: str = "opencode/muse-spark-1.3-contributor-free"
    fallback_models: list[str] = Field(
        default_factory=lambda: [
            "nvidia-direct/nvidia/nemotron-3-ultra-550b-a55b",
        ]
    )
    timeout_sec: int = 45
    max_calls_per_run: int = 14
    # LLM-driven trade-signal strategy. When enabled, an
    # ``LLMTradeStrategy`` runs alongside Momentum / MeanReversion /
    # EventDriven and asks the LLM to emit a structured
    # {asset, direction, confidence, rationale} JSON pick which is then
    # folded into the DecisionAggregator like any other strategy
    # result. The LLM re-synthesis of degraded briefings is independent
    # of this flag.
    trade_signal_enabled: bool = True
    trade_signal_min_confidence: float = 0.45
    preflight: PreflightConfig = PreflightConfig()


class GraphConfig(BaseModel):
    """Configuration for the LLM graph orchestration engine.

    Replaces the monolithic single-call LLM prediction with a diamond-
    shaped graph of narrow-scope subagent nodes: research (per-asset),
    prediction (per-asset), a checker that validates and cross-references
    all outputs, and a final pick-trade node.

    When ``enabled`` is False (the initial default), the pipeline falls
    back to the monolithic ``LLMTradeStrategy`` call.
    """

    enabled: bool = False
    fallback_to_monolithic: bool = True
    checker_contradiction_action: Literal["veto", "penalize"] = "veto"
    checker_confidence_penalty: float = 0.15
    research_timeout_sec: int = 45
    prediction_timeout_sec: int = 30
    checker_timeout_sec: int = 30
    pick_trade_timeout_sec: int = 30
    # Wall-clock ceiling for the entire graph. Per-node timeouts above
    # are per *attempt*; with a 7-model fallback chain the node-level
    # worst case runs to many minutes, which would overrun the
    # ``execution.entry.entry_window_minutes`` window given the pipeline
    # starts ~2 min before the open. When this budget is exhausted the
    # orchestrator stops starting new nodes and reports a graph failure
    # so the caller can fall back.
    total_deadline_sec: int = 360
    # LLM calls held back from ``LLMConfig.max_calls_per_run`` so the
    # monolithic fallback is still affordable after a graph failure.
    reserved_calls_for_fallback: int = 1
    # Ceiling on the confidence of a signal that rests on a single
    # deterministic strategy with no LLM corroboration. Applied in two
    # places: the published DirectionalForecast, and — critically — the
    # DecisionAggregator gate, because the forecast is computed AFTER the
    # decision and is never read back by the trading path, so capping the
    # forecast alone changes nothing about what actually trades.
    #
    # Sized deliberately BELOW both strategy gates
    # (momentum.min_confidence 0.40, llm.trade_signal_min_confidence 0.45)
    # so an uncorroborated signal cannot clear them on its own. That is the
    # whole point: a value above 0.40 dampens the number without blocking
    # the trade.
    #
    # Evidence (final run per day, 2026-07-29..08-28): trades taken on a
    # lone deterministic strategy went 1 win / 4 losses for -$64, while
    # corroborated or LLM-backed trades went 5/12 for +$158. Every solo
    # trade fired between 0.55 and 0.80 confidence, so nothing below 0.55
    # would have been filtered by the existing gates. Small sample (n=5) —
    # the asymmetry is that a blocked trade costs an opportunity, while an
    # uncorroborated one has so far cost money.
    unsupported_confidence_cap: float = 0.35


class PremarketConfig(BaseModel):
    """Live pre-market price/volume reconstruction (see ``src.ingestion.premarket``).

    Fixes the 2026-09-11 incident where the atlas Finnhub snapshot --
    taken ~06:02 PT, before Finnhub has any pre-market price -- was fed
    into the pipeline as if it were TODAY's quote. Outside market hours
    Finnhub's ``/quote`` returns the prior completed session's close as
    "current price" (and the session before that as "previous close"),
    so the pipeline was computing YESTERDAY's own gap/move, not today's
    (see ``Quote.prior_session_close``). QQQ that morning: snapshot
    ``current_price`` 708.69 was exactly Thursday 9/10's close; QQQ
    actually opened Friday at 715.44 (+0.95%), a ~1.6% miss on the
    strike computed from the stale price.

    Live price/volume come from Alpaca 1-minute bars on the IEX feed --
    this paper account has no SIP/consolidated-tape subscription (see
    ``AlpacaBrokerClient.get_minute_bars``). IEX pre-market volume for
    SPY/QQQ is a small, noisy slice of true consolidated volume: an
    empirical probe (2026-08-20..09-10, 16 trading days) found median
    cumulative pre-market volume to the 09:28 ET cutoff of ~1,779
    shares for QQQ and ~610 for SPY, ranging day to day from 0 to
    8,425 -- an absolute floor would be meaningless at that scale and
    variance, so :attr:`min_volume_ratio` is relative to this same
    account's own trailing-day norm at the same clock time, which
    cancels out the fixed (small) IEX/consolidated ratio.
    """

    enabled: bool = True
    # Matches the pipeline's cron run time (9:28 AM ET, 2 min before the
    # 9:30 open) -- see AGENTS.md "Schedule".
    cutoff_et: str = "09:28"
    session_start_et: str = "04:00"
    # Trailing window for the volume-reliability comparison.
    lookback_days: int = 10
    # Minimum fraction of the trailing lookback_days median cumulative
    # pre-market volume (same symbol, same clock cutoff) for today's
    # pre-market move to be treated as reliable ANALYSIS EVIDENCE in the
    # LLM prompt -- below this, the prompt is told the move is thin and
    # to weight it lightly. MECHANICS (strike selection, the forecast
    # target strike, the gap used by gap-fade) are unaffected and always
    # use the live pre-market price regardless of this flag -- a strike
    # computed from yesterday's close is wrong regardless of volume.
    # 0.5 (half the trailing median) is a reasoned default given the
    # empirical spread documented above, not fitted from trade outcomes
    # -- there isn't yet a labeled sample of "thin premarket day"
    # outcomes to fit against.
    min_volume_ratio: float = 0.5
    # MECHANICS price source order (2026-09-11 follow-up review): (a) the
    # live underlying quote's bid/ask midpoint, (b) the last pre-market
    # bar's close if recent, (c) the prior session's close (logged as a
    # WARNING). A zero- or thin-volume morning has no recent trade, but
    # Alpaca still returns a live bid/ask (market makers quote without
    # anyone trading) -- e.g. 2026-09-11 09:29 ET, QQQ's live quote read
    # 715.64 against a real 715.44 open, while the last pre-market TRADE
    # could have been an hour stale. A bar older than this (minutes,
    # measured against the session's cutoff) is treated as too stale to
    # use for mechanics OR to count as reliable evidence, regardless of
    # volume_ratio. The historical replay has no cheap way to reconstruct
    # a past bid/ask quote, so it skips tier (a) and relies on (b)/(c)
    # only -- see fetch_premarket_quote's `quote_fetcher` parameter.
    max_bar_age_min: float = 15.0
    # Maximum bid/ask spread, as a percentage of the midpoint, for the
    # live quote to be trusted as a mechanics price (tier (a) above). A
    # very wide spread means the "midpoint" isn't a meaningful price.
    max_quote_spread_pct: float = 0.1


class Settings(BaseSettings):
    """Root application settings loaded from YAML or environment variables."""

    general: GeneralConfig = GeneralConfig()
    atlas_briefing: AtlasBriefingConfig = AtlasBriefingConfig()
    finnhub: FinnhubConfig = FinnhubConfig()
    alpha_vantage: AlphaVantageConfig = AlphaVantageConfig()
    brave: BraveConfig = BraveConfig()
    reddit: RedditConfig = RedditConfig()
    unusual_whales: UnusualWhalesConfig = UnusualWhalesConfig()
    rss_feeds: list[RSSFeedItem] = Field(default_factory=list)
    strategies: StrategiesConfig = StrategiesConfig()
    risk: RiskConfig = RiskConfig()
    gap_fade: GapFadeConfig = GapFadeConfig()
    premarket: PremarketConfig = PremarketConfig()
    mcp: MCPConfig = MCPConfig()
    logging: LoggingConfig = LoggingConfig()
    llm: LLMConfig = LLMConfig()
    graph: GraphConfig = GraphConfig()
    execution: ExecutionConfig = ExecutionConfig()

    model_config = ConfigDict(env_nested_delimiter="__")

    @classmethod
    def from_yaml(cls, path: str | Path) -> Settings:
        """Load settings from a YAML configuration file.

        Args:
            path: Path to the YAML file. If the file does not exist,
                returns default settings.

        Returns:
            A populated Settings instance.
        """
        path = Path(path).expanduser().resolve()
        if not path.exists():
            return cls()
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    def resolve_env_vars(self) -> Settings:
        """Resolve ``${VAR}`` placeholders in settings from environment variables.

        Returns:
            A new Settings instance with environment variables substituted.
        """

        def _resolve(value: object) -> object:
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_key = value[2:-1]
                return os.environ.get(env_key, "")
            if isinstance(value, dict):
                return {k: _resolve(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_resolve(v) for v in value]
            return value

        resolved = _resolve(self.model_dump())
        return Settings(**resolved)
