"""Application configuration and settings management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings


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


class MeanReversionConfig(StrategyConfig):
    """Configuration for the mean-reversion trading strategy."""

    rsi_oversold: int = 35
    rsi_overbought: int = 65


class EventDrivenConfig(StrategyConfig):
    """Configuration for the event-driven trading strategy."""

    min_confidence: float = 0.45
    catalyst_window_hours: int = 17


class StrategiesConfig(BaseModel):
    """Container holding configuration for all trading strategies."""

    momentum: MomentumConfig = MomentumConfig()
    mean_reversion: MeanReversionConfig = MeanReversionConfig()
    event_driven: EventDrivenConfig = EventDrivenConfig()


class RiskConfig(BaseModel):
    """Configuration for trade risk guardrails."""

    max_loss_per_trade_usd: float = 1000.0
    max_position_size_contracts: int = 10
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


class LoggingConfig(BaseModel):
    """Configuration for logging behaviour."""

    level: str = "INFO"
    json_dir: str = "logs"


class GeneralConfig(BaseModel):
    """Top-level general configuration for the application."""

    env_mode: Literal["PAPER_ALPACA", "LIVE_ROBINHOOD"] = "PAPER_ALPACA"
    target_assets: list[str] = Field(default_factory=lambda: ["SPY", "QQQ"])
    execute: bool = False


class LLMConfig(BaseModel):
    """Configuration for the local LLM fallback used to re-synthesize a
    degraded Atlas briefing.

    The Atlas morning briefing is normally LLM-synthesised upstream.
    When the upstream LLM layer fails (e.g. the 2026-07-18 DeepSeek
    free-tier hang documented in ``LESSONS_LEARNED.md``), this project
    invokes the ``opencode`` CLI locally to re-synthesise the executive
    summary from the raw feed items.

    Models are split into two tiers by provider namespace:

    - ``zen_models`` — the free OpenCode Zen namespace (``opencode/*``).
      Tried first, in order, under a strict per-call timeout. Default
      list includes every ``-free`` Zen model currently published by
      ``opencode models``, ordered by expected quality / context fit.
    - ``paid_go_models`` — the paid OpenCode Go namespace
      (``opencode-go/*``). Tried only after every Zen model has been
      exhausted. Surface via :attr:`OpencodeLLMClient.paid_used` so
      the pipeline can log when a re-synthesis incurred a cost.

    As of 2026-07-18 these Zen IDs were observed via ``opencode models``:
    ``opencode/deepseek-v4-flash-free``, ``opencode/mimo-v2.5-free``,
    ``opencode/hy3-free``, ``opencode/nemotron-3-ultra-free``,
    ``opencode/north-mini-code-free``, ``opencode/big-pickle``.
    """

    enabled: bool = True
    opencode_path: str = "opencode"
    zen_models: list[str] = Field(
        default_factory=lambda: [
            "opencode/deepseek-v4-flash-free",
            "opencode/mimo-v2.5-free",
            "opencode/nemotron-3-ultra-free",
            "opencode/hy3-free",
        ]
    )
    paid_go_models: list[str] = Field(
        default_factory=lambda: [
            "opencode-go/glm-5.2",
            "opencode-go/kimi-k3",
            "opencode-go/qwen3.7-max",
        ]
    )
    timeout_sec: int = 60
    max_calls_per_run: int = 5
    # LLM-driven trade-signal strategy. When enabled, an
    # ``LLMTradeStrategy`` runs alongside Momentum / MeanReversion /
    # EventDriven and asks the LLM to emit a structured
    # {asset, direction, confidence, rationale} JSON pick which is then
    # folded into the DecisionAggregator like any other strategy
    # result. The LLM re-synthesis of degraded briefings is independent
    # of this flag.
    trade_signal_enabled: bool = True
    trade_signal_min_confidence: float = 0.45


class Settings(BaseSettings):
    """Root application settings loaded from YAML or environment variables."""

    general: GeneralConfig = GeneralConfig()
    atlas_briefing: AtlasBriefingConfig = AtlasBriefingConfig()
    finnhub: FinnhubConfig = FinnhubConfig()
    brave: BraveConfig = BraveConfig()
    reddit: RedditConfig = RedditConfig()
    unusual_whales: UnusualWhalesConfig = UnusualWhalesConfig()
    rss_feeds: list[RSSFeedItem] = Field(default_factory=list)
    strategies: StrategiesConfig = StrategiesConfig()
    risk: RiskConfig = RiskConfig()
    mcp: MCPConfig = MCPConfig()
    logging: LoggingConfig = LoggingConfig()
    llm: LLMConfig = LLMConfig()

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
