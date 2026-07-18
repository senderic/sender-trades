"""Application configuration and settings management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings


class AtlasBriefingConfig(BaseModel):
    """Configuration for the Atlas morning briefing directory."""

    directory: str = "~/atlas-morning-briefing"

    @property
    def resolved_directory(self) -> Path:
        """Resolve and expand the briefing directory path.

        Returns:
            Absolute path with tilde and environment variables expanded.
        """
        return Path(self.directory).expanduser().resolve()


class FinnhubConfig(BaseModel):
    """Configuration for Finnhub API access."""

    api_key: str = ""
    request_timeout_sec: int = 10


class BraveConfig(BaseModel):
    """Configuration for Brave Search API access."""

    api_key: str = ""
    news_query: str = "SPY QQQ stock market intraday trading"


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

    max_loss_per_trade_usd: float = 500.0
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


class LoggingConfig(BaseModel):
    """Configuration for logging behaviour."""

    level: str = "INFO"
    json_dir: str = "logs"


class GeneralConfig(BaseModel):
    """Top-level general configuration for the application."""

    env_mode: Literal["PAPER_ALPACA", "LIVE_ROBINHOOD"] = "PAPER_ALPACA"
    target_assets: list[str] = Field(default_factory=lambda: ["SPY", "QQQ"])
    execute: bool = False


class Settings(BaseSettings):
    """Root application settings loaded from YAML or environment variables."""

    general: GeneralConfig = GeneralConfig()
    atlas_briefing: AtlasBriefingConfig = AtlasBriefingConfig()
    finnhub: FinnhubConfig = FinnhubConfig()
    brave: BraveConfig = BraveConfig()
    rss_feeds: list[RSSFeedItem] = Field(default_factory=list)
    strategies: StrategiesConfig = StrategiesConfig()
    risk: RiskConfig = RiskConfig()
    mcp: MCPConfig = MCPConfig()
    logging: LoggingConfig = LoggingConfig()

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
