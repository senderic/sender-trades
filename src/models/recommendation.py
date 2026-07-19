from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from src.timezone import now_local


class Direction(StrEnum):
    """Option direction: CALL or PUT."""

    CALL = "CALL"
    PUT = "PUT"


class PositionIntent(StrEnum):
    """Position opening intent: buy-to-open or sell-to-open."""

    BUY_TO_OPEN = "buy_to_open"
    SELL_TO_OPEN = "sell_to_open"


class Leg(BaseModel):
    """A single leg in a multi-leg option order."""

    symbol: str
    ratio_qty: str
    side: Literal["buy", "sell"] | None = None
    position_intent: PositionIntent | None = None


class AlpacaOrderPayload(BaseModel):
    """Payload for an Alpaca order via MCP."""

    qty: str
    type: str = "market"
    time_in_force: str = "day"
    symbol: str | None = None
    side: Literal["buy", "sell"] | None = None
    position_intent: str | None = None
    limit_price: str | None = None
    client_order_id: str | None = None
    order_class: str | None = None
    legs: list[dict] | None = None

    @model_validator(mode="after")
    def _validate_legs_or_symbol(self) -> AlpacaOrderPayload:
        has_legs = self.legs is not None and len(self.legs) > 0
        has_symbol = self.symbol is not None
        if has_legs and has_symbol:
            raise ValueError("Provide either legs (multi-leg) or symbol (single-leg), not both")
        if not has_legs and not has_symbol:
            raise ValueError("Either legs or symbol must be provided")
        return self


class RobinhoodOrderPayload(BaseModel):
    """Payload for a Robinhood order via MCP."""

    symbol: str
    direction: str
    quantity: int
    order_type: str = "market"
    price: float | None = None
    time_in_force: str = "day"


class ExecutionCommand(BaseModel):
    """A command to execute a trade through an MCP broker."""

    action: str = "place_option_order"
    payload: AlpacaOrderPayload | RobinhoodOrderPayload
    env_mode: str


class TradeRecommendation(BaseModel):
    """A fully specified trade recommendation ready for execution."""

    correlation_id: str
    strategy_label: str
    asset: Literal["SPY", "QQQ"]
    direction: Direction
    confidence: float
    target_strike: float
    contracts: int
    order_type: Literal["market", "limit"] = "market"
    limit_price: float | None = None
    position_intent: PositionIntent = PositionIntent.BUY_TO_OPEN
    rationale: dict[str, Any] = Field(default_factory=dict)
    expires_at: str = ""
    must_close_before: str = "15:30"
    legs: list[Leg] | None = None


class StrategyResult(BaseModel):
    """The output of a single strategy evaluation.

    ``forecast_source_labels`` lets a strategy override the default
    ``[label]`` provenance surfaced in the forecast table's
    ``up_sources`` / ``down_sources`` columns. Deterministic
    strategies leave it as ``None`` so the aggregator falls back to
    ``[label]``. LLM-driven strategies populate it with the
    LLM-cited root provenance (e.g. ``["llm:atlas-briefing",
    "llm:reuters", "llm:watchlist:NVDA"]``) so the forecast reader can
    see what actually drove the decision rather than just the strategy
    name.

    ``predictions`` is populated by LLM-driven strategies to surface
    per-asset predictions (direction + expected move %) that feed
    directly into the :class:`DirectionalForecast` table. When set,
    the pipeline's ``_compute_forecast`` uses these predictions as
    the primary forecast for each asset.
    """

    label: str
    recommendation: TradeRecommendation | None = None
    predictions: dict[str, AssetPrediction] | None = None
    confidence: float
    debug_trace: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0
    forecast_source_labels: list[str] | None = None


class AssetPrediction(BaseModel):
    """A directional prediction for a single asset produced by a strategy."""

    asset: Literal["SPY", "QQQ"]
    direction: Literal["UP", "DOWN"]
    confidence: float
    predicted_move_pct: float
    rationale: str = ""
    sources: list[str] = Field(default_factory=list)


class AssetForecast(BaseModel):
    """Directional forecast for a single asset."""

    asset: Literal["SPY", "QQQ"]
    direction: Literal["UP", "DOWN"] | None = None
    confidence: float = 0.0
    predicted_move_pct: float = 0.0
    rationale: str = ""
    sources: list[str] = Field(default_factory=list)


class DirectionalForecast(BaseModel):
    """Aggregated directional outlook across all assets."""

    forecasts: list[AssetForecast] = Field(default_factory=list)
    market_vibe: str = ""
    generated_at: datetime = Field(default_factory=now_local)

    def table(self) -> str:
        """Return a formatted table string for terminal output."""
        lines: list[str] = []
        header = f"{'Asset':<6} {'Direction':<10} {'Confidence':>11} {'Pred. Move':>11}   Key Drivers of the Prediction"
        sep = "─" * 100
        lines.append(sep)
        lines.append(header)
        lines.append(sep)
        for f in self.forecasts:
            direction_str = f.direction if f.direction else "—"
            conf_str = f"{f.confidence:.0%}" if f.confidence > 0 else "—"
            move_str = f"{f.predicted_move_pct:+.1f}%" if f.predicted_move_pct != 0.0 else "—"
            drivers = f.rationale if f.rationale else (" · ".join(f.sources) if f.sources else "—")
            lines.append(
                f"{f.asset:<6} {direction_str:<10} {conf_str:>11} {move_str:>11}   {drivers}"
            )
        lines.append(sep)
        if self.market_vibe:
            lines.append(f"\nMarket Vibe: {self.market_vibe}")
        return "\n".join(lines)


class PredictionOutcome(BaseModel):
    date: str
    correlation_id: str = ""
    asset: Literal["SPY", "QQQ"]
    predicted_direction: Literal["UP", "DOWN"]
    confidence: float = 0.0
    rationale: str = ""
    result: Literal["success", "fail", "unknown"] = "unknown"
    details: str = ""
    open_price: float | None = None
    high_price: float | None = None
    low_price: float | None = None
    close_price: float | None = None
    triggered_at: str = ""
    duration_hours: float | None = None


class DecisionOutput(BaseModel):
    """The final decision output after aggregation and risk validation."""

    selected_label: str | None = None
    recommendation: TradeRecommendation | None = None
    forecast: DirectionalForecast | None = None
    all_results: list[StrategyResult] = Field(default_factory=list)
    rationale: str = ""
