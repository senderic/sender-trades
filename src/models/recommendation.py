from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class Direction(str, Enum):
    """Option direction: CALL or PUT."""

    CALL = "CALL"
    PUT = "PUT"


class PositionIntent(str, Enum):
    """Position opening intent: buy-to-open or sell-to-open."""

    BUY_TO_OPEN = "buy_to_open"
    SELL_TO_OPEN = "sell_to_open"


class Leg(BaseModel):
    """A single leg in a multi-leg option order."""

    symbol: str
    ratio_qty: str
    side: Optional[Literal["buy", "sell"]] = None
    position_intent: Optional[PositionIntent] = None


class AlpacaOrderPayload(BaseModel):
    """Payload for an Alpaca order via MCP."""

    qty: str
    type: str = "market"
    time_in_force: str = "day"
    symbol: Optional[str] = None
    side: Optional[Literal["buy", "sell"]] = None
    position_intent: Optional[str] = None
    limit_price: Optional[str] = None
    client_order_id: Optional[str] = None
    order_class: Optional[str] = None
    legs: Optional[list[dict]] = None

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
    price: Optional[float] = None
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
    limit_price: Optional[float] = None
    position_intent: PositionIntent = PositionIntent.BUY_TO_OPEN
    rationale: dict[str, Any] = Field(default_factory=dict)
    expires_at: str = ""
    must_close_before: str = "15:30"
    legs: Optional[list[Leg]] = None


class StrategyResult(BaseModel):
    """The output of a single strategy evaluation."""

    label: str
    recommendation: Optional[TradeRecommendation] = None
    confidence: float
    debug_trace: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0


class DecisionOutput(BaseModel):
    """The final decision output after aggregation and risk validation."""

    selected_label: Optional[str] = None
    recommendation: Optional[TradeRecommendation] = None
    all_results: list[StrategyResult] = Field(default_factory=list)
    rationale: str = ""
