"""Pydantic models for the execution engine."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class TradeState(Enum):
    """States in the trade lifecycle state machine."""

    CREATED = "created"
    VALIDATING = "validating"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    EXITS_PLACED = "exits_placed"
    TP_FILLED = "tp_filled"
    SL_FILLED = "sl_filled"
    FORCE_CLOSED = "force_closed"
    CLOSED = "closed"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        """Return True if this state is terminal (no further transitions)."""
        return self in (
            TradeState.CLOSED,
            TradeState.REJECTED,
            TradeState.EXPIRED,
            TradeState.FAILED,
        )


VALID_TRANSITIONS: dict[TradeState, set[TradeState]] = {
    TradeState.CREATED: {TradeState.VALIDATING, TradeState.FAILED},
    TradeState.VALIDATING: {TradeState.SUBMITTED, TradeState.FAILED},
    TradeState.SUBMITTED: {
        TradeState.ACKNOWLEDGED,
        TradeState.PARTIALLY_FILLED,
        TradeState.FILLED,
        TradeState.REJECTED,
        TradeState.EXPIRED,
        TradeState.FAILED,
    },
    TradeState.ACKNOWLEDGED: {
        TradeState.PARTIALLY_FILLED,
        TradeState.FILLED,
        TradeState.REJECTED,
        TradeState.EXPIRED,
        TradeState.FAILED,
    },
    TradeState.PARTIALLY_FILLED: {
        TradeState.PARTIALLY_FILLED,
        TradeState.FILLED,
        TradeState.REJECTED,
        TradeState.EXPIRED,
        TradeState.FAILED,
    },
    TradeState.FILLED: {TradeState.EXITS_PLACED, TradeState.FAILED},
    TradeState.EXITS_PLACED: {
        TradeState.TP_FILLED,
        TradeState.SL_FILLED,
        TradeState.FORCE_CLOSED,
        TradeState.CLOSED,
        TradeState.FAILED,
    },
    TradeState.TP_FILLED: {TradeState.CLOSED, TradeState.FAILED},
    TradeState.SL_FILLED: {TradeState.CLOSED, TradeState.FAILED},
    TradeState.FORCE_CLOSED: {TradeState.CLOSED, TradeState.FAILED},
    TradeState.CLOSED: set(),
    TradeState.REJECTED: set(),
    TradeState.EXPIRED: set(),
    TradeState.FAILED: set(),
}


class InvalidTransitionError(ValueError):
    """Raised when a lifecycle transition is not allowed."""

    def __init__(self, from_state: TradeState, to_state: TradeState) -> None:
        super().__init__(f"Cannot transition from {from_state.value} to {to_state.value}")


class OrderResult(BaseModel):
    """Result of an order submission or query through the broker."""

    order_id: str
    status: str
    symbol: str = ""
    side: str = ""
    order_type: str = ""
    qty: str = ""
    filled_qty: str = "0"
    filled_avg_price: str | None = None
    limit_price: str | None = None
    created_at: str = ""
    updated_at: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)


class TrailingConfig(BaseModel):
    """Configuration for trailing stop behaviour."""

    enabled: bool = True
    activate_after_pct: float = 30.0
    trail_pct: float = 15.0


class EntryConfig(BaseModel):
    """Configuration for trade entry (order submission)."""

    order_type: Literal["market", "limit"] = "limit"
    limit_offset_pct: float = 5.0
    entry_window_minutes: int = 15


class ExitConfig(BaseModel):
    """Configuration for trade exit strategy."""

    take_profit_pct: float = 100.0
    stop_loss_pct: float = -50.0
    trailing: TrailingConfig = TrailingConfig()
    time_deadline_est: str = "15:25"


class TenacityConfig(BaseModel):
    """Configuration for Tenacity retry behaviour on API calls."""

    max_attempts: int = 3
    min_wait_sec: float = 1.0
    max_wait_sec: float = 30.0
    backoff_multiplier: float = 2.0


class ExecutionConfig(BaseModel):
    """Top-level configuration for the execution engine."""

    entry: EntryConfig = EntryConfig()
    exit_strategy: ExitConfig = ExitConfig()
    tenacity: TenacityConfig = TenacityConfig()
    max_concurrent_trades: int = 1


class TradeRecord(BaseModel):
    """Complete audit trail record for a single trade."""

    trade_id: str
    correlation_id: str
    asset: str
    direction: str
    contracts: int
    entry_price: float
    entry_order_id: str = ""
    entry_filled_at: str = ""
    exit_price: float | None = None
    exit_reason: (
        Literal["take_profit", "stop_loss", "force_close", "expired", "rejected", "error"] | None
    ) = None
    final_pnl: float | None = None
    final_pnl_pct: float | None = None
    duration_seconds: float | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


class LifecycleEvent(BaseModel):
    """A single event recorded during the trade lifecycle."""

    trade_id: str
    state_from: str | None = None
    state_to: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
