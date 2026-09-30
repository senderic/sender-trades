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


class ExitAdvisorConfig(BaseModel):
    """Configuration for the LLM-driven profit-exit advisor.

    Replaces the fixed +100% take-profit for 0DTE options. A replay of
    72 asset-days (2026-07-22..09-10) found holding winners to expiry
    pays only on a handful of big days (avg ~+71%/trade in premium
    units, top 3 days ~45% of total gains); the resting +100% limit
    order cuts those days off exactly, and the fixed TP/SL combo fell to
    ~+7-9%. When enabled, ``ExecutionEngine`` skips the resting +100% TP
    order (placing :attr:`safety_cap_pct` instead, if set) and
    ``src.execution.intraday_monitor`` consults this model at
    event-driven cadence points (see
    ``src.execution.exit_advisor.decide_trigger``) to decide HOLD vs
    EXIT. The stop-loss (-50%), the 15:25 ET time deadline, and the
    12:20 PM PT safety-close sweep are all unaffected hard rails that
    always run first -- this config only governs the *profit* exit path.
    """

    enabled: bool = False
    # Falls back to LLMConfig.primary_model when unset (see
    # src.execution.exit_advisor.resolve_model). Deliberately no
    # fallback chain of its own -- a single heavy model, consulted
    # sparingly, is the point; an unavailable/degraded model instead
    # enforces `trailing` above (see exit_advisor.py docstring).
    model: str | None = None
    # Per-attempt timeout. Must stay well inside the 3-minute cron
    # interval so one advisor call can never make a monitor pass overrun
    # into the next scheduled run.
    timeout_sec: float = 55.0
    # Cadence thresholds -- see src.execution.exit_advisor.decide_trigger.
    profit_step_pct: float = 50.0
    giveback_from_peak_pct: float = 25.0
    periodic_interval_min: float = 15.0
    final_window_min: float = 30.0
    max_calls_per_trade: int = 12
    # Resting safety-cap limit order placed at Alpaca instead of the
    # fixed +100% TP, so total advisor unavailability still has *some*
    # ceiling. `None` places no resting order at all and relies solely
    # on the advisor + trailing-stop fallback + safety-close sweep.
    safety_cap_pct: float | None = 400.0


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
    exit_advisor: ExitAdvisorConfig = ExitAdvisorConfig()
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
