"""0DTE options trade execution engine.

Direct Alpaca API integration with Tenacity retries, automatic exit
management (take-profit, stop-loss, trailing stop, time-based hard
close), and full audit-trail logging.
"""

from __future__ import annotations

from src.execution.client import AlpacaBrokerClient
from src.execution.context import TradeContext
from src.execution.engine import ExecutionEngine
from src.execution.exit_manager import ExitManager
from src.execution.lifecycle import TradeLifecycle, TradeState
from src.execution.models import (
    EntryConfig,
    ExecutionConfig,
    ExitConfig,
    OrderResult,
    TradeRecord,
    TrailingConfig,
)
from src.execution.retry import al_api_retry

__all__ = [
    "AlpacaBrokerClient",
    "EntryConfig",
    "ExecutionConfig",
    "ExecutionEngine",
    "ExitConfig",
    "ExitManager",
    "OrderResult",
    "TradeContext",
    "TradeLifecycle",
    "TradeRecord",
    "TradeState",
    "TrailingConfig",
    "al_api_retry",
]
