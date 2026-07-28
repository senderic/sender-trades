"""Exit strategy management for 0DTE options trades.

Handles take-profit, stop-loss, trailing stop, and time-based exits.
"""

from __future__ import annotations

from datetime import datetime, time
from typing import TYPE_CHECKING, Any

import structlog

from src.execution.models import ExitConfig
from src.timezone import ET_TZ

if TYPE_CHECKING:
    pass

logger = structlog.get_logger()


class ExitManager:
    """Calculates and manages exit levels for a 0DTE options position.

    When the position fills, the exit manager calculates the take-profit
    and stop-loss levels, then monitors for trailing stop activation and
    time-deadline enforcement.
    """

    def __init__(self, config: ExitConfig):
        """Initialize the exit manager with exit strategy configuration.

        Args:
            config: Exit strategy parameters.
        """
        self.config = config
        self.entry_price: float | None = None
        self._tp_level: float | None = None
        self._sl_level: float | None = None
        self._peak_pnl_pct: float = 0.0
        self._trailing_active: bool = False
        self._trail_level: float | None = None

    def on_entry_filled(self, entry_premium: float) -> None:
        """Configure exit levels after the entry order is filled.

        Args:
            entry_premium: The average fill price per contract.
        """
        if entry_premium <= 0:
            raise ValueError(f"Entry premium must be positive, got {entry_premium}")

        self.entry_price = entry_premium
        self._tp_level = round(entry_premium * (1 + self.config.take_profit_pct / 100.0), 2)
        self._sl_level = round(entry_premium * (1 + self.config.stop_loss_pct / 100.0), 2)
        logger.info(
            "exit_levels_set",
            entry_price=entry_premium,
            take_profit=self._tp_level,
            stop_loss=self._sl_level,
        )

    @property
    def tp_level(self) -> float | None:
        """Current take-profit trigger price."""
        return self._tp_level

    @property
    def sl_level(self) -> float | None:
        """Current stop-loss trigger price."""
        return self._sl_level

    @property
    def trail_level(self) -> float | None:
        """Current trailing stop level (if active)."""
        return self._trail_level

    @property
    def trailing_active(self) -> bool:
        """Whether the trailing stop is currently active."""
        return self._trailing_active

    def evaluate(self, current_price: float) -> dict[str, Any]:
        """Evaluate the current price against all exit conditions.

        Args:
            current_price: Current mark/quote price of the option.

        Returns:
            Dict with exit trigger flags, current PnL, and active levels.
        """
        if self.entry_price is None or self.entry_price <= 0:
            return {
                "triggered": False,
                "trigger_type": None,
                "current_pnl_pct": 0.0,
                "tp_level": self._tp_level,
                "sl_level": self._sl_level,
                "trail_level": self._trail_level,
                "trailing_active": self._trailing_active,
            }

        pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100.0
        pnl_pct = round(pnl_pct, 2)

        self._update_trailing(pnl_pct)

        triggered = False
        trigger_type = None

        if self._sl_level is not None and current_price <= self._sl_level:
            triggered = True
            trigger_type = "stop_loss"
        elif self._tp_level is not None and current_price >= self._tp_level:
            triggered = True
            trigger_type = "take_profit"
        elif (
            self._trailing_active
            and self._trail_level is not None
            and current_price <= self._trail_level
        ):
            triggered = True
            trigger_type = "trailing_stop"
        elif self.is_time_deadline_approaching():
            triggered = True
            trigger_type = "time_deadline"

        return {
            "triggered": triggered,
            "trigger_type": trigger_type,
            "current_pnl_pct": pnl_pct,
            "tp_level": self._tp_level,
            "sl_level": self._sl_level,
            "trail_level": self._trail_level,
            "trailing_active": self._trailing_active,
        }

    def _update_trailing(self, current_pnl_pct: float) -> None:
        if not self.config.trailing.enabled or self.entry_price is None:
            return

        if current_pnl_pct > self._peak_pnl_pct:
            self._peak_pnl_pct = current_pnl_pct

        if not self._trailing_active:
            if current_pnl_pct >= self.config.trailing.activate_after_pct:
                self._trailing_active = True
                self._trail_level = round(
                    self.entry_price
                    * (1 + (current_pnl_pct - self.config.trailing.trail_pct) / 100.0),
                    2,
                )
                logger.info(
                    "trailing_stop_activated",
                    peak_pnl_pct=current_pnl_pct,
                    trail_level=self._trail_level,
                )
        else:
            new_trail = round(
                self.entry_price
                * (1 + (self._peak_pnl_pct - self.config.trailing.trail_pct) / 100.0),
                2,
            )
            if new_trail > (self._trail_level or 0):
                self._trail_level = new_trail
                logger.debug(
                    "trailing_stop_adjusted",
                    peak_pnl_pct=self._peak_pnl_pct,
                    trail_level=self._trail_level,
                )

    def is_time_deadline_approaching(self, now: datetime | None = None) -> bool:
        """Check if the time deadline for closing has been reached.

        Args:
            now: Current datetime (defaults to now in ET).

        Returns:
            True if the deadline has passed.
        """
        current = now if now is not None else datetime.now(ET_TZ)
        parts = self.config.time_deadline_est.split(":")
        deadline = time(int(parts[0]), int(parts[1]))
        return current.time() >= deadline

    def build_tp_order(self, occ_symbol: str, contracts: int) -> dict[str, Any]:
        """Build a take-profit limit order specification.

        Args:
            occ_symbol: OCC option symbol.
            contracts: Number of contracts.

        Returns:
            Dict with order parameters for the Alpaca API.
        """
        return {
            "symbol": occ_symbol,
            "qty": contracts,
            "side": "sell",
            "type": "limit",
            "limit_price": self._tp_level,
            "time_in_force": "gtc",
            "order_class": "simple",
        }

    def build_sl_order(self, occ_symbol: str, contracts: int) -> dict[str, Any]:
        """Build a stop-loss order specification.

        Args:
            occ_symbol: OCC option symbol.
            contracts: Number of contracts.

        Returns:
            Dict with order parameters for the Alpaca API.
        """
        return {
            "symbol": occ_symbol,
            "qty": contracts,
            "side": "sell",
            "type": "stop",
            "stop_price": self._sl_level,
            "time_in_force": "gtc",
            "order_class": "simple",
        }

    def build_market_close_order(
        self, occ_symbol: str, contracts: int, client_order_id: str | None = None
    ) -> dict[str, Any]:
        """Build a market order to force-close the position.

        Args:
            occ_symbol: OCC option symbol.
            contracts: Number of contracts.
            client_order_id: Optional client-specified order ID.

        Returns:
            Dict with order parameters for the Alpaca API.
        """
        order: dict[str, Any] = {
            "symbol": occ_symbol,
            "qty": contracts,
            "side": "sell",
            "type": "market",
            "time_in_force": "day",
            "order_class": "simple",
        }
        if client_order_id:
            order["client_order_id"] = client_order_id
        return order
