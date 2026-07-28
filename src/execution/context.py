"""Trade audit trail — structured JSON logging for full traceability."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from src.models.recommendation import TradeRecommendation

logger = structlog.get_logger()


class TradeContext:
    """Bundles trade identity and writes structured audit JSON.

    Links ``correlation_id`` → ``trade_id`` → all order lifecycle
    events. Writes to ``logs/<date>/trade-<trade_id>.json``.
    """

    def __init__(
        self,
        trade_id: str,
        correlation_id: str,
        recommendation: TradeRecommendation,
        log_dir: str | Path = "logs",
    ):
        """Initialize a trade context for audit logging.

        Args:
            trade_id: Unique trade identifier (UUID).
            correlation_id: Pipeline run identifier linking to the decision.
            recommendation: The trade recommendation that spawned this trade.
            log_dir: Root directory for audit log output.
        """
        self.trade_id = trade_id
        self.correlation_id = correlation_id
        self.recommendation = recommendation
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.entries: list[dict[str, Any]] = []
        self._start_time = datetime.now(UTC)

    @property
    def audit_path(self) -> Path:
        """Path to the audit JSON file for this trade."""
        day_dir = self.log_dir / datetime.now(UTC).strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        return day_dir / f"trade-{self.trade_id}.json"

    def record_entry(self, event_type: str, **fields: Any) -> None:
        """Record a structured event entry in the audit trail.

        Args:
            event_type: Short label for the event (e.g. 'order_submitted').
            **fields: Arbitrary key-value data to record.
        """
        entry: dict[str, Any] = {
            "trade_id": self.trade_id,
            "correlation_id": self.correlation_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            **fields,
        }
        self.entries.append(entry)
        with open(self.audit_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        logger.debug(
            "trade_audit_recorded",
            trade_id=self.trade_id,
            event_type=event_type,
        )

    def record_monitoring_snapshot(
        self,
        current_price: float,
        current_pnl_pct: float,
        tp_level: float,
        sl_level: float,
        trailing_active: bool = False,
        trail_level: float | None = None,
    ) -> None:
        """Record a periodic PnL monitoring snapshot.

        Args:
            current_price: Current mark price of the option.
            current_pnl_pct: Unrealized PnL as a percentage of entry.
            tp_level: Current take-profit trigger level.
            sl_level: Current stop-loss trigger level.
            trailing_active: Whether trailing stop is currently active.
            trail_level: Current trailing stop level if active.
        """
        self.record_entry(
            "monitoring_snapshot",
            current_price=current_price,
            current_pnl_pct=current_pnl_pct,
            tp_level=tp_level,
            sl_level=sl_level,
            trailing_active=trailing_active,
            trail_level=trail_level,
        )

    def record_adjustment(self, reason: str, **details: Any) -> None:
        """Record a manual or automatic adjustment to exit parameters.

        Args:
            reason: Human-readable reason for the adjustment.
            **details: Arbitrary details about the adjustment.
        """
        self.record_entry("adjustment", reason=reason, **details)

    def finalize(
        self,
        exit_reason: str,
        exit_price: float,
        final_pnl: float,
        final_pnl_pct: float,
        lifecycle_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Write the final trade summary and close the audit trail.

        Args:
            exit_reason: How the trade exited (take_profit, stop_loss, etc.).
            exit_price: Final exit price per contract.
            final_pnl: Realized profit/loss in dollars.
            final_pnl_pct: Realized PnL as a percentage.
            lifecycle_events: Optional state machine event list.

        Returns:
            The complete trade summary dict.
        """
        end_time = datetime.now(UTC)
        duration = (end_time - self._start_time).total_seconds()

        summary: dict[str, Any] = {
            "trade_id": self.trade_id,
            "correlation_id": self.correlation_id,
            "asset": self.recommendation.asset,
            "direction": self.recommendation.direction.value,
            "contracts": self.recommendation.contracts,
            "entry_strike": self.recommendation.target_strike,
            "exit_reason": exit_reason,
            "exit_price": exit_price,
            "final_pnl": final_pnl,
            "final_pnl_pct": final_pnl_pct,
            "duration_seconds": round(duration, 2),
            "started_at": self._start_time.isoformat(),
            "ended_at": end_time.isoformat(),
            "events": lifecycle_events or [],
            "entries": list(self.entries),
            "recommendation": self.recommendation.model_dump(),
        }

        self._write_final_summary(summary)
        return summary

    def _write_final_summary(self, summary: dict[str, Any]) -> None:
        path = self.audit_path.with_name(f"trade-{self.trade_id}.json")
        if path.exists():
            path.rename(path.with_suffix(".json.bak"))
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(
            "trade_audit_finalized",
            trade_id=self.trade_id,
            exit_reason=summary["exit_reason"],
            final_pnl=summary["final_pnl"],
        )
