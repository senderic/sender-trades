"""Trade execution engine — orchestrates the full lifecycle of a 0DTE trade."""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog

from src.execution.client import AlpacaBrokerClient
from src.execution.context import TradeContext
from src.execution.exit_manager import ExitManager
from src.execution.lifecycle import InvalidTransitionError, TradeLifecycle
from src.execution.models import ExecutionConfig, OrderResult, TradeState
from src.mcp.schemas import occ_option_symbol
from src.models.recommendation import TradeRecommendation
from src.timezone import ET_TZ

# src.config and src.engine.decision are imported lazily (inside __init__ /
# execute()) rather than at module level: src.config imports
# src.execution.models for ExecutionConfig, which triggers this package's
# __init__.py, which imports this very module -- a top-level `from
# src.config import ...` here would try to read attributes off src.config
# while it is still mid-import and raise ImportError. TYPE_CHECKING keeps
# the type hints working for static analysis without re-introducing the
# cycle at runtime.
if TYPE_CHECKING:
    from src.config import RiskConfig

logger = structlog.get_logger()


def _safe_error(exception: BaseException) -> dict[str, Any]:
    """Parse an exception into a safe, serializable dict."""
    raw = str(exception)
    try:
        return json.loads(raw) if isinstance(raw, str) and raw.startswith("{") else {"message": raw}
    except json.JSONDecodeError:
        return {"message": raw}


class ExecutionEngineError(Exception):
    """Non-recoverable error during trade execution."""

    def __init__(self, message: str, trade_id: str = "") -> None:
        super().__init__(message)
        self.trade_id = trade_id


class ExecutionEngine:
    """Orchestrates the full lifecycle of a 0DTE options trade.

    From entry submission through exit management to final close,
    every state transition and API action is retried (via Tenacity)
    and logged to the structured audit trail.
    """

    def __init__(
        self,
        client: AlpacaBrokerClient,
        exec_config: ExecutionConfig,
        log_dir: str | None = None,
        risk_config: RiskConfig | None = None,
    ):
        """Initialize the execution engine.

        Args:
            client: Configured AlpacaBrokerClient.
            exec_config: Execution configuration (entry, exit, retry params).
            log_dir: Root directory for audit logs (defaults to ``logs/``).
            risk_config: Risk guardrail config, used for the premium gate
                (see :meth:`execute`). Defaults to ``RiskConfig()`` when
                omitted so existing callers/tests keep working.
        """
        self.client = client
        self.exec_config = exec_config
        self.log_dir = log_dir or "logs"
        if risk_config is None:
            from src.config import RiskConfig as _RiskConfig  # deferred, see module import note

            risk_config = _RiskConfig()
        self.risk_config = risk_config
        self._monitor_interval: float = 30.0
        self._wait_for_option_open: bool = True
        self._option_open_wait_cap_sec: float = 240.0
        self._option_open_buffer_sec: float = 5.0

    @staticmethod
    def _extract_spot(underlying: dict[str, Any] | None) -> float | None:
        """Coalesce an underlying quote dict into a single spot price.

        Prefers the last trade price, then the bid/ask midpoint, then the
        ask alone. Shared by the pre-open entry-pricing fallback (when no
        option quote is available yet) and ``DecisionAggregator.premium_gate``
        (which needs a live spot to compute the OTM-distance-aware
        breakeven -- see ``execute()``).
        """
        if not underlying:
            return None
        last = underlying.get("last")
        bid = underlying.get("bid")
        ask = underlying.get("ask")
        if last and last > 0:
            return float(last)
        if bid and ask and bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        if ask and ask > 0:
            return float(ask)
        return None

    async def _await_option_quote(self, occ_symbol: str) -> dict[str, Any] | None:
        """Fetch a live option quote, waiting for the market open if needed.

        Options only trade during RTH (9:30 AM ET), so a quote requested at
        the 9:28 AM ET submission window is unavailable. This retries once
        the market opens so the entry limit is priced off the live option
        ask rather than the underlying-spot fallback when possible.

        Args:
            occ_symbol: OCC option symbol to quote.

        Returns:
            The option quote dict (bid/ask), or None if still unavailable.
        """
        quote = await self.client.get_option_quote(occ_symbol)
        if quote and quote.get("ask"):
            return quote

        if not self._wait_for_option_open:
            return quote

        now = datetime.now(ET_TZ)
        open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
        seconds_to_open = (open_time - now).total_seconds()
        if 0 < seconds_to_open <= self._option_open_wait_cap_sec:
            await asyncio.sleep(seconds_to_open + self._option_open_buffer_sec)
            quote = await self.client.get_option_quote(occ_symbol)
            if quote and quote.get("ask"):
                return quote

        return quote

    async def execute(self, rec: TradeRecommendation, correlation_id: str) -> dict[str, Any]:
        """Execute a trade recommendation through the full lifecycle.

        Args:
            rec: The trade recommendation to execute.
            correlation_id: Pipeline correlation ID for traceability.

        Returns:
            Dict with trade summary (trade_id, result, pnl, etc.).
        """
        trade_id = uuid.uuid4().hex[:12]
        lifecycle = TradeLifecycle(trade_id)
        ctx = TradeContext(
            trade_id,
            correlation_id,
            rec,
            log_dir=self.log_dir,
            execution_config=self.exec_config.model_dump(),
        )

        try:
            lifecycle.transition(
                TradeState.VALIDATING, {"asset": rec.asset, "direction": rec.direction.value}
            )
            lifecycle.transition(TradeState.SUBMITTED)

            option_type = "C" if rec.direction.value == "CALL" else "P"
            expiry = rec.expires_at
            occ_sym = occ_option_symbol(rec.asset, expiry, rec.target_strike, option_type)
            quote = await self._await_option_quote(occ_sym)
            spot = None
            if not quote or not quote.get("ask"):
                underlying = await self.client.get_underlying_quote(rec.asset)
                spot = self._extract_spot(underlying)

            if quote and quote.get("ask"):
                # Premium is only known now (options don't quote pre-market),
                # so the breakeven gate runs here rather than at decision
                # time. Fetch a live underlying quote specifically for the
                # gate -- it's not the same `spot` fetched above, which
                # only happens when there's NO option ask (mutually
                # exclusive with this branch). See
                # DecisionAggregator.premium_gate for the OTM-distance-aware
                # breakeven calculation; when the underlying quote is
                # unavailable, premium_gate falls back to a less strict
                # ask/strike approximation and logs that it did.
                gate_underlying = await self.client.get_underlying_quote(rec.asset)
                gate_spot = self._extract_spot(gate_underlying)

                from src.engine.decision import DecisionAggregator  # deferred, see import note

                premium_reason = DecisionAggregator.premium_gate(
                    rec, quote.get("ask"), gate_spot, self.risk_config
                )
                if premium_reason:
                    logger.warning(
                        "premium_gate_blocked",
                        trade_id=trade_id,
                        asset=rec.asset,
                        direction=rec.direction.value,
                        reason=premium_reason,
                    )
                    ctx.record_entry(
                        "premium_gate_blocked",
                        reason=premium_reason,
                        ask=quote.get("ask"),
                        underlying=gate_spot,
                    )
                    lifecycle.transition(TradeState.REJECTED, {"reason": "premium_gate"})
                    return ctx.finalize(
                        exit_reason="premium_gate_blocked",
                        exit_price=0.0,
                        final_pnl=0.0,
                        final_pnl_pct=0.0,
                        lifecycle_events=lifecycle.event_summary(),
                    )

            order_data = self.client.build_entry_order(
                rec, occ_symbol=occ_sym, quote=quote, spot=spot
            )

            ctx.record_entry(
                "entry_submitted",
                occ_symbol=occ_sym,
                contracts=rec.contracts,
                order_type=order_data["type"],
            )

            entry_result = await self.client.submit_order(order_data)
            ctx.record_entry(
                "entry_order_response",
                order_id=entry_result.order_id,
                status=entry_result.status,
            )

            if entry_result.status == "rejected":
                lifecycle.transition(TradeState.REJECTED, {"order_id": entry_result.order_id})
                return ctx.finalize(
                    exit_reason="rejected",
                    exit_price=0.0,
                    final_pnl=0.0,
                    final_pnl_pct=0.0,
                    lifecycle_events=lifecycle.event_summary(),
                )
            if entry_result.status == "expired":
                lifecycle.transition(TradeState.EXPIRED, {"order_id": entry_result.order_id})
                return ctx.finalize(
                    exit_reason="expired",
                    exit_price=0.0,
                    final_pnl=0.0,
                    final_pnl_pct=0.0,
                    lifecycle_events=lifecycle.event_summary(),
                )

            lifecycle.transition(TradeState.ACKNOWLEDGED, {"order_id": entry_result.order_id})

            fill_result = await self._wait_for_fill(
                entry_result.order_id,
                timeout_minutes=self.exec_config.entry.entry_window_minutes,
            )

            if fill_result is None:
                await self.client.cancel_order(entry_result.order_id)
                lifecycle.transition(TradeState.EXPIRED, {"order_id": entry_result.order_id})
                return ctx.finalize(
                    exit_reason="expired",
                    exit_price=0.0,
                    final_pnl=0.0,
                    final_pnl_pct=0.0,
                    lifecycle_events=lifecycle.event_summary(),
                )

            filled_qty = int(fill_result.filled_qty) if fill_result.filled_qty else 0
            if filled_qty < rec.contracts:
                lifecycle.transition(
                    TradeState.PARTIALLY_FILLED,
                    {"filled_qty": filled_qty, "ordered_qty": rec.contracts},
                )

            entry_price = (
                float(fill_result.filled_avg_price) if fill_result.filled_avg_price else 0.0
            )
            lifecycle.transition(
                TradeState.FILLED,
                {"filled_qty": filled_qty, "avg_price": entry_price},
            )
            ctx.record_entry(
                "entry_filled",
                order_id=fill_result.order_id,
                filled_qty=filled_qty,
                avg_price=entry_price,
            )

            exit_mgr = ExitManager(self.exec_config.exit_strategy)
            exit_mgr.on_entry_filled(entry_price)

            tp_spec = exit_mgr.build_tp_order(occ_sym, filled_qty)
            tp_result = await self.client.submit_order(tp_spec)
            lifecycle.transition(
                TradeState.EXITS_PLACED,
                {"tp_order_id": tp_result.order_id, "exit_via_cron": True},
            )
            ctx.record_entry(
                "exits_placed",
                tp_order_id=tp_result.order_id,
                tp_level=exit_mgr.tp_level,
                sl_level=exit_mgr.sl_level,
                note="TP placed at Alpaca. SL enforced by intraday monitor cron (~3 min, 9:30-15:25 ET); safety-close sweep (12:20 PM PT) is the final backstop.",
            )

            lifecycle.transition(TradeState.CLOSED)
            return ctx.finalize(
                exit_reason="pending",
                exit_price=0.0,
                final_pnl=0.0,
                final_pnl_pct=0.0,
                lifecycle_events=lifecycle.event_summary(),
            )

        except InvalidTransitionError as e:
            logger.error("execution_invalid_transition", trade_id=trade_id, error=str(e))
            with contextlib.suppress(InvalidTransitionError):
                lifecycle.transition(TradeState.FAILED, {"error": str(e)})
            return ctx.finalize(
                exit_reason="error",
                exit_price=0.0,
                final_pnl=0.0,
                final_pnl_pct=0.0,
                lifecycle_events=lifecycle.event_summary(),
            )
        except Exception as e:
            logger.error("execution_error", trade_id=trade_id, error=str(e))
            with contextlib.suppress(InvalidTransitionError):
                lifecycle.transition(TradeState.FAILED, {"error": _safe_error(e)})
            ctx.record_entry("execution_error", error=str(e), error_detail=_safe_error(e))
            return ctx.finalize(
                exit_reason="error",
                exit_price=0.0,
                final_pnl=0.0,
                final_pnl_pct=0.0,
                lifecycle_events=lifecycle.event_summary(),
            )

    async def _wait_for_fill(self, order_id: str, timeout_minutes: int = 5) -> OrderResult | None:
        """Poll an order until it fills, rejects, expires, or times out.

        Args:
            order_id: Alpaca order ID to poll.
            timeout_minutes: Maximum time to wait for a fill.

        Returns:
            The filled :class:`OrderResult`, or None if expired/timeout.
        """
        deadline = datetime.now(ET_TZ).timestamp() + timeout_minutes * 60
        terminal_states = {"filled", "partially_filled", "rejected", "canceled", "expired"}
        sleep_sec = 2.0

        while datetime.now(ET_TZ).timestamp() < deadline:
            result = await self.client.get_order(order_id)
            if result.status in terminal_states:
                if result.status in ("filled", "partially_filled"):
                    return result
                return None
            await asyncio.sleep(sleep_sec)
            sleep_sec = min(sleep_sec * 1.5, 10.0)

        return None

    async def _monitor_exits(
        self,
        exit_mgr: ExitManager,
        tp_order_id: str,
        occ_symbol: str,
        contracts: int,
        lifecycle: TradeLifecycle,
        ctx: TradeContext,
    ) -> dict[str, Any]:
        """Monitor exit orders until one fills or time expires.

        Polls quote data and evaluates exit conditions, adjusting
        trailing stops as needed. Only a TP limit order is placed
        at Alpaca; SL and trailing stops are managed in-app by
        submitting a market sell when triggered.

        Args:
            exit_mgr: Configured exit manager.
            tp_order_id: Take-profit order ID at Alpaca.
            occ_symbol: OCC option symbol.
            contracts: Number of contracts.
            lifecycle: Trade lifecycle tracker.
            ctx: Trade audit context.

        Returns:
            Finalized trade summary dict.
        """
        while not lifecycle.is_terminal:
            if exit_mgr.is_time_deadline_approaching():
                logger.info("monitor_time_deadline", trade_id=ctx.trade_id)
                await self.client.cancel_order(tp_order_id)
                from uuid import uuid4

                close_spec = exit_mgr.build_market_close_order(
                    occ_symbol, contracts, client_order_id=uuid4().hex[:12]
                )
                await self.client.submit_order(close_spec)
                lifecycle.transition(TradeState.FORCE_CLOSED)
                lifecycle.transition(TradeState.CLOSED)
                return ctx.finalize(
                    exit_reason="force_close",
                    exit_price=0.0,
                    final_pnl=0.0,
                    final_pnl_pct=0.0,
                    lifecycle_events=lifecycle.event_summary(),
                )

            quote = await self.client.get_option_quote(occ_symbol)
            current_price: float | None = None

            if quote:
                bid = quote.get("bid") or 0
                ask = quote.get("ask") or 0
                if bid > 0 and ask > 0:
                    current_price = (bid + ask) / 2.0
                elif ask > 0:
                    current_price = ask

            if current_price is None:
                tp_order = await self.client.get_order(tp_order_id)
                if tp_order.status in ("filled", "canceled", "expired", "rejected"):
                    return await self._resolve_exit(
                        tp_order_id, exit_mgr, lifecycle, ctx, occ_symbol, contracts
                    )
                await asyncio.sleep(self._monitor_interval)
                continue

            evaluation = exit_mgr.evaluate(current_price)
            ctx.record_monitoring_snapshot(
                current_price=current_price,
                current_pnl_pct=evaluation["current_pnl_pct"],
                tp_level=evaluation["tp_level"] or 0,
                sl_level=evaluation["sl_level"] or 0,
                trailing_active=evaluation["trailing_active"],
                trail_level=evaluation.get("trail_level"),
            )

            if evaluation["triggered"]:
                trigger = evaluation["trigger_type"]
                await self.client.cancel_order(tp_order_id)

                if trigger == "take_profit":
                    lifecycle.transition(TradeState.TP_FILLED)
                    lifecycle.transition(TradeState.CLOSED)
                    return ctx.finalize(
                        exit_reason="take_profit",
                        exit_price=current_price,
                        final_pnl=(current_price - (exit_mgr.entry_price or 0)) * contracts * 100,
                        final_pnl_pct=evaluation["current_pnl_pct"],
                        lifecycle_events=lifecycle.event_summary(),
                    )

                from uuid import uuid4

                close_spec = exit_mgr.build_market_close_order(
                    occ_symbol, contracts, client_order_id=uuid4().hex[:12]
                )
                await self.client.submit_order(close_spec)

                if trigger == "time_deadline":
                    lifecycle.transition(TradeState.FORCE_CLOSED)
                else:
                    lifecycle.transition(TradeState.SL_FILLED)
                lifecycle.transition(TradeState.CLOSED)
                return ctx.finalize(
                    exit_reason="force_close" if trigger == "time_deadline" else "stop_loss",
                    exit_price=current_price,
                    final_pnl=(current_price - (exit_mgr.entry_price or 0)) * contracts * 100,
                    final_pnl_pct=evaluation["current_pnl_pct"],
                    lifecycle_events=lifecycle.event_summary(),
                )

            if exit_mgr.trailing_active:
                ctx.record_adjustment(
                    "trailing_check",
                    trail_level=exit_mgr.trail_level,
                    peak_pnl_pct=evaluation["current_pnl_pct"],
                )

            await asyncio.sleep(self._monitor_interval)

        return ctx.finalize(
            exit_reason="error",
            exit_price=0.0,
            final_pnl=0.0,
            final_pnl_pct=0.0,
            lifecycle_events=lifecycle.event_summary(),
        )

    async def _resolve_exit(
        self,
        tp_order_id: str,
        exit_mgr: ExitManager,
        lifecycle: TradeLifecycle,
        ctx: TradeContext,
        occ_symbol: str,
        contracts: int,
    ) -> dict[str, Any]:
        tp = await self.client.get_order(tp_order_id)

        if tp.status == "filled":
            lifecycle.transition(TradeState.TP_FILLED)
            lifecycle.transition(TradeState.CLOSED)
            fill_price = float(tp.filled_avg_price or 0)
            return ctx.finalize(
                exit_reason="take_profit",
                exit_price=fill_price,
                final_pnl=(fill_price - (exit_mgr.entry_price or 0))
                * (int(tp.filled_qty) if tp.filled_qty else 0)
                * 100,
                final_pnl_pct=(
                    (fill_price - (exit_mgr.entry_price or 0)) / (exit_mgr.entry_price or 1) * 100
                ),
                lifecycle_events=lifecycle.event_summary(),
            )

        lifecycle.transition(TradeState.FAILED)
        return ctx.finalize(
            exit_reason="error",
            exit_price=0.0,
            final_pnl=0.0,
            final_pnl_pct=0.0,
            lifecycle_events=lifecycle.event_summary(),
        )
