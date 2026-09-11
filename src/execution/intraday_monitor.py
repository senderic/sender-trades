"""Intraday stop-loss enforcement for 0DTE options trades.

The pipeline places only the take-profit at Alpaca and then exits; the
in-app ``ExitManager`` stop-loss / trailing / time-deadline logic is
therefore dead code within the pipeline process. Historically every losing
trade bled through to the once-daily 12:20 PM PT ``safety_close.sh`` sweep,
realizing an average ~-86% instead of the configured -50%.

This module restores the stop-loss as a standalone, stateless, idempotent
monitor run by cron during market hours. It reads pending trade audits from
``logs/<date>/trade-*.json``, polls the live option mark, and force-closes
any position whose mark has fallen to or below its recorded ``sl_level``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from datetime import time as dtime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from src.execution import exit_advisor
from src.execution.client import AlpacaBrokerClient
from src.json_utils import load_json_tolerant
from src.timezone import ET_TZ

if TYPE_CHECKING:
    from src.execution.models import ExecutionConfig

logger = structlog.get_logger()

# Market-hours guard: only act while the underlying + options are liquid.
MONITOR_OPEN = dtime(9, 30)
MONITOR_CLOSE = dtime(15, 25)

PENDING_REASONS = (None, "pending")

# Fill-confirmation polling for a close order (SL / advisor_exit /
# trailing_stop all go through the same close path). Bounded well under
# the 3-minute cron interval -- 5 attempts * 3s = 15s worst case.
CLOSE_FILL_POLL_ATTEMPTS = 5
CLOSE_FILL_POLL_INTERVAL_SEC = 3.0
# Each unfilled retry escalates (lower limit_mult = more aggressive,
# closer to/through the bid) up to this floor.
CLOSE_LIMIT_MULT_STEP = 0.1
CLOSE_LIMIT_MULT_FLOOR = 0.5


@dataclass
class OpenTrade:
    """A parsed, still-open trade extracted from an audit JSON file."""

    path: Path
    trade_id: str
    data: dict[str, Any]
    occ_symbol: str | None = None
    tp_level: float | None = None
    sl_level: float | None = None
    entry_price: float | None = None
    contracts: int = 1
    tp_order_id: str | None = None
    asset: str | None = None
    direction: str | None = None
    strategy: str | None = None


def load_audit_file(path: Path) -> dict[str, Any]:
    """Load a trade audit JSON file.

    ``TradeContext`` (see ``src.execution.context``) rewrites the whole
    file atomically on every event, so a current file is always one
    complete JSON object -- including while the trade is still open, not
    just after ``finalize()``. Some files predate that fix and were left
    as several JSON objects concatenated in one file, which plain
    ``json.load``/``json.loads`` cannot parse. See
    :func:`src.json_utils.load_json_tolerant` for how both shapes are
    handled.
    """
    return load_json_tolerant(path.read_text())


def extract_open_trade(data: dict[str, Any], path: Path) -> OpenTrade | None:
    """Parse a trade audit into an :class:`OpenTrade`, or None if resolved."""
    if data.get("exit_reason") not in PENDING_REASONS:
        return None

    trade = OpenTrade(path=path, trade_id=data.get("trade_id", ""), data=data)
    trade.asset = data.get("asset")
    trade.direction = data.get("direction")
    rec = data.get("recommendation") or {}
    trade.strategy = rec.get("strategy_label")
    trade.contracts = int(data.get("contracts") or rec.get("contracts") or 1)

    for entry in data.get("entries") or []:
        et = entry.get("event_type")
        if et == "entry_submitted" and trade.occ_symbol is None:
            trade.occ_symbol = entry.get("occ_symbol")
        elif et == "entry_filled" and trade.entry_price is None:
            ep = entry.get("fill_price") or entry.get("avg_price")
            if ep is not None:
                trade.entry_price = float(ep)
        elif et == "exits_placed":
            if trade.tp_level is None:
                trade.tp_level = entry.get("tp_level")
            if trade.sl_level is None:
                trade.sl_level = entry.get("sl_level")
            if trade.tp_order_id is None:
                trade.tp_order_id = entry.get("tp_order_id")

    if trade.occ_symbol is None:
        logger.warning("intraday_monitor_no_symbol", trade_id=trade.trade_id, path=str(path))
        return None
    if trade.sl_level is None:
        logger.warning("intraday_monitor_no_sl", trade_id=trade.trade_id, path=str(path))
        return None
    return trade


def should_trigger(mark: float, sl_level: float) -> bool:
    """Return True when the option mark has fallen to or below ``sl_level``."""
    return mark <= sl_level


def mark_from_quote(quote: dict[str, Any] | None) -> float | None:
    """Derive a mark price from an option snapshots quote dict."""
    if not quote:
        return None
    bid = quote.get("bid")
    ask = quote.get("ask")
    if bid and ask and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if ask and ask > 0:
        return float(ask)
    return None


def compute_pnl(
    entry_price: float | None, exit_price: float, contracts: int
) -> tuple[float, float]:
    """Return (pnl_dollars, pnl_pct) for an exit at ``exit_price``."""
    if not entry_price or entry_price <= 0:
        return 0.0, 0.0
    pnl = (exit_price - entry_price) * contracts * 100
    pnl_pct = (exit_price / entry_price - 1.0) * 100.0
    return round(pnl, 2), round(pnl_pct, 3)


def in_market_hours(now: datetime | None = None) -> bool:
    """Return True when ``now`` falls within the intraday monitor window."""
    current = now if now is not None else datetime.now(ET_TZ)
    return MONITOR_OPEN <= current.time() <= MONITOR_CLOSE


def write_result(
    trade: OpenTrade,
    exit_price: float,
    exit_reason: str,
    pnl: float,
    pnl_pct: float,
    sell_order_id: str | None = None,
    now: datetime | None = None,
) -> None:
    """Idempotently write the resolved exit into the audit file.

    Marks ``exit_reason`` so neither this monitor nor ``safety_close.sh``
    re-processes the trade, and persists so a crash between the sell and
    the write cannot double-sell.
    """
    data = dict(trade.data)
    data["exit_reason"] = exit_reason
    data["exit_price"] = exit_price
    data["final_pnl"] = pnl
    data["final_pnl_pct"] = pnl_pct
    now = now if now is not None else datetime.now(ET_TZ)
    data["ended_at"] = now.isoformat()

    events = data.get("entries") if isinstance(data.get("entries"), list) else None
    if events is None and isinstance(data.get("events"), list):
        events = data["events"]
    if isinstance(events, list):
        if sell_order_id:
            events.append(
                {
                    "event_type": "exit_filled",
                    "fill_price": exit_price,
                    "order_id": sell_order_id,
                    "timestamp": now.isoformat(),
                }
            )
        events.append(
            {
                "event_type": "lifecycle",
                "state": "CLOSED",
                "exit_reason": exit_reason,
                "timestamp": now.isoformat(),
            }
        )
        data["entries"] = events

    tmp = trade.path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, trade.path)


def write_close_attempt(
    trade: OpenTrade,
    attempted_exit_reason: str,
    attempted_limit_price: float,
    order_id: str | None,
    now: datetime | None = None,
) -> int:
    """Record an UNFILLED close attempt without resolving the trade.

    Deliberately leaves ``exit_reason`` untouched (still "pending" / not
    yet set) so the trade is still picked up as open by
    :func:`extract_open_trade` on the NEXT monitor run and the close is
    retried -- more aggressively, see :func:`_close_position` -- instead
    of the trade being silently orphaned: writing a terminal
    ``exit_reason`` for a sell that never actually filled would leave
    the real position open at Alpaca while the audit already reads as
    closed, with nothing left to notice or retry it.

    Returns:
        The new ``close_attempts`` count (used to escalate aggressiveness
        on the next retry).
    """
    data = dict(trade.data)
    attempts = int(data.get("close_attempts", 0) or 0) + 1
    data["close_attempts"] = attempts
    now = now if now is not None else datetime.now(ET_TZ)

    events = data.get("entries") if isinstance(data.get("entries"), list) else None
    if events is None and isinstance(data.get("events"), list):
        events = data["events"]
    entry = {
        "event_type": "close_attempt_unfilled",
        "attempted_exit_reason": attempted_exit_reason,
        "attempted_limit_price": attempted_limit_price,
        "order_id": order_id,
        "attempt_number": attempts,
        "timestamp": now.isoformat(),
    }
    if isinstance(events, list):
        events.append(entry)
        data["entries"] = events

    trade.data = data
    tmp = trade.path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, trade.path)
    return attempts


async def _poll_fill(client: AlpacaBrokerClient, order_id: str) -> Any | None:
    """Poll an order until it fills or reaches a terminal non-fill state.

    Returns the filled :class:`OrderResult` only when ``status ==
    "filled"``. Returns ``None`` for anything else (still open,
    partially filled, rejected, cancelled, or unreachable) -- callers
    must treat all of those as "did not resolve the position" and never
    write a terminal ``exit_reason``. Partially-filled orders are
    deliberately treated the same as unfilled here rather than closing
    out a partial position: a true partial-fill reconciliation (residual
    qty accounting) is out of scope for this monitor and would risk
    over-selling on the next retry.
    """
    for _ in range(CLOSE_FILL_POLL_ATTEMPTS):
        try:
            order = await client.get_order(order_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("intraday_monitor_poll_fill_error", order_id=order_id, error=str(exc))
            return None
        if order.status == "filled":
            return order
        if order.status in ("rejected", "canceled", "expired"):
            return None
        await asyncio.sleep(CLOSE_FILL_POLL_INTERVAL_SEC)
    return None


async def _close_position(
    trade: OpenTrade,
    client: AlpacaBrokerClient,
    mark: float,
    exit_reason: str,
    limit_mult: float,
    now: datetime | None,
) -> dict[str, Any] | None:
    """Cancel any resting TP, sell at an aggressive limit, and CONFIRM the fill.

    Shared by the stop-loss hard rail and the profit-exit advisor path --
    both close a position the same way, they differ only in why. A 0DTE
    bid/ask spread can be wider than ``limit_mult``'s cushion, so the
    sell is not assumed to have filled just because it was submitted:
    this polls for an actual fill (see :func:`_poll_fill`) before ever
    writing a terminal ``exit_reason``. If it doesn't fill, the order is
    cancelled, the attempt is recorded via :func:`write_close_attempt`
    (which leaves the trade "pending"), and the next monitor run retries
    with a more aggressive limit (``limit_mult`` steps down by
    ``CLOSE_LIMIT_MULT_STEP`` per prior attempt, floored at
    ``CLOSE_LIMIT_MULT_FLOOR``).

    Returns:
        The closed-trade summary dict when the sell actually filled, or
        ``None`` when it did not -- callers must NOT treat ``None`` as a
        failure to be ignored; the trade is still open and will be
        retried on the next pass.
    """
    if trade.tp_order_id:
        try:
            await client.cancel_order(trade.tp_order_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "intraday_monitor_cancel_tp_failed",
                trade_id=trade.trade_id,
                tp_order_id=trade.tp_order_id,
                error=str(exc),
            )

    prior_attempts = int(trade.data.get("close_attempts", 0) or 0)
    effective_mult = max(
        limit_mult - CLOSE_LIMIT_MULT_STEP * prior_attempts, CLOSE_LIMIT_MULT_FLOOR
    )
    limit_price = round(max(mark * effective_mult, 0.01), 2)
    sell = await client.submit_order(
        {
            "symbol": trade.occ_symbol,
            "qty": trade.contracts,
            "side": "sell",
            "type": "limit",
            "limit_price": limit_price,
            "time_in_force": "day",
        }
    )
    sell_order_id = getattr(sell, "order_id", None)

    filled_order = await _poll_fill(client, sell_order_id) if sell_order_id else None
    if filled_order is None:
        if sell_order_id:
            try:
                await client.cancel_order(sell_order_id)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "intraday_monitor_cancel_unfilled_close_failed",
                    trade_id=trade.trade_id,
                    order_id=sell_order_id,
                    error=str(exc),
                )
        attempts = write_close_attempt(trade, exit_reason, limit_price, sell_order_id, now)
        logger.warning(
            "intraday_monitor_close_unfilled",
            trade_id=trade.trade_id,
            exit_reason=exit_reason,
            attempted_limit_price=limit_price,
            attempt=attempts,
        )
        return None

    exit_price = float(filled_order.filled_avg_price) if filled_order.filled_avg_price else mark
    pnl, pnl_pct = compute_pnl(trade.entry_price, exit_price, trade.contracts)
    write_result(
        trade,
        exit_price=exit_price,
        exit_reason=exit_reason,
        pnl=pnl,
        pnl_pct=pnl_pct,
        sell_order_id=filled_order.order_id,
        now=now,
    )
    logger.info(
        "intraday_monitor_closed",
        trade_id=trade.trade_id,
        exit_price=exit_price,
        exit_reason=exit_reason,
        pnl=pnl,
        pnl_pct=pnl_pct,
    )
    return {
        "trade_id": trade.trade_id,
        "asset": trade.asset,
        "direction": trade.direction,
        "symbol": trade.occ_symbol,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "pnl": pnl,
        "succeeded": True,
    }


async def monitor_trade(
    trade: OpenTrade,
    client: AlpacaBrokerClient,
    now: datetime | None = None,
    exec_config: ExecutionConfig | None = None,
    llm_primary_model: str = "",
) -> dict[str, Any] | None:
    """Poll one open trade's mark and close it if a rail (or the advisor) fires.

    Hard rail FIRST, always, unconditionally: the -50% stop-loss below is
    evaluated and can close the trade before anything else runs. Only
    once it does NOT fire does the (optional) profit-exit advisor get a
    turn -- see ``src.execution.exit_advisor``, which manages only the
    profit side of the exit and can never touch the stop-loss.

    Returns an info dict when triggered, else None.
    """
    if trade.occ_symbol is None or trade.sl_level is None:
        return None

    quote = await client.get_option_quote(trade.occ_symbol)
    mark = mark_from_quote(quote)
    if mark is None:
        logger.debug("intraday_monitor_no_quote", trade_id=trade.trade_id, symbol=trade.occ_symbol)
        return None

    # Hard rail: stop-loss. Deterministic, always first, cannot be
    # overridden or delayed by the advisor below.
    if should_trigger(mark, trade.sl_level):
        logger.info(
            "intraday_monitor_sl_triggered",
            trade_id=trade.trade_id,
            asset=trade.asset,
            direction=trade.direction,
            symbol=trade.occ_symbol,
            mark=round(mark, 2),
            sl_level=trade.sl_level,
        )
        return await _close_position(trade, client, mark, "stop_loss", 0.7, now)

    if exec_config is not None and exec_config.exit_advisor.enabled:
        from src.execution.engine import ExecutionEngine  # deferred, avoids import-time coupling

        underlying = await client.get_underlying_quote(trade.asset or "")
        underlying_spot = ExecutionEngine._extract_spot(underlying)

        decision = await exit_advisor.process(
            trade,
            mark,
            quote,
            underlying_spot,
            exec_config,
            llm_primary_model,
            now=now,
        )
        if decision is not None and decision.should_exit:
            return await _close_position(
                trade, client, mark, decision.exit_reason, decision.limit_mult, now
            )

    return None


def find_open_trades(log_dir: str | Path, trade_date: str | None = None) -> list[OpenTrade]:
    """Scan the day's audit files for still-open trades."""
    root = Path(log_dir).expanduser().resolve()
    target_date = trade_date or datetime.now(ET_TZ).strftime("%Y-%m-%d")
    day_dir = root / target_date
    if not day_dir.is_dir():
        return []

    open_trades: list[OpenTrade] = []
    for fname in sorted(day_dir.iterdir()):
        if not fname.name.startswith("trade-") or not fname.name.endswith(".json"):
            continue
        if fname.name.endswith(".bak") or fname.name.endswith(".tmp"):
            continue
        try:
            data = load_audit_file(fname)
        except Exception as exc:
            logger.debug("intraday_monitor_read_error", path=str(fname), error=str(exc))
            continue
        trade = extract_open_trade(data, fname)
        if trade is not None:
            open_trades.append(trade)
    return open_trades


def _load_exec_config(config_path: str) -> tuple[ExecutionConfig, str]:
    """Load ``execution:`` config + ``llm.primary_model`` for this run.

    Never raises: a broken/missing config file must not break the
    stop-loss rail, which does not depend on this at all -- only the
    optional profit-exit advisor does. Falls back to defaults
    (``exit_advisor.enabled = False``) on any failure.
    """
    from src.config import Settings  # deferred, see module import notes elsewhere in this package

    try:
        settings = Settings.from_yaml(config_path).resolve_env_vars()
        return settings.execution, settings.llm.primary_model
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("intraday_monitor_config_load_failed", path=config_path, error=str(exc))
        from src.execution.models import ExecutionConfig as _ExecutionConfig

        return _ExecutionConfig(), "opencode/muse-spark-1.3-contributor-free"


async def run(
    log_dir: str | Path = "logs",
    trade_date: str | None = None,
    config_path: str = "config.yaml",
) -> dict[str, Any]:
    """Run one pass of the intraday stop-loss monitor."""
    now = datetime.now(ET_TZ)
    result: dict[str, Any] = {
        "ran_at": now.isoformat(),
        "in_market_hours": in_market_hours(now),
        "open_trades_found": 0,
        "closed_trades": [],
        "action_taken": False,
    }

    if not in_market_hours(now):
        result["skipped"] = "outside_market_hours"
        return result

    trades = find_open_trades(log_dir, trade_date)
    result["open_trades_found"] = len(trades)
    if not trades:
        return result

    paper = os.environ.get("PAPER_MODE", "1") != "0"
    api_key = os.environ.get("APCA_API_KEY_ID", "")
    secret_key = os.environ.get("APCA_API_SECRET_KEY", "")
    if not api_key or not secret_key:
        result["skipped"] = "no_alpaca_keys"
        return result

    exec_config, llm_primary_model = _load_exec_config(config_path)

    client = AlpacaBrokerClient(api_key, secret_key, paper=paper)
    try:
        for trade in trades:
            outcome = await monitor_trade(
                trade,
                client,
                now=now,
                exec_config=exec_config,
                llm_primary_model=llm_primary_model,
            )
            if outcome:
                result["closed_trades"].append(outcome)
                result["action_taken"] = True
    finally:
        if client._http is not None:
            await client._http.aclose()
        if client._data_http is not None:
            await client._data_http.aclose()

    return result


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Intraday 0DTE stop-loss monitor")
    parser.add_argument("--log-dir", default=os.environ.get("LOG_DIR", "logs"))
    parser.add_argument("--date", default=None, help="Override trade date (YYYY-MM-DD)")
    parser.add_argument(
        "--config", default=os.environ.get("CONFIG_PATH", "config.yaml"), help="Path to config YAML"
    )
    args = parser.parse_args(argv)

    result = await run(log_dir=args.log_dir, trade_date=args.date, config_path=args.config)
    logger.info("intraday_monitor_run_complete", **result)
    return 0


if __name__ == "__main__":
    sys.exit(__import__("asyncio").run(main()))
