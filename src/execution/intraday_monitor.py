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
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from datetime import time as dtime
from pathlib import Path
from typing import Any

import structlog

from src.execution.client import AlpacaBrokerClient
from src.json_utils import load_json_tolerant
from src.timezone import ET_TZ

logger = structlog.get_logger()

# Market-hours guard: only act while the underlying + options are liquid.
MONITOR_OPEN = dtime(9, 30)
MONITOR_CLOSE = dtime(15, 25)

PENDING_REASONS = (None, "pending")


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


async def monitor_trade(
    trade: OpenTrade,
    client: AlpacaBrokerClient,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Poll one open trade's mark and close it if the stop is breached.

    Returns an info dict when triggered, else None.
    """
    if trade.occ_symbol is None or trade.sl_level is None:
        return None

    quote = await client.get_option_quote(trade.occ_symbol)
    mark = mark_from_quote(quote)
    if mark is None:
        logger.debug("intraday_monitor_no_quote", trade_id=trade.trade_id, symbol=trade.occ_symbol)
        return None

    if not should_trigger(mark, trade.sl_level):
        return None

    logger.info(
        "intraday_monitor_sl_triggered",
        trade_id=trade.trade_id,
        asset=trade.asset,
        direction=trade.direction,
        symbol=trade.occ_symbol,
        mark=round(mark, 2),
        sl_level=trade.sl_level,
    )

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

    limit_price = round(max(mark * 0.7, 0.01), 2)
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

    exit_price = mark
    pnl, pnl_pct = compute_pnl(trade.entry_price, mark, trade.contracts)
    write_result(
        trade,
        exit_price=exit_price,
        exit_reason="stop_loss",
        pnl=pnl,
        pnl_pct=pnl_pct,
        sell_order_id=getattr(sell, "order_id", None),
        now=now,
    )
    logger.info(
        "intraday_monitor_closed",
        trade_id=trade.trade_id,
        exit_price=exit_price,
        exit_reason="stop_loss",
        pnl=pnl,
        pnl_pct=pnl_pct,
    )
    return {
        "trade_id": trade.trade_id,
        "asset": trade.asset,
        "direction": trade.direction,
        "symbol": trade.occ_symbol,
        "exit_price": exit_price,
        "pnl": pnl,
        "succeeded": True,
    }


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


async def run(log_dir: str | Path = "logs", trade_date: str | None = None) -> dict[str, Any]:
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

    client = AlpacaBrokerClient(api_key, secret_key, paper=paper)
    try:
        for trade in trades:
            outcome = await monitor_trade(trade, client, now=now)
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
    args = parser.parse_args(argv)

    result = await run(log_dir=args.log_dir, trade_date=args.date)
    logger.info("intraday_monitor_run_complete", **result)
    return 0


if __name__ == "__main__":
    sys.exit(__import__("asyncio").run(main()))
