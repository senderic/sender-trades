#!/bin/bash
# safety-close.sh — force-close open Alpaca paper positions before 0DTE expiry,
# then reconcile every pending trade audit against Alpaca order history so the
# learning loop records BOTH wins and losses (not just force-closes).
# Runs at 12:20 PM PT / 3:20 PM ET, 5 min before hard deadline (15:25).

set -u

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"

PROJECT_ENV="$DIR/.env"
if [ -f "$PROJECT_ENV" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$PROJECT_ENV"
  set +a
fi

cd "$DIR" || exit 1

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="$DIR/logs/cron/safety-close-$TIMESTAMP.log"
mkdir -p "$DIR/logs/cron"

# Serialize with the intraday monitor so the two writers cannot race on the
# same audit JSON (the monitor may fire the SL while this sweep reconciles).
(
  flock -n 200 || exit 0
  uv run python -c "
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, QueryOrderStatus
import structlog, os, sys, json, pytz, time
import datetime as _dt
from src.json_utils import load_json_tolerant

logger = structlog.get_logger()

key = os.environ.get('APCA_API_KEY_ID', '')
secret = os.environ.get('APCA_API_SECRET_KEY', '')
if not key:
    print('No Alpaca keys — skipping')
    sys.exit(0)

tc = TradingClient(key, secret, paper=True)
la_tz = pytz.timezone('America/Los_Angeles')
today_str = _dt.datetime.now(la_tz).strftime('%Y-%m-%d')
LOGS_DIR = os.path.join(os.getcwd(), 'logs')

# ---------------------------------------------------------------------------
# Phase 1: force-close any still-open positions.
# ---------------------------------------------------------------------------
positions = tc.get_all_positions()
if positions:
    logger.info('safety_close_start', count=len(positions))
    for p in positions:
        # Cancel any open orders for this symbol FIRST
        for o in tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[str(p.symbol)])):
            status = str(o.status)
            if str(o.symbol) == str(p.symbol) and status not in ('filled', 'canceled', 'expired', 'rejected', 'done_for_day'):
                tc.cancel_order_by_id(str(o.id))
                logger.info('safety_close_cancelled_order', symbol=p.symbol, order_id=str(o.id))

        # Close via LIMIT sell at an aggressive price (marketable but avoids
        # the 'uncovered option contracts' rejection market orders hit).
        current_price = float(p.current_price) if p.current_price and float(p.current_price) > 0 else 0.05
        limit_price = round(max(current_price * 0.7, 0.01), 2)
        try:
            req = LimitOrderRequest(
                symbol=p.symbol, qty=int(p.qty),
                side=OrderSide.SELL, type=OrderType.LIMIT,
                limit_price=limit_price,
                time_in_force=TimeInForce.DAY,
            )
            result = tc.submit_order(req)
            logger.info('safety_close_sold', symbol=p.symbol, qty=p.qty,
                         limit_price=req.limit_price, order_id=str(result.id),
                         pnl_pct=str(p.unrealized_plpc))
        except Exception as e:
            logger.error('safety_close_failed', symbol=p.symbol, error=str(e))
    # Give the aggressive limit sells a moment to fill before reconciling.
    time.sleep(8)
else:
    logger.info('safety_close_no_positions', count=0)

# ---------------------------------------------------------------------------
# Phase 2: reconcile EVERY pending trade audit against Alpaca order history.
# Runs unconditionally so take-profit wins (already closed during the day) are
# also recorded, not just the force-close losses.
# ---------------------------------------------------------------------------
log_root = LOGS_DIR
if not os.path.isdir(log_root):
    sys.exit(0)

for day_dir in sorted(os.listdir(log_root)):
    day_path = os.path.join(log_root, day_dir)
    if not os.path.isdir(day_path) or not day_dir.startswith('20'):
        continue
    for fname in sorted(os.listdir(day_path)):
        if not fname.startswith('trade-') or not fname.endswith('.json') or fname.endswith('.bak'):
            continue
        fpath = os.path.join(day_path, fname)
        try:
            with open(fpath, 'r') as fh:
                # Tolerates both a normal single-JSON-object file and a
                # legacy file left as several concatenated JSON objects
                # by a trade that never reached finalize() (see
                # src.json_utils and src.execution.context.TradeContext).
                trade_data = load_json_tolerant(fh.read())
        except Exception:
            continue
        if not trade_data:
            continue

        exit_reason = trade_data.get('exit_reason')
        if exit_reason not in (None, 'pending'):
            continue

        occ_symbol = None
        tp_level = None
        for entry in trade_data.get('entries', trade_data.get('events', [])):
            et = entry.get('event_type')
            if et == 'entry_submitted' and not occ_symbol:
                occ_symbol = entry.get('occ_symbol')
            if et == 'exits_placed' and tp_level is None:
                tp_level = entry.get('tp_level')

        if not occ_symbol:
            continue

        # Fetch ALL orders for this option contract.
        orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, symbols=[occ_symbol]))

        buy_filled = None
        sell_filled = None
        for o in orders:
            if o.status.value != 'filled':
                continue
            if o.side.value == 'buy' and buy_filled is None:
                buy_filled = o
            elif o.side.value == 'sell' and sell_filled is None:
                sell_filled = o

        # Entry price: prefer the audit's own entry_filled, fall back to the
        # Alpaca buy fill average.
        entry_price = None
        for entry in trade_data.get('entries', trade_data.get('events', [])):
            if entry.get('event_type') == 'entry_filled':
                ep = entry.get('fill_price') or entry.get('avg_price')
                if ep is not None:
                    entry_price = float(ep)
                break
        if entry_price is None and buy_filled is not None and buy_filled.filled_avg_price:
            entry_price = float(buy_filled.filled_avg_price)

        exit_price = None
        reason = None
        sell_order_id = None

        if sell_filled is not None:
            exit_price = float(sell_filled.filled_avg_price) if sell_filled.filled_avg_price else 0.0
            sell_order_id = str(sell_filled.id)
            limit = float(sell_filled.limit_price) if sell_filled.limit_price else None
            # A sell whose limit matches the TP level is a take-profit fill;
            # anything else (aggressive low limit) is a safety-close force-out.
            if tp_level is not None and limit is not None and abs(limit - float(tp_level)) < 0.001:
                reason = 'take_profit'
            else:
                reason = 'safety_close'
        elif buy_filled is not None and day_dir < today_str:
            # Bought on a prior day, never sold → 0DTE expired worthless.
            exit_price = 0.0
            reason = 'expired_worthless'
        elif buy_filled is None:
            # Entry never filled → no capital at risk; mark resolved as unfilled.
            reason = 'unfilled'
        else:
            # Bought today, not yet sold — leave pending for next reconciliation.
            continue

        # Compute PnL when we have both legs.
        pnl = None
        pnl_pct = None
        if exit_price is not None and entry_price and entry_price > 0:
            pnl = round((exit_price - entry_price) * 100, 2)
            pnl_pct = round((exit_price / entry_price - 1) * 100, 3)

        trade_data['exit_reason'] = reason
        trade_data['exit_price'] = exit_price if exit_price is not None else 0.0
        trade_data['final_pnl'] = pnl if pnl is not None else 0.0
        trade_data['final_pnl_pct'] = pnl_pct if pnl_pct is not None else 0.0
        trade_data['ended_at'] = _dt.datetime.now(la_tz).isoformat()

        events_list = trade_data.get('entries') or trade_data.get('events')
        if isinstance(events_list, list):
            if sell_order_id:
                events_list.append({
                    'event_type': 'exit_filled',
                    'fill_price': exit_price,
                    'order_id': sell_order_id,
                    'timestamp': _dt.datetime.now(la_tz).isoformat(),
                })
            events_list.append({
                'event_type': 'lifecycle',
                'state': 'CLOSED',
                'exit_reason': reason,
                'timestamp': _dt.datetime.now(la_tz).isoformat(),
            })

        tmp_path = fpath + '.tmp'
        with open(tmp_path, 'w') as fh:
            json.dump(trade_data, fh, indent=2, default=str)
        os.replace(tmp_path, fpath)

        logger.info('safety_close_audit_reconciled',
                     trade_id=trade_data.get('trade_id'),
                     asset=trade_data.get('asset'),
                     direction=trade_data.get('direction'),
                     strategy=(trade_data.get('recommendation') or {}).get('strategy_label'),
                     entry_price=entry_price,
                     exit_price=exit_price,
                     exit_reason=reason,
                     pnl=pnl,
                     pnl_pct=pnl_pct)
" > "$LOG_FILE" 2>&1
) 200>"$DIR/.safety_close.lock"

logger -t sender-trades "safety-close complete log=$LOG_FILE"
