#!/bin/bash
# safety-close.sh — force-close all Alpaca paper positions before 0DTE expiry.
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

uv run python -c "
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, QueryOrderStatus
import structlog, os, sys, json, pytz
from datetime import timezone

logger = structlog.get_logger()

key = os.environ.get('APCA_API_KEY_ID', '')
secret = os.environ.get('APCA_API_SECRET_KEY', '')
if not key:
    print('No Alpaca keys — skipping')
    sys.exit(0)

tc = TradingClient(key, secret, paper=True)

positions = tc.get_all_positions()
if not positions:
    logger.info('safety_close_no_positions', count=0)
    sys.exit(0)

logger.info('safety_close_start', count=len(positions))

LOGS_DIR = os.path.join(os.getcwd(), 'logs')

for p in positions:
    # Cancel any open orders for this symbol FIRST
    for o in tc.get_orders():
        status = str(o.status)
        if str(o.symbol) == str(p.symbol) and status not in ('filled', 'canceled', 'expired', 'rejected', 'done_for_day'):
            tc.cancel_order_by_id(str(o.id))
            logger.info('safety_close_cancelled_order', symbol=p.symbol, order_id=str(o.id))

    # Close via LIMIT sell at an aggressive price (marketable but avoids "uncovered" error)
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

# --- Audit writeback: mark safety-close fills in trade JSONs ---
import datetime as _dt
la_tz = pytz.timezone('America/Los_Angeles')
today_str = _dt.datetime.now(la_tz).strftime('%Y-%m-%d')
log_date_dir = os.path.join(LOGS_DIR, today_str)

if os.path.isdir(log_date_dir):
    for fname in sorted(os.listdir(log_date_dir)):
        if not fname.startswith('trade-') or not fname.endswith('.json') or fname.endswith('.bak'):
            continue
        fpath = os.path.join(log_date_dir, fname)
        try:
            with open(fpath, 'r') as fh:
                trade_data = json.load(fh)
        except Exception:
            continue

        exit_reason = trade_data.get('exit_reason')
        if exit_reason not in (None, 'pending'):
            continue

        occ_symbol = None
        for entry in trade_data.get('entries', trade_data.get('events', [])):
            if entry.get('event_type') == 'entry_submitted':
                occ_symbol = entry.get('occ_symbol')
                break

        if not occ_symbol:
            continue

        exit_price = None
        sell_order_id = None
        for o in tc.get_orders():
            if str(o.symbol) == occ_symbol and str(o.side) == 'sell' and str(o.status) == 'filled':
                fp = float(o.filled_avg_price) if o.filled_avg_price and float(o.filled_avg_price) > 0 else None
                if fp:
                    exit_price = fp
                    sell_order_id = str(o.id)
                    break

        if not exit_price:
            continue

        entry_price = None
        for entry in trade_data.get('entries', trade_data.get('events', [])):
            if entry.get('event_type') == 'entry_filled':
                ep = entry.get('fill_price') or entry.get('avg_price')
                if ep is not None:
                    entry_price = float(ep)
                break

        pnl = None
        pnl_pct = None
        if entry_price and entry_price > 0:
            pnl = round((exit_price - entry_price) * 100, 2)
            pnl_pct = round((exit_price / entry_price - 1) * 100, 3)

        trade_data['exit_reason'] = 'safety_close'
        trade_data['exit_price'] = exit_price
        trade_data['final_pnl'] = pnl
        trade_data['final_pnl_pct'] = pnl_pct
        trade_data['ended_at'] = _dt.datetime.now(la_tz).isoformat()

        exit_event = {
            'event_type': 'exit_filled',
            'fill_price': exit_price,
            'order_id': sell_order_id,
            'timestamp': _dt.datetime.now(la_tz).isoformat(),
        }

        events_list = trade_data.get('entries') or trade_data.get('events')
        if isinstance(events_list, list):
            events_list.append(exit_event)
            events_list.append({
                'event_type': 'lifecycle',
                'state': 'CLOSED',
                'timestamp': _dt.datetime.now(la_tz).isoformat(),
            })

        with open(fpath, 'w') as fh:
            json.dump(trade_data, fh, indent=2, default=str)

        logger.info('safety_close_audit_updated', trade_id=trade_data.get('trade_id'),
                     exit_price=exit_price, pnl=pnl)
" > "$LOG_FILE" 2>&1

logger -t sender-trades "safety-close complete log=$LOG_FILE"
