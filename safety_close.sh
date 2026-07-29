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
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
import structlog, os, sys

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

for p in positions:
    try:
        req = MarketOrderRequest(
            symbol=p.symbol, qty=int(p.qty),
            side=OrderSide.SELL, type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
        )
        tc.submit_order(req)
        logger.info('safety_close_sold', symbol=p.symbol, qty=p.qty, pnl_pct=p.unrealized_plpc)
    except Exception as e:
        logger.error('safety_close_failed', symbol=p.symbol, error=str(e))

# Cancel any lingering open orders
for o in tc.get_orders():
    status = str(o.status)
    if status not in ('filled', 'canceled', 'expired', 'rejected', 'done_for_day'):
        tc.cancel_order_by_id(str(o.id))
        logger.info('safety_close_cancelled_order', order_id=str(o.id))
" > "$LOG_FILE" 2>&1

logger -t sender-trades "safety-close complete log=$LOG_FILE"
