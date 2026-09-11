#!/bin/bash
# intraday_monitor.sh — enforce the -50% stop-loss intraday for open 0DTE
# positions. The pipeline places only the TP at Alpaca and exits; this cron
# polls the live option mark and force-closes any trade at/below its sl_level.
# Runs every 3 min 9:30 AM - 12:20 PM PT. Idempotent: marks exit_reason so
# the safety-close sweep (12:20 PM PT) and later passes skip it.
# Also guards against overlapping runs via flock on .intraday_monitor.lock.

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
LOG_FILE="$DIR/logs/cron/intraday-$TIMESTAMP.log"
mkdir -p "$DIR/logs/cron"

(
  flock -n 200 || exit 0
  uv run python -m src.execution.intraday_monitor > "$LOG_FILE" 2>&1
) 200>"$DIR/.intraday_monitor.lock"

logger -t sender-trades "intraday-monitor complete log=$LOG_FILE"
