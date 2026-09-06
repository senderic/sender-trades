#!/bin/bash
# sender-trades LLM model availability preflight — runs ~15 min before
# run_trades.sh (28 6 * * 1-5), so a dead/slow model is caught and the
# chain reordered before the trading run needs it, not discovered mid-run
# at the cost of a full node timeout.
#
# This is advisory telemetry only: it always exits 0 and never blocks
# run_trades.sh, which runs on its own schedule regardless of this
# script's outcome.
#
# Add to crontab (targets ~6:13 AM PT, 15 minutes before the 6:28 run):
#   13 6 * * 1-5 /home/eric/sender-trades/scripts/preflight.sh
#
# (Not installed automatically — add the line above with `crontab -e`.)

set -u

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# cron ships a minimal PATH — make sure uv / uvx / opencode are reachable
export PATH="$HOME/.local/bin:/home/linuxbrew/.linuxbrew/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Reuse the market-data + Gmail secrets already configured for the morning
# briefing (same source order as run_trades.sh).
ATLAS_ENV="$HOME/atlas-morning-briefing/.env"
if [ -f "$ATLAS_ENV" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ATLAS_ENV"
  set +a
fi

# Source project .env for Alpaca / opencode credentials.
PROJECT_ENV="$DIR/.env"
if [ -f "$PROJECT_ENV" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$PROJECT_ENV"
  set +a
fi

cd "$DIR" || exit 1

LOG_DIR="$DIR/logs/cron"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/preflight-$STAMP.log"

uv run python -m src.preflight > "$LOG_FILE" 2>&1
RC=$?

logger -t sender-trades-preflight "run complete rc=$RC log=$LOG_FILE"

exit $RC
