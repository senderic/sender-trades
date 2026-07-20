#!/bin/bash
# sender-trades daily cron runner.
# Picks up the latest Atlas morning briefing snapshot from ~/atlas-morning-briefing,
# runs the pipeline in dry-run mode, and emails the directional forecast + trade idea.

set -u

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# cron ships a minimal PATH — make sure uv / uvx / opencode are reachable
export PATH="$HOME/.local/bin:/home/linuxbrew/.linuxbrew/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Reuse the market-data + Gmail secrets already configured for the morning briefing
# (FINNHUB_API_KEY, BRAVE_API_KEY, GMAIL_USER, GMAIL_APP_PASSWORD, RECIPIENT_EMAIL, ...).
ATLAS_ENV="$HOME/atlas-morning-briefing/.env"
if [ -f "$ATLAS_ENV" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ATLAS_ENV"
  set +a
fi

cd "$DIR" || exit 1

LOG_DIR="$DIR/logs/cron"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/sender-trades-$STAMP.log"

# Dry-run by default (no MCP order execution). --email dispatches the forecast.
uv run python -m src.main --dry-run --email > "$LOG_FILE" 2>&1
RC=$?

logger -t sender-trades "run complete rc=$RC log=$LOG_FILE"

exit $RC
