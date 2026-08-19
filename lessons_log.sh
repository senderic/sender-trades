#!/bin/bash
# lessons-log.sh — post-market analysis that updates LESSONS_LEARNED.md.
# Runs at 2:00 PM PT / 5:00 PM ET, after market close.
# Reads the day's audit logs, compares predictions vs actuals, writes entry.

set -u

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:/home/linuxbrew/.linuxbrew/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

ATLAS_ENV="$HOME/atlas-morning-briefing/.env"
if [ -f "$ATLAS_ENV" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ATLAS_ENV"
  set +a
fi

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
LOG_FILE="$LOG_DIR/lessons-log-$STAMP.log"

uv run python -m src.lessons_analyzer > "$LOG_FILE" 2>&1
RC=$?

logger -t sender-trades "lessons-log complete rc=$RC log=$LOG_FILE"

exit $RC
