from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

LA_TZ = ZoneInfo("America/Los_Angeles")
ET_TZ = ZoneInfo("America/New_York")


def today_local() -> date:
    return datetime.now(LA_TZ).date()


def now_local() -> datetime:
    return datetime.now(LA_TZ)


def format_la(dt: datetime) -> str:
    d = dt.astimezone(LA_TZ)
    return d.strftime("%Y-%m-%d %I:%M %p %Z")
