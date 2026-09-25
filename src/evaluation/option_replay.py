"""Replay trades on REAL historical 0DTE option prices (Alpaca 1-minute bars).

Shared by the offline ``scripts/scoreboard.py`` (two-year backtest over
signal x moneyness x exit) and :mod:`src.evaluation.forward_scorecard`
(daily scoring of the pipeline's own forecasts against baselines).

Alpaca serves 1-minute bars for EXPIRED SPY/QQQ 0DTE contracts back to at
least mid-2024 -- pass an explicit start/end, or the endpoint returns an
empty result. Everything fetched is cached under ``logs/scoreboard/cache/``.

No look-ahead: a :class:`DaySetup` uses only prior sessions' daily bars
and pre-market 1-minute bars up to 09:28 ET. Entries price off the 09:30
option bar (``vwap`` matched 20 real fills to ~1% median error on
2026-09-25). Exits mirror production: stop/trailing checks on the
intraday monitor's 3-minute cadence, a take-profit as a resting limit,
and a 15:20 ET safety close.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"
OUT_DIR = LOG_DIR / "scoreboard"
CACHE = OUT_DIR / "cache"
ET = ZoneInfo("America/New_York")
DATA_URL = "https://data.alpaca.markets"
ASSETS = ("SPY", "QQQ")

PREMARKET_CUTOFF = (9, 28)  # pipeline runs 09:28 ET
ENTRY_MINUTE = (9, 30)
SAFETY_CLOSE = (15, 20)  # safety_close.sh, 12:20 PT
POLL_EVERY_MIN = 3  # intraday_monitor.sh */3
STRIKE_HALF_WIDTH = 8  # fetch strikes round(s0) +/- 8

# Signed OTM distance: positive = out of the money.
MONEYNESS = {"itm0.3": -0.003, "atm": 0.0, "otm0.3": 0.003, "otm0.6": 0.006}
PROD_MONEYNESS = "otm0.6"  # compute_otm_strike(delta_target=0.30)


# ─────────────────────────────── data access ───────────────────────────────


def _load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class Alpaca:
    """Minimal cached client for Alpaca's market-data API."""

    def __init__(self) -> None:
        _load_env()
        self.http = httpx.Client(
            base_url=DATA_URL,
            headers={
                "APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"],
                "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"],
            },
            timeout=30,
        )
        self.calls = 0

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(6):
            self.calls += 1
            try:
                r = self.http.get(path, params=params)
            except httpx.HTTPError:
                time.sleep(2**attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2**attempt)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"Alpaca request kept failing: {path} {params}")

    def bars(self, path: str, params: dict[str, Any]) -> dict[str, list[dict]]:
        """Fetch every page of a multi-symbol bars endpoint."""
        out: dict[str, list[dict]] = defaultdict(list)
        params = {**params, "limit": 10000}
        while True:
            data = self._get(path, params)
            for sym, rows in (data.get("bars") or {}).items():
                out[sym].extend(rows)
            token = data.get("next_page_token")
            if not token:
                return dict(out)
            params["page_token"] = token


def _cached(path: Path, fetch):
    if path.exists():
        return json.loads(path.read_text())
    value = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return value


def _utc(day: date, hm: tuple[int, int]) -> str:
    return datetime(day.year, day.month, day.day, *hm, tzinfo=ET).isoformat()


def daily_bars(api: Alpaca, sym: str, start: date, end: date) -> dict[str, dict]:
    path = CACHE / "daily" / f"{sym}_{start}_{end}.json"

    def fetch():
        rows = api.bars(
            "/v2/stocks/bars",
            {
                "symbols": sym,
                "timeframe": "1Day",
                "start": (start - timedelta(days=15)).isoformat(),
                # SIP refuses windows reaching into the last 15 minutes.
                "end": min(
                    datetime.combine(end + timedelta(days=1), datetime.min.time(), ET),
                    datetime.now(ET) - timedelta(minutes=20),
                ).isoformat(),
                "feed": "sip",
                "adjustment": "raw",
            },
        ).get(sym, [])
        return {_bar_et(b).date().isoformat(): b for b in rows}

    return _cached(path, fetch)


def minute_bars(api: Alpaca, sym: str, day: date) -> list[dict]:
    return _cached(
        CACHE / "stock" / sym / f"{day}.json",
        lambda: api.bars(
            "/v2/stocks/bars",
            {
                "symbols": sym,
                "timeframe": "1Min",
                "start": _utc(day, (4, 0)),
                "end": _utc(day, (16, 0)),
                "feed": "sip",
                "adjustment": "raw",
            },
        ).get(sym, []),
    )


def occ(asset: str, day: date, cp: str, strike: float) -> str:
    return f"{asset}{day:%y%m%d}{cp}{round(strike * 1000):08d}"


def option_bars(api: Alpaca, asset: str, day: date, center: int) -> dict[str, list[dict]]:
    """1-minute bars for the 0DTE strike grid around ``center``, 09:30-15:30 ET."""
    syms = [
        occ(asset, day, cp, k)
        for k in range(center - STRIKE_HALF_WIDTH, center + STRIKE_HALF_WIDTH + 1)
        for cp in ("C", "P")
    ]
    return _cached(
        CACHE / "option" / asset / f"{day}_{center}.json",
        lambda: api.bars(
            "/v1beta1/options/bars",
            {
                "symbols": ",".join(syms),
                "timeframe": "1Min",
                "start": _utc(day, (9, 30)),
                "end": _utc(day, (15, 31)),
            },
        ),
    )


def _bar_et(bar: dict) -> datetime:
    return datetime.fromisoformat(bar["t"].replace("Z", "+00:00")).astimezone(ET)


def _hm(bar: dict) -> tuple[int, int]:
    t = _bar_et(bar)
    return (t.hour, t.minute)


# ─────────────────────────────── day setup ────────────────────────────────


@dataclass
class DaySetup:
    day: date
    asset: str
    s0: float  # last pre-market price at/before 09:28 ET (the mechanics price)
    prev_close: float
    prev2_close: float
    prev_range_pct: float  # prior session (high-low)/close
    open_: float
    close_1520: float
    options: dict[str, list[dict]]
    features: dict[str, float] = field(default_factory=dict)

    @property
    def gap_pct(self) -> float:
        return (self.s0 / self.prev_close - 1) * 100


def build_day(api: Alpaca, asset: str, day: date, daily: dict[str, dict]) -> DaySetup | None:
    keys = sorted(k for k in daily if k < day.isoformat())
    if len(keys) < 2 or day.isoformat() not in daily:
        return None
    prev, prev2 = daily[keys[-1]], daily[keys[-2]]
    mins = minute_bars(api, asset, day)
    pre = [b for b in mins if _hm(b) <= PREMARKET_CUTOFF]
    reg = [b for b in mins if ENTRY_MINUTE <= _hm(b) <= SAFETY_CLOSE]
    if not pre or not reg:
        return None
    s0 = pre[-1]["c"]
    opts = option_bars(api, asset, day, round(s0))
    atm_c = opts.get(occ(asset, day, "C", round(s0)))
    if not atm_c:
        return None  # no same-day expiry listed that day
    setup = DaySetup(
        day=day,
        asset=asset,
        s0=s0,
        prev_close=prev["c"],
        prev2_close=prev2["c"],
        prev_range_pct=(prev["h"] - prev["l"]) / prev["c"] * 100,
        open_=reg[0]["o"],
        close_1520=reg[-1]["c"],
        options=opts,
    )
    # Implied move: ATM straddle at the open as a % of spot -- what the
    # market is charging for today's move, known at entry time.
    c = entry_price(opts.get(occ(asset, day, "C", round(s0))) or [], "vwap")
    p = entry_price(opts.get(occ(asset, day, "P", round(s0))) or [], "vwap")
    if c and p:
        setup.features["implied_move_pct"] = (c + p) / s0 * 100
    setup.features["abs_gap_pct"] = abs(setup.gap_pct)
    setup.features["prev_range_pct"] = setup.prev_range_pct
    return setup


# ─────────────────────────── entry / exit model ───────────────────────────


def entry_price(bars: list[dict], model: str) -> float | None:
    """Entry price from the 09:30 bar. ``model`` is one of the bar fields."""
    first = next((b for b in bars if _hm(b) == ENTRY_MINUTE), None)
    if first is None:
        first = next((b for b in bars if _hm(b) <= (9, 32)), None)
    if first is None:
        return None
    return {"open": first["o"], "vwap": first["vw"], "high": first["h"], "close": first["c"]}[model]


@dataclass(frozen=True)
class ExitPolicy:
    name: str
    take_profit_pct: float | None = None  # resting limit
    stop_loss_pct: float | None = -50.0  # polled
    trail_activate_pct: float | None = None  # polled, P&L points below peak
    trail_points: float | None = None


EXIT_POLICIES = [
    ExitPolicy("tp100_sl50", take_profit_pct=100.0),  # production until 2026-09-14
    ExitPolicy("sl50_trail30/15", trail_activate_pct=30.0, trail_points=15.0),  # advisor fallback
    ExitPolicy("sl50_trail50/50", trail_activate_pct=50.0, trail_points=50.0),
    ExitPolicy("sl50_hold", stop_loss_pct=-50.0),
    ExitPolicy("hold_to_close", stop_loss_pct=None),
]


def simulate_exit(
    bars: list[dict], entry: float, policy: ExitPolicy, exit_slip: float
) -> tuple[float, str]:
    """Walk the option's minute bars after entry and return ``(exit_price, reason)``."""
    session = [b for b in bars if ENTRY_MINUTE < _hm(b) <= SAFETY_CLOSE]
    peak = 0.0
    last = entry
    tp = entry * (1 + policy.take_profit_pct / 100) if policy.take_profit_pct else None
    for b in session:
        if tp is not None and b["h"] >= tp:
            return tp, "take_profit"
        last = b["c"]
        _, m = _hm(b)
        if m % POLL_EVERY_MIN:
            continue
        pnl = (last / entry - 1) * 100
        peak = max(peak, pnl)
        if policy.stop_loss_pct is not None and pnl <= policy.stop_loss_pct:
            return last * (1 - exit_slip), "stop_loss"
        if (
            policy.trail_activate_pct is not None
            and peak >= policy.trail_activate_pct
            and pnl <= peak - policy.trail_points
        ):
            return last * (1 - exit_slip), "trailing_stop"
    return last * (1 - exit_slip), "safety_close"


def strike_for(s0: float, direction: str, moneyness: str) -> int:
    m = MONEYNESS[moneyness]
    return round(s0 * (1 + m)) if direction == "CALL" else round(s0 * (1 - m))


# ─────────────────────────────── signals ─────────────────────────────────


def _sign(x: float) -> str | None:
    return "CALL" if x > 0 else "PUT" if x < 0 else None


BASELINE_SIGNALS = {
    "gap_follow": lambda d: _sign(d.gap_pct),
    "gap_fade": lambda d: _sign(-d.gap_pct),
    "repeat_yesterday": lambda d: _sign(d.prev_close - d.prev2_close),
    "always_call": lambda _d: "CALL",
    "always_put": lambda _d: "PUT",
    # Upper bound: perfect knowledge of the entry->safety-close direction.
    "oracle": lambda d: _sign(d.close_1520 - d.s0),
}
