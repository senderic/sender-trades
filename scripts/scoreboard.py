"""Scoreboard: score trade-selection policies on REAL historical 0DTE option prices.

Why this exists: the live record (23 filled trades, two outliers carrying the
whole P&L) can't tell skill from luck, and ``scripts/simulate_exits.py``
reconstructs option marks with Black-Scholes, which missed real exits by
>30% on 8 of 10 trades. Alpaca serves real 1-minute bars for expired
SPY/QQQ 0DTE contracts back to at least mid-2024, so this script scores
every combination of

    direction signal  x  strike moneyness  x  exit policy

over every trading day with a same-day expiry, using the option's actual
traded prices -- no pricing model. The LLM is scored the same way, from the
cached replay predictions under ``logs/replay/`` (zero new LLM calls).

No look-ahead: signals see only the prior sessions' daily bars and
pre-market 1-minute bars up to 09:28 ET (the pipeline's cutoff). Entries
are priced from the 09:30 option bar, calibrated against the real fills in
``logs/<date>/trade-*.json`` (see ``calibrate``). Exits follow production
mechanics: the stop-loss and trailing stop are checked on the intraday
monitor's 3-minute cron cadence, a take-profit is a resting limit (fills
on any bar whose high reaches it), and anything open is closed at the
15:20 ET safety close.

Usage::

    uv run python scripts/scoreboard.py --start 2024-06-03 --end 2026-09-24

Everything fetched is cached under ``logs/scoreboard/cache/`` so reruns
are free. Outputs: ``logs/scoreboard/report.md`` and ``trades.csv``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx

ROOT = Path(__file__).resolve().parents[1]
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


def _json_loose(text: Any) -> Any:
    for _ in range(3):
        if not isinstance(text, str):
            return text
        s = text.strip()
        if s.startswith("```"):
            s = s.strip("`").removeprefix("json").strip()
        try:
            text = json.loads(s)
        except ValueError:
            i, j = s.find("{"), s.rfind("}")
            if i < 0 or j < 0:
                return None
            try:
                text = json.loads(s[i : j + 1])
            except ValueError:
                return None
    return text


def load_llm_signals() -> dict[str, dict[tuple[str, str], str]]:
    """Per-asset directions from each cached replay model (no LLM calls)."""
    out: dict[str, dict[tuple[str, str], str]] = {}
    for model_dir in sorted((LOG_DIR / "replay").iterdir()):
        if not model_dir.is_dir() or model_dir.name in ("premarket", "premarket_cache"):
            continue
        preds: dict[tuple[str, str], str] = {}
        for f in model_dir.glob("*.json"):
            rec = json.loads(f.read_text())
            if not rec.get("ok"):
                continue
            body = _json_loose(rec.get("response"))
            table = body.get("predictions") if isinstance(body, dict) else None
            if not isinstance(table, dict):
                continue
            for asset, p in table.items():
                if not isinstance(p, dict):
                    continue
                d = str(p.get("direction", "")).upper()
                sig = {"UP": "CALL", "DOWN": "PUT"}.get(d)
                if sig and asset in ASSETS:
                    preds[(rec.get("date") or f.stem, asset)] = sig
        if preds:
            out["llm:" + model_dir.name.replace("__", "/", 1)] = preds
    return out


def load_live_trades() -> list[dict[str, Any]]:
    """Real filled entries from the trade audits, for calibration and comparison."""
    rows = []
    for f in sorted(glob.glob(str(LOG_DIR / "2026-*/trade-*.json"))):
        d = json.loads(Path(f).read_text())
        evs = [e for e in (d.get("entries") or []) + (d.get("events") or []) if isinstance(e, dict)]
        fill = next((e for e in evs if e.get("event_type") == "entry_filled"), None)
        sym = next((e.get("occ_symbol") for e in evs if e.get("occ_symbol")), None)
        if not fill or not sym or len(sym) != 18 or d.get("final_pnl") is None:
            continue
        rows.append(
            {
                "day": Path(f).parent.name,
                "asset": d.get("asset"),
                "direction": d.get("direction"),
                "occ": sym,
                "fill": fill.get("avg_price"),
                "fill_ts": fill.get("timestamp"),
                "contracts": d.get("contracts") or 1,
                "final_pnl": d.get("final_pnl"),
            }
        )
    return rows


# ─────────────────────────────── statistics ────────────────────────────────


def summarize(pnls: list[float], rets: list[float], seed: int = 7) -> dict[str, Any]:
    n = len(pnls)
    if not n:
        return {"n": 0}
    rng = random.Random(seed)
    boots = sorted(statistics.fmean(rng.choices(pnls, k=n)) for _ in range(2000))
    top2 = sum(sorted(pnls, reverse=True)[:2])
    return {
        "n": n,
        "win": sum(p > 0 for p in pnls) / n,
        "total": sum(pnls),
        "mean": statistics.fmean(pnls),
        "ci": (boots[50], boots[1949]),
        "median": statistics.median(pnls),
        "mean_ret": statistics.fmean(rets),
        "ex_top2": sum(pnls) - top2,
    }


# ─────────────────────────────── main run ─────────────────────────────────


@dataclass
class TradeRow:
    day: str
    asset: str
    signal: str
    direction: str
    moneyness: str
    exit_policy: str
    strike: int
    entry: float
    exit: float
    reason: str
    pnl: float  # $ per 1 contract
    ret: float  # % of premium
    features: dict[str, float]


def simulate_day(
    d: DaySetup, llm: dict[str, dict[tuple[str, str], str]], entry_model: str, exit_slip: float
) -> list[TradeRow]:
    """Every signal x moneyness x exit policy for one asset-day."""
    rows: list[TradeRow] = []
    signals = {name: fn(d) for name, fn in BASELINE_SIGNALS.items()}
    for name, preds in llm.items():
        if (d.day.isoformat(), d.asset) in preds:
            signals[name] = preds[(d.day.isoformat(), d.asset)]
    for sig, direction in signals.items():
        if direction is None:
            continue
        for mny in MONEYNESS:
            k = strike_for(d.s0, direction, mny)
            bars = d.options.get(occ(d.asset, d.day, direction[0], k)) or []
            entry = entry_price(bars, entry_model)
            if not entry or entry < 0.02:
                continue
            for pol in EXIT_POLICIES:
                px, reason = simulate_exit(bars, entry, pol, exit_slip)
                rows.append(
                    TradeRow(
                        day=d.day.isoformat(),
                        asset=d.asset,
                        signal=sig,
                        direction=direction,
                        moneyness=mny,
                        exit_policy=pol.name,
                        strike=k,
                        entry=entry,
                        exit=px,
                        reason=reason,
                        pnl=(px - entry) * 100,
                        ret=(px / entry - 1) * 100,
                        features=dict(d.features),
                    )
                )
    return rows


def calibrate(api: Alpaca, live: list[dict]) -> dict[str, Any]:
    """Compare each entry-price model with the real fills at the open."""
    errs: dict[str, list[float]] = defaultdict(list)
    detail = []
    for t in live:
        ts = datetime.fromisoformat(t["fill_ts"]).astimezone(ET)
        if (ts.hour, ts.minute) > (9, 32):
            continue  # late/retried entries: not the modeled at-the-open fill
        day = date.fromisoformat(t["day"])
        bars = api.bars(
            "/v1beta1/options/bars",
            {
                "symbols": t["occ"],
                "timeframe": "1Min",
                "start": _utc(day, (9, 30)),
                "end": _utc(day, (9, 33)),
            },
        ).get(t["occ"], [])
        row = {"day": t["day"], "occ": t["occ"], "fill": t["fill"]}
        for model in ("open", "vwap", "high", "close"):
            px = entry_price(bars, model)
            row[model] = px
            if px:
                errs[model].append((t["fill"] - px) / t["fill"] * 100)
        detail.append(row)
    stats = {
        m: {
            "median_err_pct": statistics.median(v),
            "mean_abs_err_pct": statistics.fmean(map(abs, v)),
        }
        for m, v in errs.items()
        if v
    }
    return {"models": stats, "detail": detail}


def run(args: argparse.Namespace) -> None:
    api = Alpaca()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    live = load_live_trades()

    calib = calibrate(api, live)
    entry_model = args.entry_model or min(
        calib["models"], key=lambda m: calib["models"][m]["mean_abs_err_pct"]
    )
    print(f"entry model: {entry_model}  calibration: {json.dumps(calib['models'])}", flush=True)  # noqa: T201

    llm = load_llm_signals()
    days: list[DaySetup] = []
    rows: list[TradeRow] = []
    for asset in ASSETS:
        daily = daily_bars(api, asset, start, end)
        trading = sorted(
            date.fromisoformat(k) for k in daily if start.isoformat() <= k <= end.isoformat()
        )
        for i, day in enumerate(trading):
            setup = build_day(api, asset, day, daily)
            if setup:
                rows.extend(simulate_day(setup, llm, entry_model, args.exit_slip))
                # Option bars are ~12k dicts per asset-day; holding all of
                # them until the end exhausted memory on the full range.
                setup.options = {}
                days.append(setup)
            if i % 50 == 0:
                print(f"{asset} {day} ({i + 1}/{len(trading)}) api_calls={api.calls}", flush=True)  # noqa: T201

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "trades.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        feat_keys = ["implied_move_pct", "abs_gap_pct", "prev_range_pct"]
        w.writerow(
            ["day", "asset", "signal", "direction", "moneyness", "exit_policy", "strike",
             "entry", "exit", "reason", "pnl", "ret", *feat_keys]
        )  # fmt: skip
        for r in rows:
            w.writerow(
                [r.day, r.asset, r.signal, r.direction, r.moneyness, r.exit_policy, r.strike,
                 round(r.entry, 4), round(r.exit, 4), r.reason, round(r.pnl, 2), round(r.ret, 2),
                 *(round(r.features.get(k, float("nan")), 4) for k in feat_keys)]
            )  # fmt: skip
    write_report(rows, days, calib, entry_model, live, llm, args)
    print(f"done: {len(days)} asset-days, {len(rows)} simulated trades, api_calls={api.calls}")  # noqa: T201
    print(f"report: {OUT_DIR / 'report.md'}")  # noqa: T201


# ─────────────────────────────── report ─────────────────────────────────


def _fmt(s: dict[str, Any]) -> str:
    if not s.get("n"):
        return "| - | - | - | - | - | - | - |"
    lo, hi = s["ci"]
    return (
        f"| {s['n']} | {s['win']:.0%} | ${s['mean']:+.1f} | [{lo:+.1f}, {hi:+.1f}] "
        f"| {s['mean_ret']:+.1f}% | ${s['total']:+,.0f} | ${s['ex_top2']:+,.0f} |"
    )


HEADER = (
    "| n | win | mean $/trade | 95% CI | mean % of premium | total $ | total ex-top-2 |\n"
    "|---|---|---|---|---|---|---|"
)


def write_report(rows, days, calib, entry_model, live, llm, args) -> None:
    by = defaultdict(list)
    for r in rows:
        by[(r.signal, r.moneyness, r.exit_policy)].append(r)

    def stats(key_rows):
        return summarize([r.pnl for r in key_rows], [r.ret for r in key_rows])

    dates = sorted({d.day for d in days})
    split = dates[len(dates) // 2].isoformat() if dates else ""
    lines = [
        "# Scoreboard — real 0DTE option prices",
        "",
        f"Range {args.start} → {args.end}: {len(days)} asset-days with a same-day expiry "
        f"({sum(d.asset == 'SPY' for d in days)} SPY, {sum(d.asset == 'QQQ' for d in days)} QQQ). "
        f"One contract per trade, entry = 09:30 bar `{entry_model}`, exit slippage "
        f"{args.exit_slip:.0%}, safety close 15:20 ET. $ figures are per contract.",
        "",
        "## Entry-price calibration against real fills",
        "",
        "Real fill vs the 09:30 option bar (positive = real fill paid more than the model).",
        "",
        "| model | median error | mean abs error |",
        "|---|---|---|",
    ]
    for m, s in calib["models"].items():
        lines.append(f"| {m} | {s['median_err_pct']:+.1f}% | {s['mean_abs_err_pct']:.1f}% |")
    lines += [f"\n{len(calib['detail'])} fills at the open were compared.", ""]

    # 1. Baselines at production mechanics.
    lines += [
        f"## 1. Direction signals at production strike ({PROD_MONEYNESS}), each exit policy",
        "",
    ]
    for pol in EXIT_POLICIES:
        lines += [
            f"### Exit: `{pol.name}`",
            "",
            "| signal " + HEADER.split("\n")[0],
            "|---" + HEADER.split("\n")[1],
        ]
        for sig in [*BASELINE_SIGNALS, *llm]:
            s = stats(by[(sig, PROD_MONEYNESS, pol.name)])
            lines.append(f"| {sig} " + _fmt(s))
        lines.append("")

    # 2. LLM vs baselines on the SAME days only.
    lines += [
        "## 2. LLM vs baselines on the same asset-days",
        "",
        "Each LLM compared only on the days it has a cached prediction for, "
        f"at {PROD_MONEYNESS} with `sl50_trail30/15` (the advisor's deterministic fallback).",
        "",
    ]
    pol = "sl50_trail30/15"
    for name, preds in llm.items():
        keys = set(preds)
        lines += [
            f"### {name} ({len(keys)} asset-day predictions)",
            "",
            "| signal " + HEADER.split("\n")[0],
            "|---" + HEADER.split("\n")[1],
        ]
        for sig in [name, *BASELINE_SIGNALS]:
            sel = [r for r in by[(sig, PROD_MONEYNESS, pol)] if (r.day, r.asset) in keys]
            lines.append(f"| {sig} " + _fmt(stats(sel)))
        lines.append("")

    # 3. Moneyness.
    lines += [
        "## 3. Strike moneyness (all signals pooled except oracle, `sl50_trail30/15`)",
        "",
        "| moneyness " + HEADER.split("\n")[0],
        "|---" + HEADER.split("\n")[1],
    ]
    for mny in MONEYNESS:
        sel = [
            r
            for r in rows
            if r.moneyness == mny
            and r.exit_policy == pol
            and r.signal
            in ("gap_follow", "gap_fade", "repeat_yesterday", "always_call", "always_put")
        ]
        lines.append(f"| {mny} " + _fmt(stats(sel)))
    lines.append("")

    # 4. Filters: is today worth playing? Terciles fit on the first half, tested on the second.
    lines += [
        '## 4. Day filters — "is today worth playing?"',
        "",
        f"Tercile cut points are fit on days before {split} and applied unchanged to days on/after it, "
        "so the TEST column is out-of-sample. Signal pool: gap_follow, gap_fade, repeat_yesterday, "
        f"always_call, always_put at {PROD_MONEYNESS}, `sl50_trail30/15`.",
        "",
    ]
    pool = [
        r
        for r in rows
        if r.moneyness == PROD_MONEYNESS
        and r.exit_policy == pol
        and r.signal in ("gap_follow", "gap_fade", "repeat_yesterday", "always_call", "always_put")
    ]
    for feat in ("implied_move_pct", "abs_gap_pct", "prev_range_pct"):
        train_vals = sorted(
            {
                (d.day, d.asset): d.features.get(feat)
                for d in days
                if d.day.isoformat() < split and d.features.get(feat) is not None
            }.values()
        )
        if len(train_vals) < 9:
            continue
        q1, q2 = train_vals[len(train_vals) // 3], train_vals[2 * len(train_vals) // 3]
        lines += [
            f"### {feat} (cuts {q1:.2f} / {q2:.2f})",
            "",
            "| bucket | half " + HEADER.split("\n")[0],
            "|---|---" + HEADER.split("\n")[1],
        ]
        for label, lo, hi in (("low", -1e9, q1), ("mid", q1, q2), ("high", q2, 1e9)):
            for half, cond in (
                ("train", lambda r: r.day < split),
                ("TEST", lambda r: r.day >= split),
            ):
                sel = [
                    r
                    for r in pool
                    if cond(r) and r.features.get(feat) is not None and lo <= r.features[feat] < hi
                ]
                lines.append(f"| {label} | {half} " + _fmt(stats(sel)))
        lines.append("")

    # 5. Live record for reference.
    lines += [
        "## 5. Live record (for reference)",
        "",
        f"{len(live)} filled trades, total ${sum(t['final_pnl'] for t in live):+,.2f} as booked "
        "(contract counts vary; the scoreboard uses 1 contract).",
        "",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--start", default="2024-06-03")
    p.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat())
    p.add_argument("--entry-model", choices=["open", "vwap", "high", "close"], default=None,
                   help="Default: the model closest to real fills (see calibration).")  # fmt: skip
    p.add_argument("--exit-slip", type=float, default=0.03, help="Haircut on polled/close exits.")
    run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
