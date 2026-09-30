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
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

# Running this file directly puts scripts/ on sys.path[0]; add the repo root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.option_replay import (  # noqa: E402
    ASSETS,
    BASELINE_SIGNALS,
    ET,
    EXIT_POLICIES,
    LOG_DIR,
    MONEYNESS,
    OUT_DIR,
    PROD_MONEYNESS,
    Alpaca,
    DaySetup,
    _utc,
    build_day,
    daily_bars,
    entry_price,
    occ,
    simulate_exit,
    strike_for,
)


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
