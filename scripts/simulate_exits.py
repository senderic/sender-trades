"""Offline policy simulation for the profit-exit advisor.

Answers "would replacing the fixed +100% take-profit with the LLM
exit-advisor (or a plain trailing stop) have made more money?" by
reconstructing a per-minute option-mark path for each historical filled
0DTE trade that has Yahoo Finance 1-minute underlying data available
(Yahoo keeps ~30 trading days of 1m history, so roughly 2026-08-12
onward as of 2026-09-11 -- see YahooFinanceProvider in
src/ingestion/candle_providers.py), then replays four exit policies
against that reconstructed path:

    (a) current rules      -- TP +100% / SL -50% (today's behaviour)
    (b) SL-only             -- SL -50%, otherwise hold to the safety close
    (c) SL + trailing       -- SL -50%, plus the configured trailing stop
                                (execution.exit_strategy.trailing), no LLM
    (d) SL + advisor        -- SL -50%, plus the real exit-advisor logic
                                (src.execution.exit_advisor), consulting
                                opencode/muse-spark-1.3-contributor-free
                                ONLY (no fallback chain), at the same
                                event cadence production uses

Option marks are reconstructed with Black-Scholes: the implied vol is
backed out from the trade's ACTUAL entry fill price (the only real
premium we have), then held constant while spot and time-to-expiry move
minute to minute. This is a real, disclosed simplification -- see
"Caveats" below and the printed report -- not a claim that these are
the marks that would have actually printed.

All four policies reuse the poll cadence of the real intraday monitor
cron (every 3 minutes), and policy (d) reuses the actual production
code in src.execution.exit_advisor (decide_trigger, AdvisorState,
trailing_fallback_triggered, build_context/build_prompt, consult) so
the simulation exercises the same logic that will run live, not a
reimplementation of it.

This script only READS existing modules (src.execution.exit_advisor,
src.ingestion.candle_providers, src.json_utils, src.llm.client) and the
historical logs/<date>/trade-*.json audit files. It places no orders,
sends no email, and writes nothing back into logs/ -- only prints a
report to stdout.

Usage::

    uv run python scripts/simulate_exits.py \\
        --start 2026-08-12 --end 2026-09-10 \\
        --model opencode/muse-spark-1.3-contributor-free \\
        --concurrency 1 --max-calls 150

Caveats (also printed in the report):
    - Black-Scholes marks for 0DTE options are rough: a single frozen
      IV, no vol surface, no skew, no bid/ask spread, calendar-time
      (not trading-time) T. Small n (see below).
    - n is small: only trades in the ~30-day Yahoo 1m window are
      usable, and only those with BOTH a resolved outcome (not
      premium_gate_blocked/rejected/expired) AND fetchable 1m bars.
    - The script checks its own reconstruction against reality: at
      each trade's REAL exit timestamp, it compares the reconstructed
      BS mark to the REAL recorded exit_price and reports the
      deltas plainly -- if these don't line up, that is stated, not
      hidden, and the policy comparison should be read as indicative
      only, not a proven backtest.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

# Puts the repo root on sys.path so `import src.*` resolves when this
# script is run directly (`uv run python scripts/simulate_exits.py`),
# which otherwise only puts `scripts/` itself on sys.path[0].
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.execution import exit_advisor
from src.execution.exit_advisor import (
    AdvisorState,
    ConsultResult,
    build_context,
    build_prompt,
    decide_trigger,
    trailing_fallback_triggered,
    update_peak,
)
from src.execution.models import ExitAdvisorConfig, ExitConfig, TrailingConfig
from src.ingestion.candle_providers import YahooFinanceProvider
from src.json_utils import load_json_tolerant
from src.timezone import ET_TZ

RESOLVED_REASONS = ("take_profit", "stop_loss", "safety_close")
POLL_INTERVAL_MIN = 3  # matches intraday_monitor.sh cron cadence
SAFETY_CLOSE_ET = (15, 20)  # matches safety_close.sh cron (12:20 PM PT)
MARKET_CLOSE_ET = (16, 0)
RISK_FREE_RATE = 0.05
LOCAL_OPENCODE_ERROR_SUBSTRINGS = ("Failed query: insert into", "Session not found")


# ─────────────────────────── Black-Scholes ───────────────────────────


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """European Black-Scholes price. Falls back to intrinsic value at T<=0."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if is_call:
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def implied_vol(
    target_price: float, S: float, K: float, T: float, r: float, is_call: bool
) -> float | None:
    """Back out sigma from a target option price via bisection.

    Returns None when the target price can't be bracketed by BS prices
    over a wide sigma range (e.g. the entry fill was outside what BS can
    represent at this T -- happens for very cheap/deep OTM 0DTE prints).
    """
    if target_price <= 0 or T <= 0:
        return None
    lo, hi = 1e-4, 8.0
    price_lo = bs_price(S, K, T, r, lo, is_call)
    price_hi = bs_price(S, K, T, r, hi, is_call)
    if not (price_lo <= target_price <= price_hi):
        return None
    for _ in range(100):
        mid = (lo + hi) / 2.0
        p = bs_price(S, K, T, r, mid, is_call)
        if abs(p - target_price) < 1e-5:
            return mid
        if p < target_price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def time_to_expiry_years(now_et: datetime, expiry_date: date) -> float:
    close_dt = datetime(
        expiry_date.year, expiry_date.month, expiry_date.day, *MARKET_CLOSE_ET, tzinfo=ET_TZ
    )
    seconds = max((close_dt - now_et).total_seconds(), 30.0)
    return seconds / (365.25 * 24 * 3600)


# ────────────────────────── trade loading ──────────────────────────


@dataclass
class HistTrade:
    path: Path
    trade_id: str
    trade_date: date
    asset: str
    direction: str  # CALL / PUT
    strike: float
    contracts: int
    entry_price: float
    entry_time: datetime
    actual_exit_reason: str
    actual_exit_price: float | None
    actual_exit_time: datetime | None
    actual_final_pnl: float | None


def load_candidate_trades(log_dir: Path, start: date, end: date) -> list[HistTrade]:
    out: list[HistTrade] = []
    for day_dir in sorted(log_dir.iterdir()):
        if not day_dir.is_dir():
            continue
        try:
            d = datetime.strptime(day_dir.name, "%Y-%m-%d").date()
        except ValueError:
            continue
        if not (start <= d <= end):
            continue
        for fpath in sorted(day_dir.glob("trade-*.json")):
            if fpath.name.endswith((".bak", ".tmp")):
                continue
            try:
                data = load_json_tolerant(fpath.read_text())
            except Exception:
                continue
            trade = _extract_hist_trade(data, fpath, d)
            if trade is not None:
                out.append(trade)
    return out


def _extract_hist_trade(data: dict[str, Any], path: Path, trade_date: date) -> HistTrade | None:
    exit_reason = data.get("exit_reason")
    if exit_reason not in RESOLVED_REASONS:
        return None

    entry_price: float | None = None
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    for entry in data.get("entries") or []:
        et = entry.get("event_type")
        if et == "entry_filled" and entry_price is None:
            ep = entry.get("fill_price") or entry.get("avg_price")
            if ep is not None:
                entry_price = float(ep)
            ts = entry.get("timestamp")
            if ts:
                with contextlib.suppress(ValueError):
                    entry_time = datetime.fromisoformat(ts).astimezone(ET_TZ)
        if et == "exit_filled" and exit_time is None:
            ts = entry.get("timestamp")
            if ts:
                with contextlib.suppress(ValueError):
                    exit_time = datetime.fromisoformat(ts).astimezone(ET_TZ)

    if entry_price is None or entry_price <= 0 or entry_time is None:
        return None
    strike = data.get("entry_strike")
    asset = data.get("asset")
    direction = data.get("direction")
    if not strike or not asset or direction not in ("CALL", "PUT"):
        return None

    if exit_time is None:
        ended_at = data.get("ended_at")
        if ended_at:
            with contextlib.suppress(ValueError):
                exit_time = datetime.fromisoformat(ended_at).astimezone(ET_TZ)

    return HistTrade(
        path=path,
        trade_id=data.get("trade_id", ""),
        trade_date=trade_date,
        asset=asset,
        direction=direction,
        strike=float(strike),
        contracts=int(data.get("contracts") or 1),
        entry_price=entry_price,
        entry_time=entry_time,
        actual_exit_reason=exit_reason,
        actual_exit_price=data.get("exit_price"),
        actual_exit_time=exit_time,
        actual_final_pnl=data.get("final_pnl"),
    )


# ────────────────────────── market data ──────────────────────────


_BAR_CACHE: dict[tuple[str, date], list[dict] | None] = {}


async def get_bars(provider: YahooFinanceProvider, symbol: str, d: date) -> list[dict] | None:
    key = (symbol, d)
    if key in _BAR_CACHE:
        return _BAR_CACHE[key]
    try:
        bars = await provider.fetch_intraday_candles(symbol, d, resolution=1)
    except Exception:
        bars = None
    if bars:
        bars = sorted(bars, key=lambda b: b["timestamp"])
    _BAR_CACHE[key] = bars
    return bars


def bar_time(bar: dict) -> datetime:
    return datetime.fromtimestamp(bar["timestamp"], tz=ET_TZ)


def find_bar_index_at_or_after(bars: list[dict], target: datetime) -> int | None:
    for i, b in enumerate(bars):
        if bar_time(b) >= target:
            return i
    return None


def find_bar_index_at_or_before(bars: list[dict], target: datetime) -> int | None:
    idx = None
    for i, b in enumerate(bars):
        if bar_time(b) <= target:
            idx = i
        else:
            break
    return idx


def summarize_bars(bars: list[dict]) -> dict[str, Any]:
    """Same shape as exit_advisor.gather_market_context's per-symbol summary."""
    if not bars:
        return {"available": False}
    opens = [b["open"] for b in bars]
    highs = [b["high"] for b in bars]
    lows = [b["low"] for b in bars]
    closes = [b["close"] for b in bars]
    volumes = [b.get("volume", 0) or 0 for b in bars]
    total_vol = sum(volumes) or 1
    typical = [(hi + lo + c) / 3.0 for hi, lo, c in zip(highs, lows, closes, strict=True)]
    vwap = sum(t * v for t, v in zip(typical, volumes, strict=True)) / total_vol
    recent = bars[-24:]
    return {
        "available": True,
        "open": round(opens[0], 4),
        "high": round(max(highs), 4),
        "low": round(min(lows), 4),
        "last": round(closes[-1], 4),
        "vwap": round(vwap, 4),
        "pct_change_from_open": (
            round((closes[-1] / opens[0] - 1.0) * 100.0, 3) if opens[0] else None
        ),
        "recent_5min_bars": [{"t": b["timestamp"], "c": round(b["close"], 4)} for b in recent[::5]],
    }


# ──────────────────────── reconstructed path ────────────────────────


@dataclass
class ReconstructedTrade:
    hist: HistTrade
    iv: float | None
    entry_spot: float | None
    poll_indices: list[int] = field(default_factory=list)  # indices into `bars`
    bars: list[dict] = field(default_factory=list)
    other_bars: list[dict] = field(default_factory=list)
    unreconstructable_reason: str = ""


async def reconstruct_trade(provider: YahooFinanceProvider, hist: HistTrade) -> ReconstructedTrade:
    other_asset = "QQQ" if hist.asset == "SPY" else "SPY"
    bars = await get_bars(provider, hist.asset, hist.trade_date)
    other_bars = await get_bars(provider, other_asset, hist.trade_date) or []
    if not bars:
        return ReconstructedTrade(
            hist=hist, iv=None, entry_spot=None, unreconstructable_reason="no_1m_bars"
        )

    entry_idx = find_bar_index_at_or_after(bars, hist.entry_time)
    if entry_idx is None:
        entry_idx = find_bar_index_at_or_before(bars, hist.entry_time)
    if entry_idx is None:
        return ReconstructedTrade(
            hist=hist, iv=None, entry_spot=None, unreconstructable_reason="entry_time_outside_bars"
        )

    entry_spot = bars[entry_idx]["close"]
    is_call = hist.direction == "CALL"
    t0 = time_to_expiry_years(bar_time(bars[entry_idx]), hist.trade_date)
    iv = implied_vol(hist.entry_price, entry_spot, hist.strike, t0, RISK_FREE_RATE, is_call)
    if iv is None:
        return ReconstructedTrade(
            hist=hist,
            iv=None,
            entry_spot=entry_spot,
            bars=bars,
            other_bars=other_bars,
            unreconstructable_reason="iv_unbracketed",
        )

    close_target = datetime(
        hist.trade_date.year,
        hist.trade_date.month,
        hist.trade_date.day,
        *SAFETY_CLOSE_ET,
        tzinfo=ET_TZ,
    )
    close_idx = find_bar_index_at_or_before(bars, close_target)
    if close_idx is None or close_idx <= entry_idx:
        close_idx = len(bars) - 1

    poll_indices = list(range(entry_idx, close_idx + 1, POLL_INTERVAL_MIN))
    if poll_indices[-1] != close_idx:
        poll_indices.append(close_idx)

    return ReconstructedTrade(
        hist=hist,
        iv=iv,
        entry_spot=entry_spot,
        poll_indices=poll_indices,
        bars=bars,
        other_bars=other_bars,
    )


def mark_at(rt: ReconstructedTrade, idx: int) -> float:
    b = rt.bars[idx]
    t = time_to_expiry_years(bar_time(b), rt.hist.trade_date)
    return bs_price(
        b["close"], rt.hist.strike, t, RISK_FREE_RATE, rt.iv or 0.0, rt.hist.direction == "CALL"
    )


def pnl_pct_at(rt: ReconstructedTrade, idx: int) -> float:
    mark = mark_at(rt, idx)
    return round((mark / rt.hist.entry_price - 1.0) * 100.0, 3)


# ───────────────────────────── policies ─────────────────────────────


@dataclass
class PolicyOutcome:
    policy: str
    exit_reason: str
    exit_mark: float
    pnl_pct: float
    pnl_dollars: float
    calls_used: int = 0


def _finalize(
    rt: ReconstructedTrade, policy: str, idx: int, reason: str, calls_used: int = 0
) -> PolicyOutcome:
    mark = mark_at(rt, idx)
    pnl_pct = round((mark / rt.hist.entry_price - 1.0) * 100.0, 3)
    pnl_dollars = round((mark - rt.hist.entry_price) * rt.hist.contracts * 100, 2)
    return PolicyOutcome(policy, reason, round(mark, 4), pnl_pct, pnl_dollars, calls_used)


def policy_current_rules(rt: ReconstructedTrade) -> PolicyOutcome:
    for idx in rt.poll_indices:
        pnl = pnl_pct_at(rt, idx)
        if pnl <= -50:
            return _finalize(rt, "current_rules", idx, "stop_loss")
        if pnl >= 100:
            return _finalize(rt, "current_rules", idx, "take_profit")
    return _finalize(rt, "current_rules", rt.poll_indices[-1], "held_to_close")


def policy_sl_only(rt: ReconstructedTrade) -> PolicyOutcome:
    for idx in rt.poll_indices:
        pnl = pnl_pct_at(rt, idx)
        if pnl <= -50:
            return _finalize(rt, "sl_only", idx, "stop_loss")
    return _finalize(rt, "sl_only", rt.poll_indices[-1], "held_to_close")


def policy_sl_trailing(rt: ReconstructedTrade, trailing_cfg: TrailingConfig) -> PolicyOutcome:
    peak = 0.0
    for idx in rt.poll_indices:
        pnl = pnl_pct_at(rt, idx)
        if pnl <= -50:
            return _finalize(rt, "sl_trailing", idx, "stop_loss")
        peak = max(peak, pnl)
        if trailing_cfg.enabled and peak >= trailing_cfg.activate_after_pct:
            trail_level = peak - trailing_cfg.trail_pct
            if pnl <= trail_level:
                return _finalize(rt, "sl_trailing", idx, "trailing_stop")
    return _finalize(rt, "sl_trailing", rt.poll_indices[-1], "held_to_close")


LOCAL_OPENCODE_MAX_RETRIES = 2


def _sim_invoke(model: str, prompt: str, timeout_sec: int) -> str | None:
    """Consult the model through exactly the production call path.

    An earlier version shelled out to a bare ``opencode run --dir /tmp
    --pure``, which never loaded the ``exit-advisor`` agent's system
    prompt (.opencode/agent/exit-advisor.md). The model then got the trade
    context without its instructions -- including "respond with ONLY a
    JSON object" -- answered in prose, and nearly every consult failed to
    parse, so policy (d) silently measured the trailing-stop fallback
    instead of the advisor. Delegating to the production invoke keeps the
    simulation and the live monitor from drifting apart again.
    """
    return exit_advisor._default_invoke(model, prompt, timeout_sec)


async def consult_with_retry(
    prompt: str, model: str, timeout_sec: float, max_retries: int = LOCAL_OPENCODE_MAX_RETRIES
) -> tuple[ConsultResult, int]:
    """Retry a consult up to ``max_retries`` times, but ONLY when the
    failure looks like a local opencode error (concurrent sessions from
    another agent), not a genuine model/timeout failure.

    Returns (result, physical_call_count).
    """
    attempts = 0
    result: ConsultResult | None = None
    for attempt in range(max_retries + 1):
        result = await exit_advisor.consult(prompt, model, timeout_sec, invoke_fn=_sim_invoke)
        attempts += 1
        if result.ok:
            return result, attempts
        if attempt < max_retries and any(
            s in (result.error or "") for s in LOCAL_OPENCODE_ERROR_SUBSTRINGS
        ):
            continue
        return result, attempts
    assert result is not None
    return result, attempts


@dataclass
class CallBudget:
    max_calls: int
    used: int = 0
    deadline_ts: float | None = None

    def exhausted(self) -> bool:
        if self.used >= self.max_calls:
            return True
        return self.deadline_ts is not None and time.monotonic() >= self.deadline_ts


async def policy_sl_advisor(
    rt: ReconstructedTrade,
    advisor_cfg: ExitAdvisorConfig,
    exit_cfg: ExitConfig,
    model: str,
    budget: CallBudget,
) -> PolicyOutcome:
    state = AdvisorState()
    calls_this_trade = 0
    for idx in rt.poll_indices:
        pnl = pnl_pct_at(rt, idx)
        if pnl <= -50:
            return _finalize(rt, "sl_advisor", idx, "stop_loss", calls_this_trade)

        update_peak(state, pnl)
        b = rt.bars[idx]
        mins_left = (
            datetime(
                rt.hist.trade_date.year,
                rt.hist.trade_date.month,
                rt.hist.trade_date.day,
                *SAFETY_CLOSE_ET,
                tzinfo=ET_TZ,
            )
            - bar_time(b)
        ).total_seconds() / 60.0

        trigger = decide_trigger(pnl, state, bar_time(b), mins_left, advisor_cfg)
        if trigger is None:
            continue

        if state.calls_made >= advisor_cfg.max_calls_per_trade or budget.exhausted():
            if trailing_fallback_triggered(pnl, state, exit_cfg):
                return _finalize(rt, "sl_advisor", idx, "trailing_stop", calls_this_trade)
            continue

        causal_bars = rt.bars[: idx + 1]
        causal_other = [b2 for b2 in rt.other_bars if b2["timestamp"] <= b["timestamp"]]
        market_context = {
            "underlying_path": summarize_bars(causal_bars),
            "co_movement": {
                ("QQQ" if rt.hist.asset == "SPY" else "SPY"): summarize_bars(causal_other)
            },
        }
        context = build_context(
            trigger=trigger,
            asset=rt.hist.asset,
            direction=rt.hist.direction,
            prediction={
                "direction": rt.hist.direction,
                "note": "historical replay -- original rationale not reattached",
            },
            entry_time=rt.hist.entry_time.isoformat(),
            entry_price=rt.hist.entry_price,
            current_mark=round(mark_at(rt, idx), 4),
            current_bid=None,
            current_ask=None,
            underlying_spot=b["close"],
            strike=rt.hist.strike,
            pnl_pct=pnl,
            peak_pnl_pct=state.peak_pnl_pct,
            market_context=market_context,
            minutes_to_deadline=mins_left,
            sl_level=round(rt.hist.entry_price * 0.5, 4),
            time_deadline_est=exit_cfg.time_deadline_est,
            calls_made=state.calls_made,
            max_calls_per_trade=advisor_cfg.max_calls_per_trade,
        )
        prompt = build_prompt(context)

        state.calls_made += 1
        result, physical_calls = await consult_with_retry(prompt, model, advisor_cfg.timeout_sec)
        budget.used += physical_calls
        calls_this_trade += physical_calls

        if not result.ok:
            print(  # noqa: T201
                f"    [WARN] consult failed trigger={trigger} error={result.error!r}",
                file=sys.stderr,
            )
            if trailing_fallback_triggered(pnl, state, exit_cfg):
                return _finalize(rt, "sl_advisor", idx, "trailing_stop", calls_this_trade)
            continue

        state.apply_trail_tightening(result.trail_stop_pct, exit_cfg.trailing.trail_pct)
        if result.action == "EXIT":
            return _finalize(rt, "sl_advisor", idx, "advisor_exit", calls_this_trade)
        # HOLD: fall through to the next poll. Even once the budget is
        # exhausted, later polls in THIS loop still hit the
        # `state.calls_made >= cap or budget.exhausted()` branch above,
        # which keeps checking the deterministic trailing-stop fallback
        # -- the SL check at the top of the loop always keeps running
        # regardless. Nothing here may skip straight to held_to_close.

    return _finalize(rt, "sl_advisor", rt.poll_indices[-1], "held_to_close", calls_this_trade)


# ────────────────────────── reconciliation check ──────────────────────────


def check_reconstruction(rt: ReconstructedTrade) -> dict[str, Any] | None:
    """Compare the reconstructed mark at the REAL exit time to the REAL exit price."""
    if rt.iv is None or rt.hist.actual_exit_time is None or rt.hist.actual_exit_price is None:
        return None
    idx = find_bar_index_at_or_before(rt.bars, rt.hist.actual_exit_time)
    if idx is None:
        idx = find_bar_index_at_or_after(rt.bars, rt.hist.actual_exit_time)
    if idx is None:
        return None
    recon_mark = mark_at(rt, idx)
    real = rt.hist.actual_exit_price
    diff = recon_mark - real
    pct_diff = (diff / real * 100.0) if real else None
    return {
        "trade_id": rt.hist.trade_id,
        "actual_exit_reason": rt.hist.actual_exit_reason,
        "real_exit_price": real,
        "reconstructed_mark": round(recon_mark, 4),
        "diff": round(diff, 4),
        "pct_diff": round(pct_diff, 1) if pct_diff is not None else None,
    }


# ────────────────────────────── main ──────────────────────────────


async def run(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    trades = load_candidate_trades(log_dir, start, end)
    print(f"Found {len(trades)} resolved trades in {start}..{end}:")  # noqa: T201
    for t in trades:
        print(  # noqa: T201
            f"  {t.trade_date} {t.asset} {t.direction} entry=${t.entry_price:.2f} "
            f"strike={t.strike} actual_exit={t.actual_exit_reason} "
            f"actual_pnl=${t.actual_final_pnl}"
        )

    provider = YahooFinanceProvider()
    recon: list[ReconstructedTrade] = []
    for t in trades:
        rt = await reconstruct_trade(provider, t)
        recon.append(rt)

    usable = [rt for rt in recon if rt.iv is not None]
    unusable = [rt for rt in recon if rt.iv is None]
    print(  # noqa: T201
        f"\n{len(usable)}/{len(recon)} trades have a reconstructable BS path "
        "(usable IV backed out from entry fill)."
    )
    for rt in unusable:
        print(  # noqa: T201
            f"  SKIPPED {rt.hist.trade_date} {rt.hist.asset} {rt.hist.direction}: "
            f"{rt.unreconstructable_reason}"
        )

    if not usable:
        print("\nNo usable trades -- nothing to simulate.")  # noqa: T201
        return

    # ── reconstruction sanity check ──
    print("\n=== Mark-reconstruction check (reconstructed vs REAL recorded exit) ===")  # noqa: T201
    checks = []
    for rt in usable:
        c = check_reconstruction(rt)
        if c:
            checks.append(c)
            flag = (
                " <-- LARGE MISMATCH"
                if c["pct_diff"] is not None and abs(c["pct_diff"]) > 30
                else ""
            )
            print(  # noqa: T201
                f"  {c['trade_id']} ({c['actual_exit_reason']}): real=${c['real_exit_price']:.2f} "
                f"recon=${c['reconstructed_mark']:.2f} diff=${c['diff']:+.2f} "
                f"({c['pct_diff']}%){flag}"
            )
    large_mismatches = [c for c in checks if c["pct_diff"] is not None and abs(c["pct_diff"]) > 30]
    if large_mismatches:
        print(  # noqa: T201
            f"\n  HONEST CAVEAT: {len(large_mismatches)}/{len(checks)} trades show a >30% mismatch "
            "between the reconstructed BS mark and the real recorded exit price at that same "
            "moment. The frozen-IV, no-spread Black-Scholes reconstruction does NOT precisely "
            "reproduce real historical fills -- treat the policy comparison below as indicative, "
            "not as a validated backtest."
        )
    else:
        print(  # noqa: T201
            "\n  Reconstructed marks broadly track the real recorded exits at the same timestamps."
        )

    # ── policies (a)(b)(c): free, no LLM calls ──
    trailing_cfg = TrailingConfig(activate_after_pct=30, trail_pct=15)
    exit_cfg = ExitConfig(trailing=trailing_cfg, time_deadline_est="15:25")
    results: dict[str, list[PolicyOutcome]] = {
        "current_rules": [],
        "sl_only": [],
        "sl_trailing": [],
        "sl_advisor": [],
    }
    for rt in usable:
        results["current_rules"].append(policy_current_rules(rt))
        results["sl_only"].append(policy_sl_only(rt))
        results["sl_trailing"].append(policy_sl_trailing(rt, trailing_cfg))

    # ── policy (d): the real advisor, calling opencode ──
    advisor_cfg = ExitAdvisorConfig(
        enabled=True,
        model=args.model,
        timeout_sec=args.timeout_sec,
        profit_step_pct=50,
        giveback_from_peak_pct=25,
        periodic_interval_min=15,
        final_window_min=30,
        max_calls_per_trade=12,
    )
    budget = CallBudget(
        max_calls=args.max_calls,
        deadline_ts=time.monotonic() + args.time_budget_sec if args.time_budget_sec else None,
    )
    print(  # noqa: T201
        f"\n=== Running policy (d): SL + advisor ({args.model}, concurrency={args.concurrency}, "
        f"max_calls={args.max_calls}, time_budget_sec={args.time_budget_sec}) ==="
    )
    for rt in usable:
        if budget.exhausted():
            print(  # noqa: T201
                f"  [BUDGET EXHAUSTED] {rt.hist.trade_date} {rt.hist.asset}: no more opencode "
                "calls available -- still applying SL + deterministic trailing fallback (never "
                "skips the hard rail)."
            )
        # Always run the real policy function, even with the budget already
        # exhausted: it internally never calls the model once
        # budget.exhausted() is True, but it MUST keep checking the SL hard
        # rail and the deterministic trailing-stop fallback at every poll --
        # skipping straight to held_to_close here would violate "hard rails
        # always run first" for the remainder of the session.
        t0 = time.monotonic()
        outcome = await policy_sl_advisor(rt, advisor_cfg, exit_cfg, args.model, budget)
        elapsed = time.monotonic() - t0
        print(  # noqa: T201
            f"  {rt.hist.trade_date} {rt.hist.asset} {rt.hist.direction}: exit={outcome.exit_reason} "
            f"pnl=${outcome.pnl_dollars:+.2f} calls={outcome.calls_used} elapsed={elapsed:.1f}s "
            f"(budget used so far: {budget.used}/{budget.max_calls})"
        )
        results["sl_advisor"].append(outcome)

    # ── report ──
    print("\n" + "=" * 78)  # noqa: T201
    print(f"POLICY COMPARISON (usable trades only, n={len(usable)})")  # noqa: T201
    print("=" * 78)  # noqa: T201
    header = (
        f"{'Policy':<16} {'Total P&L':>12} {'Avg P&L':>10} {'Avg P&L%':>10} "
        f"{'Win rate':>9} {'Calls':>7}"
    )
    print(header)  # noqa: T201
    print("-" * len(header))  # noqa: T201
    for policy, outcomes in results.items():
        total = sum(o.pnl_dollars for o in outcomes)
        avg = total / len(outcomes) if outcomes else 0.0
        avg_pct = sum(o.pnl_pct for o in outcomes) / len(outcomes) if outcomes else 0.0
        wins = sum(1 for o in outcomes if o.pnl_dollars > 0)
        win_rate = wins / len(outcomes) * 100 if outcomes else 0.0
        calls = sum(o.calls_used for o in outcomes)
        print(  # noqa: T201
            f"{policy:<16} {total:>+12.2f} {avg:>+10.2f} {avg_pct:>+9.1f}% {win_rate:>8.0f}% {calls:>7}"
        )

    print("\nPer-trade detail:")  # noqa: T201
    for i, rt in enumerate(usable):
        row = [
            f"{rt.hist.trade_date} {rt.hist.asset} {rt.hist.direction} "
            f"(real: {rt.hist.actual_exit_reason}, ${rt.hist.actual_final_pnl})"
        ]
        for policy in results:
            o = results[policy][i]
            row.append(f"{policy}={o.exit_reason}/${o.pnl_dollars:+.2f}")
        print("  " + " | ".join(row))  # noqa: T201

    print(  # noqa: T201
        "\nCaveats: Black-Scholes marks use a single implied vol backed out from the entry "
        "fill and held constant through the session (no vol surface, no skew, no bid/ask "
        "spread, calendar-time T). n is small (see counts above) -- read this as directional "
        "evidence, not a validated backtest. See the mark-reconstruction check above for how "
        "well the reconstruction tracks real recorded exits."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline exit-policy simulation")
    default_end = date.today() - timedelta(days=1)
    default_start = date.today() - timedelta(days=30)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--start", default=default_start.isoformat())
    parser.add_argument("--end", default=default_end.isoformat())
    parser.add_argument("--model", default="opencode/muse-spark-1.3-contributor-free")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-calls", type=int, default=150)
    parser.add_argument("--timeout-sec", type=float, default=55.0)
    parser.add_argument(
        "--time-budget-sec",
        type=float,
        default=1200.0,
        help="Wall-clock cap on policy (d)'s total opencode call time, beyond which "
        "remaining trigger events fall back to the deterministic trailing rule. Not part "
        "of the production advisor -- a pragmatic safeguard for this one-off script so it "
        "terminates in bounded time. Set 0 to disable.",
    )
    args = parser.parse_args()
    if args.concurrency != 1:
        print(  # noqa: T201
            f"[WARN] --concurrency {args.concurrency} requested but this simulation only "
            "supports 1 (matches execution.max_concurrent_trades=1 and the real intraday "
            "monitor's sequential-per-trade processing) -- forcing 1.",
            file=sys.stderr,
        )
    if not args.time_budget_sec:
        args.time_budget_sec = None

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
