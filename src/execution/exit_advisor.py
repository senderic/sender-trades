"""LLM-driven profit-exit advisor for open 0DTE options trades.

Replaces the fixed +100% take-profit limit order. A replay of 72
asset-days (2026-07-22..09-10) found this strategy's out-of-the-money
0DTE options pay off on only a handful of big days -- holding a winner
to expiry averaged ~+71%/trade in premium units, with the top 3 days
carrying ~45% of total gains. Under the fixed +100% TP / -50% SL, that
fell to ~+7-9%: the resting +100% limit order fills and cuts off
exactly the days that pay.

This module is consulted by :mod:`src.execution.intraday_monitor` --
NOT on every 3-minute poll, but at event-driven cadence points (see
:func:`decide_trigger`) -- to decide HOLD vs EXIT against the heavy
model configured at ``execution.exit_advisor.model`` (falls back to
``llm.primary_model``). It never runs before the deterministic hard
rails (stop-loss, time deadline, safety-close sweep), which stay
entirely in :mod:`src.execution.intraday_monitor` / ``safety_close.sh``
and are always evaluated first, unconditionally, regardless of
anything below.

State (peak P&L, calls made, cadence bookkeeping, any advisor-tightened
trailing stop) is persisted into the trade's own audit JSON under the
``advisor_state`` key, and reloaded fresh on every monitor run --
each cron invocation is a new process, so nothing here can rely on
in-memory state surviving between polls. Persistence and the exit
decision itself both go through the same audit file already protected
by the intraday monitor's flock lock (``.intraday_monitor.lock``), so
no additional locking is needed here.

If the model call fails, times out, or returns unparseable output --
or the per-trade call budget (``max_calls_per_trade``) is exhausted --
this falls back to enforcing the configured trailing stop
(``execution.exit_strategy.trailing``) deterministically instead, and
logs a WARNING. The advisor can tighten (never loosen) that trailing
stop via its own ``trail_stop_pct`` output.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from datetime import time as dtime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from src.llm.trade_signal import _parse_pick as _extract_json
from src.timezone import ET_TZ

if TYPE_CHECKING:
    from src.execution.intraday_monitor import OpenTrade
    from src.execution.models import ExecutionConfig, ExitAdvisorConfig, ExitConfig

logger = structlog.get_logger()

VALID_ACTIONS = ("HOLD", "EXIT")


# ─────────────────────────── persisted state ────────────────────────────


@dataclass
class AdvisorState:
    """Per-trade advisor bookkeeping, persisted in the trade audit JSON.

    Reconstructed fresh from the audit file on every monitor run (see
    module docstring) via :meth:`from_dict`, mutated in memory during
    one pass, then written back via :meth:`to_dict`.
    """

    peak_pnl_pct: float = 0.0
    calls_made: int = 0
    last_pnl_bucket: int = 0
    # Peak level (in P&L %) the give-back ladder is currently anchored
    # to. Reset (along with `giveback_bucket`) whenever `peak_pnl_pct`
    # sets a new high -- see `decide_trigger`.
    last_giveback_peak: float = 0.0
    # How many `giveback_from_peak_pct` steps BELOW `last_giveback_peak`
    # have already been consulted for (peak-25 = 1, peak-50 = 2, ...).
    # Lets the give-back trigger re-fire on a continued slide instead of
    # firing once and then going silent until a new peak is set.
    giveback_bucket: int = 0
    # One-shot: has the "P&L round-tripped from a real peak back below
    # zero" trigger already fired for this trade?
    zero_cross_consulted: bool = False
    last_periodic_ts: datetime | None = None
    final_window_consulted: bool = False
    # Advisor-tightened trailing stop percentage. None means "no advisor
    # tightening yet -- the deterministic fallback trail
    # (execution.exit_strategy.trailing.trail_pct) applies only on
    # consult failure / budget exhaustion, not on every poll". Once set,
    # it is the advisor's own deliberate choice and IS enforced on every
    # poll (see `advisor_trail_triggered`), not just as a fallback. Can
    # only ever be tightened (made smaller) over the life of a trade --
    # see `apply_trail_tightening`.
    trail_stop_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_pnl_pct": self.peak_pnl_pct,
            "calls_made": self.calls_made,
            "last_pnl_bucket": self.last_pnl_bucket,
            "last_giveback_peak": self.last_giveback_peak,
            "giveback_bucket": self.giveback_bucket,
            "zero_cross_consulted": self.zero_cross_consulted,
            "last_periodic_ts": (
                self.last_periodic_ts.isoformat() if self.last_periodic_ts else None
            ),
            "final_window_consulted": self.final_window_consulted,
            "trail_stop_pct": self.trail_stop_pct,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AdvisorState:
        if not data:
            return cls()
        ts_raw = data.get("last_periodic_ts")
        parsed_ts: datetime | None = None
        if ts_raw:
            try:
                parsed_ts = datetime.fromisoformat(str(ts_raw))
            except ValueError:
                parsed_ts = None
        try:
            peak = float(data.get("peak_pnl_pct", 0.0) or 0.0)
        except (TypeError, ValueError):
            peak = 0.0
        try:
            calls = int(data.get("calls_made", 0) or 0)
        except (TypeError, ValueError):
            calls = 0
        try:
            bucket = int(data.get("last_pnl_bucket", 0) or 0)
        except (TypeError, ValueError):
            bucket = 0
        try:
            giveback_peak = float(data.get("last_giveback_peak", 0.0) or 0.0)
        except (TypeError, ValueError):
            giveback_peak = 0.0
        try:
            giveback_bucket = int(data.get("giveback_bucket", 0) or 0)
        except (TypeError, ValueError):
            giveback_bucket = 0
        trail = data.get("trail_stop_pct")
        try:
            trail = float(trail) if trail is not None else None
        except (TypeError, ValueError):
            trail = None
        return cls(
            peak_pnl_pct=peak,
            calls_made=calls,
            last_pnl_bucket=bucket,
            last_giveback_peak=giveback_peak,
            giveback_bucket=giveback_bucket,
            zero_cross_consulted=bool(data.get("zero_cross_consulted", False)),
            last_periodic_ts=parsed_ts,
            final_window_consulted=bool(data.get("final_window_consulted", False)),
            trail_stop_pct=trail,
        )

    def apply_trail_tightening(self, proposed: float | None, default_trail_pct: float) -> None:
        """Tighten the effective trailing stop, never loosen it.

        Args:
            proposed: The advisor's ``trail_stop_pct`` output, or None.
            default_trail_pct: The configured default trail percentage,
                used as the starting point when the advisor has never
                tightened it before.
        """
        if proposed is None or proposed <= 0:
            return
        current = self.trail_stop_pct if self.trail_stop_pct is not None else default_trail_pct
        if proposed < current:
            self.trail_stop_pct = proposed

    def effective_trail_pct(self, default_trail_pct: float) -> float:
        return self.trail_stop_pct if self.trail_stop_pct is not None else default_trail_pct


def update_peak(state: AdvisorState, pnl_pct: float) -> AdvisorState:
    """Update the recorded peak P&L in place. Always called every poll."""
    if pnl_pct > state.peak_pnl_pct:
        state.peak_pnl_pct = pnl_pct
    return state


def advisor_trail_triggered(pnl_pct: float, state: AdvisorState, exit_cfg: ExitConfig) -> bool:
    """Enforce the advisor's OWN tightened trailing stop, on EVERY poll.

    Unlike :func:`trailing_fallback_triggered` (which only ever runs as a
    degraded-mode fallback -- consult failure or budget exhaustion), this
    checks the trail the ADVISOR ITSELF explicitly asked for via its
    ``trail_stop_pct`` output. Once the advisor has set one
    (``state.trail_stop_pct is not None``), that is the advisor's own
    deliberate choice to protect the position between consultations --
    not a blanket rule -- so enforcing it on every poll does not
    reintroduce the tail-cutting the advisor exists to avoid; it only
    tightens the specific trades the advisor itself flagged.

    Deliberately does NOT gate on ``exit_cfg.trailing.activate_after_pct``
    the way the deterministic fallback does: the advisor setting a trail
    at all IS the activation signal, regardless of where the configured
    default would have kicked in. Still respects ``trailing.enabled`` as
    the overall kill switch.
    """
    if not exit_cfg.trailing.enabled:
        return False
    if state.trail_stop_pct is None:
        return False
    trail_level = state.peak_pnl_pct - state.trail_stop_pct
    return pnl_pct <= trail_level


# ────────────────────────────── cadence ──────────────────────────────


def decide_trigger(
    pnl_pct: float,
    state: AdvisorState,
    now: datetime,
    minutes_to_deadline: float,
    config: ExitAdvisorConfig,
) -> str | None:
    """Decide whether this poll should consult the advisor, and why.

    Mutates ``state``'s cadence bookkeeping fields (only for the trigger
    that actually fires -- at most one per call), so repeated calls at
    the same P&L/time don't re-fire the same trigger. Callers should
    call :func:`update_peak` first so ``state.peak_pnl_pct`` reflects
    this poll's ``pnl_pct`` before triggers are evaluated.

    Priority when multiple conditions are true simultaneously:
    profit_step > giveback > zero_cross > final_window > periodic.

    Returns:
        One of ``"profit_step"``, ``"giveback"``, ``"zero_cross"``,
        ``"final_window"``, ``"periodic"``, or ``None`` if nothing
        should fire this poll.
    """
    step = config.profit_step_pct
    if step > 0 and pnl_pct >= step:
        bucket = int(pnl_pct // step)
        if bucket > state.last_pnl_bucket:
            state.last_pnl_bucket = bucket
            # Any consult -- whatever triggered it -- resets the periodic
            # check-in clock, so a profit-step/giveback/final-window
            # consult right before a periodic one is due doesn't
            # immediately trigger a near-duplicate periodic consult too.
            state.last_periodic_ts = now
            return "profit_step"

    if config.giveback_from_peak_pct > 0 and state.peak_pnl_pct > 0:
        gb_step = config.giveback_from_peak_pct
        # A new high resets the ladder: it's a fresh peak to give back
        # from, not a continuation of the old descent.
        if state.peak_pnl_pct > state.last_giveback_peak:
            state.last_giveback_peak = state.peak_pnl_pct
            state.giveback_bucket = 0
        drop = state.last_giveback_peak - pnl_pct
        if drop >= gb_step:
            gb_bucket = int(drop // gb_step)
            # Re-fires at each further step down (peak-25, peak-50,
            # peak-75, ...), not just once per peak, so a winner sliding
            # all the way back to the stop keeps getting consulted
            # instead of going quiet after the first give-back.
            if gb_bucket > state.giveback_bucket:
                state.giveback_bucket = gb_bucket
                state.last_periodic_ts = now
                return "giveback"

    if (
        config.profit_step_pct > 0
        and state.peak_pnl_pct >= config.profit_step_pct
        and pnl_pct < 0
        and not state.zero_cross_consulted
    ):
        # One-shot: the position round-tripped from a real peak (at
        # least one profit_step milestone) back into a loss. The
        # give-back ladder above is anchored to peak_pnl_pct/step
        # increments and could in principle skip straight past this in
        # one poll (e.g. a sharp single-bar reversal); this is a direct,
        # unconditional backstop for that specific "was winning,
        # now losing" event regardless of the ladder's bucket math.
        state.zero_cross_consulted = True
        state.last_periodic_ts = now
        return "zero_cross"

    if (
        minutes_to_deadline <= config.final_window_min
        and minutes_to_deadline >= 0
        and not state.final_window_consulted
    ):
        state.final_window_consulted = True
        state.last_periodic_ts = now
        return "final_window"

    if pnl_pct > 0 and config.periodic_interval_min > 0:
        if state.last_periodic_ts is None:
            state.last_periodic_ts = now
            return "periodic"
        elapsed_min = (now - state.last_periodic_ts).total_seconds() / 60.0
        if elapsed_min >= config.periodic_interval_min:
            state.last_periodic_ts = now
            return "periodic"

    return None


def minutes_to_deadline(now: datetime, time_deadline_est: str) -> float:
    """Minutes remaining until ``time_deadline_est`` (``HH:MM``, ET) today."""
    parts = time_deadline_est.split(":")
    deadline_time = dtime(int(parts[0]), int(parts[1]))
    deadline_dt = now.astimezone(ET_TZ).replace(
        hour=deadline_time.hour, minute=deadline_time.minute, second=0, microsecond=0
    )
    return (deadline_dt - now.astimezone(ET_TZ)).total_seconds() / 60.0


# ───────────────────────── trailing-stop fallback ─────────────────────────


def trailing_fallback_triggered(pnl_pct: float, state: AdvisorState, exit_cfg: ExitConfig) -> bool:
    """Deterministic trailing-stop check used when the advisor is unavailable.

    Only ever consulted as a fallback (see module docstring) -- not run
    on every poll -- because a background trailing check on every poll
    would defeat the entire point of consulting the advisor for
    "let winners run" discretion.
    """
    trailing = exit_cfg.trailing
    if not trailing.enabled:
        return False
    if state.peak_pnl_pct < trailing.activate_after_pct:
        return False
    trail_pct = state.effective_trail_pct(trailing.trail_pct)
    trail_level_pct = state.peak_pnl_pct - trail_pct
    return pnl_pct <= trail_level_pct


# ─────────────────────────── response parsing ───────────────────────────


def parse_advisor_response(text: str | None) -> dict[str, Any] | None:
    """Defensively parse the exit-advisor's JSON response.

    Mirrors ``src.llm.graph._extract_json`` -- tolerant of fenced/mixed
    text, scans for a balanced ``{...}`` object. Additionally validates
    and coerces the advisor-specific schema; returns ``None`` for
    anything that doesn't yield a usable ``action``.
    """
    if not text:
        return None
    parsed = _extract_json(text)
    if not isinstance(parsed, dict):
        return None

    action = str(parsed.get("action", "")).strip().upper()
    if action not in VALID_ACTIONS:
        return None

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reason = str(parsed.get("reason", "") or "")

    trail_stop_pct = parsed.get("trail_stop_pct")
    try:
        trail_stop_pct = float(trail_stop_pct) if trail_stop_pct is not None else None
    except (TypeError, ValueError):
        trail_stop_pct = None
    if trail_stop_pct is not None and trail_stop_pct <= 0:
        trail_stop_pct = None

    return {
        "action": action,
        "confidence": confidence,
        "reason": reason,
        "trail_stop_pct": trail_stop_pct,
    }


# ────────────────────────────── model call ──────────────────────────────


def resolve_model(advisor_cfg: ExitAdvisorConfig, llm_primary_model: str) -> str:
    """Resolve the model to consult: ``exit_advisor.model`` or ``llm.primary_model``."""
    return advisor_cfg.model or llm_primary_model


@dataclass
class ConsultResult:
    """Outcome of one advisor consultation attempt."""

    ok: bool
    model: str
    latency_sec: float
    action: str | None = None
    confidence: float | None = None
    reason: str = ""
    trail_stop_pct: float | None = None
    raw: str | None = None
    error: str = ""


def _default_invoke(model: str, prompt: str, timeout_sec: int) -> str | None:
    """Real opencode call: a fresh single-model client, no fallback chain."""
    from src.config import LLMConfig  # deferred: avoids import cycle at module load
    from src.llm.client import OpencodeLLMClient

    llm_cfg = LLMConfig(
        enabled=True,
        primary_model=model,
        fallback_models=[],
        timeout_sec=timeout_sec,
        max_calls_per_run=1,
    )
    client = OpencodeLLMClient(llm_cfg)
    return client.invoke_agent("exit-advisor", prompt, timeout_sec=timeout_sec)


InvokeFn = Callable[[str, str, int], "str | None"]


async def consult(
    prompt: str,
    model: str,
    timeout_sec: float,
    invoke_fn: InvokeFn | None = None,
) -> ConsultResult:
    """Consult the exit-advisor model once, off the event loop, under a hard timeout.

    Args:
        prompt: The rendered advisor prompt (see :func:`build_prompt`).
        model: Resolved model id (see :func:`resolve_model`).
        timeout_sec: Per-attempt timeout. Kept well inside the 3-minute
            cron interval by :class:`~src.execution.models.ExitAdvisorConfig`.
        invoke_fn: Injectable ``(model, prompt, timeout_sec) -> str | None``
            callable, defaulting to a real opencode call. Tests pass a
            stub here instead of shelling out.
    """
    fn = invoke_fn or _default_invoke
    t0 = time.monotonic()
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(fn, model, prompt, int(timeout_sec)),
            timeout=timeout_sec + 5,
        )
    except TimeoutError:
        return ConsultResult(
            ok=False, model=model, latency_sec=time.monotonic() - t0, error="timeout"
        )
    except Exception as e:  # pragma: no cover - defensive
        return ConsultResult(
            ok=False,
            model=model,
            latency_sec=time.monotonic() - t0,
            error=f"{type(e).__name__}: {e}",
        )
    elapsed = time.monotonic() - t0

    if not response:
        return ConsultResult(ok=False, model=model, latency_sec=elapsed, error="empty_response")

    parsed = parse_advisor_response(response)
    if parsed is None:
        return ConsultResult(
            ok=False, model=model, latency_sec=elapsed, raw=response[:500], error="unparseable"
        )

    return ConsultResult(ok=True, model=model, latency_sec=elapsed, raw=response[:1000], **parsed)


# ────────────────────────────── prompt input ──────────────────────────────


def compute_pnl_pct(entry_price: float | None, mark: float) -> float:
    if not entry_price or entry_price <= 0:
        return 0.0
    return round((mark / entry_price - 1.0) * 100.0, 2)


async def gather_market_context(
    asset: str,
    now: datetime,
    candle_fetcher: Callable[[str, Any, int], Awaitable[list[dict] | None]] | None = None,
) -> dict[str, Any]:
    """Best-effort compressed underlying path + co-movement context.

    Never raises: any failure to fetch bars degrades to ``"unavailable"``
    fields in the returned context rather than breaking the advisor
    consultation (a missing chart is a reason for the advisor to be more
    cautious, not a reason to crash the monitor).

    Args:
        asset: The traded asset ("SPY" or "QQQ").
        now: Current time, used to pick "today" for the intraday fetch.
        candle_fetcher: Injectable ``(symbol, date, resolution_min) ->
            list[bar]`` coroutine, defaulting to Yahoo Finance via
            :class:`~src.ingestion.candle_providers.YahooFinanceProvider`.
            Each bar dict has ``timestamp/open/high/low/close/volume``.
    """
    if candle_fetcher is None:
        from src.ingestion.candle_providers import YahooFinanceProvider

        provider = YahooFinanceProvider()

        async def candle_fetcher(
            symbol: str, target_date: Any, resolution: int
        ) -> list[dict] | None:  # type: ignore[misc]
            return await provider.fetch_intraday_candles(symbol, target_date, resolution)

    other_asset = "QQQ" if asset == "SPY" else "SPY"
    today = now.astimezone(ET_TZ).date()

    async def _summary(symbol: str) -> dict[str, Any]:
        try:
            bars = await candle_fetcher(symbol, today, 5)
        except Exception as e:
            logger.debug("exit_advisor_bars_fetch_failed", symbol=symbol, error=str(e))
            bars = None
        if not bars:
            return {"available": False}

        opens = [b["open"] for b in bars]
        highs = [b["high"] for b in bars]
        lows = [b["low"] for b in bars]
        closes = [b["close"] for b in bars]
        volumes = [b.get("volume", 0) or 0 for b in bars]
        day_open = opens[0]
        day_high = max(highs)
        day_low = min(lows)
        last_close = closes[-1]
        total_vol = sum(volumes) or 1
        typical = [(hi + lo + c) / 3.0 for hi, lo, c in zip(highs, lows, closes, strict=True)]
        vwap = sum(t * v for t, v in zip(typical, volumes, strict=True)) / total_vol

        recent = bars[-24:]  # last ~2 hours of 5-min bars
        compact = [{"t": b["timestamp"], "c": round(b["close"], 4)} for b in recent]

        return {
            "available": True,
            "open": round(day_open, 4),
            "high": round(day_high, 4),
            "low": round(day_low, 4),
            "last": round(last_close, 4),
            "vwap": round(vwap, 4),
            "pct_change_from_open": (
                round((last_close / day_open - 1.0) * 100.0, 3) if day_open else None
            ),
            "recent_5min_bars": compact,
        }

    own = await _summary(asset)
    other = await _summary(other_asset)
    return {
        "underlying_path": own,
        "co_movement": {other_asset: other},
    }


def build_prompt(context: dict[str, Any]) -> str:
    """Render the advisor context dict into the user prompt text."""
    return (
        "Decide HOLD or EXIT for this open 0DTE position. Full context follows as JSON:\n\n"
        f"{json.dumps(context, indent=2, default=str)}\n\n"
        "Respond with ONLY the JSON object described in your instructions."
    )


def build_context(
    *,
    trigger: str,
    asset: str,
    direction: str,
    prediction: dict[str, Any],
    entry_time: str,
    entry_price: float,
    current_mark: float,
    current_bid: float | None,
    current_ask: float | None,
    underlying_spot: float | None,
    strike: float,
    pnl_pct: float,
    peak_pnl_pct: float,
    market_context: dict[str, Any],
    minutes_to_deadline: float,
    sl_level: float,
    time_deadline_est: str,
    calls_made: int,
    max_calls_per_trade: int,
) -> dict[str, Any]:
    """Assemble the full advisor input context dict (see AGENTS.md/prompt for shape)."""
    distance_pct = None
    if underlying_spot and underlying_spot > 0 and strike:
        if direction == "PUT":
            distance_pct = round((underlying_spot - strike) / underlying_spot * 100.0, 3)
        else:
            distance_pct = round((strike - underlying_spot) / underlying_spot * 100.0, 3)

    return {
        "trigger": trigger,
        "asset": asset,
        "direction": direction,
        "prediction": prediction,
        "entry": {"time": entry_time, "option_entry_price": entry_price},
        "current": {
            "option_mark": current_mark,
            "option_bid": current_bid,
            "option_ask": current_ask,
            "underlying_spot": underlying_spot,
        },
        "strike_distance": {
            "strike": strike,
            "distance_pct": distance_pct,
            "note": "positive = underlying still needs to move further ITM in the predicted direction",
        },
        "pnl": {"now_pct": pnl_pct, "peak_pct": peak_pnl_pct},
        **market_context,
        "minutes_to_deadline": round(minutes_to_deadline, 1),
        "rails": {
            "stop_loss_level": sl_level,
            "time_deadline_est": time_deadline_est,
            "note": "already enforced deterministically before you were consulted; informational only",
        },
        "advisor_budget": {"calls_made": calls_made, "max_calls_per_trade": max_calls_per_trade},
    }


# ───────────────────────────── orchestration ─────────────────────────────


@dataclass
class ExitDecision:
    """What :func:`process` decided this poll."""

    should_exit: bool
    exit_reason: str = ""  # "advisor_exit" | "trailing_stop"
    limit_mult: float = 0.85
    consult: ConsultResult | None = None


def _write_trade_data(path: Path, data: dict[str, Any]) -> None:
    """Atomic tmp + os.replace write, matching TradeContext's pattern."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)


def _record_consultation(
    trade: OpenTrade,
    *,
    trigger: str,
    result: ConsultResult | None,
    action_taken: str,
    now: datetime,
) -> None:
    """Append a consultation record into trade.data['entries'] in place."""
    entry: dict[str, Any] = {
        "event_type": "advisor_consultation",
        "timestamp": now.isoformat(),
        "trigger": trigger,
        "action_taken": action_taken,
    }
    if result is not None:
        entry["model"] = result.model
        entry["latency_sec"] = round(result.latency_sec, 2)
        entry["ok"] = result.ok
        if result.ok:
            entry["parsed"] = {
                "action": result.action,
                "confidence": result.confidence,
                "reason": result.reason,
                "trail_stop_pct": result.trail_stop_pct,
            }
        else:
            entry["error"] = result.error
    events = trade.data.get("entries")
    if not isinstance(events, list):
        events = []
        trade.data["entries"] = events
    events.append(entry)


async def process(
    trade: OpenTrade,
    mark: float,
    quote: dict[str, Any] | None,
    underlying_spot: float | None,
    exec_config: ExecutionConfig,
    llm_primary_model: str,
    now: datetime | None = None,
    invoke_fn: InvokeFn | None = None,
    market_context_fn: Callable[[str, datetime], Awaitable[dict[str, Any]]] | None = None,
) -> ExitDecision | None:
    """Run one monitor-pass worth of advisor logic for a single open trade.

    Precondition: the caller (``src.execution.intraday_monitor``) has
    ALREADY evaluated and cleared the stop-loss hard rail for ``mark``
    before calling this -- this function only ever manages the profit
    exit path and must never be consulted in place of the stop-loss.

    Mutates ``trade.data`` in place (advisor_state + any consultation
    entries). If this call does not result in an exit, the updated
    ``trade.data`` is written straight back to ``trade.path`` before
    returning (nothing else will persist it this pass). If it DOES
    result in an exit, the caller is expected to perform the actual
    order cancel/sell and then write the final audit via its existing
    ``write_result`` helper (which reads ``trade.data``, already
    updated here) -- this function does not write in that case, to
    avoid a duplicate write racing the caller's own.

    Returns:
        None if the trade should stay open this pass, else an
        :class:`ExitDecision` describing what triggered the exit.
    """
    now = now or datetime.now(ET_TZ)
    advisor_cfg = exec_config.exit_advisor
    exit_cfg = exec_config.exit_strategy

    state = AdvisorState.from_dict(trade.data.get("advisor_state"))
    pnl_pct = compute_pnl_pct(trade.entry_price, mark)
    update_peak(state, pnl_pct)

    # The advisor's OWN tightened trail (if any) is enforced on every
    # poll, deterministically, with no model call needed -- this is the
    # advisor's deliberate prior choice being carried out, not a new
    # decision. Checked before anything else below (including the call
    # budget, which is irrelevant here since this path never calls the
    # model).
    if advisor_trail_triggered(pnl_pct, state, exit_cfg):
        _record_consultation(
            trade, trigger="advisor_trail", result=None, action_taken="trailing_fallback", now=now
        )
        trade.data["advisor_state"] = state.to_dict()
        return ExitDecision(should_exit=True, exit_reason="trailing_stop", limit_mult=0.85)

    mins_left = minutes_to_deadline(now, exit_cfg.time_deadline_est)

    if state.calls_made >= advisor_cfg.max_calls_per_trade:
        if trailing_fallback_triggered(pnl_pct, state, exit_cfg):
            _record_consultation(
                trade,
                trigger="budget_exhausted",
                result=None,
                action_taken="trailing_fallback",
                now=now,
            )
            trade.data["advisor_state"] = state.to_dict()
            return ExitDecision(should_exit=True, exit_reason="trailing_stop", limit_mult=0.85)
        trade.data["advisor_state"] = state.to_dict()
        _write_trade_data(trade.path, trade.data)
        return None

    trigger = decide_trigger(pnl_pct, state, now, mins_left, advisor_cfg)
    if trigger is None:
        trade.data["advisor_state"] = state.to_dict()
        _write_trade_data(trade.path, trade.data)
        return None

    model = resolve_model(advisor_cfg, llm_primary_model)
    gather_fn = market_context_fn or gather_market_context
    try:
        market_context = await gather_fn(trade.asset or "", now)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("exit_advisor_market_context_failed", error=str(e))
        market_context = {"underlying_path": {"available": False}, "co_movement": {}}

    rec = trade.data.get("recommendation") or {}
    prediction = {
        "direction": trade.direction,
        "predicted_move_pct": rec.get("predicted_move_pct"),
        "rationale": (rec.get("rationale") or {}).get("llm_rationale", ""),
    }
    context = build_context(
        trigger=trigger,
        asset=trade.asset or "",
        direction=trade.direction or "",
        prediction=prediction,
        entry_time=trade.data.get("started_at", ""),
        entry_price=trade.entry_price or 0.0,
        current_mark=mark,
        current_bid=(quote or {}).get("bid"),
        current_ask=(quote or {}).get("ask"),
        underlying_spot=underlying_spot,
        strike=trade.data.get("entry_strike") or 0.0,
        pnl_pct=pnl_pct,
        peak_pnl_pct=state.peak_pnl_pct,
        market_context=market_context,
        minutes_to_deadline=mins_left,
        sl_level=trade.sl_level or 0.0,
        time_deadline_est=exit_cfg.time_deadline_est,
        calls_made=state.calls_made,
        max_calls_per_trade=advisor_cfg.max_calls_per_trade,
    )
    prompt = build_prompt(context)

    state.calls_made += 1
    result = await consult(prompt, model, advisor_cfg.timeout_sec, invoke_fn=invoke_fn)

    if not result.ok:
        logger.warning(
            "exit_advisor_consult_failed",
            trade_id=trade.trade_id,
            trigger=trigger,
            model=model,
            error=result.error,
        )
        if trailing_fallback_triggered(pnl_pct, state, exit_cfg):
            _record_consultation(
                trade, trigger=trigger, result=result, action_taken="trailing_fallback", now=now
            )
            trade.data["advisor_state"] = state.to_dict()
            return ExitDecision(should_exit=True, exit_reason="trailing_stop", limit_mult=0.85)
        _record_consultation(
            trade, trigger=trigger, result=result, action_taken="hold_on_failure", now=now
        )
        trade.data["advisor_state"] = state.to_dict()
        _write_trade_data(trade.path, trade.data)
        return None

    state.apply_trail_tightening(result.trail_stop_pct, exit_cfg.trailing.trail_pct)

    if result.action == "EXIT":
        _record_consultation(trade, trigger=trigger, result=result, action_taken="exit", now=now)
        trade.data["advisor_state"] = state.to_dict()
        logger.info(
            "exit_advisor_exit",
            trade_id=trade.trade_id,
            trigger=trigger,
            confidence=result.confidence,
            reason=result.reason,
        )
        return ExitDecision(
            should_exit=True, exit_reason="advisor_exit", limit_mult=0.9, consult=result
        )

    _record_consultation(trade, trigger=trigger, result=result, action_taken="hold", now=now)
    trade.data["advisor_state"] = state.to_dict()
    _write_trade_data(trade.path, trade.data)
    return None
