"""Forward scorecard: grade every daily forecast against baselines on real option prices.

Paper trades alone are a slow vote -- the premium gate blocks most days, so
they accumulate ~0-1 trades a day, and per-trade outcomes are noisy enough
that separating a real edge from luck takes ~250 of them. This module
counts every vote instead: each (date, asset) forecast the pipeline wrote
to ``logs/prediction-history.json`` -- traded, gate-blocked, or passed -- is
replayed as the trade production WOULD have made (0.6% OTM strike, 09:30
entry, -50% stop + 30/15 trailing stop, the advisor's deterministic
fallback) on that day's REAL option bars, next to the same trade for each
naive baseline direction (see :data:`COMPARE`). Scored with
:mod:`src.evaluation.option_replay`, the same engine as the two-year
``scripts/scoreboard.py`` backtest.

Run after the close (``lessons_log.sh``, 14:00 PT)::

    uv run python -m src.evaluation.forward_scorecard

Results persist in ``logs/forward-scorecard.json`` (one entry per scored
asset-day, never re-scored). The morning email reads that file via
:func:`render_email_html` / :func:`render_email_text`, which never raise and
never make network calls. When an execution audit shows that the premium gate
blocked a candidate, the row also contains ``blocked_trade``: a replay of the
exact recommended strike and contract count, separate from the standardized
forecast trade used for fair baseline comparisons.
"""

from __future__ import annotations

import html
import json
import random
import statistics
import sys
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import structlog

from src.evaluation.option_replay import (
    BASELINE_SIGNALS,
    ET,
    EXIT_POLICIES,
    LOG_DIR,
    PROD_MONEYNESS,
    Alpaca,
    DaySetup,
    build_day,
    daily_bars,
    entry_price,
    occ,
    simulate_exit,
    strike_for,
)

logger = structlog.get_logger()

SCORECARD_PATH = LOG_DIR / "forward-scorecard.json"
HISTORY_PATH = LOG_DIR / "prediction-history.json"
EXIT_POLICY = next(p for p in EXIT_POLICIES if p.name == "sl50_trail30/15")
ENTRY_MODEL = "vwap"
EXIT_SLIP = 0.03
# Baselines every forecast is compared against, on the same asset-days.
COMPARE = ("gap_follow", "always_call", "repeat_yesterday", "gap_fade", "always_put")
# Rough sample size needed to resolve a ~7%-of-premium edge (2026-09-25
# backtest: per-trade SD ~56% of premium -> (1.96 * 56 / 7)^2 ~ 250).
TARGET_N = 250
# Bars for a session are only complete (and safe to cache) well after the
# close; SIP data also lags 15 minutes.
SESSION_SETTLED = time(16, 30)
_DIRECTION = {"UP": "CALL", "DOWN": "PUT"}


# ──────────────────────────────── scoring ────────────────────────────────


def system_forecasts(history: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    """Map ``(date, asset)`` -> CALL/PUT from the prediction history.

    NEUTRAL / missing directions are the pipeline declining to forecast and
    are not scored. A later entry for the same asset-day wins (reruns).
    """
    out: dict[tuple[str, str], str] = {}
    for entry in history:
        direction = _DIRECTION.get(str(entry.get("predicted_direction", "")).upper())
        if direction and entry.get("date") and entry.get("asset"):
            out[(str(entry["date"]), str(entry["asset"]))] = direction
    return out


def blocked_trades(log_dir: Path = LOG_DIR) -> dict[tuple[str, str], dict[str, Any]]:
    """Load exact recommendations rejected by the premium gate from trade audits."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(log_dir.glob("????-??-??/trade-*.json")):
        try:
            audit = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(audit, dict) or audit.get("exit_reason") != "premium_gate_blocked":
            continue
        asset = str(audit.get("asset") or "")
        direction = str(audit.get("direction") or "").upper()
        try:
            strike = float(audit.get("entry_strike") or 0)
            contracts = int(audit.get("contracts") or 1)
        except (TypeError, ValueError):
            continue
        if not asset or direction not in {"CALL", "PUT"} or strike <= 0:
            continue
        out[(path.parent.name, asset)] = {
            "trade_id": audit.get("trade_id"),
            "direction": direction,
            "strike": strike,
            "contracts": contracts,
        }
    return out


def _simulate(
    setup: DaySetup, direction: str | None, strike: float | None = None
) -> dict[str, Any] | None:
    """Replay one production-style trade in ``direction`` on ``setup``'s real bars."""
    if direction is None:
        return None
    strike = (
        round(strike) if strike is not None else strike_for(setup.s0, direction, PROD_MONEYNESS)
    )
    bars = setup.options.get(occ(setup.asset, setup.day, direction[0], strike)) or []
    entry = entry_price(bars, ENTRY_MODEL)
    if not entry or entry < 0.02:
        return None
    exit_px, reason = simulate_exit(bars, entry, EXIT_POLICY, EXIT_SLIP)
    return {
        "direction": direction,
        "strike": strike,
        "entry": round(entry, 4),
        "exit": round(exit_px, 4),
        "reason": reason,
        "ret": round((exit_px / entry - 1) * 100, 2),
        "pnl": round((exit_px - entry) * 100, 2),
    }


def score_asset_day(
    setup: DaySetup,
    system_direction: str,
    blocked_trade: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score the system's forecast and every baseline on one asset-day."""
    row = {
        "date": setup.day.isoformat(),
        "asset": setup.asset,
        "system": _simulate(setup, system_direction),
        "baselines": {name: _simulate(setup, BASELINE_SIGNALS[name](setup)) for name in COMPARE},
        "gap_pct": round(setup.gap_pct, 3),
    }
    if blocked_trade:
        blocked_result = _simulate(
            setup,
            blocked_trade.get("direction"),
            blocked_trade.get("strike"),
        )
        if blocked_result:
            contracts = int(blocked_trade.get("contracts") or 1)
            blocked_result.update(
                trade_id=blocked_trade.get("trade_id"),
                contracts=contracts,
                total_pnl=round(blocked_result["pnl"] * contracts, 2),
            )
            row["blocked_trade"] = blocked_result
    return row


def _settled(day: date, now: datetime) -> bool:
    return now >= datetime.combine(day, SESSION_SETTLED, ET)


def load_scorecard(path: Path = SCORECARD_PATH) -> dict[str, dict[str, Any]]:
    try:
        rows = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(rows, list):
        return {}
    return {f"{r['date']}|{r['asset']}": r for r in rows if isinstance(r, dict) and "date" in r}


def update(
    api: Alpaca | None = None,
    history_path: Path = HISTORY_PATH,
    scorecard_path: Path = SCORECARD_PATH,
    log_dir: Path = LOG_DIR,
    now: datetime | None = None,
) -> dict[str, dict[str, Any]]:
    """Score every settled, not-yet-scored forecast and persist the scorecard."""
    now = now or datetime.now(ET)
    scored = load_scorecard(scorecard_path)
    try:
        history = json.loads(history_path.read_text())
    except (OSError, ValueError):
        logger.warning("forward_scorecard_no_history", path=str(history_path))
        return scored
    forecasts = system_forecasts(history if isinstance(history, list) else [])
    blocked = blocked_trades(log_dir)
    pending = {
        key: d
        for key, d in forecasts.items()
        if f"{key[0]}|{key[1]}" not in scored and _settled(date.fromisoformat(key[0]), now)
    }
    if not pending:
        logger.info("forward_scorecard_up_to_date", scored=len(scored))
        return scored

    api = api or Alpaca()
    days = sorted(date.fromisoformat(k[0]) for k in pending)
    daily = {asset: daily_bars(api, asset, days[0], days[-1]) for asset in {k[1] for k in pending}}
    for (day_s, asset), direction in sorted(pending.items()):
        try:
            setup = build_day(api, asset, date.fromisoformat(day_s), daily[asset])
        except Exception as e:  # one bad day must not block the rest
            logger.warning("forward_scorecard_day_failed", date=day_s, asset=asset, error=str(e))
            continue
        if setup is None:
            logger.info("forward_scorecard_no_data", date=day_s, asset=asset)
            continue
        scored[f"{day_s}|{asset}"] = score_asset_day(
            setup,
            direction,
            blocked_trade=blocked.get((day_s, asset)),
        )

    rows = sorted(scored.values(), key=lambda r: (r["date"], r["asset"]))
    scorecard_path.write_text(json.dumps(rows, indent=1))
    logger.info("forward_scorecard_updated", scored=len(rows), added=len(pending))
    return scored


# ─────────────────────────────── summary ────────────────────────────────


def _bootstrap_ci(values: list[float], seed: int = 7, n_boot: int = 2000) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n_boot))
    return means[int(n_boot * 0.025)], means[int(n_boot * 0.975) - 1]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """System vs each baseline, paired on the same asset-days (units: % of premium)."""
    sys_rows = [r for r in rows if r.get("system")]
    n = len(sys_rows)
    out: dict[str, Any] = {"n": n, "target_n": TARGET_N, "baselines": {}}
    if not n:
        return out
    sys_rets = [r["system"]["ret"] for r in sys_rows]
    out["system"] = {
        "mean_ret": statistics.fmean(sys_rets),
        "win": sum(x > 0 for x in sys_rets) / n,
        "total_pnl": sum(r["system"]["pnl"] for r in sys_rows),
    }
    for name in COMPARE:
        pairs = [
            (r["system"]["ret"], r["baselines"][name]["ret"])
            for r in sys_rows
            if (r.get("baselines") or {}).get(name)
        ]
        if len(pairs) < 2:
            continue
        diffs = [s - b for s, b in pairs]
        lo, hi = _bootstrap_ci(diffs)
        verdict = "ahead" if lo > 0 else "behind" if hi < 0 else "not significant"
        out["baselines"][name] = {
            "n": len(pairs),
            "mean_ret": statistics.fmean(b for _, b in pairs),
            "edge": statistics.fmean(diffs),
            "ci": (lo, hi),
            "verdict": verdict,
        }
    return out


def headline(summary: dict[str, Any]) -> str:
    """One-sentence verdict for the email."""
    n = summary.get("n", 0)
    if not n:
        return "No forecasts scored yet."
    verdicts = [b["verdict"] for b in summary["baselines"].values()]
    if verdicts and all(v == "ahead" for v in verdicts):
        return f"System beats every baseline with statistical significance (n={n})."
    if any(v == "behind" for v in verdicts):
        return f"System is significantly BEHIND at least one naive baseline (n={n})."
    return (
        f"Not yet significant (n={n} of ~{TARGET_N} needed to resolve a ~7%/trade edge) "
        "-- treat paper P&L as noise until this flips."
    )


# ─────────────────────────────── rendering ───────────────────────────────


def _load_summary(path: Path) -> dict[str, Any] | None:
    rows = list(load_scorecard(path).values())
    return summarize(rows) if rows else None


def render_email_html(path: Path = SCORECARD_PATH) -> str:
    """Scorecard section for the morning email. Never raises; "" when unavailable."""
    try:
        s = _load_summary(path)
        if not s or not s.get("n"):
            return ""
        sysm = s["system"]
        rows = (
            f"<tr><td><strong>System forecasts</strong></td><td>{sysm['mean_ret']:+.1f}%</td>"
            f"<td>{sysm['win']:.0%} win</td><td>—</td><td>—</td></tr>"
        )
        for name, b in s["baselines"].items():
            lo, hi = b["ci"]
            color = {"ahead": "#1a7f37", "behind": "#cf222e"}.get(b["verdict"], "#57606a")
            rows += (
                f"<tr><td>{name}</td><td>{b['mean_ret']:+.1f}%</td><td>n={b['n']}</td>"
                f"<td>{b['edge']:+.1f} pts [{lo:+.1f}, {hi:+.1f}]</td>"
                f'<td style="color:{color};font-weight:600">{b["verdict"]}</td></tr>'
            )
        return f"""<h2>Forward Scorecard — every forecast vs. naive baselines</h2>
<p style="font-size:13px"><strong>{html.escape(headline(s))}</strong></p>
<table>
<thead><tr><th>Direction source</th><th>Avg return (% of premium)</th><th>Sample</th>
<th>System edge vs. this (95% CI)</th><th>Verdict</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<p style="font-size:11px;color:#8b949e">Every forecast in prediction-history (traded or not) replayed on that
day's real 0DTE option bars: 0.6% OTM strike, 09:30 entry, &minus;50% stop + 30/15 trailing stop.
Baselines take the same trade on the same days.</p>"""
    except Exception as e:
        logger.warning("forward_scorecard_render_failed", error=str(e))
        return ""


def render_email_text(path: Path = SCORECARD_PATH) -> str:
    """Plain-text scorecard for the email. Never raises; "" when unavailable."""
    try:
        s = _load_summary(path)
        if not s or not s.get("n"):
            return ""
        lines = [
            "\n\nForward Scorecard (every forecast vs. naive baselines, % of premium):",
            f"  {headline(s)}",
            f"  System: {s['system']['mean_ret']:+.1f}%/trade, {s['system']['win']:.0%} win",
        ]
        for name, b in s["baselines"].items():
            lo, hi = b["ci"]
            lines.append(
                f"  vs {name}: {b['mean_ret']:+.1f}% -> edge {b['edge']:+.1f} pts "
                f"[{lo:+.1f}, {hi:+.1f}] {b['verdict']}"
            )
        return "\n".join(lines)
    except Exception as e:
        logger.warning("forward_scorecard_render_failed", error=str(e))
        return ""


def main() -> int:
    scored = update()
    s = summarize(list(scored.values()))
    sys.stdout.write(render_email_text().lstrip() + "\n" if s.get("n") else "nothing scored\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
