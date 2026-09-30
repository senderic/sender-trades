"""Trade outcome tracking — scans audit files for post-mortem learning.

Feeds actual trade PnL back into the LLM prompt and strategy confidence
dampening so the system can learn from its mistakes.

Key data extracted from each ``trade-*.json`` audit file:
- Date, asset, option direction (CALL/PUT)
- Strategy label, entry price, entry strike
- Exit price, exit reason, PnL ($ and %)

Only resolved trades (``exit_reason`` not in ``pending``, ``None``,
``expired`` where unfilled) contribute to stats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from src.json_utils import load_json_tolerant

logger = structlog.get_logger()

# Dispositions that produced no actual fill → no PnL to learn from.
# "pending"/empty means never resolved; "expired"/"rejected"/"unfilled"
# mean the entry never filled (no capital at risk, not a loss).
# NOTE: "expired_worthless" is NOT here — that is a real loss (bought,
# expired OTM) and must be learned from.
UNRESOLVED_DISPOSITIONS = frozenset({None, "", "pending", "expired", "rejected", "unfilled"})


class TradeOutcome:
    """A single completed trade with entry + exit data."""

    __slots__ = (
        "asset",
        "correlation_id",
        "date",
        "direction",
        "entry_price",
        "entry_strike",
        "exit_price",
        "exit_reason",
        "pnl",
        "pnl_pct",
        "strategy",
        "trade_id",
    )

    def __init__(self, raw: dict[str, Any], date_str: str) -> None:
        self.date = date_str
        self.trade_id = raw.get("trade_id", "")
        self.correlation_id = raw.get("correlation_id", "")
        self.asset = raw.get("asset", "?")
        self.direction = raw.get("direction", "?")
        self.strategy = _extract_strategy(raw)
        self.entry_strike = raw.get("entry_strike", 0)

        self.exit_reason = raw.get("exit_reason", "")
        self.exit_price = float(raw.get("exit_price", 0) or 0)
        self.pnl = float(raw.get("final_pnl", 0) or 0)
        self.pnl_pct = float(raw.get("final_pnl_pct", 0) or 0)

        self.entry_price = _extract_entry_price(raw)
        if not self.entry_price and self.exit_price and self.pnl:
            # Reconstruct entry price from exit + PnL (1 contract = 100 shares)
            self.entry_price = round(self.exit_price - self.pnl / 100, 4)

    @property
    def won(self) -> bool:
        return self.pnl > 0

    @property
    def lost(self) -> bool:
        return self.pnl < 0

    @property
    def is_resolved(self) -> bool:
        return self.exit_reason not in UNRESOLVED_DISPOSITIONS

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "trade_id": self.trade_id,
            "asset": self.asset,
            "direction": self.direction,
            "strategy": self.strategy,
            "entry_price": self.entry_price,
            "entry_strike": self.entry_strike,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
        }


def load_trade_outcomes(log_dir: str | Path) -> list[TradeOutcome]:
    """Scan all ``trade-*.json`` files under ``logs/<date>/`` directories.

    Only trades with an actual fill and resolved exit are included.
    Skipped: unresolved ("pending"/empty) and never-filled entries
    ("expired"/"rejected" — no capital at risk, no PnL to learn from).
    """
    root = Path(log_dir).expanduser().resolve()
    outcomes: list[TradeOutcome] = []

    for day_dir in sorted(root.iterdir()):
        if not day_dir.is_dir() or not day_dir.name.startswith("20"):
            continue
        date_str = day_dir.name[:10]
        for trade_file in sorted(day_dir.glob("trade-*.json")):
            if trade_file.name.endswith(".bak"):
                continue
            try:
                # Tolerates both a normal single-JSON-object file and a
                # legacy file left as several concatenated JSON objects
                # by a trade that never reached finalize() -- see
                # src.json_utils and src.execution.context.TradeContext.
                data = load_json_tolerant(trade_file.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.debug("trade_tracker_read_error", path=str(trade_file), error=str(e))
                continue
            if not data:
                continue

            exit_reason = data.get("exit_reason")
            if exit_reason in UNRESOLVED_DISPOSITIONS:
                logger.debug(
                    "trade_tracker_skip_unresolved",
                    trade_id=data.get("trade_id"),
                    exit_reason=exit_reason,
                )
                continue

            outcome = TradeOutcome(data, date_str)
            # Skip trades with no actual fill and no PnL — they represent
            # failed/no-op executions, not real outcomes to learn from.
            if outcome.entry_price == 0.0 and outcome.exit_price == 0.0 and outcome.pnl == 0.0:
                logger.debug(
                    "trade_tracker_skip_no_fill",
                    trade_id=data.get("trade_id"),
                )
                continue

            outcomes.append(outcome)

    logger.info("trade_tracker_loaded", total=len(outcomes))
    return outcomes


def _compute_streak(trades: list[TradeOutcome]) -> int:
    """Compute the trailing win/loss streak from chronological trades.

    Returns a signed integer: positive for consecutive wins ending at the
    most recent trade, negative for consecutive losses, 0 for no streak.
    A break-even trade (PnL == 0) breaks the streak.
    """
    streak = 0
    for t in reversed(trades):
        if t.won:
            if streak < 0:
                break
            streak += 1
        elif t.lost:
            if streak > 0:
                break
            streak -= 1
        else:
            break
    return streak


def compute_strategy_stats(
    outcomes: list[TradeOutcome],
) -> dict[str, dict[str, Any]]:
    """Compute per-strategy win/loss stats from resolved trade outcomes.

    Returns a dict keyed by strategy label with:
    - wins, losses, total, win_rate
    - total_pnl, avg_pnl
    - current_streak (positive=win streak, negative=loss streak)
    """
    if not outcomes:
        return {}

    strategies: dict[str, list[TradeOutcome]] = {}
    for o in sorted(outcomes, key=lambda x: x.date):
        strategies.setdefault(o.strategy, []).append(o)

    stats: dict[str, dict[str, Any]] = {}
    for label, trades in strategies.items():
        wins = sum(1 for t in trades if t.won)
        losses = sum(1 for t in trades if t.lost)
        total = wins + losses
        pnl_total = sum(t.pnl for t in trades)

        # Current streak: consecutive same-sign outcomes ending at most recent
        streak = _compute_streak(trades)

        stats[label] = {
            "wins": wins,
            "losses": losses,
            "total": total,
            "win_rate": round(wins / total, 3) if total > 0 else 0.0,
            "total_pnl": round(pnl_total, 2),
            "avg_pnl": round(pnl_total / total, 2) if total > 0 else 0.0,
            "current_streak": streak,
            "last_date": trades[-1].date if trades else "",
        }

    return stats


def compute_direction_stats(
    outcomes: list[TradeOutcome],
) -> dict[str, dict[str, Any]]:
    """Compute per-asset + per-direction win/loss stats.

    Returns dict keyed like ``"SPY:CALL"`` with wins/losses/win_rate/streak.
    """
    stats: dict[str, dict[str, list[TradeOutcome]]] = {}
    for o in sorted(outcomes, key=lambda x: x.date):
        key = f"{o.asset}:{o.direction}"
        stats.setdefault(key, []).append(o)

    result: dict[str, dict[str, Any]] = {}
    for key, trades in stats.items():
        wins = sum(1 for t in trades if t.won)
        losses = sum(1 for t in trades if t.lost)
        total = wins + losses
        streak = _compute_streak(trades)
        result[key] = {
            "wins": wins,
            "losses": losses,
            "total": total,
            "win_rate": round(wins / total, 3) if total > 0 else 0.0,
            "total_pnl": round(sum(t.pnl for t in trades), 2),
            "current_streak": streak,
        }

    return result


def format_outcomes_for_prompt(
    outcomes: list[TradeOutcome],
    max_items: int = 12,
) -> str:
    """Format trade outcomes as a concise prompt section for the LLM.

    Includes recent per-trade PnL, per-strategy stats, and per-direction
    stats so the LLM can learn from actual trade results.
    """
    if not outcomes:
        return ""

    resolved = [o for o in outcomes if o.is_resolved]
    if not resolved:
        return ""

    recent = sorted(resolved, key=lambda o: o.date, reverse=True)[:max_items]
    strategy_stats = compute_strategy_stats(resolved)
    direction_stats = compute_direction_stats(resolved)

    parts: list[str] = ["Actual trade results (not predictions — real money outcomes):"]

    # Per-trade summary
    lines: list[str] = []
    for o in recent:
        sign = "+" if o.pnl > 0 else "-" if o.pnl < 0 else ""
        lines.append(
            f"  - {o.date}: {o.asset} {o.direction} ({o.strategy}) "
            f"entry ${o.entry_price:.2f} → exit ${o.exit_price:.2f} "
            f"({o.exit_reason}) — PnL: {sign}${abs(o.pnl):.2f} "
            f"({sign}{abs(o.pnl_pct):.1f}%)"
        )
    parts.append("Recent trades:\n" + "\n".join(lines))

    # Per-strategy summary
    if strategy_stats:
        strat_lines: list[str] = []
        for label, s in sorted(strategy_stats.items()):
            streak_str = (
                f"streak: {'W' if s['current_streak'] > 0 else 'L'}{abs(s['current_streak'])}"
                if s["current_streak"] != 0
                else "no streak"
            )
            strat_lines.append(
                f"  - {label}: {s['wins']}W/{s['losses']}L "
                f"({s['win_rate']:.0%} win) total PnL ${s['total_pnl']:+.2f}, "
                f"{streak_str}"
            )
        parts.append("Per-strategy record:\n" + "\n".join(strat_lines))

    # Per-direction summary
    if direction_stats:
        dir_lines: list[str] = []
        for key, s in sorted(direction_stats.items()):
            if s["total"] == 0:
                continue
            streak_str = (
                f"{'W' if s['current_streak'] > 0 else 'L'}{abs(s['current_streak'])} streak"
                if s["current_streak"] != 0
                else ""
            )
            note = f" (⚠️ {streak_str})" if streak_str else ""
            dir_lines.append(
                f"  - {key}: {s['wins']}W/{s['losses']}L "
                f"({s['win_rate']:.0%}) PnL ${s['total_pnl']:+.2f}{note}"
            )
        parts.append("Per-asset/direction record:\n" + "\n".join(dir_lines))

    # Learning hint for the LLM
    parts.append(
        "Learning directive: use this trade history to AVOID repeating losing "
        "patterns. If a strategy on a particular asset/direction has lost "
        "consecutively, do NOT recommend that combination again unless there "
        "is overwhelming new evidence (gap, catalyst, sentiment) that changes "
        "the thesis. Prefer strategies with positive win rates and recent wins."
    )

    return "\n".join(parts)


def _extract_entry_price(data: dict[str, Any]) -> float:
    """Extract entry fill price from trade audit entries."""
    for entry in data.get("entries", []):
        if entry.get("event_type") == "entry_filled":
            price = entry.get("avg_price") or entry.get("fill_price")
            if price is not None:
                return float(price)
    return 0.0


def _extract_strategy(data: dict[str, Any]) -> str:
    """Extract strategy label from trade recommendation."""
    rec = data.get("recommendation", {})
    return rec.get("strategy_label", "unknown")
