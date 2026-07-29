from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

import structlog

logger = structlog.get_logger()

ASSETS = ["SPY", "QQQ"]


def find_logs_dir(root: str = "logs") -> Path:
    return Path(root).expanduser().resolve()


def find_today_audits(log_dir: Path, *, target_date: date | None = None) -> list[dict]:
    day_dir = log_dir / (target_date or _today_pacific()).isoformat()
    if not day_dir.is_dir():
        return []
    audits = []
    for f in sorted(day_dir.glob("trade-*.json")):
        if f.name.endswith(".bak"):
            continue
        try:
            audits.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return audits


def find_today_summaries(log_dir: Path, *, target_date: date | None = None) -> list[dict]:
    day_dir = log_dir / (target_date or _today_pacific()).isoformat()
    if not day_dir.is_dir():
        return []
    summaries = []
    for f in sorted(day_dir.glob("summary-*.json")):
        try:
            summaries.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return summaries


def _today_pacific() -> date:
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("America/Los_Angeles")).date()


def _legacy_outcome_label(forecast: dict, daily_ohlc: dict | None) -> str:
    if daily_ohlc is None:
        return "unknown"
    direction = forecast.get("direction", "").upper()
    move_pct = abs(forecast.get("predicted_move_pct", 0) or 0)
    o = float(daily_ohlc["o"])
    h = float(daily_ohlc["h"])
    lo = float(daily_ohlc["l"])
    c_val = float(daily_ohlc["c"])
    pct = (c_val - o) / o * 100 if o else 0

    if move_pct >= 0.1:
        target = o * (1 + move_pct / 100)
        hit = h >= target if direction == "UP" else lo <= target
        return "success" if hit else "fail"
    if direction == "UP":
        return "success" if pct > 0 else "fail"
    return "success" if pct < 0 else "fail"


def fetch_daily_ohlc(symbol: str, *, target_date: date | None = None) -> dict | None:
    try:
        from datetime import datetime as dt

        import yfinance as yf

        ticker = yf.Ticker(symbol)
        h = ticker.history(period="5d")
        if h.empty:
            return None
        want = target_date or _today_pacific()
        want_ts = dt(want.year, want.month, want.day)
        mask = h.index.normalize() == want_ts
        row = h[mask].iloc[0] if mask.any() else h.iloc[-1]
        return {
            "o": float(row["Open"]),
            "h": float(row["High"]),
            "l": float(row["Low"]),
            "c": float(row["Close"]),
        }
    except Exception:
        return None


def today_letter() -> str:
    days = ["M", "T", "W", "Th", "F", "Sa", "Su"]
    return days[_today_pacific().weekday()]


def format_pnl(num: float | int) -> str:
    sign = "+" if num >= 0 else ""
    return f"{sign}${num:,.2f}"


def _format_move_pct(f: dict) -> str:
    val = f.get("predicted_move_pct") or f.get("expected_move_pct") or 0
    return f"{float(val):.1f}%"


def _forecast_for_asset(forecast_list: list[dict], asset: str) -> dict | None:
    for f in forecast_list:
        if f.get("asset") == asset:
            return f
    return None


def _actual_move(ohlc: dict | None) -> float:
    if ohlc is None:
        return 0.0
    o = ohlc["o"]
    c = ohlc["c"]
    return ((c - o) / o * 100) if o else 0.0


def _trade_filled(trade: dict) -> bool:
    return any(ev.get("state_to") == "filled" for ev in trade.get("events", []))


def _trade_fill_price(trade: dict) -> float | None:
    for ev in trade.get("events", []):
        if ev.get("state_to") == "filled":
            return ev.get("metadata", {}).get("avg_price")
    return None


def _trade_closed(trade: dict) -> bool:
    return trade.get("exit_reason", "") not in ("", "error")


def build_lessons_md(
    *,
    target_date: date | None = None,
    audits: list[dict],
    summaries: list[dict],
    market_data: dict[str, dict | None],
) -> str:
    dt = target_date or _today_pacific()
    header = f"## {dt.isoformat()} — Post-market analysis"

    parts = [header]

    # --- Prediction accuracy ---
    if summaries:
        summary = summaries[0]
        forecast_list = summary.get("decision", {}).get("forecast", {}).get("forecasts", [])
        recommendation = summary.get("decision", {}).get("recommendation") or {}
        selected_label = summary.get("decision", {}).get("selected_label", "")

        parts.append("")
        parts.append("### Prediction accuracy")

        for asset in ASSETS:
            f = _forecast_for_asset(forecast_list, asset)
            if not f:
                parts.append(f"- **{asset}**: no prediction")
                continue

            direction = f.get("direction", "?").upper()
            confidence = f.get("confidence", 0)
            move_pct = _format_move_pct(f)
            ohlc = market_data.get(asset)
            outcome = _legacy_outcome_label(f, ohlc)
            actual = _actual_move(ohlc)

            outcome_label = ":white_check_mark: HIT" if outcome == "success" else ":x: MISS"
            parts.append(
                f"- {outcome_label} **{asset} {direction}** | {confidence:.0%} conf | predicted {move_pct} | actual {actual:+.2f}%"
            )
            rationale = f.get("rationale", "")
            if rationale:
                parts.append(f"  _{rationale[:200]}_")

        if recommendation and recommendation.get("asset"):
            parts.append(
                f"\n**Best trade**: {recommendation['asset']} {recommendation['direction']}, strategy={selected_label}"
            )

    # --- Trade execution results ---
    parts.append("")
    parts.append("### Trade execution")

    if audits:
        filled = [t for t in audits if _trade_filled(t)]
        orphaned = [t for t in filled if not _trade_closed(t)]

        parts.append(f"**{len(audits)}** orders submitted, **{len(filled)}** filled")

        if orphaned:
            parts.append(
                f"**{len(orphaned)}** positions were not closed by the engine "
                f"(engine exit error). These were closed manually or by safety-close."
            )

        for tr in filled:
            asset = tr.get("asset", "?")
            direction = tr.get("direction", "?")
            fill_price = _trade_fill_price(tr)
            pnl = float(tr.get("final_pnl", 0) or 0)
            reason = tr.get("exit_reason", "?")
            line = f"  {asset} {direction} @ ${fill_price:.2f}/contract"
            if pnl:
                line += f" → {format_pnl(pnl)} ({reason})"
            else:
                line += f" → not closed by engine ({reason})"
            parts.append(line)

        engine_pnl = sum(float(t.get("final_pnl", 0) or 0) for t in filled)
        if engine_pnl:
            parts.append(f"\nEngine-tracked PnL: {format_pnl(engine_pnl)}")
    else:
        parts.append("No trades executed today.")

    # --- System issues ---
    issues: list[str] = []
    error_trades = [t for t in audits if t.get("exit_reason") == "error" and _trade_filled(t)]
    if error_trades:
        issues.append(
            f"Engine failed to close {len(error_trades)} filled positions (exit monitoring died with pipeline)"
        )

    if issues:
        parts.append("")
        parts.append("### System issues")
        for issue in issues:
            parts.append(f"- {issue}")

    # --- Market context ---
    parts.append("")
    parts.append("### Daily market")
    for asset in ASSETS:
        ohlc = market_data.get(asset)
        if ohlc:
            pct = _actual_move(ohlc)
            parts.append(
                f"  {asset}: O=${ohlc['o']:.2f} H=${ohlc['h']:.2f} L=${ohlc['l']:.2f} C=${ohlc['c']:.2f} ({pct:+.2f}%)"
            )
        else:
            parts.append(f"  {asset}: no data")

    # --- Cumulative record ---
    history = _load_prediction_history()
    if history:
        deduped = _dedupe_history(history)
        wins = sum(1 for h in deduped if h.get("result") == "success")
        valid = len(deduped)
        rate = wins / valid * 100 if valid > 0 else 0
        parts.append("")
        parts.append(f"**Cumulative prediction record**: {wins}/{valid} ({rate:.0f}%)")

    return "\n".join(parts) + "\n"


def _load_prediction_history(log_dir: str | Path = "logs") -> list[dict]:
    path = Path(log_dir).expanduser().resolve() / "prediction-history.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _dedupe_history(history: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    result: list[dict] = []
    for h in history:
        key = (h.get("date"), h.get("asset"), h.get("predicted_direction"))
        if key not in seen:
            seen.add(key)
            result.append(h)
    return result


def update_lessons_log(
    entry: str,
    lessons_path: str = "LESSONS_LEARNED.md",
) -> None:
    path = Path(lessons_path).expanduser().resolve()
    existing = path.read_text() if path.exists() else ""

    marker = "# Lessons Learned\n"
    insert_at = existing.find(marker)
    if insert_at == -1:
        path.write_text(marker + "\n" + entry + "\n---\n\n")
        return

    after_header = existing.find("---", insert_at + len(marker))
    if after_header == -1:
        new_content = (
            existing[: insert_at + len(marker)]
            + "\n"
            + entry
            + "\n---\n"
            + existing[insert_at + len(marker) :]
        )
    else:
        new_content = existing[: after_header + 4] + "\n" + entry + existing[after_header + 4 :]
    path.write_text(new_content)


def run(
    log_dir: str | Path = "logs",
    target_date: date | None = None,
    lessons_path: str = "LESSONS_LEARNED.md",
) -> int:
    dt = target_date or _today_pacific()
    ld = Path(log_dir).expanduser().resolve()

    if dt.weekday() >= 5:
        logger.info("lesson_skip_weekend", date=dt.isoformat())
        return 0

    day_dir = ld / dt.isoformat()
    if not day_dir.is_dir():
        logger.info("lesson_no_logs", date=dt.isoformat())
        return 0

    audits = find_today_audits(ld, target_date=dt)
    summaries = find_today_summaries(ld, target_date=dt)

    market_data: dict[str, dict | None] = {}
    for sym in ASSETS:
        market_data[sym] = fetch_daily_ohlc(sym, target_date=dt)

    entry = build_lessons_md(
        target_date=dt,
        audits=audits,
        summaries=summaries,
        market_data=market_data,
    )

    update_lessons_log(entry, lessons_path=lessons_path)
    logger.info("lesson_written", date=dt.isoformat(), trades=len(audits))
    return 0


def main() -> None:
    code = run()
    sys.exit(code)


if __name__ == "__main__":
    main()
