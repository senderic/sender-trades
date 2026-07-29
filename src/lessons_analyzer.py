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
        import pandas as pd
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        h = ticker.history(period="5d")
        if h.empty:
            return None
        want = target_date or _today_pacific()
        want_ts = pd.Timestamp(want.year, want.month, want.day, tz=h.index.tz)
        norm = h.index.normalize()
        mask = norm == want_ts
        if mask.any():
            row = h[mask].iloc[0]
        else:
            diffs = (norm - want_ts).to_series().abs()
            row = h.iloc[diffs.argmin()]
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


def _format_pct(val: float) -> str:
    return f"{val:+.2f}%"


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


def _extract_sources_from_summary(summary: dict) -> dict[str, list[str]]:
    """Extract per-asset sources from all strategy predictions."""
    results: dict[str, list[str]] = {}
    for r in summary.get("decision", {}).get("all_results", []):
        predictions = r.get("predictions") or {}
        if isinstance(predictions, dict):
            for asset, pred in predictions.items():
                if isinstance(pred, dict) and asset in ASSETS:
                    sources = pred.get("sources", [])
                    if asset not in results:
                        results[asset] = []
                    for s in sources:
                        if s not in results[asset]:
                            results[asset].append(s)
    return results


def _strategy_accuracy_summary(summary: dict, market_data: dict[str, dict | None]) -> str:
    """Build a per-strategy accuracy breakdown."""
    lines: list[str] = []
    all_r = summary.get("decision", {}).get("all_results", [])

    for r in all_r:
        label = r.get("label", "unknown")
        predictions = r.get("predictions") or {}
        if not isinstance(predictions, dict) or not predictions:
            lines.append(f"- **{label}**: no per-asset predictions available")
            continue

        asset_results: list[str] = []
        for asset in ASSETS:
            pred = predictions.get(asset)
            if not isinstance(pred, dict):
                continue
            direction = pred.get("direction", "?").upper()
            ohlc = market_data.get(asset)
            outcome = _legacy_outcome_label(pred, ohlc)
            mark = "HIT" if outcome == "success" else "MISS"
            asset_results.append(f"{asset} {direction}: {mark}")

        if asset_results:
            lines.append(f"- **{label}**: {' | '.join(asset_results)}")
        else:
            lines.append(f"- **{label}**: no SPY/QQQ prediction")

    return "\n".join(lines)


def _source_effectiveness(summary: dict, market_data: dict[str, dict | None]) -> str:
    """Tally which input sources appear in winning vs losing predictions."""
    win_sources: set[str] = set()
    loss_sources: set[str] = set()

    for r in summary.get("decision", {}).get("all_results", []):
        predictions = r.get("predictions") or {}
        if not isinstance(predictions, dict):
            continue
        for asset, pred in predictions.items():
            if not isinstance(pred, dict) or asset not in ASSETS:
                continue
            sources = pred.get("sources", [])
            outcome = _legacy_outcome_label(pred, market_data.get(asset))
            if outcome == "success":
                win_sources.update(sources)
            else:
                loss_sources.update(sources)

    if not win_sources and not loss_sources:
        return "no source data available"

    lines: list[str] = []

    win_only = win_sources - loss_sources
    loss_only = loss_sources - win_sources
    both = win_sources & loss_sources

    if both:
        lines.append(f"  Sources in both wins & losses: {', '.join(sorted(both))}")
    if win_only:
        lines.append(f"  Sources only in winning predictions: {', '.join(sorted(win_only))}")
    if loss_only:
        lines.append(f"  Sources only in losing predictions: {', '.join(sorted(loss_only))}")

    if not lines:
        return "no distinct source patterns"

    return "\n".join(lines)


def _catalyst_extract(summary: dict) -> str:
    """Extract key catalysts/themes mentioned in the LLM rationale."""
    r = summary.get("decision", {}).get("recommendation") or {}
    rationale = r.get("rationale", {})
    text = rationale.get("llm_rationale", "") if isinstance(rationale, dict) else ""
    if not text:
        return "no catalyst data"

    keywords = [
        "regulation",
        "FOMC",
        "earnings",
        "CPI",
        "jobs",
        "China",
        "AI",
        "tech",
        "recession",
        "selloff",
        "rally",
        "gap",
        "breach",
        "sanction",
        "defense",
        "spending",
        "tariff",
        "recalibration",
        "open source",
        "agent",
        "rotation",
    ]
    found = [kw for kw in keywords if kw.lower() in text.lower()]
    if found:
        return f"  Key catalysts cited: {', '.join(found)}"
    return "  No standard catalyst keywords detected"


def _vibe_context(summary: dict) -> str:
    """Capture the pre-market context visible to the LLM."""
    forecast_list = summary.get("decision", {}).get("forecast", {}).get("forecasts", [])
    forecast = summary.get("decision", {}).get("forecast", {})
    market_vibe = forecast.get("market_vibe", "")
    lines: list[str] = []

    if market_vibe:
        lines.append(f"  Market vibe: {market_vibe[:300]}")

    for f in forecast_list:
        asset = f.get("asset", "?")
        direction = f.get("direction", "?").upper()
        move_pct = _format_move_pct(f)
        rationale = f.get("rationale", "")
        lines.append(f"  **{asset}**: predicted {direction} {move_pct}")
        if rationale:
            lines.append(f"    _{rationale[:250]}_")

    return "\n".join(lines) if lines else "no context data"


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

    summary = summaries[0] if summaries else None

    # --- Vibe context: what the LLM saw ---
    if summary:
        parts.append("")
        parts.append("### Pre-market context (what the model saw)")
        parts.append(_vibe_context(summary))
        parts.append("")
        parts.append("### Key catalysts")
        parts.append(_catalyst_extract(summary))

    # --- Prediction accuracy ---
    if summary:
        forecast_list = summary.get("decision", {}).get("forecast", {}).get("forecasts", [])
        recommendation = summary.get("decision", {}).get("recommendation") or {}
        selected_label = summary.get("decision", {}).get("selected_label", "")

        parts.append("")
        parts.append("### Prediction outcomes")
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

            outcome_label = "HIT" if outcome == "success" else "MISS"
            emoji = ":white_check_mark:" if outcome == "success" else ":x:"
            parts.append(
                f"- {emoji} **{asset} {direction}** | {confidence:.0%} conf | "
                f"predicted {move_pct} | actual {_format_pct(actual)} ({outcome_label})"
            )

        if recommendation and recommendation.get("asset"):
            parts.append(
                f"\n**Best trade**: {recommendation['asset']} {recommendation['direction']}, "
                f"strategy={selected_label}"
            )

        # --- Strategy-level accuracy breakdown ---
        parts.append("")
        parts.append("### Strategy accuracy breakdown")
        parts.append(_strategy_accuracy_summary(summary, market_data))

        # --- Source effectiveness ---
        parts.append("")
        parts.append("### Source effectiveness (which signals helped)")
        parts.append(_source_effectiveness(summary, market_data))

    # --- Trade execution ---
    parts.append("")
    parts.append("### Trade execution")
    if audits:
        filled = [t for t in audits if _trade_filled(t)]
        orphaned = [t for t in filled if not _trade_closed(t)]
        parts.append(f"**{len(audits)}** orders submitted, **{len(filled)}** filled")

        if orphaned:
            parts.append(
                f"**{len(orphaned)}** positions were not closed by the engine. "
                f"Closed manually or by safety-close."
            )

        for tr in filled:
            asset = tr.get("asset", "?")
            direction = tr.get("direction", "?")
            fill_price = _trade_fill_price(tr)
            pnl = float(tr.get("final_pnl", 0) or 0)
            reason = tr.get("exit_reason", "?")
            line = f"  {asset} {direction} @ ${fill_price:.2f}/contract"
            if pnl:
                line += f" -> {format_pnl(pnl)} ({reason})"
            else:
                line += f" -> not closed by engine ({reason})"
            parts.append(line)

        engine_pnl = sum(float(t.get("final_pnl", 0) or 0) for t in filled)
        if engine_pnl:
            parts.append(f"\nEngine-tracked PnL: {format_pnl(engine_pnl)}")
    else:
        parts.append("No trades executed today.")

    # --- System issues ---
    if audits:
        error_trades = [t for t in audits if t.get("exit_reason") == "error" and _trade_filled(t)]
        if error_trades:
            parts.append("")
            parts.append("### System issues")
            parts.append(
                f"- Engine failed to close {len(error_trades)} filled positions "
                f"(exit monitoring died with pipeline)"
            )

    # --- Daily market summary ---
    parts.append("")
    parts.append("### Daily market")
    for asset in ASSETS:
        ohlc = market_data.get(asset)
        if ohlc:
            pct = _actual_move(ohlc)
            parts.append(
                f"  {asset}: O=${ohlc['o']:.2f} H=${ohlc['h']:.2f} L=${ohlc['l']:.2f} "
                f"C=${ohlc['c']:.2f} ({_format_pct(pct)})"
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
