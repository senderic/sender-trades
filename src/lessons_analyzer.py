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
    direction = (forecast.get("direction") or "").upper()
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


def _raw_move_pct(f: dict) -> float:
    return float(f.get("predicted_move_pct") or f.get("expected_move_pct") or 0)


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
        direction = (f.get("direction") or "?").upper()
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
    summary = summaries[0] if summaries else None
    parts: list[str] = []

    # --- YAML frontmatter ---
    parts.append(f"## {dt.isoformat()}")
    parts.append("")
    parts.append("```yaml")

    tag_list: list[str] = []

    if summary:
        forecast_list = summary.get("decision", {}).get("forecast", {}).get("forecasts", [])
        recommendation = summary.get("decision", {}).get("recommendation") or {}

        parts.append(f"date: {dt.isoformat()}")

        for asset in ASSETS:
            f = _forecast_for_asset(forecast_list, asset)
            if not f:
                continue
            direction = (f.get("direction") or "?").upper()
            confidence = f.get("confidence", 0)
            predicted = _raw_move_pct(f)
            ohlc = market_data.get(asset)
            outcome = _legacy_outcome_label(f, ohlc)
            actual = _actual_move(ohlc)
            hit_text = "HIT" if outcome == "success" else "MISS"
            note = ""
            if ohlc:
                h = ohlc["h"]
                lo = ohlc["l"]
                c_val = ohlc["c"]
                o_open = ohlc["o"]
                close_move = (c_val - o_open) / o_open * 100
                if outcome == "success":
                    close_dir = "UP" if close_move > 0 else "DOWN"
                    if close_dir != direction:
                        note = f"hit target, but reversed — closed {_format_pct(close_move)}"
                    elif direction == "UP":
                        note = f"hit target (H=${h:.2f})"
                    else:
                        note = f"hit target (L=${lo:.2f})"
                else:
                    note = f"never hit target, closed {_format_pct(close_move)}"

            parts.append(f"{asset.lower()}:")
            parts.append(f"  direction: {direction}")
            parts.append(f"  confidence: {confidence}")
            parts.append(f"  predicted_move_pct: {predicted}")
            parts.append(f"  actual_move_pct: {actual:.2f}")
            parts.append(f"  result: {hit_text}")
            if note:
                parts.append(f'  note: "{note}"')

            # Tag collection — only tag pattern when confident AND correct
            if hit_text == "HIT" and confidence >= 0.6:
                tag_list.append(f"pattern:{asset.lower()}-{direction.lower()}-reliable")
            if hit_text == "MISS" and confidence >= 0.5:
                tag_list.append(f"pattern:{asset.lower()}-{direction.lower()}-unreliable")
            if outcome == "success" and abs(actual) > abs(predicted) * 1.3:
                tag_list.append(f"pattern:{asset.lower()}-amplifies")

        if recommendation and recommendation.get("asset"):
            rec = recommendation
            parts.append(
                f'best_trade: "{rec["asset"]} {rec["direction"]}'
                f" @ ${rec.get('target_strike', '?')}, "
                f'strategy={rec.get("strategy_label", "?")}"'
            )

        # Source tags — only tag sources from confident winning predictions
        sources_result = _source_tag_dict(summary, market_data)
        for tag in sources_result.get("tags", []):
            if tag not in tag_list:
                tag_list.append(tag)

    # Trade execution data
    filled = [t for t in audits if _trade_filled(t)]
    trade_filled = len(filled) > 0
    parts.append(f"trade_filled: {str(trade_filled).lower()}")
    engine_pnl = sum(float(t.get("final_pnl", 0) or 0) for t in filled)
    parts.append(f"engine_pnl: {engine_pnl:.2f}")

    # System issue tags
    error_trades = [t for t in audits if t.get("exit_reason") == "error" and _trade_filled(t)]
    if error_trades:
        tag_list.append("system:exit-monitoring-dies")
    if audits and not filled:
        tag_list.append("system:no-trades-filled")

    # Build tag list
    if tag_list:
        parts.append("tags:")
        for t in sorted(set(tag_list)):
            parts.append(f"  - {t}")

    parts.append("```")

    # --- Pre-market context ---
    if summary:
        parts.append("")
        parts.append("### Pre-market context")
        parts.append("")
        forecast_list = summary.get("decision", {}).get("forecast", {}).get("forecasts", [])
        forecast = summary.get("decision", {}).get("forecast", {})
        market_vibe = forecast.get("market_vibe", "")
        if market_vibe:
            parts.append(f"Market vibe: {market_vibe[:400]}")
            parts.append("")

        catalysts = _catalyst_extract(summary)
        if catalysts != "no catalyst data":
            parts.append(f"Key catalysts: {catalysts.replace('  Key catalysts cited: ', '')}")
            parts.append("")

    # --- Prediction table ---
    if summary:
        parts.append("### What we predicted")
        parts.append("")
        parts.append("| Asset | Direction | Confidence | Predicted Move | Rationale (truncated) |")
        parts.append("|-------|-----------|------------|----------------|-----------------------|")
        forecast_list = summary.get("decision", {}).get("forecast", {}).get("forecasts", [])
        for f in forecast_list:
            asset = f.get("asset", "?")
            direction = (f.get("direction") or "?").upper()
            confidence = f"{f.get('confidence', 0):.0%}"
            move = _format_move_pct(f)
            rationale = f.get("rationale", "")[:120]
            parts.append(f"| {asset} | {direction} | {confidence} | {move} | {rationale} |")
        parts.append("")

    # --- Actuals table ---
    parts.append("### What actually happened")
    parts.append("")
    parts.append("| Asset | Open | High | Low | Close | Move % | Result |")
    parts.append("|-------|------|------|-----|-------|--------|--------|")
    if summary:
        forecast_list = summary.get("decision", {}).get("forecast", {}).get("forecasts", [])
        for f in forecast_list:
            asset = f.get("asset", "?")
            direction = (f.get("direction") or "?").upper()
            ohlc = market_data.get(asset)
            if ohlc:
                outcome = _legacy_outcome_label(f, ohlc)
                actual = _actual_move(ohlc)
                hit_text = ":white_check_mark: HIT" if outcome == "success" else ":x: MISS"
                parts.append(
                    f"| {asset} | ${ohlc['o']:.2f} | ${ohlc['h']:.2f} | "
                    f"${ohlc['l']:.2f} | ${ohlc['c']:.2f} | {_format_pct(actual)} | {hit_text} |"
                )
    else:
        for asset in ASSETS:
            ohlc = market_data.get(asset)
            if ohlc:
                pct = _actual_move(ohlc)
                parts.append(
                    f"| {asset} | ${ohlc['o']:.2f} | ${ohlc['h']:.2f} | "
                    f"${ohlc['l']:.2f} | ${ohlc['c']:.2f} | {_format_pct(pct)} | N/A |"
                )
    parts.append("")

    # --- Trade execution ---
    parts.append("### Trade execution")
    parts.append("")
    if audits:
        filled_list = [t for t in audits if _trade_filled(t)]
        orphaned_list = [t for t in filled_list if not _trade_closed(t)]
        parts.append(f"**{len(audits)}** orders submitted, **{len(filled_list)}** filled")
        if orphaned_list:
            parts.append(
                f"**{len(orphaned_list)}** positions were not closed by the engine (closed manually or by safety-close)."
            )
        parts.append("")
        for tr in filled_list:
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
            parts.append(f"- {line}")
        engine_total = sum(float(t.get("final_pnl", 0) or 0) for t in filled_list)
        if engine_total:
            parts.append(f"\nEngine-tracked PnL: {format_pnl(engine_total)}")
    else:
        parts.append("No trades executed today.")
    parts.append("")

    # --- Strategy summary (compact) ---
    if summary:
        strategy_result = _strategy_accuracy_compact(summary, market_data)
        if strategy_result:
            parts.append("### Strategy summary")
            parts.append("")
            parts.append(strategy_result)
            parts.append("")

    # --- Cumulative record ---
    history = _load_prediction_history()
    if history:
        deduped = _dedupe_history(history)
        wins = sum(1 for h in deduped if h.get("result") == "success")
        valid = len(deduped)
        rate = wins / valid * 100 if valid > 0 else 0
        parts.append(f"**Cumulative prediction record**: {wins}/{valid} ({rate:.0f}%)")

    return "\n".join(parts) + "\n"


def _strategy_accuracy_compact(summary: dict, market_data: dict[str, dict | None]) -> str:
    lines: list[str] = []
    lines.append("| Strategy | SPY | QQQ |")
    lines.append("|----------|-----|-----|")
    for r in summary.get("decision", {}).get("all_results", []):
        label = r.get("label", "unknown")
        predictions = r.get("predictions") or {}
        if not isinstance(predictions, dict):
            lines.append(f"| {label} | — | — |")
            continue
        cols: list[str] = [label]
        for asset in ASSETS:
            pred = predictions.get(asset)
            if isinstance(pred, dict):
                direction = (pred.get("direction") or "?").upper()
                outcome = _legacy_outcome_label(pred, market_data.get(asset))
                icon = ":white_check_mark:" if outcome == "success" else ":x:"
                cols.append(f"{icon} {direction}")
            else:
                cols.append("—")
        lines.append(f"| {' | '.join(cols)} |")
    return "\n".join(lines)


def _source_tag_dict(summary: dict, market_data: dict[str, dict | None]) -> dict:
    win_tags: list[str] = []
    loss_tags: list[str] = []
    for r in summary.get("decision", {}).get("all_results", []):
        predictions = r.get("predictions") or {}
        if not isinstance(predictions, dict):
            continue
        for asset, pred in predictions.items():
            if not isinstance(pred, dict) or asset not in ASSETS:
                continue
            sources = pred.get("sources", [])
            outcome = _legacy_outcome_label(pred, market_data.get(asset))
            for src in sources:
                tag = f"source:{src.replace(':', '-').replace(' ', '-')}"
                if outcome == "success":
                    win_tags.append(tag)
                    win_tags.append(f"source:{src.replace(':', '-').replace(' ', '-')}-reliable")
                else:
                    loss_tags.append(tag)
                    loss_tags.append(f"source:{src.replace(':', '-').replace(' ', '-')}-unreliable")
    return {"tags": win_tags + loss_tags}


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

    if not existing:
        header = (
            "# Lessons Learned\n\n"
            "## Quick Index\n\n"
            "| Date | SPY | QQQ | Result | Key Lesson | Tags |\n"
            "|------|-----|-----|--------|------------|------|\n\n"
            "## Persistent Patterns\n\n"
            "*Patterns updated manually. See daily entries for raw data.*\n\n"
            "---\n\n"
        )
        path.write_text(header + entry + "\n---\n\n")
        return

    # Find insertion point: after "---" that follows Persistent Patterns,
    # before the first daily date entry.
    patterns_marker = "## Persistent Patterns"
    patterns_pos = existing.find(patterns_marker)
    if patterns_pos == -1:
        # Old format — find first "---" after header
        first_dash = existing.find("---")
        if first_dash == -1:
            path.write_text(existing + "\n" + entry + "\n")
            return
        new_content = existing[: first_dash + 4] + "\n" + entry + "\n" + existing[first_dash + 4 :]
        path.write_text(new_content)
        return

    # Find the "---" that closes the Patterns section
    rest = existing[patterns_pos + len(patterns_marker) :]
    dash_pos = rest.find("\n---\n")
    if dash_pos == -1:
        path.write_text(existing + "\n" + entry + "\n")
        return

    insert_at = patterns_pos + len(patterns_marker) + dash_pos + 5  # after \n---\n
    new_content = (
        existing[:insert_at] + "\n" + entry + "\n---\n\n" + existing[insert_at:].lstrip("\n")
    )
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
