from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import structlog

from src.ingestion.candle_providers import CandleProvider
from src.models.recommendation import PredictionOutcome
from src.timezone import LA_TZ, today_local

logger = structlog.get_logger()

HISTORY_FILENAME = "prediction-history.json"


async def find_previous_business_day(
    log_dir: str | Path,
    provider: CandleProvider,
    max_skip: int = 7,
) -> date | None:
    cand = today_local() - timedelta(days=1)
    for _ in range(max_skip):
        if cand.weekday() >= 5:
            cand -= timedelta(days=1)
            continue
        try:
            candle = await provider.fetch_daily_candle("SPY", cand)
        except Exception:
            candle = None
        if candle is not None:
            return cand
        cand -= timedelta(days=1)
    return None


def read_previous_forecasts(
    log_dir: str | Path,
    target_date: date,
) -> list[dict]:
    day_dir = Path(log_dir).expanduser().resolve() / target_date.isoformat()
    if not day_dir.is_dir():
        logger.warning("prev_prediction_no_log_dir", day_dir=str(day_dir))
        return []

    summaries = sorted(day_dir.glob("summary-*.json"), reverse=True)
    if not summaries:
        logger.warning("prev_prediction_no_summary", day_dir=str(day_dir))
        return []

    try:
        with open(summaries[0]) as f:
            summary = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("prev_prediction_read_error", path=str(summaries[0]), error=str(e))
        return []

    forecasts_raw = summary.get("decision", {}).get("forecast", {}).get("forecasts", [])
    if not forecasts_raw:
        logger.warning("prev_prediction_no_forecasts", path=str(summaries[0]))
        return []

    results: list[dict] = []
    for f in forecasts_raw:
        asset = f.get("asset")
        if asset not in ("SPY", "QQQ"):
            continue

        direction: str | None = f.get("direction")
        confidence: float = f.get("confidence", 0.0)

        if not direction:
            up = f.get("up_confidence", 0.0) or 0.0
            down = f.get("down_confidence", 0.0) or 0.0
            if up > down and up > 0.01:
                direction = "UP"
                confidence = up
            elif down > up and down > 0.01:
                direction = "DOWN"
                confidence = down
            else:
                continue

        results.append(
            {
                "asset": asset,
                "direction": direction,
                "confidence": round(confidence, 4),
                "predicted_move_pct": f.get("predicted_move_pct")
                or f.get("expected_move_pct", 0.0),
                "rationale": f.get("rationale", ""),
                "sources": f.get("sources", []),
                "correlation_id": summary.get("correlation_id", ""),
            }
        )

    return results


def check_outcome(
    pred: dict,
    daily_candle: dict | None,
    hourly_candles: list[dict] | None,
    biz_date: date | None = None,
) -> PredictionOutcome:
    asset = pred["asset"]
    direction = pred["direction"]
    confidence = pred["confidence"]
    rationale = pred.get("rationale", "")
    cid = pred.get("correlation_id", "")
    move_pct = pred.get("predicted_move_pct", 0.0) or 0.0
    pred_date = (biz_date or today_local()).isoformat()

    if daily_candle is None:
        return PredictionOutcome(
            date=pred_date,
            correlation_id=cid,
            asset=asset,
            predicted_direction=direction,
            confidence=confidence,
            rationale=rationale,
            result="unknown",
            details="No trading data available for this date.",
        )

    o = float(daily_candle["o"][0])
    h = float(daily_candle["h"][0])
    lo = float(daily_candle["l"][0])
    c_val = float(daily_candle["c"][0])

    target_strike: float | None = None
    if abs(move_pct) >= 0.1:
        target_strike = round(o * (1 + move_pct / 100), 2)

    triggered_at_str = ""
    duration_h: float | None = None

    if target_strike is not None:
        price_moved = h >= target_strike if direction == "UP" else lo <= target_strike
        threshold_label = f"target strike ${target_strike:.2f}"
    else:
        price_moved = h > o if direction == "UP" else lo < o
        threshold_label = f"the open of ${o:.2f}"

    if hourly_candles and price_moved:
        threshold = target_strike if target_strike is not None else o
        if direction == "UP":
            for i, candle in enumerate(hourly_candles):
                if candle["high"] >= threshold:
                    ts = datetime.fromtimestamp(candle["timestamp"], tz=LA_TZ)
                    triggered_at_str = ts.strftime("%I:%M %p %Z")
                    count = 1
                    for j in range(i + 1, len(hourly_candles)):
                        if hourly_candles[j]["high"] >= threshold:
                            count += 1
                        else:
                            break
                    duration_h = count
                    break
        else:
            for i, candle in enumerate(hourly_candles):
                if candle["low"] <= threshold:
                    ts = datetime.fromtimestamp(candle["timestamp"], tz=LA_TZ)
                    triggered_at_str = ts.strftime("%I:%M %p %Z")
                    count = 1
                    for j in range(i + 1, len(hourly_candles)):
                        if hourly_candles[j]["low"] <= threshold:
                            count += 1
                        else:
                            break
                    duration_h = count
                    break

    result: str
    details_parts: list[str] = []

    if not price_moved:
        result = "fail"
        details_parts.append(
            f"{asset} opened at ${o:.2f} and never hit {threshold_label}. "
            f"Daily range: ${lo:.2f} - ${h:.2f} | Close: ${c_val:.2f}"
        )
    else:
        result = "success"
        if triggered_at_str:
            parts = [f"Hit {threshold_label} at {triggered_at_str}"]
            if duration_h is not None and duration_h > 1:
                parts.append(
                    f"held above for ~{duration_h} hours"
                    if direction == "UP"
                    else f"held below for ~{duration_h} hours"
                )
            parts.append(f"Daily range: ${lo:.2f} - ${h:.2f} | Close: ${c_val:.2f}")
            details_parts.append(" | ".join(parts))
        else:
            details_parts.append(
                f"{asset} hit {threshold_label}. "
                f"Daily range: ${lo:.2f} - ${h:.2f} | Close: ${c_val:.2f}"
            )

    oc_correct = (c_val > o) if direction == "UP" else (c_val < o)

    return PredictionOutcome(
        date=pred_date,
        correlation_id=cid,
        asset=asset,
        predicted_direction=direction,
        confidence=confidence,
        rationale=rationale,
        result=result,
        target_strike=target_strike,
        details="".join(details_parts),
        open_price=o,
        high_price=h,
        low_price=lo,
        close_price=c_val,
        open_close_correct=oc_correct,
        triggered_at=triggered_at_str,
        duration_hours=duration_h,
        sources=pred.get("sources", []),
    )


def _history_path(log_dir: str | Path) -> Path:
    return Path(log_dir).expanduser().resolve() / HISTORY_FILENAME


def load_history(log_dir: str | Path) -> list[dict]:
    path = _history_path(log_dir)
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("prediction_history_read_error", path=str(path), error=str(e))
        return []


def append_outcomes(log_dir: str | Path, outcomes: list[PredictionOutcome]) -> None:
    """Merge new prediction outcomes into the history file, keyed by ``(date, asset)``.

    Idempotent on ``(date, asset)``: re-running the pipeline for a day
    that was already recorded REPLACES the existing record for that key
    with the newer one rather than appending a second one. Duplicates
    would otherwise silently double-weight that day when the history is
    fed back to the LLM via :func:`format_history_for_prompt`.

    Ordering is preserved: a replaced record keeps its original position
    (so the file still reads chronologically), and a genuinely new
    ``(date, asset)`` key is appended at the end, same as before. Any
    pre-existing duplicate keys already in the file are collapsed to
    their last occurrence as a side effect of the merge.

    Args:
        log_dir: Directory containing the prediction history file.
        outcomes: New outcomes to record.
    """
    if not outcomes:
        return
    path = _history_path(log_dir)
    existing = load_history(log_dir)

    indexed: dict[tuple[str, str], dict] = {
        (entry.get("date", ""), entry.get("asset", "")): entry for entry in existing
    }
    order: list[tuple[str, str]] = list(indexed.keys())

    for outcome in outcomes:
        key = (outcome.date, outcome.asset)
        if key not in indexed:
            order.append(key)
        indexed[key] = outcome.model_dump(mode="json")

    merged = [indexed[key] for key in order]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(merged, f, indent=2)
    except OSError as e:
        logger.error("prediction_history_write_error", path=str(path), error=str(e))


def _compute_per_asset_record(
    history: list[dict],
) -> dict[str, dict[str, int]]:
    """Compute per-asset success/fail counts from prediction history.

    Deduplicates by (date, asset) to avoid counting reruns multiple times.

    Args:
        history: List of prediction outcome dicts.

    Returns:
        Dict keyed by asset with ``success``, ``fail``, and ``total`` counts.
        Also includes per-direction breakdowns (``up_success``, etc.).
    """
    seen: set[tuple[str, str, str]] = set()
    records: dict[str, dict[str, int]] = {}

    for entry in history:
        asset = entry.get("asset", "?")
        date_str = entry.get("date", "")
        result = entry.get("result", "unknown")
        direction = entry.get("predicted_direction", "")

        dedup_key = (date_str, asset, direction)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        if asset not in records:
            records[asset] = {
                "success": 0,
                "fail": 0,
                "total": 0,
                "up_success": 0,
                "up_fail": 0,
                "down_success": 0,
                "down_fail": 0,
            }
        records[asset]["total"] += 1
        if result == "success":
            records[asset]["success"] += 1
            if direction == "UP":
                records[asset]["up_success"] += 1
            elif direction == "DOWN":
                records[asset]["down_success"] += 1
        elif result == "fail":
            records[asset]["fail"] += 1
            if direction == "UP":
                records[asset]["up_fail"] += 1
            elif direction == "DOWN":
                records[asset]["down_fail"] += 1

    return records


def _compute_source_reliability(
    history: list[dict],
) -> dict[str, dict[str, int]]:
    """Compute per-source hit rate from prediction history.

    Extracts ``sources`` from each entry, strips the ``llm:`` prefix,
    and tallies correct/total per source tag.

    Args:
        history: List of prediction outcome dicts with ``sources`` field.

    Returns:
        Dict keyed by source tag with ``correct`` and ``total`` counts.
        Only includes sources seen at least twice.
    """
    stats: dict[str, dict[str, int]] = {}

    for entry in history:
        result = entry.get("result", "unknown")
        sources = entry.get("sources", [])
        if not isinstance(sources, list):
            continue
        for src in sources:
            if not isinstance(src, str):
                continue
            clean = src.removeprefix("llm:")
            if not clean:
                continue
            if clean not in stats:
                stats[clean] = {"correct": 0, "total": 0}
            stats[clean]["total"] += 1
            if result == "success":
                stats[clean]["correct"] += 1

    return {k: v for k, v in stats.items() if v["total"] >= 2}


def format_history_for_prompt(
    history: list[dict],
    max_items: int = 5,
) -> str:
    if not history:
        return ""

    total = len(history)
    successes = sum(1 for h in history if h.get("result") == "success")

    parts: list[str] = [
        f"Overall prediction record: {successes}/{total} successful "
        f"({successes / total * 100:.0f}%)"
        if total > 0
        else "No prior predictions."
    ]

    per_asset = _compute_per_asset_record(history)
    if per_asset:
        asset_lines: list[str] = []
        for asset, counts in sorted(per_asset.items()):
            pct = counts["total"]
            up_total = counts["up_success"] + counts["up_fail"]
            down_total = counts["down_success"] + counts["down_fail"]
            bits: list[str] = [
                f"{counts['success']}/{pct} ({counts['success'] / pct * 100:.0f}%)",
            ]
            if up_total > 0:
                bits.append(f"UP {counts['up_success']}/{up_total}")
            if down_total > 0:
                bits.append(f"DOWN {counts['down_success']}/{down_total}")
            asset_lines.append(f"  - {asset}: {' — '.join(bits)}")
        parts.append("Per-asset prediction record:\n" + "\n".join(asset_lines))

    source_rel = _compute_source_reliability(history)
    if source_rel:
        src_lines: list[str] = []
        for src, counts in sorted(source_rel.items(), key=lambda kv: kv[1]["total"], reverse=True):
            pct = counts["total"]
            corr = counts["correct"]
            src_lines.append(f"  - {src}: {corr}/{pct} ({corr / pct * 100:.0f}%)")
        parts.append("Source reliability (from past predictions):\n" + "\n".join(src_lines))

    recent = sorted(history, key=lambda h: h.get("date", ""), reverse=True)

    failures_shown = [h for h in recent if h.get("result") == "fail"][:max_items]
    if failures_shown:
        parts.append("Recent misses:")
        for h in failures_shown:
            date_str = h.get("date", "?")
            asset = h.get("asset", "?")
            pred_dir = h.get("predicted_direction", "?")
            conf = h.get("confidence", 0)
            details = h.get("details", "")
            parts.append(f"  - {date_str}: {asset} {pred_dir} ({conf:.0%} confidence) — {details}")

    successes_shown = [h for h in recent if h.get("result") == "success"]
    remaining = max_items - len(failures_shown)
    if remaining > 0 and successes_shown:
        parts.append("Recent wins:")
        for h in successes_shown[:remaining]:
            date_str = h.get("date", "?")
            asset = h.get("asset", "?")
            pred_dir = h.get("predicted_direction", "?")
            conf = h.get("confidence", 0)
            details = h.get("details", "")
            parts.append(f"  - {date_str}: {asset} {pred_dir} ({conf:.0%} confidence) — {details}")

    return "\n".join(parts)
