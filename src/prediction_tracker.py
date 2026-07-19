from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import structlog

from src.ingestion.fetcher import FinnhubFetcher
from src.models.recommendation import PredictionOutcome
from src.timezone import LA_TZ, today_local

logger = structlog.get_logger()

HISTORY_FILENAME = "prediction-history.json"


async def find_previous_business_day(
    log_dir: str | Path,
    fetcher: FinnhubFetcher,
    max_skip: int = 7,
) -> date | None:
    cand = today_local() - timedelta(days=1)
    for _ in range(max_skip):
        if cand.weekday() >= 5:
            cand -= timedelta(days=1)
            continue
        try:
            candle = await fetcher.fetch_daily_candle("SPY", cand)
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
) -> PredictionOutcome:
    asset = pred["asset"]
    direction = pred["direction"]
    confidence = pred["confidence"]
    rationale = pred.get("rationale", "")
    cid = pred.get("correlation_id", "")
    pred_date = today_local().isoformat()

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

    triggered_at_str = ""
    duration_h: float | None = None

    price_moved = h > o if direction == "UP" else lo < o

    if hourly_candles and price_moved:
        if direction == "UP":
            for i, candle in enumerate(hourly_candles):
                if candle["high"] > o:
                    ts = datetime.fromtimestamp(candle["timestamp"], tz=LA_TZ)
                    triggered_at_str = ts.strftime("%I:%M %p %Z")
                    count = 1
                    for j in range(i + 1, len(hourly_candles)):
                        if hourly_candles[j]["high"] > o:
                            count += 1
                        else:
                            break
                    duration_h = count
                    break
        else:
            for i, candle in enumerate(hourly_candles):
                if candle["low"] < o:
                    ts = datetime.fromtimestamp(candle["timestamp"], tz=LA_TZ)
                    triggered_at_str = ts.strftime("%I:%M %p %Z")
                    count = 1
                    for j in range(i + 1, len(hourly_candles)):
                        if hourly_candles[j]["low"] < o:
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
            f"{asset} opened at ${o:.2f} and never {'rose above' if direction == 'UP' else 'fell below'} "
            f"that level. Daily range: ${lo:.2f} - ${h:.2f} | Close: ${c_val:.2f}"
        )
    else:
        result = "success"
        if triggered_at_str:
            parts = [f"Triggered at {triggered_at_str}"]
            if duration_h is not None and duration_h > 1:
                parts.append(f"held for ~{duration_h} hours")
            parts.append(f"Daily range: ${lo:.2f} - ${h:.2f} | Close: ${c_val:.2f}")
            details_parts.append(" | ".join(parts))
        else:
            details_parts.append(
                f"{asset} {'rose above' if direction == 'UP' else 'fell below'} "
                f"the open of ${o:.2f}. "
                f"Daily range: ${lo:.2f} - ${h:.2f} | Close: ${c_val:.2f}"
            )

    return PredictionOutcome(
        date=pred_date,
        correlation_id=cid,
        asset=asset,
        predicted_direction=direction,
        confidence=confidence,
        rationale=rationale,
        result=result,
        details="".join(details_parts),
        open_price=o,
        high_price=h,
        low_price=lo,
        close_price=c_val,
        triggered_at=triggered_at_str,
        duration_hours=duration_h,
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
    if not outcomes:
        return
    path = _history_path(log_dir)
    existing = load_history(log_dir)
    existing.extend(o.model_dump(mode="json") for o in outcomes)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)
    except OSError as e:
        logger.error("prediction_history_write_error", path=str(path), error=str(e))


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
