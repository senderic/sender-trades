import collections
import json
from pathlib import Path

import pytest

from src.models.recommendation import PredictionOutcome
from src.prediction_tracker import (
    _compute_per_asset_record,
    _compute_source_reliability,
    append_outcomes,
    format_history_for_prompt,
    load_history,
)


def _entry(date, asset, direction, result, sources=None):
    return {
        "date": date,
        "asset": asset,
        "predicted_direction": direction,
        "confidence": 0.60,
        "result": result,
        "details": f"{asset} {direction} {result}",
        "sources": sources or [],
    }


class TestComputePerAssetRecord:
    def test_basic_counts(self) -> None:
        history = [
            _entry("2026-07-29", "QQQ", "DOWN", "success"),
            _entry("2026-07-29", "SPY", "UP", "fail"),
            _entry("2026-07-30", "QQQ", "UP", "success"),
            _entry("2026-07-30", "SPY", "DOWN", "fail"),
        ]
        records = _compute_per_asset_record(history)
        assert records["QQQ"]["success"] == 2
        assert records["QQQ"]["fail"] == 0
        assert records["QQQ"]["total"] == 2
        assert records["SPY"]["success"] == 0
        assert records["SPY"]["fail"] == 2
        assert records["SPY"]["total"] == 2

    def test_deduplicates_same_date_asset_direction(self) -> None:
        history = [
            _entry("2026-07-29", "QQQ", "DOWN", "success"),
            _entry("2026-07-29", "QQQ", "DOWN", "success"),
            _entry("2026-07-29", "QQQ", "DOWN", "fail"),
        ]
        records = _compute_per_asset_record(history)
        assert records["QQQ"]["total"] == 1

    def test_direction_breakdown(self) -> None:
        history = [
            _entry("2026-07-29", "QQQ", "DOWN", "success"),
            _entry("2026-07-30", "QQQ", "UP", "fail"),
        ]
        records = _compute_per_asset_record(history)
        assert records["QQQ"]["down_success"] == 1
        assert records["QQQ"]["down_fail"] == 0
        assert records["QQQ"]["up_success"] == 0
        assert records["QQQ"]["up_fail"] == 1

    def test_empty_history(self) -> None:
        assert _compute_per_asset_record([]) == {}


class TestComputeSourceReliability:
    def test_basic_counts(self) -> None:
        history = [
            _entry(
                "2026-07-29", "QQQ", "DOWN", "success", sources=["market:QQQ", "news-sentiment"]
            ),
            _entry("2026-07-30", "QQQ", "UP", "fail", sources=["market:QQQ", "news-sentiment"]),
            _entry("2026-07-30", "SPY", "DOWN", "fail", sources=["market:SPY"]),
        ]
        rel = _compute_source_reliability(history)
        assert rel["market:QQQ"]["correct"] == 1
        assert rel["market:QQQ"]["total"] == 2
        assert rel["news-sentiment"]["correct"] == 1
        assert rel["news-sentiment"]["total"] == 2
        assert "market:SPY" not in rel

    def test_strips_llm_prefix(self) -> None:
        history = [
            _entry(
                "2026-07-29",
                "QQQ",
                "DOWN",
                "success",
                sources=["llm:market:QQQ", "llm:reuters:tech"],
            ),
            _entry("2026-07-30", "QQQ", "DOWN", "success", sources=["market:QQQ", "reuters:tech"]),
        ]
        rel = _compute_source_reliability(history)
        assert "market:QQQ" in rel
        assert rel["market:QQQ"]["total"] == 2
        assert "reuters:tech" in rel

    def test_requires_minimum_two_occurrences(self) -> None:
        history = [
            _entry("2026-07-29", "QQQ", "DOWN", "success", sources=["only-once"]),
        ]
        rel = _compute_source_reliability(history)
        assert "only-once" not in rel

    def test_empty_sources_handled(self) -> None:
        history = [
            _entry("2026-07-29", "QQQ", "DOWN", "success", sources=[]),
            _entry("2026-07-29", "SPY", "UP", "fail", sources=None),
        ]
        rel = _compute_source_reliability(history)
        assert rel == {}

    def test_non_list_sources_skipped(self) -> None:
        history = [
            _entry("2026-07-29", "QQQ", "DOWN", "success", sources="not a list"),
        ]
        rel = _compute_source_reliability(history)
        assert rel == {}


class TestFormatHistoryForPrompt:
    def test_includes_per_asset_section(self) -> None:
        history = [
            _entry("2026-07-29", "QQQ", "DOWN", "success", sources=["market:QQQ"]),
            _entry("2026-07-29", "SPY", "UP", "fail", sources=["market:SPY"]),
            _entry("2026-07-30", "QQQ", "DOWN", "success", sources=["market:QQQ"]),
        ]
        prompt = format_history_for_prompt(history)
        assert "Per-asset prediction record:" in prompt
        assert "QQQ:" in prompt
        assert "SPY:" in prompt

    def test_includes_source_reliability_section(self) -> None:
        history = [
            _entry(
                "2026-07-29", "QQQ", "DOWN", "success", sources=["market:QQQ", "news-sentiment"]
            ),
            _entry(
                "2026-07-30", "QQQ", "DOWN", "success", sources=["market:QQQ", "news-sentiment"]
            ),
            _entry("2026-07-30", "SPY", "UP", "fail", sources=["market:SPY", "news-sentiment"]),
        ]
        prompt = format_history_for_prompt(history)
        assert "Source reliability" in prompt
        assert "market:QQQ" in prompt

    def test_empty_history_returns_empty_string(self) -> None:
        assert format_history_for_prompt([]) == ""

    def test_no_sources_skips_reliability_section(self) -> None:
        history = [
            _entry("2026-07-29", "QQQ", "DOWN", "success"),
        ]
        prompt = format_history_for_prompt(history)
        assert "Source reliability" not in prompt


def _outcome(date, asset, direction="UP", confidence=0.6, result="success"):
    return PredictionOutcome(
        date=date,
        correlation_id="cid-1",
        asset=asset,
        predicted_direction=direction,
        confidence=confidence,
        result=result,
    )


class TestAppendOutcomes:
    def test_new_date_asset_appends(self, tmp_path) -> None:
        log_dir = tmp_path / "logs"
        append_outcomes(str(log_dir), [_outcome("2026-08-14", "SPY")])
        append_outcomes(str(log_dir), [_outcome("2026-08-15", "SPY")])

        history = load_history(str(log_dir))
        assert len(history) == 2
        assert [(h["date"], h["asset"]) for h in history] == [
            ("2026-08-14", "SPY"),
            ("2026-08-15", "SPY"),
        ]

    def test_rerun_same_date_asset_replaces_not_appends(self, tmp_path) -> None:
        log_dir = tmp_path / "logs"
        append_outcomes(str(log_dir), [_outcome("2026-08-14", "SPY", result="fail")])
        append_outcomes(str(log_dir), [_outcome("2026-08-14", "SPY", result="success")])

        history = load_history(str(log_dir))
        assert len(history) == 1
        assert history[0]["result"] == "success"

    def test_replace_keeps_original_position(self, tmp_path) -> None:
        log_dir = tmp_path / "logs"
        append_outcomes(
            str(log_dir),
            [
                _outcome("2026-08-14", "SPY"),
                _outcome("2026-08-14", "QQQ"),
            ],
        )
        # Re-run SPY only -- it must keep its original (first) slot, not
        # move to the end, so the file still reads chronologically.
        append_outcomes(str(log_dir), [_outcome("2026-08-14", "SPY", result="fail")])

        history = load_history(str(log_dir))
        assert [(h["date"], h["asset"]) for h in history] == [
            ("2026-08-14", "SPY"),
            ("2026-08-14", "QQQ"),
        ]
        assert history[0]["result"] == "fail"

    def test_empty_outcomes_is_noop(self, tmp_path) -> None:
        log_dir = tmp_path / "logs"
        append_outcomes(str(log_dir), [])
        assert load_history(str(log_dir)) == []


class TestPredictionHistoryFileHasNoDuplicates:
    def test_cleaned_history_file_has_no_duplicate_date_asset_pairs(self) -> None:
        history_path = Path(__file__).resolve().parent.parent / "logs" / "prediction-history.json"
        if not history_path.exists():
            pytest.skip("logs/prediction-history.json not present in this environment")
        history = json.loads(history_path.read_text())
        counts = collections.Counter((r["date"], r["asset"]) for r in history)
        duplicates = [key for key, count in counts.items() if count > 1]
        assert duplicates == []
