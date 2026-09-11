"""Tests for the LESSONS_LEARNED.md scorer (src.lessons_analyzer).

Covers the 2026-09-10 fix to `_legacy_outcome_label`'s sign bug (it used
to compute the target strike from `abs(predicted_move_pct)`, which put
the target on the WRONG SIDE of the open for every DOWN prediction and
made DOWN calls register HIT far more often than they should have) and
the new three-way `_outcome_note` labeling ("target reached" /
"touched then reversed" / "direction open->close [correct|wrong]").
"""

from __future__ import annotations

from src.lessons_analyzer import _legacy_outcome_label, _outcome_note, build_lessons_md


class TestLegacyOutcomeLabelSignFix:
    """Regression coverage for the 2026-09-09 QQQ DOWN false-HIT bug.

    Real numbers from logs/prediction-history.json (the strict,
    correctly-signed source of truth): QQQ predicted DOWN -0.8%, opened
    716.40, low 714.02, high 719.70, closed 716.31. The correct target is
    BELOW the open (716.40 * (1 - 0.008) = 710.67); the low (714.02) never
    reached it, so this should score "fail" -- prediction_tracker.py
    agrees (result: "fail" in prediction-history.json). The buggy version
    used abs(-0.8) and placed the target ABOVE the open (716.40 * 1.008 =
    722.13), then checked `lo <= target`, which is nearly always true and
    silently registered "success".
    """

    def test_down_prediction_uses_correctly_signed_target(self) -> None:
        forecast = {"direction": "DOWN", "predicted_move_pct": -0.8}
        ohlc = {"o": 716.40, "h": 719.70, "l": 714.02, "c": 716.31}
        assert _legacy_outcome_label(forecast, ohlc) == "fail"

    def test_down_prediction_success_when_target_actually_touched(self) -> None:
        forecast = {"direction": "DOWN", "predicted_move_pct": -0.8}
        ohlc = {"o": 716.40, "h": 719.70, "l": 705.00, "c": 706.00}
        assert _legacy_outcome_label(forecast, ohlc) == "success"

    def test_up_prediction_unaffected_by_sign_fix(self) -> None:
        # UP predictions were never affected (abs() was a no-op for a
        # positive predicted_move_pct), so this must keep working.
        forecast = {"direction": "UP", "predicted_move_pct": 0.8}
        ohlc = {"o": 716.40, "h": 723.00, "l": 714.02, "c": 720.00}
        assert _legacy_outcome_label(forecast, ohlc) == "success"

    def test_matches_prediction_tracker_check_outcome_semantics(self) -> None:
        """Same (direction, move, OHLC) as prediction_tracker.check_outcome
        must now agree with the strict source of truth."""
        from src.prediction_tracker import check_outcome

        pred = {
            "asset": "QQQ",
            "direction": "DOWN",
            "confidence": 0.65,
            "predicted_move_pct": -0.8,
        }
        daily_candle = {"o": [716.40], "h": [719.70], "l": [714.02], "c": [716.31]}
        strict = check_outcome(pred, daily_candle, None)

        forecast = {"direction": "DOWN", "predicted_move_pct": -0.8}
        ohlc = {"o": 716.40, "h": 719.70, "l": 714.02, "c": 716.31}
        legacy = _legacy_outcome_label(forecast, ohlc)

        assert legacy == strict.result == "fail"


class TestOutcomeNote:
    def test_target_reached_and_held(self) -> None:
        ohlc = {"o": 100.0, "h": 100.5, "l": 98.5, "c": 98.7}
        note = _outcome_note("DOWN", ohlc, hit=True)
        assert "target reached" in note
        assert "reversed" not in note

    def test_touched_then_reversed(self) -> None:
        ohlc = {"o": 100.0, "h": 100.5, "l": 98.5, "c": 101.2}
        note = _outcome_note("DOWN", ohlc, hit=True)
        assert "touched then reversed" in note

    def test_never_hit_but_direction_correct(self) -> None:
        ohlc = {"o": 100.0, "h": 100.2, "l": 99.5, "c": 99.8}
        note = _outcome_note("DOWN", ohlc, hit=False)
        assert "direction open→close correct" in note

    def test_never_hit_and_direction_wrong(self) -> None:
        ohlc = {"o": 100.0, "h": 100.2, "l": 99.5, "c": 100.3}
        note = _outcome_note("DOWN", ohlc, hit=False)
        assert "direction open→close wrong" in note


class TestCumulativeRecordStrict:
    def test_excludes_unknown_from_denominator(self, tmp_path, monkeypatch) -> None:
        import json as _json

        history = [
            {
                "date": "2026-09-01",
                "asset": "SPY",
                "predicted_direction": "UP",
                "result": "success",
            },
            {"date": "2026-09-02", "asset": "SPY", "predicted_direction": "DOWN", "result": "fail"},
            {
                "date": "2026-09-03",
                "asset": "SPY",
                "predicted_direction": "UP",
                "result": "unknown",
            },
        ]
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "prediction-history.json").write_text(_json.dumps(history))

        monkeypatch.chdir(tmp_path)
        entry = build_lessons_md(
            audits=[],
            summaries=[],
            market_data={"SPY": None, "QQQ": None},
        )
        # 1/2 (50%), not 1/3 -- the "unknown" entry must not dilute the rate.
        assert "1/2 (50%)" in entry
        assert "1/3" not in entry
