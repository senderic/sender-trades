"""Unit tests for the pure scoring/aggregation functions in
``scripts/replay_models.py``.

These tests deliberately avoid touching the network, the opencode CLI,
or any real atlas-morning-briefing/logs data — they exercise only the
scoring math and response-parsing helpers, which is what the replay
harness's correctness actually hinges on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# scripts/ is not a package on sys.path by default when running under
# pytest's rootdir; add the repo root explicitly so `import scripts.*`
# resolves the same way `uv run python scripts/replay_models.py` does.
# This must run before the module import below is reached.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import replay_models as rm  # noqa: E402  # isort: skip


# ─────────────────────────── score_direction ───────────────────────────


def test_score_direction_up_correct():
    assert rm.score_direction("UP", open_=100.0, close=101.0) is True


def test_score_direction_up_wrong():
    assert rm.score_direction("UP", open_=100.0, close=99.0) is False


def test_score_direction_down_correct():
    assert rm.score_direction("DOWN", open_=100.0, close=98.0) is True


def test_score_direction_down_wrong():
    assert rm.score_direction("DOWN", open_=100.0, close=100.5) is False


def test_score_direction_flat_close_counts_as_wrong_either_way():
    # close == open is neither a gain nor a loss; both directions read False.
    assert rm.score_direction("UP", open_=100.0, close=100.0) is False
    assert rm.score_direction("DOWN", open_=100.0, close=100.0) is False


# ─────────────────────────── score_target_hit ───────────────────────────


def test_target_hit_up_reaches_target():
    # Open 100, predicted +1% -> target 101. High of 101.5 clears it.
    assert rm.score_target_hit("UP", 1.0, open_=100.0, high=101.5, low=99.0) is True


def test_target_hit_up_falls_short():
    assert rm.score_target_hit("UP", 1.0, open_=100.0, high=100.5, low=99.0) is False


def test_target_hit_down_reaches_target():
    assert rm.score_target_hit("DOWN", -1.0, open_=100.0, high=100.2, low=98.5) is True


def test_target_hit_down_falls_short():
    assert rm.score_target_hit("DOWN", -1.0, open_=100.0, high=100.2, low=99.5) is False


def test_target_hit_small_magnitude_degrades_to_open_threshold():
    # |move_pct| < 0.1 -> threshold is just the open itself.
    assert rm.score_target_hit("UP", 0.02, open_=100.0, high=100.01, low=99.9) is True
    assert rm.score_target_hit("UP", 0.02, open_=100.0, high=100.0, low=99.9) is False


# ─────────────────────────── score_expiry_return ───────────────────────────
#
# otm_pct=0.0 below isolates the return-scaling formula from the OTM
# distance (i.e. tests the "as if ATM" case); separate tests below cover
# OTM's effect specifically.


def test_expiry_return_correct_direction_scales_with_move():
    # UP, open 100 -> close 100.30 (+0.30% move), ATM, premium 0.15% -> ret = 0.30/0.15 - 1 = 1.0
    ret = rm.score_expiry_return("UP", open_=100.0, close=100.30, otm_pct=0.0, premium_pct=0.15)
    assert ret == pytest.approx(1.0)


def test_expiry_return_exact_breakeven_at_one_premium_width():
    # Move exactly equal to the premium -> intrinsic == premium paid -> ret == 0.
    ret = rm.score_expiry_return("UP", open_=100.0, close=100.15, otm_pct=0.0, premium_pct=0.15)
    assert ret == pytest.approx(0.0)


def test_expiry_return_wrong_direction_is_full_loss():
    # Predicted UP but closed down -> intrinsic 0 -> ret == -1 (full premium loss).
    ret = rm.score_expiry_return("UP", open_=100.0, close=99.5, otm_pct=0.0, premium_pct=0.15)
    assert ret == pytest.approx(-1.0)


def test_expiry_return_flat_close_is_full_loss():
    ret = rm.score_expiry_return("UP", open_=100.0, close=100.0, otm_pct=0.0, premium_pct=0.15)
    assert ret == pytest.approx(-1.0)


def test_expiry_return_down_direction_scales_with_move():
    ret = rm.score_expiry_return("DOWN", open_=100.0, close=99.70, otm_pct=0.0, premium_pct=0.15)
    assert ret == pytest.approx(1.0)


def test_expiry_return_degenerate_open_is_full_loss():
    assert rm.score_expiry_return("UP", open_=0.0, close=1.0) == -1.0


def test_expiry_return_move_short_of_otm_is_full_loss_even_if_correct_direction():
    # Correct direction (+0.30%) but the option is struck 0.37% OTM ->
    # never went in the money -> full -100% loss, same as a wrong call.
    ret = rm.score_expiry_return("UP", open_=100.0, close=100.30, otm_pct=0.37, premium_pct=0.10)
    assert ret == pytest.approx(-1.0)


def test_expiry_return_move_past_otm_scales_by_excess_only():
    # +0.60% move, 0.37% OTM -> intrinsic = 0.23%; premium 0.10% -> ret = 0.23/0.10 - 1 = 1.3
    ret = rm.score_expiry_return("UP", open_=100.0, close=100.60, otm_pct=0.37, premium_pct=0.10)
    assert ret == pytest.approx(1.3)


# ─────────────────────────── score_tp_touch ───────────────────────────


def test_tp_touch_up_reaches_two_premium_widths():
    # ATM (otm=0): 2x 0.15% of 100 = 0.30 threshold; high - open = 0.35 clears it.
    assert rm.score_tp_touch("UP", open_=100.0, high=100.35, low=99.8, otm_pct=0.0, premium_pct=0.15) is True


def test_tp_touch_up_falls_short_of_two_premium_widths():
    # high - open = 0.20 clears one premium-width but not two.
    assert rm.score_tp_touch("UP", open_=100.0, high=100.20, low=99.8, otm_pct=0.0, premium_pct=0.15) is False


def test_tp_touch_down_reaches_two_premium_widths():
    assert rm.score_tp_touch("DOWN", open_=100.0, high=100.1, low=99.65, otm_pct=0.0, premium_pct=0.15) is True


def test_tp_touch_down_falls_short_of_two_premium_widths():
    assert rm.score_tp_touch("DOWN", open_=100.0, high=100.1, low=99.85, otm_pct=0.0, premium_pct=0.15) is False


def test_tp_touch_accounts_for_otm_distance():
    # otm=0.37, premium=0.10 -> threshold = 0.37 + 2*0.10 = 0.57 (0.57 pts on 100).
    assert rm.score_tp_touch("UP", open_=100.0, high=100.58, low=99.8, otm_pct=0.37, premium_pct=0.10) is True
    assert rm.score_tp_touch("UP", open_=100.0, high=100.50, low=99.8, otm_pct=0.37, premium_pct=0.10) is False


# ─────────────────────────── confidence_bucket ───────────────────────────


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        (0.0, "0.0-0.5"),
        (0.49, "0.0-0.5"),
        (0.5, "0.5-0.6"),
        (0.65, "0.6-0.7"),
        (0.79, "0.7-0.8"),
        (0.8, "0.8-1.0"),
        (1.0, "0.8-1.0"),
    ],
)
def test_confidence_bucket(confidence, expected):
    assert rm.confidence_bucket(confidence) == expected


# ─────────────────────────── wilson_ci ───────────────────────────


def test_wilson_ci_zero_n_returns_zero_interval():
    assert rm.wilson_ci(0, 0) == (0.0, 0.0)


def test_wilson_ci_bounds_stay_within_unit_interval():
    lo, hi = rm.wilson_ci(10, 10)
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_ci_contains_point_estimate():
    successes, n = 7, 10
    lo, hi = rm.wilson_ci(successes, n)
    assert lo <= successes / n <= hi


def test_wilson_ci_widens_with_smaller_sample():
    lo_small, hi_small = rm.wilson_ci(5, 10)
    lo_large, hi_large = rm.wilson_ci(50, 100)
    assert (hi_small - lo_small) > (hi_large - lo_large)


# ─────────────────────────── normalize_prediction / parsing ───────────────────────────


def test_normalize_prediction_valid():
    out = rm.normalize_prediction(
        {"direction": "UP", "confidence": 0.7, "predicted_move_pct": 1.25, "rationale": "x", "sources": ["reuters:foo"]}
    )
    assert out == {
        "direction": "UP",
        "confidence": 0.7,
        "predicted_move_pct": 1.25,
        "rationale": "x",
        "sources": ["reuters:foo"],
    }


def test_normalize_prediction_clamps_confidence():
    out = rm.normalize_prediction({"direction": "DOWN", "confidence": 5.0, "predicted_move_pct": -1.0})
    assert out is not None
    assert out["confidence"] == 1.0


def test_normalize_prediction_invalid_direction_is_none():
    assert rm.normalize_prediction({"direction": "SIDEWAYS", "confidence": 0.5}) is None


def test_normalize_prediction_non_dict_is_none():
    assert rm.normalize_prediction("not a dict") is None
    assert rm.normalize_prediction(None) is None


def test_normalize_prediction_non_numeric_confidence_defaults_zero():
    out = rm.normalize_prediction({"direction": "UP", "confidence": "high"})
    assert out is not None
    assert out["confidence"] == 0.0


def test_parse_monolithic_predictions_happy_path():
    response = json.dumps(
        {
            "predictions": {
                "SPY": {"direction": "UP", "confidence": 0.6, "predicted_move_pct": 0.5},
                "QQQ": {"direction": "DOWN", "confidence": 0.55, "predicted_move_pct": -0.4},
            },
            "market_vibe": "mixed",
        }
    )
    out = rm.parse_monolithic_predictions(response)
    assert set(out.keys()) == {"SPY", "QQQ"}
    assert out["SPY"]["direction"] == "UP"
    assert out["QQQ"]["direction"] == "DOWN"


def test_parse_monolithic_predictions_partial_abstain():
    # Only SPY present -> QQQ is absent (== abstain for that asset).
    response = json.dumps({"predictions": {"SPY": {"direction": "UP", "confidence": 0.6}}})
    out = rm.parse_monolithic_predictions(response)
    assert set(out.keys()) == {"SPY"}


def test_parse_monolithic_predictions_unparseable_returns_empty():
    assert rm.parse_monolithic_predictions("not json at all") == {}
    assert rm.parse_monolithic_predictions("") == {}


def test_parse_monolithic_predictions_ignores_unknown_assets():
    response = json.dumps({"predictions": {"TSLA": {"direction": "UP", "confidence": 0.9}}})
    assert rm.parse_monolithic_predictions(response) == {}


# ─────────────────────────── aggregate ───────────────────────────


def test_aggregate_basic_counts():
    records = [
        rm.ReplayRecord(
            date="2026-08-01", asset="SPY", model="model-a", status="predict",
            direction="UP", confidence=0.7, predicted_move_pct=1.0,
            direction_correct=True, target_hit=True, expiry_return=1.0, tp_touch=True,
        ),
        rm.ReplayRecord(
            date="2026-08-02", asset="SPY", model="model-a", status="predict",
            direction="DOWN", confidence=0.6, predicted_move_pct=-1.0,
            direction_correct=False, target_hit=False, expiry_return=-1.0, tp_touch=False,
        ),
        rm.ReplayRecord(date="2026-08-03", asset="SPY", model="model-a", status="abstain"),
        rm.ReplayRecord(date="2026-08-04", asset="SPY", model="model-a", status="fail", error="timeout"),
    ]
    rows = rm.aggregate(records)
    row = rows["model-a"]
    assert row["n"] == 4
    assert row["predicted_n"] == 2
    assert row["direction_accuracy"] == 0.5
    assert row["abstain_rate"] == 0.25
    assert row["fail_rate"] == 0.25
    assert row["target_hit_accuracy"] == 0.5
    assert row["mean_expiry_return"] == 0.0
    assert row["expiry_win_rate"] == 0.5
    assert row["tp_touch_rate"] == 0.5


def test_aggregate_separates_models():
    records = [
        rm.ReplayRecord(
            date="2026-08-01", asset="SPY", model="model-a", status="predict",
            direction="UP", confidence=0.9, direction_correct=True,
        ),
        rm.ReplayRecord(
            date="2026-08-01", asset="SPY", model="model-b", status="predict",
            direction="DOWN", confidence=0.9, direction_correct=False,
        ),
    ]
    rows = rm.aggregate(records)
    assert rows["model-a"]["direction_accuracy"] == 1.0
    assert rows["model-b"]["direction_accuracy"] == 0.0


def test_aggregate_all_abstain_direction_accuracy_is_none():
    records = [rm.ReplayRecord(date="2026-08-01", asset="SPY", model="model-a", status="abstain")]
    rows = rm.aggregate(records)
    assert rows["model-a"]["direction_accuracy"] is None
    assert rows["model-a"]["target_hit_accuracy"] is None
    assert rows["model-a"]["mean_expiry_return"] is None
    assert rows["model-a"]["expiry_win_rate"] is None
    assert rows["model-a"]["tp_touch_rate"] is None


def test_aggregate_calibration_buckets_by_confidence():
    records = [
        rm.ReplayRecord(
            date=f"2026-08-{i:02d}", asset="SPY", model="model-a", status="predict",
            direction="UP", confidence=0.9, direction_correct=(i % 2 == 0),
        )
        for i in range(1, 5)
    ]
    rows = rm.aggregate(records)
    calibration = rows["model-a"]["calibration"]
    assert "0.8-1.0" in calibration
    assert calibration["0.8-1.0"]["n"] == 4
    assert calibration["0.8-1.0"]["accuracy"] == 0.5


# ─────────────────────────── find_excluded_models ───────────────────────────


def test_find_excluded_models_detects_all_credit_exhausted_failures():
    records = [
        rm.ReplayRecord(
            date="2026-08-01", asset="SPY", model="broke-model", status="fail",
            error="...but can only afford 9532...add more credits...",
        ),
        rm.ReplayRecord(
            date="2026-08-02", asset="QQQ", model="broke-model", status="fail",
            error="...insufficient credits...",
        ),
    ]
    excluded = rm.find_excluded_models(records)
    assert "broke-model" in excluded
    assert "out of credits" in excluded["broke-model"]


def test_find_excluded_models_excludes_even_with_some_earlier_successes():
    # A model that ran fine for a while and then hit a credit wall is still
    # excluded — its later dates are systematically missing, which biases
    # the sample rather than just shrinking it. The count of successes is
    # surfaced in the reason string for transparency.
    records = [
        rm.ReplayRecord(date="2026-08-01", asset="SPY", model="ok-then-broke", status="predict", direction="UP"),
        rm.ReplayRecord(date="2026-08-01", asset="QQQ", model="ok-then-broke", status="predict", direction="UP"),
        rm.ReplayRecord(
            date="2026-08-02", asset="QQQ", model="ok-then-broke", status="fail",
            error="...can only afford...",
        ),
    ]
    excluded = rm.find_excluded_models(records)
    assert "ok-then-broke" in excluded
    assert "n=2 ok" in excluded["ok-then-broke"]


def test_find_excluded_models_ignores_non_credit_failures():
    records = [
        rm.ReplayRecord(date="2026-08-01", asset="SPY", model="flaky-model", status="fail", error="timeout after 150s"),
    ]
    assert rm.find_excluded_models(records) == {}


# ─────────────────────────── no-look-ahead date filtering ───────────────────────────


def test_history_before_excludes_same_and_future_dates():
    history = [
        {"date": "2026-08-01", "asset": "SPY"},
        {"date": "2026-08-05", "asset": "SPY"},
        {"date": "2026-08-10", "asset": "SPY"},
    ]

    def fake_load_history(_log_dir):
        return history

    original = rm.load_history
    rm.load_history = fake_load_history  # type: ignore[assignment]
    try:
        from datetime import date

        result = rm.history_before(date(2026, 8, 5))
    finally:
        rm.load_history = original  # type: ignore[assignment]

    dates = {h["date"] for h in result}
    assert dates == {"2026-08-01"}


# ─────────────────────────── estimate_option_pricing_from_fills ───────────────────────────


def test_estimate_option_pricing_from_fills_falls_back_when_no_files(tmp_path):
    pricing = rm.estimate_option_pricing_from_fills(log_dir=tmp_path)
    assert pricing["n"] == 0
    assert pricing["otm_pct"] == rm.DEFAULT_OTM_PCT
    assert pricing["premium_pct"] == rm.DEFAULT_PREMIUM_PCT


def test_estimate_option_pricing_from_fills_joins_real_fills(tmp_path):
    # One synthetic resolved trade: SPY CALL struck 3.00 above a 100.00 open,
    # filled at 0.20 -> OTM% = 3.00/100*100 = 3.0%, premium% = 0.20/100*100 = 0.2%.
    history_path = tmp_path / "prediction-history.json"
    history_path.write_text(
        json.dumps([{"date": "2026-08-01", "asset": "SPY", "open_price": 100.0}])
    )
    day_dir = tmp_path / "2026-08-01"
    day_dir.mkdir()
    trade = {
        "asset": "SPY",
        "direction": "CALL",
        "entry_strike": 103.0,
        "entries": [{"event_type": "entry_filled", "avg_price": 0.20}],
    }
    (day_dir / "trade-abc123.json").write_text(json.dumps(trade))

    pricing = rm.estimate_option_pricing_from_fills(log_dir=tmp_path)
    assert pricing["n"] == 1
    assert pricing["otm_pct"] == pytest.approx(3.0)
    assert pricing["premium_pct"] == pytest.approx(0.2)


def test_estimate_option_pricing_from_fills_signs_itm_and_skips_unsigned(tmp_path):
    # A PUT struck ABOVE the open is in the money: its distance must be
    # negative, not counted as OTM (the 2026-08-18 / 09-01 fills). A fill
    # with no CALL/PUT direction can't be signed and is skipped.
    history_path = tmp_path / "prediction-history.json"
    history_path.write_text(
        json.dumps([{"date": "2026-08-01", "asset": "QQQ", "open_price": 100.0}])
    )
    day_dir = tmp_path / "2026-08-01"
    day_dir.mkdir()
    filled = [{"event_type": "entry_filled", "avg_price": 0.50}]
    (day_dir / "trade-itm.json").write_text(
        json.dumps({"asset": "QQQ", "direction": "PUT", "entry_strike": 101.0, "entries": filled})
    )
    (day_dir / "trade-unsigned.json").write_text(
        json.dumps({"asset": "QQQ", "entry_strike": 104.0, "entries": filled})
    )

    pricing = rm.estimate_option_pricing_from_fills(log_dir=tmp_path)
    assert pricing["n"] == 1
    assert pricing["otm_pct"] == pytest.approx(-1.0)


def test_estimate_option_pricing_from_fills_skips_unresolved_and_bak(tmp_path):
    history_path = tmp_path / "prediction-history.json"
    history_path.write_text(
        json.dumps([{"date": "2026-08-01", "asset": "SPY", "open_price": 100.0}])
    )
    day_dir = tmp_path / "2026-08-01"
    day_dir.mkdir()
    # No entry_filled event -> not a real fill -> skipped.
    (day_dir / "trade-nofill.json").write_text(json.dumps({"asset": "SPY", "entry_strike": 103.0, "entries": []}))
    # .bak files are always skipped.
    (day_dir / "trade-abc123.json.bak").write_text(
        json.dumps({"asset": "SPY", "entry_strike": 103.0, "entries": [{"event_type": "entry_filled", "avg_price": 0.2}]})
    )
    pricing = rm.estimate_option_pricing_from_fills(log_dir=tmp_path)
    assert pricing["n"] == 0


# ─────────────────────────── premium_for_otm ───────────────────────────


def test_premium_for_otm_at_atm_returns_atm_assumption():
    p = rm.premium_for_otm(0.0, atm_premium_pct=0.30, anchor_otm_pct=0.37, anchor_premium_pct=0.10)
    assert p == pytest.approx(0.30)


def test_premium_for_otm_at_anchor_returns_anchor_premium():
    p = rm.premium_for_otm(0.37, atm_premium_pct=0.30, anchor_otm_pct=0.37, anchor_premium_pct=0.10)
    assert p == pytest.approx(0.10)


def test_premium_for_otm_extrapolates_beyond_anchor():
    # Linear: slope = (0.10 - 0.30) / 0.37 per 1% OTM; at 0.6% OTM the
    # premium should be below the anchor's 0.10% (further OTM = cheaper).
    p = rm.premium_for_otm(0.6, atm_premium_pct=0.30, anchor_otm_pct=0.37, anchor_premium_pct=0.10)
    assert p < 0.10


def test_premium_for_otm_never_returns_non_positive():
    # Even a pathological anchor shouldn't produce a zero/negative premium
    # that would divide-by-zero downstream.
    p = rm.premium_for_otm(100.0, atm_premium_pct=0.30, anchor_otm_pct=0.37, anchor_premium_pct=0.10)
    assert p > 0.0


# ─────────────────────────── compute_sensitivity_table ───────────────────────────


def test_compute_sensitivity_table_recomputes_from_stored_prices():
    records = [
        rm.ReplayRecord(
            date="2026-08-01", asset="SPY", model="model-a", status="predict",
            direction="UP", open_price=100.0, close_price=100.60,
        ),
    ]
    table, premiums = rm.compute_sensitivity_table(
        records, otm_grid=(0.0, 0.37), anchor_otm_pct=0.37, anchor_premium_pct=0.10
    )
    assert 0.0 in premiums
    assert 0.37 in premiums
    # At OTM 0.37 with the anchor premium 0.10: intrinsic = 0.6-0.37=0.23 -> ret = 0.23/0.10-1 = 1.3
    assert table["model-a"][0.37] == pytest.approx(1.3)


def test_compute_sensitivity_table_ignores_non_predict_records():
    records = [
        rm.ReplayRecord(date="2026-08-01", asset="SPY", model="model-a", status="fail", error="x"),
        rm.ReplayRecord(date="2026-08-02", asset="SPY", model="model-a", status="abstain"),
    ]
    table, _ = rm.compute_sensitivity_table(records, otm_grid=(0.0,))
    assert table == {}
