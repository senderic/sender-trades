"""Tests for the forward scorecard (daily forecasts vs baselines on real option bars)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import patch

from src.evaluation import forward_scorecard as fs
from src.evaluation.option_replay import ET, DaySetup

DAY = date(2026, 9, 22)
AFTER_CLOSE = datetime(2026, 9, 22, 17, 0, tzinfo=ET)
MIDDAY = datetime(2026, 9, 22, 12, 0, tzinfo=ET)


def _bar(hh: int, mm: int, price: float) -> dict:
    t = datetime(DAY.year, DAY.month, DAY.day, hh, mm, tzinfo=ET).astimezone(UTC)
    return {
        "t": t.isoformat().replace("+00:00", "Z"),
        "o": price,
        "h": price,
        "l": price,
        "c": price,
        "vw": price,
    }


def _path(start: float, end: float) -> list[dict]:
    return [_bar(9, 30, start), _bar(12, 0, (start + end) / 2), _bar(15, 20, end)]


def _setup(asset: str = "QQQ") -> DaySetup:
    # s0=100 -> 0.6% OTM strikes: CALL 101, PUT 99. The call doubles, the put
    # decays -- so CALL forecasts win and PUT forecasts lose.
    return DaySetup(
        day=DAY,
        asset=asset,
        s0=100.0,
        prev_close=99.0,  # gap up -> gap_follow = CALL, gap_fade = PUT
        prev2_close=100.0,  # yesterday down -> repeat_yesterday = PUT
        prev_range_pct=1.0,
        open_=100.0,
        close_1520=101.5,
        options={
            f"{asset}260922C00101000": _path(1.0, 2.0),
            f"{asset}260922C00100000": _path(1.5, 2.25),
            f"{asset}260922P00099000": _path(1.0, 0.8),
        },
    )


def _history(tmp_path: Path, entries: list[dict]) -> Path:
    p = tmp_path / "prediction-history.json"
    p.write_text(json.dumps(entries))
    return p


class TestForecasts:
    def test_maps_directions_and_skips_neutral(self) -> None:
        got = fs.system_forecasts(
            [
                {"date": "2026-09-22", "asset": "QQQ", "predicted_direction": "UP"},
                {"date": "2026-09-22", "asset": "SPY", "predicted_direction": "NEUTRAL"},
                {"date": "2026-09-23", "asset": "SPY", "predicted_direction": "DOWN"},
            ]
        )
        assert got == {("2026-09-22", "QQQ"): "CALL", ("2026-09-23", "SPY"): "PUT"}

    def test_later_entry_wins(self) -> None:
        got = fs.system_forecasts(
            [
                {"date": "2026-09-22", "asset": "QQQ", "predicted_direction": "UP"},
                {"date": "2026-09-22", "asset": "QQQ", "predicted_direction": "DOWN"},
            ]
        )
        assert got[("2026-09-22", "QQQ")] == "PUT"


class TestScoring:
    def test_scores_system_and_baselines_on_same_bars(self) -> None:
        row = fs.score_asset_day(_setup(), "CALL")
        assert row["system"]["strike"] == 101
        assert row["system"]["ret"] > 0
        assert row["baselines"]["gap_follow"]["direction"] == "CALL"
        assert row["baselines"]["gap_fade"]["direction"] == "PUT"
        assert row["baselines"]["repeat_yesterday"]["direction"] == "PUT"
        assert row["baselines"]["gap_fade"]["ret"] < 0

    def test_scores_blocked_trade_at_exact_recommended_strike(self) -> None:
        row = fs.score_asset_day(
            _setup(),
            "CALL",
            blocked_trade={
                "trade_id": "blocked-1",
                "direction": "CALL",
                "strike": 100.0,
                "contracts": 2,
            },
        )

        assert row["blocked_trade"]["trade_id"] == "blocked-1"
        assert row["blocked_trade"]["strike"] == 100
        assert row["blocked_trade"]["contracts"] == 2
        assert row["blocked_trade"]["pnl"] == 68.25
        assert row["blocked_trade"]["total_pnl"] == 136.5


class TestBlockedTrades:
    def test_loads_gate_blocked_recommendation_from_trade_audit(self, tmp_path: Path) -> None:
        day_dir = tmp_path / "2026-09-22"
        day_dir.mkdir()
        (day_dir / "trade-blocked-1.json").write_text(
            json.dumps(
                {
                    "trade_id": "blocked-1",
                    "asset": "QQQ",
                    "direction": "CALL",
                    "contracts": 2,
                    "entry_strike": 100.0,
                    "exit_reason": "premium_gate_blocked",
                }
            )
        )

        assert fs.blocked_trades(tmp_path) == {
            ("2026-09-22", "QQQ"): {
                "trade_id": "blocked-1",
                "direction": "CALL",
                "strike": 100.0,
                "contracts": 2,
            }
        }


class TestUpdate:
    def test_backfills_settled_days_and_never_rescores(self, tmp_path: Path) -> None:
        hist = _history(
            tmp_path, [{"date": "2026-09-22", "asset": "QQQ", "predicted_direction": "UP"}]
        )
        card = tmp_path / "scorecard.json"
        with (
            patch.object(fs, "daily_bars", return_value={}),
            patch.object(fs, "build_day", return_value=_setup()) as build,
        ):
            fs.update(api=object(), history_path=hist, scorecard_path=card, now=AFTER_CLOSE)
            fs.update(api=object(), history_path=hist, scorecard_path=card, now=AFTER_CLOSE)
        assert build.call_count == 1
        rows = json.loads(card.read_text())
        assert [(r["date"], r["asset"]) for r in rows] == [("2026-09-22", "QQQ")]

    def test_attaches_exact_blocked_trade_replay(self, tmp_path: Path) -> None:
        hist = _history(
            tmp_path, [{"date": "2026-09-22", "asset": "QQQ", "predicted_direction": "UP"}]
        )
        day_dir = tmp_path / "2026-09-22"
        day_dir.mkdir()
        (day_dir / "trade-blocked-1.json").write_text(
            json.dumps(
                {
                    "trade_id": "blocked-1",
                    "asset": "QQQ",
                    "direction": "CALL",
                    "contracts": 2,
                    "entry_strike": 100.0,
                    "exit_reason": "premium_gate_blocked",
                }
            )
        )
        card = tmp_path / "scorecard.json"

        with (
            patch.object(fs, "daily_bars", return_value={}),
            patch.object(fs, "build_day", return_value=_setup()),
        ):
            fs.update(
                api=object(),
                history_path=hist,
                scorecard_path=card,
                log_dir=tmp_path,
                now=AFTER_CLOSE,
            )

        [row] = json.loads(card.read_text())
        assert row["blocked_trade"]["trade_id"] == "blocked-1"
        assert row["blocked_trade"]["strike"] == 100
        assert row["blocked_trade"]["total_pnl"] == 136.5

    def test_skips_unsettled_session(self, tmp_path: Path) -> None:
        # Scoring mid-session would cache partial option bars forever.
        hist = _history(
            tmp_path, [{"date": "2026-09-22", "asset": "QQQ", "predicted_direction": "UP"}]
        )
        card = tmp_path / "scorecard.json"
        with patch.object(fs, "build_day") as build:
            fs.update(api=object(), history_path=hist, scorecard_path=card, now=MIDDAY)
        build.assert_not_called()
        assert not card.exists()

    def test_one_failing_day_does_not_block_others(self, tmp_path: Path) -> None:
        hist = _history(
            tmp_path,
            [
                {"date": "2026-09-22", "asset": "QQQ", "predicted_direction": "UP"},
                {"date": "2026-09-22", "asset": "SPY", "predicted_direction": "UP"},
            ],
        )
        card = tmp_path / "scorecard.json"

        def build(api, asset, day, daily):
            if asset == "QQQ":
                raise RuntimeError("alpaca down")
            return _setup("SPY")

        with (
            patch.object(fs, "daily_bars", return_value={}),
            patch.object(fs, "build_day", side_effect=build),
        ):
            fs.update(api=object(), history_path=hist, scorecard_path=card, now=AFTER_CLOSE)
        assert [r["asset"] for r in json.loads(card.read_text())] == ["SPY"]


def _row(sys_ret: float, base_ret: float) -> dict:
    trade = {"ret": sys_ret, "pnl": sys_ret}
    base = {"ret": base_ret, "pnl": base_ret}
    return {"system": trade, "baselines": dict.fromkeys(fs.COMPARE, base)}


class TestSummary:
    def test_consistent_edge_is_ahead(self) -> None:
        rows = [_row(10 + (i % 3), -5 + (i % 2)) for i in range(40)]
        s = fs.summarize(rows)
        assert all(b["verdict"] == "ahead" for b in s["baselines"].values())
        assert "beats every baseline" in fs.headline(s)

    def test_consistent_deficit_is_behind(self) -> None:
        rows = [_row(-20 + (i % 3), 5) for i in range(40)]
        assert "BEHIND" in fs.headline(fs.summarize(rows))

    def test_noise_is_not_significant(self) -> None:
        rows = [_row(100 if i % 2 else -100, 100 if i % 3 else -100) for i in range(20)]
        s = fs.summarize(rows)
        assert {b["verdict"] for b in s["baselines"].values()} == {"not significant"}
        assert "Not yet significant" in fs.headline(s)

    def test_empty(self) -> None:
        assert fs.headline(fs.summarize([])) == "No forecasts scored yet."


class TestRender:
    def test_renders_from_file(self, tmp_path: Path) -> None:
        card = tmp_path / "scorecard.json"
        rows = [
            {"date": f"2026-09-{d:02d}", "asset": "QQQ", **_row(10 * (-1) ** d, -5 * (-1) ** d)}
            for d in range(1, 7)
        ]
        card.write_text(json.dumps(rows))
        html = fs.render_email_html(card)
        assert "Forward Scorecard" in html
        assert "P&amp;L" in html
        assert "vs gap_follow" in fs.render_email_text(card)

    def test_never_raises_on_bad_file(self, tmp_path: Path) -> None:
        card = tmp_path / "scorecard.json"
        card.write_text("{not json")
        assert fs.render_email_html(card) == ""
        assert fs.render_email_text(card) == ""
        card.write_text(json.dumps([{"date": "x", "asset": "Q", "system": {"bogus": 1}}]))
        assert fs.render_email_html(card) == ""
        assert fs.render_email_text(tmp_path / "missing.json") == ""
