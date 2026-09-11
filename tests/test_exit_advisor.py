"""Unit tests for the LLM-driven profit-exit advisor."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.execution import exit_advisor
from src.execution.exit_advisor import (
    AdvisorState,
    ExitDecision,
    advisor_trail_triggered,
    build_context,
    build_prompt,
    compute_pnl_pct,
    decide_trigger,
    minutes_to_deadline,
    parse_advisor_response,
    resolve_model,
    trailing_fallback_triggered,
    update_peak,
)
from src.execution.intraday_monitor import OpenTrade, extract_open_trade, monitor_trade
from src.execution.models import ExecutionConfig, ExitAdvisorConfig, ExitConfig, TrailingConfig
from src.timezone import ET_TZ


def _now(hour: int, minute: int, day: int = 3) -> datetime:
    return datetime(2026, 9, day, hour, minute, tzinfo=ET_TZ)


# ─────────────────────────────── cadence ───────────────────────────────


class TestDecideTrigger:
    def test_profit_step_first_crossing(self) -> None:
        cfg = ExitAdvisorConfig()
        state = AdvisorState()
        update_peak(state, 55.0)
        trig = decide_trigger(55.0, state, _now(10, 0), 120, cfg)
        assert trig == "profit_step"
        assert state.last_pnl_bucket == 1

    def test_profit_step_does_not_refire_within_same_bucket(self) -> None:
        cfg = ExitAdvisorConfig()
        state = AdvisorState(last_pnl_bucket=1, peak_pnl_pct=55.0, last_periodic_ts=_now(10, 0))
        trig = decide_trigger(58.0, state, _now(10, 0), 120, cfg)
        assert trig is None

    def test_profit_step_fires_again_at_next_step(self) -> None:
        cfg = ExitAdvisorConfig()
        state = AdvisorState(last_pnl_bucket=1, peak_pnl_pct=55.0)
        update_peak(state, 101.0)
        trig = decide_trigger(101.0, state, _now(10, 0), 120, cfg)
        assert trig == "profit_step"
        assert state.last_pnl_bucket == 2

    def test_giveback_fires_once_in_profit(self) -> None:
        cfg = ExitAdvisorConfig()
        state = AdvisorState(peak_pnl_pct=80.0, last_pnl_bucket=1)
        # 80 peak, now 50 => giveback of 30 >= 25 threshold
        trig = decide_trigger(50.0, state, _now(10, 0), 120, cfg)
        assert trig == "giveback"
        assert state.last_giveback_peak == 80.0

    def test_giveback_does_not_refire_at_same_drop_level(self) -> None:
        cfg = ExitAdvisorConfig()
        state = AdvisorState(
            peak_pnl_pct=80.0,
            last_pnl_bucket=1,
            last_giveback_peak=80.0,
            giveback_bucket=1,  # already consulted for peak-25
            last_periodic_ts=_now(10, 0),
        )
        trig = decide_trigger(50.0, state, _now(10, 0), 120, cfg)
        assert trig is None

    def test_giveback_refires_at_next_step_down_same_peak(self) -> None:
        """Fix (2026-09-11 review): a winner sliding from +80 through +50
        (peak-25) then +25 (peak-50) must be consulted again at the
        second step, not go quiet after the first give-back consult."""
        cfg = ExitAdvisorConfig()
        state = AdvisorState(
            peak_pnl_pct=80.0,
            last_pnl_bucket=1,
            last_giveback_peak=80.0,
            giveback_bucket=1,  # already consulted at peak-25 (pnl 50)
            last_periodic_ts=_now(10, 0),
        )
        # Continues sliding: 80 - 25 = 55 -- drop 25 (bucket 1, already
        # consulted). Now drops further to 25 -- drop 55 (bucket 2, new).
        trig = decide_trigger(25.0, state, _now(10, 5), 120, cfg)
        assert trig == "giveback"
        assert state.giveback_bucket == 2

    def test_giveback_refires_repeatedly_down_the_ladder(self) -> None:
        cfg = ExitAdvisorConfig(giveback_from_peak_pct=25)
        # last_pnl_bucket=2 -- the +100% profit_step milestone has
        # already been consulted for, so it doesn't fire again and
        # confound the giveback-ladder assertions below.
        state = AdvisorState(peak_pnl_pct=100.0, last_pnl_bucket=2)
        update_peak(state, 100.0)
        # Establish the ladder anchor.
        trig1 = decide_trigger(74.0, state, _now(10, 0), 120, cfg)  # drop 26 -> bucket 1
        assert trig1 == "giveback"
        assert state.giveback_bucket == 1
        trig2 = decide_trigger(74.0, state, _now(10, 3), 120, cfg)  # unchanged -> no refire
        assert trig2 is None
        trig3 = decide_trigger(49.0, state, _now(10, 6), 120, cfg)  # drop 51 -> bucket 2
        assert trig3 == "giveback"
        assert state.giveback_bucket == 2
        trig4 = decide_trigger(24.0, state, _now(10, 9), 120, cfg)  # drop 76 -> bucket 3
        assert trig4 == "giveback"
        assert state.giveback_bucket == 3

    def test_giveback_refires_at_new_higher_peak(self) -> None:
        cfg = ExitAdvisorConfig()
        state = AdvisorState(
            peak_pnl_pct=80.0, last_pnl_bucket=1, last_giveback_peak=80.0, giveback_bucket=1
        )
        update_peak(state, 120.0)
        trig = decide_trigger(90.0, state, _now(10, 0), 120, cfg)
        assert trig == "giveback"
        assert state.last_giveback_peak == 120.0
        assert state.giveback_bucket == 1  # ladder reset against the new peak, then re-armed

    def test_giveback_requires_having_been_in_profit(self) -> None:
        cfg = ExitAdvisorConfig()
        state = AdvisorState(peak_pnl_pct=0.0)
        trig = decide_trigger(-40.0, state, _now(10, 0), 120, cfg)
        assert trig is None

    def test_final_window_fires_once(self) -> None:
        cfg = ExitAdvisorConfig(final_window_min=30)
        state = AdvisorState()
        trig = decide_trigger(5.0, state, _now(10, 0), 25.0, cfg)
        assert trig == "final_window"
        assert state.final_window_consulted is True
        # A second poll a minute later, still inside the window, must not refire.
        trig2 = decide_trigger(6.0, state, _now(10, 1), 24.0, cfg)
        assert trig2 is None

    def test_periodic_fires_first_time_in_profit(self) -> None:
        cfg = ExitAdvisorConfig(periodic_interval_min=15)
        state = AdvisorState()
        trig = decide_trigger(10.0, state, _now(10, 0), 120, cfg)
        assert trig == "periodic"
        assert state.last_periodic_ts == _now(10, 0)

    def test_periodic_waits_for_interval(self) -> None:
        cfg = ExitAdvisorConfig(periodic_interval_min=15)
        state = AdvisorState(last_periodic_ts=_now(10, 0))
        trig = decide_trigger(10.0, state, _now(10, 10), 120, cfg)
        assert trig is None
        trig2 = decide_trigger(10.0, state, _now(10, 16), 120, cfg)
        assert trig2 == "periodic"

    def test_periodic_never_fires_out_of_profit(self) -> None:
        cfg = ExitAdvisorConfig(periodic_interval_min=15)
        state = AdvisorState()
        trig = decide_trigger(-5.0, state, _now(10, 0), 120, cfg)
        assert trig is None

    def test_priority_profit_step_over_periodic(self) -> None:
        cfg = ExitAdvisorConfig()
        state = AdvisorState()
        update_peak(state, 60.0)
        trig = decide_trigger(60.0, state, _now(10, 0), 120, cfg)
        assert trig == "profit_step"

    def test_no_trigger_when_flat_and_no_deadline_pressure(self) -> None:
        cfg = ExitAdvisorConfig()
        state = AdvisorState()
        trig = decide_trigger(0.0, state, _now(10, 0), 120, cfg)
        assert trig is None


class TestZeroCrossTrigger:
    def test_fires_once_after_round_trip_from_real_peak(self) -> None:
        cfg = ExitAdvisorConfig(profit_step_pct=50)
        state = AdvisorState(
            peak_pnl_pct=80.0, last_pnl_bucket=1, last_giveback_peak=80.0, giveback_bucket=3
        )
        trig = decide_trigger(-10.0, state, _now(10, 0), 120, cfg)
        assert trig == "zero_cross"
        assert state.zero_cross_consulted is True

    def test_does_not_refire(self) -> None:
        cfg = ExitAdvisorConfig(profit_step_pct=50)
        state = AdvisorState(
            peak_pnl_pct=80.0,
            last_giveback_peak=80.0,
            giveback_bucket=4,  # drop of 100 (80 - (-20)) already consulted via the ladder
            zero_cross_consulted=True,
            last_periodic_ts=_now(10, 0),
        )
        trig = decide_trigger(-20.0, state, _now(10, 3), 120, cfg)
        assert trig is None

    def test_does_not_fire_below_profit_step_threshold(self) -> None:
        """Peak never reached a real milestone (profit_step_pct) -- no round-trip event."""
        cfg = ExitAdvisorConfig(
            profit_step_pct=50, giveback_from_peak_pct=1000
        )  # isolate from giveback
        state = AdvisorState(peak_pnl_pct=20.0)
        trig = decide_trigger(-5.0, state, _now(10, 0), 120, cfg)
        assert trig is None

    def test_does_not_fire_while_still_positive(self) -> None:
        cfg = ExitAdvisorConfig(profit_step_pct=50)
        state = AdvisorState(
            peak_pnl_pct=80.0,
            last_giveback_peak=80.0,
            giveback_bucket=3,
            last_periodic_ts=_now(10, 0),
        )
        trig = decide_trigger(5.0, state, _now(10, 0), 120, cfg)
        assert trig is None


class TestAdvisorTrailTriggered:
    def test_none_when_advisor_never_tightened(self) -> None:
        exit_cfg = ExitConfig(
            trailing=TrailingConfig(enabled=True, activate_after_pct=30, trail_pct=15)
        )
        state = AdvisorState(peak_pnl_pct=50.0)  # no trail_stop_pct set
        assert advisor_trail_triggered(10.0, state, exit_cfg) is False

    def test_triggers_below_advisor_tightened_level(self) -> None:
        exit_cfg = ExitConfig(
            trailing=TrailingConfig(enabled=True, activate_after_pct=30, trail_pct=15)
        )
        state = AdvisorState(peak_pnl_pct=50.0, trail_stop_pct=5.0)
        # trail level = 50 - 5 = 45; pnl 44 <= 45 -> triggered
        assert advisor_trail_triggered(44.0, state, exit_cfg) is True

    def test_ignores_activate_after_pct_gate(self) -> None:
        """The advisor's own tightening IS the activation signal -- unlike
        the deterministic fallback, this does not wait for
        activate_after_pct to be crossed."""
        exit_cfg = ExitConfig(
            trailing=TrailingConfig(enabled=True, activate_after_pct=90, trail_pct=15)
        )
        state = AdvisorState(
            peak_pnl_pct=10.0, trail_stop_pct=2.0
        )  # peak well below activate_after_pct
        # trail level = 10 - 2 = 8; pnl 7 <= 8 -> triggered even though peak (10) < activate_after_pct (90)
        assert advisor_trail_triggered(7.0, state, exit_cfg) is True

    def test_disabled_kill_switch_still_respected(self) -> None:
        exit_cfg = ExitConfig(
            trailing=TrailingConfig(enabled=False, activate_after_pct=30, trail_pct=15)
        )
        state = AdvisorState(peak_pnl_pct=50.0, trail_stop_pct=5.0)
        assert advisor_trail_triggered(0.0, state, exit_cfg) is False

    def test_not_triggered_above_trail_level(self) -> None:
        exit_cfg = ExitConfig(
            trailing=TrailingConfig(enabled=True, activate_after_pct=30, trail_pct=15)
        )
        state = AdvisorState(peak_pnl_pct=50.0, trail_stop_pct=5.0)
        assert advisor_trail_triggered(46.0, state, exit_cfg) is False


class TestMinutesToDeadline:
    def test_before_deadline(self) -> None:
        mins = minutes_to_deadline(_now(15, 0), "15:25")
        assert mins == pytest.approx(25.0)

    def test_after_deadline_is_negative(self) -> None:
        mins = minutes_to_deadline(_now(15, 30), "15:25")
        assert mins < 0


# ────────────────────────────── state I/O ──────────────────────────────


class TestAdvisorState:
    def test_round_trip(self) -> None:
        state = AdvisorState(
            peak_pnl_pct=75.5,
            calls_made=3,
            last_pnl_bucket=1,
            last_giveback_peak=75.5,
            last_periodic_ts=_now(10, 0),
            final_window_consulted=True,
            trail_stop_pct=10.0,
        )
        restored = AdvisorState.from_dict(state.to_dict())
        assert restored == state or restored.to_dict() == state.to_dict()

    def test_from_dict_none_returns_defaults(self) -> None:
        state = AdvisorState.from_dict(None)
        assert state.peak_pnl_pct == 0.0
        assert state.calls_made == 0

    def test_from_dict_tolerates_garbage(self) -> None:
        state = AdvisorState.from_dict(
            {"peak_pnl_pct": "bad", "calls_made": None, "last_periodic_ts": "not-a-date"}
        )
        assert state.peak_pnl_pct == 0.0
        assert state.calls_made == 0
        assert state.last_periodic_ts is None

    def test_apply_trail_tightening_tightens(self) -> None:
        state = AdvisorState()
        state.apply_trail_tightening(10.0, default_trail_pct=15.0)
        assert state.trail_stop_pct == 10.0

    def test_apply_trail_tightening_never_loosens(self) -> None:
        state = AdvisorState(trail_stop_pct=8.0)
        state.apply_trail_tightening(12.0, default_trail_pct=15.0)
        assert state.trail_stop_pct == 8.0

    def test_apply_trail_tightening_ignores_none_and_nonpositive(self) -> None:
        state = AdvisorState()
        state.apply_trail_tightening(None, default_trail_pct=15.0)
        assert state.trail_stop_pct is None
        state.apply_trail_tightening(-5.0, default_trail_pct=15.0)
        assert state.trail_stop_pct is None

    def test_effective_trail_pct_defaults(self) -> None:
        state = AdvisorState()
        assert state.effective_trail_pct(15.0) == 15.0
        state.trail_stop_pct = 9.0
        assert state.effective_trail_pct(15.0) == 9.0


# ──────────────────────────── trailing fallback ────────────────────────────


class TestTrailingFallback:
    def test_not_triggered_before_activation(self) -> None:
        exit_cfg = ExitConfig(
            trailing=TrailingConfig(enabled=True, activate_after_pct=30, trail_pct=15)
        )
        state = AdvisorState(peak_pnl_pct=20.0)
        assert trailing_fallback_triggered(15.0, state, exit_cfg) is False

    def test_triggered_after_giveback_past_trail(self) -> None:
        exit_cfg = ExitConfig(
            trailing=TrailingConfig(enabled=True, activate_after_pct=30, trail_pct=15)
        )
        state = AdvisorState(peak_pnl_pct=50.0)
        # trail level = 50 - 15 = 35; pnl 34 <= 35 => triggered
        assert trailing_fallback_triggered(34.0, state, exit_cfg) is True

    def test_uses_tightened_trail(self) -> None:
        exit_cfg = ExitConfig(
            trailing=TrailingConfig(enabled=True, activate_after_pct=30, trail_pct=15)
        )
        state = AdvisorState(peak_pnl_pct=50.0, trail_stop_pct=5.0)
        # trail level = 50 - 5 = 45; pnl 44 <= 45 => triggered (tighter than default 35)
        assert trailing_fallback_triggered(44.0, state, exit_cfg) is True
        # With the default 15 this would NOT have triggered at 44.
        state2 = AdvisorState(peak_pnl_pct=50.0)
        assert trailing_fallback_triggered(44.0, state2, exit_cfg) is False

    def test_disabled_never_triggers(self) -> None:
        exit_cfg = ExitConfig(
            trailing=TrailingConfig(enabled=False, activate_after_pct=30, trail_pct=15)
        )
        state = AdvisorState(peak_pnl_pct=90.0)
        assert trailing_fallback_triggered(0.0, state, exit_cfg) is False


# ───────────────────────────── response parsing ─────────────────────────────


class TestParseAdvisorResponse:
    def test_valid_exit(self) -> None:
        parsed = parse_advisor_response(
            '{"action": "EXIT", "confidence": 0.8, "reason": "reversal", "trail_stop_pct": 5.0}'
        )
        assert parsed == {
            "action": "EXIT",
            "confidence": 0.8,
            "reason": "reversal",
            "trail_stop_pct": 5.0,
        }

    def test_valid_hold_no_trail(self) -> None:
        parsed = parse_advisor_response(
            '{"action": "hold", "confidence": 0.6, "reason": "trend intact"}'
        )
        assert parsed["action"] == "HOLD"
        assert parsed["trail_stop_pct"] is None

    def test_fenced_json(self) -> None:
        text = '```json\n{"action": "EXIT", "confidence": 0.9, "reason": "done"}\n```'
        parsed = parse_advisor_response(text)
        assert parsed is not None
        assert parsed["action"] == "EXIT"

    def test_prose_wrapped_json(self) -> None:
        text = 'Here is my call:\n{"action": "HOLD", "confidence": 0.5, "reason": "ok"}\nThanks.'
        parsed = parse_advisor_response(text)
        assert parsed is not None
        assert parsed["action"] == "HOLD"

    def test_missing_action_is_none(self) -> None:
        assert parse_advisor_response('{"confidence": 0.5, "reason": "x"}') is None

    def test_invalid_action_is_none(self) -> None:
        assert parse_advisor_response('{"action": "MAYBE", "confidence": 0.5}') is None

    def test_empty_text_is_none(self) -> None:
        assert parse_advisor_response("") is None
        assert parse_advisor_response(None) is None

    def test_garbage_text_is_none(self) -> None:
        assert parse_advisor_response("not json at all") is None

    def test_confidence_clamped(self) -> None:
        parsed = parse_advisor_response('{"action": "HOLD", "confidence": 5.0, "reason": "x"}')
        assert parsed["confidence"] == 1.0
        parsed2 = parse_advisor_response('{"action": "HOLD", "confidence": -1.0, "reason": "x"}')
        assert parsed2["confidence"] == 0.0

    def test_negative_trail_stop_pct_dropped(self) -> None:
        parsed = parse_advisor_response(
            '{"action": "HOLD", "confidence": 0.5, "reason": "x", "trail_stop_pct": -3}'
        )
        assert parsed["trail_stop_pct"] is None

    def test_bad_confidence_type_defaults_zero(self) -> None:
        parsed = parse_advisor_response('{"action": "HOLD", "confidence": "high", "reason": "x"}')
        assert parsed["confidence"] == 0.0


# ────────────────────────────── misc helpers ──────────────────────────────


class TestMisc:
    def test_compute_pnl_pct(self) -> None:
        assert compute_pnl_pct(1.0, 1.5) == 50.0
        assert compute_pnl_pct(1.0, 0.5) == -50.0

    def test_compute_pnl_pct_zero_entry(self) -> None:
        assert compute_pnl_pct(None, 1.5) == 0.0
        assert compute_pnl_pct(0.0, 1.5) == 0.0

    def test_resolve_model_uses_override(self) -> None:
        cfg = ExitAdvisorConfig(model="opencode/some-other-model")
        assert (
            resolve_model(cfg, "opencode/muse-spark-1.3-contributor-free")
            == "opencode/some-other-model"
        )

    def test_resolve_model_falls_back_to_primary(self) -> None:
        cfg = ExitAdvisorConfig(model=None)
        assert resolve_model(cfg, "opencode/muse-spark-1.3-contributor-free") == (
            "opencode/muse-spark-1.3-contributor-free"
        )

    def test_build_context_distance_put(self) -> None:
        ctx = build_context(
            trigger="profit_step",
            asset="QQQ",
            direction="PUT",
            prediction={},
            entry_time="t",
            entry_price=1.0,
            current_mark=1.5,
            current_bid=1.4,
            current_ask=1.6,
            underlying_spot=710.0,
            strike=715.0,
            pnl_pct=50.0,
            peak_pnl_pct=50.0,
            market_context={"underlying_path": {}, "co_movement": {}},
            minutes_to_deadline=60.0,
            sl_level=0.5,
            time_deadline_est="15:25",
            calls_made=1,
            max_calls_per_trade=12,
        )
        # PUT: (spot - strike)/spot -- underlying below strike is favorable => positive
        assert ctx["strike_distance"]["distance_pct"] == pytest.approx(
            round((710.0 - 715.0) / 710.0 * 100.0, 3), abs=1e-6
        )
        assert ctx["pnl"]["now_pct"] == 50.0

    def test_build_prompt_contains_context_json(self) -> None:
        prompt = build_prompt({"foo": "bar"})
        assert '"foo": "bar"' in prompt
        assert "HOLD" in prompt or "JSON" in prompt


# ──────────────────────────── process() orchestration ────────────────────────────


def _trade_data(
    *,
    entry_price: float = 1.0,
    sl_level: float = 0.5,
    tp_order_id: str | None = "tp-1",
    advisor_state: dict | None = None,
    predicted_move_pct: float = -0.9,
) -> dict:
    return {
        "trade_id": "adv-trade-1",
        "correlation_id": "corr-1",
        "asset": "QQQ",
        "direction": "PUT",
        "contracts": 1,
        "entry_strike": 715.0,
        "exit_reason": "pending",
        "exit_price": None,
        "final_pnl": None,
        "final_pnl_pct": None,
        "started_at": "2026-09-03T13:30:01+00:00",
        "ended_at": None,
        "advisor_state": advisor_state,
        "entries": [
            {
                "event_type": "entry_submitted",
                "occ_symbol": "QQQ260903P00715000",
                "contracts": 1,
                "order_type": "limit",
            },
            {
                "event_type": "entry_filled",
                "order_id": "buy-order",
                "filled_qty": 1,
                "avg_price": entry_price,
            },
            {
                "event_type": "exits_placed",
                "tp_order_id": tp_order_id,
                "tp_level": 4.0,
                "sl_level": sl_level,
            },
        ],
        "recommendation": {
            "strategy_label": "llm_trade",
            "asset": "QQQ",
            "direction": "PUT",
            "confidence": 0.6,
            "predicted_move_pct": predicted_move_pct,
            "rationale": {"llm_rationale": "test rationale"},
        },
    }


def _write_trade(tmp_path: Path, data: dict) -> OpenTrade:
    path = tmp_path / "trade-adv-trade-1.json"
    path.write_text(json.dumps(data))
    trade = extract_open_trade(data, path)
    assert trade is not None
    return trade


async def _no_market_context(asset: str, now: datetime) -> dict:
    return {"underlying_path": {"available": False}, "co_movement": {}}


class TestProcessNoTrigger:
    @pytest.mark.asyncio
    async def test_no_trigger_persists_peak_and_returns_none(self, tmp_path: Path) -> None:
        trade = _write_trade(tmp_path, _trade_data())
        exec_cfg = ExecutionConfig(exit_advisor=ExitAdvisorConfig(enabled=True))
        decision = await exit_advisor.process(
            trade,
            mark=1.0,  # pnl 0% -- no trigger conditions met
            quote={"bid": 0.95, "ask": 1.05},
            underlying_spot=710.0,
            exec_config=exec_cfg,
            llm_primary_model="opencode/muse-spark-1.3-contributor-free",
            now=_now(10, 0),
            market_context_fn=_no_market_context,
        )
        assert decision is None
        on_disk = json.loads(trade.path.read_text())
        assert on_disk["advisor_state"]["peak_pnl_pct"] == 0.0


class TestProcessAdvisorTrail:
    """Fix (2026-09-11 review): an advisor-tightened trail must be
    enforced on EVERY poll, not only as a consult-failure fallback."""

    @pytest.mark.asyncio
    async def test_tightened_trail_exits_without_calling_model(self, tmp_path: Path) -> None:
        # Simulate a prior consult that tightened the trail to 5% and
        # left the peak at 80%. No cadence trigger is due at this poll
        # (last_periodic_ts just set, no new profit_step/giveback level
        # crossed) -- yet a 44% pnl (peak 80 - trail 5 = 75 level) must
        # still force an exit, and must NOT call the model to do it.
        state = AdvisorState(
            peak_pnl_pct=80.0,
            calls_made=2,
            last_pnl_bucket=1,
            last_giveback_peak=80.0,
            giveback_bucket=0,
            trail_stop_pct=5.0,
            last_periodic_ts=_now(10, 0),
        ).to_dict()
        trade = _write_trade(tmp_path, _trade_data(entry_price=1.0, advisor_state=state))

        called = {"n": 0}

        def invoke_fn(model, prompt, timeout):
            called["n"] += 1
            return '{"action": "HOLD", "confidence": 0.5, "reason": "x"}'

        exec_cfg = ExecutionConfig(exit_advisor=ExitAdvisorConfig(enabled=True))
        decision = await exit_advisor.process(
            trade,
            mark=1.44,  # pnl 44% <= 80 - 5 = 75 trail level
            quote={"bid": 1.4, "ask": 1.5},
            underlying_spot=710.0,
            exec_config=exec_cfg,
            llm_primary_model="opencode/muse-spark-1.3-contributor-free",
            now=_now(10, 1),
            invoke_fn=invoke_fn,
            market_context_fn=_no_market_context,
        )
        assert called["n"] == 0  # never consulted -- deterministic trail did the work
        assert decision is not None
        assert decision.should_exit is True
        assert decision.exit_reason == "trailing_stop"

    @pytest.mark.asyncio
    async def test_untightened_default_trail_does_not_exit_mid_session(
        self, tmp_path: Path
    ) -> None:
        """Without an advisor tightening, the configured DEFAULT trail is
        fallback-only and must NOT fire on a plain healthy poll."""
        state = AdvisorState(
            peak_pnl_pct=80.0,
            calls_made=2,
            last_pnl_bucket=1,
            last_giveback_peak=80.0,
            giveback_bucket=0,
            trail_stop_pct=None,
            last_periodic_ts=_now(10, 0),
        ).to_dict()
        trade = _write_trade(tmp_path, _trade_data(entry_price=1.0, advisor_state=state))

        called = {"n": 0}

        def invoke_fn(model, prompt, timeout):
            called["n"] += 1
            return '{"action": "HOLD", "confidence": 0.5, "reason": "x"}'

        exec_cfg = ExecutionConfig(
            exit_advisor=ExitAdvisorConfig(enabled=True),
            exit_strategy=ExitConfig(
                trailing=TrailingConfig(enabled=True, activate_after_pct=30, trail_pct=15)
            ),
        )
        # pnl 70% is only 10 points below peak (80) -- inside the default
        # 15% trail, and no cadence trigger is due -- so this poll must
        # be a pure no-op (no exit, no model call).
        decision = await exit_advisor.process(
            trade,
            mark=1.70,
            quote={"bid": 1.65, "ask": 1.75},
            underlying_spot=710.0,
            exec_config=exec_cfg,
            llm_primary_model="opencode/muse-spark-1.3-contributor-free",
            now=_now(10, 1),
            invoke_fn=invoke_fn,
            market_context_fn=_no_market_context,
        )
        assert called["n"] == 0
        assert decision is None


class TestProcessExit:
    @pytest.mark.asyncio
    async def test_advisor_exit_action(self, tmp_path: Path) -> None:
        trade = _write_trade(tmp_path, _trade_data())

        def sync_invoke(model, prompt, timeout):
            return '{"action": "EXIT", "confidence": 0.9, "reason": "thesis broken"}'

        exec_cfg = ExecutionConfig(exit_advisor=ExitAdvisorConfig(enabled=True, profit_step_pct=50))
        decision = await exit_advisor.process(
            trade,
            mark=1.5,  # +50% pnl -> profit_step trigger
            quote={"bid": 1.4, "ask": 1.6},
            underlying_spot=710.0,
            exec_config=exec_cfg,
            llm_primary_model="opencode/muse-spark-1.3-contributor-free",
            now=_now(10, 0),
            invoke_fn=sync_invoke,
            market_context_fn=_no_market_context,
        )
        assert decision is not None
        assert decision.should_exit is True
        assert decision.exit_reason == "advisor_exit"
        assert decision.consult is not None
        assert decision.consult.action == "EXIT"
        # process() must NOT write to disk itself on exit -- caller does via write_result.
        # advisor_state should be updated in the in-memory trade.data though.
        assert trade.data["advisor_state"]["calls_made"] == 1
        consultations = [
            e for e in trade.data["entries"] if e.get("event_type") == "advisor_consultation"
        ]
        assert len(consultations) == 1
        assert consultations[0]["trigger"] == "profit_step"
        assert consultations[0]["action_taken"] == "exit"
        assert consultations[0]["parsed"]["action"] == "EXIT"

    @pytest.mark.asyncio
    async def test_advisor_hold_persists_and_tightens_trail(self, tmp_path: Path) -> None:
        trade = _write_trade(tmp_path, _trade_data())

        def sync_invoke(model, prompt, timeout):
            return '{"action": "HOLD", "confidence": 0.7, "reason": "still trending", "trail_stop_pct": 5.0}'

        exec_cfg = ExecutionConfig(
            exit_advisor=ExitAdvisorConfig(enabled=True, profit_step_pct=50),
            exit_strategy=ExitConfig(trailing=TrailingConfig(trail_pct=15)),
        )
        decision = await exit_advisor.process(
            trade,
            mark=1.5,
            quote={"bid": 1.4, "ask": 1.6},
            underlying_spot=710.0,
            exec_config=exec_cfg,
            llm_primary_model="opencode/muse-spark-1.3-contributor-free",
            now=_now(10, 0),
            invoke_fn=sync_invoke,
            market_context_fn=_no_market_context,
        )
        assert decision is None
        on_disk = json.loads(trade.path.read_text())
        assert on_disk["advisor_state"]["trail_stop_pct"] == 5.0
        assert on_disk["advisor_state"]["calls_made"] == 1


class TestProcessFallback:
    @pytest.mark.asyncio
    async def test_failed_call_logs_and_holds_when_no_trailing_trigger(
        self, tmp_path: Path
    ) -> None:
        trade = _write_trade(tmp_path, _trade_data())

        def failing_invoke(model, prompt, timeout):
            return None

        exec_cfg = ExecutionConfig(exit_advisor=ExitAdvisorConfig(enabled=True, profit_step_pct=50))
        decision = await exit_advisor.process(
            trade,
            mark=1.5,
            quote={"bid": 1.4, "ask": 1.6},
            underlying_spot=710.0,
            exec_config=exec_cfg,
            llm_primary_model="opencode/muse-spark-1.3-contributor-free",
            now=_now(10, 0),
            invoke_fn=failing_invoke,
            market_context_fn=_no_market_context,
        )
        assert decision is None
        on_disk = json.loads(trade.path.read_text())
        consultations = [
            e for e in on_disk["entries"] if e.get("event_type") == "advisor_consultation"
        ]
        assert consultations[0]["action_taken"] == "hold_on_failure"
        assert consultations[0]["ok"] is False

    @pytest.mark.asyncio
    async def test_failed_call_falls_back_to_trailing_stop_when_active(
        self, tmp_path: Path
    ) -> None:
        # Peak already at 80%, trailing activates at 30% with trail 15% ->
        # trail level 65%; current pnl 60% is below it, so a failed
        # consult must fall back to closing via the trailing stop.
        state = AdvisorState(peak_pnl_pct=80.0, calls_made=1, last_pnl_bucket=1).to_dict()
        trade = _write_trade(tmp_path, _trade_data(entry_price=1.0, advisor_state=state))

        def failing_invoke(model, prompt, timeout):
            raise RuntimeError("boom")

        exec_cfg = ExecutionConfig(
            exit_advisor=ExitAdvisorConfig(enabled=True, giveback_from_peak_pct=1000),
            exit_strategy=ExitConfig(
                trailing=TrailingConfig(enabled=True, activate_after_pct=30, trail_pct=15)
            ),
        )
        decision = await exit_advisor.process(
            trade,
            mark=1.6,  # pnl = 60%
            quote={"bid": 1.5, "ask": 1.7},
            underlying_spot=710.0,
            exec_config=exec_cfg,
            llm_primary_model="opencode/muse-spark-1.3-contributor-free",
            now=_now(11, 30),
            invoke_fn=failing_invoke,
            market_context_fn=_no_market_context,
        )
        assert decision is not None
        assert decision.should_exit is True
        assert decision.exit_reason == "trailing_stop"

    @pytest.mark.asyncio
    async def test_budget_exhausted_skips_model_and_checks_trailing(self, tmp_path: Path) -> None:
        state = AdvisorState(peak_pnl_pct=80.0, calls_made=12, last_pnl_bucket=1).to_dict()
        trade = _write_trade(tmp_path, _trade_data(advisor_state=state))

        called = {"n": 0}

        def invoke_fn(model, prompt, timeout):
            called["n"] += 1
            return '{"action": "HOLD", "confidence": 0.5, "reason": "x"}'

        exec_cfg = ExecutionConfig(
            exit_advisor=ExitAdvisorConfig(enabled=True, max_calls_per_trade=12),
            exit_strategy=ExitConfig(
                trailing=TrailingConfig(enabled=True, activate_after_pct=30, trail_pct=15)
            ),
        )
        decision = await exit_advisor.process(
            trade,
            mark=1.6,  # pnl 60%, below trail level (80-15=65) -> should trigger
            quote={"bid": 1.5, "ask": 1.7},
            underlying_spot=710.0,
            exec_config=exec_cfg,
            llm_primary_model="opencode/muse-spark-1.3-contributor-free",
            now=_now(10, 0),
            invoke_fn=invoke_fn,
            market_context_fn=_no_market_context,
        )
        assert called["n"] == 0  # model never consulted -- budget exhausted
        assert decision is not None
        assert decision.exit_reason == "trailing_stop"

    @pytest.mark.asyncio
    async def test_unparseable_response_treated_as_failure(self, tmp_path: Path) -> None:
        trade = _write_trade(tmp_path, _trade_data())

        def invoke_fn(model, prompt, timeout):
            return "I cannot decide right now."

        exec_cfg = ExecutionConfig(exit_advisor=ExitAdvisorConfig(enabled=True, profit_step_pct=50))
        decision = await exit_advisor.process(
            trade,
            mark=1.5,
            quote={"bid": 1.4, "ask": 1.6},
            underlying_spot=710.0,
            exec_config=exec_cfg,
            llm_primary_model="opencode/muse-spark-1.3-contributor-free",
            now=_now(10, 0),
            invoke_fn=invoke_fn,
            market_context_fn=_no_market_context,
        )
        assert decision is None
        on_disk = json.loads(trade.path.read_text())
        consultations = [
            e for e in on_disk["entries"] if e.get("event_type") == "advisor_consultation"
        ]
        assert consultations[0]["error"] == "unparseable"


# ─────────────────────── monitor_trade rails-first ordering ───────────────────────


class TestMonitorTradeRailsFirst:
    @pytest.mark.asyncio
    async def test_sl_triggers_before_advisor_is_ever_consulted(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        data = _trade_data(entry_price=1.0, sl_level=0.5)
        path = tmp_path / "trade-adv-trade-1.json"
        path.write_text(json.dumps(data))
        trade = extract_open_trade(data, path)
        assert trade is not None

        client = AsyncMock()
        client.get_option_quote = AsyncMock(
            return_value={"bid": 0.4, "ask": 0.5}
        )  # mark 0.45 <= sl 0.5
        client.cancel_order = AsyncMock()
        client.submit_order = AsyncMock(return_value=type("R", (), {"order_id": "sell-1"})())
        client.get_order = AsyncMock(
            return_value=type(
                "R", (), {"status": "filled", "filled_avg_price": "0.45", "order_id": "sell-1"}
            )()
        )
        client.get_underlying_quote = AsyncMock(return_value={"last": 710.0})

        advisor_called = {"n": 0}

        async def fake_process(*args, **kwargs):
            advisor_called["n"] += 1
            return None

        monkeypatch.setattr(exit_advisor, "process", fake_process)

        exec_cfg = ExecutionConfig(exit_advisor=ExitAdvisorConfig(enabled=True))
        outcome = await monitor_trade(
            trade, client, now=_now(10, 0), exec_config=exec_cfg, llm_primary_model="m"
        )

        assert outcome is not None
        assert outcome["exit_reason"] == "stop_loss"
        assert advisor_called["n"] == 0  # advisor never consulted -- SL rail took priority
        client.get_underlying_quote.assert_not_called()

    @pytest.mark.asyncio
    async def test_advisor_consulted_when_sl_not_triggered(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        data = _trade_data(entry_price=1.0, sl_level=0.5)
        path = tmp_path / "trade-adv-trade-1.json"
        path.write_text(json.dumps(data))
        trade = extract_open_trade(data, path)
        assert trade is not None

        client = AsyncMock()
        client.get_option_quote = AsyncMock(
            return_value={"bid": 1.4, "ask": 1.6}
        )  # mark 1.5, well above SL
        client.get_underlying_quote = AsyncMock(return_value={"last": 710.0})
        client.cancel_order = AsyncMock()
        client.submit_order = AsyncMock(return_value=type("R", (), {"order_id": "sell-1"})())
        client.get_order = AsyncMock(
            return_value=type(
                "R", (), {"status": "filled", "filled_avg_price": "1.5", "order_id": "sell-1"}
            )()
        )

        async def fake_process(*args, **kwargs):
            return ExitDecision(should_exit=True, exit_reason="advisor_exit", limit_mult=0.9)

        monkeypatch.setattr(exit_advisor, "process", fake_process)

        exec_cfg = ExecutionConfig(exit_advisor=ExitAdvisorConfig(enabled=True))
        outcome = await monitor_trade(
            trade, client, now=_now(10, 0), exec_config=exec_cfg, llm_primary_model="m"
        )

        assert outcome is not None
        assert outcome["exit_reason"] == "advisor_exit"

    @pytest.mark.asyncio
    async def test_advisor_disabled_never_calls_process(self, tmp_path: Path, monkeypatch) -> None:
        data = _trade_data(entry_price=1.0, sl_level=0.5)
        path = tmp_path / "trade-adv-trade-1.json"
        path.write_text(json.dumps(data))
        trade = extract_open_trade(data, path)
        assert trade is not None

        client = AsyncMock()
        client.get_option_quote = AsyncMock(return_value={"bid": 1.4, "ask": 1.6})

        called = {"n": 0}

        async def fake_process(*args, **kwargs):
            called["n"] += 1
            return None

        monkeypatch.setattr(exit_advisor, "process", fake_process)

        exec_cfg = ExecutionConfig(exit_advisor=ExitAdvisorConfig(enabled=False))
        outcome = await monitor_trade(
            trade, client, now=_now(10, 0), exec_config=exec_cfg, llm_primary_model="m"
        )
        assert outcome is None
        assert called["n"] == 0

    @pytest.mark.asyncio
    async def test_no_exec_config_behaves_like_today(self, tmp_path: Path) -> None:
        data = _trade_data(entry_price=1.0, sl_level=0.5)
        path = tmp_path / "trade-adv-trade-1.json"
        path.write_text(json.dumps(data))
        trade = extract_open_trade(data, path)
        assert trade is not None

        client = AsyncMock()
        client.get_option_quote = AsyncMock(return_value={"bid": 1.4, "ask": 1.6})

        outcome = await monitor_trade(trade, client, now=_now(10, 0))
        assert outcome is None
