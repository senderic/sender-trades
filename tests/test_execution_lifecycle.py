"""Unit tests for TradeLifecycle state machine."""

from __future__ import annotations

from src.execution.lifecycle import TradeLifecycle
from src.execution.models import InvalidTransitionError, TradeState


class TestTradeLifecycleTransitions:
    def test_valid_forward_progression(self) -> None:
        lc = TradeLifecycle("trade-1")
        assert lc.state == TradeState.CREATED
        lc.transition(TradeState.VALIDATING)
        assert lc.state == TradeState.VALIDATING
        lc.transition(TradeState.SUBMITTED)
        assert lc.state == TradeState.SUBMITTED
        lc.transition(TradeState.FILLED)
        assert lc.state == TradeState.FILLED
        lc.transition(TradeState.EXITS_PLACED)
        assert lc.state == TradeState.EXITS_PLACED

    def test_created_cannot_jump_to_filled(self) -> None:
        lc = TradeLifecycle("trade-1")
        try:
            lc.transition(TradeState.FILLED)
            assert False, "Expected InvalidTransitionError"
        except InvalidTransitionError:
            pass

    def test_cannot_transition_from_terminal(self) -> None:
        lc = TradeLifecycle("trade-1")
        lc._state = TradeState.CLOSED
        try:
            lc.transition(TradeState.EXITS_PLACED)
            assert False, "Expected InvalidTransitionError"
        except InvalidTransitionError:
            pass

    def test_cannot_transition_backwards(self) -> None:
        lc = TradeLifecycle("trade-1")
        lc.transition(TradeState.VALIDATING)
        lc.transition(TradeState.SUBMITTED)
        try:
            lc.transition(TradeState.VALIDATING)
            assert False, "Expected InvalidTransitionError"
        except InvalidTransitionError:
            pass

    def test_rejected_is_terminal(self) -> None:
        lc = TradeLifecycle("trade-1")
        lc.transition(TradeState.VALIDATING)
        lc.transition(TradeState.SUBMITTED)
        lc.transition(TradeState.REJECTED)
        assert lc.is_terminal is True
        assert TradeState.REJECTED.is_terminal is True
        try:
            lc.transition(TradeState.CLOSED)
            assert False, "Expected InvalidTransitionError"
        except InvalidTransitionError:
            pass

    def test_expired_is_terminal(self) -> None:
        lc = TradeLifecycle("trade-1")
        lc._state = TradeState.EXPIRED
        assert lc.is_terminal is True

    def test_closed_is_terminal(self) -> None:
        lc = TradeLifecycle("trade-1")
        lc._state = TradeState.CLOSED
        assert lc.is_terminal is True

    def test_failed_is_terminal(self) -> None:
        assert TradeState.FAILED.is_terminal is True

    def test_not_terminal_mid_lifecycle(self) -> None:
        lc = TradeLifecycle("trade-1")
        lc.transition(TradeState.VALIDATING)
        lc.transition(TradeState.SUBMITTED)
        assert lc.is_terminal is False


class TestTradeLifecycleEvents:
    def test_records_initial_event(self) -> None:
        lc = TradeLifecycle("trade-1")
        events = lc.events
        assert len(events) >= 1
        assert events[0].state_to == "created"
        assert events[0].state_from is None

    def test_records_all_transitions(self) -> None:
        lc = TradeLifecycle("trade-1")
        lc.transition(TradeState.VALIDATING)
        lc.transition(TradeState.SUBMITTED)
        lc.transition(TradeState.FILLED)
        events = lc.events
        assert len(events) == 4

    def test_event_includes_metadata(self) -> None:
        lc = TradeLifecycle("trade-1")
        event = lc.transition(TradeState.VALIDATING, {"source": "engine"})
        assert event.metadata["source"] == "engine"

    def test_event_summary_is_serializable(self) -> None:
        lc = TradeLifecycle("trade-1")
        lc.transition(TradeState.VALIDATING)
        lc.transition(TradeState.SUBMITTED, {"order_id": "abc123"})
        summary = lc.event_summary()
        assert len(summary) == 3
        assert summary[2]["metadata"]["order_id"] == "abc123"
        assert "timestamp" in summary[0]
        assert "state_from" in summary[0]
        assert "state_to" in summary[0]

    def test_take_profit_exit_path(self) -> None:
        lc = TradeLifecycle("trade-1")
        lc.transition(TradeState.VALIDATING)
        lc.transition(TradeState.SUBMITTED)
        lc.transition(TradeState.FILLED)
        lc.transition(TradeState.EXITS_PLACED)
        lc.transition(TradeState.TP_FILLED)
        lc.transition(TradeState.CLOSED)
        assert lc.is_terminal is True
        assert lc.state == TradeState.CLOSED

    def test_stop_loss_exit_path(self) -> None:
        lc = TradeLifecycle("trade-1")
        lc.transition(TradeState.VALIDATING)
        lc.transition(TradeState.SUBMITTED)
        lc.transition(TradeState.FILLED)
        lc.transition(TradeState.EXITS_PLACED)
        lc.transition(TradeState.SL_FILLED)
        lc.transition(TradeState.CLOSED)
        assert lc.is_terminal is True

    def test_force_close_exit_path(self) -> None:
        lc = TradeLifecycle("trade-1")
        lc.transition(TradeState.VALIDATING)
        lc.transition(TradeState.SUBMITTED)
        lc.transition(TradeState.FILLED)
        lc.transition(TradeState.EXITS_PLACED)
        lc.transition(TradeState.FORCE_CLOSED)
        lc.transition(TradeState.CLOSED)
        assert lc.is_terminal is True


class TestInvalidTransitionError:
    def test_message_contains_both_states(self) -> None:
        err = InvalidTransitionError(TradeState.CREATED, TradeState.FILLED)
        msg = str(err)
        assert "created" in msg
        assert "filled" in msg
