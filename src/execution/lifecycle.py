"""Trade lifecycle state machine with structured event recording."""

from __future__ import annotations

from typing import Any

import structlog

from src.execution.models import (
    VALID_TRANSITIONS,
    InvalidTransitionError,
    LifecycleEvent,
    TradeState,
)

logger = structlog.get_logger()


class TradeLifecycle:
    """State machine that tracks a trade from creation through closure.

    Every state transition is recorded as a :class:`LifecycleEvent` for
    the audit trail. Invalid transitions raise
    :class:`InvalidTransitionError`.
    """

    def __init__(self, trade_id: str, initial_state: TradeState = TradeState.CREATED):
        """Initialize the lifecycle for a given trade.

        Args:
            trade_id: Unique trade identifier.
            initial_state: Starting state (defaults to ``CREATED``).
        """
        self.trade_id = trade_id
        self._state = initial_state
        self._events: list[LifecycleEvent] = []

        self._record_event(None, initial_state, {})
        logger.info(
            "trade_lifecycle_init",
            trade_id=trade_id,
            state=initial_state.value,
        )

    @property
    def state(self) -> TradeState:
        """Current state of the trade."""
        return self._state

    @property
    def events(self) -> list[LifecycleEvent]:
        """All recorded lifecycle events."""
        return list(self._events)

    @property
    def is_terminal(self) -> bool:
        """Return True if the trade has reached a terminal state."""
        return self._state.is_terminal

    def transition(
        self, to_state: TradeState, metadata: dict[str, Any] | None = None
    ) -> LifecycleEvent:
        """Transition from the current state to a new state.

        Args:
            to_state: The target state to transition to.
            metadata: Arbitrary metadata to attach to the event.

        Returns:
            The recorded :class:`LifecycleEvent`.

        Raises:
            InvalidTransitionError: If the transition is not allowed.
        """
        meta = metadata or {}
        allowed = VALID_TRANSITIONS.get(self._state, set())
        if to_state not in allowed:
            raise InvalidTransitionError(self._state, to_state)

        return self._record_event(self._state, to_state, meta)

    def _record_event(
        self, from_state: TradeState | None, to_state: TradeState, metadata: dict[str, Any]
    ) -> LifecycleEvent:
        event = LifecycleEvent(
            trade_id=self.trade_id,
            state_from=from_state.value if from_state else None,
            state_to=to_state.value,
            metadata=metadata,
        )
        self._events.append(event)
        self._state = to_state
        logger.info(
            "trade_lifecycle_transition",
            trade_id=self.trade_id,
            state_from=from_state.value if from_state else None,
            state_to=to_state.value,
        )
        return event

    def event_summary(self) -> list[dict[str, Any]]:
        """Return a lightweight summary of all events for serialization.

        Returns:
            List of event dicts with timestamp, state_from, state_to, and metadata.
        """
        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "state_from": e.state_from,
                "state_to": e.state_to,
                "metadata": e.metadata,
            }
            for e in self._events
        ]
