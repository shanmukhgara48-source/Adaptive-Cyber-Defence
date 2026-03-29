"""
Event Bus for the Adaptive Cyber Defense Simulator.

Provides pub/sub event routing between simulation modules.
All events are instances of models.state.Event.

Features:
- Subscribe handlers by event type
- Priority queue processing (HIGH before LOW)
- Dead letter queue for unhandled events
- Thread-safe publish/subscribe
- Chained events (handlers may publish new events)
"""

from __future__ import annotations

import heapq
import threading
import time
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional

from ..models.state import Event


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

class EventBus:
    """
    Central publish/subscribe event bus.

    Usage::

        bus = EventBus()
        bus.subscribe("THREAT_DETECTED", handler)
        bus.publish(Event(type="THREAT_DETECTED", payload={"id": "t-001"}))
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
        self._dead_letter: List[Event] = []
        self._history: deque = deque(maxlen=10000)
        self._counter: int = 0   # tie-breaker for heap

    # -----------------------------------------------------------------------
    # Subscribe / unsubscribe
    # -----------------------------------------------------------------------

    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        """Register a handler for a specific event type."""
        with self._lock:
            if handler not in self._handlers[event_type]:
                self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        """Remove a handler for a specific event type."""
        with self._lock:
            handlers = self._handlers.get(event_type, [])
            if handler in handlers:
                handlers.remove(handler)

    # -----------------------------------------------------------------------
    # Publish
    # -----------------------------------------------------------------------

    def publish(self, event: Event) -> int:
        """
        Publish an event. Handlers are called synchronously in priority order.

        Returns number of handlers that received the event.
        """
        if not isinstance(event, Event):
            event = Event(type=str(event))

        # Ensure monotonic timestamp
        with self._lock:
            if self._history and event.timestamp <= self._history[-1].timestamp:
                event = Event(
                    type=event.type,
                    payload=event.payload,
                    timestamp=self._history[-1].timestamp + 1e-9,
                    source=event.source,
                    priority=event.priority,
                )
            self._history.append(event)
            handlers = list(self._handlers.get(event.type, []))

        if not handlers:
            with self._lock:
                self._dead_letter.append(event)
            return 0

        called = 0
        for handler in handlers:
            try:
                handler(event)
                called += 1
            except Exception:
                pass   # isolate handler failures
        return called

    def publish_many(self, events: List[Event]) -> None:
        """Publish a list of events sorted by priority (highest first)."""
        sorted_events = sorted(events, key=lambda e: -e.priority)
        for event in sorted_events:
            self.publish(event)

    # -----------------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all subscribers, dead letters, and history."""
        with self._lock:
            self._handlers.clear()
            self._dead_letter.clear()
            self._history.clear()
            self._counter = 0

    @property
    def dead_letter_queue(self) -> List[Event]:
        """Events that had no subscribers."""
        with self._lock:
            return list(self._dead_letter)

    @property
    def history(self) -> List[Event]:
        """All events published since last reset (up to 10,000)."""
        with self._lock:
            return list(self._history)

    def history_for_type(self, event_type: str) -> List[Event]:
        with self._lock:
            return [e for e in self._history if e.type == event_type]


# ---------------------------------------------------------------------------
# Global singleton (optional convenience — engines can use their own)
# ---------------------------------------------------------------------------

_global_bus: Optional[EventBus] = None


def get_global_bus() -> EventBus:
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


def reset_global_bus() -> None:
    global _global_bus
    _global_bus = EventBus()
