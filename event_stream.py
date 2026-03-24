
"""Quantyze event streaming scaffolds.

This module defines the EventStream class, which sequences Event objects and
passes them to the matching engine during a simulation or replay.

Copyright Information
===============================

Copyright (c) 2026 Cade McNelly, Nicolas Miranda Cantanhede,
Sahand Samadirand
"""

from datetime import datetime
from orders import Event
from matching_engine import MatchingEngine

class EventStream:
    """Sequence Event objects and send them to a matching engine.

    Representation Invariants:
    - self.speed >= 0
    - all(isinstance(event, Event) for event in self.source)
    """

    source: list[Event]
    engine: MatchingEngine
    speed: float
    running: bool
    current_ts: datetime

    def __init__(self, source: list[Event], engine: MatchingEngine, speed: float = 0.0) -> None:
        """Initialize this event stream with a source, engine, and replay speed.

        Preconditions:
        - speed >= 0
        """
        self.source = source
        self.engine = engine
        self.speed = speed
        self.running = False
        self.current_ts = datetime.now()

    def start(self) -> None:
        """Start processing events from this stream.

        Preconditions:
        - self.running is False
        """
        pass

    def stop(self) -> None:
        """Stop processing events from this stream.

        Preconditions:
        - self.running is True
        """
        pass

    def emit(self, event: Event) -> list[dict]:
        """Send one event to the matching engine and return resulting fills.

        Preconditions:
        - event is a valid Event
        """
        pass

    def run_all(self) -> None:
        """Process every event currently stored in this stream's source.

        Preconditions:
        - self.source is finite
        """
        pass
