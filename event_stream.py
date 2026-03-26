
"""Quantyze event streaming scaffolds.

This module defines the EventStream class, which sequences Event objects and
passes them to the matching engine during a simulation or replay.

Copyright Information
===============================

Copyright (c) 2026 Cade McNelly, Nicolas Miranda Cantanhede,
Sahand Samadirand
"""

from datetime import datetime
import time

from orders import Event
from matching_engine import MatchingEngine


class EventStream:
    """Sequence Event objects and send them to a matching engine.

    Instance Attributes:
    - source: the finite list of events to replay through the engine
    - engine: the matching engine that processes emitted events
    - speed: the replay speed multiplier used during event emission
    - running: whether the stream is currently active
    - current_ts: the timestamp of the most recently emitted event

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
        """Initialize this stream with a source of events and a matching engine.

        Preconditions:
        - speed >= 0
        """
        self.source = source
        self.engine = engine
        self.speed = speed
        self.running = False
        self.current_ts = datetime.now()

    def start(self) -> None:
        """Mark this stream as running and process events from its source.

        Preconditions:
        - self.running is False
        """
        self.running = True
        self.run_all()

    def stop(self) -> None:
        """Mark this stream as stopped so no further events are emitted.

        Preconditions:
        - self.running is True
        """
        self.running = False

    def emit(self, event: Event) -> list[dict]:
        """Emit one event to the matching engine and return any fill records.

        Preconditions:
        - event is a valid Event
        """
        if self.speed > 0:
            time.sleep(1 / self.speed)
        self.current_ts = event.timestamp
        return self.engine.process_event(event)

    def run_all(self) -> None:
        """Process each event in this stream's source until exhausted or stopped.

        Preconditions:
        - self.source is finite
        """
        for event in self.source:
            if not self.running:
                break
            self.emit(event)

        self.running = False
