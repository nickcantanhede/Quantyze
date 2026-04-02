"""Quantyze internal dataset generator.

Module Description
==================
This module recreates the packaged ``sample_internal.csv`` and
``huge_internal.csv`` files in Quantyze's canonical event format. The small
dataset is a fixed deterministic internal event stream, and the large dataset
is built by repeating that stream with shifted timestamps and suffixed order
ids so the events remain independent.

Copyright Information
===============================

Copyright (c) 2026 Cade McNelly, Nicolas Miranda Cantanhede,
Sahand Samadirand
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta
from pathlib import Path

from orders import Event

SAMPLE_DATASET_PATH = "sample_internal.csv"
HUGE_DATASET_PATH = "huge_internal.csv"
INTERNAL_DATASET_START = datetime.fromisoformat("2026-03-30T23:27:45.717728")
SAMPLE_EVENT_COUNT = 120
BLOCK_GAP_SECONDS = 600
HUGE_BLOCK_COUNT = 120


def _new_event(
    offset_seconds: int,
    order_id: str,
    side: str,
    order_type: str,
    price: float | None,
    quantity: float
) -> Event:
    """Return one validated internal dataset event."""
    event = Event(
        INTERNAL_DATASET_START + timedelta(seconds=offset_seconds),
        order_id,
        side,
        order_type,
        price,
        quantity,
    )
    event.validate()
    return event


def generate_sample_events() -> list[Event]:
    """Return the deterministic packaged small internal dataset.

    >>> sample_events = generate_sample_events()
    >>> len(sample_events)
    120
    >>> sample_events[0].order_id
    'seed_bid_0'
    >>> sample_events[12].order_type
    'cancel'
    >>> sample_events[-1].order_id
    'rest_ask_119'
    """
    events = [
        _new_event(0, "seed_bid_0", "buy", "limit", 99.9, 12.0),
        _new_event(1, "seed_ask_0", "sell", "limit", 100.1, 12.0),
        _new_event(2, "seed_bid_1", "buy", "limit", 99.8, 10.0),
        _new_event(3, "seed_ask_1", "sell", "limit", 100.2, 10.0),
        _new_event(4, "seed_bid_2", "buy", "limit", 99.7, 8.0),
        _new_event(5, "seed_ask_2", "sell", "limit", 100.3, 8.0),
    ]

    for event_index in range(6, SAMPLE_EVENT_COUNT):
        cycle_index = (event_index - 6) // 8
        cycle_step = (event_index - 6) % 8
        cycle_base = 6 + cycle_index * 8
        actions = (
            ("rest_bid", "buy", "limit", 99.7, 4.0, cycle_base),
            ("rest_ask", "sell", "limit", 100.3, 4.0, cycle_base + 1),
            ("cross_buy", "buy", "limit", 100.2, 6.0, cycle_base + 2),
            ("cross_sell", "sell", "limit", 99.8, 6.0, cycle_base + 3),
            ("market_buy", "buy", "market", None, 4.0, cycle_base + 4),
            ("market_sell", "sell", "market", None, 4.0, cycle_base + 5),
            ("rest_bid", "buy", "cancel", None, 0.0, cycle_base),
            ("rest_ask", "sell", "cancel", None, 0.0, cycle_base + 1),
        )
        prefix, side, order_type, price, quantity, order_number = actions[cycle_step]
        events.append(
            _new_event(event_index, f"{prefix}_{order_number}", side, order_type, price, quantity)
        )

    return events


def generate_huge_events(sample_events: list[Event], block_count: int = HUGE_BLOCK_COUNT) -> list[Event]:
    """Return the repeated packaged large internal dataset.

    >>> sample_events = generate_sample_events()
    >>> huge_events = generate_huge_events(sample_events, 2)
    >>> len(huge_events)
    240
    >>> huge_events[0].order_id
    'seed_bid_0__0'
    >>> huge_events[120].order_id
    'seed_bid_0__1'
    """
    if not sample_events:
        return []

    stride_seconds = (sample_events[-1].timestamp - sample_events[0].timestamp).seconds + BLOCK_GAP_SECONDS
    repeated_events = []
    for block_index in range(block_count):
        block_offset = timedelta(seconds=block_index * stride_seconds)
        for event in sample_events:
            repeated_event = Event(
                event.timestamp + block_offset,
                f"{event.order_id}__{block_index}",
                event.side,
                event.order_type,
                event.price,
                event.quantity,
            )
            repeated_event.validate()
            repeated_events.append(repeated_event)
    return repeated_events


def write_events_csv(path: str, events: list[Event], line_terminator: str = "\n") -> None:
    """Write ``events`` to ``path`` in Quantyze's canonical internal CSV format."""
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator=line_terminator)
        writer.writerow(["timestamp", "order_id", "side", "order_type", "price", "quantity"])
        for event in events:
            writer.writerow([
                event.timestamp.isoformat(),
                event.order_id,
                event.side,
                event.order_type,
                "" if event.price is None else event.price,
                event.quantity,
            ])


def generate_internal_datasets(
    sample_path: str = SAMPLE_DATASET_PATH,
    huge_path: str = HUGE_DATASET_PATH
) -> tuple[int, int]:
    """Generate the packaged internal CSV pair and return their event counts."""
    sample_events = generate_sample_events()
    huge_events = generate_huge_events(sample_events)
    write_events_csv(sample_path, sample_events, "\n")
    write_events_csv(huge_path, huge_events, "\r\n")
    return len(sample_events), len(huge_events)


def main() -> None:
    """Generate the packaged internal CSV files in the current working directory."""
    sample_count, huge_count = generate_internal_datasets()
    print(f"Wrote {sample_count} events to {Path(SAMPLE_DATASET_PATH).resolve()}")
    print(f"Wrote {huge_count} events to {Path(HUGE_DATASET_PATH).resolve()}")


if __name__ == "__main__":
    if os.environ.get("QUANTYZE_RUN_PYTA") == "1":
        import doctest
        import python_ta

        doctest.testmod()

        python_ta.check_all(config={
            "extra-imports": ["csv", "os", "datetime", "pathlib", "orders", "doctest", "python_ta"],
            "allowed-io": ["write_events_csv", "main"],
            "max-line-length": 100
        })
    else:
        main()
