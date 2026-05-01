"""Quantyze simulation-analysis coordinator.

Module Description
==================
This module coordinates the analytics flow for a completed Quantyze simulation.
It gathers replay events, matching-engine metrics, execution logs, and final
order-book state, then combines feature extraction and market-regime
interpretation into one report payload.

Copyright Information
===============================

Copyright (c) 2026 Nicolas Miranda Cantanhede
"""

from collections.abc import Sequence
from typing import Any

from core.matching_engine import MatchingEngine
from core.order_book import OrderBook
from core.orders import Event
from data.data_loader import DataLoader


def analyze_simulation(
    engine: MatchingEngine,
    book: OrderBook,
    loader: DataLoader | None = None,
    events: Sequence[Event] | None = None,
    depth_levels: int = 10,
) -> dict[str, Any]:
    """Return one complete analytics report for a finished simulation."""
    ...


def _loader_events(loader: DataLoader | None) -> list[Event]:
    """Return replay events from a loader, or an empty list if unavailable."""
    ...


def _source_format(loader: DataLoader | None) -> str:
    """Return the loader source format, or ``"unknown"`` if unavailable."""
    ...
