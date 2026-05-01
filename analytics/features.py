"""Quantyze market-analysis feature helpers.

Module Description
==================
This module contains small analytics helpers for extracting raw measurements
from replayed events, execution records, and the final order-book state. These
helpers are intended to measure facts only; higher-level interpretation belongs
in the regime and analysis modules.

Copyright Information
===============================

Copyright (c) 2026 Nicolas Miranda Cantanhede
"""

from collections.abc import Sequence
from typing import Any

from core.order_book import OrderBook
from core.orders import Event


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a division result that is safe when the denominator is zero."""
    
    return numerator / denominator if denominator != 0 else 0.0


def event_type_counts(events: Sequence[Event]) -> dict[str, int]:
    """Return counts for limit, market, and cancel events in the replay."""
    
    types_so_far = {'limit': 0, 'market': 0, 'cancel': 0}

    for event in events:
        types_so_far[event.order_type] += 1

    return types_so_far


def event_side_counts(events: Sequence[Event]) -> dict[str, int]:
    """Return counts for buy-side and sell-side incoming events."""

    sides_so_far = {'buy': 0, 'sell': 0}

    for event in events:
        sides_so_far[event.side] += 1

    return sides_so_far


def trade_side_counts(trades: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Return counts for buy-side and sell-side completed executions."""
    
    sides_so_far = {'buy': 0, 'sell': 0}

    for trade in trades:
        side = trade.get('side')
        if side in {'buy', 'sell'}:
            sides_so_far[side] += 1

    return sides_so_far


def total_trade_quantity(trades: Sequence[dict[str, Any]]) -> float:
    """Return the total filled quantity across all execution records."""
    
    total_so_far = 0.0

    for trade in trades:
        if 'filled_qty' in trade:
            total_so_far += trade['filled_qty']

    return total_so_far


def depth_totals(book: OrderBook, levels: int = 10) -> dict[str, float]:
    """Return total bid depth and ask depth for the requested book levels."""
    
    depths = {'bid_depth': 0.0, 'ask_depth': 0.0}

    book_snapshot = book.depth_snapshot(levels)
    depths['bid_depth'] = sum(volume for _, volume in book_snapshot['bids'])
    depths['ask_depth'] = sum(volume for _, volume in book_snapshot['asks'])

    return depths


def depth_imbalance(bid_depth: float, ask_depth: float) -> float:
    """Return the signed liquidity imbalance between bid and ask depth."""
    
    return safe_ratio(bid_depth - ask_depth, bid_depth + ask_depth)
