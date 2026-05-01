"""Quantyze API payload helpers.

Module Description
==================
This module contains the JSON payload helpers shared by the Quantyze browser
API. It serializes book summary, depth, metrics, trades, execution logs, and
open orders from the live in-memory runtime state.

Copyright Information
===============================

Copyright (c) 2026 Nicolas Miranda Cantanhede
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.matching_engine import MatchingEngine
from ml.neural_net import Agent
from core.order_book import OrderBook
from core.price_level import PriceLevel


def price_level_top(level: PriceLevel | None) -> dict[str, float] | None:
    """Serialize the best bid or ask level for JSON, or None when empty."""
    if level is None:
        return None
    return {"price": float(level.price), "volume": float(level.volume)}


def clamp_int(value: Any, default: int, low: int, high: int) -> int:
    """Parse ``value`` as an int, fall back to ``default``, then clamp it."""
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, parsed))


def book_summary_payload(book: OrderBook, agent: Agent | None) -> dict[str, Any]:
    """Return the payload used by ``GET /api/book/summary``.

    >>> from datetime import datetime
    >>> from core.orders import Event, Order
    >>> order_book = OrderBook()
    >>> order_book.add_limit_order(Order(Event(datetime(2026, 1, 1, 9, 30), 'b1', 'buy', 'limit', 99.5, 1.0)))
    >>> order_book.add_limit_order(Order(Event(datetime(2026, 1, 1, 9, 30, 1), 's1', 'sell', 'limit', 100.5, 1.0)))
    >>> payload = book_summary_payload(order_book, None)
    >>> payload['spread']
    1.0
    >>> payload['mid_price']
    100.0
    >>> payload['agent'] is None
    True
    """

    payload_summary: dict[str, Any] = {"best_bid": price_level_top(book.best_bid()),
                                       "best_ask": price_level_top(book.best_ask()), "spread": book.spread(),
                                       "mid_price": book.mid_price(),
                                       "agent": {"current_pnl": agent.current_pnl()} if agent is not None else None}
    return payload_summary


def book_depth_payload(book: OrderBook, levels: int) -> dict[str, Any]:
    """Return the payload used by ``GET /api/book/depth``."""
    raw = book.depth_snapshot(levels)
    bids = [{"price": price, "volume": volume} for price, volume in raw["bids"]]
    asks = [{"price": price, "volume": volume} for price, volume in raw["asks"]]
    return {"levels": levels, "bids": bids, "asks": asks}


def metrics_payload(engine: MatchingEngine) -> dict[str, Any]:
    """Return the payload used by ``GET /api/metrics``."""
    return engine.compute_metrics()


def _load_trade_records(log_path: str | None) -> list[dict]:
    """Return trade records loaded from ``log_path``, or an empty list on failure."""
    if log_path is None:
        return []

    path = Path(log_path)
    if not path.is_file():
        return []

    try:
        with path.open(encoding="utf-8") as file:
            loaded = json.load(file)
    except (OSError, json.JSONDecodeError):
        return []

    if isinstance(loaded, list):
        return loaded
    return []


def trades_payload(
    book: OrderBook,
    limit: int,
    offset: int,
    log_path: str | None,
) -> dict[str, Any]:
    """Return the payload used by ``GET /api/trades``."""
    records: list[dict] = list(book.trade_log)
    if not records and log_path:
        records = _load_trade_records(log_path)

    total = len(records)
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "trades": records[offset: offset + limit],
    }


def execution_log_payload(
    engine: MatchingEngine,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    """Return the payload used by ``GET /api/execution-log``."""
    total = len(engine.execution_log)
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "entries": engine.execution_log[offset: offset + limit],
    }


def open_orders_payload(book: OrderBook) -> dict[str, Any]:
    """Return the payload used by ``GET /api/orders/open``."""
    return {
        "count": len(book.order_index),
        "orders": [order.to_dict() for order in book.order_index.values()],
    }


if __name__ == '__main__':
    import doctest
    import python_ta

    doctest.testmod()

    python_ta.check_all(config={
        'extra-imports': [
            'json', 'pathlib', 'typing', 'core.matching_engine',
            'ml.neural_net', 'core.order_book', 'core.price_level', 'doctest', 'python_ta'
        ],
        'allowed-io': ['_load_trade_records'],
        'max-line-length': 120
    })
