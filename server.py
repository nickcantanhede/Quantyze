"""Quantyze Flask HTTP API for post-simulation book and engine state.

Module Description
==================
This module exposes JSON endpoints for health checks, book summary, depth,
matching metrics, trade log, and execution records. It is intended to be
constructed after a simulation run with the live ``OrderBook`` and
``MatchingEngine`` instances created in ``main.py``.

Copyright Information
===============================

Copyright (c) 2026 Cade McNelly, Nicolas Miranda Cantanhede,
Sahand Samadirand
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

from matching_engine import MatchingEngine
from neural_net import Agent
from order_book import OrderBook
from price_level import PriceLevel

_DEFAULT_DEPTH_LEVELS = 10
_MAX_DEPTH_LEVELS = 500
_DEFAULT_TRADE_LIMIT = 200
_MAX_TRADE_LIMIT = 10_000


def _price_level_top(level: PriceLevel | None) -> dict[str, float] | None:
    """Serialize best bid or ask level for JSON, or None when the side is empty."""

    if level is None:
        return None
    return {"price": float(level.price), "volume": float(level.volume)}


def _clamp_int(value: Any, default: int, low: int, high: int) -> int:
    """Parse ``value`` as int, fall back to ``default``, then clamp to ``[low, high]``."""

    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, parsed))


def _api_health_payload() -> dict[str, str]:
    """Body for ``GET /api/health``."""

    return {"status": "ok", "service": "quantyze"}


def _api_book_summary_payload(book: OrderBook, agent: Agent | None) -> dict[str, Any]:
    """Body for ``GET /api/book/summary``."""

    payload: dict[str, Any] = {
        "best_bid": _price_level_top(book.best_bid()),
        "best_ask": _price_level_top(book.best_ask()),
        "spread": book.spread(),
        "mid_price": book.mid_price(),
    }
    if agent is not None:
        payload["agent"] = {"current_pnl": agent.current_pnl()}
    else:
        payload["agent"] = None
    return payload


def _api_book_depth_payload(book: OrderBook, levels: int) -> dict[str, Any]:
    """Body for ``GET /api/book/depth`` given a resolved level count."""

    raw = book.depth_snapshot(levels)
    bids = [{"price": p, "volume": v} for p, v in raw["bids"]]
    asks = [{"price": p, "volume": v} for p, v in raw["asks"]]
    return {"levels": levels, "bids": bids, "asks": asks}


def _api_metrics_payload(engine: MatchingEngine) -> dict[str, Any]:
    """Body for ``GET /api/metrics``."""

    return engine.compute_metrics()


def _api_trades_payload(
    book: OrderBook,
    limit: int,
    offset: int,
    log_path: str | None,
) -> dict[str, Any]:
    """Body for ``GET /api/trades`` with query params already resolved."""

    records: list[dict] = list(book.trade_log)
    if not records and log_path:
        path = Path(log_path)
        if path.is_file():
            try:
                with path.open(encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, list):
                    records = loaded
            except (OSError, json.JSONDecodeError):
                records = []

    total = len(records)
    slice_records = records[offset: offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "trades": slice_records,
    }


def _api_execution_log_payload(
    engine: MatchingEngine,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    """Body for ``GET /api/execution-log`` with query params already resolved."""

    total = len(engine.execution_log)
    entries = engine.execution_log[offset: offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "entries": entries,
    }


def _api_open_orders_payload(book: OrderBook) -> dict[str, Any]:
    """Body for ``GET /api/orders/open``."""

    return {
        "count": len(book.order_index),
        "orders": [o.to_dict() for o in book.order_index.values()],
    }


def create_app(
    book: OrderBook,
    engine: MatchingEngine,
    *,
    agent: Agent | None = None,
    log_path: str | None = None,
) -> Flask:
    """Build a Flask app that reads the given book and engine after simulation."""

    app = Flask(__name__)
    app.config["QUANTYZE_LOG_PATH"] = log_path

    @app.get("/api/health")
    def health() -> Any:
        """JSON liveness probe for the Quantyze API."""

        return jsonify(_api_health_payload())

    @app.get("/api/book/summary")
    def book_summary() -> Any:
        """JSON best bid/ask, spread, mid, and optional agent P&L."""

        return jsonify(_api_book_summary_payload(book, agent))

    @app.get("/api/book/depth")
    def book_depth() -> Any:
        """JSON depth-of-book snapshot; ``levels`` query param caps rows per side."""

        levels = _clamp_int(
            request.args.get("levels"),
            _DEFAULT_DEPTH_LEVELS,
            1,
            _MAX_DEPTH_LEVELS,
        )
        return jsonify(_api_book_depth_payload(book, levels))

    @app.get("/api/metrics")
    def metrics() -> Any:
        """JSON aggregate matching-engine counters and slippage summary."""

        return jsonify(_api_metrics_payload(engine))

    @app.get("/api/trades")
    def trades() -> Any:
        """JSON paginated trade log from memory or optional flush file."""

        limit = _clamp_int(
            request.args.get("limit"),
            _DEFAULT_TRADE_LIMIT,
            1,
            _MAX_TRADE_LIMIT,
        )
        offset = _clamp_int(request.args.get("offset"), 0, 0, _MAX_TRADE_LIMIT)
        path = app.config.get("QUANTYZE_LOG_PATH")
        log = str(path) if path else None
        return jsonify(_api_trades_payload(book, limit, offset, log))

    @app.get("/api/execution-log")
    def execution_log() -> Any:
        """JSON paginated fill records from the matching engine execution log."""

        limit = _clamp_int(
            request.args.get("limit"),
            _DEFAULT_TRADE_LIMIT,
            1,
            _MAX_TRADE_LIMIT,
        )
        offset = _clamp_int(request.args.get("offset"), 0, 0, _MAX_TRADE_LIMIT)
        return jsonify(_api_execution_log_payload(engine, limit, offset))

    @app.get("/api/orders/open")
    def open_orders() -> Any:
        """Resting orders still in the in-memory order index (post-simulation snapshot)."""

        return jsonify(_api_open_orders_payload(book))

    return app


def run_server(app: Flask, port: int, host: str = "127.0.0.1") -> None:
    """Start the Flask development server (blocks until stopped)."""

    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == '__main__':
    import doctest
    import python_ta

    doctest.testmod()

    python_ta.check_all(config={
        'extra-imports': [
            'json', 'pathlib', 'typing', 'flask', 'matching_engine',
            'neural_net', 'order_book', 'price_level', 'doctest', 'python_ta'
        ],
        'allowed-io': ['_api_trades_payload', 'run_server'],
        'max-line-length': 120
    })
