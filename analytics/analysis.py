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

from analytics.features import (
    depth_imbalance,
    depth_totals,
    event_side_counts,
    event_type_counts,
    safe_ratio,
    total_trade_quantity,
    trade_side_counts,
)

from analytics.regimes import classify_market_regime, regime_summary
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

    if events is not None:
        replay_events = list(events)
    else:
        replay_events = _loader_events(loader)

    event_count = len(replay_events)
    
    metrics = engine.compute_metrics()
    fill_count = metrics['fill_count']
    cancel_count = metrics['cancel_count']
    fill_rate = safe_ratio(fill_count, event_count)
    cancel_rate = safe_ratio(cancel_count, event_count)

    depth = depth_totals(book, depth_levels)
    bid_depth, ask_depth = depth['bid_depth'], depth['ask_depth']
    depth_bias = depth_imbalance(bid_depth, ask_depth)
    spread = book.spread()
    mid_price = book.mid_price()

    trades = engine.execution_log.copy()
    trade_counts = trade_side_counts(trades)
    buy_trade_count = trade_counts['buy']
    sell_trade_count = trade_counts['sell']
    trade_count = len(trades)
    total_quantity = total_trade_quantity(trades)
    order_flow_imbalance = safe_ratio(buy_trade_count - sell_trade_count, trade_count)

    regime = classify_market_regime(
        event_count=event_count,
        trade_count=trade_count,
        fill_rate=fill_rate,
        cancel_rate=cancel_rate,
        spread=spread,
        bid_depth=bid_depth,
        ask_depth=ask_depth,
        depth_imbalance=depth_bias,
        order_flow_imbalance=order_flow_imbalance,
    )

    summary = regime_summary(regime)

    return {
        "source_format": _source_format(loader),
        "event_count": event_count,
        "event_mix": event_type_counts(replay_events),
        "side_mix": event_side_counts(replay_events),
        "fill_count": fill_count,
        "cancel_count": cancel_count,
        "fill_rate": fill_rate,
        "cancel_rate": cancel_rate,
        "trade_count": trade_count,
        "buy_trade_count": buy_trade_count,
        "sell_trade_count": sell_trade_count,
        "order_flow_imbalance": order_flow_imbalance,
        "total_executed_quantity": total_quantity,
        "average_execution_size": safe_ratio(total_quantity, trade_count),
        "depth_levels": depth_levels,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "depth_imbalance": depth_bias,
        "spread": spread,
        "mid_price": mid_price,
        "liquidity_regime": regime,
        "summary": summary,
    }


def _loader_events(loader: DataLoader | None) -> list[Event]:
    """Return replay events from a loader, or an empty list if unavailable."""
    
    if loader is None:
        return []
    else:
        return loader.events.copy()


def _source_format(loader: DataLoader | None) -> str:
    """Return the loader source format, or ``"unknown"`` if unavailable."""
    
    if loader is None or loader.source_format is None:
        return "unknown"
    else:
        return loader.source_format

